"""Opt-in internal training runtime. No mutable brain reads, tools, or auth-store writes.

Wire with install(app, ...) from main. Uses existing API channel authentication.
Snapshots and conversation pins survive process restarts in a private SQLite store.
"""
import asyncio
import contextlib
import hashlib
import json
import re
import sqlite3
from pathlib import Path

from fastapi import HTTPException, Request

PREFIX = '/chatbot-api/v1/training'
ID = re.compile(r'^[A-Za-z0-9_-]{1,100}$')
# A branch is the deployment's label for one training bot (a shop, a team); the allowlist sets it, never the caller.
BRANCH = re.compile(r'^[A-Za-z0-9_-]{1,20}$')


def validate_manifest(raw):
    if not isinstance(raw, str) or len(raw.encode()) > 1_000_000:
        raise ValueError('invalid manifest size')
    files = json.loads(raw)
    if not isinstance(files, dict) or not files.get('Agent.md'):
        raise ValueError('Agent.md required')
    for name, text in files.items():
        if not isinstance(name, str) or not isinstance(text, str):
            raise ValueError('invalid file')
        if name != 'Agent.md' and not (name.startswith('tai-lieu/') and name.endswith('.md')):
            raise ValueError('file outside training manifest')
        if any(part in ('', '.', '..') for part in name.split('/')) or '\\' in name:
            raise ValueError('invalid path')
    return files


class SnapshotStore:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS snapshots (
              id TEXT PRIMARY KEY, account TEXT NOT NULL, branch TEXT NOT NULL,
              version TEXT NOT NULL, hash TEXT NOT NULL, manifest TEXT NOT NULL,
              provider TEXT NOT NULL, model TEXT NOT NULL, reasoning TEXT NOT NULL DEFAULT '',
              UNIQUE(account,branch,version));
            CREATE TABLE IF NOT EXISTS sessions (
              account TEXT NOT NULL, conversation TEXT NOT NULL, snapshot TEXT NOT NULL,
              PRIMARY KEY(account,conversation));
            CREATE TABLE IF NOT EXISTS turns (
              account TEXT NOT NULL, message TEXT NOT NULL, conversation TEXT NOT NULL,
              snapshot TEXT NOT NULL, request TEXT NOT NULL, response TEXT NOT NULL,
              PRIMARY KEY(account,message));
            ''')
            if 'reasoning' not in {row[1] for row in db.execute('PRAGMA table_info(snapshots)')}:
                db.execute("ALTER TABLE snapshots ADD COLUMN reasoning TEXT NOT NULL DEFAULT ''")
        Path(path).chmod(0o600)
        self.lock = asyncio.Lock()

    @contextlib.contextmanager
    def connect(self):
        # sqlite3's own `with connection` only commits, it never closes: every call would leave a handle open.
        db = sqlite3.connect(self.path, timeout=1)
        try:
            db.row_factory = sqlite3.Row
            db.execute('PRAGMA foreign_keys=ON')
            db.execute('PRAGMA synchronous=FULL')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def create(self, account, branch, body, provider, model, reasoning=""):
        version, raw = body.get('version'), body.get('manifest_json')
        if not isinstance(branch, str) or not BRANCH.fullmatch(branch) or not isinstance(version, str) or not ID.fullmatch(version):
            raise ValueError('invalid branch/version')
        validate_manifest(raw)
        digest = hashlib.sha256(raw.encode()).hexdigest()
        if body.get('hash') != digest or not provider or not model:
            raise ValueError('hash or model missing')
        identity = json.dumps([account, branch, version, digest, provider, model, reasoning], separators=(',', ':'))
        sid = 'snap_' + hashlib.sha256(identity.encode()).hexdigest()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT * FROM snapshots WHERE account=? AND branch=? AND version=?',
                             (account, branch, version)).fetchone()
            if old and old['id'] != sid:
                raise ValueError('version is immutable')
            db.execute('INSERT OR IGNORE INTO snapshots(id,account,branch,version,hash,manifest,provider,model,reasoning) VALUES(?,?,?,?,?,?,?,?,?)',
                       (sid, account, branch, version, digest, raw, provider, model, reasoning))
        return self.inspect(account, branch, sid)

    def inspect(self, account, branch, sid):
        with self.connect() as db:
            row = db.execute('SELECT * FROM snapshots WHERE id=? AND account=? AND branch=?',
                             (sid, account, branch)).fetchone()
        if not row:
            raise ValueError('snapshot not found')
        row = dict(row)
        validate_manifest(row['manifest'])
        if hashlib.sha256(row['manifest'].encode()).hexdigest() != row['hash']:
            raise ValueError('snapshot corrupt')
        return row

    def _before_turn(self, account, branch, sid, body):
        """(duplicate response or None, snapshot, request json, history). Reads only; runs in a worker thread."""
        snapshot = self.inspect(account, branch, sid)
        cid, mid, text = body.get('conversation_id'), body.get('message_id'), body.get('text')
        if not all(isinstance(x, str) and x for x in (cid, mid, text)) or len(cid) > 60 or len(mid) > 100 or len(text) > 4000:
            raise ValueError('invalid turn')
        request = json.dumps(body, sort_keys=True, ensure_ascii=False)
        with self.connect() as db:
            dup = self._duplicate(db, account, mid, cid, sid, request)
            if dup is not None:
                return dup, snapshot, request, []
            self._check_pin(db, account, cid, sid)
            history = []
            for row in db.execute('SELECT request,response FROM turns WHERE account=? AND conversation=? ORDER BY rowid', (account, cid)):
                history.extend([{'role': 'user', 'content': json.loads(row['request'])['text']},
                                {'role': 'assistant', 'content': json.loads(row['response'])['reply']}])
        return None, snapshot, request, history[-40:]

    @staticmethod
    def _duplicate(db, account, mid, cid, sid, request):
        old = db.execute('SELECT * FROM turns WHERE account=? AND message=?', (account, mid)).fetchone()
        if not old:
            return None
        if old['snapshot'] != sid or old['conversation'] != cid or old['request'] != request:
            raise ValueError('idempotency conflict')
        return dict(json.loads(old['response']), duplicate=True)

    @staticmethod
    def _check_pin(db, account, cid, sid):
        pin = db.execute('SELECT snapshot FROM sessions WHERE account=? AND conversation=?', (account, cid)).fetchone()
        if pin and pin['snapshot'] != sid:
            raise ValueError('conversation is pinned to a different snapshot')

    def _commit_turn(self, account, sid, body, request, response):
        """Pin + dedup + the turn in one transaction. Re-checks both under the write lock, since another process may
        have answered the same message or pinned the conversation while the model was running."""
        cid, mid = body['conversation_id'], body['message_id']
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            dup = self._duplicate(db, account, mid, cid, sid, request)
            if dup is not None:
                return dup
            self._check_pin(db, account, cid, sid)
            db.execute('INSERT OR IGNORE INTO sessions VALUES(?,?,?)', (account, cid, sid))
            db.execute('INSERT INTO turns VALUES(?,?,?,?,?,?)', (account, mid, cid, sid, request, json.dumps(response)))
        return response

    async def turn(self, account, branch, sid, body, answer):
        # SQLite work runs in worker threads; the write transaction is NOT held while the model answers, so a slow
        # model never keeps the store locked. A crash before the commit leaves nothing behind and the turn can be
        # retried. Turns are serialized in this process (a low-volume training service).
        async with self.lock:
            dup, snapshot, request, history = await asyncio.to_thread(self._before_turn, account, branch, sid, body)
            if dup is not None:
                return dup
            out = await asyncio.wait_for(answer(snapshot, body['text'], history), timeout=90)
            if not isinstance(out, dict) or not out.get('reply') or out.get('model') != snapshot['model'] or out.get('provider') != snapshot['provider']:
                raise ValueError('model attestation missing')
            response = dict(ok=True, reply=out['reply'], model=out['model'], provider=out['provider'],
                            runtime_id=sid, branch=branch, version=snapshot['version'], hash=snapshot['hash'],
                            silent=False, handed_over=False, turn_id=body['message_id'])
            return await asyncio.to_thread(self._commit_turn, account, sid, body, request, response)


def attestation(row):
    return {k: row[k] for k in ('branch', 'version', 'hash', 'provider', 'model')} | {
        'runtime_id': row['id'], 'immutable': True, 'training_only': True, 'channels': ['api'], 'schema_version': 1}


def install(app, *, path, enabled, accounts, authenticate, model_for, answer):
    """accounts is an explicit {API account ID: branch label} deployment allowlist, never client data.

    authenticate must reuse channels.api.gate AND enforce dedicated API-only bot configuration.
    model_for resolves the effective configured model; answer uses that exact route without tools.
    """
    frozen = SnapshotStore(path) if enabled else None

    def access(request):
        if not enabled:
            raise HTTPException(503, 'training_runtime_disabled')
        account = authenticate(request)
        aid = account['id']
        branch = accounts.get(aid)
        if not isinstance(branch, str) or not BRANCH.fullmatch(branch):
            raise HTTPException(403, 'training_account_not_allowed')
        return aid, branch, account

    async def body(request):
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 1_100_000:
                raise HTTPException(413, 'training_body_too_large')
        try:
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError()
            return value
        except (ValueError, TypeError):
            raise HTTPException(400, 'training_body_invalid') from None

    @app.post(PREFIX + '/snapshots')
    async def training_snapshot_create(request: Request):
        aid, branch, account = access(request)
        data = await body(request)
        if data.get('branch') != branch:
            raise HTTPException(403, 'training_branch_mismatch')
        provider, model, *settings = model_for(account)
        try:
            row = await asyncio.to_thread(frozen.create, aid, branch, data, provider, model, settings[0] if settings else "")
            return attestation(row)
        except (ValueError, TypeError, sqlite3.Error):
            raise HTTPException(409, 'training_snapshot_rejected') from None

    @app.get(PREFIX + '/snapshots/{sid}')
    async def training_snapshot_inspect(sid: str, request: Request):
        aid, branch, _ = access(request)
        try:
            return attestation(await asyncio.to_thread(frozen.inspect, aid, branch, sid))
        except (ValueError, sqlite3.Error):
            raise HTTPException(404, 'training_snapshot_not_found') from None

    @app.post(PREFIX + '/snapshots/{sid}/turn')
    async def training_snapshot_turn(sid: str, request: Request):
        aid, branch, _ = access(request)
        data = await body(request)
        try:
            return await frozen.turn(aid, branch, sid, data, answer)
        except (ValueError, sqlite3.Error, asyncio.TimeoutError):
            raise HTTPException(409, 'training_turn_rejected') from None
    return frozen
