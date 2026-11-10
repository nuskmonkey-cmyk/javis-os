"""Opt-in training snapshot runtime (training_snapshots.py + training_snapshot_wiring.py).

    python tests/run.py training_snapshots      (no network, fake engine)

A server of the owner freezes one version of a training bot (Agent.md + tai-lieu/*.md) and replays conversations on
exactly that version. This file pins: the version is immutable and hashed, a conversation stays on the snapshot it
started on, message_id dedup survives a restart, the account allowlist decides HN/SG (never the caller), the bot must
be API-only and within the API ceiling, only plain API routes are frozen, and a turn whose engine reports a different
model (or none) is refused.
"""
from _paths import ROOT, SERVER  # noqa: E402,F401
import os
import tempfile

os.environ["JAVIS_STATE_DIR"] = tempfile.mkdtemp(prefix="javis-training-snap-")
os.environ["BRAINS_DIR"] = tempfile.mkdtemp(prefix="javis-training-snap-brains-")
os.environ.pop("JAVIS_TRAINING_SNAPSHOTS_ENABLED", None)
import asyncio
import hashlib
import json
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from training_snapshots import SnapshotStore, install, PREFIX


class SnapshotTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'store.sqlite3'
        self.store = SnapshotStore(self.path)
        self.calls = []

    def tearDown(self):
        self.temp.cleanup()

    def create(self, account='hn', branch='HN', version='v1', text='before'):
        manifest = json.dumps({'Agent.md': text, 'tai-lieu/gia.md': 'dữ liệu'}, ensure_ascii=False)
        return self.store.create(account, branch, {'version': version, 'manifest_json': manifest,
                                 'hash': hashlib.sha256(manifest.encode()).hexdigest()}, 'test', 'model-1')

    async def answer(self, snapshot, text, history):
        self.calls.append((snapshot, history))
        return {'reply': json.loads(snapshot['manifest'])['Agent.md'], 'provider': 'test', 'model': 'model-1'}

    async def send(self, snapshot, mid='m1', cid='c1', text='hello'):
        return await self.store.turn('hn', 'HN', snapshot['id'],
                {'message_id': mid, 'conversation_id': cid, 'text': text}, self.answer)

    async def test_pin_restart_dedup_and_next_version(self):
        v1 = self.create()
        self.assertEqual((await self.send(v1))['reply'], 'before')
        self.store = SnapshotStore(self.path)
        self.assertTrue((await self.send(v1))['duplicate'])
        self.assertEqual(len(self.calls), 1)
        v2 = self.create(version='v2', text='after')
        with self.assertRaises(ValueError):
            await self.send(v2, mid='m2')
        self.assertEqual((await self.send(v1, mid='m2'))['reply'], 'before')
        self.assertEqual((await self.send(v2, mid='m3', cid='new'))['reply'], 'after')
        self.assertEqual(self.calls[1][1][1]['content'], 'before')

    async def test_immutable_account_branch_and_idempotency(self):
        v1 = self.create()
        self.assertEqual(self.create()['id'], v1['id'])
        with self.assertRaises(ValueError): self.create(text='changed')
        with self.assertRaises(ValueError): self.store.inspect('sg', 'SG', v1['id'])
        with self.assertRaises(ValueError): self.store.inspect('hn', 'SG', v1['id'])
        await self.send(v1)
        with self.assertRaises(ValueError): await self.send(v1, text='different')

    async def test_bad_model_and_failure_do_not_commit(self):
        v1 = self.create()
        async def bad(*_): return {'reply': 'hi', 'model': 'different', 'provider': 'test'}
        with self.assertRaises(ValueError):
            await self.store.turn('hn', 'HN', v1['id'], {'message_id':'x','conversation_id':'x','text':'x'}, bad)
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM turns').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT count(*) FROM sessions').fetchone()[0], 0)

    async def test_concurrent_duplicate_one_model_call(self):
        v1 = self.create()
        results = await asyncio.gather(self.send(v1), self.send(v1))
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(results[1]['duplicate'])

    async def test_snapshot_dedup_survives_process_exit(self):
        v = self.create()
        script = """
import asyncio,sys
from training_snapshots import SnapshotStore
store=SnapshotStore(sys.argv[1])
async def answer(s,t,h):
    if sys.argv[3]=='read': raise AssertionError('duplicate called model')
    return {'reply':'persisted','model':'model-1','provider':'test'}
result=asyncio.run(store.turn('hn','HN',sys.argv[2],{'message_id':'process-m','conversation_id':'process-c','text':'hi'},answer))
assert result['reply']=='persisted'
if sys.argv[3]=='read': assert result['duplicate']
"""
        for phase in ('write','read'):
            result = subprocess.run([sys.executable, '-c', script, str(self.path), v['id'], phase], capture_output=True, text=True,
                                    env=dict(os.environ, PYTHONPATH=str(SERVER)))
            self.assertEqual(result.returncode, 0, result.stderr)

    async def test_paths_hash_and_corruption(self):
        for files in ({'Agent.md':'x','tai-lieu/../x.md':'x'}, {'Agent.md':'x','elsewhere.md':'x'}, {'tai-lieu/a.md':'x'}):
            raw=json.dumps(files)
            with self.assertRaises(ValueError):
                self.store.create('hn','HN',{'version':'v1','manifest_json':raw,'hash':hashlib.sha256(raw.encode()).hexdigest()},'test','model-1')
        v=self.create()
        with self.store.connect() as db:
            db.execute('UPDATE snapshots SET manifest=? WHERE id=?', ('{"Agent.md":"tampered"}',v['id']))
        with self.assertRaises(ValueError): self.store.inspect('hn','HN',v['id'])


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
    def tearDown(self): self.temp.cleanup()
    def app(self, enabled=True):
        app=FastAPI()
        def auth(req):
            key=req.headers.get('authorization')
            if key not in ('Bearer hn','Bearer sg'): raise HTTPException(401)
            return {'id':key.split()[1]}
        async def answer(s,t,h): return {'reply':'model mock','provider':'test','model':'model-1'}
        install(app,path=Path(self.temp.name)/'db',enabled=enabled,accounts={'hn':'HN','sg':'SG'},
                authenticate=auth,model_for=lambda _:('test','model-1',''),answer=answer)
        return TestClient(app)
    def test_endpoint_auth_branch_and_attestation(self):
        with self.app() as c:
            raw='{"Agent.md":"fixed"}'
            data={'branch':'HN','version':'v1','manifest_json':raw,'hash':hashlib.sha256(raw.encode()).hexdigest()}
            self.assertEqual(c.post(PREFIX+'/snapshots',json=data).status_code,401)
            h={'Authorization':'Bearer hn'}
            self.assertEqual(c.post(PREFIX+'/snapshots',json=dict(data,branch='SG'),headers=h).status_code,403)
            out=c.post(PREFIX+'/snapshots',json=data,headers=h)
            self.assertEqual(out.status_code,200)
            sid=out.json()['runtime_id']
            self.assertTrue(out.json()['immutable'])
            self.assertEqual(c.get(PREFIX+'/snapshots/'+sid,headers={'Authorization':'Bearer sg'}).status_code,404)
            result=c.post(PREFIX+'/snapshots/'+sid+'/turn',headers=h,json={'message_id':'m','conversation_id':'c','text':'hi'})
            self.assertEqual(result.status_code,200)
            self.assertEqual(result.json()['runtime_id'],sid)
            self.assertEqual(result.json()['model'],'model-1')
    def test_disabled_does_not_create_store(self):
        with self.app(False) as c:
            self.assertEqual(c.post(PREFIX+'/snapshots',json={}).status_code,503)
        self.assertFalse((Path(self.temp.name)/'db').exists())



class WiringTests(unittest.TestCase):
    """register_main with the real Javis modules swapped for fakes: what the composition root itself enforces."""

    def run_wired(self, env, route=('mock-provider', 'api', '', 'model-fixed'), session_route=None):
        import os
        import types
        from unittest.mock import patch
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.bot = {'enabled': True, 'muc_quyen': 'suggest', 'accounts': ['hn']}
        bot = self.bot
        api = types.SimpleNamespace(
            gate=lambda auth, ip, h: ({'id': 'hn'}, None) if auth == 'Bearer mock' else (None, (401, {'code': 'unauthorized'})),
            _bot_of=lambda _: bot,
            ceiling_problem=lambda b: '' if b.get('muc_quyen') in ('suggest', 'read_docs') else 'above ceiling')
        config = types.SimpleNamespace(STATE_DIR=Path(tmp.name), read_settings=lambda: {'model': {}})
        chatbot = types.SimpleNamespace(account_ids_of=lambda b: b['accounts'])
        self.actual = ['model-fixed']

        async def stream(*args):
            yield {'type': 'meta', 'model': self.actual[0]}
            yield {'type': 'text', 'content': 'mock output'}
        ns = {'app': FastAPI(), '_chat_provider_bot': lambda *a: route,
              '_chat_provider_for_session': lambda *a: session_route or route,
              '_chat_reasoning': lambda *a: '', '_api_stream': stream,
              'usage_store': types.SimpleNamespace(record=lambda *a: None)}
        from training_snapshot_wiring import register_main
        with patch.dict(os.environ, env), patch.dict('sys.modules', {
                'channels': types.SimpleNamespace(api=api), 'chatbot_store': chatbot, 'config': config}):
            register_main(ns)
        self.state_dir = Path(tmp.name)
        return TestClient(ns['app'])

    ON = {'JAVIS_TRAINING_SNAPSHOTS_ENABLED': 'true', 'JAVIS_TRAINING_SNAPSHOT_ACCOUNTS': '{"hn":"HN"}'}
    URL = '/chatbot-api/v1/training/snapshots'
    RAW = '{"Agent.md":"fixed"}'
    PACKET = {'branch': 'HN', 'version': 'v1', 'manifest_json': RAW, 'hash': hashlib.sha256(RAW.encode()).hexdigest()}
    H = {'Authorization': 'Bearer mock'}

    def test_auth_api_only_ceiling_and_model_evidence(self):
        with self.run_wired(self.ON) as client:
            self.assertEqual(client.post(self.URL, json=self.PACKET).status_code, 401)
            self.bot['accounts'].append('customer-page')
            self.assertEqual(client.post(self.URL, headers=self.H, json=self.PACKET).status_code, 403)
            self.bot['accounts'] = ['hn']
            self.bot['muc_quyen'] = 'full'
            self.assertEqual(client.post(self.URL, headers=self.H, json=self.PACKET).status_code, 403)
            self.bot['muc_quyen'] = 'suggest'
            result = client.post(self.URL, headers=self.H, json=self.PACKET)
            self.assertEqual(result.status_code, 200)
            url = self.URL + '/' + result.json()['runtime_id'] + '/turn'
            turn = {'message_id': 'm1', 'conversation_id': 'c1', 'text': 'hello'}
            self.assertEqual(client.post(url, headers=self.H, json=turn).status_code, 200)
            self.actual[0] = 'wrong-model'
            turn['message_id'] = 'm2'
            self.assertEqual(client.post(url, headers=self.H, json=turn).status_code, 409)

    def test_cli_engine_and_moving_model_are_not_frozen(self):
        with self.run_wired(self.ON, route=('claude-cli', 'cli', '', 'model-fixed')) as client:
            self.assertEqual(client.post(self.URL, headers=self.H, json=self.PACKET).status_code, 409)
        with self.run_wired(self.ON, route=('mock-provider', 'api', '', 'latest')) as client:
            self.assertEqual(client.post(self.URL, headers=self.H, json=self.PACKET).status_code, 409)

    def test_fallback_to_main_model_is_refused(self):
        with self.run_wired(self.ON, session_route=('other', 'api', '', 'main-model')) as client:
            sid = client.post(self.URL, headers=self.H, json=self.PACKET).json()['runtime_id']
            out = client.post(self.URL + '/' + sid + '/turn', headers=self.H,
                              json={'message_id': 'm1', 'conversation_id': 'c1', 'text': 'hi'})
            self.assertEqual(out.status_code, 409)

    def test_off_by_default_and_bad_allowlist_does_not_crash(self):
        with self.run_wired({}) as client:
            self.assertEqual(client.post(self.URL, headers=self.H, json=self.PACKET).status_code, 503)
        with self.run_wired(dict(self.ON, JAVIS_TRAINING_SNAPSHOT_ACCOUNTS='{"hn":"not a label!"}')) as client:
            self.assertEqual(client.post(self.URL, headers=self.H, json=self.PACKET).status_code, 503)
        with self.run_wired(dict(self.ON, JAVIS_TRAINING_SNAPSHOT_ACCOUNTS='not json')) as client:
            self.assertEqual(client.post(self.URL, headers=self.H, json=self.PACKET).status_code, 503)
        self.assertFalse((self.state_dir / 'training-runtime.sqlite3').exists())


class MainWiringTests(unittest.TestCase):
    def test_main_registers_routes_and_keeps_the_prefix_slash(self):
        import main
        self.assertIn('/chatbot-api/v1/training/', main._AUTH_PUBLIC_PREFIX)
        self.assertNotIn('/chatbot-api/v1/training', main._AUTH_PUBLIC_PREFIX)
        from _paths import moi_duong_dan
        paths = set(moi_duong_dan(main.app))
        for p in ('/chatbot-api/v1/training/snapshots', '/chatbot-api/v1/training/snapshots/{sid}',
                  '/chatbot-api/v1/training/snapshots/{sid}/turn', '/chatbot-api/v1/turn'):
            self.assertIn(p, paths)
        # Off by default: nothing is written to the state dir at import.
        import config
        self.assertFalse((config.STATE_DIR / 'training-runtime.sqlite3').exists())


if __name__ == '__main__':
    unittest.main()
