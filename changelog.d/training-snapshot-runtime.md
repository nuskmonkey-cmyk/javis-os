---
bump: minor
title: Training bots can run on a frozen, versioned copy of their documents
---
## vi
### Thêm mới
- **Bot luyện tập chạy trên bản tài liệu đã đóng băng (tắt mặc định).** Một máy chủ của chủ gửi Agent.md và tai-lieu/*.md của một phiên bản, Javis lưu nguyên văn kèm mã băm và model, rồi trả lời từng lượt đúng trên bản đó. Cuộc hội thoại bắt đầu ở bản nào thì ở lại bản đó, lượt gửi lại không gọi model lần hai. Chỉ chạy với tài khoản kênh API có trong danh sách JAVIS_TRAINING_SNAPSHOT_ACCOUNTS, bot chỉ trực kênh API và không vượt mức Đọc tài liệu. Bot tiếp khách thật không đổi gì.

## en
### Added
- **Training bots can answer on a frozen copy of their documents (off by default).** A server of the owner sends one version of Agent.md and tai-lieu/*.md; Javis stores it verbatim with its hash and model and answers each turn on exactly that version. A conversation stays on the version it started on, and a resent turn does not call the model again. Only API channel accounts listed in JAVIS_TRAINING_SNAPSHOT_ACCOUNTS can use it, the bot must serve the API channel alone and stay within Read documents. Bots that talk to customers are unchanged.
