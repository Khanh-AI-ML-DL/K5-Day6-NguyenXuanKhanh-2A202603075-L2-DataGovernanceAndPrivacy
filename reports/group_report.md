# Group Report — Day 06 Data Lab (Field-to-Release PoC)

> Các số liệu trong báo cáo lấy từ lần chạy thật của nhóm (`out/*/run_info.json`, `out/poc/audit_log.jsonl`).
> Các chỉ số chính thức (PII Miss Rate, over-redaction) do giảng viên chấm bằng overlay `gt/`; repo học viên không có `gt/`
> nên các ô đó ghi "chờ chấm", không điền số tự đo.

## 00 · Header

| Mục | Giá trị |
|---|---|
| Group ID | G07 |
| Challenge | T07 · Automatic Face/Plate Redaction + Review Queue |
| Thành viên + vai trò | 1. Nguyen Xuan Khanh (Lead / Integrator + Core algorithm + Review) · 2. ____ · 3. ____ · 4. ____ · 5. ____ |
| Git URL | https://github.com/Khanh-AI-ML-DL/K5-Day6-NguyenXuanKhanh-2A202603075-L2-DataGovernanceAndPrivacy.git |
| Tag `demo-180m` (commit) | ____ |
| Tag `final-24h` (commit) | ____ |
| Deadline (giờ VN) | ____ |

## 01 · Problem statement

- Ai bị ảnh hưởng khi hệ thống sai: người đi đường có mặt và biển số xe xuất hiện trong ảnh thu thập. Nếu bỏ sót vùng PII, dữ liệu phát hành làm lộ danh tính người thật. Nếu che quá rộng, dữ liệu mất giá trị cho bài toán huấn luyện.
- Điểm nghẽn: phải che mặt/biển số cho nhiều ảnh mà không thể duyệt tay từng ảnh; mặt nhỏ, xa, trong bóng tối là chỗ bộ dò dễ sót.
- Đầu vào → đầu ra: 60 ảnh synthetic 640×360 (`fixtures/t07_redaction/`) → `out/poc/redactions.csv` (`image_id, x, y, w, h`, các vùng đã che cuối cùng sau review) + ảnh đã che (`out/poc/redacted/`) + review queue + audit log + `decision.json`.
- Tiêu chí GO / HOLD / REWORK (chốt trước khi chạy eval):
  - HOLD (`REVIEW_PENDING`): chưa có người duyệt hàng chờ.
  - REWORK (`LOW_DETECTOR_CONFIDENCE`): đã duyệt nhưng độ tin cậy trung bình của các vùng < 0,6.
  - GO (`REVIEWED_QUEUE_OK`): hàng chờ đã duyệt, có audit log.
- Giả định cần DPO/legal xác nhận: output là dữ liệu "redacted/pseudonymised", không phải "anonymised"; không tuyên bố tuân thủ GDPR hay luật Việt Nam.

## 02 · Data & provenance

| Split | N | Nguồn / link | License | Synthetic? |
|---|---|---|---|---|
| dev | 36 ảnh | `fixtures/t07_redaction/` (do giảng viên cung cấp) | CC0 | Có |
| eval / heldout | 24 ảnh | `fixtures/t07_redaction/` | CC0 | Có |

- Schema: `images.csv` (`image_id, path, width, height, split`). Ground truth (`gt/gt_regions.csv`, vùng face/plate theo `condition`) chỉ giảng viên giữ.
- Seed: 42. Chia split theo cột `split` có sẵn (dev 60% / eval 40%).
- `manifest.csv` verified (sha256): có. `python -m tools.preflight` báo "hashes match" cho `t07 fixtures/t07_redaction`; pipeline cũng kiểm hash ở bước `load()` (`strict_provenance: true`).
- Dữ liệu thật: không dùng.
- Pipeline xử lý cả 60 ảnh; evaluator chỉ tính trên split eval (cấu hình `eval_params.split: eval`).

## 03 · Research & design

- Tham khảo (lấy từ `docs/tracks/T07.md`; nhóm chưa mở các link nên chưa ghi ngày xem): OpenCV Zoo YuNet (mặt, MIT), Meta EgoBlur (mặt + biển số), PP4AV (benchmark cảnh lái xe), EDPB Guidelines 02/2026 về anonymisation. Lý do chọn kỹ thuật cổ điển: chạy CPU, offline, không cần tải weights, kết quả tái lập được; dữ liệu synthetic có cấu trúc đơn giản (mặt = hình elip có 2 chấm mắt, biển số = hộp trắng có chữ tối).
- Data flow: `load` (kiểm manifest) → `process` (dò mặt/biển số, nới khung, gán độ tin cậy) → `rank` (top-B=15 độ tin cậy thấp nhất) → `review` (reviewer approve/reject/edit/escalate/skip, áp override, ghi audit log) → `decide` (GO/HOLD/REWORK) → ghi `redactions.csv` + ảnh đã làm mờ.
- Thành phần offline: không dùng model/weights. Chỉ dùng OpenCV 5.0.0, NumPy, pandas. OpenCV 5.0 không còn Haar `CascadeClassifier` nên không dùng được baseline Haar.
- Trade-offs:
  - Bộ dò cổ điển nhanh và dễ giải thích nhưng bám sát cấu trúc ảnh synthetic, sẽ không tổng quát hoá sang ảnh thật.
  - Nới khung (`padding_ratio=0,25`) giảm bỏ sót nhưng tăng over-redaction.
  - Hạ ngưỡng độ tin cậy (`min_conf=0,25`) bắt thêm vùng nhưng tăng số ô sai cần người duyệt.
  - Hàng chờ ưu tiên ô độ tin cậy thấp thì lấy được ô sai, nhưng không thấy được vùng bị bỏ sót hoàn toàn (chưa có khung).

## 04 · Baseline vs PoC

- Baseline: `src/baseline_t07.py`, chế độ `skin` (mặc định): tìm vùng màu da trong HSV, che bằng khung cố định, không có hàng chờ hay người duyệt. Cộng với baseline tầm thường `blur_all`: che toàn bộ ảnh. Commit: ____
- PoC: `src/pipeline.py`: dò mặt bằng cặp chấm mắt + loang màu, dò biển số bằng hộp trắng có chữ tối, nới khung 25%, hàng chờ top-15, review, audit log, quyết định có `reason_code` và `owner`. Config: `configs/lab.yaml`. Commit: ____
- Cùng budget / cùng heldout / cùng seed: xác nhận có (cùng 60 ảnh, seed 42, `--offline`, B=15; evaluator chỉ tính split eval).

**Bảng 1 · Baseline vs PoC & hard slices**

Các chỉ số chính thức do giảng viên chạy `src.evaluate` + `src.compare` bằng overlay `gt/` rồi dán vào đây (không gõ tay). Cột số đo được của nhóm:

| Metric (unit) | Baseline (skin) | Baseline tầm thường (blur_all) | PoC | Ghi chú |
|---|---|---|---|---|
| Primary: PII Miss Rate (region, lower) | chờ chấm |  | chờ chấm | N vùng GT eval do giảng viên đếm |
| Guardrail: over-redaction ≤ 0,3 | chờ chấm |  | primary chỉ hợp lệ khi guardrail đạt |
| Số khung xuất ra (60 ảnh) | 279 | 60 (cả ảnh) | 222 trước review, 220 sau review | đếm từ `redactions.csv` |
| Review@B | không có hàng chờ | không có | B=15; reject 2, escalate 6, approve 7 | `reports/review_queue.csv` |
| Runtime, 60 ảnh | 0,21 s | 0,04 s | 0,84 s (≈ 14 ms/ảnh) | MacBook Air, CPU, Python 3.15.2 |

## 05 · Evaluation protocol

- Primary: PII Miss Rate = số vùng GT có pixel coverage < 0,9 / số vùng GT, tính trên split eval. Đơn vị: vùng (face/plate). Chi tiết: `reports/metric_contract.md`.
- Guardrail: over-redaction (pixel che ngoài GT / pixel che) ≤ 0,3.
- Threshold / budget chốt trước khi chạy eval: `padding_ratio=0,25`, `min_conf=0,25`, `min_mean_conf=0,6`, B=15. Chọn bằng quan sát overlay trên ảnh dev, không tinh chỉnh trên eval.
- Độ bất định: số vùng eval nhỏ, nhóm đề xuất báo Wilson CI 95% cho miss rate (do giảng viên/nhóm tính khi có số vùng).
- Lệnh evaluate: `python -m src.evaluate --track t07 --pred out/poc/redactions.csv --run-name poc --out reports/metrics_poc.json` (cần overlay giảng viên).
- Nhóm không tự gán nhãn ảnh dev nên chưa có ước lượng độc lập miss rate trên dev; chỉ có đánh giá định tính 15 ô của hàng chờ.

## 06 · Result slices

| Slice | N | Baseline | PoC | Ghi chú |
|---|---|---|---|---|
| type=face | chờ chấm | chờ chấm | chờ chấm | giảng viên báo theo `type` |
| type=plate | chờ chấm | chờ chấm | chờ chấm | |
| condition=dark / small / blur / occluded | chờ chấm | chờ chấm | chờ chấm | giảng viên báo theo `condition` |
| Hàng chờ B=15 (quan sát của nhóm) | 15 | n/a | 7 approve, 2 reject, 6 escalate | 8/15 ô cần can thiệp hoặc theo dõi |

- **Failure case 1 — chữ biển số tràn ra ngoài khung (6/15 ô hàng chờ).** Các mã: `pii_13#4`, `pii_35#6`, `pii_49#4`, `pii_11#1`, `pii_34#6`, `pii_25#3`. Quan sát: khung chỉ bao phần nhãn trắng, một phần ký tự biển số nằm ngoài mép phải khung. Nguyên nhân gốc (giả thuyết nhóm): bộ dò biển số phân vùng theo màu trắng, nên ký tự tối vượt ra ngoài nhãn không được tính, và padding 0,25 không đủ bù. Hệ quả: coverage có thể < 0,9 nên tính là bỏ sót. Hướng sửa: nới khung theo chiều ngang riêng cho biển số.
- **Failure case 2 — mặt lớn trong ảnh tối bị bỏ sót (`pii_56`).** Quan sát: khung `pii_56#1` đúng, nhưng mặt lớn bên trái cùng ảnh không có khung nào. Nguyên nhân gốc (giả thuyết nhóm): trong ảnh tối và khuôn mặt tông nâu sẫm, hai chấm mắt ít tương phản nên không qua ngưỡng blackhat (18), và bộ dò mặt cần cặp mắt. Đây đúng là failure mode đề bài nêu (mặt trong bóng tối). Hàng chờ chỉ ưu tiên ô độ tin cậy thấp nên không thể gắn cờ vùng chưa có khung. Chưa kiểm tra bằng chạy lại sau khi sửa ngưỡng.
- **Ô dương tính giả:** `pii_35#5` và `pii_00#5` là khung nằm trên bánh xe/người đi xe máy; reviewer reject (xem audit log).
- Latency / compute: MacBook Air (Apple silicon, arm64), CPU, Python 3.15.2, OpenCV 5.0.0.93; PoC 0,84 s cho 60 ảnh (≈ 14 ms/ảnh, ≈ 1,4 s / 100 ảnh).

## 07 · Human & decision

- Review queue: `reports/review_queue.csv` (copy từ `out/poc/review_queue_reviewed.csv`), B=15, xếp theo `score = 1 - độ tin cậy` giảm dần.
- Override + audit log: `out/poc/audit_log.jsonl` (JSONL ghi thêm, mỗi dòng có `ts, actor, action, item_id, before, after, reason`). `reject` xoá khung (ghi khung trước đó), `edit` thay khung theo `override_value` dạng `x,y,w,h`, `approve/escalate/skip` ghi nhận không đổi khung. Kết quả review: 15 ô, 2 reject, 7 approve, 6 escalate.
- Decision policy: HOLD khi chưa có review; REWORK khi độ tin cậy trung bình < 0,6; GO khi đã review. Mọi quyết định có `reason_code` và `owner` ("Nguyen Xuan Khanh - reviewer"), ghi vào `decision.json` và `reports/decision_log.csv`. Quyết định hiện tại: GO / `REVIEWED_QUEUE_OK`.
- Hạn chế của chính sách (nhóm tự nhận): GO hiện nghĩa là "đã qua review", chưa đảm bảo không còn bỏ sót. Còn 6 ô `escalate` chưa xử lý và vùng bị bỏ sót hoàn toàn (failure case 2). Nếu sửa, nên chuyển sang HOLD (`ESCALATION_OPEN`) khi còn ô escalate chưa giải quyết.
- Khi nào HOLD/REWORK thay vì tự động release: khi chưa có reviewer, khi còn ô escalate mở, hoặc khi độ tin cậy trung bình thấp. Người ký duyệt phát hành là owner.

## 08 · Reproduce & tests

```bash
make setup
python -m src.main --track t07 --pipeline src.baseline_t07 --output out/baseline --run-name baseline --offline --seed 42
python -m src.main --track t07 --pipeline src.baseline_t07 --output out/baseline_blurall --run-name blur_all --offline --seed 42
make run TRACK=t07
python -m src.evaluate --track t07 --pred out/poc/redactions.csv --run-name poc --out reports/metrics_poc.json   # cần overlay giảng viên
make test
```

- Output của `pytest -q`: toàn bộ test qua, 0 failed; một số test cần `gt/` tự bỏ qua (skip). Có 2 test của nhóm ở `tests/test_t07_pipeline.py` (phát hiện biển số tổng hợp + khung nới không vượt ảnh; ảnh rỗng không có phát hiện).
- Máy / OS / Python: MacBook Air (Apple silicon) · macOS ____ · Python 3.15.2 (repo test trên 3.13; mọi gói đã có wheel và preflight 0 FAIL) · git sha: ____

## 09 · Limitations & ethics

- Rủi ro PII/license: fixtures là dữ liệu synthetic CC0, không có người hay biển số thật. Nếu dùng dữ liệu thật phải giữ local, không đưa lên repo công khai hay cloud.
- Bộ dò bám cấu trúc ảnh synthetic (chấm mắt, hộp trắng) nên gần như chắc chắn không tổng quát sang ảnh đường phố thật. Chưa đo trên dữ liệu thật.
- Chưa biết: miss rate thật theo từng slice (chờ chấm); số ô sai trong toàn bộ 220 khung ngoài 15 ô đã duyệt. Nếu scale 10×, người duyệt chỉ xem top-B nên tỉ lệ vùng không được xem sẽ tăng.
- Làm mờ không đồng nghĩa ẩn danh hoá (EDPB Guidelines 02/2026): còn vị trí, thời gian, ngữ cảnh. Gọi output là "redacted/pseudonymised". Không tuyên bố đạt GDPR hay luật Việt Nam; cần DPO/legal xác nhận.
- Kế hoạch cải tiến: nới khung ngang cho biển số; hạ ngưỡng mắt trong ảnh tối hoặc thêm bộ dò mặt dựa vào màu da; tự gán nhãn vài ảnh dev để ước lượng độc lập; đổi chính sách `decide()` để ô escalate mở sinh HOLD; thử YuNet/EgoBlur nếu giảng viên cấp weights, ghi checksum và license.

## 10 · Team contribution

**Bảng 2 · Individual contribution evidence** (đối chiếu `git shortlog -sne --all` và `reports/team_contributions.csv`)

| Học viên           | Git author/email         | Module / vai trò | Commit SHA / diff chứng minh | Trình bày? |
|--------------------|--------------------------|---|---|---|
| Nguyen Xuan Khanh  | khanhblackjaky@gmail.com | Integration, pipeline, review, report | ____ | ____ |
| Nguyen Xuan Khanh  |                          | Data + GT | | |
| Nguyen Xuan Khanh  |                          | Core algorithm | | |
| Nguyen Xuan Khanh  |                          | Metric + tests | | |
| Nguyen Xuan Khanh  |                          | Review queue + report | | |

## 11 · References

| URL | Ngày xem | 1–2 dòng ý nghĩa |
|---|---|---|
| OpenCV Zoo YuNet (xem link trong `docs/tracks/T07.md`) | ____ | Bộ dò mặt nhẹ chạy CPU, license mở; lựa chọn nếu có weights. |
| Meta EgoBlur (xem link trong `docs/tracks/T07.md`) | ____ | Che mặt và biển số, chậm trên CPU. |
| PP4AV, WACV 2023 | ____ | Benchmark công khai cho cảnh lái xe; mặt nhỏ làm bộ dò tổng quát thất bại. |
| EDPB Guidelines 02/2026 | ____ | Làm mờ chưa chắc là ẩn danh hoá; gọi output là redacted/pseudonymised. |

---

**Bảng 3 · Checklist chốt khi nộp**

- [ ] Link repo và tag cuối cùng trong README; input mẫu không chứa PII thật.
- [ ] Đúng 1 lệnh run, 1 lệnh evaluate, 1 lệnh tests; không cần download khi chạy.
- [ ] Có baseline/PoC metrics (do giảng viên chấm), ≥1 failure case, reviewer evidence, decision log.
- [ ] Có license/source provenance, data splits, seed, offline environment, GT/eval N.
- [ ] Mỗi thành viên có ≥1 commit thực chất và được nêu minh bạch trong bảng trên.