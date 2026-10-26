# Bắt nhịp lại dự án

Đọc file này nếu bạn quay lại sau một thời gian, hoặc vừa nhận dự án.
Nó trả lời: đang làm gì, các từ viết tắt nghĩa là gì, đã biết được gì,
và còn lại gì.

Chi tiết kỹ thuật ở [`../README.md`](../README.md) và
[`DEVIATIONS.md`](DEVIATIONS.md). File này là lớp trên.

---

## 1. Đồ án này là gì

Tái lập và kiểm toán bài **Fox** (ICML 2026), một phương pháp giảm ảo giác
(*hallucination*) cho mô hình ngôn ngữ-thị giác, không cần huấn luyện lại.

Hai câu hỏi, theo thứ tự:

1. **Số của paper có tái lập được không?**
2. **Nếu có, nó đến từ cơ chế mà paper mô tả hay từ chỗ khác?**

Câu 1 đã trả lời xong phần lớn. Câu 2 là nơi kết quả thú vị nằm.

---

## 2. Từ điển thuật ngữ

### 2.1 Mô hình và bài toán

| Từ | Nghĩa |
|---|---|
| **LVLM** | Large Vision-Language Model — mô hình nhận ảnh + văn bản, sinh văn bản |
| **LLaVA-1.5-7B** | LVLM cụ thể dùng trong mọi thí nghiệm ở đây. 7 tỉ tham số, fp16 nặng 13.16 GiB |
| **hallucination** | Mô hình mô tả vật thể **không có** trong ảnh. Vấn đề paper muốn giảm |
| **Fox** | Tên phương pháp của paper. Cũng là tên repo này (`fox-repro`) |

### 2.2 Hai benchmark

> **POPE và CHAIR là THƯỚC ĐO, không phải phương pháp.** Chúng không cạnh
> tranh với Fox — chúng là cái dùng để chấm Fox. Nhầm chỗ này sẽ đọc sai
> toàn bộ bảng kết quả.
>
> - **Phương pháp** (cái được so sánh với nhau): `sampling` (baseline), `fox`,
>   VCD, ICD, OPERA, CausalMM, SID. Đây đều là cách *giải mã* — chúng thay
>   đổi mô hình sinh token thế nào.
> - **Thước đo** (cái dùng để chấm): POPE và CHAIR.
>
> Mỗi ô trong bảng kết quả là *(một phương pháp) chấm bởi (một thước đo)*.
> Ví dụ "fox trên CHAIR_S = 46.74".

| Từ | Nghĩa |
|---|---|
| **POPE** | Polling-based Object Probing Evaluation. 3000 câu hỏi **có/không** dạng *"Is there a dog in the image?"*. Sinh đúng **1 token**. Đo khả năng **phân biệt** |
| **POPE random / popular / adversarial** | Ba split, độ khó tăng dần. `random` lấy vật thể ngẫu nhiên; `popular` lấy vật thể hay xuất hiện; `adversarial` lấy vật thể **hay đi cùng** vật thể có thật — khó nhất, và là nơi paper tuyên bố cải thiện lớn nhất |
| **CHAIR** | Caption Hallucination Assessment with Image Relevance (Rohrbach et al., 2018). Một **thước đo**, không phải phương pháp. Cho model **mô tả ảnh** (~100 từ), rồi đối chiếu danh sách vật thể nó nhắc tới với chú thích thật của COCO, đếm phần bịa. 500 ảnh COCO val2014. Đo khả năng **sinh** |
| **COCO val2014** | Bộ ảnh gốc, 40,504 ảnh. POPE và CHAIR đều lấy ảnh từ đây |

### 2.3 Các chỉ số

| Từ | Nghĩa | Cao hay thấp là tốt |
|---|---|---|
| **accuracy** | POPE: tỉ lệ trả lời đúng | cao |
| **F1** | POPE: trung bình điều hoà precision/recall | cao |
| **yes-ratio** | POPE: tỉ lệ trả lời "yes". **Không phải trang trí** — một phương pháp có thể nâng accuracy chỉ bằng cách trả lời "no" nhiều hơn | — (chẩn đoán) |
| **CHAIR_S** | **S = sentence.** Tỉ lệ **caption** chứa ít nhất một vật thể bịa | **thấp** |
| **CHAIR_I** | **I = instance.** Tỉ lệ **lần nhắc vật thể** là bịa | **thấp** |
| **recall** | tỉ lệ vật thể thật được nhắc tới | cao |

Hai chỉ số CHAIR đếm trên hai đơn vị khác nhau:

```
CHAIR_S = (số caption có >= 1 vật thể bịa) / (tổng số caption)
CHAIR_I = (số lần nhắc vật thể bị bịa)     / (tổng số lần nhắc vật thể)
```

Số thật của baseline (greedy, 500 ảnh):

| | tử số | mẫu số | kết quả |
|---|---|---|---|
| CHAIR_S | 246 caption có bịa | 500 caption | **49.20%** |
| CHAIR_I | 520 lần nhắc bị bịa | 3762 lần nhắc | **13.82%** |

Cùng một caption đóng góp vào cả hai theo cách khác nhau. Ví dụ ảnh `117527`
(xem [`../samples/`](../samples/)): caption nhắc vật thể 14 lần, trong đó 5
lần là bịa → đóng góp **1** vào tử số CHAIR_S, và **5/14** vào CHAIR_I.

Vì sao cần cả hai: CHAIR_S nói *"bao nhiêu caption có vấn đề"*, CHAIR_I nói
*"vấn đề nặng cỡ nào"*. Một mô hình bịa đúng một vật thể trong mọi caption có
CHAIR_S cao nhưng CHAIR_I thấp. CHAIR_S cũng nhiễu hơn nhiều — ở n=500 nó chỉ
phân giải được khoảng 8.85 điểm, còn CHAIR_I có 3762 lần nhắc nên phân giải
được ~2.24 điểm.

> **Bẫy quan trọng:** CHAIR_S có thể bị "gian lận" bằng cách **nói ít đi** —
> caption nhắc ít vật thể thì bịa ít vật thể. Vì vậy mọi số CHAIR trong repo
> này luôn đi kèm `recall` và `#correct objects`. Có một bộ kiểm tự động
> (`RECALL_DROP_FLAG`) gắn cờ ⚠ cho arm nào recall tụt ≥10%. Nó đã bắt được
> hai arm thật: `ld=bf16` và `vcd`.

### 2.4 Cách sinh văn bản

| Từ | Nghĩa |
|---|---|
| **sampling** / **nucleus sampling** | Lấy mẫu ngẫu nhiên từ phân phối, `top_p=0.9`, `temperature=1.0`. **Có ngẫu nhiên** → đổi seed là đổi kết quả. Đây là chế độ chính của paper |
| **greedy** | Luôn lấy token xác suất cao nhất (argmax). **Tất định** → chạy lại cho kết quả giống hệt từng byte. Đã kiểm chứng: 50/50 mẫu khớp giữa seed 42 và 997 |
| **top_p** | Chỉ lấy mẫu từ nhóm token nhỏ nhất có tổng xác suất ≥ 0.9 |
| **max_new_tokens** | POPE = 1, CHAIR = 512 (lấy từ code gốc) |

### 2.5 Cơ chế Fox — các ký hiệu

| Ký hiệu | Nghĩa |
|---|---|
| **γ (gamma)** | Hằng số trừ khỏi attention logit. Code gốc hardcode **10000.0**, paper không hề nhắc |
| **k** | Tỉ lệ head bị can thiệp mỗi layer. Code **0.35**, paper nói **0.45** |
| **τ_JS (tau_js)** | Ngưỡng Jensen-Shannon divergence để hợp nhất hai nhánh. Code **0.15**, paper nói **0.2** |
| **α (alpha)** | Hệ số khuếch đại khi hợp nhất. **2.0**, cả hai khớp |
| **β (beta)** | Tỉ lệ cắt khi ước lượng xung đột. **0.1**, cả hai khớp |
| **layer window** | Các layer bị can thiệp. Code **2–15**, paper nói **2–13** |
| **m_sys** | Khối lượng attention trên các key **trước ảnh** (system prompt). Paper gọi đây là *"primary anchor for latent priors"* — gốc của đường tắt cần cắt |
| **H_vis** | Entropy attention trên token ảnh |
| **S = m_sys · H_vis** | Điểm chọn head: head nào vừa bám system prompt vừa tán loạn trên ảnh thì bị can thiệp |

### 2.6 Các "arm" (nhánh thí nghiệm)

Một **arm** là một cấu hình được so sánh. Tất cả đều chạy trên cùng dữ liệu,
cùng seed, chỉ khác đúng một thứ.

| Arm | Làm gì | Vai trò |
|---|---|---|
| `sampling` | Không can thiệp gì | **baseline** — mọi gap đo từ đây |
| `fox` (`as_implemented`) | Đúng như code gốc: trừ γ khỏi **mọi** key, hai nhánh, hợp nhất | phương pháp của paper |
| `fox off` | Giữ hợp nhất hai nhánh, **tắt** can thiệp | tách phần đóng góp của riêng fusion |
| `fox ulp_control` | **Chỉ** làm tròn logit lên lưới fp16, **không trừ γ** | nếu bằng `fox` → cơ chế là làm tròn |
| `fox noise_control` | Nhiễu ngẫu nhiên cùng biên độ thay vì trừ hằng số | loại trừ "nhiễu bất kỳ cũng được" |
| `fox as_described` | Chỉ bias trên key của **token ảnh**, đúng như Appendix B **mô tả** | cái paper nói vs cái code làm |
| `fox boost_visual` | **Tăng** attention lên token ảnh (ngược dấu) | thí nghiệm paper lập luận từ đó mà **không bao giờ chạy** |
| `fox ld=fp32` / `ld=bf16` | Lưu logit ở fp32 / bf16 thay vì fp16 | trục mà cơ chế thật nằm trên |
| `vcd` | Nhánh nhiễu kiểu VCD trong fusion của Fox. **Không phải VCD thật** — xem DEVIATIONS | tham chiếu, không so với paper được |

### 2.7 Số học dấu chấm động

| Từ | Nghĩa |
|---|---|
| **logit** | Điểm số trước softmax |
| **fp16 / bf16 / fp32** | Độ chính xác lưu trữ: 16-bit nửa chính xác / 16-bit "brain float" / 32-bit |
| **ULP** | *Unit in the Last Place* — khoảng cách giữa hai số biểu diễn được liền kề. Càng lớn thì lưới càng thô |
| **requantisation** | Việc ép một số về lưới ULP của kiểu dữ liệu, tức làm tròn |

ULP của fp16 tại các γ (đo được, không phải suy ra):

| γ | ULP fp16 |
|---|---|
| 1e2 | 0.0625 |
| 1e3 | 0.5 |
| **1e4** (mặc định của code) | **8.0** |
| 6e4 | 32.0 |

Logit thật có biên độ O(1)–O(30). Lưới 8.0 là **thô bằng cả dải tín hiệu**.

### 2.8 Vận hành

| Từ | Nghĩa |
|---|---|
| **job** | Một lệnh `generate` — một (arm, benchmark, split, decoding, seed) |
| **run file** | File JSONL một job ghi ra. Tên dẫn xuất từ cấu hình, nên hai cấu hình khác nhau **không thể** ghi đè nhau |
| **variant hash** | Phần băm trong tên file, từ các trường làm đổi kết quả |
| **provenance** | Siêu dữ liệu ghi trong **từng record**: commit, GPU, torch version, mọi hyperparameter. Dùng để phát hiện run bị trộn |
| **session** | Một lần mở Kaggle notebook. Tối đa 12 h, **độc lập hoàn toàn** — đĩa trống, phải tải lại dữ liệu |
| **round** (r1…r5) | Một đợt chạy. Tag zip là `fox_r<N>_s<M>` |
| **seed** | Hạt giống ngẫu nhiên. Chỉ ảnh hưởng `sampling`, không ảnh hưởng `greedy` |
| **arm identity** | Chuỗi định danh arm trong report, ví dụ `fox ulp_control`, `fox ld=bf16` |

### 2.9 Thống kê

| Từ | Nghĩa |
|---|---|
| **paired / ghép đôi** | So sánh hai arm trên **cùng** mẫu và **cùng** seed. Loại bỏ phương sai do độ khó mẫu |
| **Wilson interval** | Khoảng tin cậy cho một tỉ lệ nhị thức |
| **McNemar exact** | Kiểm định ghép đôi cho accuracy. Chỉ các cặp **bất đồng** mang thông tin |
| **bootstrap** | Lấy mẫu lại để ước lượng khoảng tin cậy khi không có công thức |
| **cluster bootstrap** | Bootstrap theo **caption** cho CHAIR_I — vì 3800 lần nhắc đến từ 500 caption **không phải** 3800 quan sát độc lập |
| **Student-t** | Phân phối dùng khi số mẫu nhỏ. Ở 10 seed, hệ số 95% là 2.262 chứ không phải 1.960 |
| **TOST** | *Two One-Sided Tests*. Đảo giả thuyết: H₀ = "khác nhau ít nhất Δ". Bác bỏ được → **bằng chứng dương cho tương đương**. Cần vì p cao của kiểm định thường **không** chứng minh được hai thứ giống nhau |
| **Holm-Bonferroni** | Hiệu chỉnh đa so sánh. Report có hàng chục so sánh; ở α=0.05 cứ 20 cái thì 1 cái "có ý nghĩa" do nhiễu |
| **equivalence margin (Δ)** | Khoảng chênh lệch vẫn coi là "như nhau". Là một **phán đoán**, nên luôn được in ra cạnh kết quả |

> **Hiểu nhầm cần tránh:** `p = 0.671` **không** nghĩa là "67.1% khả năng hai
> arm tương đương". Nó là xác suất quan sát được dữ liệu này **nếu** chúng
> tương đương. Và p cao thì **càng dễ đạt khi đo càng nhiễu** — 2 seed cho p
> cao hơn 10 seed. Đó chính là lý do phải dùng TOST.

---

## 3. Paper nói gì, và ta kiểm chứng gì

**Luận điểm của paper:** ảo giác đến từ một "đường tắt bệnh lý"
`X_sys → H_R → Y_t` — mô hình bám vào **system prompt** (tiên nghiệm ngôn ngữ)
thay vì vào **ảnh**. Fox tìm các head có hành vi đó (`S = m_sys · H_vis`),
**triệt tiêu** chúng ở layer 2–15, rồi hợp nhất hai nhánh forward.

**Điều đọc code phát hiện ra:** phép "triệt tiêu" là trừ γ khỏi **mọi** key
position tại một query position. Mà softmax **bất biến với phép tịnh tiến** —
và softmax của HuggingFace ổn định số học, cộng dồn ở fp32. Nên phép trừ
**triệt tiêu hoàn toàn**. Thứ duy nhất sống sót là phép **cast ngược về fp16**,
ép hàng logit lên lưới ULP 8.0.

Tức là: nếu giả thuyết này đúng, hiệu ứng là một **artifact làm tròn**, không
phải một can thiệp nhân quả. Toàn bộ các arm ở §2.6 tồn tại để kiểm điều đó.

---

## 4. Đã chạy những gì

| Round | Session | Nội dung | Kết quả |
|---|---|---|---|
| **r1** | 1 | Session thử đầu tiên | Prompt sai (tự chế) → baseline POPE cao hơn paper 5.13 điểm |
| **r2** | 1 | Prompt **đúng từng byte** với code gốc | baseline 77.17 vs paper 76.77 ✅ |
| **r3** | 12 | 117 job: POPE ×3 split ×10 seed, CHAIR ×10 seed, greedy, mọi arm cơ chế | 240,100 generation, 0 lỗi |
| **r4** | 6 | `ld=bf16` ×10 seed, `boost_visual` ×10 seed, 4 draw CHAIR mới, `ld=fp32` ×10 seed | **Session 1–2 chạy nhầm trên CPU** → mất 20.8 h, `fp32` và chair_draws không chạy |
| **r5** | 3 (chưa chạy) | Làm lại phần r4 bị mất | 19.9 h |

Tổng cộng: **165 job** trong plan, **146 xong**, **313,047 generation** đã lưu.

---

## 5. Kết quả hiện tại

### 5.1 Tái lập

| artefact | gap của mình [CI95] | paper | kết luận |
|---|---|---|---|
| POPE-adv, sampling (10 seed) | +5.03 [+4.68, +5.38] | +5.16 | **tái lập** |
| POPE-random, sampling (3 seed) | +4.16 [+3.86, +4.45] | +4.20 | **tái lập** |
| POPE-popular, sampling (3 seed) | +4.13 [+2.63, +5.64] | +4.17 | điểm khớp, **chưa phân giải sau Holm** |
| POPE ×3 split, greedy | +0.20 / +1.00 / +3.33 | +0.10 / +1.14 / +3.33 | **tái lập** |
| CHAIR_S, sampling (10 seed) | −6.98 [−9.25, −4.71] | −11.20 | **không** |
| CHAIR_I, sampling (10 seed) | −2.67 [−3.37, −1.96] | −4.28 | **không** |

Mọi **baseline** POPE khớp trong 0.05 điểm → harness, dữ liệu, protocol đều
đúng. Những gì đo được là về phương pháp, không phải về setup.

### 5.2 Cơ chế (POPE-adversarial, sampling, 10 seed)

| arm | gap vs baseline | TOST vs `fox` (Δ=±0.5) |
|---|---|---|
| `fox` | +5.03 [+4.68, +5.38] | — |
| `fox ulp_control` | +5.11 [+4.91, +5.30] | **tương đương**, p=0.0154 |
| `fox ld=bf16` | +5.05 [+4.69, +5.41] | **tương đương**, p=0.0035 |
| `fox off` | +2.21 [+1.84, +2.58] | không, p=1.000 |
| `fox boost_visual` | +3.74 [+3.46, +4.02] | không, p=0.9998 |
| `fox ld=fp32` (greedy, n=3000) | **+0.03 [−0.06, +0.09]**, p=1.00 | hiệu ứng **biến mất** |

### 5.3 CHAIR (sampling, ghép đôi theo seed)

| arm | CHAIR_S gap | CHAIR_I gap | recall |
|---|---|---|---|
| `fox` (n=10) | −6.98 | −2.67 | 76.72 |
| **`boost_visual`** (n=9) | **−13.22** | **−4.77** | 76.23 |
| paper (cho Fox) | −11.20 | −4.28 | — |

`boost_visual` vs `fox` ghép đôi: CHAIR_S −6.42 [−7.74, −5.10], CHAIR_I
−2.17 [−2.51, −1.83], cả hai p<1e-4. **Recall không giảm** → không phải
degeneracy.

---

## 6. Ba phát hiện chính

**1. Hiệu ứng tái lập được, nhưng lời giải thích thì không.**
`ulp_control` (chỉ làm tròn, không trừ γ) và `bf16` (lưới thô hơn) đều **tương
đương thống kê** với `fox`. `ld=fp32` (giữ γ, bỏ làm tròn) đưa hiệu ứng về
**0.03 điểm**. Cơ chế là **phép cast logit**, không phải can thiệp attention.

**2. Cái paper *mô tả* không chạy.** `as_described` — đúng Appendix B — chỉ
được +0.40 điểm. Cái code *làm* thì chạy, và nó chạy bằng làm tròn.

**3. Tiền đề nhân quả không đứng, và thí nghiệm bị bỏ qua lại thắng.**
Bỏ system prompt đi thì gain **tăng** (+4.37 vs +3.33) — trong khi `m_sys`
được cho là gốc của đường tắt. Còn `boost_visual` — tăng attention ảnh thật,
đúng thứ câu chuyện của paper khuyến nghị — **thắng Fox trên CHAIR** mà không
mất recall.

---

## 7. Bản đồ repo

```
src/foxrepro/
  fox/
    reference.py     Chép nguyên văn code gốc. KHÔNG được dọn dẹp, tối ưu
                     hay sửa bug. Đây là "oracle" số học
    kernel.py        Bản viết lại sạch, đã chứng minh bit-identical với
                     reference.py trên tensor ngẫu nhiên, mọi dtype
    attention.py     Nối vào transformers qua AttentionInterface; vòng lặp
                     decode hai nhánh; chặn chạy trên CPU
  bench/
    generate.py      Chạy một job, ghi JSONL, resume được
    methods.py       Định nghĩa các arm, dựng prompt
    plan.py          165 job, chia session, theo dõi tiến độ, ingest/archive
    evaluate.py      Chấm điểm (CPU, không cần GPU) từ JSONL
    chair.py         Bộ chấm CHAIR
    stats.py         Wilson, McNemar, bootstrap, Student-t, TOST, Holm
    report.py        Dựng báo cáo Markdown
  data/
    loaders.py       Nạp POPE/CHAIR, dựng prompt template
    fetch_images.py  Tải đúng ảnh session cần (không phải 6.2 GB zip)
  assets/            POPE annotations, 5 danh sách ảnh CHAIR, synonym list

results/runs/        313k generation đã lưu — clone là có sẵn
results/reports/     Báo cáo đã chốt
docs/DEVIATIONS.md   MỌI khác biệt so với code gốc, phân loại theo việc
                     nó có thể làm đổi số hay không
scripts/
  require_gpu.py     Chặn session chạy nhầm trên CPU
  probe_host.py      Kiểm tra host trước khi tải 14 GB
run_kaggle_fox.sh    Điểm vào trên Kaggle
```

**403 test.** Chạy: `PYTHONPATH=src python3 -m unittest discover -s tests -t .`

---

## 8. Quy trình chạy

### Trên Kaggle

Sidebar → **Accelerator = GPU T4 × 2**. Bắt buộc — một T4 không đủ chứa
13.16 GiB fp16; hai cái thì đủ khi shard. **Không** dùng quantization: nó sẽ
làm đổi chính đại lượng đang nghiên cứu.

```bash
!git clone -q https://github.com/KiRzEa/fox.git /kaggle/working/fox
%cd /kaggle/working/fox
!bash run_kaggle_fox.sh 1        # 1..N, xem scripts/run_plan.json
```

Script tự: cài → kiểm GPU (dừng nếu thiếu) → tải đúng ảnh cần → nạp kết quả
cũ từ `results/runs` → chạy → đóng gói zip sau **mỗi** job.

**Mỗi session độc lập hoàn toàn.** Đĩa trống, phải tải lại. Resume hoạt động
được là vì `results/runs/` **được commit vào repo** (330 MB JSONL → 18.5 MB
trong git), nên `git clone` đã mang theo mọi job đã xong.

### Ở máy local, sau khi tải zip về `runs_inbox/`

```bash
python3 -m foxrepro.bench.plan ingest
python3 -m foxrepro.bench.plan archive
A=~/.cache/fox-repro/local/data/annotations
python3 -m foxrepro.bench.evaluate \
    --instances $A/instances_val2014.json \
    --captions  $A/captions_val2014.json \
    --pope-dir src/foxrepro/assets/pope --format both
python3 -m foxrepro.bench.report --out results/reports/final_rN.md
```

### Tạo lại plan

```bash
python3 -m foxrepro.bench.plan plan --budget-hours 10 --round rN \
    --calibrate logs/_job_times_r<N-1>.tsv --out scripts/run_plan.json
```

---

## 9. Còn lại gì

**r5 — 3 session, 19.9 h:**

| # | giờ | nội dung | để làm gì |
|---|---|---|---|
| 1 | 9.79 | chair_draws d1–d3 · `ld=fp32` seed 42–43 | đo confound tập ảnh CHAIR |
| 2 | 9.99 | `ld=fp32` seed 44–51 · chair_draws d4 | khép lập luận cơ chế |
| 3 | 1.51 | `boost_visual` CHAIR seed 45 (tiếp từ 447/500) | đủ 10 seed |

**Tuỳ chọn, chưa lên lịch:** POPE-popular lên 10 seed (~19 h) để nó sống sót
qua Holm.

**Sau r5 thì nên dừng chạy và bắt đầu viết.**

---

## 10. Những lần đã sai — đừng lặp lại

Mỗi cái đều tốn thời gian thật và đều đã có test chặn.

| Lỗi | Hậu quả | Đã chặn bằng |
|---|---|---|
| Nạp checkpoint chưa convert | vision tower khởi tạo **ngẫu nhiên**, caption trôi chảy nhưng không liên quan ảnh, không metric nào bắt được | `_refuse_unconverted_checkpoint` |
| Prompt tự chế | baseline POPE cao hơn paper 5.13 điểm | `reference_prompt()` + test byte-identical |
| Kết luận "release không gửi system prompt" | sai — nó đến từ `eval_configs/*.yaml`, nối bằng `+` **không có dấu cách** | test khớp từng byte |
| `job_id()` thiếu seed | 10 seed dùng chung một id, 9 job báo pending mãi mãi | test id duy nhất |
| `N_SAMPLES.get(..., 0)` | split mới trông như **miễn phí**, budget mất nghĩa | giờ `raise` |
| `group_by_seed` gộp run khác kích thước | probe 50 mẫu bị bình quân vào arm 500 mẫu | số mẫu nằm trong khoá |
| `_index` khoá theo `method` | mọi control arm đụng với `fox` | khoá theo **arm** |
| Calibration lấy dòng cuối | retry lỗi xoá mất đo tốt → job 93 phút bị tin là 6 giây | bỏ rc≠0, **cộng** các dòng cùng job |
| Kaggle chưa gắn GPU | **20.8 h mất trắng**, 724 s/caption thay vì 7.8 s | `require_gpu.py` + `_refuse_cpu_only_host` |
| Đọc CHAIR_S mà không xem recall | `bf16` nhìn như thắng 23.6 điểm, thật ra recall tụt 79→60 | cờ ⚠ tự động |
| Dùng p cao làm bằng chứng tương đương | luận điểm trung tâm dựa trên nền yếu | TOST |
| Không hiệu chỉnh đa so sánh | tự mắc đúng lỗi đang phê phán paper | Holm |
