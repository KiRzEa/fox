# Tổng quan đồ án — tái lập và kiểm toán Fox (ICML 2026)

Tài liệu này dành cho người **mới tham gia đồ án**. Đọc hết là đủ hiểu đồ án
làm gì, đang ở đâu, và bạn nhận được việc gì. Muốn đi sâu thì có bản đồ ở
mục 9.

---

## 1. Đồ án làm gì, trong một đoạn

Paper **Fox** (Liu Yu et al., ICML 2026, PMLR v306 — *"Dismantling Pathological
Shortcuts: A Causal Framework for Faithful LVLM Decoding"*) nói rằng mô hình
thị giác–ngôn ngữ bịa ra vật thể không có trong ảnh vì một số **attention head
cụ thể** ở các bước giải mã quan trọng tách khỏi bằng chứng thị giác và bám vào
tiên nghiệm ngôn ngữ neo ở system prompt. Fox đề xuất chẩn đoán các head đó,
can thiệp vào attention logit của chúng, rồi hợp nhất hai nhánh.

Đồ án này **tái lập** các con số đó trên phần cứng của mình, và **kiểm toán**
xem cơ chế mà paper mô tả có thật là thứ tạo ra hiệu ứng hay không.

Hai việc đó khác nhau, và sự khác nhau chính là đóng góp của đồ án: chúng ta
tái lập được hiệu ứng nhưng **không** tái lập được lời giải thích.

## 2. Vì sao đồ án này đáng làm

Một bài tái lập chỉ xác nhận "số có ra giống không" thì giá trị thấp. Chỗ có
giá trị là: một phương pháp có thể cho ra đúng con số đã công bố **vì một lý do
hoàn toàn khác** với lý do mà paper viết. Khi đó bảng số liệu không phát hiện
được gì — phải thiết kế thí nghiệm riêng để phát hiện.

Đó là điều đã xảy ra ở đây, và nó xảy ra ở **ba** phương pháp khác nhau (Fox,
OPERA, CausalMM). Xem mục 5.

## 3. Phạm vi — và những gì nằm ngoài

**Phần cứng:** Kaggle Notebook, 2× Tesla T4 16 GB, quota ~30 GPU-h/tuần. Mỗi
session tối đa 12h nên ta luôn lên kế hoạch dưới 11h.

| | |
|---|---|
| **Backbone** | `llava-hf/llava-1.5-7b-hf`, fp16, `device_map="auto"` (shard qua 2 GPU) |
| **Benchmark** | POPE (3 split × 3000 câu hỏi yes/no) và CHAIR (500 ảnh COCO val2014, sinh caption) |
| **Phương pháp** | `sampling` (baseline), `fox`, `vcd`, `icd`, `sid`, `causalmm`, `opera` |

**Nằm ngoài, và vì sao** — ghi rõ để báo cáo nói được thay vì để lỗ trống:

- **InstructBLIP / Shikra** (cột 2 và 3 của mọi bảng chính). InstructBLIP chạy
  một *bản copy thứ hai* của phần can thiệp trong release, bỏ `m_sys` và dùng
  layer 4–10 với k=0.4 — là một phương pháp khác, không phải đổi tham số.
  Shikra không có bản chuyển đổi HF được bảo trì.
- **MME / MMBench / MathVista** (Figure 5 và phụ lục). Cần harness viết mới;
  đáng chú ý là release của Fox **không hề có script MME** dù báo cáo kết quả
  MME.
- **GPT-4V judge** (Figure 6). Cần API key trả phí. Protocol nằm ở Appendix
  C.3. Đang tạm hoãn theo quyết định của nhóm.

Không bao giờ dùng **checkpoint lượng tử hoá** (gptq-4bit, 8bit). Đại lượng
đang nghiên cứu *chính là* hiệu ứng độ chính xác số học, nên lượng tử hoá sẽ
phá đúng thứ ta đang đo.

## 4. Nguyên tắc thiết kế quan trọng nhất

Năm trong sáu phương pháp so sánh là **tự implement** từ source của từng
paper. Nên repo có một cấu trúc hai tầng mà bạn cần hiểu trước khi sửa bất kỳ
dòng nào:

```
src/foxrepro/fox/reference.py   ← chép NGUYÊN VĂN từ code đã phát hành
src/foxrepro/fox/kernel.py      ← bản port sạch, test khẳng định TRÙNG BIT
```

`reference.py` là **oracle số học**. Nó xấu, nó giữ nguyên cả những chỗ code
gốc không khớp với văn bản paper. Đó là chủ ý: nó trả lời câu hỏi *"code của ta
có bằng code của họ?"*.

Nhưng nó **không** trả lời được hai câu hỏi khác:

1. *Code của họ có bằng paper của họ?* — Thường là không. Ví dụ: harness của
   SID không chạy nổi VCD; sampler hai-counterfactual của CausalMM là dead
   code; nửa retrospection của OPERA không thể kích hoạt trên POPE.
2. *Cơ chế có thực sự hoạt động trên model thật?* — Một phương pháp contrastive
   mà nhánh âm của nó **không** tệ hơn vẫn cho ra một con số accuracy. Một
   phương pháp mà quy tắc của nó chưa bao giờ đổi một quyết định nào cũng vậy.
   Không bảng nào nhìn thấy điều đó.

Nên có tầng thứ ba: [`docs/VALIDATION.md`](VALIDATION.md) — **6 dự đoán được
đăng ký trước** (V1–V6), viết xuống *trước khi* chạy, mỗi dự đoán kèm cách đọc
kết quả cho mọi khả năng. Hiện trạng: **18 dự đoán đúng, 1 sai.**

> Đây là nguyên tắc văn hoá của đồ án: nếu một thí nghiệm có thể được giải
> thích theo nhiều cách sau khi thấy số, thì phải chọn cách giải thích
> **trước**. Cái sai duy nhất tính đến nay là một dự đoán của chúng ta, không
> phải một bug — và nó được ghi lại đúng như vậy.

## 5. Phát hiện chính

### 5.1 Về Fox — hiệu ứng tái lập được, lời giải thích thì không

POPE-adversarial, sampling, 10 seed:

| arm | gap vs baseline | TOST vs `fox` (Δ=±0.5) |
|---|---|---|
| `fox` | +5.03 [+4.68, +5.38] | — |
| `fox ulp_control` | +5.11 [+4.91, +5.30] | **tương đương**, p=0.0154 |
| `fox logit_dtype=bf16` | +5.05 [+4.69, +5.41] | **tương đương**, p=0.0035 |
| `fox off` | +2.21 [+1.84, +2.58] | không, p=1.000 |
| `fox logit_dtype=fp32` (greedy, n=3000) | **+0.03 [−0.06, +0.09]** | hiệu ứng **biến mất** |

Đọc bảng này từ dưới lên:

- `ulp_control` **chỉ làm tròn** logit, không trừ γ, không can thiệp attention
  — và tương đương thống kê với Fox đầy đủ.
- `logit_dtype=fp32` **giữ nguyên** can thiệp attention nhưng bỏ phép làm tròn
  — và hiệu ứng về 0.03 điểm.

Vậy cơ chế là **phép cast logit**, không phải can thiệp attention.

Thêm hai điểm:

- **Cái paper *mô tả* không chạy.** Arm `as_described` — làm đúng Appendix B —
  chỉ được +0.40 điểm. Cái code *làm* thì chạy.
- **Tiền đề nhân quả không đứng.** Bỏ system prompt đi thì gain **tăng**
  (+4.37 vs +3.33), trong khi `m_sys` được cho là gốc của đường tắt. Và
  `boost_visual` — tăng attention ảnh thật, đúng thứ câu chuyện của paper
  khuyến nghị và paper bỏ qua — **thắng Fox trên CHAIR** (−13.22 vs −6.98
  CHAIR_S) mà **không mất recall**.

### 5.2 Về OPERA — con số POPE của nó là beam search

OPERA là ba cơ chế xếp lên nhau nhưng paper chỉ đặt tên cho một: (1) beam
search 5 beam, (2) hạn chế cứng 5 candidate, (3) hình phạt "over-trust" thưởng
cho việc chú ý vào ảnh.

Phân rã phải làm ở **protocol 10 token của chính OPERA**, vì ở harness 1 token
của Fox thì cả phương pháp chỉ được +0.07 — không có gì để phân rã. Ở 10 token
nó được +2.33, và ba arm chia nhau +2.33 đó:

| arm | accuracy | vs baseline | trùng `opera` | cặp có thông tin | p |
|---|---|---|---|---|---|
| `opera` | 81.47 | +2.33 | — | — | — |
| `opera_no_penalty` | **81.47** | +2.33 | 1496/1500 | 2 (1–1) | 1.0 |
| `opera_unrestricted` | **81.47** | +2.33 | 1474/1500 | 24 (12–12) | 1.0 |

Ba arm, một con số. **Toàn bộ +2.33 là của beam search.** Không cơ chế nào
trong hai cơ chế còn lại đóng góp gì đo được.

Và cách phát biểu phải chính xác hơn "hình phạt vô tác dụng". Telemetry:

| | 1 token | 10 token |
|---|---|---|
| độ tản hình phạt trên 5 candidate | 0.067 | 0.082 |
| độ tản logit thô | 9.099 | 7.889 |
| tỉ lệ | 136× | **96×** |
| số bước hình phạt đổi top candidate | 0.00033/câu | **0.0913/câu** |

Ở 10 token hình phạt **có** tác dụng — nó đổi top candidate ở khoảng 0.9% các
bước, chừng 274 lần trên 3000 câu, so với **đúng 1 bước** ở 1 token. Nhưng
trong 274 lần đó chỉ 4 lần đi tới câu trả lời cuối, và tỉ số là 1–1. Hình phạt
sắp xếp lại những candidate mà beam dù sao cũng hồi phục được. Đó là một tuyên
bố khác với "hình phạt không thể tác dụng", và là tuyên bố duy nhất mà dữ liệu
10-token cho phép.

### 5.3 Về CausalMM — counterfactual thị giác là một mô tả của thứ code không làm

Mỗi hàng attention của CLIP là softmax trên 577 key, nên ma trận tổng bằng 577
và trung bình mỗi ô là `1/577`. Shuffle một ma trận **không đổi được trung
bình**, và bilinear resize hay mean-over-heads cũng không. Hệ quả: "nhánh
counterfactual" chỉ là ảnh bị hạ tỉ lệ ~580 lần, tức gần như mù.

Đo trên model thật, greedy, 3000 câu, ghép đôi trên cùng sample:

| cặp | trùng nhau | cặp có thông tin | tỉ số | p |
|---|---|---|---|---|
| `shuffle` vs `cf_constant` | 2988/3000 (99.6%) | 12 | 9–3 | **0.146** |
| `shuffle` vs `cf_blind` | 2954/3000 (98.5%) | 46 | 10–36 | **1.6e-4** |
| `cf_constant` vs `cf_blind` | 2942/3000 (98.1%) | 58 | 13–45 | **3.0e-5** |

Accuracy: `shuffle` 76.67 · `cf_constant` 76.47 · `cf_blind` **77.53**.

Hàng đầu xác nhận lập luận, và giờ đủ cỡ mẫu để nói: thay attention đã shuffle
bằng **đúng** `1/n_keys` là không phân biệt được với shuffle. Counterfactual
thị giác của CausalMM **là một contrast tiên nghiệm ngôn ngữ**.

Hàng thứ hai là dự đoán **sai** duy nhất trong cả bộ kiểm chứng, và tín hiệu
phát hiện ra nó ở 500 câu (5–0, p=0.0625) giữ đúng ở 3000 câu: 36–10,
p=1.6e-4. Nhánh mù hoàn toàn là tiên nghiệm ngôn ngữ thuần nhất, nên trừ nó đi
loại bỏ nhiều `Yes` do tiên nghiệm sinh ra hơn là trừ một nhánh còn giữ 0.17%
ảnh. **CausalMM sẽ điểm cao hơn nếu nó làm ít hơn.** Lập luận 1/577 đúng 99%,
và 1% còn lại chạy ngược lại chính phương pháp.

Một con số cần mang sang Table 1: `causalmm` greedy được 76.67 so với baseline
greedy **79.03**. Dưới sampling nó *thắng* (+1.13), dưới greedy nó *thua* 2.36.
Phần lợi của nó phụ thuộc vào mặt nạ plausibility và phép rút ngẫu nhiên, không
phải vào contrast.

### 5.4 Về phần chẩn đoán — đo trực tiếp các con số paper công bố

Ba mục 5.1–5.3 nói về phần **can thiệp**. Round 13 đo phần **chẩn đoán** — thứ
mà toàn bộ bài báo dựa vào để biện minh cho can thiệp. Một pass forward-only,
3000 câu POPE-adversarial, không sinh text, 0.71 s/câu. Chạy lại bằng
`python scripts/analyse_diagnostics.py`.

**Khác với 5.1–5.3, mục này không phải dự đoán tiền-đăng ký.** Gate vẫn là
18 đúng / 1 sai. Đây là phép đo, không phải phép xác nhận — script phân tích
được viết trước khi chạy, nhưng không ai đặt cược vào con số nào trước.

**(a) Mẫu số của Eq. 13 có dấu mà không gì quyết định.** Eq. 13 là
`s_n = Σ m_sys(n)·ΔAUC / Σ ΔAUC`. Mẫu số là một **vô hướng toàn cục**, nên nếu
nó âm thì nó không làm méo thứ hạng — nó **đảo** thứ hạng, và AUC thành
`1 − AUC`.

| | tail=1 | tail=32 |
|---|---|---|
| Eq. 13 đúng như viết | **0.1310** | 0.6568 |
| paper | 0.8226 | 0.7626 |
| mẫu số Σ ΔAUC | **−2.8821** | +0.7769 |
| độ lớn nó cộng, Σ\|ΔAUC\| | 74.5528 | 24.6462 |
| ⇒ mẫu số chỉ là phần dư | **3.9 %** | 3.2 % |
| chia cho Σ\|ΔAUC\| | **0.8690** | 0.6568 |
| held-out | **0.8735** | 0.6645 |

Dấu đó là phần dư của 1024 số hạng gần như triệt tiêu nhau (504 dương, 520 âm).
Paper công bố hai số đều > 0.5, nghĩa là phần dư của họ dương hai lần; của
chúng ta âm ở tail=1. **Tín hiệu là thật và mạnh hơn paper nói** — sửa mẫu số
thì được 0.8690 in-sample, 0.8735 held-out. Đây không phải tái lập thất bại:
đây là tái lập được hiệu ứng, cộng với một lỗi ở công thức báo cáo nó.

**(b) Và hướng thì ngược.** Caption Figure 13 của paper nói rõ: *điểm cao hơn
= dựa vào system token nhiều hơn*, hallucination là lớp dương. Nhưng:

- `m_sys` lấy trung bình trên cả 1024 head dự đoán hallucination **ở mức ngẫu
  nhiên** (AUC 0.4626). Toàn bộ 0.87 đến từ trọng số ΔAUC từng head.
- Tính theo layer, hallucination làm **tăng** `m_sys` ở chỉ **11/32** layer. Năm
  layer có tín hiệu mạnh nhất đều chạy ngược: L23 **0.237**, L22 0.299,
  L26 0.341, L21 0.343, L13 0.370.

Nói cách khác: cái phân biệt được là ==**head nào** dồn chú ý vào system prompt,
không phải **bao nhiêu**==. Và ở các layer giữa-cuối, nhiều chú ý system đi với
**ít** hallucination hơn. Một probe đọc dấu vân 1024 chiều không phải bằng
chứng cho "đường tắt nhân quả"; nó là bằng chứng rằng forward lúc hallucinate
có *hình dạng* khác.

`H_vis` là nửa tiền đề **đúng**: hallucination làm tăng entropy chú ý thị giác
ở 24/32 layer, mạnh nhất L23 (+0.1077). Đúng hướng, nhưng yếu.

**(c) Quy tắc chọn cửa sổ của Figure 7 không chọn ra cửa sổ mà code ship.**
Trung bình |AUC−0.5| trên một cửa sổ là thứ quy tắc đó cực đại hoá. So ở đúng
độ rộng 14 layer của cửa sổ đã ship:

| cửa sổ | `H_vis` | `m_sys` |
|---|---|---|
| code đã ship, L2–15 | 0.0305 | 0.0458 |
| appendix (thủ công), L2–13 | 0.0283 | 0.0402 |
| **cửa sổ 14-layer tốt nhất** | **0.0571** (L11–24) | **0.1075** (L13–26) |
| toàn bộ 32 layer | 0.0373 | 0.0762 |

Cửa sổ đã ship **thấp hơn mức trung bình của toàn bộ layer** trên cả hai thống
kê. Quy tắc được phát biểu là "giữ đoạn liền mạch early-to-mid vượt ngưỡng" và
paper không đưa ngưỡng; ở mức nửa đỉnh, các đoạn liền mạch là {12,13}, {15,16},
{21–24}, {26}, {30} — **không có đoạn early-to-mid nào để giữ**, tín hiệu nằm ở
giữa-cuối.

**Ba điều này khép lại một vòng.** Fox can thiệp lên `S = m_sys · H_vis` tại
L2–15. Giờ cả ba lựa chọn đó đều có một phép đo chống lại: `m_sys` ngược hướng
ở nơi nó mạnh nhất, `H_vis` đúng hướng nhưng yếu, và L2–15 là cửa sổ dưới trung
bình theo đúng tiêu chí của paper. Ba lý do độc lập để can thiệp **không** nên
hoạt động như mô tả — và 5.1 đã cho thấy nó hoạt động bằng thứ khác: phép cast
logit fp16.

**(d) Table 13 báo một đại lượng không được định nghĩa.** Đo được
0.1057 / 0.1602 / 0.7341, tổng **1.0000**. Eq. 11 định nghĩa mỗi `m` là khối
lượng chú ý trên một dải khoá, ba dải phân hoạch toàn bộ khoá nên tổng buộc
phải bằng 1. Paper ghi tổng 2.01 ⇒ ba số đó **không** phải thứ Eq. 11 định
nghĩa, và paper không đưa định nghĩa nào khác.

Còn "Tail ROC-AUC 1.00"? Caption nói nó *"treats system-token responses as the
positive class"* — nó phân biệt **nhóm token**, không phải hallucinated vs
faithful. Chúng ta tái lập **đúng**: 1.0000 ở mức sample (min `m_sys` 0.7113 vs
max của hai cái kia 0.1731), 0.9381 ở mức từng head. Tức là ==con số duy nhất
trong Table 13 mà ta tái lập đến từng chữ số là con số không nói gì về
hallucination==, và nó gần như hiển nhiên khi `m_sys` là số lớn nhất trong ba
trung bình. Khẳng định định tính mà nó được đưa ra để chống lưng thì **đúng**,
và đúng mạnh: 73 % khối lượng chú ý nằm trên system prefix.

### 5.5 Về Figure 8 — chọn head ngẫu nhiên giữ được 90 % hiệu ứng

Figure 8 là phép tự-kiểm của chính paper: Fox so với chọn head **ngẫu nhiên** ở
cùng ngân sách K, và paper kết luận *"targeting specific risky mediators is
superior to stochastic head suppression"*.

Arm `random_heads` đã chạy từ **r6** — 10 seed POPE, 3 seed CHAIR — và **nằm
trong archive không ai chấm** cho đến r13. Ở K = 0.35 (11/32 head):

| arm | accuracy | so baseline | so `fox` (CI 90 %) |
|---|---|---|---|
| `fox` | 81.75 | **+5.03** | — |
| `fox random_heads` | 81.27 | **+4.54** | **−0.490** [−0.816, −0.164] |
| `fox off` | 78.92 | +2.21 | −2.827 |

Thay toàn bộ phần chẩn đoán nhân quả bằng `torch.randperm` mất **0.49 trong
5.03 điểm**. Khoảng tin cậy không chứa 0, nên Figure 8 **không sai** — nó là
một phát biểu đúng về một hiệu ứng dưới một điểm, được vẽ như bằng chứng cho
câu chuyện nhân quả mà 4.54 điểm còn lại không chống lưng.

Trên CHAIR thì head ngẫu nhiên **tốt hơn**, và ở recall không đổi:

| | `fox` | `random_heads` | paper |
|---|---|---|---|
| CHAIR_S | 48.47 | **42.27** | 46.40 |
| CHAIR_I | 13.89 | **12.11** | 12.90 |
| recall | 75.79 | 76.11 | — |
| object đúng | 1182 | 1187 | — |

Recall **không** tụt (76.11 vs 75.79), số object đúng không đổi, caption dài
như nhau — nên đây không phải cái bẫy "nói ít nên ảo ít" mà `vcd_fox_fusion`
bị gắn cờ. ==Chọn head ngẫu nhiên tái lập con số CHAIR của paper sát hơn chính
phần chẩn đoán của paper.==

Cái Figure 8 thực sự quét là Δ theo **K**, và ta chỉ có một K. **r14 đã quét
xong** `k ∈ {0.10, 0.20, 0.45}`, 3 seed, full 3000:

| K | head bị sửa | chẩn đoán | ngẫu nhiên | chênh [90 %] |
|---|---|---|---|---|
| 0.10 | 3 | 79.40 | 79.14 | **+0.26 [−0.20, +0.71]** — không phân biệt được với 0 |
| 0.20 | 6 | 81.16 | 79.79 | +1.37 [+0.91, +1.80] |
| 0.45 | 14 | 82.28 | 81.49 | +0.79 [+0.36, +1.22] |

Ở k = 0.10 — ba head, nơi chọn đúng ba cái phải quan trọng nhất — thứ hạng
**không đáng gì**. Và so với baseline: ngẫu nhiên ở k = 0.45 (+4.77) **ăn
đứt** chẩn đoán ở k = 0.20 (+4.43). ==Hiệu ứng mua bằng **số lượng** head bị
sửa, không phải bằng việc sửa head nào.==

### 5.6 Cửa sổ layer — dự đoán của nhóm **sai**, và cái sai mới là phát hiện

V8 dự đoán cửa sổ không quan trọng (±0.5). Đo ra ngược hẳn, cùng độ rộng 14
layer nên số layer bị làm tròn là như nhau:

| cửa sổ | accuracy | vs L2–15 | vs `fox off` (không can thiệp gì) |
|---|---|---|---|
| **L2–15** (code ship) | 81.56 | — | **+2.59 [+2.09, +3.09]** |
| L11–24 (mạnh nhất cho `H_vis`) | 78.49 | **−3.07** | −0.48 — **tệ hơn không làm gì** |
| L13–26 (mạnh nhất cho `m_sys`) | 79.31 | **−2.24** | +0.34, p = 0.18 — **bằng không làm gì** |

Cột cuối mới là chỗ sắc. Chuyển cửa sổ sang giữa-cuối thì toàn bộ +2.59 của
phần can thiệp **biến mất**. Nó phải xảy ra **sớm**.

Điều này sửa lời giải thích của chính nhóm theo một hướng, và siết paper theo
hướng khác:

- **Chống lại nhóm:** cách hiểu "requantisation" cũ coi vị trí là không quan
  trọng. Sai. Sửa ở layer 2 còn 30 layer phía sau để khuếch đại, sửa ở layer
  26 chỉ còn 6.
- **Chống lại paper, nặng hơn:** r13 đo tín hiệu chẩn đoán theo tiêu chí của
  chính paper và nó **mạnh nhất ở L11–26** — đúng chỗ can thiệp vô dụng.
  ==Chỗ paper chẩn đoán và chỗ can thiệp có tác dụng là **hai chỗ khác
  nhau**.==

**r15 đã quét xong, và đường cong là một vách đá, không phải dốc.** Cùng độ
rộng 14 layer, chỉ đổi vị trí bắt đầu:

| start | cửa sổ | acc | vs `fox off` |
|---|---|---|---|
| 0 | L0–13 | 81.61 | +2.64 |
| 2 | **L2–15** (ship) | 81.56 | +2.59 |
| 4 | L4–17 | 81.53 | +2.57 |
| 8 | L8–21 | 79.29 | +0.32, p = 0.23 |
| 11 | L11–24 | 78.49 | −0.48 |
| 13–18 | … | 79.3–79.4 | +0.34 … +0.47 |

Bắt đầu ở 0, 2, 4 → **không phân biệt được với nhau**, đều cho đủ +2.6. Bắt
đầu từ 8 trở đi → **bằng không can thiệp**, và phẳng tới hết mô hình.

Quét thêm độ rộng từ L2 rồi lấy hiệu, định vị được hiệu ứng nằm ở đâu:

| dải layer | đóng góp |
|---|---|
| 2–3 | +0.02 |
| 4–5 | +0.92 |
| **6–9** | **+1.38** |
| 10–15 | +0.27 |
| 16–23 | −0.02 |
| 24–31 | −0.24 |

~~**89 % hiệu ứng nằm ở layer 4–9.**~~ — ==**r16 đã rút lại khẳng định
này**==, xem 5.7. Phép trừ là số học đúng nhưng **không phải một phép quy
công hợp lệ**: mỗi hiệu là giá trị biên của một layer *trong bối cảnh cửa sổ
nó được thêm vào*, và r16 đo thấy giá trị biên đó dao động hơn một điểm tuỳ
đường đi. Thứ còn đứng: cần một **dải layer sớm liên tục**, và hiệu ứng
**bão hoà**. Sửa quá layer 15 thì trơ hoặc hơi có hại — cái này vẫn đúng.

Hệ quả: **không có gì trong việc chọn cửa sổ của Fox là có tác dụng.** Chỉ
cần cửa sổ phủ đủ layer sớm, mà cửa sổ sớm nào cũng phủ. Và r13 đo tín hiệu chẩn
đoán mạnh nhất ở **L11–26** — toàn bộ nằm trong vùng trơ.

> [!warning] Lời giải thích của nhóm cũng chưa đủ
> r14 thêm "độ sâu lan truyền" vào cách hiểu requantisation. Vách đá **không**
> ủng hộ cách hiểu đó: nếu là lan truyền thì layer 10 phải đáng giá hơi ít hơn
> layer 9, chứ không phải bằng 0, và layer 16–31 vẫn phải đáng giá chút ít.
> Một bậc thang ở 9–10 rồi phẳng là hiện tượng khác, và tài liệu này **chưa có
> lời giải thích cho nó**.

### 5.7 Không layer nào là **cần thiết** — và tôi đã rút lại một khẳng định

r16 đi từng layer một, đo bằng **bước nhảy ghép đôi** (L2-k so với L2-(k−1)
trên cùng sample). Hai cửa sổ chia sẻ gần hết nên bước nhảy đo chính xác hơn
nhiều (±0.2) so với khoảng tin cậy của từng cửa sổ với `off` (±0.45).

| layer k | bước nhảy | 90 % CI | |
|---|---|---|---|
| 3 | **+1.12** | [+0.88, +1.37] | có |
| 4 | −0.10 | [−0.30, +0.09] | không |
| 5 | −0.18 | [−0.37, +0.01] | không |
| 6 | −0.01 | [−0.18, +0.16] | không |
| 7 | **+0.60** | [+0.40, +0.81] | có |
| 8 | **+0.71** | [+0.49, +0.93] | có |
| 9, 10 | +0.08, +0.08 | chứa 0 | không |
| 11 | −0.18 | [−0.32, −0.02] | **âm** |
| 12 | **+0.43** | [+0.24, +0.62] | có |
| 13–15 | −0.09, +0.06, −0.08 | chứa 0 | không |

Nhưng — và đây mới là chỗ quan trọng — **không layer nào là cần thiết**:

| cửa sổ | layer | vs `off` |
|---|---|---|
| L2–3 | chỉ 2, 3 | **+1.22** |
| L4–17 | 4…17, **không có 2, 3** | **+2.57** |
| L2–15 | 2…15 | +2.59 |
| L2–6 | 2…6, **không có 7, 8** | +0.93 |
| L8–21 | 8…21, **không có 2–7** | +0.32 |

Layer 3 đáng +1.12 khi nó là layer thứ hai trong cửa sổ, và đáng **+0.02**
khi 4–15 đã có sẵn. L4–17 bỏ cả hai layer mà bảng bước nhảy làm ra vẻ quý
nhất, vẫn giữ 99 % hiệu ứng.

Nói bằng số: L2–3 (+1.22) cộng L4–17 (+2.57) sẽ là +3.79 **nếu cộng được**.
Đo thật, L2–15 cho **+2.59** — hụt **−1.20**, gần nửa tổng hiệu ứng.
==Hiệu ứng **dưới-cộng tính** mạnh.==

> [!warning] Rút lại "89 % hiệu ứng nằm ở layer 4–9" của r15
> Con số đó đến từ phép trừ các cửa sổ tích luỹ. **Số học đúng** — tổng
> telescoping dọc một đường đi là chính xác — nhưng **quy công thì sai**.
> Những hiệu đó là giá trị biên *trong bối cảnh cụ thể*, và r16 cho thấy nó
> đổi hơn một điểm tuỳ đường đi. Đọc chúng như "tầm quan trọng nội tại của
> từng layer" là lỗi của tôi, ở commit `f9e0916`.
>
> Thứ còn đứng, yếu hơn và khác: **cần một dải layer sớm bắt đầu từ khoảng
> layer 7 trở về trước, và hiệu ứng bão hoà khi đã đủ.** Layer nào cụ thể thì
> phần lớn là thay thế được cho nhau.

Với ba cách giải thích:

- **Paper không khá hơn.** Tín hiệu chẩn đoán vẫn đỉnh ở L11–26, mà layer
  9, 10, 13, 14, 15 đo được là **không đóng góp gì**, layer 11 còn hơi âm.
  Không gì trong phần chẩn đoán dự đoán được layer 3, 7, 8.
- **Cách giải thích "độ sâu" của nhóm giờ **sai hẳn**,** không chỉ là thiếu
  bằng chứng. Lan truyền theo độ sâu thì phải giảm đơn điệu theo chỉ số
  layer. Layer 4–6 không đóng góp gì trong khi 7 và 8 góp 1.31 — không phải
  giảm dần.
- **Requantisation hợp nhất trong ba.** Nếu cơ chế là làm tròn fp16 các
  logit attention thì cái quan trọng là **đủ nhiều** layer sớm bị làm tròn,
  không phải layer nào — và hiệu ứng phải bão hoà. Nó bão hoà, rất gắt, và
  tính dưới-cộng chính là hình dạng của bão hoà.

Vẫn **chưa ai giải thích được** vì sao vùng sống kết thúc quanh layer 7–8.
r16 thu hẹp câu hỏi chứ không đóng nó: khoảng chưa đo giờ là start 5, 6, 7 —
giữa L4–17 (chạy được) và L8–21 (không).

### 5.8 Cái cổng JSD — nửa còn lại, và nó đã đo được một phần

Mọi phép đo ở trên đều nhắm vào **phần chẩn đoán**. Nhưng cái còn lại để giải
thích Fox là **luật hợp nhất**: `fox off` giữ nguyên dual pass và fusion, tắt
hoàn toàn can thiệp, vẫn được +2.21 trong +5.03. Table 14 của paper là
ablation của đúng chỗ đó, và code phát hành không có cờ nào cho hai trong ba
dòng của nó. r15 thêm `--fuse-strategy` (V9).

Điều hay: **không cần chạy mới cũng đã biết một nửa câu trả lời.** Mỗi record
Fox đã lưu `d_t` và `lambda` theo từng sample, nên hành vi của cổng là đo
được, không phải đoán. Với `tau_js = 0.15`:

| | `d_t` trung bình | λ trung bình | % sample từng đạt `d_t ≥ tau_js` |
|---|---|---|---|
| POPE-adv, 3000 | 0.0161 | 1.9964 | **0.2 %** |
| CHAIR, 500 | 0.1285 | 1.3848 | **100 %** |

POPE sinh **một** token nên `d_t` là một số cho mỗi sample, và 2993/3000 nằm
dưới ngưỡng. Vậy trên POPE, `fixed` và `hybrid` **là cùng một run** trên
2993/3000 sample — cổng gần như không bao giờ mở. CHAIR sinh tới 512 token nên
mọi caption đều vượt ngưỡng ở đâu đó. ==Một trục, hai chế độ hoàn toàn khác
nhau, nên V9 chạy cả hai benchmark.==

**r15 đã chạy, và cả hai phần đều rõ.**

Trên POPE, cổng **không làm gì**:

| luật | acc | vs `hybrid` | vs baseline |
|---|---|---|---|
| `hybrid` (Eq. 7, ship) | 81.56 | — | +4.84 |
| `fixed` (λ = α luôn) | 81.91 | +0.36, p = 0.17 | +5.19 |
| `dynamic` (λ = d_t luôn) | 76.74 | **−4.81** | **+0.02** |

==Bỏ sàn λ là bỏ luôn phương pháp== — `dynamic` hơn baseline đúng 0.02 điểm.
Còn `fixed` thì **bằng** `hybrid`. Tức ngưỡng của Eq. 7 không đóng góp gì
trên chính benchmark mang con số accuracy đầu bảng của paper.

Trên CHAIR — nơi cổng thật sự mở — **thứ tự của paper bị đảo ngược**:

| arm | CHAIR_S | CHAIR_I | recall | object đúng | words |
|---|---|---|---|---|---|
| `sampling` | 54.40 | 16.90 | 75.64 | 1180 | 102.1 |
| `fox off` | 49.20 | 14.05 | **78.57** | 1226 | 99.2 |
| `hybrid` | 48.47 | 13.89 | 75.79 | 1182 | 96.3 |
| **`fixed`** | **32.20** | **10.12** | 70.73 | 1103 | 102.8 |
| `dynamic` | 49.47 | 14.77 | 76.84 | 1199 | 99.5 |

| | tốt nhất | giữa | tệ nhất |
|---|---|---|---|
| Table 14 | hybrid 46.40 | dynamic 47.68 | **fixed 49.20** |
| đo được | **fixed 32.20** | hybrid 48.47 | dynamic 49.47 |

`hybrid` của ta (48.47) nằm trong khoảng dịch 5.20 điểm của tập ảnh so với
46.40 của paper — khớp. Nhưng `fixed` 32.20 so với 49.20 là lệch **17 điểm**,
ngoài xa khoảng đó. ==Dòng paper xếp bét lại đứng đầu, cách 16 điểm.==

**Kiểm recall trước khi mừng**, theo đúng cạm bẫy của chính project: `fixed`
giảm CHAIR_I 27 % tương đối (13.89 → 10.12) trong khi recall giảm 6.7 %
(75.79 → 70.73). Đánh đổi có lợi, **không phải thắng không công** — nhưng
cũng **không** phải cái bẫy "nói ít nên ảo ít": caption của `fixed` là **dài
nhất** trong mọi arm (102.8 từ). Nó viết nhiều hơn mà gọi tên ít object hơn.

Và một dòng nữa đáng đọc: `fox off` đạt CHAIR_S 49.20 ở **recall cao nhất**
(78.57). Phương pháp đầy đủ được 48.47 ở recall 75.79 — tức trên CHAIR, phần
can thiệp mua 0.73 điểm CHAIR_S bằng 2.8 điểm recall.

## 6. Tình trạng hiện tại

- **348/358 job đã chạy.** r16 về 5/6 session: **V11 đã chấm đủ** (xem 5.7), còn Table 1 thiếu 8 job vì **session 5 không về**.
- Hai job `L2-14` (seed 43, 44) bị cắt ở tường session (`rc = -15`), còn 2972/3000 và 2599/3000 record. Đã ghép đôi trên phần giao nên vẫn dùng được; cả hai bước nhảy liên quan đều không có ý nghĩa thống kê.
- Gate kiểm chứng: **20 đúng / 4 sai**.
- Hàng đợi `pending` của bot **đã chạy qua trên GPU thật**: 10 session khai
  báo, 4 chạy ngay (2 account còn quota × trần 2 session), **6 xếp hàng** và
  được cron đẩy dần khi có chỗ. Cả 10 đều xong, 0 lỗi. Trước đó nhánh này mới
  chỉ kiểm ở độ sâu 1 trên notebook CPU. Xem [[kaggle-ci-pipeline]].
  Cả bốn cái sai đều là dự đoán của nhóm, không phải bug: `cf_blind` (5.3),
  **V8**, **V10** (5.6) và **V11** (5.7) — và V11 sai còn kéo theo việc rút
  lại một khẳng định r15 đã viết.
- Mọi **baseline** POPE khớp paper trong 0.05 điểm → harness, dữ liệu, protocol
  đều đúng. Những gì đo được là về phương pháp, không phải về setup.

Tái lập so với paper:

| artefact | của mình [CI95] | paper | kết luận |
|---|---|---|---|
| POPE-adv, sampling (10 seed) | +5.03 [+4.68, +5.38] | +5.16 | **tái lập** |
| POPE-random, sampling (3 seed) | +4.16 [+3.86, +4.45] | +4.20 | **tái lập** |
| POPE-popular, sampling (3 seed) | +4.13 [+2.63, +5.64] | +4.17 | điểm khớp, chưa phân giải sau Holm |
| POPE ×3 split, greedy | +0.20 / +1.00 / +3.33 | +0.10 / +1.14 / +3.33 | **tái lập** |
| CHAIR_S, sampling (10 seed) | −6.98 [−9.25, −4.71] | −11.20 | **không** |
| CHAIR_I, sampling (10 seed) | −2.67 [−3.37, −1.96] | −4.28 | **không** |
| Table 6 (tỉ lệ latency) | 2.03× | 1.87× | **tái lập** |

CHAIR không tái lập, và lý do đã đo được: khoảng cách dịch **5.20 điểm** qua
năm lần rút tập ảnh khác nhau, nên khoảng tin cậy một-lần-rút đánh giá thấp độ
bất định. Tập ảnh gốc của paper không thể phục hồi.

### So sánh bảy phương pháp (POPE-adversarial, protocol dùng chung)

Sampling, trung bình qua seed:

| phương pháp | n seed | acc | F1 | gap |
|---|---|---|---|---|
| `fox` | 10 | 81.76 | 82.13 | **+5.03** |
| `sid` | 3 | 79.67 | 80.75 | +2.94 |
| `vcd_fox_fusion` | 3 | 78.81 | 79.07 | +2.09 |
| `causalmm` | 3 | 77.86 | 80.13 | +1.13 |
| `icd` | 3 | 77.20 | 79.15 | +0.48 |
| `sampling` (baseline) | 10 | 76.72 | 78.93 | — |
| `vcd` | 3 | 75.80 | 78.34 | **−0.92** |

Greedy, n=3000, một lần chạy mỗi arm (tiền định nên một lần là đủ):

| phương pháp | acc | F1 | gap |
|---|---|---|---|
| `fox` | 82.37 | 82.65 | +3.33 |
| `opera`, protocol 10 token của OPERA | 81.13 | 81.88 | **+2.10** |
| `sid` | 80.23 | 81.29 | +1.20 |
| `opera`, harness 1 token của Fox | 79.10 | 80.95 | +0.07 |
| `sampling` (baseline) | 79.03 | 80.89 | — |
| `icd` | 78.60 | 80.41 | −0.43 |
| `vcd` | 76.93 | 79.08 | −2.10 |

Hai chỗ bắt buộc phải chú ý khi đọc hai bảng này:

- **`vcd` và `vcd_fox_fusion` là hai thứ khác nhau.** `vcd` là VCD thật, dùng
  quy tắc **trừ** của chính nó, và nó xuống **dưới** baseline ở cả hai bảng.
  `vcd_fox_fusion` là ảnh nhiễu của VCD đặt vào quy tắc **cộng** của Fox, và
  được +2.09. Cùng một nhánh âm, hai quy tắc hợp nhất, lệch nhau 3 điểm — nên
  bất kỳ con số "VCD" nào cũng phải nói rõ nó dùng quy tắc nào. Thêm nữa, code
  đã phát hành của Fox **không chạy nổi** VCD và ICD của chính nó (cả hai cờ
  đều dẫn vào nhánh chết), nên không xác định được số trong paper từ đâu ra.

- **Độ dài sinh văn bản quyết định OPERA.** Cùng một implementation: 1 token
  thì +0.07, 10 token thì +2.10. Beam search chấm **cả chuỗi**, nên token đầu
  phụ thuộc có bao nhiêu token theo sau. Fox chạy lại các phương pháp so sánh
  trong harness 1-token của nó, nên ô OPERA trong Table 1 của Fox là hàng
  1-token — hàng mà OPERA gần như không làm gì.

## 7. Quy trình làm việc

GPU ở trên Kaggle, kết quả đi về repo qua git. Có **hai** lệnh đều tên `ingest`
và chúng chạy **ngược chiều nhau** — bỏ lệnh nào cũng mất việc:

```bash
# TRÊN KAGGLE, trước khi chạy: nạp archive đã commit vào runs dir của session
python -m foxrepro.bench.plan ingest results/runs
# (run_kaggle_fox.sh đã gọi sẵn bên trong)
bash run_kaggle_fox.sh <N>
```

```bash
# Ở LOCAL, sau session: thả zip vào runs_inbox/ rồi
python -m foxrepro.bench.plan ingest     # đọc runs_inbox/
python -m foxrepro.bench.plan archive    # ghi vào results/runs/ để commit
python scripts/check_validation.py       # chấm V1–V6
python -m foxrepro.bench.plan plan --budget-hours 10 --round r<N+1>
git commit && git push
```

Ba quy tắc rút ra từ việc mất GPU-h thật:

1. **Push ngay sau khi commit.** Kaggle clone từ remote, nên một commit chưa
   push là một commit không tồn tại. Đã từng chạy 4 session bằng code cũ vì
   quên push.
2. **Ingest cả zip của session bị crash.** Nó thường chứa những job đã xong
   trước lúc crash.
3. **Tăng số round mỗi lần đóng băng plan lại.** Nếu không, session 1 mới sẽ
   ghi đè zip cũ cùng tên.

## 8. Còn lại gì — việc có thể nhận

**r16 đã đóng băng**: 54 job, 49.5 GPU-h, **6 session, thứ tự `priority`**.

| session | nội dung | h |
|---|---|---|
| 1–3 | **V11** — vách đá layer, mỗi session **một seed trọn vẹn của đường cong** | 9.17 mỗi cái |
| 4–6 | Table 1, hai split còn thiếu cho VCD / ICD / SID / CausalMM | 7.33 mỗi cái |

==Dùng `--order priority` chứ không phải `balanced`, và đây là lựa chọn có
chủ ý.== r15 drain trong nửa ngày vì hai account còn đầy quota; round này
lập khi quota gần cạn nên session có thể nằm chờ nhiều ngày. `balanced` rải
đều các tầng nên **không tiền tố nào của các session trả lời được gì** — sai
hẳn khi không biết mấy session sẽ chạy trước khi hết cửa sổ quota. `priority`
đặt việc quyết định vào session 1, và hàng đợi của bot là FIFO theo số
session nên session 1 cũng là cái được phóng đầu tiên.

Thêm một tính chất của cách nhóm job: mỗi session cliff là **một seed đủ 10
window**, nên mất một session là mất một seed ở mọi điểm, chứ không phải mất
mọi seed ở vài điểm. Dừng sau session 2 vẫn còn đường cong 2 seed đọc được.

Sau r16, mọi mục còn lại đều **cần code mới trước khi xin quota**; đánh giá
chi phí lấy từ [`docs/PAPER_AUDIT.md`](PAPER_AUDIT.md).

> [!warning] Plan rỗng thì đừng chạy `plan plan`
> `plan plan` trên một plan đã cạn sẽ ghi đè `scripts/run_plan.json` thành
> `round: r1, n_sessions: 0`. **Thêm job trước, đóng băng sau.**

Thứ tự đề xuất, theo "cần code mới" tăng dần:

1. **r17** — Table 4 / 12 (bộ chọn anchor), mục duy nhất còn lại chạm trực
   tiếp vào một bảng của paper. "x_last only" và "mean-pool all tokens" cần
   query vector của các vị trí **trước** bước hiện tại, mà KV cache chỉ giữ K
   và V, không giữ Q. Phải thêm chỗ lưu Q theo từng bước — máy móc mới, không
   phải một tham số.
2. **r18+** — MME, MMBench, MathVista (mỗi cái một harness mới).
3. **r19+** — backbone mới. Đắt nhất và cần tổng quát hoá loader.

Câu hỏi vách đá layer 9–10 **đã vào r16** (V11). Lưu ý về thiết kế: quét
**tích luỹ** `L2-k` chứ không phải từng layer một. Một layer đơn lẻ đáng
khoảng 0.35–0.45 điểm trong khi 3 seed trên 3000 sample chỉ phân giải được
~0.4 — quét từng layer sẽ **thiếu lực thống kê đúng chỗ cần sắc nhất**, và
trả về một đường cong nhiễu trông như có cấu trúc. Mỗi điểm tích luỹ đo
0.9–2.6 điểm, rồi lấy hiệu hai điểm liền nhau mới ra đóng góp từng layer.

Không tái lập được, có lý do, **không phải lỗi implementation**:

| mục | vì sao |
|---|---|
| CHAIR tuyệt đối | `chair_eval.py` lấy `os.listdir` của một thư mục chưa publish, shuffle rồi giữ 500 ảnh — tập ảnh của paper **không phục hồi được kể cả về nguyên tắc**. Ta đo được khoảng dịch **5.20 điểm** qua năm lần rút, nên con số tuyệt đối không so được; so sánh *giữa hai arm của ta* trên cùng tập ảnh thì vẫn chắc |
| Figure 6 / Table 5 (GPT-4V judge) | API trả phí, n=50, và model judge đã **deprecated** — không ai tái lập được, kể cả tác giả |
| Figure 16/17/21/22 | InstructBLIP / Shikra, nằm ngoài phạm vi đã chốt |
| Figure 10/11/20 | ví dụ định tính, không có gì để tái lập |

| việc | GPU-h | cần code mới? | giá trị |
|---|---|---|---|
| ~~**Table 13 + Figure 3, 7, 1a**~~ — **đã chạy r13 (2026-11-12)** | 0.6 thực đo | xong | ba phát hiện, xem `VALIDATION.md` §r13: mẫu số của Eq. 13 là phần dư ~4 %, dấu của nó quyết định AUC là 0.13 hay 0.87; `m_sys` **ngược hướng** paper ở các layer mạnh nhất; cửa sổ L2–15 mà code ship **dưới mức trung bình** theo đúng tiêu chí chọn cửa sổ của paper |
| ~~**Table 14 + Figure 19**~~ — quy tắc hợp nhất — **trong r15, V9** | 14.6 | xong: `--fuse-strategy`, mặc định khẳng định bit-equal với `reference_fuse` | cao: arm `off` cho thấy riêng phần hợp nhất mang +2.21 trong +5.03 của Fox. Đã đo được từ archive: cổng chỉ mở trên **0.2 %** sample POPE nhưng **100 %** caption CHAIR |
| **Table 4 / 12** — vị trí truy vấn | ~17 (hoặc ~5 nếu chỉ POPE-adv) | có: bộ chọn anchor | trung bình-cao: code gốc dùng `tail_n = 1`, một vị trí, **không phải** hai anchor như paper mô tả. Mục này đo cái giá của lựa chọn đó |
| **MME harness** | ~4 | có, viết từ đầu | trung bình: release không có script MME |
| **MMBench** | ~6 | có (SID có script tham khảo) | trung bình |
| **MathVista** | ~8 | có, không repo nào có script | thấp-trung bình |
| **Backbone mới** (Qwen3-VL-8B, InternVL3-8B) | ~20 | có: tổng quát hoá loader; `locate_image_span` đang hard-code 576 token | cao nhưng đắt |
| **GPT-4V judge** | 0 GPU, tốn API | có | đang hoãn |

`docs/PAPER_AUDIT.md` đã được cập nhật ở r13 (SID/CausalMM không còn ghi
BLOCKED, Table 13/15 và Figure 3/7/8 đã có kết quả).

## 9. Bản đồ tài liệu

Đọc theo thứ tự này:

| file | dùng khi |
|---|---|
| **`docs/CS2229_Overview.md`** | file này — tổng quan, đọc trước |
| [`docs/CATCHUP.md`](CATCHUP.md) | **từ điển thuật ngữ đầy đủ** (POPE, CHAIR, ULP, TOST, Holm…) + lịch sử. Đọc khi gặp từ không hiểu. Một số bảng số liệu trong đó cũ hơn file này |
| [`docs/VALIDATION.md`](VALIDATION.md) | 6 dự đoán đăng ký trước và kết quả. Đọc trước khi tin bất kỳ con số nào của một phương pháp so sánh |
| [`docs/DEVIATIONS.md`](DEVIATIONS.md) | mọi chỗ code của ta lệch khỏi release, và vì sao. Dài nhất, tra cứu theo mục |
| [`docs/PAPER_AUDIT.md`](PAPER_AUDIT.md) | kiểm kê 15 bảng + 22 hình: cái nào chạy được, giá bao nhiêu (đang lỗi thời) |
| [`README.md`](../README.md) | quickstart và layout repo |

Bản đồ code:

```
src/foxrepro/
  fox/reference.py    chép nguyên văn từ release — oracle số học
  fox/kernel.py       port sạch, test khẳng định trùng bit với trên
  fox/attention.py    FoxState, đăng ký attention, vòng giải mã hai nhánh
  fox/opera.py        beam search của OPERA (không có rollback — xem DEVIATIONS)
  bench/methods.py    mỗi phương pháp là một class, kèm các control arm
  bench/plan.py       ma trận mục tiêu, theo dõi tiến độ, chia session theo budget
  bench/generate.py   CLI chạy một job
  bench/evaluate.py   POPE/CHAIR → metric
  bench/report.py     bảng Markdown kèm khoảng tin cậy
  bench/stats.py      TOST, Holm, bootstrap ghép đôi
scripts/check_validation.py   chấm V1–V6 (CPU, không cần GPU)
```

## 10. Cạm bẫy — đừng lặp lại

Mỗi dòng dưới đây đã tốn thời gian thật và giờ đều có test chặn.

| Lỗi | Hậu quả |
|---|---|
| Nạp checkpoint chưa convert | vision tower khởi tạo **ngẫu nhiên**; caption trôi chảy nhưng không liên quan ảnh, **không metric nào bắt được** |
| Tự viết prompt | baseline POPE cao hơn paper 5.13 điểm |
| Kaggle chưa gắn GPU | **20.8 h mất trắng**, 724 s/caption thay vì 7.8 s |
| Đọc CHAIR_S mà không xem recall | `bf16` nhìn như thắng 23.6 điểm, thật ra recall tụt 79→60 |
| Dùng p-value cao làm bằng chứng tương đương | luận điểm trung tâm dựa trên nền yếu → giờ dùng **TOST** |
| Không hiệu chỉnh đa so sánh | tự mắc đúng lỗi đang phê phán paper → giờ dùng **Holm** |
| `hash()` của Python cho seed theo sample | `hash()` trên `str` bị salt theo tiến trình, nên cùng một lệnh không tái lập được token của chính nó → giờ dùng `zlib.crc32` |
| Quên `@torch.inference_mode()` | graph autograd giữ KV buffer; job 10-token OOM ngay sample đầu |
| Đo token-identity dưới **sampling** | một lần rút multinomial lật bất cứ khi nào logit nhích, nên hai arm có cùng phân phối vẫn khác nhau từng token. Test identity **phải** chạy greedy |
| **Chạy arm rồi không chấm** | `fox random_heads` — phép tự-kiểm quan trọng nhất của paper — chạy từ r6, nằm trong `plan list`, trong mọi zip, và **không tài liệu nào nhắc đến nó suốt 7 round**. Chấm nó mất 0 GPU-h và cho ra phát hiện 5.5. Sinh job và chấm job là hai việc khác nhau; `plan list` nói cái gì đã *chạy*, không nói cái gì đã được *đọc* |
| `plan plan --order balanced` trên plan đã chạy gần hết | `partition_jobs` chia plan **như lúc viết**, không phải **như còn lại**: r15 ra 6 session × 47 h trên ngân sách 10 h, vì 241 job resume trong vài giây bị tính giá đầy đủ. Không job nào chạy hai lần (resume khoá theo run file) nhưng `n_sessions` của manifest là con số CI dùng để phóng session. Nhánh `priority` vốn đã lọc; đã thêm `skip_done` cho nhánh `balanced` + test |
| Chạy `plan plan` khi plan đã cạn | ghi đè `scripts/run_plan.json` thành plan rỗng `round: r1, n_sessions: 0`, xoá luôn bản ghi round vừa chạy. Tệ hơn: nếu sửa `kaggle-run.yml` thành `sessions: 0` cho khớp thì publish release sẽ chạy **không gì cả, im lặng**. Hai test guard chặn đúng chỗ này — **thêm job trước, đóng băng sau** |
| Chia cho một tổng có dấu | xem 5.4(a): mẫu số của Eq. 13 là phần dư ~4 % của 1024 số hạng triệt tiêu nhau, nên dấu của nó — chứ không phải cơ chế nào — quyết định AUC là 0.13 hay 0.87. Khi một công thức chuẩn hoá bằng `Σ x`, hỏi ngay `Σ x` có thể bằng 0 hay đổi dấu không |

Dòng `token-identity` và dòng `plan plan` là bài học tổng quát nhất: **một thí nghiệm có thể trả lời sai
câu hỏi mà vẫn chạy trót lọt.** Khi thiết kế arm mới, hỏi trước: nếu cơ chế này
hoàn toàn không hoạt động, arm này có cho ra kết quả khác không?
