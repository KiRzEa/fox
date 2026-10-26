# Dữ liệu trông như thế nào

Ba ảnh thật, lấy từ **COCO val2014** — đúng nguồn mà cả POPE lẫn CHAIR dùng.
Mỗi ảnh ở đây xuất hiện trong **cả hai** benchmark, nên bạn thấy được cùng
một ảnh được hỏi theo hai cách khác nhau.

Mọi câu hỏi, nhãn, ground truth và caption trong file này đều **đọc ngược từ
file thật** mà dự án đã chấm (`samples.json` được sinh tự động), không có gì
gõ tay.

```
samples/
  images/COCO_val2014_000000117527.jpg
  images/COCO_val2014_000000142774.jpg
  images/COCO_val2014_000000562155.jpg
  samples.json      cùng nội dung dưới dạng máy đọc được
```

Ảnh lấy trực tiếp từ `http://images.cocodataset.org/val2014/<tên file>` —
không cần tải zip 6.2 GB.

---

## Ảnh 1 — `142774`: ví dụ rõ nhất về ba split POPE

![bus và xe máy](images/COCO_val2014_000000142774.jpg)

Cảnh đường phố: xe buýt xanh, một xe máy chở hai người, lề cỏ.
**Ground truth COCO: `bus`, `motorcycle`, `person`.**

POPE hỏi 6 câu cho mỗi ảnh ở mỗi split — 3 câu trả lời **yes**, 3 câu **no**.
Phần `yes` **giống hệt nhau** ở cả ba split. Khác nhau nằm ở phần `no`:

| split | 3 vật thể hỏi mà đáp án là **no** | tại sao khó dần |
|---|---|---|
| `random` | boat, kite, train | lấy ngẫu nhiên — **không liên quan gì** đến đường phố. Dễ |
| `popular` | dining table, car, chair | vật thể **hay xuất hiện** trong COCO nói chung. Khó hơn vì mô hình có tiên nghiệm mạnh |
| `adversarial` | **car, bicycle, traffic light** | vật thể **hay đi cùng** `bus`/`motorcycle`. Khó nhất |

Đây là điểm mấu chốt: ở `adversarial`, `car` và `traffic light` **đúng là thứ
bạn mong đợi trên một con đường**. Mô hình phải nhìn ảnh để biết chúng không
có — nếu nó đoán theo tiên nghiệm ngôn ngữ ("đường thì có ô tô"), nó sai.
Đó chính là ảo giác mà paper muốn giảm, và là lý do paper tuyên bố cải thiện
lớn nhất ở split này (+5.16 điểm).

---

## Ảnh 2 — `117527`: ví dụ rõ nhất về ảo giác trong CHAIR

![bàn ăn nhà hàng](images/COCO_val2014_000000117527.jpg)

Bàn ăn đông người: nhiều người, một tô thức ăn lớn, cốc bia, giấy ăn.
**Ground truth COCO: `bowl`, `cup`, `dining table`, `person`.**

POPE `adversarial` hỏi **no** cho: `chair`, `car`, `spoon` — `spoon` rất hay
đi cùng `bowl` và `dining table`.

CHAIR thì không hỏi gì cả, nó bảo mô hình **"Please describe this image in
detail."** rồi đếm vật thể bịa. Đây là caption **thật** của baseline:

> The image depicts a group of people gathered around a dining table, enjoying
> a meal together. There are at least nine people in the scene… The table is
> filled with various food items, including a large bowl of food, multiple
> cups, and **a few sandwiches**. The table is also set with a variety of
> utensils, such as **forks, knives, and spoons**. There are multiple
> **bottles** placed around the table…

| | |
|---|---|
| nhắc tới | bottle, bowl, cup, dining table, fork, knife, person, sandwich, spoon |
| có thật | bowl, cup, dining table, person |
| **bịa** | **sandwich, fork, knife, spoon, bottle** |

Không cái nào trong số đó **vô lý** — bàn ăn thì thường có dao nĩa. Mô hình
đang mô tả *cái bàn ăn điển hình* chứ không phải *cái bàn ăn này*. Đó đúng là
định nghĩa của hallucination ở đây, và là lý do CHAIR khó hơn POPE: POPE chỉ
cần một token `Yes`/`No`, CHAIR phải đúng suốt ~120 từ.

Caption của `fox` trên cùng ảnh bịa `chair`, `knife`, `bottle` — ít hơn, nhưng
vẫn bịa.

---

## Ảnh 3 — `562155`: khi "sai" là một phán đoán tinh tế

![tự chụp trong xe](images/COCO_val2014_000000562155.jpg)

Ảnh tự chụp trong xe hơi: một người đeo kính, áo xám, cà vạt kẻ caro.
**Ground truth COCO: `car`, `person`, `tie`.**

POPE `adversarial` hỏi **no** cho: `dining table`, `truck`, `cup`.

Caption baseline bịa đúng **một** vật thể: `chair`. Nhìn ảnh thì có ghế xe —
nhưng COCO không chú thích ghế ô tô là `chair`, nên bộ chấm tính đó là bịa.

Ví dụ này đáng giữ lại vì nó cho thấy **CHAIR không hoàn hảo**: một phần
"ảo giác" thực ra là bất đồng giữa mô hình và quy ước chú thích của COCO.
Đó là một lý do nữa để luôn đọc CHAIR kèm `recall`, và để so sánh **gap giữa
hai phương pháp trên cùng bộ ảnh** thay vì đọc con số tuyệt đối.

---

## CHAIR_S và CHAIR_I được tính thế nào

CHAIR là **thước đo**, không phải một phương pháp cạnh tranh với Fox. Nó chấm
một caption bằng cách so danh sách vật thể caption nhắc tới với chú thích thật
của COCO.

```
CHAIR_S = (số caption có >= 1 vật thể bịa) / (tổng số caption)     S = sentence
CHAIR_I = (số lần nhắc vật thể bị bịa)     / (tổng số lần nhắc)     I = instance
```

Lấy baseline greedy trên 500 ảnh:

| | tử số | mẫu số | kết quả |
|---|---|---|---|
| CHAIR_S | 246 | 500 | **49.20%** |
| CHAIR_I | 520 | 3762 | **13.82%** |

Ảnh `117527` ở trên đóng góp vào cả hai, theo hai cách:
caption nhắc vật thể **14 lần**, trong đó **5 lần** là bịa
(`sandwich`, `fork`, `knife`, `spoon`, `bottle`)
→ **1** vào tử số CHAIR_S, và **5/14** vào CHAIR_I.

Ảnh `142774` không bịa gì → đóng góp **0** vào tử số CHAIR_S.

Cần cả hai vì chúng trả lời hai câu khác nhau: CHAIR_S là *"bao nhiêu caption
có vấn đề"*, CHAIR_I là *"vấn đề nặng cỡ nào"*. Và CHAIR_S nhiễu hơn hẳn — ở
n=500 nó chỉ phân giải được ~8.85 điểm, trong khi CHAIR_I có 3762 lần nhắc nên
phân giải được ~2.24 điểm.

---

## Hai benchmark cạnh nhau

| | POPE | CHAIR |
|---|---|---|
| Đầu vào | ảnh + 1 câu hỏi có/không | ảnh + *"Please describe this image in detail."* |
| Mô hình sinh | **1 token** | tới **512 token** (thực tế ~100–120 từ) |
| Quy mô | 3000 câu × 3 split | 500 ảnh |
| Đo cái gì | khả năng **phân biệt** | khả năng **sinh** |
| Chỉ số | accuracy, F1, yes-ratio | CHAIR_S, CHAIR_I, recall |
| Ground truth | nhãn yes/no có sẵn | danh sách vật thể từ `instances_val2014.json` |

Kết quả của dự án này cho thấy hai benchmark **không đo cùng một thứ**: trên
POPE, lợi thế của Fox được tái tạo chính xác bởi một arm chỉ làm tròn số
(`ulp_control`); trên CHAIR, thứ thực sự giúp là tăng attention lên token ảnh
(`boost_visual`). Xem [`../docs/CATCHUP.md`](../docs/CATCHUP.md) §5–6.

---

## Tự lấy thêm mẫu

```bash
# danh sách ảnh CHAIR đang dùng
python3 -c "import json;print(json.load(open('src/foxrepro/assets/chair_images.json'))['image_ids'][:10])"

# câu hỏi POPE của một ảnh, cả ba split
python3 - <<'PY'
import json, pathlib
img = 142774
for split in ("random", "popular", "adversarial"):
    f = next(pathlib.Path("src/foxrepro/assets/pope").glob(f"*{split}*.json"))
    for line in f.read_text().splitlines():
        r = json.loads(line)
        if int(r["image"].split("_")[-1].split(".")[0]) == img:
            print(f"{split:12s} {r['label']:3s} {r['text']}")
PY

# tải một ảnh bất kỳ
curl -O http://images.cocodataset.org/val2014/COCO_val2014_000000142774.jpg
```
