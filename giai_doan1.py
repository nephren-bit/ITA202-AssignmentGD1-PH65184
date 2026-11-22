# ITA202 - Assignment - Giai đoạn 1: Chuẩn bị dữ liệu và Xây dựng mô hình cơ sở
# Bài toán: dự đoán bệnh tim (phân loại nhị phân: 0 = không bệnh, 1 = có bệnh)
#
# Nguồn dữ liệu: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989).
#   Heart Disease [Dataset]. UCI Machine Learning Repository.
#   https://doi.org/10.24432/C52P4X (giấy phép CC BY 4.0)
#
# Các bước: 1. Khám phá -> 2. Làm sạch -> 3. Tiền xử lý -> 4. Huấn luyện -> 5. Đánh giá
#
# Nguyên tắc tránh rò rỉ dữ liệu (data leakage): những bước không cần thống kê (xóa dòng trùng,
# đổi giá trị vô lý thành NaN) làm trên toàn bộ dữ liệu; những bước phải "học" từ dữ liệu
# (ngưỡng ngoại lệ, median, mode, mean/std) chỉ học trên tập Train rồi áp dụng cho Test.
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # lưu ảnh ra file, không cần mở cửa sổ
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = BASE_DIR / "hinh"

RANDOM_STATE = 42
TEST_SIZE = 0.2

SOURCES = {
    "cleveland": "Cleveland (Mỹ)",
    "hungarian": "Hungary",
    "switzerland": "Thụy Sĩ",
    "va": "Long Beach VA (Mỹ)",
}
COLUMNS = [
    "age",  # tuổi
    "sex",  # giới tính: 1 = nam, 0 = nữ
    "cp",  # loại đau ngực: 1-4 (4 = không triệu chứng)
    "trestbps",  # huyết áp lúc nghỉ (mm Hg)
    "chol",  # cholesterol (mg/dl)
    "fbs",  # đường huyết lúc đói > 120 mg/dl
    "restecg",  # điện tâm đồ lúc nghỉ: 0-2
    "thalach",  # nhịp tim tối đa
    "exang",  # đau ngực khi gắng sức
    "oldpeak",  # độ chênh ST khi gắng sức so với lúc nghỉ
    "slope",  # độ dốc đoạn ST: 1-3
    "ca",  # số mạch máu chính nhìn thấy qua chụp huỳnh quang: 0-3
    "thal",  # kết quả xạ hình thallium: 3, 6, 7
    "num",  # chẩn đoán gốc: 0 = không bệnh, 1-4 = có bệnh
]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BINARY = ["sex", "fbs", "exang"]
CATEGORIES = {"cp": [1, 2, 3, 4], "restecg": [0, 1, 2], "slope": [1, 2, 3]}

ZERO_IS_MISSING = ["trestbps", "chol"]  # huyết áp và cholesterol không thể bằng 0
MAX_MISSING = 0.5  # cột thiếu quá 50% thì loại bỏ
# ca (66%) và thal (53%) vượt ngưỡng trên dữ liệu gốc. Ghi cố định thay vì xét lại trên từng tập Train,
# vì thal sát ngưỡng: đổi cách chia là bộ đặc trưng sẽ đổi theo. Quyết định này không dùng tới nhãn.
DROP_COLUMNS = ["ca", "thal"]

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8a85"


def tieu_de(text):
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def luu_hinh(fig, name):
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / name, dpi=150)
    plt.close(fig)
    print("Đã lưu biểu đồ:", f"hinh/{name}")


# ---------- 1. Chọn và khám phá dữ liệu ----------
def doc_du_lieu():
    frames = []
    for key, name in SOURCES.items():
        # File gốc không có dòng tiêu đề, giá trị thiếu ghi bằng dấu "?"
        part = pd.read_csv(DATA_DIR / f"processed.{key}.data", header=None, names=COLUMNS, na_values="?")
        part["source"] = name
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def kham_pha(df):
    print("Shape:", df.shape)
    print("\n--- 5 dòng đầu ---")
    print(df.head().to_string())
    print("\n--- Info ---")
    df.info()
    print("\n--- Describe ---")
    print(df.describe().T.round(2).to_string())
    print("\n--- Số ca theo cơ sở y tế ---")
    benh = (df["num"] > 0).groupby(df["source"]).agg(so_ca="size", co_benh="sum", ti_le_benh="mean")
    print(benh.round(3).to_string())
    print("\nSố dòng trùng lặp:", df.duplicated().sum())
    for col in ZERO_IS_MISSING:
        print(f"Số giá trị 0 vô lý ở cột {col}:", (df[col] == 0).sum())


# ---------- 2. Làm sạch dữ liệu ----------
def lam_sach_co_ban(df):
    df = df.drop_duplicates().reset_index(drop=True)
    for col in ZERO_IS_MISSING:
        df[col] = df[col].mask(df[col] == 0)
    df["target"] = (df["num"] > 0).astype(int)
    # Bỏ num (chính là đáp án) và source (tỉ lệ bệnh mỗi nơi rất khác nhau, model sẽ "học vẹt" theo nơi khám)
    return df.drop(columns=["num", "source"] + DROP_COLUMNS)


def hoc_lam_sach(X_train):
    """Học mọi tham số làm sạch, chỉ từ tập Train."""
    bounds = {}
    for col in NUMERIC:
        q1, q3 = X_train[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    # Cột số điền median (ít bị ngoại lệ kéo lệch), cột phân loại điền mode
    fill = {col: X_train[col].median() for col in NUMERIC}
    fill.update({col: X_train[col].mode()[0] for col in BINARY + list(CATEGORIES)})
    return {"bounds": bounds, "fill": fill}


def ap_dung_lam_sach(X, params):
    X = X.copy()
    for col, (low, high) in params["bounds"].items():
        X[col] = X[col].clip(low, high)  # kéo ngoại lệ về ngưỡng thay vì xóa cả bệnh nhân
    return X.fillna(params["fill"])


def in_ket_qua_lam_sach(X_train, params):
    print("Tỉ lệ thiếu trên Train (%):")
    print((X_train.isna().mean() * 100).round(1).to_string())
    print(f"\nCột bị loại (thiếu > {MAX_MISSING:.0%} trên dữ liệu gốc):", DROP_COLUMNS)
    print("\nNgoại lệ theo IQR trên Train:")
    for col, (low, high) in params["bounds"].items():
        count = ((X_train[col] < low) | (X_train[col] > high)).sum()
        print(f"  {col:<9} ngưỡng [{low:7.2f}, {high:7.2f}] -> {count} ngoại lệ")
    print("\nGiá trị dùng để điền NaN:")
    for col, value in params["fill"].items():
        print(f"  {col:<9} {value:g}")


# ---------- 3. Tiền xử lý ----------
def ma_hoa(X):
    # cp, restecg, slope là mã nhóm, không phải đại lượng -> one-hot (bỏ nhóm đầu làm mốc)
    X = X.copy()
    for col, values in CATEGORIES.items():
        X[col] = pd.Categorical(X[col].astype(int), categories=values)
    return pd.get_dummies(X, columns=list(CATEGORIES), drop_first=True, dtype=float)


# ---------- 4. Huấn luyện mô hình cơ sở ----------
def huan_luyen(X_train, y_train):
    return LogisticRegression(max_iter=1000).fit(X_train, y_train)


# ---------- 5. Đánh giá ----------
def tinh_chi_so(model, X, y):
    pred = model.predict(X)
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "ROC-AUC": roc_auc_score(y, model.predict_proba(X)[:, 1]),
    }


def danh_gia(model, X_train, X_test, y_train, y_test):
    moc = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    bang = pd.DataFrame(
        {
            "Đoán theo số đông": tinh_chi_so(moc, X_test, y_test),
            "LogReg (Train)": tinh_chi_so(model, X_train, y_train),
            "LogReg (Test)": tinh_chi_so(model, X_test, y_test),
        }
    )
    print(bang.round(4).to_string())

    pred = model.predict(X_test)
    print("\n--- Classification report (Test) ---")
    print(classification_report(y_test, pred, target_names=["Không bệnh", "Có bệnh"], digits=4))
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print(f"Confusion Matrix: TN = {tn}, FP = {fp}, FN = {fn} (bỏ sót bệnh), TP = {tp}")
    # MSE là chỉ số của bài toán hồi quy nên không dùng cho bài phân loại này


# ---------- Biểu đồ ----------
def ve_phan_bo_nhan(df):
    count = (df["num"] > 0).groupby(df["source"]).agg(["size", "sum"]).reindex(list(SOURCES.values()))
    x = range(len(count))
    fig, ax = plt.subplots(figsize=(8, 4))
    for offset, values, color, label in [
        (-0.2, count["size"] - count["sum"], BLUE, "Không bệnh"),
        (0.2, count["sum"], ORANGE, "Có bệnh"),
    ]:
        bars = ax.bar([i + offset for i in x], values, width=0.37, color=color, label=label)
        ax.bar_label(bars, padding=2)
    ax.set_xticks(list(x), count.index)
    ax.set(title="Số ca theo cơ sở y tế và chẩn đoán", ylabel="Số bệnh nhân", ylim=(0, 220))
    ax.legend(frameon=False)
    luu_hinh(fig, "01_phan_bo_nhan.png")


def ve_gia_tri_thieu(df):
    # Tính cả các số 0 vô lý, vì thực chất đó là giá trị thiếu
    data = df[COLUMNS[:-1]].copy()
    for col in ZERO_IS_MISSING:
        data[col] = data[col].mask(data[col] == 0)
    percent = (data.isna().mean() * 100).sort_values()
    colors = [ORANGE if v > MAX_MISSING * 100 else BLUE for v in percent]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(percent.index, percent, height=0.65, color=colors)
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.axvline(MAX_MISSING * 100, color=GRAY, linestyle="--", linewidth=1)
    ax.text(MAX_MISSING * 100 - 1, 0, "ngưỡng loại 50%", ha="right", va="center", color="#52514e")
    ax.set(title="Tỉ lệ giá trị thiếu (đã tính các số 0 vô lý)", xlabel="% số dòng", xlim=(0, 80))
    luu_hinh(fig, "02_gia_tri_thieu.png")


def ve_ngoai_le(truoc, bounds):
    # So sánh trước khi điền NaN, vì điền median làm hẹp IQR và sinh ra "ngoại lệ" giả trên biểu đồ
    sau = truoc[NUMERIC].apply(lambda s: s.clip(*bounds[s.name]))
    fig, axes = plt.subplots(2, len(NUMERIC), figsize=(11, 5.5), sharey="col")
    for row, (data, label) in enumerate([(truoc, "Trước"), (sau, "Sau")]):
        for ax, col in zip(axes[row], NUMERIC):
            ax.boxplot(
                data[col].dropna(),
                widths=0.5,
                flierprops={"markeredgecolor": ORANGE, "markersize": 4},
                medianprops={"color": BLUE, "linewidth": 2},
            )
            ax.set_title(f"{col} - {label}")
            ax.set_xticks([])
    fig.suptitle("Giá trị ngoại lệ trước và sau khi xử lý (tập Train)")
    luu_hinh(fig, "03_ngoai_le.png")


def ve_confusion_matrix(model, X_test, y_test):
    fig, ax = plt.subplots(figsize=(5, 4.3))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["Không bệnh", "Có bệnh"], cmap="Blues", colorbar=False, ax=ax
    )
    ax.set(title="Confusion Matrix - tập Test", xlabel="Dự đoán", ylabel="Thực tế")
    luu_hinh(fig, "04_confusion_matrix.png")


def ve_roc(model, X_test, y_test):
    fig, ax = plt.subplots(figsize=(5, 4.3))
    RocCurveDisplay.from_estimator(
        model, X_test, y_test, name="Logistic Regression", curve_kwargs={"color": BLUE, "linewidth": 2}, ax=ax
    )
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1, label="Đoán ngẫu nhiên")
    ax.set(title="Đường ROC - tập Test", xlabel="Tỉ lệ báo nhầm (FPR)", ylabel="Tỉ lệ phát hiện (Recall)")
    ax.legend(loc="lower right", frameon=False)
    luu_hinh(fig, "05_roc.png")


def ve_he_so(model, names):
    coef = pd.Series(model.coef_[0], index=names).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(coef.index, coef, height=0.65, color=[ORANGE if v > 0 else BLUE for v in coef])
    ax.bar_label(bars, fmt="%+.2f", padding=3)
    ax.axvline(0, color=GRAY, linewidth=1)
    ax.margins(x=0.15)
    ax.set(title="Hệ số mô hình: dương = tăng khả năng có bệnh, âm = giảm", xlabel="Hệ số (sau chuẩn hóa)")
    luu_hinh(fig, "06_he_so.png")


def main():
    IMAGE_DIR.mkdir(exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

    tieu_de("1. KHÁM PHÁ DỮ LIỆU")
    raw = doc_du_lieu()
    kham_pha(raw)
    ve_phan_bo_nhan(raw)
    ve_gia_tri_thieu(raw)

    tieu_de("2. LÀM SẠCH DỮ LIỆU")
    df = lam_sach_co_ban(raw)
    print(f"Xóa dòng trùng: {len(raw)} -> {len(df)} dòng")
    # Chia ngay tại đây để Test không ảnh hưởng tới median, mode, ngưỡng ngoại lệ, mean/std
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df.drop(columns="target"),
        df["target"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["target"],
    )
    params = hoc_lam_sach(X_train_raw)
    in_ket_qua_lam_sach(X_train_raw, params)
    X_train_clean = ap_dung_lam_sach(X_train_raw, params)
    X_test_clean = ap_dung_lam_sach(X_test_raw, params)
    print("\nNaN còn lại: Train =", X_train_clean.isna().sum().sum(), "| Test =", X_test_clean.isna().sum().sum())
    ve_ngoai_le(X_train_raw, params["bounds"])

    tieu_de("3. TIỀN XỬ LÝ")
    X_train_encoded, X_test_encoded = ma_hoa(X_train_clean), ma_hoa(X_test_clean)
    scaler = StandardScaler().fit(X_train_encoded)
    X_train, X_test = scaler.transform(X_train_encoded), scaler.transform(X_test_encoded)
    print(f"Train: {X_train.shape} ({len(X_train) / len(df):.0%}), có bệnh {y_train.mean():.1%}")
    print(f"Test:  {X_test.shape} ({len(X_test) / len(df):.0%}), có bệnh {y_test.mean():.1%}")
    print("Đặc trưng sau mã hóa:", X_train_encoded.columns.tolist())

    tieu_de("4. HUẤN LUYỆN LOGISTIC REGRESSION")
    model = huan_luyen(X_train, y_train)
    print("Hội tụ sau", model.n_iter_[0], "vòng lặp")
    print(pd.Series(model.coef_[0], index=X_train_encoded.columns).sort_values(key=abs, ascending=False).round(3).to_string())

    tieu_de("5. ĐÁNH GIÁ MÔ HÌNH CƠ SỞ")
    danh_gia(model, X_train, X_test, y_train, y_test)
    ve_confusion_matrix(model, X_test, y_test)
    ve_roc(model, X_test, y_test)
    ve_he_so(model, X_train_encoded.columns)


if __name__ == "__main__":
    main()
