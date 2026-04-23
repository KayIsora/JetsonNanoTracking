import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adjustText import adjust_text

# =========================
# 1. Đường dẫn file
# =========================
excel_path = Path(r"E:\A_HK8\Paper\BCao_GKy\DanhSachTracker_Success&FPS_OTB100.xlsx")
output_dir = excel_path.parent

# =========================
# 2. Đọc dữ liệu an toàn
# =========================
def load_tracker_block(excel_file, cols):
    raw = pd.read_excel(excel_file, header=None)

    # Lấy từ dòng 2 trở đi
    df = raw.iloc[1:, cols].copy()
    df.columns = ["Tracker", "Success", "FPS"]

    # Bỏ dòng header lặp lại
    df = df[df["Tracker"].astype(str).str.strip().str.lower() != "tracker"].copy()

    df["Tracker"] = df["Tracker"].astype(str).str.strip()
    df["Success"] = pd.to_numeric(df["Success"], errors="coerce")
    df["FPS"] = pd.to_numeric(df["FPS"], errors="coerce")

    df = df.dropna(subset=["Tracker", "Success", "FPS"]).copy()
    df = df[df["FPS"] > 0].copy()

    return df

# =========================
# 3. Hàm vẽ 1 biểu đồ
# =========================
def plot_benchmark(
    df,
    title,
    output_file,
    highlight_name="MyTracker",
    use_log_scale=True
):
    if df.empty or len(df) < 2:
        print(f"[ERROR] Không đủ dữ liệu để vẽ {title}")
        print(df)
        return

    fig, ax = plt.subplots(figsize=(13.5, 7.8))

    df_other = df[df["Tracker"] != highlight_name].copy()
    df_highlight = df[df["Tracker"] == highlight_name].copy()

    other_color = "#4C84C3"      # xanh
    highlight_color = "#F28E2B"  # cam

    # Scatter tracker thường
    ax.scatter(
        df_other["FPS"],
        df_other["Success"],
        s=230,
        color=other_color,
        edgecolors="#2b2b2b",
        linewidths=1.2,
        alpha=0.95,
        zorder=3
    )

    # Scatter MyTracker
    if not df_highlight.empty:
        ax.scatter(
            df_highlight["FPS"],
            df_highlight["Success"],
            s=380,
            color=highlight_color,
            edgecolors="#1f1f1f",
            linewidths=1.8,
            alpha=1.0,
            zorder=5
        )

    # Trục X log scale
    if use_log_scale:
        ax.set_xscale("log")
        x_fit_input = np.log10(df["FPS"].values)
        coef = np.polyfit(x_fit_input, df["Success"].values, 1)
        xfit_log = np.linspace(x_fit_input.min(), x_fit_input.max(), 300)
        yfit = coef[0] * xfit_log + coef[1]
        ax.plot(
            10 ** xfit_log,
            yfit,
            linestyle="--",
            linewidth=2.3,
            color="#555555",
            alpha=0.9,
            zorder=2
        )
    else:
        x = df["FPS"].values
        y = df["Success"].values
        coef = np.polyfit(x, y, 1)
        xfit = np.linspace(x.min(), x.max(), 300)
        yfit = coef[0] * xfit + coef[1]
        ax.plot(
            xfit,
            yfit,
            linestyle="--",
            linewidth=2.3,
            color="#555555",
            alpha=0.9,
            zorder=2
        )

    # =========================
    # Label tự động tránh chồng
    # =========================
    texts = []

    for _, row in df.iterrows():
        name = row["Tracker"]
        is_highlight = (name == highlight_name)

        txt = ax.text(
            row["FPS"],
            row["Success"],
            name,
            fontsize=14 if is_highlight else 11.5,
            fontweight="bold" if is_highlight else "normal",
            color="#1f1f1f",
            zorder=6
        )
        texts.append(txt)

    adjust_text(
        texts,
        ax=ax,
        only_move={'points': 'xy', 'text': 'xy'},
        expand_text=(1.15, 1.20),
        expand_points=(1.10, 1.15),
        force_text=(0.7, 0.8),
        force_points=(0.3, 0.4),
        arrowprops=dict(
            arrowstyle="-",
            color="gray",
            lw=0.7,
            alpha=0.65
        )
    )

    # =========================
    # Box Pearson / thống kê
    # =========================
    pearson_fps = np.corrcoef(df["FPS"], df["Success"])[0, 1]
    pearson_logfps = np.corrcoef(np.log10(df["FPS"]), df["Success"])[0, 1]

    stats_text = (
        f"Common trackers: {len(df)}\n"
        f"Pearson(FPS, SuccessRate): {pearson_fps:.3f}\n"
        f"Pearson(log10 FPS, SuccessRate): {pearson_logfps:.3f}"
    )

    ax.text(
        0.985, 0.98, stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11.5,
        bbox=dict(
            boxstyle="square,pad=0.35",
            facecolor="white",
            edgecolor="#999999",
            linewidth=1.0,
            alpha=0.95
        )
    )

    # =========================
    # Giao diện
    # =========================
    ax.set_title(title, fontsize=22, pad=16)
    ax.set_xlabel("FPS", fontsize=16)
    ax.set_ylabel("SuccessRate", fontsize=16)

    ax.grid(True, which="both", linestyle="--", linewidth=1.0, alpha=0.28)
    ax.tick_params(axis="both", labelsize=12)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#333333")

    y_min = max(0, df["Success"].min() - 0.03)
    y_max = min(1.0, df["Success"].max() + 0.03)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()

# =========================
# 4. Đọc dữ liệu
# =========================
df_otb = load_tracker_block(excel_path, [0, 1, 2])   # A:C
df_lasot = load_tracker_block(excel_path, [4, 5, 6]) # E:G

# =========================
# 5. Vẽ
# =========================
plot_benchmark(
    df_otb,
    title="OTB100 Tracker Performance: FPS vs SuccessRate",
    output_file=output_dir / "OTB100_FPS_vs_SuccessRate.png",
    highlight_name="MyTracker",
    use_log_scale=True
)

plot_benchmark(
    df_lasot,
    title="LaSOT Tracker Performance: FPS vs SuccessRate",
    output_file=output_dir / "LaSOT_FPS_vs_SuccessRate.png",
    highlight_name="MyTracker",
    use_log_scale=True
)

print("Đã lưu:")
print(output_dir / "OTB100_FPS_vs_SuccessRate.png")
print(output_dir / "LaSOT_FPS_vs_SuccessRate.png")