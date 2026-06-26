import csv
from pathlib import Path

import matplotlib.pyplot as plt


def plot_losses(histories, title):
    fig, ax = plt.subplots(figsize=(7, 4))

    for label, (train_losses, val_losses) in histories.items():
        ax.plot(train_losses, linewidth=2, label=f"{label} train")
        ax.plot(val_losses, linewidth=2, label=f"{label} validation")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title(title)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    Path("history").mkdir(exist_ok=True)
    file_name = title.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    plt.savefig(f"history/{file_name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def write_csv(rows, output_path="results.csv"):
    if isinstance(rows, dict):
        rows = [rows]
    if not rows:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists() and output_path.stat().st_size > 0

    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
