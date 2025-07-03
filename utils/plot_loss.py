import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import os
from pathlib import Path


def plot_loss(file_path="loss_log.txt", save_dir="weights/loss_curve"):
    # 确保保存目录存在
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 从文件名生成图片名（去掉.txt后缀，加上.png）
    file_name = os.path.basename(file_path)
    save_name = os.path.splitext(file_name)[0] + ".png"
    save_path = os.path.join(save_dir, save_name)

    epoch_losses = defaultdict(list)

    # 读取数据并分组
    with open(file_path, "r") as f:
        for line in f:
            epoch, loss = map(float, line.strip().split(","))
            epoch_losses[int(epoch)].append(loss)

    # 计算每个epoch的平均loss
    epochs = sorted(epoch_losses.keys())
    avg_losses = [sum(epoch_losses[e]) / len(epoch_losses[e]) for e in epochs]

    # 绘制曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, avg_losses, 'b-', lw=2, label='Average Loss per Epoch')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss Curve ({os.path.splitext(file_name)[0]})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形，避免在非交互环境下显示
    print(f"Loss curve saved to: {save_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_loss(file_path=sys.argv[1])
    else:
        plot_loss()