import os
import re
import pandas as pd
import matplotlib

matplotlib.use('Agg')  # 必须在其他 matplotlib 导入前设置
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def parse_perf_log(txt_path):
    """解析性能日志TXT文件"""
    data = {'Epoch': [], 'Epoch_Time': [], 'Batch_Time': [], 'FPS': []}

    with open(txt_path, 'r') as f:
        for line in f:
            if 'Epoch time:' in line:
                data['Epoch_Time'].append(float(re.search(r'Epoch time: ([\d.]+) s', line).group(1)))
            elif 'Avg batch time:' in line:
                data['Batch_Time'].append(float(re.search(r'Avg batch time: ([\d.]+) ms', line).group(1)))
            elif 'Train FPS:' in line:
                data['FPS'].append(float(re.search(r'Train FPS: ([\d.]+) samples/sec', line).group(1)))

    # 生成Epoch编号
    data['Epoch'] = list(range(len(data['Epoch_Time'])))

    return pd.DataFrame(data)


def plot_data(df, save_path, title):
    """绘图函数（移除了多余的 plt.close()）"""
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # 左Y轴（Epoch Time）
    ax1.plot(df['Epoch'], df['Epoch_Time'], 'b-o', label='Epoch Time (s)')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Epoch Time (s)', color='b', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 右Y轴（FPS和Batch Time）
    ax2 = ax1.twinx()
    ax2.plot(df['Epoch'], df['FPS'], 'r-s', label='FPS (samples/sec)')
    ax2.plot(df['Epoch'], df['Batch_Time'], 'g--^', label='Batch Time (ms)')
    ax2.set_ylabel('FPS / Batch Time', fontsize=12)

    # 合并图例
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc='upper right')

    plt.title(f"Training Performance: {title}", pad=20)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # 明确关闭当前figure


def plot_log(file_path, save_dir=None):
    """主处理函数（移除了重复的 plt.close()）"""
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(file_path), "log_curve")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    file_ext = os.path.splitext(file_path)[1].lower()
    file_name = os.path.basename(file_path)
    save_name = file_name.replace(file_ext, "_performance.png" if file_ext == '.txt' else "_metrics.png")
    save_path = os.path.join(save_dir, save_name)

    if file_ext == '.txt':
        df = parse_perf_log(file_path)
        plot_data(df, save_path, os.path.splitext(file_name)[0])
    elif file_ext == '.csv':
        df = pd.read_csv(file_path)
        save_name = os.path.splitext(file_name)[0] + "_metrics.png"
        save_path = os.path.join(save_dir, save_name)

        plt.figure(figsize=(12, 8))
        ax1 = plt.gca()
        ax2 = ax1.twinx()

        ax1.plot(df['epoch'], df['IoU'], 'b-', label='IoU', linewidth=2)
        ax1.set_ylabel('IoU', color='b', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='b')

        ax2.plot(df['epoch'], df['PD'], 'g-', label='PD', linewidth=2)
        ax2.plot(df['epoch'], df['FA'], 'r-', label='FA', linewidth=2)
        ax2.set_yscale('log')
        ax2.set_ylabel('PD / FA (log)', fontsize=12)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.title(f"Training Metrics: {os.path.splitext(file_name)[0]}", pad=20)

    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

    print(f"Plot saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize training logs')
    parser.add_argument("file_path", type=str, help="Path to log file (CSV or TXT)")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Custom save directory (default: auto-create log_curve folder)")
    args = parser.parse_args()

    plot_log(args.file_path, args.save_dir)