# /root/llama/LlamaFactory/data_image_generation.py

import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import os

def parse_log_file(log_path):
    """解析训练日志文件"""
    train_steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []
    learning_rates = []
    epochs = []

    with open(log_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            step = data.get('current_steps', 0)

            # 训练损失
            if 'loss' in data:
                train_steps.append(step)
                train_losses.append(data['loss'])

            # 验证损失
            if 'eval_loss' in data:
                eval_steps.append(step)
                eval_losses.append(data['eval_loss'])

            # 学习率
            if 'lr' in data:
                learning_rates.append(data['lr'])

            # epoch
            if 'epoch' in data:
                epochs.append(data['epoch'])

    return {
        'train_steps': train_steps,
        'train_losses': train_losses,
        'eval_steps': eval_steps,
        'eval_losses': eval_losses,
        'learning_rates': learning_rates,
        'epochs': epochs[:len(train_steps)]  # 确保长度一致
    }

def create_visualization(log_path, output_path=None):
    """创建训练可视化图表"""

    # 解析日志
    print(f"📂 正在解析日志: {log_path}")
    data = parse_log_file(log_path)

    # 设置样式
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🎯 Qwen2.5-1.5B LoRA Training Analysis\n(农业实体识别任务)',
                 fontsize=16, fontweight='bold', y=0.98)

    # 调整子图间距
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # ==================== 图1: 损失曲线对比 ====================
    ax1 = axes[0, 0]

    # 绘制训练损失
    ax1.plot(data['train_steps'], data['train_losses'],
             color='#2E86AB', linewidth=2, label='Training Loss', alpha=0.8)

    # 绘制验证损失
    ax1.plot(data['eval_steps'], data['eval_losses'],
             color='#E94F37', linewidth=2.5, marker='o', markersize=4,
             label='Validation Loss', alpha=0.9)

    ax1.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Loss Value', fontsize=11, fontweight='bold')
    ax1.set_title('📉 Training vs Validation Loss', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # 标注关键点
    min_eval_idx = np.argmin(data['eval_losses'])
    ax1.annotate(f'Min Val Loss: {data["eval_losses"][min_eval_idx]:.4f}\n(Step {data["eval_steps"][min_eval_idx]})',
                xy=(data['eval_steps'][min_eval_idx], data['eval_losses'][min_eval_idx]),
                xytext=(data['eval_steps'][min_eval_idx]+100, data['eval_losses'][min_eval_idx]+0.05),
                arrowprops=dict(arrowstyle='->', color='#E94F37', lw=2),
                fontsize=9, color='#E94F37', fontweight='bold')

    # 添加过拟合区域标注
    ax1.axvspan(800, 1720, alpha=0.15, color='red', label='Overfitting Zone')
    ax1.text(1260, max(data['train_losses'])*0.9, '⚠️ Overfitting',
             fontsize=10, ha='center', color='darkred', fontweight='bold')

    # ==================== 图2: 学习率变化 ====================
    ax2 = axes[0, 1]

    steps_for_lr = list(range(len(data['learning_rates'])))
    ax2.plot(steps_for_lr, data['learning_rates'],
             color='#F18F01', linewidth=2.5, label='Learning Rate')
    ax2.fill_between(steps_for_lr, data['learning_rates'], alpha=0.3, color='#F18F01')

    ax2.set_xlabel('Log Entry Index', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Learning Rate', fontsize=11, fontweight='bold')
    ax2.set_title('📊 Learning Rate Schedule (Cosine)', fontsize=13, fontweight='bold', pad=10)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # 对数坐标更好展示

    # 标注初始和最终学习率
    ax2.annotate(f'Initial LR: {data["learning_rates"][0]:.2e}',
                xy=(0, data['learning_rates'][0]),
                xytext=(50, data['learning_rates'][0]*1.5),
                arrowprops=dict(arrowstyle='->', color='#F18F01'),
                fontsize=9, color='#F18F01')
    ax2.annotate(f'Final LR: {data["learning_rates"][-1]:.2e}',
                xy=(len(data['learning_rates'])-1, data['learning_rates'][-1]),
                xytext=(len(data['learning_rates'])-200, data['learning_rates'][-1]*100),
                arrowprops=dict(arrowstyle='->', color='#F18F01'),
                fontsize=9, color='#F18F01')

    # ==================== 图3: 训练进度 ====================
    ax3 = axes[1, 0]

    # 使用epoch作为x轴
    epochs_unique = sorted(list(set(data['epochs'])))[:len(data['train_losses']//int(len(data['epochs'])/len(set(data['epochs']))))]

    # 简化：直接用step/total的比例表示进度
    progress = [s/data['train_steps'][-1]*100 for s in data['train_steps']]

    ax3.fill_between(range(len(progress)), progress, alpha=0.4, color='#4CAF50')
    ax3.plot(range(len(progress)), progress, color='#2E7D32', linewidth=2)

    ax3.set_xlabel('Training Log Entries', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Progress (%)', fontsize=11, fontweight='bold')
    ax3.set_title('✅ Training Progress', fontsize=13, fontweight='bold', pad=10)
    ax3.set_ylim([0, 105])
    ax3.axhline(y=100, color='gray', linestyle='--', alpha=0.7, label='Complete')
    ax3.grid(True, alpha=0.3)

    # 添加里程碑标注
    milestones = [25, 50, 75, 100]
    for m in milestones:
        idx = int(len(progress) * m / 100)
        if idx < len(progress):
            ax3.axhline(y=m, color='#FFC107', linestyle=':', alpha=0.6)
            ax3.text(idx+5, m+2, f'{m}%', fontsize=8, color='#FFC107')

    # ==================== 图4: 关键指标统计 ====================
    ax4 = axes[1, 1]
    ax4.axis('off')

    # 计算统计信息
    final_train_loss = data['train_losses'][-1]
    final_eval_loss = data['eval_losses'][-1]
    min_train_loss = min(data['train_losses'])
    min_eval_loss = min(data['eval_losses'])
    total_steps = data['train_steps'][-1]
    total_epochs = data['epochs'][-1] if data['epochs'] else 20

    # 过拟合检测
    overfitting_gap = final_eval_loss - min_eval_loss
    overfitting_status = "⚠️ SEVERE OVERFITTING" if overfitting_gap > 0.1 else "✅ Normal"

    stats_text = f"""
    📈 TRAINING SUMMARY REPORT
    {'─'*40}

    🎯 Model: Qwen2.5-1.5B (LoRA Fine-tuning)
    📚 Task: Agricultural Entity Recognition
    ⏱️  Total Time: ~1h 09m 39s

    {'─'*40}
    📊 LOSS METRICS
    {'─'*40}

    Final Training Loss:   {final_train_loss:.6f}
    Final Validation Loss: {final_eval_loss:.4f}

    Best Training Loss:    {min_train_loss:.6f}
    Best Validation Loss:  {min_eval_loss:.4f}

    Overfitting Gap:       {overfitting_gap:.4f}
    Status: {overfitting_status}

    {'─'*40}
    ⚙️  TRAINING CONFIGURATION
    {'─'*40}

    Total Steps:     {total_steps:,} / 1,720
    Total Epochs:    {total_epochs:.1f} / 20
    Eval Strategy:   Every 20 steps
    Learning Rate:   Cosine schedule (5e-5 → ~0)

    {'─'*40}
    💡 RECOMMENDATIONS
    {'─'*40}

    ✅ Training completed successfully!
    ⚠️  Warning: Severe overfitting detected
       - Train loss very low (~0.001)
       - Val loss increased to ~0.62

    🔧 Suggested improvements:
    • Reduce training epochs (5-10 instead of 20)
    • Add dropout or regularization
    • Use early stopping mechanism
    • Increase validation dataset size
    • Lower learning rate (1e-5)
    """

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5',
                      edgecolor='#4CAF50', alpha=0.9))

    # 保存图表
    if output_path is None:
        output_path = log_path.replace('.jsonl', '_training_analysis.png')

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()  # 关闭图形，释放内存
    print(f"✅ 图表已保存到: {output_path}")

    return output_path

if __name__ == "__main__":
    # 日志文件路径
    LOG_FILE = "/root/llama/LlamaFactory/saves/saves/Qwen2.5-1.5B/2026-08-26-11-58/trainer_log.jsonl"
    OUTPUT_FILE = "/root/llama/LlamaFactory/saves/saves/Qwen2.5-1.5B/2026-08-26-11-58/training_visualization.png"

    # 生成可视化
    create_visualization(LOG_FILE, OUTPUT_FILE)