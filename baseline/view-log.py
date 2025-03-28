import sys
import torch
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt


def main():
    path = sys.argv[1]
    results = torch.load(path)

    # 提取验证准确率并计算每个epoch的平均值
    val_acc = torch.FloatTensor(results['tracker']['val_acc'])
    val_acc = val_acc.mean(dim=1).numpy()

    # 找到最大准确率及其对应的epoch
    max_acc = val_acc.max()
    max_epoch = val_acc.argmax()

    plt.figure()
    plt.plot(val_acc, color='black')

    # 每10个epoch标记准确率
    for epoch in range(0, len(val_acc), 10):
        plt.text(epoch, val_acc[epoch], f'{val_acc[epoch]:.2f}', 
                 color='blue', fontsize=8, ha='center', va='bottom')

    # 标记最高点
    plt.scatter(max_epoch, max_acc, color='red', label=f'Max Accuracy: {max_acc:.2f} (Epoch {max_epoch})')
    plt.text(max_epoch, max_acc, f'{max_acc:.2f}', 
             color='red', fontsize=10, ha='center', va='bottom')

    plt.xlabel('Epochs')  # 添加x轴标签
    plt.ylabel('Accuracy')  # 添加y轴标签
    plt.grid(True)  # 显示网格
    plt.legend()

    plt.savefig('val_acc.png')
    print(f'Figure saved as "val_acc.png". Max accuracy: {max_acc:.2f} at epoch {max_epoch}')

if __name__ == '__main__':
    main()
