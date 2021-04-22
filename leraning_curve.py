import numpy as np
import matplotlib.pyplot as plt
import xlrd  # 导入库


def draw_result(train_loss, val_loss,val_acc,test_acc,title):
    plt.figure(figsize=(12, 8), dpi=120)
    plt.plot(range(len(train_loss)), train_loss, '-b', label='train loss')
    plt.plot(range(len(val_loss)), val_loss, '-r', label='valid loss')
    plt.plot(range(len(val_acc)), val_acc, '-y', label='valid accuracy')
    plt.plot(range(len(test_acc)), test_acc, '-c', label='test accuracy')

    plt.xlabel("Epoch")
    plt.legend(loc='upper right')
    plt.title(title)
    plt.savefig(title+".png")  # should before show method

    plt.show()


def test_draw():
    xlsx = xlrd.open_workbook('learning_curve.xlsx')
    sheet1 = xlsx.sheets()[0]
    sheet1_cols = sheet1.ncols
    sheet1_nrows = sheet1.nrows
    train_loss=[]
    val_loss=[]
    val_acc=[]
    test_acc=[]
    for i in range(sheet1_nrows):
        if i>0:
            train_loss.append(sheet1.row_values(i)[1])
            val_loss.append(sheet1.row_values(i)[2])
            val_acc.append(sheet1.row_values(i)[3])
            test_acc.append(sheet1.row_values(i)[4])
    draw_result(train_loss, val_loss,val_acc,test_acc,"learning_curve")


if __name__ == '__main__':
    test_draw()
