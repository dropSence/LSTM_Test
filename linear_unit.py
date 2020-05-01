from perceptron import Perception

# 定义激活函数
f = lambda x: x

class LinearUnit(Perception):
    def __init__(self, input_num):
        """
            初始化线性函数,设置输入参数的个数
        """
        Perception.__init__(self, input_num, f)

def get_trainning_dataset():
    """
        随便写几个人的数据
    """
    # 构建训练数据
    # 输入向量列表, 每一项是工作年限
    input_vecs = [[1, 2], [2, 1]]
    # 期望的输出列表,月薪
    labels = [1, 2]
    return input_vecs, labels


def train_linear_unit():
    """
        使用数据训练线性单元
    """
    # 创建感知器,输入的参数特征为1(一个特征,即工作年限)
    lu = LinearUnit(2)
    # 训练,迭代十轮,学习速率为0.01
    input_vecs, labels = get_trainning_dataset()
    # print(lu)
    lu.train(input_vecs, labels, 1, 0.1)
    # 返回训练好的线性单元
    return lu

if __name__ == "__main__":
    """
        训练线性单元
    """
    linear_unit = train_linear_unit()
    # 打印训练获得的权重
    print(linear_unit)
    # 测试
    str1 = "工作3.4年的月薪预计是:{}"
    str2 = "工作20年的月薪预计是:{}"
    str3 = "工作10年的月薪预计是:{}"
    str4 = "工作9年的月薪预计是:{}"
    print(str1.format(linear_unit.predict([1, 2])))
    print(str2.format(linear_unit.predict([2, 1])))
    print(str3.format(linear_unit.predict([3, 1])))
    print(str4.format(linear_unit.predict([4, 1])))
