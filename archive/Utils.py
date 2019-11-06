import pkuseg
import jieba
import pandas as pd


def Pkuseg():
    train_data = pd.read_csv(
        '/Users/kevin/Workspace/food_safety/data/train.csv')
    test_data = pd.read_csv(
        '/Users/kevin/Workspace/food_safety/data/test_new.csv')

    seg = pkuseg.pkuseg(model_name='web')  # 以默认配置加载模型
    train_data['segments'] = train_data['comment'].apply(lambda i: seg.cut(i))
    test_data['segments'] = test_data['comment'].apply(lambda i: seg.cut(i))

    train_data.to_csv(
        '/Users/kevin/Workspace/food_safety/data/segments/web_train.csv',
        index=False)
    test_data.to_csv(
        '/Users/kevin/Workspace/food_safety/data/segments/web_test.csv',
        index=False)


def Jieba():
    train_data = pd.read_csv(
        '/Users/kevin/Workspace/food_safety/data/train.csv')
    test_data = pd.read_csv(
        '/Users/kevin/Workspace/food_safety/data/test_new.csv')

    jieba.enable_parallel(64)  #并行分词开启
    train_data['segments'] = train_data['comment'].apply(
        lambda i: jieba.lcut(i))
    test_data['segments'] = test_data['comment'].apply(
        lambda i: jieba.lcut(i))

    """
    test_data['segments'] = ['/ '.join(i) for i in test_data['segments']]
    train_data['segments'] = ['/ '.join(i) for i in train_data['segments']]
    """

    train_data.to_csv(
        '/Users/kevin/Workspace/food_safety/data/segments/jieba_train.csv',
        index=False)
    test_data.to_csv(
        '/Users/kevin/Workspace/food_safety/data/segments/jieba_test.csv',
        index=False)


Jieba()
