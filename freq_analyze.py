import pandas as pd

train_file = '/Users/kevin/Workspace/food_safety/data/segments/web_train.csv'
test_file = '/Users/kevin/Workspace/food_safety/data/segments/web_test.csv'


def read_data(train_file, test_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    return train_data, test_data


train_data, test_data = read_data(train_file, test_file)

other_bad = []
bad_dict = {
    'total': 0,
    '馊': [],
    '馊_wrong': [],
    '虫': [],
    '虫_wrong': [],
    '苍蝇': [],
    '苍蝇_wrong': [],
    '蟑螂': [],
    '蟑螂_wrong': [],
    '小强': [],
    '小强_wrong': [],
    '蚊子': [],
    '蚊子_wrong': [],
    '臭': [],
    '臭_wrong': [],
    '变质': [],
    '变质_wrong': [],
    '钢丝': [],
    '钢丝_wrong': [],
    '头发': [],
    '头发_wrong': [],
    '毛发': [],
    '毛发_wrong': [],
    '霉': [],  # 24 / 15
    '霉_wrong': [],  # 2 / 2
    '反胃': [],  # 19 / 14
    '反胃_wrong': [],  # 0
    '拉肚子': [],  # 245 / 229
    '拉肚子_wrong': [],  # 7 / 7
    '地沟油': [],  # 12 / 10
    '地沟油_wrong': [],  # 0
    '食物中毒': [],  # 10 / 6
    '食物中毒_wrong': [],  # 0
    '异味': [],  # 29 / 26
    '异味_wrong': [],  # 3 / 2
    '不新鲜': [],  # 112 / 21    #NOTE
    '不新鲜_wrong': [],  # 12 / 8
    '过期': [],  # 6 / 3
    '过期_wrong': [],  # 1 / 1
    '不干净': [],  # 33 / 10
    '不干净_wrong': [],  # 2 / 2
    '疼': [],  # 30 / 21
    '疼_wrong': [],  # 3 / 2
    '痛': [],  # 30 / 9
    '痛_wrong': [],  # 7 / 7
    '怪味': [],  # 14 / 7
    '怪味_wrong': [],  # 4 / 4
    '没熟': [],  # 16 / 7
    '没熟_wrong': [],  # 9 / 9
    '吐': [],  # 79 / 21
    '吐_wrong': [],  # 26 / 25
    '无语': [],  # 55 / 7
    '无语_wrong': [],  # 38 / 38
    '腥': [],  # 25 / 7
    '腥_wrong': [],  # 10 / 10
    '硬': [],  # 27 / 4
    '硬_wrong': [],  # 41 / 38
    '恶心': [],  # 122 / 20
    '恶心_wrong': [],  # 14 / 11
    '差评': [],  # 109 / 19
    '差评_wrong': [],  # 122 / 109
    '难吃': [],  # 153 / 40
    '难吃_wrong': [],  # 427 / 364 很多人说着差评给了好评？？？
    '垃圾': [],  # 27 / 7
    '垃圾_wrong': [],  # 31 / 17
    '糊': [],  # 17 / 7
    '糊_wrong': [],  # 31 / 17
    '脏': [],  # 10 / 3
    '脏_wrong': [],  # 9 / 9
    '烂': [],  # 28 / 16
    '烂_wrong': [],  # 16 / 16
    '添加剂': [],  # 3 / 2
    '添加剂_wrong': [],  # 3 / 1
    '味精': [],  # 2 / 0
    '味精_wrong': [],  # 13 / 9
    '辣': [],  # 71 / 3
    '辣_wrong': [],  # 472 / 34
    '沙子': [],  # 2 / 1
    '沙子_wrong': [],  # 2 / 1
    '不好吃': [],  # 56 / 12
    '不好吃_wrong': [],  # 255 / 188
    '花甲': [],  # 10 / 2
    '花甲_wrong': [],  # 12 / 7
    '虾': [],  # 59 / 12
    '虾_wrong': [],  # 132 / 84
    '失望': [],  # 32 / 1
    '失望_wrong': [],  # 53 / 26
}
'''
情感分析
'''

key_list = bad_dict.keys()
isFound = False
for index, row in train_data.iterrows():
    label = row[0]
    comment = row[1]
    segment = row[2]
    for key in key_list:
        if key in comment:
            isFound = True
            if label == 1:
                bad_dict[key].append(comment)
                bad_dict['total'] += 1
            else:
                incorect = key + '_wrong'
                bad_dict[incorect].append(comment)
            break
    if not isFound and label == 1:
        other_bad.append(comment)
    isFound = False

for key in bad_dict:
    try:
        print(key + ": " + str(len(bad_dict[key])))
    except TypeError:
        print("total: " + str(bad_dict[key]))
