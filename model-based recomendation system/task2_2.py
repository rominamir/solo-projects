import xgboost as xgb
from pyspark import SparkContext
import sys
import time
import json
import pandas as pd
from sklearn.metrics import mean_squared_error
from datetime import date

out_file = 'out_2_2.csv'
train_file = 'yelp_train.csv'
test_file = 'yelp_val.csv'

#
# tip = folder_path + '/tip.json'
# user = folder_path + '/user.json'
# checkin = folder_path + '/checkin.json'
# business = folder_path + '/business.json'
# review_tr = folder_path + '/review_train.json'

tip = 'tip.json'
user = 'user.json'

checkin = 'checkin.json'
business = 'business.json'
review_tr = 'review_train.json'

sc = SparkContext()




def days(d):
    l = d.split('-')
    s = date(int(l[0]), int(l[1]), int(l[2]))
    e = date(2021, 3, 26)
    return int(str(e - s).split(' days')[0])

def x(train, test):
    l = []
    with open(train, 'r') as f:
        for i in list(f):
            s = i.split(',')
            l.append([s[0], s[1], s[2].split('\n')[0]])

    ll = []
    with open(test, 'r') as f:
        for i in list(f):
            s = i.split(',')
            ll.append([s[0], s[1]])

    return l[1:], ll[1:]


def read_user_file(file):
    l = {}
    with open(file, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = json.loads(ln)
            l[ln['user_id']] = [ln['review_count'], ln['useful'], ln['fans'], len(ln['friends']), ln['average_stars'],
                                days(ln['yelping_since'])]

    return l


def read_files():
    business_files = [checkin, business, review_tr]

    def load_json(d):
        obj = {}
        try:
            obj = json.loads(d)
        except:
            pass
        return obj

    def f(d):
        return [d[0], [x for x in d[1][0][0]] + [y for y in d[1][0][1]] + [d[1][1]]]

    tips_file = sc.textFile(tip)
    tip_file = tips_file.map(lambda ln: load_json(ln))
    tip_count = tip_file.map(lambda x: (x['business_id'], 1)).reduceByKey(lambda x, y: x + y).map(
        lambda x: (x[0], x[1]))
    like_count = tip_file.map(lambda x: (x['business_id'], int(x['likes']))).reduceByKey(lambda x, y: x + y).map(
        lambda x: (x[0], x[1]))

    review_file = sc.textFile(business_files[2])
    review_features = review_file.map(lambda ln: load_json(ln)).filter(lambda x: 'useful' in x.keys()).map(
        lambda x: (x['business_id'], x['useful']))

    business_file = sc.textFile(business_files[1])
    bus_features = business_file.map(lambda ln: load_json(ln)).filter(lambda l: 'attributes' in l.keys()).map(
        lambda x: (x['business_id'],
                   [x['review_count'], x['stars'], x['is_open'], len(x['hours'].keys()) if x.get('hours', False) else 0,
                    len(x['categories'].split(',')) if x.get('categories', False) else 0,
                    (0 if x['attributes'] == None else sum(1 for condition in x['attributes'].values() if condition))]))

    count_features = tip_count.join(like_count)
    business_features = bus_features.join(count_features)
    business_features = business_features.join(review_features)
    business_features = business_features.map(lambda x: f(x)).collectAsMap()

    train, test = x(train_file, test_file)
    user_features = read_user_file(user)

    l = []
    for i in train:
        u = user_features.get(i[0], None)
        b = business_features.get(i[1], None)
        if u is not None and b is not None:
            l.append([user_features[i[0]] + business_features[i[1]], i[2]])

    val_l = []
    for i in test:
        u = user_features.get(i[0], None)
        b = business_features.get(i[1], None)
        if u is not None and b is not None:
            val_l.append([user_features[i[0]] + business_features[i[1]]])

    with open('gxb_input_2_2.csv', 'w') as f:
        for r in l[:-1]:
            s = ''
            for i in r[0]:
                s += str(i) + ','
            s += str(r[1]) + '\n'
            f.write(s)

    with open('gxb_val_2_2.csv', 'w') as f:
        for r in val_l[:-1]:
            s = ''
            for i in r[0]:
                s += str(i) + ','
            s += '\n'
            f.write(s)

start = time.time()
read_files()


def xgb_model(model_file, val_file):
    data = pd.read_csv(model_file)
    val = pd.read_csv(val_file)
    data.columns = ['A', 'B', 'C', 'D', 'E', 'F', 'H', 'G', 'I', 'J', 'K', 'L', 'M','N', 'O', 'TT']
    val.columns = ['A', 'B', 'C', 'D', 'E', 'F', 'H', 'G', 'I', 'J', 'K', 'L', 'M','N', 'O']

    x_train = data[['A', 'B', 'C', 'D', 'E', 'F', 'H', 'G', 'I', 'J', 'K', 'L', 'M','N', 'O']]
    y_train = data['TT'].fillna(0)

    x_val = val[['A', 'B', 'C', 'D', 'E', 'F', 'H', 'G', 'I', 'J', 'K', 'L', 'M','N', 'O']]

    model = xgb.XGBRegressor(colsample_bytree=0.7, gamma=0.3, max_depth=2, min_child_weight=5, subsample=1.0)

    model.fit(x_train, y_train)
    pred = model.predict(x_val)

    validate = pd.read_csv(test_file)
    pred = pd.DataFrame(pred, columns=['prediction'])
    df = pd.concat([validate.iloc[:, 0:2], pred], axis=1)

    df.to_csv(out_file, index=False, header=True)

    return out_file


xgb_model(model_file='gxb_input_2_2.csv', val_file='gxb_val_2_2.csv')

print('xgb model' , time.time() - start)
