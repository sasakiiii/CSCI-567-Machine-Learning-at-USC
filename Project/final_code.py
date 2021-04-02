import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import os
import gc
import re
import pickle
import time
from xgboost import XGBRegressor
from itertools import product
import time

def process_train_and_test():
    #训练集清洗
    df_train =  pd.read_csv("./data/sales_train.csv")
    df_test = pd.read_csv("./data/test.csv")
    df_train = df_train[df_train.item_price > 0].reset_index(drop = True)
    df_train.loc[df_train.item_cnt_day < 1, "item_cnt_day"] = 0

    #0和57一样
    #1和58一样
    #11和10一样
    #39和40一样
    df_train.loc[df_train.shop_id == 0, "shop_id"] = 57
    df_test.loc[df_test.shop_id == 0 , "shop_id"] = 57
    df_train.loc[df_train.shop_id == 1, "shop_id"] = 58
    df_test.loc[df_test.shop_id == 1 , "shop_id"] = 58
    df_train.loc[df_train.shop_id == 11, "shop_id"] = 10
    df_test.loc[df_test.shop_id == 11, "shop_id"] = 10
    df_train.loc[df_train.shop_id == 40, "shop_id"] = 39
    df_test.loc[df_test.shop_id == 40, "shop_id"] = 39
    return df_train,df_test

def process_shops():
    #shops特征构建
    shops = pd.read_csv("./data/shops.csv")
    shops.loc[ shops.shop_name == 'Сергиев Посад ТЦ "7Я"',"shop_name" ] = 'СергиевПосад ТЦ "7Я"'
    shops["city"] = shops.shop_name.str.split(" ").map( lambda x: x[0] )
    shops["category"] = shops.shop_name.str.split(" ").map( lambda x: x[1] )

    shops["category_marketname"] = shops.shop_name.str.split("\"").map( lambda x: x[1].strip() if len(x)>=2 else x[0])

    shops.loc[shops.city == "!Якутск", "city"] = "Якутск"
    category = []
    for cat in shops.category.unique():
        print(cat, len(shops[shops.category == cat]) )
        if len(shops[shops.category == cat]) > 4:
            category.append(cat)

    shops.category = shops.category.apply( lambda x: x if (x in category) else "etc" )
    for cat in shops.category.unique():
        print(cat, len(shops[shops.category == cat]) )
    shops["shop_category"] = LabelEncoder().fit_transform( shops.category )
    shops["shop_city"] = LabelEncoder().fit_transform( shops.city )
    shops["le_category_marketname"] = LabelEncoder().fit_transform( shops.category_marketname )
    shops = shops[["shop_id", "shop_category", "shop_city","le_category_marketname"]]

    print("shops 特征构建完毕")
    return shops
def process_cats():
    #cats特征构建
    cats = pd.read_csv("./data/item_categories.csv")
    cats["type_code"] = cats.item_category_name.apply( lambda x: x.split(" ")[0] ).astype(str)
    cats.loc[ (cats.type_code == "Игровые")| (cats.type_code == "Аксессуары"), "type_code" ] = "Игры"
    category = []
    for cat in cats.type_code.unique():
        print(cat, len(cats[cats.type_code == cat]))
        if len(cats[cats.type_code == cat]) > 4: 
            category.append( cat )
    cats.type_code = cats.type_code.apply(lambda x: x if (x in category) else "etc")
    for cat in cats.type_code.unique():
        print(cat, len(cats[cats.type_code == cat]))
    cats.type_code = LabelEncoder().fit_transform(cats.type_code)
    cats["split"] = cats.item_category_name.apply(lambda x: x.split("-"))
    cats["subtype"] = cats.split.apply(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
    cats["subtype_code"] = LabelEncoder().fit_transform( cats["subtype"] )
    cats = cats[["item_category_id", "subtype_code", "type_code"]]
    print("cats特征构建完毕")
    return cats
def process_items():
    #item特征构建
    items = pd.read_csv("./data/items.csv")
    def name_correction(x):
        x = x.lower()
        x = x.partition('[')[0]
        x = x.partition('(')[0]
        x = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', x) #替换非数字字母俄文的字符
        x = x.replace('  ', ' ')
        x = x.strip()
        return x

    items["name1"], items["name2"] = items.item_name.str.split("[", 1).str
    items["name1"], items["name3"] = items.item_name.str.split("(", 1).str

    items["name2"] = items.name2.str.replace('[^A-Za-z0-9А-Яа-я]+', " ").str.lower()
    items["name3"] = items.name3.str.replace('[^A-Za-z0-9А-Яа-я]+', " ").str.lower()
    items = items.fillna('0')

    items["item_name"] = items["item_name"].apply(lambda x: name_correction(x))
    items.name2 = items.name2.apply( lambda x: x[:-1] if x !="0" else "0")

    items["type"] = items.name2.apply(lambda x: x[0:8] if x.split(" ")[0] == "xbox" else x.split(" ")[0] )
    items.loc[(items.type == "x360") | (items.type == "xbox360") | (items.type == "xbox 360") ,"type"] = "xbox 360"
    items.loc[ items.type == "", "type"] = "mac"
    items.type = items.type.apply( lambda x: x.replace(" ", "") )
    items.loc[ (items.type == 'pc' )| (items.type == 'pс') | (items.type == "pc"), "type" ] = "pc"
    items.loc[ items.type == 'рs3' , "type"] = "ps3"

    group_sum = items.groupby(["type"]).agg({"item_id": "count"})
    print( group_sum.reset_index() )
    group_sum = group_sum.reset_index()

    drop_cols = []
    for cat in group_sum.type.unique():
    #     print(group_sum.loc[(group_sum.type == cat), "item_id"].values[0])
        if group_sum.loc[(group_sum.type == cat), "item_id"].values[0] <40:
            drop_cols.append(cat)

    items.name2 = items.name2.apply( lambda x: "etc" if (x in drop_cols) else x )
    items = items.drop(["type"], axis = 1)

    items.name2 = LabelEncoder().fit_transform(items.name2)
    items.name3 = LabelEncoder().fit_transform(items.name3)

    items.drop(["item_name", "name1"],axis = 1, inplace= True)
    items.head()
    print("items特征构建完毕")
    return items
def feature_engineer(df_train,df_test,shops,items,cats):
    #时序特征
    print("时序特征构建开始")
    ts = time.time()
    df_matrix = []
    cols  = ["date_block_num", "shop_id", "item_id"]
    for i in range(34):
        sales = df_train[df_train.date_block_num == i]
        df_matrix.append( np.array(list( product( [i], sales.shop_id.unique(), sales.item_id.unique() ) ), dtype = np.int16) )

    df_matrix = pd.DataFrame( np.vstack(df_matrix), columns = cols )
    df_matrix["date_block_num"] = df_matrix["date_block_num"].astype(np.int8)
    df_matrix["shop_id"] = df_matrix["shop_id"].astype(np.int8)
    df_matrix["item_id"] = df_matrix["item_id"].astype(np.int16)
    df_matrix.sort_values( cols, inplace = True )
    time.time()- ts

    df_train["revenue"] = df_train["item_cnt_day"] * df_train["item_price"]

    #每个月每个商店每个商品月销量
    ts = time.time()
    group = df_train.groupby( ["date_block_num", "shop_id", "item_id"] ).agg( {"item_cnt_day": ["sum"]} )
    group.columns = ["item_cnt_month"]
    group.reset_index( inplace = True)
    df_matrix = pd.merge( df_matrix, group, on = cols, how = "left" )
    df_matrix["item_cnt_month"] = df_matrix["item_cnt_month"].fillna(0).clip(0,20).astype(np.float16)
    time.time() - ts


    df_test["date_block_num"] = 34
    df_test["date_block_num"] = df_test["date_block_num"].astype(np.int8)
    df_test["shop_id"] = df_test.shop_id.astype(np.int8)
    df_test["item_id"] = df_test.item_id.astype(np.int16)


    ts = time.time()

    df_matrix = pd.concat([df_matrix, df_test.drop(["ID"],axis = 1)], ignore_index=True, sort=False, keys=cols)
    df_matrix.fillna( 0, inplace = True )
    time.time() - ts


    #将shop特征，item特征，cats特征 合并入最后的数据形式
    ts = time.time()
    df_matrix = pd.merge( df_matrix, shops, on = ["shop_id"], how = "left" )
    df_matrix = pd.merge(df_matrix, items, on = ["item_id"], how = "left")
    df_matrix = pd.merge( df_matrix, cats, on = ["item_category_id"], how = "left" )
    df_matrix["shop_city"] = df_matrix["shop_city"].astype(np.int8)
    df_matrix["shop_category"] = df_matrix["shop_category"].astype(np.int8)
    df_matrix["item_category_id"] = df_matrix["item_category_id"].astype(np.int8)
    df_matrix["subtype_code"] = df_matrix["subtype_code"].astype(np.int8)
    df_matrix["name2"] = df_matrix["name2"].astype(np.int8)
    df_matrix["name3"] = df_matrix["name3"].astype(np.int16)
    df_matrix["type_code"] = df_matrix["type_code"].astype(np.int8)
    time.time() - ts

    #此函数用来计算偏移时序特征，比如上一个月，上两个月
    def lag_feature( df,lags, cols ):
        for col in cols:
            print(col)
            tmp = df[["date_block_num", "shop_id","item_id",col ]]
            for i in lags:
                shifted = tmp.copy()
                shifted.columns = ["date_block_num", "shop_id", "item_id", col + "_lag_"+str(i)]
                shifted.date_block_num = shifted.date_block_num + i
                df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
        return df

    ts = time.time()

    #计算前一、二、三个月的商品月销售数量
    df_matrix = lag_feature( df_matrix, [1,2,3], ["item_cnt_month"] )
    time.time() - ts

    #每个月平均销售数量
    ts = time.time()
    group = df_matrix.groupby( ["date_block_num"] ).agg({"item_cnt_month" : ["mean"]})
    group.columns = ["date_avg_item_cnt"]
    group.reset_index(inplace = True)

    df_matrix = pd.merge(df_matrix, group, on = ["date_block_num"], how = "left")
    df_matrix.date_avg_item_cnt = df_matrix["date_avg_item_cnt"].astype(np.float16)
    df_matrix = lag_feature( df_matrix, [1], ["date_avg_item_cnt"] )
    df_matrix.drop( ["date_avg_item_cnt"], axis = 1, inplace = True )
    time.time() - ts

    #每个月 所有商店 某个商品平均销售数量
    ts = time.time()
    group = df_matrix.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})
    group.columns = [ 'date_item_avg_item_cnt' ]
    group.reset_index(inplace=True)

    df_matrix = pd.merge(df_matrix, group, on=['date_block_num','item_id'], how='left')
    df_matrix.date_item_avg_item_cnt = df_matrix['date_item_avg_item_cnt'].astype(np.float16)
    df_matrix = lag_feature(df_matrix, [1,2,3], ['date_item_avg_item_cnt'])
    df_matrix.drop(['date_item_avg_item_cnt'], axis=1, inplace=True)
    time.time() - ts

    #每个月 某商店的平均商品销售数量
    ts = time.time()
    group = df_matrix.groupby( ["date_block_num","shop_id"] ).agg({"item_cnt_month" : ["mean"]})
    group.columns = ["date_shop_avg_item_cnt"]
    group.reset_index(inplace = True)

    df_matrix = pd.merge(df_matrix, group, on = ["date_block_num","shop_id"], how = "left")
    df_matrix.date_avg_item_cnt = df_matrix["date_shop_avg_item_cnt"].astype(np.float16)
    df_matrix = lag_feature( df_matrix, [1,2,3], ["date_shop_avg_item_cnt"] )
    df_matrix.drop( ["date_shop_avg_item_cnt"], axis = 1, inplace = True )
    time.time() - ts

    #每个月 某商店某商品的平均销售数量
    ts = time.time()
    group = df_matrix.groupby( ["date_block_num","shop_id","item_id"] ).agg({"item_cnt_month" : ["mean"]})
    group.columns = ["date_shop_item_avg_item_cnt"]
    group.reset_index(inplace = True)

    df_matrix = pd.merge(df_matrix, group, on = ["date_block_num","shop_id","item_id"], how = "left")
    df_matrix.date_avg_item_cnt = df_matrix["date_shop_item_avg_item_cnt"].astype(np.float16)
    df_matrix = lag_feature( df_matrix, [1,2,3], ["date_shop_item_avg_item_cnt"] )
    df_matrix.drop( ["date_shop_item_avg_item_cnt"], axis = 1, inplace = True )
    time.time() - ts

    #每个月 每个商店 每个商品小类的平均销售数量
    ts = time.time()
    group = df_matrix.groupby(['date_block_num', 'shop_id', 'subtype_code']).agg({'item_cnt_month': ['mean']})
    group.columns = ['date_shop_subtype_avg_item_cnt']
    group.reset_index(inplace=True)

    df_matrix = pd.merge(df_matrix, group, on=['date_block_num', 'shop_id', 'subtype_code'], how='left')
    df_matrix.date_shop_subtype_avg_item_cnt = df_matrix['date_shop_subtype_avg_item_cnt'].astype(np.float16)
    df_matrix = lag_feature(df_matrix, [1], ['date_shop_subtype_avg_item_cnt'])
    df_matrix.drop(['date_shop_subtype_avg_item_cnt'], axis=1, inplace=True)
    time.time() - ts

    #每个月 每个城市的平均商品销售数量
    ts = time.time()
    group = df_matrix.groupby(['date_block_num', 'shop_city']).agg({'item_cnt_month': ['mean']})
    group.columns = ['date_city_avg_item_cnt']
    group.reset_index(inplace=True)

    df_matrix = pd.merge(df_matrix, group, on=['date_block_num', "shop_city"], how='left')
    df_matrix.date_city_avg_item_cnt = df_matrix['date_city_avg_item_cnt'].astype(np.float16)
    df_matrix = lag_feature(df_matrix, [1], ['date_city_avg_item_cnt'])
    df_matrix.drop(['date_city_avg_item_cnt'], axis=1, inplace=True)
    time.time() - ts


    #每个月 每个城市 每个商品的平均月销售量
    ts = time.time()
    group = df_matrix.groupby(['date_block_num', 'item_id', 'shop_city']).agg({'item_cnt_month': ['mean']})
    group.columns = [ 'date_item_city_avg_item_cnt' ]
    group.reset_index(inplace=True)

    df_matrix = pd.merge(df_matrix, group, on=['date_block_num', 'item_id', 'shop_city'], how='left')
    df_matrix.date_item_city_avg_item_cnt = df_matrix['date_item_city_avg_item_cnt'].astype(np.float16)
    df_matrix = lag_feature(df_matrix, [1], ['date_item_city_avg_item_cnt'])
    df_matrix.drop(['date_item_city_avg_item_cnt'], axis=1, inplace=True)
    time.time() - ts

    #每个商品的平均售价
    ts = time.time()
    group = df_train.groupby( ["item_id"] ).agg({"item_price": ["mean"]})
    group.columns = ["item_avg_item_price"]
    group.reset_index(inplace = True)

    df_matrix = df_matrix.merge( group, on = ["item_id"], how = "left" )
    df_matrix["item_avg_item_price"] = df_matrix.item_avg_item_price.astype(np.float16)

    group = df_train.groupby( ["date_block_num","item_id"] ).agg( {"item_price": ["mean"]} )
    group.columns = ["date_item_avg_item_price"]
    group.reset_index(inplace = True)

    df_matrix = df_matrix.merge(group, on = ["date_block_num","item_id"], how = "left")
    df_matrix["date_item_avg_item_price"] = df_matrix.date_item_avg_item_price.astype(np.float16)
    lags = [1, 2, 3]
    df_matrix = lag_feature( df_matrix, lags, ["date_item_avg_item_price"] )
    for i in lags:
        df_matrix["delta_price_lag_" + str(i) ] = (df_matrix["date_item_avg_item_price_lag_" + str(i)]- df_matrix["item_avg_item_price"] )/ df_matrix["item_avg_item_price"]

    def select_trends(row) :
        for i in lags:
            if row["delta_price_lag_" + str(i)]:
                return row["delta_price_lag_" + str(i)]
        return 0

    df_matrix["delta_price_lag"] = df_matrix.apply(select_trends, axis = 1)
    df_matrix["delta_price_lag"] = df_matrix.delta_price_lag.astype( np.float16 )
    df_matrix["delta_price_lag"].fillna( 0 ,inplace = True)

    features_to_drop = ["item_avg_item_price", "date_item_avg_item_price"]
    for i in lags:
        features_to_drop.append("date_item_avg_item_price_lag_" + str(i) )
        features_to_drop.append("delta_price_lag_" + str(i) )
    df_matrix.drop(features_to_drop, axis = 1, inplace = True)
    time.time() - ts


    #每个月 每个商店的营业额
    ts = time.time()
    group = df_train.groupby( ["date_block_num","shop_id"] ).agg({"revenue": ["sum"] })
    group.columns = ["date_shop_revenue"]
    group.reset_index(inplace = True)

    df_matrix = df_matrix.merge( group , on = ["date_block_num", "shop_id"], how = "left" )
    df_matrix['date_shop_revenue'] = df_matrix['date_shop_revenue'].astype(np.float32)

    group = group.groupby(["shop_id"]).agg({ "date_block_num":["mean"] })
    group.columns = ["shop_avg_revenue"]
    group.reset_index(inplace = True )

    df_matrix = df_matrix.merge( group, on = ["shop_id"], how = "left" )
    df_matrix["shop_avg_revenue"] = df_matrix.shop_avg_revenue.astype(np.float32)
    df_matrix["delta_revenue"] = (df_matrix['date_shop_revenue'] - df_matrix['shop_avg_revenue']) / df_matrix['shop_avg_revenue']
    df_matrix["delta_revenue"] = df_matrix["delta_revenue"]. astype(np.float32)

    df_matrix = lag_feature(df_matrix, [1], ["delta_revenue"])
    df_matrix["delta_revenue_lag_1"] = df_matrix["delta_revenue_lag_1"].astype(np.float32)
    df_matrix.drop( ["date_shop_revenue", "shop_avg_revenue", "delta_revenue"] ,axis = 1, inplace = True)
    time.time() - ts

    #月份特征
    df_matrix["month"] = df_matrix["date_block_num"] % 12
    days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])
    df_matrix["days"] = df_matrix["month"].map(days).astype(np.int8)

    #距离首售日期特征
    ts = time.time()
    df_matrix["item_shop_first_sale"] = df_matrix["date_block_num"] - df_matrix.groupby(["item_id","shop_id"])["date_block_num"].transform('min')
    df_matrix["item_first_sale"] = df_matrix["date_block_num"] - df_matrix.groupby(["item_id"])["date_block_num"].transform('min')
    time.time() - ts

    #因为很多时序特征需要计算前一个月，前两个月，前三个月的内容，
    #而第一个月，第二个月，第三个月的计算这些特征时为空值，所以去除前三个月
    df_matrix = df_matrix[df_matrix["date_block_num"] > 3]
    return df_matrix

if __name__=="__main__":
    df_train,df_test = process_train_and_test()
    shops = process_shops()
    cats = process_cats()
    items = process_items()
    data = feature_engineer(df_train,df_test,shops,items,cats)

    #训练集测试集划分
    X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
    Y_train = data[data.date_block_num < 33]['item_cnt_month']
    X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
    Y_valid = data[data.date_block_num == 33]['item_cnt_month']
    X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
    del data

    ts = time.time()
    #XGBOOST模型构建
    model = XGBRegressor(
        max_depth=10,
        n_estimators=500,
        min_child_weight=0.5, 
        colsample_bytree=0.8, 
        subsample=0.8, 
        eta=0.1,
        seed=2020)

    model.fit(
        X_train, 
        Y_train, 
        eval_metric="rmse", 
        eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
        verbose=True, 
        early_stopping_rounds = 20)

    time.time() - ts

    Y_test = model.predict(X_test).clip(0, 20)

    submission = pd.DataFrame({
        "ID": df_test.index, 
        "item_cnt_month": Y_test
    })
    submission.to_csv('xgb_submission_{}.csv'.format(time.strftime("%d_%m_%Y_%H_%M_%S")), index=False)
    
    #lightgbm模型构建
    import lightgbm as lgb
    gbm = lgb.LGBMRegressor(objective='regression',
                            num_leaves=31,
                            learning_rate=0.01,
                            n_estimators=300)
    gbm.fit(X_train, Y_train,
            eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
            eval_metric='l2',
            early_stopping_rounds=10)

    gbm_Y_test = gbm.predict(X_test).clip(0, 20)
    submission = pd.DataFrame({
        "ID": df_test.index, 
        "item_cnt_month": gbm_Y_test
    })
    submission.to_csv('lgb_submission_{}.csv'.format(time.strftime("%d_%m_%Y_%H_%M_%S")), index=False)

    #catboost模型构建
    import catboost as cbt
    cbt_model = cbt.CatBoostRegressor(iterations=180,learning_rate=0.1)
    cbt_model.fit(X_train, Y_train,eval_set=(X_valid, Y_valid),verbose = True)

    cbt_Y_test = cbt_model.predict(X_test).clip(0, 20)
    submission = pd.DataFrame({
        "ID": df_test.index, 
        "item_cnt_month": cbt_Y_test
    })
    submission.to_csv('cbt_submission_{}.csv'.format(time.strftime("%d_%m_%Y_%H_%M_%S")), index=False)

    #三个模型融合
    blend_Y_test = (cbt_Y_test + gbm_Y_test + Y_test)/3
    submission = pd.DataFrame({
        "ID": df_test.index, 
        "item_cnt_month": blend_Y_test
    })
    submission.to_csv('blending_submission_{}.csv'.format(time.strftime("%d_%m_%Y_%H_%M_%S")), index=False)