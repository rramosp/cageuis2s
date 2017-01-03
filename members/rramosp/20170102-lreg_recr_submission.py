import kagglegym
import numpy as np
import pandas as pd
import xgboost as xgb

def diffs2signal(s):
    r = np.ones(len(s)+1)
    for i in range(len(s)):
        r[i+1] = r[i]*(1+s[i])
    r = r[1:]
    return r

def signal2diffs(s):
    return np.array([1]+list(s[1:]/s[:-1]))-1
    
low_y_cut = -0.086093
high_y_cut = 0.093497

env = kagglegym.make()
observation = env.reset()
df=observation.train
print (df.shape)
y_values_within = ((df['y'] > low_y_cut) & (df['y'] <high_y_cut))
gf = df.loc[y_values_within,:].copy()
gf.fillna(0,inplace=True)
gf["y_recr"] = np.zeros(len(gf))
print (gf.shape)

gf = gf.set_index(['id', "timestamp"]).reindex()

for i in gf.index.levels[0].values:
    gf.ix[i,"y_recr"] = diffs2signal(gf.ix[i].y.values) 
    
cols =[u'fundamental_7', u'technical_30', u'fundamental_15', u'fundamental_60',
       u'fundamental_20', u'technical_34', u'technical_36', u'technical_35',
       u'technical_27', u'technical_43']
cols_pos =[u'fundamental_7',
 u'fundamental_15',
 u'technical_30',
 u'fundamental_60',
 u'fundamental_20',
 u'fundamental_56',
 u'fundamental_53',
 u'fundamental_55',
 u'fundamental_57',
 u'fundamental_2',
 u'fundamental_40',
 u'fundamental_26',
 u'technical_22',
 u'fundamental_11',
 u'technical_7',
 u'technical_12',
 u'fundamental_52',
 u'fundamental_36',
 u'technical_37',
 u'fundamental_35',
 u'fundamental_48',
 u'technical_38',
 u'technical_39',
 u'technical_32',
 u'fundamental_58',
 u'derived_1',
 u'fundamental_10',
 u'fundamental_30',
 u'fundamental_0',
 u'technical_0',
 u'technical_9',
 u'fundamental_24',
 u'fundamental_63',
 u'fundamental_61',
 u'technical_42',
 u'technical_31',
 u'technical_18',
 u'fundamental_22',
 u'fundamental_23',
 u'fundamental_5']

cols_neg = [u'technical_2',
 u'fundamental_51',
 u'fundamental_31',
 u'fundamental_8',
 u'fundamental_62',
 u'technical_20',
 u'fundamental_25',
 u'technical_10',
 u'technical_11',
 u'technical_29',
 u'fundamental_45',
 u'technical_21',
 u'technical_6',
 u'technical_14',
 u'technical_19',
 u'technical_36',
 u'technical_34',
 u'technical_35',
 u'technical_43',
 u'technical_27']
cols = cols_neg
#cols =[u'fundamental_7', u'technical_30', u'fundamental_15', u'fundamental_60']
cols = ["technical_20"]
cols=cols_pos[:5]
x_train=gf[cols].values
y = gf.y_recr.values
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
lr_model = LinearRegression()

lr_model = Pipeline([('pca', PCA(n_components=2)), 
                     ('regr', LinearRegression())])
lr_model.fit(x_train, y)

last_y = {}
for i in np.unique(gf.index.levels[0]):
    last_y[i] = gf.ix[0].y_recr.values[-1]
observation = env.reset()
rewards = []
while True:
    observation.features.fillna(0, inplace=True)
    preds = lr_model.predict(observation.features[cols].values)
    prevs = np.array([last_y[i] if i in last_y.keys() else 1 for i in observation.features.id.values])
    y_target = preds/prevs-1    
    
    observation.target.y = y_target

    for i,aid in enumerate(observation.features.id.values):
        last_y[aid] = preds[i]
    
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    
    # We perform a "step" by making our prediction and getting back an updated "observation":
    observation, reward, done, info = env.step(target)
    rewards.append(reward)
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp), reward)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break