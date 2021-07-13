#%%
import pandas as pd
import numpy as np
from tqdm import tqdm


train_df=pd.read_csv('train.csv')
test_df=pd.read_csv('test.csv')

train_df.loc[train_df.임대보증금=='-', '임대보증금'] = np.nan
test_df.loc[test_df.임대보증금=='-', '임대보증금'] = np.nan
train_df['임대보증금'] = train_df['임대보증금'].astype(float)
test_df['임대보증금'] = test_df['임대보증금'].astype(float)

train_df.loc[train_df.임대료=='-', '임대료'] = np.nan
test_df.loc[test_df.임대료=='-', '임대료'] = np.nan
train_df['임대료'] = train_df['임대료'].astype(float)
test_df['임대료'] = test_df['임대료'].astype(float)


#%%
train_df['면적당보증금']=train_df['임대보증금']/train_df['전용면적']
test_df['면적당보증금']=test_df['임대보증금']/test_df['전용면적']

train_df['면적당임대료']=train_df['임대료']/train_df['전용면적']
test_df['면적당임대료']=test_df['임대료']/test_df['전용면적']

df=train_df[train_df['면적당보증금'].isnull()==False]
for i in train_df['단지코드'].drop_duplicates():
    train_df.loc[train_df['단지코드']==i,'단지별면적당보증금평균']=np.mean(df.loc[df['단지코드']==i,'면적당보증금'])
    train_df.loc[train_df['단지코드']==i,'단지별면적당임대료평균']=np.mean(df.loc[df['단지코드']==i,'면적당임대료'])


df=test_df[test_df['면적당보증금'].isnull()==False]
for i in test_df['단지코드'].drop_duplicates():
    test_df.loc[test_df['단지코드']==i,'단지별면적당보증금평균']=np.mean(df.loc[df['단지코드']==i,'면적당보증금'])
    test_df.loc[test_df['단지코드']==i,'단지별면적당임대료평균']=np.mean(df.loc[df['단지코드']==i,'면적당임대료'])


#%%
test_df.loc[test_df.단지코드.isin(['C2411']) & test_df.자격유형.isnull(), '자격유형'] = 'A'
test_df.loc[test_df.단지코드.isin(['C2253']) & test_df.자격유형.isnull(), '자격유형'] = 'C'

#%%
unique_cols = ['총세대수', '지역', '공가수', 
               '도보 10분거리 내 지하철역 수(환승노선 수 반영)',
               '도보 10분거리 내 버스정류장 수',
               '단지내주차면수', '등록차량수',
               '단지별면적당보증금평균','단지별면적당임대료평균']
train_agg = train_df.set_index('단지코드')[unique_cols].drop_duplicates()
test_agg = test_df.set_index('단지코드')[[col for col in unique_cols if col!='등록차량수']].drop_duplicates()

#%%

# 이걸 보고 순서대로 0,1,2,3,4, 입력해보기
df=pd.DataFrame(train_agg.groupby('지역')['등록차량수'].mean().sort_values())
for i,v in enumerate(df.index):
    train_agg.loc[train_agg['지역']==v,'지역']=i
    test_agg.loc[test_agg['지역']==v,'지역']=i

#%%
for i in train_df['자격유형'].unique():
    train_df['자격유형_{}'.format(i)]=0

for i in train_df['공급유형'].unique():
    train_df['공급유형_{}'.format(i)]=0

for i in train_df['단지코드'].unique():
    df=train_df[train_df['단지코드']==i]
    qual_columns=df['자격유형'].unique()
    sup_columns=df['공급유형'].unique()
    
    for z in qual_columns:
        train_df.loc[train_df['단지코드']==i,'자격유형_{}'.format(z)]=df[df['자격유형']==z]['전용면적별세대수'].sum()/df['전용면적별세대수'].sum()
    for z in sup_columns:
        train_df.loc[train_df['단지코드']==i,'공급유형_{}'.format(z)]=df[df['공급유형']==z]['전용면적별세대수'].sum()/df['전용면적별세대수'].sum()

train_df_2=train_df[['단지코드','자격유형_A', '자격유형_B', '자격유형_C', '자격유형_D',
       '자격유형_E', '자격유형_F', '자격유형_G', '자격유형_H', '자격유형_I', '자격유형_J', '자격유형_K',
       '자격유형_L', '자격유형_M', '자격유형_N', '자격유형_O', '공급유형_국민임대', '공급유형_공공임대(50년)',
       '공급유형_영구임대', '공급유형_임대상가', '공급유형_공공임대(10년)', '공급유형_공공임대(분납)',
       '공급유형_장기전세', '공급유형_공공분양', '공급유형_행복주택', '공급유형_공공임대(5년)']].drop_duplicates()
#%%

for i in test_df['자격유형'].unique():
    test_df['자격유형_{}'.format(i)]=0

for i in test_df['공급유형'].unique():
    test_df['공급유형_{}'.format(i)]=0

for i in test_df['단지코드'].unique():
    df=test_df[test_df['단지코드']==i]
    qual_columns=df['자격유형'].unique()
    sup_columns=df['공급유형'].unique()
    
    for z in qual_columns:
        test_df.loc[test_df['단지코드']==i,'자격유형_{}'.format(z)]=df[df['자격유형']==z]['전용면적별세대수'].sum()/df['전용면적별세대수'].sum()
    for z in sup_columns:
        test_df.loc[test_df['단지코드']==i,'공급유형_{}'.format(z)]=df[df['공급유형']==z]['전용면적별세대수'].sum()/df['전용면적별세대수'].sum()

#%%
test_df_2=test_df[['단지코드','자격유형_H', '자격유형_A', '자격유형_E', '자격유형_C', '자격유형_D',
       '자격유형_G', '자격유형_I', '자격유형_J', '자격유형_K', '자격유형_L', '자격유형_M', '자격유형_N',
       '공급유형_국민임대', '공급유형_영구임대', '공급유형_임대상가', '공급유형_공공임대(50년)',
       '공급유형_공공임대(10년)', '공급유형_공공임대(분납)', '공급유형_행복주택']].drop_duplicates()

#%%
train_agg=pd.merge(train_agg,train_df_2,on='단지코드')
test_agg=pd.merge(test_agg,test_df_2,on='단지코드')

train_agg=train_agg.fillna(0)
test_agg=test_agg.fillna(0)
#%%
train=train_agg.drop(['단지코드'],axis=1)
train=train_agg[['총세대수','지역', '공가수', '도보 10분거리 내 지하철역 수(환승노선 수 반영)', '도보 10분거리 내 버스정류장 수',
       '단지내주차면수', '단지별면적당보증금평균', '단지별면적당임대료평균', '자격유형_H', '자격유형_A', '자격유형_E',
       '자격유형_C', '자격유형_D', '자격유형_G', '자격유형_I', '자격유형_J', '자격유형_K', '자격유형_L',
       '자격유형_M', '자격유형_N', '공급유형_국민임대', '공급유형_영구임대', '공급유형_임대상가',
       '공급유형_공공임대(50년)', '공급유형_공공임대(10년)', '공급유형_공공임대(분납)', '공급유형_행복주택', '등록차량수']]

test=test_agg.drop(['단지코드'],axis=1)

train=train.drop(['단지별면적당보증금평균', '단지별면적당임대료평균'],axis=1)
test=test.drop(['단지별면적당보증금평균', '단지별면적당임대료평균'],axis=1)
#%%
import xgboost as xgb
from sklearn.metrics import mean_absolute_error


#%%



X, y = np.array(train.iloc[:,:-1]),np.array(train.iloc[:,-1])
data_dmatrix = xgb.DMatrix(data=X,label=y)



#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=120)

xg_reg = xgb.XGBRegressor(objective ='reg:linear',eval_metric='mae', colsample_bytree = 1, learning_rate = 0.5, max_depth = 4, alpha = 5, n_estimators = 9)
#
xg_reg.fit(X_train,y_train)

#preds = xg_reg.predict(np.array(test))
preds = xg_reg.predict(X_test)


mae = np.sqrt(mean_absolute_error(y_test, preds))
print("MAE: %f" % (mae))
#%%


xg_reg.fit(X,y)

preds = xg_reg.predict(np.array(test))

sub_df=test_agg[['단지코드']]
sub_df['Y']=preds
sub_df.columns=['code','num']
sub_df.to_csv('submission.csv',index=False)






# %%
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

xg_reg = xgb.XGBRegressor(objective ='reg:linear',eval_metric='mae', colsample_bytree = 1, learning_rate = 0.4, max_depth = 3,  n_estimators = 10)

avg=0
for i in range(20):
    scores = cross_val_score(xg_reg, X, y, cv=KFold(n_splits=5, shuffle=True), scoring='r2')

    avg+=scores.mean()
    

print(avg/20)

# %%
