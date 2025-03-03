import pandas as pd
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler, StandardScaler

train_data=pd.read_csv(r'D:\РАФ\2 курс\4сем\AI\data titan(example)\train.csv')
train_data[['Deck', 'Cabin_num', 'Side']] = train_data["Cabin"].str.split("/", expand=True)
train_data['Cabin_num'] = train_data['Cabin_num'].astype(float)

train_data = train_data.drop(columns=['Cabin', 'Name'])
missing_values = train_data.isnull().sum()

categorical_cols = ['HomePlanet','CryoSleep','Destination','VIP','Deck','Side']
num_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck','Cabin_num']

for column in categorical_cols:
    mode_value = train_data[column].mode()[0]
    train_data[column] = train_data[column].fillna(mode_value)

for column in num_columns:
    mean_value = train_data[column].mean()
    train_data[column] = train_data[column].fillna(mean_value)

after_missing_values=train_data.isnull().sum()
print(f'Missing values before fillings: {missing_values}')
print(f'Missing values after fillings:{after_missing_values}')

categorical_cols_no_bin=list(filter(lambda x:x!='CryoSleep' and x!='VIP',categorical_cols))

train_data['VIP'] = train_data['VIP'].astype(int)
train_data['Transported'] = train_data['Transported'].astype(int)
train_data['CryoSleep'] = train_data['CryoSleep'].astype(int)

scaler = StandardScaler()
train_data[num_columns] = scaler.fit_transform(train_data[num_columns])

train_data = pd.get_dummies(train_data, columns=categorical_cols_no_bin, drop_first=False)
train_data = train_data.replace({True: 1, False: 0})

train_data.to_csv("processed_titanic.csv",index=False)




