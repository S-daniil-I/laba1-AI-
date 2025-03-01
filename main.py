import pandas as pd
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler, StandardScaler

train_data=pd.read_csv(r'D:\РАФ\2 курс\4сем\AI\data titan(example)\train.csv')
missing_values = train_data.isnull().sum()

categorical_cols = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP','Name']
num_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

for column in categorical_cols:
    mode_value = train_data[column].mode()[0]
    train_data[column] = train_data[column].fillna(mode_value)

for column in num_columns:
    mean_value = train_data[column].mean()
    train_data[column] = train_data[column].fillna(mean_value)
after_missing_values=train_data.isnull().sum()
print(f'Missing values before fillings: {missing_values}')
print(f'Missing values after fillings:{after_missing_values}')

scaler = StandardScaler()
train_data[num_columns] = scaler.fit_transform(train_data[num_columns])

train_data = pd.get_dummies(train_data, columns=categorical_cols, drop_first=True)
train_data.to_csv("processed_titanic.csv",index=False)




