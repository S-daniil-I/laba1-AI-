import pandas as pd
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler, StandardScaler

train_data=pd.read_csv(r'D:\РАФ\2 курс\4сем\AI\data titan\train.csv')
#test_data= pd.read_csv(r'D:\РАФ\2 курс\4сем\AI\data titan\test.csv')
missing_values = train_data.isnull().sum()
for column in train_data.columns:
    mode_value=train_data[column].mode()[0]
    train_data.fillna(mode_value,inplace=True)
after_missing_values=train_data.isnull().sum()

categorical_cols = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
num_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

train_data[categorical_cols] = train_data[categorical_cols].astype(str)  # Преобразуем все категории в строки
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_cat = encoder.fit_transform(train_data[categorical_cols])
encoded_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out())

train_data = train_data.drop(columns=categorical_cols)
train_data = pd.concat([train_data.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

scaler = MinMaxScaler()
train_data[num_columns] = scaler.fit_transform(train_data[num_columns])
print(train_data.head())
print(after_missing_values)