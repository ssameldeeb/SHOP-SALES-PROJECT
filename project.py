import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor



df = pd.read_csv('SHOP-SALES.csv')

print(df.shape)
print(df.columns.values)
print(df.info())
print(df.head())
print(df.dtypes)

print(df["QUANTITY"].unique())

df = df.drop(["Date","HOUR","PRICE","QUANTITY","SALE CONSULTANT"], axis=1)
print(df.head())

print(df["GENDER"].value_counts())
df.loc[df["GENDER"] == "MEN","GENDER"] = 1
df.loc[df["GENDER"] == "WOMEN","GENDER"] = 2
df.loc[df["GENDER"] == "BOYS","GENDER"] = 3
df.loc[df["GENDER"] == "GIRLS","GENDER"] = 4
df.loc[df["GENDER"] == "Erkek","GENDER"] = 5
print(df.head())
sns.countplot(df["GENDER"])
plt.show()


plt.figure(figsize=(15,4))
sns.countplot(df["SIZE"])
plt.show()

print(df["COLOR"].value_counts().sort_values(ascending=False))

print(df["PRODUCT"].value_counts().sort_values(ascending=False))

La = LabelEncoder()

plt.figure(figsize=(15,4))
df["COLOR"] = La.fit_transform(df["COLOR"])
sns.countplot(df["COLOR"])
plt.show()


plt.figure(figsize=(15,4))
df["PRODUCT"] = La.fit_transform(df["PRODUCT"])
sns.countplot(df["PRODUCT"])
plt.show()


plt.figure(figsize=(15,4))
df["SIZE"] = La.fit_transform(df["SIZE"])
sns.countplot(df["SIZE"])
plt.show()

# df["VAT"] = df["VAT"].astype(int)

print(df.dtypes)
print(df.head())
print(df.isnull().sum())

x = df.drop("TOTAL",axis=1)
y = df["TOTAL"]
print(x.shape)
print(y.shape)

ss = StandardScaler()
x = ss.fit_transform(x)
print(x[:5])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=44, shuffle =True)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

Ln = LinearRegression()
Ln.fit(X_train, y_train)

print("_"*100)
print(Ln.score(X_train, y_train))
print(Ln.score(X_test, y_test))


Rf = RandomForestRegressor(n_estimators=10,max_depth=15, random_state=33)
Rf.fit(X_train,y_train)

print("_"*100)
print(Rf.score(X_train, y_train))
print(Rf.score(X_test, y_test))

y_pred = Rf.predict(X_test)
print(y_pred[:5])

result = pd.DataFrame({"y_test": y_test, "y_pred": y_pred})
# result.to_csv("result.csv",index=False)