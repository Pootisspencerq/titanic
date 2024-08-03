import pandas as pd

df = pd.read_csv("titanic.csv")

df.info()

df.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)

age1 = df[df["Pclass"] == 1]["Age"].median()
age2 = df[df["Pclass"] == 2]["Age"].median()
age3 = df[df["Pclass"] == 3]["Age"].median()

def fill_age(data):
    if pd.isnull(data["Age"]):
        if data["Pclass"] == 1:
            return age1
        elif data["Pclass"] == 2:
            return age2
        else:
            return age3
    else:
        return data["Age"]

df["Age"] = df.apply(fill_age, axis=1)
df['Embarked'].fillna('S', inplace=True)

def fill_sex(data):
    if data == 'male':
        return 1
    else:
        return 0

df["Sex"] = df["Sex"].apply(fill_sex)
df[list(pd.get_dummies(df["Embarked"]).columns)] = pd.get_dummies(df["Embarked"])
df.drop(["Embarked"], axis=1, inplace=True)

def is_alone(data):
    if data["SibSp"] + data['Parch'] == 0:
        return 1
    return 0

df["Alone"] = df.apply(is_alone, axis=1)

df.info()

df.to_csv("Cleaned_Titanic.csv", index=False)
