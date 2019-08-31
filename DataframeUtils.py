import pandas as pd

def printDfTypeAndFirstRows(df: pd.DataFrame):
    print(df[0:5])
    print(df.dtypes)

def getDataframe(path: str):
    return pd.DataFrame(pd.read_csv(path))

def convertColumnToDate(df: pd.DataFrame, columnName: str):
    return pd.to_datetime(df[columnName])

def getTrueAmout(df: pd.DataFrame, columnName: str):
    times = len(df[df[columnName].eq(True)])
    return (times, (times * 100 / df[columnName].count()))

def getFalseAmout(df: pd.DataFrame, columnName: str):
    times = len(df[df[columnName].eq(False)])
    return (times, (times * 100 / df[columnName].count()))

def getOpenCloseDifPercent(df: pd.DataFrame):
    return (df["Close"] - df["Open"])*100/df["Open"]

def getWeekDay(df: pd.DataFrame):
    return df["Date"].dt.day_name()

def closeIsHigher(df: pd.DataFrame):
    return df['Close'] > df['Open']