import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import csv
import datetime
from NewsModel import NewsModel
import DataframeUtils as dfUtils
import WatsonApiUtils


def get_up_down_accuracy_dow_jones_ds():
    dataset = np.genfromtxt("D:/MachineLearningDS/capstone/dow-jones-v1.csv", delimiter=",")
    x = dataset[:,0:4]
    #print(x[1])
    y = dataset[:,5]
    #print(y[0])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=17)
    bernNB = BernoulliNB(binarize=True)
    bernNB.fit(x_train, y_train)
    #print (bernNB)
    y_expect = y_test
    y_pred = bernNB.predict(x_test)
    print(accuracy_score(y_expect, y_pred))


def join_ds_dow_jones_and_tesla():
    dfDowJones = dfUtils.getDataframe("D:/MachineLearningDS/capstone/dow-jones-v2.csv")
    dfDowJones['Date'] = dfUtils.convertColumnToDate(dfDowJones, 'Date')
    dfUtils.printDfTypeAndFirstRows(dfDowJones)
    print("DFDowJonesSize: ", dfDowJones.shape[0])

    dfTesla = dfUtils.getDataframe("D:/MachineLearningDS/capstone/tsla.us-processed.txt")
    dfTesla['Date'] = dfUtils.convertColumnToDate(dfTesla, 'Date',)
    dfUtils.printDfTypeAndFirstRows(dfTesla)
    print("DFTeslaSize: ", dfTesla.shape[0])

    # closeHigherThanOpenDowJones = {'CloseIsHigherDowJones': dfDowJones['Close'] > dfDowJones['Open'],
    #                                'CloseIsHigherTesla': dfTesla}
    attributesNewDF = {'Date': dfDowJones['Date'],
                       'Weekday': dfUtils.getWeekDay(dfDowJones),
        'closeIsHigherDowJones': dfUtils.closeIsHigher(dfDowJones),
            'closeIsHigherTesla': dfUtils.closeIsHigher(dfTesla),
                   'difInPercentDowJones': dfUtils.getOpenCloseDifPercent(dfDowJones),
                       'difInPercentTesla': dfUtils.getOpenCloseDifPercent(dfTesla)}

    combinedDf = pd.DataFrame(attributesNewDF)
    dfUtils.printDfTypeAndFirstRows(combinedDf)
    print("closeIsHigherDowJones: ", dfUtils.getTrueAmout(combinedDf, "closeIsHigherDowJones"))
    print("closeIsLowerDowJones: ", dfUtils.getFalseAmout(combinedDf, "closeIsHigherDowJones"))
    print("closeIsHigherTesla: ", dfUtils.getTrueAmout(combinedDf, "closeIsHigherTesla"))
    print("closeIsLowerTesla: ", dfUtils.getFalseAmout(combinedDf, "closeIsHigherTesla"))
    print("MaxDowJones: ", combinedDf['difInPercentDowJones'].max())
    print("MinDowJones: ", combinedDf['difInPercentDowJones'].min())
    print("MaxTesla: ", combinedDf['difInPercentTesla'].max())
    print("MinTesla: ", combinedDf['difInPercentTesla'].min())


def build_news_dataset(source_ds_path: str, destination_path: str):
    news_data_frame = dfUtils.getDataframe(source_ds_path)
    news_data_frame = news_data_frame.drop(columns=['Label'])
    news_data_frame['Date'] = dfUtils.convertColumnToDate(news_data_frame, 'Date')
    try:
        print("Starting api calls...")
        news_model_list = []
        for index, row in news_data_frame.iterrows():
            date = row['Date']
            for i in range(25):
                column = 'Top' + (i+1).__str__()
                print('Processing: ', row[column])
                cleaned_news = clean_news(row[column])
                news_model_list.append(jsonToNewsModel(cleaned_news, date, WatsonApiUtils.watsonNlpApiCall(row[column], True)))
        write_csv(news_model_list, destination_path)
        print("Csv writing task completed successful")
    except Exception as e:
        print("An error occurred during csv writing task: " + e)
        write_csv(news_model_list, 'D:/MachineLearningDS/capstone/news-dataset-nlp.csv')



def write_csv(data: [], file_path: str):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data[0].__dict__.keys())
        writer.writeheader()
        for obj in data:
            writer.writerow(obj.__dict__)
        csvfile.close()


def clean_news(news):
    cleanedNews = news
    if cleanedNews[0:2] == "b'":
        cleanedNews = cleanedNews[2:]
    elif cleanedNews[0:2] == "b\"":
        cleanedNews = cleanedNews[2:]
    if cleanedNews[-1] == "'":
        cleanedNews = cleanedNews[0:-2]
    elif cleanedNews[-1] == "\"":
        cleanedNews = cleanedNews[0:-1]
    return cleanedNews


def jsonToNewsModel(newsSummary: str, date: datetime.date, watsonCallResult):

    news_obj = NewsModel(
        date.strftime("%Y/%m/%d"),
        newsSummary,
        NewsModel.missingValuesHandler(watsonCallResult, 'categories', 0, 'label', ""),
        NewsModel.missingValuesHandler(watsonCallResult, 'categories', 0, 'score', ""),
        NewsModel.missingValuesHandler(watsonCallResult, 'categories', 1, 'label', ""),
        NewsModel.missingValuesHandler(watsonCallResult, 'categories', 1, 'score', ""),
        NewsModel.missingValuesHandler(watsonCallResult, 'keywords', 0, 'text', ""),
        NewsModel.missingValuesHandler(watsonCallResult, 'keywords', 0, 'sentiment', 'score'),
        NewsModel.missingValuesHandler(watsonCallResult, 'keywords', 0, 'relevance', ""),
        NewsModel.missingValuesHandler(watsonCallResult, 'keywords', 0, 'emotion', 'sadness'),
        NewsModel.missingValuesHandler(watsonCallResult, 'keywords', 0, 'emotion', 'joy'),
        NewsModel.missingValuesHandler(watsonCallResult, 'keywords', 0, 'emotion', 'fear'),
        NewsModel.missingValuesHandler(watsonCallResult, 'keywords', 0, 'emotion', 'disgust'),
        NewsModel.missingValuesHandler(watsonCallResult, 'keywords', 0, 'emotion', 'anger'),
        NewsModel.missingValuesHandler(watsonCallResult, 'keywords', 1, 'text', ""),
        NewsModel.missingValuesHandler(watsonCallResult, 'keywords', 1, 'sentiment', 'score'),
        NewsModel.missingValuesHandler(watsonCallResult, 'keywords', 1, 'relevance', ""),
        NewsModel.missingValuesHandler(watsonCallResult, 'keywords', 1, 'emotion', 'sadness'),
        NewsModel.missingValuesHandler(watsonCallResult, 'keywords', 1, 'emotion', 'joy'),
        NewsModel.missingValuesHandler(watsonCallResult, 'keywords', 1, 'emotion', 'fear'),
        NewsModel.missingValuesHandler(watsonCallResult, 'keywords', 1, 'emotion', 'disgust'),
        NewsModel.missingValuesHandler(watsonCallResult, 'keywords', 1, 'emotion', 'anger'),
        NewsModel.missingValuesHandler(watsonCallResult, 'keywords', 2, 'text', ""),
        NewsModel.missingValuesHandler(watsonCallResult, 'keywords', 2, 'sentiment', 'score'),
        NewsModel.missingValuesHandler(watsonCallResult, 'keywords', 2, 'relevance', ""),
        NewsModel.missingValuesHandler(watsonCallResult, 'keywords', 2, 'emotion', 'sadness'),
        NewsModel.missingValuesHandler(watsonCallResult, 'keywords', 2, 'emotion', 'joy'),
        NewsModel.missingValuesHandler(watsonCallResult, 'keywords', 2, 'emotion', 'fear'),
        NewsModel.missingValuesHandler(watsonCallResult, 'keywords', 2, 'emotion', 'disgust'),
        NewsModel.missingValuesHandler(watsonCallResult, 'keywords', 2, 'emotion', 'anger'),
        NewsModel.missingValuesHandler(watsonCallResult, 'entities', 0, 'type', ""),
        NewsModel.missingValuesHandler(watsonCallResult, 'entities', 0, 'text', ""),
        NewsModel.missingValuesHandler(watsonCallResult, 'entities', 0, 'relevance', ""),
        NewsModel.missingValuesHandler(watsonCallResult, 'entities', 1, 'type', ""),
        NewsModel.missingValuesHandler(watsonCallResult, 'entities', 1, 'text', ""),
        NewsModel.missingValuesHandler(watsonCallResult, 'entities', 1, 'relevance', ""),
        NewsModel.missingValuesHandler(watsonCallResult, 'entities', 2, 'type', ""),
        NewsModel.missingValuesHandler(watsonCallResult, 'entities', 2, 'text', ""),
        NewsModel.missingValuesHandler(watsonCallResult, 'entities', 2, 'relevance', "")
    )
    return news_obj


def main():
    #joinDsTesla()
    #watsonNlp()
    source_news_ds_path = 'D:/MachineLearningDS/capstone/news-dataset.csv'
    destination_path = 'D:/MachineLearningDS/capstone/news-dataset-after-nlp.csv'
    build_news_dataset(source_news_ds_path, destination_path)


main()
