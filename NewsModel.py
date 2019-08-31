import datetime
from typing import Any


class NewsModel:
    date: datetime.date
    summary: str
    category1_label: str
    category1_score: float
    category2_label: str
    category2_score: float
    keyword1: str
    kw1_sentiment_score: float
    kw1_relevance: float
    kw1_sadness: float
    kw1_joy: float
    kw1_fear: float
    kw1_disgust: float
    kw1_anger: float
    keyword2: str
    kw2_sentiment_score: float
    kw2_relevance: float
    kw2_sadness: float
    kw2_joy: float
    kw2_fear: float
    kw2_disgust: float
    kw2_anger: float
    keyword3: str
    kw3_sentiment_score: float
    kw3_relevance: float
    kw3_sadness: float
    kw3_joy: float
    kw3_fear: float
    kw3_disgust: float
    kw3_anger: float
    entity1_type: str
    entity1_text: str
    entity1_relevance: float
    entity2_type: str
    entity2_text: str
    entity2_relevance: float
    entity3_type: str
    entity3_text: str
    entity3_relevance: float

    def __init__(self, date: datetime, summary: str, category1_label: str, category1_score: float, category2_label: str,
                 category2_score: float, keyword1: str, kw1_sentiment_score: float, kw1_relevance: float,
                 kw1_sadness: float, kw1_joy: float, kw1_fear: float, kw1_disgust: float, kw1_anger: float,
                 keyword2: str, kw2_sentiment_score: float, kw2_relevance: float, kw2_sadness: float, kw2_joy: float,
                 kw2_fear: float, kw2_disgust: float, kw2_anger: float, keyword3: str, kw3_sentiment_score: float,
                 kw3_relevance: float, kw3_sadness: float, kw3_joy: float, kw3_fear: float, kw3_disgust: float,
                 kw3_anger: float, entity1_type: str, entity1_text: str, entity1_relevance: float, entity2_type: str,
                 entity2_text: str, entity2_relevance: float, entity3_type: str, entity3_text: str, entity3_relevance: float):
        self.date = date
        self.summary = summary
        self.category1_label = category1_label
        self.category1_score = category1_score
        self.category2_label = category2_label
        self.category2_score = category2_score
        self.keyword1 = keyword1
        self.kw1_sentiment_score = kw1_sentiment_score
        self.kw1_relevance = kw1_relevance
        self.kw1_sadness = kw1_sadness
        self.kw1_joy = kw1_joy
        self.kw1_fear = kw1_fear
        self.kw1_disgust = kw1_disgust
        self.kw1_anger = kw1_anger
        self.keyword2 = keyword2
        self.kw2_sentiment_score = kw2_sentiment_score
        self.kw2_relevance = kw2_relevance
        self.kw2_sadness = kw2_sadness
        self.kw2_joy = kw2_joy
        self.kw2_fear = kw2_fear
        self.kw2_disgust = kw2_disgust
        self.kw2_anger = kw2_anger
        self.keyword3 = keyword3
        self.kw3_sentiment_score = kw3_sentiment_score
        self.kw3_relevance = kw3_relevance
        self.kw3_sadness = kw3_sadness
        self.kw3_joy = kw3_joy
        self.kw3_fear = kw3_fear
        self.kw3_disgust = kw3_disgust
        self.kw3_anger = kw3_anger
        self.entity1_type = entity1_type
        self.entity1_text = entity1_text
        self.entity1_relevance = entity1_relevance
        self.entity2_type = entity2_type
        self.entity2_text = entity2_text
        self.entity2_relevance = entity2_relevance
        self.entity3_type = entity3_type
        self.entity3_text = entity3_text
        self.entity3_relevance = entity3_relevance

    def __str__(self) -> str:
        return ', '.join(['{key}={value}'.format(key=key, value=self.__dict__.get(key)) for key in self.__dict__])

    def __getattribute__(self, name: str) -> Any:
        return super().__getattribute__(name)


def missingValuesHandler(watsonCallResult, layer1: str, elementNum: int, layer2: str, layer3: str):
    maxArraySize = checkMaxArraySize(watsonCallResult, layer1)
    if elementNum > maxArraySize:
        return ""
    else:
        if layer3 != "":
            return watsonCallResult[layer1][elementNum][layer2][layer3]
        else:
            return watsonCallResult[layer1][elementNum][layer2]


def checkMaxArraySize(watsonCallResult, key: str):
    count = -1
    try:
        for x in range(100):
            watsonCallResult[key][x]
            count = count + 1
    except:
        return count
    return count
