from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions, CategoriesOptions

api_key_opt1 = 'cUhH2spQAP1ncZEIFjrdJMPWNfTFmzOTkrHQeZf-NYqq'
api_key_opt2 = 'LOI9h9XjAnOYwQlrnrYFXLEbsmcExn4DiD5Hqh9GvKX_'


def getWatsonServiceInstance(retry_number: int, api_key = api_key_opt1):
    try:
        return NaturalLanguageUnderstandingV1(
            version='2018-03-16',
            url='https://gateway.watsonplatform.net/natural-language-understanding/api',
            iam_apikey='cUhH2spQAP1ncZEIFjrdJMPWNfTFmzOTkrHQeZf-NYqq')
    except Exception as e:
        print(e)
        if retry_number == 0:
            print("Retrying using key 2")
            getWatsonServiceInstance(retry_number + 1, api_key_opt2)


def watsonNlpApiCall(txtOrUrl: str, isText: bool):
    if (isText):
        return getWatsonServiceInstance(0).analyze(
            text=txtOrUrl, language='en', features=Features(entities=EntitiesOptions(), categories=CategoriesOptions(),
                                                            keywords=KeywordsOptions(sentiment=True, emotion=True))
        ).get_result()
    else:
        return getWatsonServiceInstance(0).analyze(
            url=txtOrUrl, language='en', features=Features(entities=EntitiesOptions(), categories=CategoriesOptions(),
                                                           keywords=KeywordsOptions(sentiment=True, emotion=True))
        ).get_result()
