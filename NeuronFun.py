from http import HTTPStatus
import requests
import numpy as np

# API-токен для запроса погодных условий
WEATHER_API_TOKEN = '85b36611b136f2bb3ab28e34df64c7a9'

# функция для получения ID города в openweathermap.org
def getCityId(city_name):
    try:
        res = requests.get("http://api.openweathermap.org/data/2.5/find",
                           params={'q': city_name, 'type': 'like', 'units': 'metric', 'APPID': WEATHER_API_TOKEN})
        data = res.json()
        city_id = data['list'][0]['id']
        return city_id
    except Exception as e:
        return None

# функция для Формирования входных значений
def getW(city_name):
    city_id = getCityId(city_name + ",RU")
    cntCount = 3
    try:
        res = requests.get("http://api.openweathermap.org/data/2.5/forecast",
                           params={'id': city_id, 'cnt': cntCount, 'units': 'metric', 'lang': 'ru',
                                   'APPID': WEATHER_API_TOKEN})
        data = res.json()
        averageTemp = 0
        averageWind = 0
        isRain = False
        for i in range(0, cntCount, 1):
            averageTemp += data['list'][i]['main']['feels_like']
            averageWind += data['list'][i]['wind']['speed']
            if data['list'][i]['weather'][0]['main'] == "Rain":
                isRain = True
            condish = data['list'][i]['weather'][0]['main']
        averageTemp = round((averageTemp / cntCount), 3)
        averageWind = round((averageWind / cntCount), 3)
        if condish == "Thunderstorm" or condish == "Rain" or condish == "Drizzle":
            cond = 3
        else:
            if condish == "Mist" or condish == "Haze" or condish == "Smoke":
                cond = 5
            else:
                if condish == "Clear":
                    cond = 1
                else:
                    if condish == "Clouds":
                        cond = 2
                    else:
                        if condish == "Snow":
                            cond = 4
                        else:
                            cond = 6
        return averageTemp, averageWind, cond
    except Exception as e:
        return None

# Занесение необходимых матриц весов и векторов смещения
def Appoint(W1, b1, W2, b2):
    newW1 = W1
    newb1 = b1
    newW2 = W2
    newb2 = b2
    return newW1, newb1, newW2, newb2

# Функция активации
def Relu(t):
    return np.maximum(t, 0)

# Softmax
def Softmax(t):
    out = np.exp(t)
    return out / np.sum(out)

# Функция прямого прохождения через нейронную сеть
def Predict(x, W1, b1, W2, b2):
    t1 = x @ W1 + b1
    h1 = Relu(t1)
    t2 = h1 @ W2 + b2
    z = Softmax(t2)
    return z

# Для получения id пользователя
def get_pk_from_url(url, token, user_id):
    response = requests.get(
        url,
        headers={'Authorization': 'Token ' + token},
        params={'user_id': str(user_id)}
    )
    if response.status_code == HTTPStatus.NOT_FOUND:
        return None

    data = response.json()
    url = data[0]['url']
    url_data = url.split('/')
    pk = url_data[-2]
    return pk