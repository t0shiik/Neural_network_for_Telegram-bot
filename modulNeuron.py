import numpy as np
import requests
from Head import HeadW1, HeadW2, Headb1, Headb2, HeadId
from Add import AddW1, AddW2, Addb1, Addb2, AddId
from Body import BodyW1, BodyW2, Bodyb1, Bodyb2, BodyId
from Leg import LegW1, LegW2, Legb1, Legb2, LegId
from Foot import FootW1, FootW2, Footb1, Footb2, FootId
from NeuronFun import getW, Appoint, Predict, get_pk_from_url

def RecNeuron(userid):
    # Формирование входных значений

    # Получение города пользователя
    PROFILE_URL = 'https://helperweathbot.pythonanywhere.com/api/profile/'
    BACKEND_TOKEN = 'e90095ef6add6a7db95d2c8000bc02a5a8c7305a'
    pk = get_pk_from_url(url=PROFILE_URL, token=BACKEND_TOKEN, user_id=userid)
    response = requests.get(f'{PROFILE_URL}{pk}/', headers={'Authorization': 'Token ' + BACKEND_TOKEN})
    city_name = response.json()['city']

    # Получение погодных условий
    x = getW(city_name) # Погодные условия

    # Получение пола пользователя
    gender = response.json()['gender']
    if gender == 'Женский':
        gender = 0
    else:
        gender = 1
    x = np.array(x)
    x = np.append(x, gender)

    # Формирование рекомендаций головного убора
    mind = Appoint(HeadW1, Headb1, HeadW2, Headb2)
    chance = Predict(x, mind[0], mind[1], mind[2], mind[3])
    PredClassHead = np.argmax(chance)

    # Формирование рекомендаций аксессуаров
    mind = Appoint(AddW1, Addb1, AddW2, Addb2)
    chance = Predict(x, mind[0], mind[1], mind[2], mind[3])
    PredClassAdd = np.argmax(chance)

    # Формирование рекомендаций одежды для торса
    mind = Appoint(BodyW1, Bodyb1, BodyW2, Bodyb2)
    chance = Predict(x, mind[0], mind[1], mind[2], mind[3])
    PredClassBody = np.argmax(chance)

    # Формирование рекомендаций одежды на ноги
    mind = Appoint(LegW1, Legb1, LegW2, Legb2)
    chance = Predict(x, mind[0], mind[1], mind[2], mind[3])
    PredClassLeg = np.argmax(chance)

    # Формирование рекомендаций обуви
    mind = Appoint(FootW1, Footb1, FootW2, Footb2)
    chance = Predict(x, mind[0], mind[1], mind[2], mind[3])
    PredClassFoot = np.argmax(chance)

    # Формирование общей рекомендации
    FullArr = []
    FullArr.append(HeadId[PredClassHead])
    FullArr.append(AddId[PredClassAdd])
    FullArr.append(BodyId[PredClassBody])
    FullArr.append(LegId[PredClassLeg])
    FullArr.append(FootId[PredClassFoot])
    ClothID = []
    for i in range(len(FullArr)):
        for j in range(len(FullArr[i])):
            if FullArr[i][j] != 0:
                ClothID.append(FullArr[i][j])

    # URL для достпуа к списку одежды
    BASE_URL = 'https://helperweathbot.pythonanywhere.com/api/clothes/'

    # Получаем список всей одежды
    response = requests.get(
        BASE_URL,
        headers={'Authorization': 'Token ' + BACKEND_TOKEN}
    )

    # хранит возвращаемое значение - массив названий одежды
    Clothes = []

    # Формирование выходного массива
    for i in range(len(ClothID)):
        for item in range(len(response.json())):
            if response.json()[item]['url'] == BASE_URL + str(ClothID[i]) + '/':
                Clothes.append(response.json()[item]['name'])
                print(response.json()[item]['name'])

    return Clothes