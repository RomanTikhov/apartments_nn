import streamlit as st
import pickle
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

st.write("# Узнай актуальную цену своей квартиры в Нижнем Новгороде")

livrooms_count = st.slider('### Количество комнта (0, если квартира-студия):',
    min_value=0,
    max_value=8,
    value = 1,                           
    step=1)

district = st.selectbox("Район:",
    ['Советский', 'Приокский', 'Канавинский', 'Автозаводский',
       'Нижегородский', 'Московский', 'Сормовский', 'Новинский',
       'Ленинский'])

year = st.slider('### Год постройки:',
    min_value=1920,
    max_value=2027,
    value = 2010,
    step=1)

level = st.slider('### Этаж:',
    min_value=0,
    max_value=25,
    step=1)

house_levels = st.slider('### Всего этаже в доме:',
    min_value=1,
    max_value=28,
    step=1) 

const_tech = st.selectbox("Материал стен:",
    ['кирпич', 'блок+утеплитель', 'шлакоблок', 'панель',
       'монолитный железобетон', 'поризованный керамический блок',
       'стеновая панель на деревянном каркасе', 'дерево'])

area = st.slider('### Общая площадь квартиры:',
    min_value=12.0,
    max_value=200.0,
    #value=2000.0,
    step=0.5)

liv_area = st.slider('### Жилая площадь квартиры:',
    min_value=7.0,
    max_value=150.0,
    #value=2000.0,
    step=0.5)

kitchen_area = st.slider('### Площадь кухни / столовой:',
    min_value=0.0,
    max_value=60.0,
    #value=2000.0,
    step=0.5)

appart =dict(zip(
    ['livrooms_count', 'district', 'year', 'level', 'house_levels', 'const_tech', 'area', 'liv_area', 'kitchen_area'],
    [livrooms_count, district, year, level, house_levels, const_tech, area, liv_area, kitchen_area]))
new = pd.DataFrame(appart, index=[0])
st.write(new)

model_pkl_file = "models/apartment_prices_regression.pkl"
with open(model_pkl_file, 'rb') as file:
    model = pickle.load(file)

st.write(f"## Оценочная стоимость квартиры: {int(np.exp(model.predict(new))-1)} руб.")
