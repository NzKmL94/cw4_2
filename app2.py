# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
from datetime import datetime
startTime = datetime.now()
# import znanych nam bibliotek

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

filename = "model.sv"
model = pickle.load(open(filename,'rb'))
# otwieramy wcześniej wytrenowany model
zdrowy_d = {0:"Nie", 1:"Tak"}
pclass_d = {0:"Pierwsza",1:"Druga", 2:"Trzecia"}
embarked_d = {0:"Cherbourg", 1:"Queenstown", 2:"Southampton"}
# o ile wcześniej kodowaliśmy nasze zmienne, to teraz wprowadzamy etykiety z ich nazewnictwem

def main():

	st.set_page_config(page_title="Twoje Zdrowie")
	overview = st.container()
	left, right = st.columns(2)
	prediction = st.container()

	st.image("https://img.freepik.com/premium-wektory/budynek-szpitala-w-stylu-plaski_768258-359.jpg?w=900")

	with overview:
		st.title("Twoje Zdrowie")

	with left:
		leki = st.slider("Leki", min_value=0, max_value=4)
		wiek = st.slider("Wiek", value=1, min_value=11, max_value=77)
		wzrost = st.slider("Wzrost", min_value=159, max_value=200)

	with right:
		objawy = st.slider("Objawy", min_value=1, max_value=5)
		chorwsp = st.slider("Chroby współistniejące", min_value=0, max_value=5, step=1)


	data = [[ objawy, wiek, chorwsp, wzrost, leki]]
	survival = model.predict(data)
	s_confidence = model.predict_proba(data)

	with prediction:
		st.subheader("Czy taka osoba jest zdrowa?")
		st.subheader(("Tak" if survival[0] == 1 else "Nie"))
		st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))

if __name__ == "__main__":
    main()
