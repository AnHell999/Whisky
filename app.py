import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import  stopwords
from nltk.stem import PorterStemmer
import numpy as np
import altair as alt
from itertools import cycle
from st_aggrid import  GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances

nltk.download('punkt')
nltk.download('stopwords')

global df

#Tratamiento del dataset
url = 'https://raw.githubusercontent.com/AnHell999/Whisky/main/scotch_review.csv'
df = pd.read_csv(url)
df = df.drop(['currency', 'Unnamed: 0'], axis=1)
df = df.rename(columns={'review.point':'review'})
df[['review']].astype(int)
df['price'] = pd.to_numeric(df['price'])
df_copy  = df.copy(True) 


#Preparar los datagrid
return_mode_value = DataReturnMode.FILTERED
update_mode_value = GridUpdateMode.MODEL_CHANGED
fit_columns_on_grid_load = True               

################################################################
def procesarDf(df_proc):
  ps = PorterStemmer()
  procesedTxt = []

  for row in df_proc.itertuples(index=False):

    text = word_tokenize(row[4])
    stops = set(stopwords.words("english"))
    text = [ps.stem(w) for w in text if not w in stops and w.isalnum()]
    text = " ".join(text)

    procesedTxt.append(text)

  df_proc['procesed_desc'] = pd.Series(procesedTxt)
  return df_proc

################################
def getMatrix(df_proc):  
  BoWModel = TfidfVectorizer()
  BoWModel.fit(df_proc['procesed_desc'])
  TextsBow = BoWModel.transform(df_proc['procesed_desc'])

  return pairwise_distances(TextsBow,TextsBow,metric='cosine')
################################       

# DB
import sqlite3
conn = sqlite3.connect('data.db')
c = conn.cursor()

# Functions
def create_table():
    c.execute('CREATE TABLE IF NOT EXISTS blogtable(author TEXT,title TEXT,article TEXT,postdate DATE)')


def view_all_notes():
    c.execute('SELECT * FROM blogtable')
    data = c.fetchall()
    return data

def view_all_titles():
    c.execute('SELECT DISTINCT title FROM blogtable')
    data = c.fetchall()
    return data

def get_nombre_Whisky(nombre,d):  
    i = d.loc[:, 'name'] == nombre
    df_nmobre = d.loc[i]
    return df_nmobre

def get_categoria_Whisky(categoria,d):  
    i = d.loc[:, 'category'] == categoria
    df_categoria = d.loc[i]
    return df_categoria

def get_precio_Whisky(precio,d):  
    i = d.loc[:, 'price'] == precio
    df_precio = d.loc[i]
    return df_precio

def filtradoNumerico(numericCol, min, max, ascent, d):

  df_precio = d.loc[(d.loc[:,numericCol] >= min) & (d.loc[:, numericCol] <= max)]
  df_precio = df_precio.sort_values(numericCol, ascending=ascent)
  return df_precio
    
def getIndex(value,d):
  i = d.loc[:, 'name'] == value
  df_tmp = d.loc[i]
  i2 = df_tmp.index.tolist()
  return i2[0]

def filtradoTextual(value,d):  
  i = d.loc[:, 'category'] == value
  df_single_malt = d.loc[i]
  return df_single_malt

def rango(index):
  precio_base = df_copy.price[index]
  min = precio_base/2
  max = precio_base*2
  d = filtradoNumerico('price',min,max,True,df_copy)
  d = filtradoTextual(df_copy.iloc[index].category, d)
  d = d.reset_index(drop=True)
  return d

def recomeda(index):
  df_to_recomend = procesarDf(df_copy)
  df_to_recomend = rango(index)

  index2 = getIndex(df_copy.name.iloc[index], df_to_recomend)

  matrixDescription = getMatrix(df_to_recomend)

  distance = list(enumerate(matrixDescription[index2]))
  ordenados_por_resena = sorted(distance, key=lambda x: x[1])
  top_index = ordenados_por_resena[1:11]
  top = [i[0] for i in top_index]

  return df_to_recomend.iloc[top]

def addWhisky(new_row, dfadd):                                                         #la nueva fila viene en forma de array y se trata dentro de la funcion 
  return dfadd.append(new_row, ignore_index=True)                   

# Layout Templates
html_temp = """
<div style="background-color:#955624;padding:10px;border-radius:10px">
<h1 style="color:white;text-align:center;">WISHKY</h1>
</div>
"""
title_temp ="""
<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:10px;">
<h4 style="color:white;text-align:center;">{}</h1>
<img src="https://www.w3schools.com/howto/img_avatar.png" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;" >
<h6>Author:{}</h6>
<br/>
<br/> 
<p style="text-align:justify">{}</p>
</div>
"""
article_temp ="""
<div style="background-color:#464e5f;padding:10px;border-radius:5px;margin:10px;">
<h4 style="color:white;text-align:center;">{}</h1>
<h6>Author:{}</h6> 
<h6>Post Date: {}</h6>
<img src="https://www.w3schools.com/howto/img_avatar.png" alt="Avatar" style="vertical-align: middle;width: 50px;height: 50px;border-radius: 50%;" >
<br/>
<br/>
<p style="text-align:justify">{}</p>
</div>
"""
head_message_temp ="""
<div style="background-color:#464e5f;padding:10px;border-radius:5px;margin:10px;">
<h4 style="color:white;text-align:center;">{}</h1>
<img src="https://www.w3schools.com/howto/img_avatar.png" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;">
<h6>Author:{}</h6> 
<h6>Post Date: {}</h6> 
</div>
"""
full_message_temp ="""
<div style="background-color:silver;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
<p style="text-align:justify;color:black;padding:10px">{}</p>
</div>
"""

def main():

    st.markdown(html_temp.format('royalblue','white'),unsafe_allow_html=True)

    menu = ["Home","Recomendador de whisky","Nuestros Whiskys","Iniciar Sesión","Registrarse"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Iniciar Sesión":
        
        username = st.sidebar.text_input("Usuario")
        password = st.sidebar.text_input("Contraseña",type='password')

        if st.sidebar.checkbox("Iniciar Sesión"):

            create_usertable()
            hashed_pswd = make_hashes(password)

            result = login_user(username,check_hashes(password,hashed_pswd))
            if result:
                
                st.success("Hola {}".format(username))  

                menuUser = ["Recomendador de whiskys", "Mis whiskys"]
                choiceUser= st.sidebar.selectbox("Opciones de usuario registrado", menuUser)
                if choiceUser == "Recomendador de whiskys":
                    nombres = df['name'].unique()
                    st.write("")
                    nombre = st.selectbox('Introduce tu Whisky favorito\n', nombres)
  

                    index = getIndex(nombre,df)
                
                    recomendacion = recomeda(index)
                    recomendacionP=recomendacion[['name','price']]
                
                    st.caption("Tus 10 recomendaciones personalizadas:")

                    gb = GridOptionsBuilder.from_dataframe(recomendacionP)
                    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
                    gb.configure_selection('single')                    
                    gb.configure_grid_options(domLayout='normal')
                    gridOptions = gb.build()

                    grid_response = AgGrid(
                        recomendacionP, 
                        gridOptions=gridOptions,
                        height=315, 
                        width='100%',
                        data_return_mode=return_mode_value, 
                        update_mode=update_mode_value,
                        fit_columns_on_grid_load=fit_columns_on_grid_load,
                        allow_unsafe_jscode=True, 
                        )

                    recomendacionP = grid_response['data']
                    selected = grid_response['selected_rows']
                    selected_df = pd.DataFrame(selected)

                    with st.spinner("Displaying results..."):
                        chart_data = recomendacionP.loc[:,['name']].assign(source='total')

                        if not selected_df.empty:
                            selected_data = selected_df.loc[:,['name']].assign(source='selection')
                            chart_data = pd.concat([chart_data, selected_data])


                        st.subheader("Whisky seleccionado:")
                        st.write(grid_response['selected_rows'])

                    if st.button("Guardar Whisky"):
                        if selected_df.empty:
                            st.caption("No puedes almacenar si no has seleccionado ningún whisky") 
                        else:
                            add_whisky(selected_df['name'])
                    
                elif choiceUser == "Mis whiskys":  
                    mis_whiskys = viewWhiskys() 

                    gb = GridOptionsBuilder.from_dataframe(mis_whiskys)
                    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
                    gb.configure_selection('single')                    
                    gb.configure_grid_options(domLayout='normal')
                    gridOptions = gb.build()

                    grid_response = AgGrid(
                    mis_whiskys, 
                    gridOptions=gridOptions,
                    height=150, 
                    width='100%',
                    data_return_mode=return_mode_value, 
                    update_mode=update_mode_value,
                    fit_columns_on_grid_load=fit_columns_on_grid_load,
                    )

                    left, right = st.columns(2)
                    with left:
                        st.button("👍")
                    with right:
                        st.button("👎")

                    if st.text_input("comm"):
                        st.write("comment")
            else:
                st.warning("Incorrect Username/Password")

    elif choice == "Registrarse":
        st.subheader("Crear nueva cuenta")
        new_user = st.text_input("Usuario")
        new_password = st.text_input("Contraseña",type='password')

        if st.button("Registro"):
            create_usertable()
            add_userdata(new_user,make_hashes(new_password))
            st.success("Has creado exitosamente tu cuenta")
            st.info("Ir al menú de Inicio de Sesión para correctamente iniciar sesión")


    elif choice == "Home":

        st.subheader("Encuentra tu whisky ideal")
        df_muestra = df

        categorias = df['category'].unique()
        categoria = st.selectbox('Categoria', categorias)
        df_muestra = df[df['category'] == categoria]

        st.write("Precio")
        precios = df_muestra['price'].unique()
        precio_min = st.selectbox('Precio minimo', precios)
        precio_max = st.selectbox('Precio maximo', precios)
        if precio_min > precio_max:
            precio_min = precios.min()
            precio_max = precios.max()
        df_muestra = filtradoNumerico('price',precio_min,precio_max,True,df_muestra)

        ratings = df_muestra['review'].unique()
        rg_rev = st.slider('Rating',int(ratings.min()),int(ratings.max()),(80,85))                 
        df_muestra = filtradoNumerico('review',rg_rev[0],rg_rev[1],True,df_muestra)
        
        df_muestra  

          

    elif choice == "Recomendador de whisky":
        nombres = df['name'].unique()
        st.write("")
        nombre = st.selectbox('Tu Whisky favorito\n', nombres)
        df[df['name'] == nombre]

        index = getIndex(nombre,df)
    
        recomendacion = recomeda(index)
        st.dataframe(recomendacion[['name','description']])

    elif choice == "Nuestros Whiskys":

        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
        gb.configure_selection('single')                    
        gb.configure_grid_options(domLayout='normal')
        gb.configure_pagination()
        gridOptions = gb.build()

        grid_response = AgGrid(
        df, 
        gridOptions=gridOptions,
        height=600, 
        width='100%',
        data_return_mode=return_mode_value, 
        update_mode=update_mode_value,
        fit_columns_on_grid_load=fit_columns_on_grid_load,
        )

        recomendacionP = grid_response['data']
        selected = grid_response['selected_rows']
        selected_df = pd.DataFrame(selected)
 
        with st.spinner("Displaying results..."):
            chart_data = recomendacionP.loc[:,['name']].assign(source='total')

            if not selected_df.empty:
                selected_data = selected_df.loc[:,['name']].assign(source='selection')
                chart_data = pd.concat([chart_data, selected_data])

            st.subheader("Whisky seleccionado:")
            st.write(grid_response['selected_rows'])


import hashlib
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

# DB  Functions
def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT,whiskys TEXT UNIQUE)')

def dropusers():
    c.execute('DROP TABLE userstable')


def add_userdata(username,password):
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
    conn.commit()

def add_whisky(whisky):
    try:
        if whisky is not None:
            c.execute('INSERT INTO userstable(whiskys) VALUES (?)',(whisky))
            conn.commit()
    except:
        st.error('Ya has guardado este whisky')
    else:
        st.success('Whisky guardado')

def login_user(username,password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
    data = c.fetchall()
    return data

def view_all_users():
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data

def viewWhiskys():
    toret = pd.DataFrame()
    c.execute('SELECT whiskys FROM userstable')
    data= c.fetchall()
    for i in data:
        if str(i[0]) != 'None':
            index = getIndex(i[0], df)
            new_row = df.iloc[index]
            toret = addWhisky(new_row,toret) 
    return toret


if __name__ == '__main__':
    main()