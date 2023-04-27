import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

model = pickle.load(open('knn.pkl', 'rb'))
dataset= pd.read_csv('Set-6.csv')


X = dataset.iloc[:, [4,7,8,9,12,13]].values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

def predict_note_authentication(UserID,Age,EstimatedSalary):
    output= model.predict(sc.transform([[Age,EstimatedSalary]]))
    print("Purchased", output)
    if output==[1]:
        prediction="Item will be purchased"
    else:
        prediction="Item will not be purchased"
    print(prediction)
    return prediction

def main():
    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center>
   </div>
   </div>
   </div>
"""
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Item Purchase Prediction using knn Algorithm")
    UserID = st.text_input("UserID","")
    Gender = st.selectbox(
    "Gender",
    ("Male", "Female", "Others")
    )
    
    Age = st.number_input('Insert a Age',18,60)
    #Age = st.text_input("Age","Type Here")
    EstimatedSalary = st.number_input("Insert EstimatedSalary",15000,150000)
    result=""
    
    if st.button("SVM Prediction"):
      result=predict_note_authentication(UserID,Age,EstimatedSalary)
      st.success('SVM Model has predicted {}'.format(result))
      
    if st.button("About"):
      st.header("Developed by Divy Mradia")
      st.subheader("Student , Department of Computer Engineering")
    html_temp = """
    <div class="" style="background-color:orange;" >
    <div class="clearfix">           
    <div class="col-md-12">
    <center><p style="font-size:20px;color:white;margin-top:10px;">Machine Learning Experiment : Support Vector Machine</p></center> 
    </div>
    </div>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
if __name__=='__main__':
  main()
      