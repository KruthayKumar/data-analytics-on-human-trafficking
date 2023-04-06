 import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import lime.lime_tabular
from sklearn.tree import DecisionTreeClassifier
from lime.lime_tabular import LimeTabularExplainer
from lime import lime_text




# Load the Boston housing dataset
data2 = pd.read_csv(r"C:\Users\chand\Downloads\table 2.txt")
data4 = pd.read_csv(r"C:\Users\chand\Downloads\table 3.txt")
data5 = pd.read_csv(r"C:\Users\chand\Downloads\table 4.txt")
data6 = pd.read_csv(r"C:\Users\chand\Downloads\table 5.txt")
data7 = pd.read_csv(r"C:\Users\chand\Downloads\table 6.txt")
data = pd.read_csv(r"C:\Users\chand\OneDrive\Desktop\table_1.txt")
df = pd.read_csv(r"C:\Users\chand\OneDrive\Desktop\project_csv.txt")




nav = st.sidebar.radio("Navigation",['Home','Descriptive','Diagnostic','Prediction','Prescriptive'])

if nav == 'Home':
    st.title("Data Analytics On Human Trafficking")
    st.write('Welcome to the site!!!')
    st.image("https://www.acamstoday.org/wp-content/uploads/2022/04/HANDS-REACHING-MM.jpg")
    st.write("Human Trafficking is a serious issue all over the world, and needs to be countered effectively. The major objective of the site is to provide the users with various recent trends related to Human Trafficking in India. Hope you look over the four types of analytical trends, and contribute to our dataset.")
    
if nav == 'Descriptive':
    st.header('Descriptive')
    st.subheader('Know what actually happened regarding with the Trafficking cases')
    st.image('https://www.springboard.com/blog/wp-content/uploads/2021/08/what-does-a-data-analyst-do-2022-career-guide.png')
    if st.checkbox(" Trend of Human Trafficking over the years"):
        plt.plot(data4.columns,data4.iloc[0])
        plt.xlabel('Year')
        plt.ylabel('Number Of Cases')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    if st.checkbox(" Total Cases By State/UTs"):
        data2['Total Crimes'] = data2[['Forced Labour','Sexual Exploitation for Prostitution','Other forms of sexual exploitation','Domestic Servitude','Forced marriage','Petty Crimes',                                                                  'Begging', 'Drug Peddling','Removal of Organs','Other reasons']].sum(axis=1)
        data2 = data2.drop(labels=[28,37,38], axis=0)
        fig = plt.figure(figsize=(18, 8))
        plt.xticks(rotation=90)
        plt.bar(data2['States/UTs'], data2['Total Crimes'])
        st.pyplot(plt)        
    if st.checkbox("Male and Female Proportion"):
        data = data.drop(labels=[28,37,38], axis=0)
        fig = go.Figure(data=[
        go.Bar(name='Male', x=data.States_UTs, y=data.Total_Male),
        go.Bar(name='Female', x= data.States_UTs, y= data.Total_Female)
        ])
        # Change the bar mode
        fig.update_layout(barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    if st.checkbox("Foreign National Cases"):
        x = ['Srilanka', 'Nepal', 'Bangladesh', 'Other']
        m = [38, 2, 0,0]
        f = [0, 6, 26,35]
        fig = plt.figure(figsize=(10, 8))
        plt.xticks(rotation=90)
        # plot bars in stack manner
        plt.bar(x, m, color='purple')
        plt.bar(x, f, bottom=m, color='y')
        plt.xlabel("Countries")
        plt.ylabel("Value")
        plt.legend(["Male", "Female"])
        plt.title("Country Wise Cases")
        plt.ylim(0,40)
        st.pyplot(plt)

   
    if st.checkbox("Cases based on Type of Trafficking"):
        k = st.selectbox(" Which chart do you prefer ?", ("Pie chart", "Bar chart"))
        if k == 'Pie chart':
            data2 = pd.read_csv(r"C:\Users\chand\Downloads\table 2.txt")
            data2['Total Crimes'] = data2[['Forced Labour','Sexual Exploitation for Prostitution','Other forms of sexual exploitation','Domestic Servitude','Forced marriage','Petty Crimes',                                                                  'Begging', 'Drug Peddling','Removal of Organs','Other reasons']].sum(axis=1)
            data3 = data2.drop(['Sl. No.'], axis=1)
            data3 = data3.drop(['States/UTs'], axis=1)
            data3 = data3.drop(['Total Persons'], axis= 1)
            data3 = data3.drop(['Total Crimes'], axis= 1)
            data3 = data3.drop(['Other forms of sexual exploitation'], axis= 1)
            data3 = data3.drop(['Drug Peddling'], axis= 1)
            fig = plt.figure(figsize=(18, 8))
            plt.xticks(rotation=90)
            mycolors = ['r','g','b', 'c', 'm','y','k', 'w', "hotpink"]
            labels= data3.columns
            plt.pie(data3.iloc[38], colors = mycolors)
            plt.legend(labels)
            st.pyplot(plt)
        if k == 'Bar chart':
            data2 = pd.read_csv(r"C:\Users\chand\Downloads\table 2.txt")
            data2['Total Crimes'] = data2[['Forced Labour','Sexual Exploitation for Prostitution','Other forms of sexual exploitation','Domestic Servitude','Forced marriage','Petty Crimes',                                                                  'Begging', 'Drug Peddling','Removal of Organs','Other reasons']].sum(axis=1)
            data3 = data2.drop(['Sl. No.'], axis=1)
            data3 = data3.drop(['States/UTs'], axis=1)
            data3 = data3.drop(['Total Persons'], axis= 1)
            data3 = data3.drop(['Total Crimes'], axis= 1)
            data3 = data3.drop(['Other forms of sexual exploitation'], axis= 1)
            data3 = data3.drop(['Drug Peddling'], axis= 1)
            fig = plt.figure(figsize=(18, 8))
            plt.xticks(rotation=90)
            plt.bar(data3.columns,data3.iloc[38],color=['black', 'red', 'green', 'blue', 'cyan','black','chocolate','peru','magenta'])
            plt.show()
            st.pyplot(plt)
        
if nav == 'Diagnostic':
    st.header('Diagnostic')
    st.subheader('Know why things happened like they did')
    st.image('https://softteco.com/media/June%202020/Vata-analytics.png')
    if st.checkbox('Why Only Some States have high trafficking cases?'):
        fig = go.Figure(data=[
        go.Bar(name='Literacy_rate', x=data5.State_UTs, y=data5.Literacy_rate, marker={'color': 'green'}),
        go.Bar(name='Performance_Score', x= data5.State_UTs, y= data5.Performance_Score, marker={'color': 'violet'}),
        go.Bar(name='Poverty_rate', x=data5.State_UTs, y=data5.Poverty_rate, marker={'color': 'black'}),
        go.Bar(name='crime_rate', x= data5.State_UTs, y= data5.crime_rate, marker={'color': 'red'})
        ])
        # Change the bar mode
        fig.update_layout(barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        st.write(" Ideally, performance score and literacy rate should be high, while poverty rate and crime rate should be low. This explains why Assam has more number of cases. ")
    if st.checkbox('Why Youth is most affected?'):
        plt.plot(data6.Age_Group,data6.Physical_suitability)
        plt.plot(data6.Age_Group,data6.Need_Of_Work)
        plt.xlabel("Age Group")
        plt.ylabel("Value")
        plt.title('multiple plots')
        st.pyplot(plt)
        st.write("Blue line - > Physical Suitability, Orange line - > Need Of Work")
        st.write(" Considering the Physical Suitability and Need Of Work ratio, youth is more at risk")
    if st.checkbox('Why female population is more at risk?'):
        fig = go.Figure(data=[
        go.Bar(name='Male', x=data7.Criteria, y=data7.Male, marker={'color': 'steelblue'}),
        go.Bar(name='Female', x= data7.Criteria, y= data7.Female, marker={'color': 'firebrick'})
        ])
        # Change the bar mode
        fig.update_layout(barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        st.write(" Considering the parameters, female population is more at risk")


df.GENDER.replace({'M':1,'F':0},inplace = True)
df.AGE.replace({'Youth':0,'Middle-aged':1,'Old-aged':2},inplace = True)
df.EDUC.replace({'N':0,'Y':1},inplace = True)
df.POVERTY.replace({'UNDER':0,'AVG':1,'ABOVE':2},inplace = True)
df.WORKING.replace({'N':0,'Y':1},inplace = True)
df.FRAUD.replace({'N':0,'Y':1},inplace = True)
df.OP.replace({'N':0,'Y':1},inplace = True)
x = np.array(df.iloc[:,[1,2,3,4,5,6,7,]])
x.reshape(-1,1)
y = np.array(df.iloc[:,[8]])
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=50)
dtree = DecisionTreeClassifier(max_depth=500, random_state=300)
dtree.fit(X_train,y_train)

if nav == 'Prediction':
    st.header('Prediction')
    st.image('https://elu.nl/wp-content/uploads/2020/12/expert-talk-data-science-data-analytics-machine-learning.jpg')
    st.subheader('Predict the Human Trafficking Chance')
    GENDER = st.selectbox("Gender of the person (= 1 if Male; 0 otherwise)", [0, 1])
    AGE = st.number_input("Age of the person", min_value=0, max_value=2, value=1,step=1)
    STATE = st.number_input("State/UTs ID", min_value=1, max_value=36, value=1,step=1)
    EDUC = st.selectbox("Received atleast seconday education", [0, 1])
    POVERTY_LINE = st.number_input(" Poverty line", min_value=0, max_value=2, value=0,step=1)
    WORKING = st.number_input("Does the person currently work sufficiently", min_value=0, max_value=1, value=1)
    FRAUD = st.number_input("Is the person receiving any information about better work and more pay in a shady manner", min_value=0, max_value=1, value=0,step=1)
    arr = np.array([[GENDER,AGE,STATE,EDUC,POVERTY_LINE,WORKING,FRAUD]])
    if st.checkbox('See the state/UTs ID'):
        data = pd.read_csv(r"C:\Users\chand\OneDrive\Desktop\table_1.txt")
        data = data.drop(labels=[28,37,38], axis=0)
        dt = data[['Sl.No','States_UTs']]
        st.write("Take Sl.No as State/UTs ID")
        st.table(dt.style.hide_index())
    d = dtree.predict(arr) 
    if st.button('Predict'):
       if d[0]== 0: 
           st.write(" You may not fall in the trap of Human Trafficking")
       if d[0]==1:
           st.write(" You may fall in the trap of Human Trafficking")
    
    st.header('Contribute to Our Dataset')
    Gender = st.number_input("Gender of the person (= 1 if Male; 0 otherwise)",min_value=0, max_value=1, value=1,step=1)
    Age = st.selectbox("Person's age", [0, 1,2])
    State = st.number_input("State/UTs ID", min_value=1, max_value=36, value=2,step=1)
    Educ= st.number_input("Received atleast seconday education",min_value=0, max_value=1, value=0,step=1)
    Poverty = st.number_input(" Poverty line", min_value=0, max_value=2, value=2,step=1)
    Working = st.number_input("Does the person currently work sufficiently", min_value=0, max_value=1, value=0)
    Fraud = st.number_input("Is the person receiving any information about better work and more pay in a shady manner", min_value=0, max_value=1, value=1,step=1)
    Op = st.selectbox("Is the person a victim", [0,1])
    if st.button('submit'):
        to_add = {"GENDER":[Gender],"AGE":[Age],"STATE":[State],"EDUC":[Educ],"POVERTY_LINE":[Poverty],"WORKING":[Working],"FRAUD":[Fraud],"OP":[Op]}
        to_add = pd.DataFrame(to_add)
        to_add.to_csv(r"C:\Users\chand\OneDrive\Desktop\project2_csv.txt", mode='a', header= False ,index=False)
        st.success("Submitted")

if nav == 'Prescriptive':
    st.header('Prescriptive Analytics')
    st.image('https://bigdataanalyticsnews.com/wp-content/uploads/2019/12/data-analyst.png')
    if st.button("View the main features"):
        x = 'EDUC'
        y = 'FRAUD'
        z = 'WORKING'
        w = 'POVERTY'
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.bar(x, -0.71, width=0.95 , color='r')
        ax.bar(y, 0.64, width=0.95, color='b')
        ax.bar(z, -0.66, width=0.95 , color='r')
        ax.bar(w, -0.61, width=0.95 , color='r')
        st.pyplot(fig)
        st.write(" As shown the key features are education, poverty status, working status, and fraud. Therefore, these need to be monitored effectively to reduce trafficking.")
    if st.checkbox("View explanation of the prediction"):
        GENDER = st.selectbox("Gender of the person (= 1 if Male; 0 otherwise)", [0, 1])
        AGE = st.number_input("Age of the person", min_value=0, max_value=2, value=1,step=1)
        STATE = st.number_input("State/UTs ID", min_value=1, max_value=36, value=1,step=1)
        EDUC = st.selectbox("Received atleast seconday education", [0, 1])
        POVERTY_LINE = st.number_input(" Poverty line", min_value=0, max_value=2, value=0,step=1)
        WORKING = st.number_input("Does the person currently work sufficiently", min_value=0, max_value=1, value=1)
        FRAUD = st.number_input("Is the person receiving any information about better work and more pay in a shady manner", min_value=0, max_value=1, value=0,step=1)
        arr = np.array([[GENDER,AGE,STATE,EDUC,POVERTY_LINE,WORKING,FRAUD]])
        if st.checkbox('See the state/UTs ID'):
            data = pd.read_csv(r"C:\Users\chand\OneDrive\Desktop\table_1.txt")
            data = data.drop(labels=[28,37,38], axis=0)
            dt = data[['Sl.No','States_UTs']]
            st.write("Take Sl.No as State/UTs ID")
            st.table(dt.style.hide_index())
        explainer = LimeTabularExplainer(X_train, feature_names=[ 'GENDER', 'AGE', 'STATE', 'EDUC', 'POVERTY', 'WORKING', 'FRAUD'], class_names=['OP'], mode='regression')
        arr = arr.ravel()
        exp = explainer.explain_instance(arr, dtree.predict,num_features=6)
        if st.button("see"):
            exp.as_pyplot_figure()
            plt.tight_layout()
            st.pyplot(plt)
            arr = np.array([[GENDER,AGE,STATE,EDUC,POVERTY_LINE,WORKING,FRAUD]])
            d = dtree.predict(arr) 
            if d[0]== 0: 
                st.write(" You may not fall in the trap of Human Trafficking")
            if d[0]==1:
                st.write(" You may fall in the trap of Human Trafficking")
            st.write(" Please improve areas related to green coloured features, if any. For example, if poverty is green it shows one needs to improve their standard of living")

