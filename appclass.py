import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import pickle

st.set_page_config(
    page_title="Credit Default App",
    page_icon="🔥",
    layout="wide"
    )

@st.cache_data
def load_data():
    data = pd.read_csv("prosper_data_app_dev.csv")
    return(data)

@st.cache_resource
def load_model():
    filename="finalized_default_model_class2026.sav"
    loaded_model = pickle.load(open(filename,"rb"))  
    return(loaded_model)


data = load_data()
model = load_model()


st.title("Sharky's Credit Default App")

st.markdown("🔥🔥🔥This is my first app! This is application is a stremalit dashborad that can be used to *analyze* and **predict** customer default")

st.header("Customer Explorer")

row1_col1, row1_col2, row1_col3 = st.columns([1,1,1])

rate = row1_col1.slider("Interest the Customer has to pay?",
                 data["borrower_rate"].min(),
                 data["borrower_rate"].max(),
                 (0.07,0.24) )


income = row1_col2.slider("Monthly Income of Customers",
                          data["monthly_income"].min(),
                          data["monthly_income"].max(),
                 (2000.0,30000.0))


mask = ~data.columns.isin(["employment_status","loan_default"])
names = data.loc[:,mask].columns

variable = row1_col3.selectbox("Select your variable to compare", names)

filtered_data = data.loc[(data["borrower_rate"] >= rate[0]) & 
                         (data["borrower_rate"] <= rate[1]) &
                         (data["monthly_income"] >= income[0]) & 
                         (data["monthly_income"] <= income[1]),]

if st.checkbox("Show Filtered Data", True):
    st.write(filtered_data)



row2_col1, row2_col2 = st.columns([1,1])


row2_col2.write("col2")


barplotdata = filtered_data[["loan_default",variable]].groupby("loan_default").mean()

fig1, ax1 = plt.subplots(figsize=((8, 3.7)))
ax1.bar(barplotdata.index.astype(str), barplotdata[variable], color="#fc8d62")
ax1.set_ylabel(variable)

row2_col1.pyplot(fig1)


st.header("Predicting Customer Default")

uploaded_data = st.file_uploader("choose a file for uplaod")

if uploaded_data is not None: 
    new_customers = pd.read_csv(uploaded_data)
    new_customers = pd.get_dummies(new_customers, drop_first=True)
    new_customers["predicted_default"] = model.predict(new_customers)
    
    st.download_button(label="Scored Customer Data",
                       data = new_customers.to_csv().encode("utf-8"),
                       file_name="scored_customer_data.csv")
    
    
    

