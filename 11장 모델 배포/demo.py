# demo.py  
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="데모 애플리케이션",
    page_icon=":shark:",
    layout="wide"
)

df = pd.read_csv("../datasets/non_linear.csv")

st.header(body="Demo Application")
st.subheader(body="non_linear.csv")

x = st.sidebar.selectbox(label="X 축", options=df.columns, index=0)
y = st.sidebar.selectbox(label="Y 축", options=df.columns, index=1)

col1, col2 = st.columns(2)
with col1:
    st.dataframe(data=df, height=500, use_container_width=True)
with col2:
    st.line_chart(data=df, x=x, y=y, height=500)
