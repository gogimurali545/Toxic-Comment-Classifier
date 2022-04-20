import streamlit as st
import plotly.express as px
import pickle as plk
import pandas as pd
st.title("Toxic comment classification")

text = st.text_input("Enter your text", placeholder='Enter your text here')

with open ('model_pkl', 'rb') as f:
    lr = plk.load(f)

with open ('vectorizer_pkl', 'rb') as f:
    vectorize = plk.load(f)
toxic_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
[prediction] = lr.predict(vectorize.transform([text])).toarray()
df = pd.DataFrame({'label': toxic_labels, 'prediction': prediction})
fig = px.bar(df, x='label', y='prediction')
fig.update_layout(
    xaxis = dict(
    showgrid=True,
    ticks="outside",
    tickson="boundaries",
    ticklen=20
    ),
    yaxis = dict(
        ticks=""
    )
)
st.plotly_chart(fig)