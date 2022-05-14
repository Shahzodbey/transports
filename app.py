import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

#title
st.title('Transportni guruhlaydigan model')

#model
model = load_learner('transport_model.pkl')

#file joylash
file = st.file_uploader('Rasm yuklash', type=['gif', 'png', 'jpg', 'jpeg', 'svg'])

if file:
    #PIL corvert
    img = PILImage.create(file)
    st.image(file)

    #prediction
    pred, pred_id, probs = model.predict(img)
    st.success(f"Natija: {pred}")
    st.info(f'Ehtimollik: {probs[pred_id]*100:.1f}%')

    #plotting
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)