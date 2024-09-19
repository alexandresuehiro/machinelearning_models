import streamlit as st
import pandas as pd
import statsmodels
import matplotlib
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import statsmodels.api as sm


# Define ARIMA prediction function
def arima_predictions(data, order, steps):
    
    # Data preprocessing (example: differencing)
    data_diff = data.diff().dropna()
    
    model = ARIMA(data_diff, order=order)
    model_fit = model.fit()
    
    # Model diagnostics
    residuals = model_fit.resid
    fig, ax = plt.subplots(1,2)
    residuals.plot(title="Residuals", ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1])
    st.pyplot(fig)
    st.write(sm.stats.acorr_ljungbox(residuals, lags=[10], return_df=True))
    
    predictions = model_fit.predict(start=len(data_diff), end=len(data_diff)+steps-1)
    
    return predictions

# Streamlit app
st.title("ARIMA Prediction Tool")

# Data input
data_file = st.file_uploader("Upload CSV file", type="csv")
if data_file is not None:
    df = pd.read_csv(data_file, sep=";")
    
    # Column selection
    selected_column = st.selectbox("Select column for prediction", df.columns)

    # Order input
    order = st.text_input("Order (p,d,q)", "0,1,1")
    order = tuple(map(int, order.split(",")))
    
    # Steps input
    steps = st.number_input("Steps", 12)
    
    # Predict button
    if st.button("Predict"):
        predictions = arima_predictions(df[selected_column], order, steps)
        st.write(predictions)
