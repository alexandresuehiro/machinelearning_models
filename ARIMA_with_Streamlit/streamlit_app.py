import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Define ARIMA prediction function for multiple columns
def arima_predictions_multivariate(data, order, steps):
    # Data preprocessing (example: differencing)
    data_diff = data.diff().dropna()
    
    model = ARIMA(data_diff, order=order)
    model_fit = model.fit()
    
    # Model diagnostics
    residuals = model_fit.resid
    
    # Plotting residuals for each column
    for col in residuals.columns:
        fig, ax = plt.subplots(1,2)
        residuals[col].plot(title=f"Residuals for {col}", ax=ax[0])
        residuals[col].plot(kind='kde', title='Density', ax=ax[1])
        st.pyplot(fig)
        st.write(sm.stats.acorr_ljungbox(residuals[col], lags=[10], return_df=True))
    
    predictions = model_fit.predict(start=len(data_diff), end=len(data_diff)+steps-1)
    
    return predictions

# Streamlit app
st.title("ARIMA Prediction Tool (Multivariate)")

# Data input
data_file = st.file_uploader("Upload CSV file", type="csv")
if data_file is not None:
    df = pd.read_csv(data_file, sep=';')

    # Column selection
    selected_columns = st.multiselect("Select columns for prediction", df.columns)
    if selected_columns:  # Check if any columns are selected
        # Order input
        order = st.text_input("Order (p,d,q)", "0,1,1")
        order = tuple(map(int, order.split(",")))

        # Steps input
        steps = st.number_input("Steps", 12)

        # Predict button
        if st.button("Predict"):
            predictions = arima_predictions_multivariate(df[selected_columns], order, steps)
            st.write(predictions)