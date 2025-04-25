from flask import Flask, render_template, request
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
import pickle
import os

app = Flask(__name__)

# Load and preprocess dataset
def load_and_preprocess_data():
    df = pd.read_csv("data/crime_rates.csv")
    df.dropna(subset=['Date'], inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y %I:%M:%S %p", errors='coerce')
    df = df[df['Date'].notna()]
    df = df[df['Date'] >= '2010-01-01']
    df = df[df['Primary Type'].notna()]
    df['Month'] = df['Date'].dt.to_period('M').dt.to_timestamp()
    monthly_crime = df.groupby('Month').size().reset_index(name='CrimeCount')
    monthly_crime = monthly_crime.rename(columns={"Month": "ds", "CrimeCount": "y"})
    return monthly_crime

@app.route('/', methods=['GET', 'POST'])
def index():
    result_plot = None
    error_message = ""
    if request.method == 'POST':
        model_name = request.form['model']
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        model_path = os.path.join("models", model_name)
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        except:
            error_message = "Failed to load the selected model."
            return render_template('index.html', result=None, error=error_message)

        future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        future_df = pd.DataFrame({'ds': future_dates})

        try:
            forecast = model.predict(future_df)
            forecast_range = forecast[['ds', 'yhat']].copy()
            avg_prediction = round(forecast_range['yhat'].mean(), 2)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast_range['ds'], y=forecast_range['yhat'],
                                     mode='lines+markers', name='Prediction'))
            fig.update_layout(title='Crime Count Forecast', xaxis_title='Date', yaxis_title='Predicted Crimes')
            result_plot = pyo.plot(fig, include_plotlyjs=False, output_type='div')

            return render_template('result.html', plot=result_plot, avg=avg_prediction)

        except Exception as e:
            error_message = f"Prediction failed: {str(e)}"

    return render_template('index.html', result=None, error=error_message)

if __name__ == '__main__':
    app.run(debug=True)