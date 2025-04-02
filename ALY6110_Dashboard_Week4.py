import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = "ad_viz_plotval_data (1).csv"
df = pd.read_csv(file_path)

# Convert Date to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Rename key columns for convenience
df.rename(columns={
    'Daily Max 8-hour CO Concentration': 'CO_Concentration',
    'Daily AQI Value': 'AQI'
}, inplace=True)

# Standardize CO Concentration for clustering
scaler = StandardScaler()
df['CO_Concentration_Scaled'] = scaler.fit_transform(df[['CO_Concentration']])

# Perform clustering using K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(df[['CO_Concentration_Scaled']])

# Initialize Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Carbon Monoxide Analysis Dashboard"),
    
    dcc.DatePickerRange(
        id='date-picker',
        start_date=df['Date'].min(),
        end_date=df['Date'].max(),
        display_format='YYYY-MM-DD'
    ),
    
    html.Div([
        dcc.Graph(id='time-series', style={'width': '48%', 'display': 'inline-block'}),
        dcc.Graph(id='scatterplot', style={'width': '48%', 'display': 'inline-block'})
    ]),
    
    html.Div([
        dcc.Graph(id='clustering', style={'width': '48%', 'display': 'inline-block'}),
        dcc.Graph(id='boxplot', style={'width': '48%', 'display': 'inline-block'})
    ]),
    
    dcc.Graph(id='histogram', style={'width': '100%'})
])

@app.callback(
    [
        Output('time-series', 'figure'),
        Output('scatterplot', 'figure'),
        Output('clustering', 'figure'),
        Output('boxplot', 'figure'),
        Output('histogram', 'figure')
    ],
    [Input('date-picker', 'start_date'), Input('date-picker', 'end_date')]
)
def update_charts(start_date, end_date):
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    # Time-Series Line Chart
    fig1 = px.line(filtered_df, x='Date', y='CO_Concentration', title='CO Levels Over Time')
    
    # Scatterplot - CO Levels vs. AQI
    fig2 = px.scatter(filtered_df, x='CO_Concentration', y='AQI', title='CO Concentration vs AQI')
    
    # Clustering Analysis Chart
    fig3 = px.scatter(filtered_df, x='Date', y='CO_Concentration', color=filtered_df['Cluster'].astype(str),
                      title='CO Level Clusters Over Time')
    
    # Boxplot - Identifying Anomalies
    fig4 = px.box(filtered_df, y='CO_Concentration', title='Boxplot of CO Levels')
    
    # Histogram - Distribution of CO Levels
    fig5 = px.histogram(filtered_df, x='CO_Concentration', nbins=10, title='Distribution of CO Levels')
    
    return fig1, fig2, fig3, fig4, fig5

if __name__ == '__main__':
    app.run_server(debug=True)

