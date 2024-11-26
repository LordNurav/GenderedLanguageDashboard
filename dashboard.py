import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
from io import BytesIO
import base64

# Load preprocessed data
data = pd.read_csv('results.csv')

# Aggregate data by gender
gender_aggregated = data.groupby('gender').agg({
    'descriptor': 'count',
    'sentiment': ['mean', 'std']
}).reset_index()
gender_aggregated.columns = ['gender', 'descriptor_count', 'sentiment_mean', 'sentiment_std']

# Initialize Dash app
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Gendered Language Analysis", style={'text-align': 'center'}),
    html.Div([
        dcc.Dropdown(
            id='book-dropdown',
            options=[{'label': book, 'value': book} for book in data['book'].unique()],
            multi=True,
            value=data['book'].unique(),
            placeholder="Select books to include",
        ),
    ], style={'width': '80%', 'margin': 'auto'}),
    html.Div([
        dcc.Graph(id='sentiment-bar', style={'width': '48%', 'display': 'inline-block'}),
        dcc.Graph(id='sentiment-distribution', style={'width': '48%', 'display': 'inline-block'}),
    ]),
    html.Div(id='gender-wordcloud')  # Dynamic content for word clouds
])

# Filter data based on selected books
def filter_data(selected_books):
    return data[data['book'].isin(selected_books)]

# Bar chart showing average sentiment by gender
@app.callback(
    Output('sentiment-bar', 'figure'),
    Input('book-dropdown', 'value')
)
def update_sentiment_bar(selected_books):
    filtered_data = filter_data(selected_books)
    aggregated = filtered_data.groupby('gender').agg({
        'descriptor': 'count',
        'sentiment': ['mean', 'std']
    }).reset_index()
    aggregated.columns = ['gender', 'descriptor_count', 'sentiment_mean', 'sentiment_std']

    fig = px.bar(
        aggregated,
        x='gender',
        y='sentiment_mean',
        error_y='sentiment_std',
        color='gender',
        title="Average Sentiment by Gender",
        labels={'sentiment_mean': 'Mean Sentiment'},
    )
    fig.update_layout(yaxis=dict(title='Sentiment (Positive or Negative)'))
    return fig

# Histogram of sentiment distribution by gender
@app.callback(
    Output('sentiment-distribution', 'figure'),
    Input('book-dropdown', 'value')
)
def update_sentiment_distribution(selected_books):
    filtered_data = filter_data(selected_books)
    fig = px.histogram(
        filtered_data,
        x='sentiment',
        color='gender',
        nbins=20,
        title="Sentiment Distribution by Gender",
        labels={'sentiment': 'Sentiment Score'}
    )
    fig.update_layout(xaxis=dict(title='Sentiment Score'), yaxis=dict(title='Count'))
    return fig

# Word cloud of descriptors by gender
@app.callback(
    Output('gender-wordcloud', 'children'),
    Input('book-dropdown', 'value')
)
def update_gender_wordcloud(selected_books):
    filtered_data = filter_data(selected_books)
    genders = filtered_data['gender'].unique()

    # Helper function to create word clouds
    def generate_wordcloud(gender):
        descriptors = ' '.join(filtered_data[filtered_data['gender'] == gender]['descriptor'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(descriptors)
        img = BytesIO()
        wordcloud.to_image().save(img, format='PNG')
        img.seek(0)
        return base64.b64encode(img.read()).decode('utf-8')

    # Render word clouds for each gender
    return html.Div([
        html.Div([
            html.H4(f"{gender.capitalize()} Descriptors", style={'text-align': 'center'}),
            html.Img(
                src=f"data:image/png;base64,{generate_wordcloud(gender)}",
                style={'display': 'block', 'margin': 'auto', 'max-width': '100%'}
            )
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'})
        for gender in genders
    ])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
