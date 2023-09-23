import matplotlib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import string
import plotly.express as px
st.set_option('deprecation.showPyplotGlobalUse', False)


# Define a function to remove punctuation from text
def remove_punctuation(text):
    if '%' in str(text):
        # Remove '%' and convert to float
        return float(str(text).replace('%', ''))
    else:
        exclude = set(string.punctuation) - set('.')
        return "".join([char for char in str(text) if char not in exclude])

# Load the data into a DataFrame
df = pd.read_csv('world-data-2023.csv')

df.fillna(0, inplace=True)  

st.title('Metro Mappers - An Urban Planning Toolkit')

selected_countries = st.multiselect('', df['Country'].unique())

if not selected_countries:
        st.warning("Please select at least one country.")

st.write(f"### Bar Chart of Population by Country")

fig, ax = plt.subplots()
for country in selected_countries:            
            # Filter the data for the selected country
            filtered_data = df[df['Country'] == country]

            # Create a bar chart
            ax.bar(filtered_data['Urban_population'], filtered_data['Country'])
ax.set_xlabel('Population')
ax.set_ylabel('Countries')
st.pyplot(fig)
st.write('\n')  # 


# top 10 countries 
# df['Country'] = str(df['Country'])
st.write("### Select a column to show the top 10 countries:")
selected_column = st.selectbox('', df.columns)

# Calculate the top 10 countries based on the selected column
top_10_countries = df.sort_values(by=selected_column, ascending=False).head(10)

# Display the top 10 countries in a table
st.write(f"Top 10 Countries by {selected_column}:")
# st.write(top_10_countries)

fig, ax = plt.subplots()
ax.bar(top_10_countries['Country'], top_10_countries[selected_column])
plt.xticks(rotation=90)
ax.set_xlabel('Country')
ax.set_ylabel(selected_column)
st.pyplot(fig)



#  Chloropleth map

st.write('Chloropleth Map to Display Population Density')

custom_color_scale = [
        [4000, 'rgb(255,255,178)'],
        [8000, 'rgb(254,204,92)'],
        [10000, 'rgb(253,141,60)'],
        [50000, 'rgb(240,59,32)'],
        [1, 'rgb(189,0,38)']
    ]



fig = px.choropleth(df, 
                        locations='Country',  # Column with country names
                        locationmode='country names',  # Use country names directly
                        color='Urban_population',  # The column with values to determine shading
                        hover_name='Country',  # Country names to display on hover
                        color_continuous_scale=custom_color_scale,  # Choose a color scale
                        projection="natural earth")  # Choose the map projection



fig.update_geos(
    showcoastlines=True,
    coastlinecolor="Black",
    showland=True,
    landcolor="LightGray",
    showocean=True,
    oceancolor="LightBlue",
)


st.plotly_chart(fig)


# Heatmap

heatmap_columns = [
    'Co2-Emissions',
    'CPI',
    'GDP',
    'Out of pocket health expenditure',
    'Population',
    'Total tax rate',
    'Land Area(Km2)',
]

countries_heatmap = [
    'China',
    'India',
    'United States',
    'Brazil',
    'Indonesia',
    'Japan',
    'Russia',
    'Nigeria',
    'Mexico',
    'Pakistan'
]

for column in heatmap_columns:
    # Exclude the target column
    df[column] = df[column].apply(lambda x: remove_punctuation(x))
    df[column] = pd.to_numeric(df[column], errors='coerce')


filtered_df = df[df['Country'].isin(countries_heatmap)]

# Filter the data based on selected columns
selected_data = df[heatmap_columns]

try:
    # Check if the selected data is empty or has no variation
    if selected_data.empty or selected_data.std().sum() == 0:
        st.warning("The selected columns have no data or no variation. Please choose different columns.")
        # return

    # Create a heatmap
    st.write("### Heatmap for Affecting Factors")
    corr = selected_data.corr()  # Calculate correlation matrix
    sns.set(style="white")
    mask = np.triu(corr)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', mask=mask)
    st.pyplot()

except Exception as e:
    st.error(f"An error occurred: {str(e)}")