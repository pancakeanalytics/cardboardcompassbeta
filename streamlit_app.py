import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Set the page configuration
st.set_page_config(
    page_title="CardBoard Compass",
    page_icon=":compass:",  # You can change the icon if you prefer
    layout="centered",
)



# Set the URL for the dataset and image
data_url = 'https://pancakebreakfaststats.com/wp-content/uploads/2024/08/data_file.xlsx'
image_url = 'https://pancakebreakfaststats.com/wp-content/uploads/2024/08/017_logo.png'

# Load and clean the data
@st.cache
def load_and_clean_data(url):
    data = pd.read_excel(url)
    data = clean_data(data)
    # Exclude Lorcana category
    data = data[data['Category'] != 'Lorcana']
    return data

# Function to clean the data
def clean_data(data):
    # Convert Month to numeric format
    data['Month'] = pd.to_datetime(data['Month'], format='%B').dt.month
    
    # Combine Year and Month into a datetime column
    data['Date'] = pd.to_datetime(data[['Year', 'Month']].assign(DAY=1))
    
    # Sort the data by Date
    data = data.sort_values(by='Date').reset_index(drop=True)
    
    return data

# Function to perform Holt-Winters forecast
def holt_winters_forecast(data, periods=12):
    model = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=12)
    fit = model.fit()
    forecast = fit.forecast(periods)
    
    # Generate a datetime index for the forecast
    last_date = data.index[-1]
    forecast_index = pd.date_range(start=last_date, periods=periods + 1, freq='M')[1:]
    forecast.index = forecast_index
    
    # Calculate confidence intervals manually
    forecast_std = np.std(fit.resid)
    conf_int = pd.DataFrame({
        'lower': forecast - 1.96 * forecast_std,
        'upper': forecast + 1.96 * forecast_std
    }, index=forecast_index)
    
    return forecast, conf_int

# Function to calculate MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

# Function to bucket MACD trends
def bucket_macd_trends(macd_diff):
    conditions = [
        (macd_diff > 0.02),  # High upward trend
        (macd_diff > 0.005), # Medium upward trend
        (macd_diff > -0.005),# Low upward trend
        (macd_diff > -0.02), # Low downward trend
        (macd_diff <= -0.02) # High downward trend
    ]
    choices = ['High Upward', 'Medium Upward', 'Low Upward', 'Low Downward', 'High Downward']
    return np.select(conditions, choices, default='Neutral')

# Function to calculate the best time to buy cards
def calculate_best_buy_time(data):
    monthly_avg = data.groupby(data['Date'].dt.month).agg({'market_value': 'mean'})
    best_month = monthly_avg['market_value'].idxmin()
    return monthly_avg, best_month

# Function to run all analyses for a given category
def run_analysis(category_data, category_name):
    # Aggregate data by month
    monthly_data = category_data.groupby(['Date']).agg({'market_value': 'sum'}).reset_index()
    time_series = monthly_data.set_index('Date')['market_value']
    
    # Perform Holt-Winters forecast
    forecast, conf_int = holt_winters_forecast(time_series)
    
    # Calculate MACD and signal line
    macd, signal = calculate_macd(time_series)
    macd_diff = macd - signal
    trend_buckets = bucket_macd_trends(macd_diff)
    
    # Calculate the best time to buy cards
    monthly_avg, best_month = calculate_best_buy_time(monthly_data)
    
    return {
        'time_series': time_series,
        'forecast': forecast,
        'conf_int': conf_int,
        'macd': macd,
        'signal': signal,
        'trend_buckets': trend_buckets,
        'monthly_avg': monthly_avg,
        'best_month': best_month
    }

# Load and clean the data
data = load_and_clean_data(data_url)

# Streamlit app layout
# Display the image
st.image(image_url, use_column_width=True)

st.title("CardBoard Compass: Expand Your Collecting Knowledge Through Data")

# Filter options
st.sidebar.header("Filter Options")
categories = ['Fortnite', 'Marvel', 'Pokemon', 'Star Wars', 'Magic the Gathering', 'Baseball', 'Basketball', 'Football', 'Hockey', 'Soccer']
selected_category = st.sidebar.selectbox("Select Category", categories)

# Add option to compare two categories
compare = st.sidebar.checkbox("Compare Two Categories")
if compare:
    selected_category_2 = st.sidebar.selectbox("Select Second Category", categories)

# Filter the data based on user selection
filtered_data = data[data['Category'] == selected_category]

# Explanation of CardBoard Compass
st.write("Welcome to CardBoard Compass, your guide to navigating the world of trading cards with data and analytics. While analytics are not necessary for collecting, we offer these insights to help collectors who want to deepen their understanding of the market and make informed decisions.")
st.write("All data in this analysis is sourced from eBay, which offers the broadest range of card conditions and types of collectors. eBay’s extensive marketplace ensures that we capture a wide variety of market trends, from high-end graded cards to raw, ungraded cards.")
st.write("The market value used in our analysis is carefully weighted based on the total number of sellers per month, which allows for a more accurate comparison between different trading card categories. This weighting helps to control for outliers and noise in the data, ensuring that the insights provided are reflective of true market trends rather than skewed by abnormal sales or infrequent transactions.")
st.write("By understanding these trends, you can gain an edge in your collecting journey, making informed decisions about when to buy and sell cards based on a more stable and reliable dataset.")




# Analysis explanation
st.subheader(f"Overview of the Analysis for {selected_category}")
st.write(f"In this analysis, we will explore the market value trends for the {selected_category} category. The following steps will be taken:")
st.write("- **Holt-Winters Forecast**: A forecasting technique that considers seasonal trends in the data, projecting the market value over the next 12 months. The forecast will be displayed along with confidence intervals to show the potential range of future values.")
st.write("- **MACD Analysis**: The Moving Average Convergence Divergence (MACD) will be analyzed to identify potential upward or downward trends in the market. We will classify these trends as high, medium, or low, and visualize them on a chart.")
st.write("- **Best Time to Buy**: Based on historical data, we will determine the best time of the year to buy cards in this category, helping you make informed decisions on future purchases.")






# Run analysis for the first category
analysis_results = run_analysis(filtered_data, selected_category)

# Display results for the first category
st.subheader(f"Analysis Results for {selected_category}")

# Time series plot with forecast and confidence intervals
fig, ax = plt.subplots()
analysis_results['time_series'].plot(ax=ax, label='Observed')
analysis_results['forecast'].plot(ax=ax, label='Forecast')
ax.fill_between(analysis_results['forecast'].index, analysis_results['conf_int']['lower'], analysis_results['conf_int']['upper'], color='gray', alpha=0.2)
ax.set_title(f"12-Month Forecast for {selected_category}")
ax.set_ylabel('Market Value')
ax.legend()
st.pyplot(fig)

# Display forecast results in a table
st.subheader("Forecast Results")
forecast_table = pd.DataFrame({
    'Month': analysis_results['forecast'].index.strftime('%Y-%m'),
    'Forecast': analysis_results['forecast'],
    'Lower Bound': analysis_results['conf_int']['lower'],
    'Upper Bound': analysis_results['conf_int']['upper']
})
st.write(forecast_table)

# Display statement on the projected percentage change
initial_value = analysis_results['time_series'].iloc[-1]
final_value = analysis_results['forecast'].iloc[-1]
percentage_change = ((final_value - initial_value) / initial_value) * 100
forecast_end_date = analysis_results['forecast'].index[-1].strftime('%B %Y')

st.subheader("Projected Percentage Change")

# Enhanced explanation
if percentage_change < 0:
    st.write(f"The projected market value change for the {selected_category} category over the next 12 months is {percentage_change:.2f}%.")
    st.write(f"This negative trend suggests a potential decline in market values, which could present a good buying opportunity for collectors looking to acquire cards in the {selected_category} category at lower prices. The forecast reaches its end point in {forecast_end_date}, providing a strategic window for making purchases.")
elif percentage_change > 0:
    st.write(f"The projected market value change for the {selected_category} category over the next 12 months is {percentage_change:.2f}%.")
    st.write(f"This positive trend indicates that market values for {selected_category} cards are expected to rise. Collectors may want to act quickly to purchase cards before prices increase further. The forecasted increase culminates in {forecast_end_date}, so timing your acquisitions accordingly could be beneficial.")
else:
    st.write(f"The projected market value change for the {selected_category} category over the next 12 months is 0.00%.")
    st.write(f"This suggests stability in the market value for {selected_category} cards. Collectors may see this as a neutral period, where prices are not expected to fluctuate significantly, allowing for steady purchases throughout the year. The forecast remains stable through {forecast_end_date}.")

# MACD plot with color shading for trend buckets
st.subheader(f"MACD Analysis for {selected_category}")
fig, ax = plt.subplots()
analysis_results['macd'].plot(ax=ax, label='MACD')
analysis_results['signal'].plot(ax=ax, label='Signal')

# Loop through MACD trends and add shading based on trend buckets
for i in range(len(analysis_results['trend_buckets']) - 1):  # Loop through to the second-to-last element
    if analysis_results['trend_buckets'][i] == 'High Upward':
        ax.axvspan(analysis_results['macd'].index[i], analysis_results['macd'].index[i+1], color='green', alpha=0.3)
    elif analysis_results['trend_buckets'][i] == 'Medium Upward':
        ax.axvspan(analysis_results['macd'].index[i], analysis_results['macd'].index[i+1], color='lightgreen', alpha=0.3)
    elif analysis_results['trend_buckets'][i] == 'Low Upward':
        ax.axvspan(analysis_results['macd'].index[i], analysis_results['macd'].index[i+1], color='yellow', alpha=0.3)
    elif analysis_results['trend_buckets'][i] == 'Low Downward':
        ax.axvspan(analysis_results['macd'].index[i], analysis_results['macd'].index[i+1], color='orange', alpha=0.3)
    elif analysis_results['trend_buckets'][i] == 'High Downward':
        ax.axvspan(analysis_results['macd'].index[i], analysis_results['macd'].index[i+1], color='red', alpha=0.3)

ax.set_title(f"MACD and Signal Line for {selected_category}")
ax.set_ylabel('Value')
ax.legend()
st.pyplot(fig)

# Display statement on the most recent trend bucket
recent_trend = analysis_results['trend_buckets'][-1]

st.subheader("Most Recent MACD Trend")

# Enhanced explanation
if recent_trend == 'High Upward':
    st.write(f"The most recent MACD trend for the {selected_category} category indicates a **High Upward** trend. This suggests that the market value for {selected_category} cards is experiencing strong momentum and is likely to continue rising in the near term.")
    st.write(f"For collectors, this could be an opportune moment to hold onto your cards or consider selling them to capitalize on the increasing values. If you're looking to buy, you may want to act quickly before prices climb further.")
elif recent_trend == 'Medium Upward':
    st.write(f"The most recent MACD trend for the {selected_category} category shows a **Medium Upward** trend. This indicates moderate positive momentum in the market, with prices likely to increase gradually.")
    st.write(f"Collectors might find this a suitable time to purchase {selected_category} cards before prices rise significantly. It's also a good moment to hold if you already own cards in this category, as the upward trend may continue.")
elif recent_trend == 'Low Upward':
    st.write(f"The most recent MACD trend for the {selected_category} category suggests a **Low Upward** trend. The market is experiencing slight positive momentum, with prices edging upward.")
    st.write(f"This could be a favorable time for cautious buying, as the market shows signs of growth. Collectors holding {selected_category} cards might choose to wait and see if the trend strengthens before selling.")
elif recent_trend == 'Low Downward':
    st.write(f"The most recent MACD trend for the {selected_category} category indicates a **Low Downward** trend. Market values are slightly declining, signaling a potential slowdown.")
    st.write(f"Collectors may want to monitor the market closely during this period. It could be a time to hold off on buying until the trend stabilizes or shows signs of reversal. If you own cards in this category, consider holding rather than selling during this mild downturn.")
elif recent_trend == 'Medium Downward':
    st.write(f"The most recent MACD trend for the {selected_category} category shows a **Medium Downward** trend. Prices are moderately declining, reflecting a more noticeable decrease in market value.")
    st.write(f"This might be a good time to avoid purchasing {selected_category} cards, as prices could continue to drop. If you're a collector holding these cards, it may be wise to hold onto them until the market shows signs of recovery.")
elif recent_trend == 'High Downward':
    st.write(f"The most recent MACD trend for the {selected_category} category indicates a **High Downward** trend. This suggests significant downward momentum, with market values potentially falling sharply.")
    st.write(f"For collectors, this may not be the best time to buy, as prices are likely to continue decreasing. If you own {selected_category} cards, consider holding onto them or waiting for the market to stabilize before making any selling decisions.")
else:
    st.write(f"The most recent MACD trend for the {selected_category} category is **Neutral**. This indicates a lack of strong momentum in either direction, with market values remaining relatively stable.")
    st.write(f"Collectors might view this as a period of market equilibrium, where buying, holding, or selling decisions can be made without the pressure of significant market shifts. It's a good time to make strategic decisions based on individual collecting goals.")

# Explanation of MACD vs. Long-Term Forecast
st.write("It's important to note that the MACD trend speaks to the short-term momentum in the market, rather than the long-term forecast. For instance, while the MACD might indicate a high upward trend, signaling short-term gains, the long-term forecast could predict a significant dip in values. Collectors should consider both the short-term MACD trends and the long-term forecasts when making buying or selling decisions.")

# Best time to buy bar chart
st.subheader("Best Time to Buy Cards")
fig, ax = plt.subplots()
analysis_results['monthly_avg'].plot(kind='bar', ax=ax, legend=False)
ax.set_title(f"Average Market Value by Month for {selected_category}")
ax.set_xlabel('Month')
ax.set_ylabel('Average Market Value')
st.pyplot(fig)

# Enhanced explanation for the best time to buy
best_month_name = pd.to_datetime(f'{analysis_results["best_month"]}', format='%m').strftime('%B')
st.subheader(f"Best Time to Buy {selected_category} Cards")
st.write(f"The analysis of historical market values suggests that the best time to buy cards in the {selected_category} category is in {best_month_name}. During this month, the market tends to experience lower average values, providing an ideal opportunity for collectors to make purchases at more favorable prices.")
st.write(f"This trend is likely driven by seasonal factors, such as decreased demand or increased supply during {best_month_name}, which results in a temporary dip in prices. For collectors looking to expand their collection, this month offers a strategic advantage to acquire {selected_category} cards before prices potentially rise again.")
st.write(f"However, it's important to consider the broader market context and any upcoming events or releases that could impact the market. While {best_month_name} historically presents lower prices, staying informed about market conditions will help you make the most of this buying opportunity.")

# Final Read Out for the selected trading card category
st.subheader(f"Final Read Out for {selected_category} Cards")

# Integrating the results from the forecast, MACD, and best time to buy
if percentage_change < 0:
    st.write(f"**Forecast Insight:** The 12-month forecast for {selected_category} cards indicates a projected market value decrease of {percentage_change:.2f}%, with the trend continuing until {forecast_end_date}. This suggests a potential drop in prices, which could be advantageous for buyers looking to acquire cards at lower prices.")
else:
    st.write(f"**Forecast Insight:** The 12-month forecast for {selected_category} cards projects an increase in market value by {percentage_change:.2f}%, with the trend continuing until {forecast_end_date}. Collectors should consider purchasing now before prices rise further.")

st.write(f"**MACD Trend:** The most recent MACD trend for {selected_category} shows a **{recent_trend}** trend. This reflects the current market momentum and suggests that the {selected_category} card market is {('gaining' if 'Upward' in recent_trend else 'losing')} momentum. Collectors should consider this when deciding whether to buy, hold, or sell.")

st.write(f"**Best Time to Buy:** Historical data indicates that {best_month_name} is the best time to purchase {selected_category} cards, as this month typically sees the lowest average market value. If you're planning to add to your collection, {best_month_name} could be an optimal time to do so.")

# Final Summary and Collector's Strategy
st.write(f"**Collector's Strategy:** Given the forecasted {('increase' if percentage_change > 0 else 'decrease')} in market value and the current {recent_trend} trend, collectors should consider {('buying now' if percentage_change > 0 else 'monitoring the market closely')} and planning for potential purchases during {best_month_name}. This approach allows you to capitalize on the forecasted trends while taking advantage of the best buying opportunities within the year.")
st.write(f"Ultimately, your strategy should align with your collecting goals, whether you're looking to buy, hold, or sell {selected_category} cards. Staying informed and responsive to market changes will help you make the most of your collection.")

# Disclaimer
st.write("**Disclaimer:** This is not financial advice. It is an analytics read of the data and an interpretation of the results. Collecting can often be more of an artform than a science, and this analysis represents the scientific perspective. Please consider the artistic and personal aspects of collecting as well when making your decisions.")






# If comparing two categories, run the same analysis for the second category and display comparisons
if compare:
    filtered_data_2 = data[data['Category'] == selected_category_2]
    analysis_results_2 = run_analysis(filtered_data_2, selected_category_2)
    
    st.subheader(f"Comparison with {selected_category_2}")
    
    # Comparison of projected percentage change
    initial_value_2 = analysis_results_2['time_series'].iloc[-1]
    final_value_2 = analysis_results_2['forecast'].iloc[-1]
    percentage_change_2 = ((final_value_2 - initial_value_2) / initial_value_2) * 100
    
    st.write(f"Projected percentage change for {selected_category}: {percentage_change:.2f}%")
    st.write(f"Projected percentage change for {selected_category_2}: {percentage_change_2:.2f}%")
    
    # Comparison of most recent MACD trend
    recent_trend_2 = analysis_results_2['trend_buckets'][-1]
    
    st.write(f"Most recent MACD trend for {selected_category}: {recent_trend}")
    st.write(f"Most recent MACD trend for {selected_category_2}: {recent_trend_2}")
    
    # Comparison of best time to buy
    best_month_name_2 = pd.to_datetime(f'{analysis_results_2["best_month"]}', format='%m').strftime('%B')
    
    st.write(f"Best time to buy cards in {selected_category}: {best_month_name}")
    st.write(f"Best time to buy cards in {selected_category_2}: {best_month_name_2}")
    
    # Plot comparison of forecasted values
    st.subheader(f"Forecast Comparison for {selected_category} and {selected_category_2}")
    fig, ax = plt.subplots()
    analysis_results['time_series'].plot(ax=ax, label=f'{selected_category} Observed')
    analysis_results['forecast'].plot(ax=ax, label=f'{selected_category} Forecast')
    analysis_results_2['time_series'].plot(ax=ax, label=f'{selected_category_2} Observed', linestyle='--')
    analysis_results_2['forecast'].plot(ax=ax, label=f'{selected_category_2} Forecast', linestyle='--')
    ax.fill_between(analysis_results['forecast'].index, analysis_results['conf_int']['lower'], analysis_results['conf_int']['upper'], color='gray', alpha=0.2)
    ax.fill_between(analysis_results_2['forecast'].index, analysis_results_2['conf_int']['lower'], analysis_results_2['conf_int']['upper'], color='blue', alpha=0.2)
    ax.set_title(f"Forecast Comparison between {selected_category} and {selected_category_2}")
    ax.set_ylabel('Market Value')
    ax.legend()
    st.pyplot(fig)
    
    # Comparison of MACD plots
    st.subheader(f"MACD Comparison for {selected_category} and {selected_category_2}")
    fig, ax = plt.subplots()
    analysis_results['macd'].plot(ax=ax, label=f'{selected_category} MACD')
    analysis_results['signal'].plot(ax=ax, label=f'{selected_category} Signal')
    analysis_results_2['macd'].plot(ax=ax, label=f'{selected_category_2} MACD', linestyle='--')
    analysis_results_2['signal'].plot(ax=ax, label=f'{selected_category_2} Signal', linestyle='--')

    # Loop through MACD trends for the first category and add shading based on trend buckets
    for i in range(len(analysis_results['trend_buckets']) - 1):
        if analysis_results['trend_buckets'][i] == 'High Upward':
            ax.axvspan(analysis_results['macd'].index[i], analysis_results['macd'].index[i+1], color='green', alpha=0.3)
        elif analysis_results['trend_buckets'][i] == 'Medium Upward':
            ax.axvspan(analysis_results['macd'].index[i], analysis_results['macd'].index[i+1], color='lightgreen', alpha=0.3)
        elif analysis_results['trend_buckets'][i] == 'Low Upward':
            ax.axvspan(analysis_results['macd'].index[i], analysis_results['macd'].index[i+1], color='yellow', alpha=0.3)
        elif analysis_results['trend_buckets'][i] == 'Low Downward':
            ax.axvspan(analysis_results['macd'].index[i], analysis_results['macd'].index[i+1], color='orange', alpha=0.3)
        elif analysis_results['trend_buckets'][i] == 'High Downward':
            ax.axvspan(analysis_results['macd'].index[i], analysis_results['macd'].index[i+1], color='red', alpha=0.3)

    # Loop through MACD trends for the second category and add shading based on trend buckets
    for i in range(len(analysis_results_2['trend_buckets']) - 1):
        if analysis_results_2['trend_buckets'][i] == 'High Upward':
            ax.axvspan(analysis_results_2['macd'].index[i], analysis_results_2['macd'].index[i+1], color='blue', alpha=0.3)
        elif analysis_results_2['trend_buckets'][i] == 'Medium Upward':
            ax.axvspan(analysis_results_2['macd'].index[i], analysis_results_2['macd'].index[i+1], color='lightblue', alpha=0.3)
        elif analysis_results_2['trend_buckets'][i] == 'Low Upward':
            ax.axvspan(analysis_results_2['macd'].index[i], analysis_results_2['macd'].index[i+1], color='cyan', alpha=0.3)
        elif analysis_results_2['trend_buckets'][i] == 'Low Downward':
            ax.axvspan(analysis_results_2['macd'].index[i], analysis_results_2['macd'].index[i+1], color='orange', alpha=0.3)
        elif analysis_results_2['trend_buckets'][i] == 'High Downward':
            ax.axvspan(analysis_results_2['macd'].index[i], analysis_results_2['macd'].index[i+1], color='purple', alpha=0.3)

    ax.set_title(f"MACD and Signal Line Comparison between {selected_category} and {selected_category_2}")
    ax.set_ylabel('Value')
    ax.legend()
    st.pyplot(fig)

    # Comparison of best time to buy bar charts
    st.subheader(f"Best Time to Buy Comparison for {selected_category} and {selected_category_2}")
    fig, ax = plt.subplots()
    analysis_results['monthly_avg'].plot(kind='bar', ax=ax, color='blue', alpha=0.6, position=0, width=0.4, label=f'{selected_category}')
    analysis_results_2['monthly_avg'].plot(kind='bar', ax=ax, color='green', alpha=0.6, position=1, width=0.4, label=f'{selected_category_2}')
    ax.set_title(f"Best Time to Buy Comparison between {selected_category} and {selected_category_2}")
    ax.set_xlabel('Month')
    ax.set_ylabel('Average Market Value')
    ax.legend()
    st.pyplot(fig)

# Final Read Out for the selected trading card categories
    st.subheader(f"Final Read Out: {selected_category} vs. {selected_category_2}")

# Integrating the results from the forecast, MACD, and best time to buy for both categories
# Forecast Insight Comparison
    if percentage_change < 0 and percentage_change_2 < 0:
        st.write(f"**Forecast Insight:** Both {selected_category} and {selected_category_2} are projected to experience a decrease in market value over the next 12 months, with drops of {percentage_change:.2f}% and {percentage_change_2:.2f}%, respectively. Collectors might consider waiting for further price drops before buying.")
        st.write(f"**Outlook:** **{selected_category_2 if percentage_change_2 > percentage_change else selected_category}** has the better forecast outlook, with a smaller projected decrease in value.")
    elif percentage_change > 0 and percentage_change_2 > 0:
        st.write(f"**Forecast Insight:** Both {selected_category} and {selected_category_2} are forecasted to increase in market value by {percentage_change:.2f}% and {percentage_change_2:.2f}%, respectively. This could be a good time to purchase cards from either category before prices rise.")
        st.write(f"**Outlook:** **{selected_category if percentage_change > percentage_change_2 else selected_category_2}** has the better forecast outlook, with a higher projected increase in value.")
    else:
        st.write(f"**Forecast Insight:** The {selected_category} category is projected to {('increase' if percentage_change > 0 else 'decrease')} by {percentage_change:.2f}%, while {selected_category_2} is expected to {('increase' if percentage_change_2 > 0 else 'decrease')} by {percentage_change_2:.2f}%. Collectors might prioritize {('buying' if percentage_change > 0 else 'monitoring')} {selected_category} cards, while considering {('buying' if percentage_change_2 > 0 else 'monitoring')} {selected_category_2} cards as well, depending on their collecting goals.")
        st.write(f"**Outlook:** **{selected_category if percentage_change > percentage_change_2 else selected_category_2}** has the better forecast outlook based on the projected percentage change.")

    # MACD Trend Comparison
    st.write(f"**MACD Trend:** The most recent MACD trend shows a **{recent_trend}** trend for {selected_category} and a **{recent_trend_2}** trend for {selected_category_2}. This suggests that {selected_category} is currently {('gaining' if 'Upward' in recent_trend else 'losing')} momentum, while {selected_category_2} is {('gaining' if 'Upward' in recent_trend_2 else 'losing')} momentum.")
    st.write(f"**Trend:** **{selected_category if 'Upward' in recent_trend else selected_category_2}** has the better MACD trend, indicating stronger momentum in the market.")

    # Final Summary and Collector's Strategy
    st.write(f"**Collector's Strategy:** Considering the forecast, {selected_category if percentage_change > percentage_change_2 else selected_category_2} has the better long-term outlook, while {selected_category if 'Upward' in recent_trend else selected_category_2} shows stronger short-term momentum. Collectors might prioritize purchasing {selected_category if percentage_change > 0 else selected_category_2} cards to take advantage of the better forecasted growth, but also consider the short-term trends to time their purchases effectively.")
    st.write(f"For those looking to diversify, balancing acquisitions between these categories based on their market dynamics could be a strategic approach. Stay flexible and responsive to market changes, leveraging both the forecast and MACD insights to optimize your collecting strategy.")

    # Disclaimer
    st.write("**Disclaimer:** This is not financial advice. It is an analytics read of the data and an interpretation of the results. Collecting can often be more of an artform than a science, and this analysis represents the scientific perspective. Please consider the artistic and personal aspects of collecting as well when making your decisions.")

# Thank You Section with Blue Subheader
st.markdown("<h2 style='color:blue;'>Thank You for Using CardBoard Compass</h2>", unsafe_allow_html=True)

st.write("We hope you found CardBoard Compass helpful in expanding your collecting knowledge through data-driven insights. Whether you're a seasoned collector or just starting out, our goal is to provide you with the tools to make informed decisions in the ever-changing world of trading cards.")

st.write("### Use Cases for CardBoard Compass:")
st.write("- **Market Forecasting:** Understand future market trends with our 12-month forecasts, helping you time your purchases and sales effectively.")
st.write("- **Short-Term Trend Analysis:** Leverage the MACD analysis to gain insights into the short-term momentum of your favorite trading card categories.")
st.write("- **Optimal Buying Times:** Identify the best times to buy cards based on historical market value trends, maximizing your investment in the hobby.")
st.write("- **Category Comparison:** Compare different trading card categories to diversify your collection and make strategic decisions based on market data.")

st.write("### About Pancake Analytics:")
st.write("Pancake Analytics is a leader in providing cutting-edge data analytics solutions for the collectibles market. Our expertise spans across various domains, offering deep insights into trading cards, comics, and other collectibles. We specialize in helping collectors, investors, and enthusiasts make data-driven decisions to enhance their collections and investments.")
st.write("Thank you for choosing CardBoard Compass powered by Pancake Analytics. We're here to help you navigate the exciting world of trading cards with confidence and clarity.")



if __name__ == "__main__":
    # The script will automatically run when you execute it with Streamlit.
    pass
