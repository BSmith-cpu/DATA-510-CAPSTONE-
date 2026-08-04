from calendar import month
from altair import value
from numpy import long
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import scipy.stats as stats


zillow_data = pd.read_csv("/Users/brandonsmith/DATA-510-CAPSTONE-/DATA-510-CAPSTONE-/data/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv")
hpi_at_metro = pd.read_csv("/Users/brandonsmith/DATA-510-CAPSTONE-/DATA-510-CAPSTONE-/data/hpi_at_metro.csv")
cbsa_2025 = pd.read_csv("/Users/brandonsmith/DATA-510-CAPSTONE-/DATA-510-CAPSTONE-/data/cbsa-est2025-alldata.csv", encoding='iso-8859-1')
cbsa_2019 = pd.read_csv("/Users/brandonsmith/DATA-510-CAPSTONE-/DATA-510-CAPSTONE-/data/cbsa-est2019-alldata.csv", encoding='iso-8859-1')

#print(zillow_data.head())
#print(zillow_data.columns.tolist()) # this was the main issue even before doing the EDA, just looking up the raw csv file. we saw they were counting the years/data inputs through the columns which is an issue. the next step will be to flip the data into long format.
#print(zillow_data.dtypes)


# We are needing to reshape the zillow_data to a long format
dates_cols = [c for c in zillow_data.columns if not pd.isna(pd.to_datetime(c, errors='coerce'))]
id_cols = [c for c in zillow_data.columns if c not in dates_cols]

zillow_long = zillow_data.melt(id_vars = id_cols, value_vars = dates_cols, var_name = 'Date', value_name = 'Value')
zillow_long['Date'] = pd.to_datetime(zillow_long['Date'])


#print(zillow_long.head())
#print(zillow_long.columns.tolist()) # by flipping it to long format, we were able to have 7 columns - RegionID, SizeRank, RegionName, RegionType, StateName, Date, and Value

# Zillow EDA

# Time Series Plot
ts_national = zillow_long.groupby('Date')['Value'].mean().reset_index()
plt.figure(figsize =(12,5))
plt.plot(ts_national['Date'], ts_national['Value'])
plt.title('National Average Zillow Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.tight_layout()
#plt.show()
#plt.savefig('/Users/brandonsmith/DATA-510-CAPSTONE-/DATA-510-CAPSTONE-/data/zillow_plots/timeseriesplot.png', dpi=100, bbox_inches='tight')

# Missing Values Heatmap
plt.figure(figsize=(14, 6))
sns.heatmap(zillow_long.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.xlabel('Date')
plt.ylabel('Region')
plt.tight_layout()
#plt.show()
#plt.savefig('/Users/brandonsmith/DATA-510-CAPSTONE-/DATA-510-CAPSTONE-/data/zillow_plots/missing_values_heatmap.png', dpi=100, bbox_inches='tight')

# Checking Seasonality
zillow_long['month'] = zillow_long['Date'].dt.month
seasonality = zillow_long.groupby('month')['Value'].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(data=seasonality, x='month', y='Value')
plt.title('Average Zillow Prices by Month')
plt.xlabel('Month')
plt.ylabel('Average Price')
plt.xticks(ticks=range(1, 13), labels=[
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
])
plt.tight_layout()
#plt.show()
#plt.savefig('/Users/brandonsmith/DATA-510-CAPSTONE-/DATA-510-CAPSTONE-/data/zillow_plots/seasonality_plot.png', dpi=100, bbox_inches='tight')


# Distribution of Prices
plt.figure(figsize=(12, 6))
sns.histplot(zillow_long['Value'], bins=50, kde=True)
plt.title('Distribution of Zillow Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.tight_layout()
#plt.show()
#plt.savefig('/Users/brandonsmith/DATA-510-CAPSTONE-/DATA-510-CAPSTONE-/data/zillow_plots/distribution_plot.png', dpi=100, bbox_inches='tight')

# Boxplot by State
top_regions = zillow_long.groupby('RegionName')['Value'].mean().nlargest(15).index
long_top = zillow_long[zillow_long['RegionName'].isin(top_regions)]

plt.figure(figsize=(14, 7))
sns.boxplot(data=long_top, x='RegionName', y='Value', hue='RegionName', palette='Set2', legend=False)
plt.title('Zillow Prices by Region (Top 15)', fontsize=14, fontweight='bold')
plt.xlabel('Region', fontsize=12)
plt.ylabel('Price ($)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
#plt.show()
#plt.savefig('/Users/brandonsmith/DATA-510-CAPSTONE-/DATA-510-CAPSTONE-/data/zillow_plots/boxplot_by_state.png', dpi=100, bbox_inches='tight')

# House Price Index at Metro Level
# They are missing column headers so we are adding those in 
hpi_at_metro.columns = ['Metro', 'Area Code', 'Year', 'Quarter', 'Value', 'Difference']
hpi_at_metro.to_csv('/Users/brandonsmith/DATA-510-CAPSTONE-/DATA-510-CAPSTONE-/data/hpi_at_metro.csv', index=False)
# Loading the file back in
hpi_at_metro = pd.read_csv('/Users/brandonsmith/DATA-510-CAPSTONE-/DATA-510-CAPSTONE-/data/hpi_at_metro.csv')


# Isolating our Training Cities
training_cities = hpi_at_metro[hpi_at_metro['Metro'].isin(['Boise City, ID', 'Tampa, FL (MSAD)', '"Austin-Round Rock-San Marcos, TX'])]
print(training_cities.head())
print(training_cities.dtypes)

# Value and Difference are strings 

#hpi_at_metro['Value'] = hpi_at_metro['Value'].str.replace('$', '').str.replace(',', '').astype(float)
#hpi_at_metro['Difference'] = hpi_at_metro['Difference'].str.replace('$', '').str.replace(',', '').astype(float)

#Tried the conversion with the code above, it failed since instead of having NANs or 0s they decided to use dashes("-"), we can run a simple replace function that replaces the dash with a 0, making the strip and replace easier

hpi_at_metro['Value'] = hpi_at_metro['Value'].str.replace('-', '0').str.replace('$', '').str.replace(',', '').astype(float)
hpi_at_metro['Difference'] = hpi_at_metro['Difference'].str.replace('-', '0').str.replace('(', '').str.replace(')', '').str.replace('$', '').str.replace(',', '').str.strip().astype(float)


# Missing Values Heatmap
plt.figure(figsize=(14, 6))
sns.heatmap(hpi_at_metro.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.xlabel('Date')
plt.ylabel('Region')
plt.tight_layout()
#plt.show()
#plt.savefig('/Users/brandonsmith/DATA-510-CAPSTONE-/DATA-510-CAPSTONE-/data/hpi_at_metro_plots/missing_values_heatmap.png', dpi=100, bbox_inches='tight')


# Time Series Showing All Cities Growth
plt.figure(figsize=(12, 5))

# Plot each training city separately
for city in ['Boise City, ID', 'Tampa, FL (MSAD)', 'Austin-Round Rock-San Marcos, TX']:
    city_data = hpi_at_metro[hpi_at_metro['Metro'] == city]
    # Sort by Year and Quarter to ensure proper line plotting
    city_data = city_data.sort_values(['Year', 'Quarter'])
    plt.plot(city_data['Year'], city_data['Value'], marker='o', label=city, linewidth=2)

plt.title('HPI  over Time by Training City')
plt.xlabel('Year')
plt.ylabel('HPI Value')
plt.legend()
plt.grid(True, alpha=0.3)
ax = plt.gca()
ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
plt.tight_layout()
#plt.show()
#plt.savefig('/Users/brandonsmith/DATA-510-CAPSTONE-/DATA-510-CAPSTONE-/data/hpi_at_metro_plots/hpi_over_time_by_city.png', dpi=100, bbox_inches='tight')