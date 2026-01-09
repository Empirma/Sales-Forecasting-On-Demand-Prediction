import os
import pandas as pd
import numpy as np
import re
import holidays
from pytz import timezone
import regex as re
from datetime import datetime, timedelta



eg_holidays = holidays.EG(language='en_US')

def count_items_in_order(lineitem_Qty):
    return sum(lineitem_Qty)


def get_holiday_name(date):
    return eg_holidays.get(date)


def get_holiday_dates(year):
    holiday_dates = {}
    
    holiday_dates['Christmas'] = datetime(year, 12, 25)
    holiday_dates['Valentine\'s Day'] = datetime(year, 2, 14)
    holiday_dates['Mother\'s Day'] = datetime(year, 3, 21)

    holiday_dates['Black Friday'] = get_nth_weekday_of_month(year, 11, 3, 5)

    
    return holiday_dates


def is_ramadan(date_series):
    ramadan_dates = {
        2022: {'start': '2022-04-02', 'end': '2022-05-01'},
        2023: {'start': '2023-03-23', 'end': '2023-04-21'},
        2024: {'start': '2024-03-11', 'end': '2024-04-09'},
        2025: {'start': '2025-03-01', 'end': '2025-03-30'}
    }
    
    result = pd.Series(0, index=date_series.index)
    
    for year, dates in ramadan_dates.items():
        start = pd.to_datetime(dates['start'],utc=True).tz_convert('Africa/Cairo')
        end = pd.to_datetime(dates['end'],utc=True).tz_convert('Africa/Cairo')
        mask = (date_series >= start) & (date_series <= end)
        result[mask] = 1
        
    return result


def is_eid(date_series):
    eid_dates = {
        2022: {
            'eid_al_fitr_start': '2022-05-02', 'eid_al_fitr_end': '2022-05-04',
            'eid_al_adha_start': '2022-07-09', 'eid_al_adha_end': '2022-07-12'
        },
        2023: {
            'eid_al_fitr_start': '2023-04-22', 'eid_al_fitr_end': '2023-04-24',
            'eid_al_adha_start': '2023-06-28', 'eid_al_adha_end': '2023-07-01'
        },
        2024: {
            'eid_al_fitr_start': '2024-04-10', 'eid_al_fitr_end': '2024-04-12',
            'eid_al_adha_start': '2024-06-16', 'eid_al_adha_end': '2024-06-19'
        },
        2025: {
            'eid_al_fitr_start': '2025-03-31', 'eid_al_fitr_end': '2025-04-02',
            'eid_al_adha_start': '2025-06-06', 'eid_al_adha_end': '2025-06-09'
        }
    }
    
    result = pd.Series(0, index=date_series.index)
    
    for year, dates in eid_dates.items():
        fitr_start = pd.to_datetime(dates['eid_al_fitr_start'])
        fitr_end = pd.to_datetime(dates['eid_al_fitr_end'])
        fitr_mask = (date_series >= fitr_start) & (date_series <= fitr_end)
        
        adha_start = pd.to_datetime(dates['eid_al_adha_start'])
        adha_end = pd.to_datetime(dates['eid_al_adha_end'])
        adha_mask = (date_series >= adha_start) & (date_series <= adha_end)
        
        combined_mask = fitr_mask | adha_mask
        result[combined_mask] = 1
        
    return result


def get_nth_weekday_of_month(year, month, weekday, n):
    date = datetime(year, month, 1)
    
    days_until_first = (weekday - date.weekday()) % 7
    first_occurrence = date + timedelta(days=days_until_first)
    
    result = first_occurrence + timedelta(days=7 * (n - 1))
    
    return result


def days_to_next_holiday(date, holidays_dict):
    if not isinstance(date, datetime):
        date = pd.to_datetime(date)
    
    # Find the next holiday
    future_holidays = {name: holiday_date for name, holiday_date in holidays_dict.items() if holiday_date >= date}
    
    # If no future holidays in current year, look at next year
    if not future_holidays:
        next_year_holidays = get_holiday_dates(date.year + 1)
        next_holiday_name = min(next_year_holidays, key=lambda x: next_year_holidays[x])
        next_holiday_date = next_year_holidays[next_holiday_name]
    else:
        next_holiday_name = min(future_holidays, key=lambda x: future_holidays[x])
        next_holiday_date = future_holidays[next_holiday_name]
    
    # Calculate days difference
    days_difference = (next_holiday_date - date).days
    
    return days_difference, next_holiday_name


def calculate_days_to_next_shopping_holiday(dates):
    """Calculate days to next shopping holiday for a series of dates."""
    # Pre-compute holiday dates for relevant years
    years_range = range(pd.to_datetime(min(dates)).year, pd.to_datetime(max(dates)).year + 2)
    holiday_dates_by_year = {year: get_holiday_dates(year) for year in years_range}
    
    days_list, next_holiday_list = [], []
    
    for date in dates:
        date_dt = pd.to_datetime(date).tz_localize(None) if hasattr(pd.to_datetime(date), 'tz') else pd.to_datetime(date)
        year = date_dt.year
        
        future_holidays = {name: date for name, date in holiday_dates_by_year[year].items() if date >= date_dt}
        
        if not future_holidays:
            # Look at next year if no future holidays in current year
            next_year = holiday_dates_by_year[year + 1]
            next_holiday = min(next_year.items(), key=lambda x: x[1])
        else:
            next_holiday = min(future_holidays.items(), key=lambda x: x[1])
        
        days_list.append((next_holiday[1] - date_dt).days)
        next_holiday_list.append(next_holiday[0])
    
    return pd.Series(days_list), pd.Series(next_holiday_list)



def load_data(file_path):
    """
    Load data from a file based on its extension (CSV, Excel, JSON).

    Parameters:
    file_path (str): Path to the file.

    Returns:
    pd.DataFrame: Loaded data as a DataFrame, or None if loading fails.
    """
    try:
        ext = os.path.splitext(file_path)[-1]
        ext = ext.lower()

        if ext == '.csv':
            df = pd.read_csv(file_path)
        elif ext in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        elif ext == '.json':
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        return df
    except:
        print(f"Error loading file: {file_path}. Please check the file format and path.")
        exit()


def investigating_data(df):
    """
    Investigate the data by checking for missing values and data types.
    
    Parameters:
    data (pd.DataFrame): Data to investigate.
    
    Returns:
    None
    """
    print("Data Info:")
    print(df.info())

    print("\nMissing Values:")
    df.isnull().sum()  

    print("\nDuplicated Values:")
    df.duplicated().sum()

    print("\nData Description:")
    print(df.describe())


def split_numeric_categorical(df):
    """
    Split a DataFrame into numeric and categorical columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    tuple: (numeric_df, categorical_df)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    numeric_df = df.select_dtypes(include=['number'])
    categorical_df = df.select_dtypes(exclude=['number'])

    return numeric_df, categorical_df


def handle_unneeded_columns(df):
    """
    Handle unneeded columns in the data.
    
    Parameters:
    df (pd.DataFrame): Data to process.
    
    Returns:
    pd.DataFrame: Data with unneeded columns handled.
    """
    # Drop billing-related columns
    billing_columns = df.columns[df.columns.str.contains('Billing')]
    df.drop(billing_columns, axis=1, inplace=True, errors='ignore')

    # Drop specific unwanted columns, if they exist
    columns_to_drop = [
        'Source', 'Risk Level', 'Tags', 'Shipping Country', 'Shipping City',
        'Lineitem fulfillment status', 'Lineitem taxable',
        'Lineitem requires shipping', 'Taxes', 'Currency',
        'Fulfilled at', 'Fulfillment Status', 'Paid at', 'Financial Status'
    ]
    df.drop(columns_to_drop, axis=1, inplace=True, errors='ignore')

    print("Unneeded columns dropped.")
    return df



def handle_missing_columns(df):
    """
    Handle missing columns in the data.
    
    Parameters:
    data (pd.DataFrame): Data to process.
    
    Returns:
    pd.DataFrame: Data with missing columns handled.
    """
    columns = df.columns[df.notna().sum() == 0]
    
    df.drop(columns, axis=1, inplace=True)
    print(f"Missing columns dropped.")
    return df


def handle_missing_rows(df):
    """
    Handle missing rows in the data.
    
    Parameters:
    data (pd.DataFrame): Data to process.
    
    Returns:
    pd.DataFrame: Data with missing rows handled.
    """

    # Forward-fill order-level details
    order_columns = ["Subtotal", "Shipping", "Total", "Discount Amount", "Shipping Method",
                    "Shipping Province", "Shipping Province Name", "Payment Method", "Created at"]
    df[order_columns] = df[order_columns].fillna(method="ffill")

    df['Lineitem compare at price'] = df['Lineitem compare at price'].fillna(0)

    # Create a dictionary to map Lineitem name to Lineitem sku
    sku_mapping = df.dropna(subset=['Lineitem sku']).set_index('Lineitem name')['Lineitem sku'].to_dict()

    # Fill missing Lineitem sku values using the mapping
    df['Lineitem sku'] = df.apply(
        lambda row: sku_mapping[row['Lineitem name']] if pd.isnull(row['Lineitem sku']) and row['Lineitem name'] in sku_mapping else row['Lineitem sku'],
        axis=1
    )

    # Fill any remaining missing Lineitem sku values with 'Unknown'
    df['Lineitem sku'] = df['Lineitem sku'].fillna('Unknown')

    # Encoding lineitem sku
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit_transform(df['Lineitem name'])
    df['Lineitem sku_made'] = le.transform(df['Lineitem name'])

    df.isnull().sum()

    # df.to_csv('data/depi_ungrouped.csv', index=False)
    # print("Missing rows handled and saved to depi_ungrouped.csv.")

    return df


def is_float(value):
    try:
        float(value)
        return True
    except:
        return False


def grouping_data(df):
    """
    Group data by 'Name' and aggregate values safely, handling bad or missing numeric strings.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: Grouped and cleaned DataFrame.
    """

    # Group and aggregate
    df_grouped = df.groupby("Name").agg({
        "Created at": "first",
        "Subtotal": "first",
        "Shipping": "first",
        "Total": "first",
        "Discount Amount": "first",
        "Shipping Method": "first",
        "Shipping Province": "first",
        "Shipping Province Name": "first",
        "Payment Method": "first",

        "Lineitem name": lambda x: [item.strip() for item in x],
        "Lineitem price": lambda x: [float(p) for p in x if is_float(p)],
        "Lineitem compare at price": lambda x: [float(p) for p in x if is_float(p)],
        "Lineitem sku": lambda x: [sku for sku in x],
        "Lineitem sku_made": lambda x: [int(sku) for sku in x if str(sku).isdigit()],
        "Lineitem quantity": lambda x: [int(float(q)) for q in x if is_float(q)],
        "Lineitem discount": lambda x: [float(d) for d in x if is_float(d)],

        "Cancelled at": "first",
        "Refunded Amount": "first"
    }).reset_index()

    # Convert datetime
    df_grouped['Created at'] = pd.to_datetime(
        df_grouped['Created at'],
        format='%Y-%m-%d %H:%M:%S %z',
        utc=True
    ).dt.tz_convert(timezone('Etc/GMT-2'))

    return df_grouped


def preprocess_ungrouped(df):

    df = handle_unneeded_columns(df)
    df = handle_missing_columns(df)
    df = handle_missing_rows(df)

    return df


def feature_engineering(df):
    """
    Perform feature engineering on the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.DataFrame: The DataFrame with new features added.
    """
    # df['Created at'] = pd.to_datetime(df['Created at'],format='%Y-%m-%d %H:%M:%S %z',utc=True).dt.tz_convert(timezone('Etc/GMT-2'))
    # df['Cancelled'] = pd.to_datetime(df['Cancelled at'],format='%Y-%m-%d %H:%M:%S %z',utc=True).dt.tz_convert(timezone('Etc/GMT-2'))
    # df['day_type'] = 'Weekday'
    # df['Lineitem price'] = df['Lineitem price'].str.split(',').apply(lambda x: [float(p) for p in x])
    # df['Lineitem name'] = df['Lineitem name'].str.split(',').apply(lambda x: [item.strip() for item in x])
    # df['Lineitem quantity'] = df['Lineitem quantity'].str.split(',').apply(lambda x: [int(float(q)) for q in x])
    # df['Lineitem sku'] = df['Lineitem sku'].str.split(',').apply(lambda x: [sku.strip() for sku in x])
    # df['Lineitem sku_made'] = df['Lineitem sku_made'].str.split(',').apply(lambda x: [int(sku) for sku in x])
    # df['Lineitem compare at price'] = df['Lineitem compare at price'].str.split(',').apply(lambda x: [float(p) for p in x])
    # df['Lineitem discount'] = df['Lineitem discount'].str.split(',').apply(lambda x: [float(d) for d in x])

    
    df['Cancelled'] = df['Cancelled at'].notnull().astype(int)

    df['hour'] = df['Created at'].dt.hour
    df['month'] = df['Created at'].dt.month
    df['year'] = df['Created at'].dt.year

    df['day_of_week'] = df['Created at'].dt.day_name()
    df['year_month'] = df['Created at'].dt.to_period('M')

    df['yearly_orders'] = df.groupby('year')['Created at'].transform('count')
    df['monthly_orders'] = df.groupby('year_month')['Created at'].transform('count')


    df['season'] = pd.cut(df['Created at'].dt.month, 
                        bins=[0,3,6,9,12], 
                        labels=['Winter', 'Spring', 'Summer', 'Fall'])


    df['holiday_name'] = df['Created at'].apply(get_holiday_name)
    df['is_holiday'] = df['holiday_name'].notna()
    df['is_weekend'] = df['Created at'].dt.dayofweek.isin([4, 5])

    df.loc[df['is_weekend'], 'day_type'] = 'Weekend'
    df.loc[df['is_holiday'], 'day_type'] = 'Holiday'



    df['item_count']= df['Lineitem quantity'].apply(count_items_in_order)

    # df.to_csv('data/depi_grouped.csv',index=False)
    
    return df


def time_series_features(df):
    """
    Create time series features from the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.DataFrame: The DataFrame with time series features added.
    """
    df_ts = df.copy()
    df_ts = df_ts.set_index('Created at')

    # 1. ROLLING AVERAGES AND STATISTICS
    daily_sales = df_ts.resample('D')[['Total', 'Subtotal', 'item_count']].sum()

    # Create rolling features
    daily_sales['total_7d_avg'] = daily_sales['Subtotal'].rolling(window=7).mean()
    daily_sales['items_7d_avg'] = daily_sales['item_count'].rolling(window=7).mean()
    daily_sales['total_7d_std'] = daily_sales['Subtotal'].rolling(window=7).std()


    daily_sales['items_30d_avg'] = daily_sales['item_count'].rolling(window=30).mean()
    daily_sales['total_30d_avg'] = daily_sales['Subtotal'].rolling(window=30).mean()
    daily_sales['total_30d_std'] = daily_sales['Subtotal'].rolling(window=30).std()

    daily_sales['total_90d_avg'] = daily_sales['Subtotal'].rolling(window=90).mean()
    daily_sales['items_90d_avg'] = daily_sales['item_count'].rolling(window=90).mean()
    daily_sales['total_90d_std'] = daily_sales['Subtotal'].rolling(window=90).std()

    # 2. LAG FEATURES
    daily_sales['total_prev_day'] = daily_sales['Subtotal'].shift(1)
    daily_sales['total_prev_week'] = daily_sales['Subtotal'].shift(7)
    daily_sales['total_prev_month'] = daily_sales['Subtotal'].shift(30)
    daily_sales['total_prev_season'] = daily_sales['Subtotal'].shift(90)

    daily_sales['items_prev_day'] = daily_sales['item_count'].shift(1)
    daily_sales['items_prev_week'] = daily_sales['item_count'].shift(7)
    daily_sales['items_prev_month'] = daily_sales['item_count'].shift(30)
    daily_sales['items_prev_season'] = daily_sales['item_count'].shift(90)



    # 3. RATIOS AND GROWTH METRICS
    daily_sales['total_growth_1d'] = daily_sales['Subtotal'] / daily_sales['total_prev_day'] - 1
    daily_sales['total_growth_1w'] = daily_sales['Subtotal'] / daily_sales['total_prev_week'] - 1
    daily_sales['total_growth_1m'] = daily_sales['Subtotal'] / daily_sales['total_prev_month'] - 1
    daily_sales['total_growth_1s'] = daily_sales['Subtotal'] / daily_sales['total_prev_season'] - 1

    # 4. SEASONALITY FEATURES
    daily_sales['quarter'] = daily_sales.index.quarter
    daily_sales['day_of_year'] = daily_sales.index.dayofyear
    daily_sales['week_of_year'] = daily_sales.index.isocalendar().week

    # 5. FOURIER TRANSFORMS FOR CYCLICAL PATTERNS
    daily_sales['week_sin'] = np.sin(2 * np.pi * daily_sales.index.dayofweek / 7)
    daily_sales['week_cos'] = np.cos(2 * np.pi * daily_sales.index.dayofweek / 7)

    # Yearly cyclical pattern
    daily_sales['year_sin'] = np.sin(2 * np.pi * daily_sales.index.dayofyear / 365)
    daily_sales['year_cos'] = np.cos(2 * np.pi * daily_sales.index.dayofyear / 365)


    #daily_sales['avg_item_price'] = df_ts.groupby(pd.Grouper(freq='D'))[].mean()




    daily_order_count = df_ts.groupby(pd.Grouper(freq='D')).size()


    # 9. ADVANCED TIME-WINDOW FEATURES
    daily_sales['avg_order_value'] = daily_sales['Subtotal'] / daily_order_count
    daily_sales['avg_items_per_order'] = daily_sales['item_count'] / daily_order_count

    # 10. COMBINE WITH EXTERNAL FACTORS
    shopping_holidays = ['Black Friday', 'Christmas', 'Valentine\'s Day', 'Mother\'s Day']


    days_series, holiday_series = calculate_days_to_next_shopping_holiday(daily_sales.index)
    daily_sales['days_to_next_shopping_holiday'] = days_series
    daily_sales['next_shopping_holiday'] = holiday_series

    daily_sales['pre_holiday_period'] = daily_sales['days_to_next_shopping_holiday'].apply(
        lambda x: 1 if x <= 7 else 0  # 1 week before holiday
    )


    daily_sales = daily_sales.reset_index()

    # daily_sales['is_ramadan'] = is_ramadan(df['Created at'])
    # daily_sales['is_eid'] = is_eid(daily_sales.index)
    # daily_sales['is_eid_al_fitr'] = is_eid_al_fitr(daily_sales.index)  
    # daily_sales['is_eid_al_adha'] = is_eid_al_adha(daily_sales.index)  
    # daily_sales['pre_ramadan'] = pre_ramadan_period(daily_sales.index)  
    # post_holiday_mask = daily_sales['next_shopping_holiday'].shift(-1) != daily_sales['next_shopping_holiday']
    # daily_sales['post_holiday_period'] = (post_holiday_mask & (daily_sales['days_to_next_shopping_holiday'].shift(-1) > 350)).astype(int)


    # daily_sales.to_csv('data/depi_time_series.csv',index = False)

    return daily_sales


# Extract and normalize
def extract_category(text):
    pattern = re.compile(
    r'(?i)\b('
    r'hand ?(plain|plaine)? ?towel?s?|'
    r'towel?s?|'
    r'face ?(plain|plaine)? ?towel?s?|'
    r'beach ?(plain|plaine)? ?towel?s?|'
    r'bathmat ?(plain|plaine)? ?towel?s?|'
    r'kitchen ?(plain|plaine)? ?towel?s?|'
    r'bathmat?s?|'
    r'bath ?mat?s? ?(plain|plaine)? ?towel?s?|'
    r'beach ?mat?s?|'
    r'mattress ?protector?s?|'
    r'mattress ?topper?s?|'
    r'bathrobe?s?|'
    r'pillow?s?|'
    r'pillow ?case?s?|'
    r'coverlet?s?|'
    r'bed ?sheet?s?|'
    r'fitted ?sheet?s?|'
    r'blanket?s?|'
    r'cushion?s?|'
    r'duvet?s?|'
    r'curtain?s?|'
    r'bundle?s?|'
    r'set?s?|'
    r'slipper?s?|'
    r'apron?s?|'
    r'pouch?s?|'
    r'mat?s?|'
    r'throw?s?|'
    r'bag?s?|'
    r'candle?s?|'
    r'robe?s?'
    r')\b'
)

    normalize_map = {
        'hand towel': 'hand towel',
        'hand towels': 'hand towel',
        'hand plain towel': 'hand towel',
        'hand plain towels': 'hand towel' ,
        'towel': 'towel',
        'towels': 'towel',
        'kitchen towel': 'kitchen towel',
        'kitchen towels': 'kitchen towel',
        'kitchen plain towel': 'kitchen towel',
        'face towel': 'face towel',
        'face towels': 'face towel',
        'face plain towel': 'face towel',
        'beach towel': 'beach towel',
        'beach towels': 'beach towel',
        'beach plain towel': 'beach towel',
        'bathmat towel': 'bath mat',
        'bathmat towels': 'bath mat',
        'bath mat towel': 'bath mat',
        'bath mat towels': 'bath mat',
        'bathmat': 'bath mat',
        'bath mat': 'bath mat',
        'beach mat': 'beach mat',
        'beach mats': 'beach mat',
        'mattress protector': 'mattress protector',
        'mattress protectors': 'mattress protector',
        'mattress topper': 'mattress topper',
        'mattress toppers': 'mattress topper',
        'bathrobe': 'bathrobe',
        'bathrobes': 'bathrobe',
        'pillow': 'pillow',
        'pillows': 'pillow',
        'pillowcase': 'pillow case',
        'pillowcases': 'pillow case',
        'pillow case': 'pillow case',
        'pillow cases': 'pillow case',
        'coverlet': 'coverlet',
        'coverlets': 'coverlet',
        'bed sheet': 'bed sheet',
        'bed sheets': 'bed sheet',
        'bedsheet': 'bed sheet',
        'bedsheets': 'bed sheet',
        'fitted sheet': 'fitted sheet',
        'fitted sheets': 'fitted sheet',
        'blanket': 'blanket',
        'blankets': 'blanket',
        'cushion': 'cushion',
        'cushions': 'cushion',
        'duvet': 'duvet',
        'duvets': 'duvet',
        'curtain': 'curtain',
        'curtains': 'curtain',
        'bundle': 'bundle',
        'bundles': 'bundle',
        'set': 'set',
        'sets': 'set',
        'slipper': 'slipper',
        'slippers': 'slipper',
        'apron': 'apron',
        'aprons': 'apron',
        'pouch': 'pouch',
        'pouches': 'pouch',
        'mat': 'mat',
        'mats': 'mat',
        'throw': 'coverlet',
        'throws': 'coverlet',
        'bag': 'bag',
        'bags': 'bag',
        'candle': 'candle',
        'candles': 'candle',
        'robe': 'bathrobe',
        'robes': 'bathrobe',
    }
    match = pattern.search(str(text).lower())
    if match:
        return normalize_map.get(match.group(1).lower(), match.group(1).lower())
    return None

def handle_category(df):
    """
    Handle the 'Lineitem name' column in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.DataFrame: The DataFrame with the 'Lineitem name' column processed.
    """
    df['category'] = df['Lineitem name'].apply(extract_category)
    
    df['category'] = df['category'].fillna('other')
    categories_to_replace = ['beach mat', 'other', 'curtain', 'bag', 'bundle', 'pouch', 'candle']
    df['category'] = df['category'].apply(lambda x: 'other' if x in categories_to_replace else x)

    
    return df