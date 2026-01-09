import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
from pytz import timezone
import regex as re
import dash
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import plotly.figure_factory as ff
import plotly.io as pio
import plotly.subplots as sp
import plotly.colors as pc


def scatter_plot(df, x, y, color=None):
    """
    Plot interactive scatter plot using Plotly in Streamlit.
    """
    fig = px.scatter(df, x=x, y=y, color=color, title=f'{y} vs {x}')
    st.plotly_chart(fig, use_container_width=True)


def line_plot(df, x, y):
    """
    Create a line plot using Plotly Express.
    """
    fig = px.line(df, x=x, y=y, title="Interactive Time Series Analysis")
    st.plotly_chart(fig, use_container_width=True)

def box_plot(df, x):
    """
    Create a box plot using Plotly Express.
    """
    # Create a box plot using Plotly
    fig = go.Figure()
    fig.add_trace(go.Box(x=df[x], name=x))

    fig.update_layout(
        title_text=f"Boxplot of {x}",
        showlegend=False,
        height=500,
        width=600
    )

    st.plotly_chart(fig, use_container_width=True)

def histogram(df, x):
    """
    Create a histogram plot using Plotly Express.
    """
    # Create a box plot using Plotly
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df[x], name=x))

    fig.update_layout(
        title_text=f"Histogram of {x}",
        showlegend=False,
        height=500,
        width=600
    )

    st.plotly_chart(fig, use_container_width=True)

def pie_chart(df, x):
    """
    Create a pie chart using Plotly Express.
    """
    # Create a pie chart using Plotly
    fig = go.Figure()
    fig.add_trace(go.Pie(labels=df[x].value_counts().index, values=df[x].value_counts().values))

    fig.update_layout(
        title_text=f"Pie Chart of {x}",
        showlegend=False,
        height=500,
        width=600
    )

    st.plotly_chart(fig, use_container_width=True)


def bar_plot(df, x, y):
    """
    Create a bar plot using Plotly Express.
    """
    # Create a bar plot using Plotly
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df[x], y=df[y], name=x))

    fig.update_layout(
        title_text=f"Bar Plot of {x} vs {y}",
        showlegend=False,
        height=500,
        width=600
    )
    st.plotly_chart(fig, use_container_width=True)


    



def correlation_matrix(corr, method='pearson'):
    """
    Create a correlation matrix using Seaborn and Matplotlib.
    """
    fig = ff.create_annotated_heatmap(
    z=corr.values, 
    x=list(corr.columns), 
    y=list(corr.index), 
    annotation_text=corr.round(2).values, 
    colorscale="tempo",
    showscale=True
    )

    fig.update_layout(title="Interactive Correlation Matrix", height=600, width=700)

    st.plotly_chart(fig, use_container_width=True)


def top_20_BSP(df):
    """
        Create a bar chart for the top 20 best-selling products.
        :param df: A pandas DataFrame containing 'Lineitem name' and 'Lineitem quantity' columns.
    """
    # Validate required columns
    if 'Lineitem name' not in df.columns or 'Lineitem quantity' not in df.columns:
        raise ValueError("DataFrame must contain 'Lineitem name' and 'Lineitem quantity' columns.")

    try:
        # Ensure 'Lineitem name' and 'Lineitem quantity' are lists or iterable
        df['Lineitem name'] = df['Lineitem name'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
        df['Lineitem quantity'] = pd.to_numeric(df['Lineitem quantity'], errors='coerce')

        # Explode columns and concatenate them
        exploded_names = df.explode('Lineitem name')
        exploded_quantities = exploded_names.explode('Lineitem quantity')  # Ensure both columns match after processing

        # Handle missing values (drop rows with NaN in 'Lineitem quantity')
        exploded_quantities = exploded_quantities.dropna(subset=['Lineitem quantity'])

        # Convert 'Lineitem quantity' to numeric
        exploded_quantities['Lineitem quantity'] = pd.to_numeric(exploded_quantities['Lineitem quantity'], errors='coerce')

        # Group by product name and calculate the total quantity
        top_products = exploded_quantities.groupby('Lineitem name')['Lineitem quantity'].sum().sort_values(ascending=False).head(20)

        # Plot the data
        fig = px.bar(
            top_products,
            x=top_products.values,
            y=top_products.index,
            orientation='h',
            title="Top 20 Best-Selling Products",
            labels={'x': 'Number of Orders', 'y': 'Product Name'},
            text=top_products.values,
        )

        # Update layout for better visualization
        fig.update_layout(
            xaxis_title="Number of Orders",
            yaxis_title="Product Name",
            height=600,
            width=900,
            yaxis={'categoryorder': 'total ascending'}
        )
        fig.update_traces(marker_color='royalblue', textposition='outside')

        # Display the chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")


def seasonal(df):
    """
    Perform seasonal analysis and create multiple subplots for visualization.
    :param df: A pandas DataFrame with 'year', 'season', 'Name', 'Total', and 'year_month' columns.
    """
    # Validate required columns
    required_columns = {'year', 'season', 'Name', 'Total', 'year_month'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    try:
        # Ensure 'season' is categorical
        if not pd.api.types.is_categorical_dtype(df['season']):
            df['season'] = pd.Categorical(df['season'])

        # Handle missing data in 'Total' and 'Name'
        df = df.dropna(subset=['Total', 'Name'])

        # Yearly orders and revenue
        yearly_orders = df.groupby('year').agg({
            'Name': 'count',
            'Total': 'sum'
        }).rename(columns={'Name': 'order_count'})

        # Seasonal analysis
        seasonal_analysis = df.groupby('season', observed=False).agg({
            'Name': 'count',
            'Total': ['mean', 'sum'],
        })

        # Monthly trends
        monthly_trend = df.groupby('year_month').agg({
            'Name': 'count',
            'Total': 'sum'
        })

        # Convert indexes to strings for plotting
        yearly_orders.index = yearly_orders.index.astype(str)
        seasonal_analysis.index = seasonal_analysis.index.astype(str)
        monthly_trend.index = monthly_trend.index.astype(str)

        # Create subplots
        fig = sp.make_subplots(
            rows=4, cols=1,
            subplot_titles=[
                "Total Revenue by Year",
                "Average Order Value by Season",
                "Total Revenue by Season",
                "Monthly Revenue Trend"
            ],
            vertical_spacing=0.15
        )

        # Plot yearly revenue
        fig.add_trace(
            go.Bar(x=yearly_orders.index, y=yearly_orders['Total'], marker_color="lightgreen", name="Total Revenue"),
            row=1, col=1
        )

        # Plot average order value by season
        fig.add_trace(
            go.Scatter(x=seasonal_analysis.index, y=seasonal_analysis['Total']['mean'],
                       mode='lines+markers', line=dict(color="orange"), name="Avg Order Value"),
            row=2, col=1
        )

        # Plot total revenue by season
        fig.add_trace(
            go.Scatter(x=seasonal_analysis.index, y=seasonal_analysis['Total']['sum'],
                       mode='lines+markers', line=dict(color="red"), name="Total Revenue"),
            row=3, col=1
        )

        # Plot monthly revenue trend
        fig.add_trace(
            go.Scatter(x=monthly_trend.index, y=monthly_trend['Total'],
                       mode='lines+markers', line=dict(color="blue"), name="Monthly Revenue"),
            row=4, col=1
        )

        # Update layout for the figure
        fig.update_layout(
            height=1000, width=1280,
            title="Revenue & Seasonal Analysis",
            showlegend=False,
            xaxis4=dict(title="Year-Month"),
            xaxis3=dict(title="Season"),
            xaxis2=dict(title="Season"),
            xaxis1=dict(title="Year"),
            yaxis1=dict(title="Revenue"),
            yaxis2=dict(title="Average Order Value"),
            yaxis3=dict(title="Total Revenue"),
            yaxis4=dict(title="Total Revenue"),
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")



def order_analysis(df):
    """
    Create an Order Size and Value Analysis plot using Plotly.
    :param df: A pandas DataFrame with 'item_count' and 'Total' columns.
    """
    # Validate required columns
    required_columns = {'item_count', 'Total'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    try:
        # Handle missing or invalid data
        df = df.dropna(subset=['item_count', 'Total'])
        df['item_count'] = pd.to_numeric(df['item_count'], errors='coerce')
        df['Total'] = pd.to_numeric(df['Total'], errors='coerce')

        # Count distribution of items per order
        order_size_counts = df['item_count'].value_counts().sort_index()

        # Calculate average order value by item count
        avg_order_value = df.groupby('item_count')['Total'].mean()

        # Create subplots
        fig = sp.make_subplots(
            rows=2, cols=1,
            subplot_titles=["Distribution of Items per Order", "Average Order Value by Number of Items"]
        )

        # Plot order size distribution
        fig.add_trace(
            go.Bar(
                x=order_size_counts.index,
                y=order_size_counts.values,
                name="Order Size Distribution",
                marker_color="royalblue"
            ),
            row=1, col=1
        )

        # Plot average order value
        fig.add_trace(
            go.Scatter(
                x=avg_order_value.index,
                y=avg_order_value.values,
                mode='lines+markers',
                name="Average Order Value",
                line=dict(color="orange"),
                marker=dict(size=8)
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            height=800, width=1280,
            title_text="Order Size and Value Analysis",
            xaxis1=dict(title="Number of Items in Order"),
            yaxis1=dict(title="Number of Orders"),
            xaxis2=dict(title="Number of Items in Order"),
            yaxis2=dict(title="Average Order Value"),
            showlegend=False
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")

def top_20_HPI(df):
    """
    Identify and visualize the top 20 products with the highest price increases year-over-year.
    :param df: A pandas DataFrame containing the required columns.
    """

    # Validate required columns
    required_columns = {'season', 'year', 'year_month', 'Shipping Province Name', 'Created at', 'Lineitem price', 'Lineitem name'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    # Ensure 'Created at' column is datetime
    df['Created at'] = pd.to_datetime(df['Created at'], errors='coerce')

    # Flatten the product and price data
    product_prices = []
    for idx, row in df.iterrows():
        if isinstance(row['Lineitem price'], list) and isinstance(row['Lineitem name'], list):
            for price, product in zip(row['Lineitem price'], row['Lineitem name']):
                product_prices.append({
                    'year': row['Created at'].year,
                    'product': product,
                    'price': price
                })

    product_df = pd.DataFrame(product_prices)

    # Convert 'price' to numeric and handle NaN values
    product_df['price'] = pd.to_numeric(product_df['price'], errors='coerce').fillna(0)

    # Calculate yearly average prices for each product
    yearly_prices = product_df.groupby(['year', 'product'])['price'].mean().reset_index()

    # Pivot table for yearly prices
    price_pivot = yearly_prices.pivot(index='product', columns='year', values='price').fillna(0)

    # Calculate year-over-year price changes
    yoy_changes = pd.DataFrame()
    for year in range(price_pivot.columns.min() + 1, price_pivot.columns.max() + 1):
        yoy = ((price_pivot[year] - price_pivot[year - 1]) / price_pivot[year - 1]) * 100
        yoy = yoy.replace([np.inf, -np.inf], np.nan)  # Replace infinite values with NaN
        yoy_changes[f'{year}_YoY%'] = yoy

    # Identify the latest year for YoY analysis
    latest_year = f'{price_pivot.columns.max()}_YoY%'
    if latest_year not in yoy_changes.columns:
        raise ValueError(f"No YoY data available for the latest year: {price_pivot.columns.max()}")

    # Get the top 20 products with the highest price increases
    top_20_increases = yoy_changes[latest_year].sort_values(ascending=False).head(20)

    # Plot the data using Plotly
    fig = px.bar(
        x=top_20_increases.values,
        y=top_20_increases.index,
        orientation="h",
        title=f"Top 20 Products with Highest Price Increases in {price_pivot.columns.max()}",
        labels={"x": "Price Increase (%)", "y": "Product"},
        color=top_20_increases.values,
        color_continuous_scale="Reds"
    )

    fig.update_layout(
        xaxis_title="Price Increase (%)",
        yaxis_title="Product",
        height=600,
        width=900,
        yaxis=dict(categoryorder="total ascending")
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # except Exception as e:
    #     st.error(f"An error occurred during processing: {e}")


def holiday_analysis(df):
    """
    Create a holiday analysis plot using Plotly.
    :param df: A pandas DataFrame with 'is_holiday', 'Lineitem name', 'Lineitem price', 'holiday_name', and 'Created at' columns.
    """
    # Validate required columns
    required_columns = {'is_holiday', 'Lineitem name', 'Lineitem price', 'holiday_name', 'Created at'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    # Ensure 'is_holiday' is boolean or convertible to boolean
    if not pd.api.types.is_bool_dtype(df['is_holiday']):
        try:
            df['is_holiday'] = df['is_holiday'].astype(bool)
        except Exception:
            raise ValueError("'is_holiday' column must be of boolean type or convertible to boolean.")

    # Filter holiday orders
    if not df['is_holiday'].any():
        st.info("No holiday data available.")
        return

    holiday_orders = df[df['is_holiday']].copy()
    if holiday_orders.empty:
        st.info("No holiday data available.")
        return

    # Flatten holiday data
    holiday_items = []
    for _, row in holiday_orders.iterrows():
        items = row['Lineitem name']
        prices = row['Lineitem price']
        holiday = row['holiday_name']
        date = pd.to_datetime(row['Created at'], errors='coerce')

        # Ensure valid data
        if not isinstance(items, list) or not isinstance(prices, list):
            continue

        for item, price in zip(items, prices):
            holiday_items.append({
                'holiday': holiday,
                'item': item.strip(),
                'price': pd.to_numeric(price, errors='coerce'),
                'date': date
            })

    # Create a DataFrame for holiday items
    holiday_items_df = pd.DataFrame(holiday_items).dropna()
    
    # Aggregate holiday data
    holiday_analysis = holiday_items_df.groupby(['holiday', 'item']).agg({
        'item': 'count',
        'price': ['mean', 'sum']
    }).round(2)

    # Flatten MultiIndex columns and rename them
    holiday_analysis.columns = ['quantity_sold', 'avg_price', 'total_revenue']
    holiday_analysis = holiday_analysis.reset_index()

    # Top 10 holiday items
    top_holiday_items = holiday_items_df.groupby('item')['item'].count().sort_values(ascending=False).head(10)
    if top_holiday_items.empty:
        st.info("No data available for top holiday items.")
    else:
        fig1 = px.bar(
            x=top_holiday_items.index,
            y=top_holiday_items.values,
            title="Top 10 Items Sold During Holidays",
            labels={"x": "Item Name", "y": "Quantity Sold"},
            color=top_holiday_items.values,
            color_continuous_scale="Blues"
        )
        fig1.update_layout(xaxis_tickangle=-45, height=500, width='auto')
        st.plotly_chart(fig1, use_container_width=True)

    # Holiday sales summary
    holiday_sales = holiday_items_df.groupby('holiday')['item'].count()
    if holiday_sales.empty:
        st.info("No data available for holiday sales.")
    else:
        fig2 = px.bar(
            x=holiday_sales.index,
            y=holiday_sales.values,
            title="Number of Items Sold by Holiday",
            labels={"x": "Holiday", "y": "Number of Items"},
            color=holiday_sales.values,
            color_continuous_scale="Oranges"
        )
        fig2.update_layout(xaxis_tickangle=-45, height=500, width='auto')
        st.plotly_chart(fig2, use_container_width=True)


def seasonal_Revenue(df):
    """
    Create a seasonal revenue plot using Plotly.
    :param df: A pandas DataFrame with 'year', 'season', and 'Total' columns.
    """
    # Validate required columns
    required_columns = {'year', 'season', 'Total'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    try:
        # Ensure 'season' is categorical
        if not pd.api.types.is_categorical_dtype(df['season']):
            df['season'] = pd.Categorical(df['season'])

        # Handle missing or invalid data
        df = df.dropna(subset=['year', 'season', 'Total'])
        df['Total'] = pd.to_numeric(df['Total'], errors='coerce')

        # Group data by year and season
        seasonal_orders = df.groupby(['year', 'season'], observed=False)['Total'].sum().unstack()
        seasonal_orders.index = seasonal_orders.index.astype(str)

        if seasonal_orders.empty:
            st.info("No data available to plot seasonal revenue patterns.")
            return

        # Create bar plot
        fig = px.bar(
            seasonal_orders,
            x=seasonal_orders.index,
            y=seasonal_orders.columns,
            title="Seasonal Revenue Patterns by Year",
            labels={"value": "Total Revenue", "variable": "Season", "x": "Year"},
            barmode="group"
        )

        # Update layout
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Total Revenue",
            legend_title="Season",
            height=600,
            width=900
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")

def plot_demand_forecast(test, forecast, item):
    """
    Plot the forecasted demand against the actual demand for a specific item.
    
    Args: 
        test (pd.DataFrame): Actual demand data for the test set.
        forecast (pd.Series or list): Forecasted demand data.
        item (str): The item for which the forecast is made.
    """

    print("Attempt to plot demand forecast")
    df_item = test[test['category'] == item].copy()
    df_item = df_item['Lineitem quantity'].resample('D').sum()
    df_item = pd.DataFrame(df_item)


    plt.figure(figsize=(16, 6))
    sns.lineplot(x=df_item.index, y=df_item['Lineitem quantity'], label='Actual Demand', color='blue')
    print("1")
    sns.lineplot(x=df_item.index, y=forecast['predicted_mean'], label='Forecasted Demand', color='orange', linestyle='--')
    print("2")

    plt.title(f"Demand Forecast for Item: {item}")
    plt.xlabel("Time Index")  # Consider changing to a column like 'date' if available
    plt.ylabel("Quantity")
    plt.legend()
    plt.grid()
    print(3)
    
    # Save the plot before displaying it
    plt.savefig(f'imgs/demand_forecast_{item}.png')
    plt.show()
