######################################## IMPORTS #######################################################################
import pandas as pd
import numpy as np
import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html
import plotly
import plotly.graph_objs as go
import time

######################################## APP INITIALIZATION#############################################################
app = dash.Dash(__name__)

######################################## READING DATA ##################################################################
start = time.perf_counter()
heatmap_df = pd.read_excel("../data/df_heatmap.xlsx", index_col=0)  # Read the Heatmap Data
market_df = pd.read_excel('../data/df_market_data.xlsx')  # Read the Market Data
market_df.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
dates = market_df["timestamp"].tolist()  # Save the dates
i = 0  # Counter
end = time.perf_counter()
print(f"Time elapsed in milliseconds for reading the data: {round((end - start) * 1000)}")
print()

######################################## FUNCTIONS #####################################################################
def create_bins(min_value, step_percentage, max_percentage):
    """Create the bins and labels for a continuous valued heatmap

    Args:
        min_value (float): the min value you want the bins to be based off
        step_percentage (float): the rate of increase to your min valuse
        max_percentage (float): the maximum rate of increase

    Raises:
        ValueError: raise error if the max percentage is bigger than the step percentage

    Returns:
        bins, labels (tuple): return the bins, labels to be inputted to the heatmap
    """

    if (step_percentage > max_percentage):
        raise ValueError('The step percentage must be less or equal to the max percentage.')

    current_value = min_value

    step = min_value * (step_percentage / 100)

    number_intervals = int(max_percentage // step_percentage)

    intervals = [current_value]

    for i in range(number_intervals):
        current_value = current_value + step
        intervals.append(current_value)

    intervals = np.array(intervals)

    for i in range(len(intervals)):
        if i == 0:
            intervals[i] -= 0.001 * abs(intervals[i])
        elif i == len(intervals) - 1:
            intervals[i] += 0.001 * abs(intervals[i])

    bins = pd.IntervalIndex.from_breaks(intervals)

    labels = np.arange(1, len(bins) + 1)

    return bins, labels

######################################## APP LAYOUT ####################################################################
app.layout = html.Div(
    [
        dcc.Graph(id = 'live-graph', animate = True),   # Initialize the graph
        dcc.Interval(
            id = 'graph-update',
            interval = 2 * 1000, # The interval time in milliseconds that is why we multiply 2 by 1000
            n_intervals = 0 # The value 0 means running indifnitely
        ),
    ]
)
  
######################################## APP CALLBACKS #################################################################
@app.callback(
    Output('live-graph', 'figure'), # output to the graph component 
    [ Input('graph-update', 'n_intervals') ] # The number of intervals that you want the graph to be updated on
)
  
def update_graph_scatter(n_intervals):
    """Updating the graph every second

    Args:
        n_intervals (int): The number of intervals that you want the graph to be updated on and 0 means indifinitely

    Returns:
        data, layout (dict): return a dictionary of the data and layouts and send it to the graph to be updated
    """
    
    start = time.perf_counter()

    #  Read the counter

    global i
    
    # Copy the data up to specific date based on the counter

    market_df_temp = market_df.iloc[:i+1].copy()

    binned_heatmap_df = heatmap_df.iloc[:,:i+1].copy()

    # Increase the counter which indicates to the time for the next second

    if i != len(dates):
        i+=1
    else:
        end = time.perf_counter()
        print(f"Time elapsed in milliseconds for Executing the graph: { round( (end-start) * 1000 ) }")
        return 
    
    # Find the global min and max based on the market dataframe 

    min_value= market_df_temp[["open","high","low","close"]].min().min()
    
    max_value= market_df_temp[["open","high","low","close"]].max().max()
    
    # Creat the bins and labels for the heatmap 
    
    bins,labels = create_bins(min_value = min_value,
                             step_percentage= 0.2,
                             max_percentage= 1.2
                            )
    
    
    # Binning the heatmap based on the bins and labels

    binned_heatmap_df['bins_range'] = pd.cut(binned_heatmap_df.index, bins=bins,include_lowest=True)


    # Create the binned dataframe and group by bins_range based on the sum

    binned_sum_df = binned_heatmap_df.groupby("bins_range").sum()
    
    # Saving the start and end step value in a list

    bins_step = [float(x) for x in str(bins[0])[1:-1].split(", ")]
    
    # Calculating the step value
    
    dy = bins_step[1]-bins_step[0]

    # Creating the heatmap graph

    trace1 = go.Heatmap(
        x = binned_sum_df.columns,
        z = binned_sum_df,
        x0 = 0.5,
        y0= min_value + 0.5,
        dy = dy,
        colorscale = 'Blues'

    )
    
    # Creating the candle stick graph

    trace2 = go.Candlestick(x=market_df_temp['timestamp'],
                    open=market_df_temp['open'],
                    high=market_df_temp['high'],
                    low=market_df_temp['low'],
                    close=market_df_temp['close'])

    # Append the graphs to the data list
    
    data = [trace1, trace2]

    # Create the layout

    layout = go.Layout(
        yaxis_range=[min_value-500,max_value+500],
        xaxis_range=["2021-04-03 19:33:00","2021-04-03 19:33:25"],
    )
    
    end = time.perf_counter()

    print(f"Time elapsed in milliseconds for Executing the graph: { round( (end-start) * 1000 ) }")

    # Return the data and layout and send it to the graph to plot it

    return {'data': data,
            'layout' : layout}

######################################## MAIN ########################################################################## 
if __name__ == '__main__':
    app.run_server()