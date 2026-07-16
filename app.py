import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Portfolio Optimization Demo",
    layout="wide"
)

st.title("Portfolio Optimization using Efficient Frontier")
st.markdown("### Brute Force Optimization (1% Weight Intervals)")

# ----------------------------------------------------
# Load Data from GitHub
# ----------------------------------------------------

GITHUB_FILE = "https://raw.githubusercontent.com/srm36524/Optimization/main/Yearly%20Returns.xlsx"

@st.cache_data
def load_data():
    df = pd.read_excel(GITHUB_FILE)
    return df

try:
    df = load_data()
except Exception as e:
    st.error("Unable to load Excel from GitHub")
    st.error(e)
    st.stop()

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

st.sidebar.header("Portfolio Settings")

year_column = df.columns[0]

years = df[year_column].tolist()

start_year = st.sidebar.selectbox(
    "Start Year",
    years,
    index=0
)

end_year = st.sidebar.selectbox(
    "End Year",
    years,
    index=len(years)-1
)

if start_year > end_year:
    st.sidebar.error("Start year should be less than End year.")
    st.stop()

risk_free = st.sidebar.number_input(
    "Risk Free Rate (%)",
    min_value=0.0,
    max_value=20.0,
    value=6.0,
    step=0.25
)/100

return_method = st.sidebar.radio(
    "Return Method",
    [
        "Arithmetic Mean",
        "Geometric Mean"
    ]
)

show_all = st.sidebar.checkbox(
    "Show All Portfolios",
    value=False
)

# ----------------------------------------------------
# Filter Selected Years
# ----------------------------------------------------

df = df[
    (df[year_column] >= start_year) &
    (df[year_column] <= end_year)
]

returns = df.iloc[:,1:].copy()

returns = returns.replace("%","",regex=True)

returns = returns.astype(float)

returns = returns/100

asset_names = returns.columns.tolist()

st.subheader("Selected Data")

st.dataframe(df,use_container_width=True)

# ----------------------------------------------------
# Calculate Statistics
# ----------------------------------------------------

if return_method=="Arithmetic Mean":
    mean_returns=returns.mean()
else:
    mean_returns=((1+returns).prod()**(1/len(returns)))-1

cov_matrix=returns.cov()

st.subheader("Average Returns")

avg=pd.DataFrame({
    "Asset":asset_names,
    "Average Return (%)":(mean_returns*100).round(2)
})

st.dataframe(avg,use_container_width=True)

st.subheader("Covariance Matrix")

st.dataframe(cov_matrix,use_container_width=True)
# ----------------------------------------------------
# Generate All Portfolio Combinations
# ----------------------------------------------------

st.subheader("Generating Portfolio Combinations...")

results = []

progress = st.progress(0)

total_portfolios = 5151
counter = 0

for w1 in range(101):

    for w2 in range(101 - w1):

        w3 = 100 - w1 - w2

        weights = np.array([w1, w2, w3]) / 100

        # Expected Return
        portfolio_return = np.dot(weights, mean_returns)

        # Portfolio Variance
        portfolio_variance = np.dot(
            weights.T,
            np.dot(cov_matrix.values, weights)
        )

        # Portfolio Risk
        portfolio_risk = np.sqrt(portfolio_variance)

        # Sharpe Ratio
        if portfolio_risk > 0:
            sharpe = (portfolio_return - risk_free) / portfolio_risk
        else:
            sharpe = np.nan

        row = {}

        for i, asset in enumerate(asset_names):
            row[asset] = round(weights[i] * 100, 0)

        row["Return"] = portfolio_return
        row["Risk"] = portfolio_risk
        row["Variance"] = portfolio_variance
        row["Sharpe"] = sharpe

        results.append(row)

        counter += 1
        progress.progress(counter / total_portfolios)

progress.empty()

results = pd.DataFrame(results)

st.success(f"Generated {len(results):,} portfolios.")

# ----------------------------------------------------
# Portfolio Statistics
# ----------------------------------------------------

st.subheader("Portfolio Statistics")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Portfolios",
    f"{len(results):,}"
)

col2.metric(
    "Highest Return",
    f"{results['Return'].max()*100:.2f}%"
)

col3.metric(
    "Lowest Risk",
    f"{results['Risk'].min()*100:.2f}%"
)

col4.metric(
    "Maximum Sharpe",
    f"{results['Sharpe'].max():.3f}"
)

# ----------------------------------------------------
# Best Portfolios
# ----------------------------------------------------

max_sharpe = results.loc[
    results["Sharpe"].idxmax()
]

min_variance = results.loc[
    results["Risk"].idxmin()
]

col1, col2 = st.columns(2)

with col1:

    st.subheader("Maximum Sharpe Portfolio")

    st.dataframe(
        max_sharpe.to_frame(),
        use_container_width=True
    )

with col2:

    st.subheader("Minimum Variance Portfolio")

    st.dataframe(
        min_variance.to_frame(),
        use_container_width=True
    )

# ----------------------------------------------------
# Allocation Tables
# ----------------------------------------------------

st.subheader("Portfolio Allocations")

alloc1 = pd.DataFrame({
    "Asset": asset_names,
    "Weight (%)": [
        max_sharpe[a]
        for a in asset_names
    ]
})

alloc2 = pd.DataFrame({
    "Asset": asset_names,
    "Weight (%)": [
        min_variance[a]
        for a in asset_names
    ]
})

left, right = st.columns(2)

with left:
    st.markdown("### Maximum Sharpe Allocation")
    st.dataframe(
        alloc1,
        use_container_width=True
    )

with right:
    st.markdown("### Minimum Variance Allocation")
    st.dataframe(
        alloc2,
        use_container_width=True
    )
    # ----------------------------------------------------
# Efficient Frontier
# ----------------------------------------------------

st.subheader("Efficient Frontier")

# Sort portfolios by Risk
sorted_results = results.sort_values(
    by=["Risk", "Return"]
).reset_index(drop=True)

# Extract Efficient Frontier
frontier = []

best_return = -999999

for i in range(len(sorted_results)-1, -1, -1):

    row = sorted_results.iloc[i]

    if row["Return"] >= best_return:

        frontier.append(row)

        best_return = row["Return"]

frontier = pd.DataFrame(frontier)

frontier = frontier.sort_values(
    by="Risk"
)

# ----------------------------------------------------
# Plot
# ----------------------------------------------------

fig = go.Figure()

# All portfolios
fig.add_trace(

    go.Scatter(

        x=results["Risk"]*100,

        y=results["Return"]*100,

        mode="markers",

        name="Portfolios",

        marker=dict(

            size=7,

            color=results["Sharpe"],

            colorscale="Viridis",

            showscale=True,

            colorbar=dict(
                title="Sharpe"
            ),

            opacity=0.75

        ),

        customdata=results[asset_names].values,

        hovertemplate=
        "<b>Portfolio</b><br>"
        "Risk : %{x:.2f}%<br>"
        "Return : %{y:.2f}%<br>"
        "Sharpe : %{marker.color:.3f}<br><br>"
        + "<b>Weights</b><br>"
        + asset_names[0] + ": %{customdata[0]:.0f}%<br>"
        + asset_names[1] + ": %{customdata[1]:.0f}%<br>"
        + asset_names[2] + ": %{customdata[2]:.0f}%<extra></extra>"

    )

)

# Efficient Frontier
fig.add_trace(

    go.Scatter(

        x=frontier["Risk"]*100,

        y=frontier["Return"]*100,

        mode="lines",

        name="Efficient Frontier",

        line=dict(

            color="black",

            width=4

        )

    )

)

# Maximum Sharpe
fig.add_trace(

    go.Scatter(

        x=[max_sharpe["Risk"]*100],

        y=[max_sharpe["Return"]*100],

        mode="markers",

        name="Maximum Sharpe",

        marker=dict(

            size=18,

            color="red",

            symbol="star"

        ),

        hovertemplate=

        "<b>Maximum Sharpe Portfolio</b><br>"

        f"Return : {max_sharpe['Return']*100:.2f}%<br>"

        f"Risk : {max_sharpe['Risk']*100:.2f}%<br>"

        f"Sharpe : {max_sharpe['Sharpe']:.3f}"

        "<extra></extra>"

    )

)

# Minimum Variance
fig.add_trace(

    go.Scatter(

        x=[min_variance["Risk"]*100],

        y=[min_variance["Return"]*100],

        mode="markers",

        name="Minimum Variance",

        marker=dict(

            size=18,

            color="green",

            symbol="diamond"

        ),

        hovertemplate=

        "<b>Minimum Variance Portfolio</b><br>"

        f"Return : {min_variance['Return']*100:.2f}%<br>"

        f"Risk : {min_variance['Risk']*100:.2f}%<br>"

        f"Sharpe : {min_variance['Sharpe']:.3f}"

        "<extra></extra>"

    )

)

fig.update_layout(

    height=750,

    title="Efficient Frontier (Brute Force 1% Optimization)",

    xaxis_title="Portfolio Risk (%)",

    yaxis_title="Expected Return (%)",

    hovermode="closest",

    template="plotly_white",

    legend=dict(

        orientation="h",

        y=1.05

    )

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ----------------------------------------------------
# Portfolio Summary
# ----------------------------------------------------

st.subheader("Portfolio Summary")

summary = pd.DataFrame({

    "Portfolio":[

        "Maximum Sharpe",

        "Minimum Variance"

    ],

    "Return (%)":[

        round(max_sharpe["Return"]*100,2),

        round(min_variance["Return"]*100,2)

    ],

    "Risk (%)":[

        round(max_sharpe["Risk"]*100,2),

        round(min_variance["Risk"]*100,2)

    ],

    "Sharpe":[

        round(max_sharpe["Sharpe"],3),

        round(min_variance["Sharpe"],3)

    ]

})

st.dataframe(
    summary,
    use_container_width=True
)
# ----------------------------------------------------
# Efficient Frontier
# ----------------------------------------------------

st.subheader("Efficient Frontier")

# Sort portfolios by Risk
sorted_results = results.sort_values(
    by=["Risk", "Return"]
).reset_index(drop=True)

# Extract Efficient Frontier
frontier = []

best_return = -999999

for i in range(len(sorted_results)-1, -1, -1):

    row = sorted_results.iloc[i]

    if row["Return"] >= best_return:

        frontier.append(row)

        best_return = row["Return"]

frontier = pd.DataFrame(frontier)

frontier = frontier.sort_values(
    by="Risk"
)

# ----------------------------------------------------
# Plot
# ----------------------------------------------------

fig = go.Figure()

# All portfolios
fig.add_trace(

    go.Scatter(

        x=results["Risk"]*100,

        y=results["Return"]*100,

        mode="markers",

        name="Portfolios",

        marker=dict(

            size=7,

            color=results["Sharpe"],

            colorscale="Viridis",

            showscale=True,

            colorbar=dict(
                title="Sharpe"
            ),

            opacity=0.75

        ),

        customdata=results[asset_names].values,

        hovertemplate=
        "<b>Portfolio</b><br>"
        "Risk : %{x:.2f}%<br>"
        "Return : %{y:.2f}%<br>"
        "Sharpe : %{marker.color:.3f}<br><br>"
        + "<b>Weights</b><br>"
        + asset_names[0] + ": %{customdata[0]:.0f}%<br>"
        + asset_names[1] + ": %{customdata[1]:.0f}%<br>"
        + asset_names[2] + ": %{customdata[2]:.0f}%<extra></extra>"

    )

)

# Efficient Frontier
fig.add_trace(

    go.Scatter(

        x=frontier["Risk"]*100,

        y=frontier["Return"]*100,

        mode="lines",

        name="Efficient Frontier",

        line=dict(

            color="black",

            width=4

        )

    )

)

# Maximum Sharpe
fig.add_trace(

    go.Scatter(

        x=[max_sharpe["Risk"]*100],

        y=[max_sharpe["Return"]*100],

        mode="markers",

        name="Maximum Sharpe",

        marker=dict(

            size=18,

            color="red",

            symbol="star"

        ),

        hovertemplate=

        "<b>Maximum Sharpe Portfolio</b><br>"

        f"Return : {max_sharpe['Return']*100:.2f}%<br>"

        f"Risk : {max_sharpe['Risk']*100:.2f}%<br>"

        f"Sharpe : {max_sharpe['Sharpe']:.3f}"

        "<extra></extra>"

    )

)

# Minimum Variance
fig.add_trace(

    go.Scatter(

        x=[min_variance["Risk"]*100],

        y=[min_variance["Return"]*100],

        mode="markers",

        name="Minimum Variance",

        marker=dict(

            size=18,

            color="green",

            symbol="diamond"

        ),

        hovertemplate=

        "<b>Minimum Variance Portfolio</b><br>"

        f"Return : {min_variance['Return']*100:.2f}%<br>"

        f"Risk : {min_variance['Risk']*100:.2f}%<br>"

        f"Sharpe : {min_variance['Sharpe']:.3f}"

        "<extra></extra>"

    )

)

fig.update_layout(

    height=750,

    title="Efficient Frontier (Brute Force 1% Optimization)",

    xaxis_title="Portfolio Risk (%)",

    yaxis_title="Expected Return (%)",

    hovermode="closest",

    template="plotly_white",

    legend=dict(

        orientation="h",

        y=1.05

    )

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ----------------------------------------------------
# Portfolio Summary
# ----------------------------------------------------

st.subheader("Portfolio Summary")

summary = pd.DataFrame({

    "Portfolio":[

        "Maximum Sharpe",

        "Minimum Variance"

    ],

    "Return (%)":[

        round(max_sharpe["Return"]*100,2),

        round(min_variance["Return"]*100,2)

    ],

    "Risk (%)":[

        round(max_sharpe["Risk"]*100,2),

        round(min_variance["Risk"]*100,2)

    ],

    "Sharpe":[

        round(max_sharpe["Sharpe"],3),

        round(min_variance["Sharpe"],3)

    ]

})

st.dataframe(
    summary,
    use_container_width=True
)
