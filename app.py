import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="Portfolio Optimization Lab",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Portfolio Optimization Laboratory")
st.caption("Brute Force • Monte Carlo • Efficient Frontier • Capital Market Line")

# ----------------------------------------------------
# Load Data
# ----------------------------------------------------

GITHUB_FILE = "https://raw.githubusercontent.com/srm36524/Optimization/main/Yearly%20Returns.xlsx"

@st.cache_data
def load_returns():

    df = pd.read_excel(GITHUB_FILE)

    return df

try:

    df = load_returns()

except Exception as e:

    st.error("Unable to load Excel file from GitHub.")

    st.exception(e)

    st.stop()

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

st.sidebar.title("⚙ Portfolio Settings")

# Years

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

    st.sidebar.error("Invalid year selection.")

    st.stop()

# ----------------------------------------------------
# Return Method
# ----------------------------------------------------

return_method = st.sidebar.radio(

    "Average Return",

    [

        "Arithmetic Mean",

        "Geometric Mean"

    ]

)

# ----------------------------------------------------
# Optimization Method
# ----------------------------------------------------

optimizer = st.sidebar.selectbox(

    "Optimization Method",

    [

        "Brute Force",

        "Monte Carlo",

        "Both"

    ]

)

# ----------------------------------------------------
# Weight Interval
# ----------------------------------------------------

weight_interval = st.sidebar.selectbox(

    "Weight Interval",

    [

        10,

        5,

        2,

        1

    ],

    index=3

)

# ----------------------------------------------------
# Monte Carlo
# ----------------------------------------------------

monte_carlo = st.sidebar.slider(

    "Monte Carlo Portfolios",

    1000,

    100000,

    20000,

    step=1000

)

# ----------------------------------------------------
# Risk Free Rate
# ----------------------------------------------------

risk_free = st.sidebar.number_input(

    "Risk Free Rate (%)",

    0.0,

    20.0,

    6.0,

    step=0.25

)/100

# ----------------------------------------------------
# Constraints
# ----------------------------------------------------

required_return = st.sidebar.number_input(

    "Required Return (%)",

    -20.0,

    50.0,

    10.0,

    step=0.5

)/100

maximum_risk = st.sidebar.number_input(

    "Maximum Risk (%)",

    0.0,

    100.0,

    30.0,

    step=0.5

)/100

min_weight = st.sidebar.slider(

    "Minimum Weight (%)",

    0,

    100,

    0

)

max_weight = st.sidebar.slider(

    "Maximum Weight (%)",

    0,

    100,

    100

)

allow_short = st.sidebar.checkbox(

    "Allow Short Selling",

    value=False

)

show_all = st.sidebar.checkbox(

    "Show All Portfolios",

    value=False

)

# ----------------------------------------------------
# Filter Data
# ----------------------------------------------------

returns_df = df[

    (df[year_column] >= start_year) &
    (df[year_column] <= end_year)

].copy()

returns = returns_df.iloc[:,1:].copy()

returns = returns.replace("%","",regex=True)

returns = returns.astype(float)

returns = returns/100

assets = returns.columns.tolist()

# ----------------------------------------------------
# Statistics
# ----------------------------------------------------

if return_method == "Arithmetic Mean":

    mean_returns = returns.mean()

else:

    mean_returns = (

        (1+returns).prod()

        **

        (1/len(returns))

    )-1

cov_matrix = returns.cov()

correlation = returns.corr()

# ----------------------------------------------------
# Display
# ----------------------------------------------------

st.subheader("Selected Period")

st.write(f"**{start_year} - {end_year}**")

left,right = st.columns(2)

with left:

    st.markdown("### Annual Returns")

    st.dataframe(

        returns_df,

        use_container_width=True

    )

with right:

    st.markdown("### Average Returns")

    avg = pd.DataFrame({

        "Asset":assets,

        "Average Return (%)":

        (mean_returns*100).round(2)

    })

    st.dataframe(

        avg,

        use_container_width=True

    )

st.subheader("Covariance Matrix")

st.dataframe(

    cov_matrix,

    use_container_width=True

)

st.subheader("Correlation Matrix")

st.dataframe(

    correlation,

    use_container_width=True

)

# ----------------------------------------------------
# Helper Function
# ----------------------------------------------------

def portfolio_statistics(weights):

    port_return = np.dot(

        weights,

        mean_returns

    )

    variance = np.dot(

        weights.T,

        np.dot(cov_matrix,weights)

    )

    risk = np.sqrt(variance)

    sharpe = (

        port_return-risk_free

    )/risk if risk>0 else np.nan

    return port_return,risk,variance,sharpe
    # ----------------------------------------------------
# Brute Force Portfolio Generator
# ----------------------------------------------------

from itertools import combinations_with_replacement

@st.cache_data(show_spinner=False)
def generate_weight_combinations(n_assets, step_percent):

    """
    Generates all weight combinations
    that sum to 100%.
    Works for any number of assets.
    """

    step = step_percent

    units = int(100 / step)

    weights = []

    def recurse(remaining, assets_left, current):

        if assets_left == 1:

            weights.append(
                current + [remaining]
            )

            return

        for i in range(remaining + 1):

            recurse(
                remaining - i,
                assets_left - 1,
                current + [i]
            )

    recurse(units, n_assets, [])

    weights = np.array(weights)

    weights = weights * step / 100

    return weights

# ----------------------------------------------------
# Generate Brute Force Portfolios
# ----------------------------------------------------

def brute_force_optimizer():

    st.subheader("Brute Force Optimization")

    weight_matrix = generate_weight_combinations(

        len(assets),

        weight_interval

    )

    progress = st.progress(0)

    results = []

    total = len(weight_matrix)

    for i, weights in enumerate(weight_matrix):

        # Constraints

        if not allow_short:

            if np.any(weights < 0):
                continue

        if np.any(weights * 100 < min_weight):
            continue

        if np.any(weights * 100 > max_weight):
            continue

        port_return, risk, variance, sharpe = \
            portfolio_statistics(weights)

        if port_return < required_return:
            continue

        if risk > maximum_risk:
            continue

        row = {}

        for j, asset in enumerate(assets):

            row[asset] = round(
                weights[j] * 100,
                2
            )

        row["Return"] = port_return
        row["Risk"] = risk
        row["Variance"] = variance
        row["Sharpe"] = sharpe

        results.append(row)

        if i % 50 == 0:

            progress.progress(i / total)

    progress.empty()

    results = pd.DataFrame(results)

    if len(results) == 0:

        st.warning(
            "No feasible portfolios found."
        )

        return None

    results = results.sort_values(

        "Sharpe",

        ascending=False

    ).reset_index(drop=True)

    st.success(

        f"{len(results):,} feasible portfolios generated."

    )

    return results

# ----------------------------------------------------
# Run Brute Force
# ----------------------------------------------------

brute_results = None

if optimizer in ["Brute Force", "Both"]:

    brute_results = brute_force_optimizer()

    if brute_results is not None:

        st.subheader("Top 10 Portfolios")

        display = brute_results.copy()

        display["Return"] = (
            display["Return"] * 100
        ).round(2)

        display["Risk"] = (
            display["Risk"] * 100
        ).round(2)

        display["Sharpe"] = display[
            "Sharpe"
        ].round(3)

        st.dataframe(

            display.head(10),

            use_container_width=True

        )

        max_sharpe = brute_results.iloc[0]

        min_variance = brute_results.loc[
            brute_results["Risk"].idxmin()
        ]

        st.subheader("Optimal Portfolios")

        c1, c2 = st.columns(2)

        with c1:

            st.metric(

                "Maximum Sharpe",

                f"{max_sharpe['Sharpe']:.3f}"

            )

            st.dataframe(

                max_sharpe.to_frame(),

                use_container_width=True

            )

        with c2:

            st.metric(

                "Minimum Risk",

                f"{min_variance['Risk']*100:.2f}%"

            )

            st.dataframe(

                min_variance.to_frame(),

                use_container_width=True

            )
# ----------------------------------------------------
# Monte Carlo Portfolio Optimizer
# ----------------------------------------------------

@st.cache_data(show_spinner=False)
def monte_carlo_optimizer(n_portfolios):

    np.random.seed(42)

    portfolios = []

    progress = st.progress(0)

    for i in range(n_portfolios):

        # Generate random weights

        if allow_short:

            w = np.random.randn(len(assets))
            w = w / np.sum(np.abs(w))

        else:

            w = np.random.random(len(assets))
            w = w / np.sum(w)

        # Apply weight constraints

        if np.any(w*100 < min_weight):
            continue

        if np.any(w*100 > max_weight):
            continue

        port_return, risk, variance, sharpe = \
            portfolio_statistics(w)

        if port_return < required_return:
            continue

        if risk > maximum_risk:
            continue

        row = {}

        for j, asset in enumerate(assets):
            row[asset] = round(w[j]*100,2)

        row["Return"] = port_return
        row["Risk"] = risk
        row["Variance"] = variance
        row["Sharpe"] = sharpe

        portfolios.append(row)

        if i % 200 == 0:
            progress.progress(i/n_portfolios)

    progress.empty()

    return pd.DataFrame(portfolios)

# ----------------------------------------------------
# Run Monte Carlo
# ----------------------------------------------------

mc_results = None

if optimizer in ["Monte Carlo","Both"]:

    st.subheader("Monte Carlo Optimization")

    mc_results = monte_carlo_optimizer(
        monte_carlo
    )

    if len(mc_results)==0:

        st.warning(
            "No feasible Monte Carlo portfolios."
        )

    else:

        st.success(
            f"{len(mc_results):,} feasible portfolios generated."
        )

        mc_results = mc_results.sort_values(
            "Sharpe",
            ascending=False
        ).reset_index(drop=True)

        mc_best = mc_results.iloc[0]

        mc_min = mc_results.loc[
            mc_results["Risk"].idxmin()
        ]

        c1,c2 = st.columns(2)

        with c1:

            st.metric(
                "Monte Carlo Max Sharpe",
                f"{mc_best['Sharpe']:.3f}"
            )

            st.dataframe(
                mc_best.to_frame(),
                use_container_width=True
            )

        with c2:

            st.metric(
                "Monte Carlo Minimum Risk",
                f"{mc_min['Risk']*100:.2f}%"
            )

            st.dataframe(
                mc_min.to_frame(),
                use_container_width=True
            )

# ----------------------------------------------------
# Comparison Table
# ----------------------------------------------------

if brute_results is not None and mc_results is not None:

    st.subheader("Brute Force vs Monte Carlo")

    comparison = pd.DataFrame({

        "Metric":[

            "Portfolios",

            "Maximum Sharpe",

            "Minimum Risk",

            "Highest Return"

        ],

        "Brute Force":[

            len(brute_results),

            round(
                brute_results.iloc[0]["Sharpe"],3
            ),

            round(
                brute_results["Risk"].min()*100,2
            ),

            round(
                brute_results["Return"].max()*100,2
            )

        ],

        "Monte Carlo":[

            len(mc_results),

            round(
                mc_results.iloc[0]["Sharpe"],3
            ),

            round(
                mc_results["Risk"].min()*100,2
            ),

            round(
                mc_results["Return"].max()*100,2
            )

        ]

    })

    st.dataframe(
        comparison,
        use_container_width=True
    )

# ----------------------------------------------------
# Dataset to Plot
# ----------------------------------------------------

if optimizer=="Brute Force":
    portfolios = brute_results

elif optimizer=="Monte Carlo":
    portfolios = mc_results

else:
    portfolios = pd.concat(
        [brute_results,mc_results],
        ignore_index=True
    )

if portfolios is not None:

    portfolios = portfolios.drop_duplicates()

    portfolios = portfolios.sort_values(
        "Risk"
    ).reset_index(drop=True)
# ==========================================================
# Efficient Frontier
# ==========================================================

st.header("📈 Efficient Frontier")

if portfolios is not None and len(portfolios) > 0:

    # ------------------------------------------------------
    # Efficient Frontier
    # ------------------------------------------------------

    frontier = []

    max_return = -999

    sorted_portfolios = portfolios.sort_values(
        ["Risk", "Return"]
    ).reset_index(drop=True)

    for i in range(len(sorted_portfolios)-1, -1, -1):

        row = sorted_portfolios.iloc[i]

        if row["Return"] >= max_return:

            frontier.append(row)

            max_return = row["Return"]

    frontier = pd.DataFrame(frontier)

    frontier = frontier.sort_values("Risk")

    # ------------------------------------------------------
    # Maximum Sharpe
    # ------------------------------------------------------

    max_sharpe = portfolios.loc[
        portfolios["Sharpe"].idxmax()
    ]

    # ------------------------------------------------------
    # Minimum Variance
    # ------------------------------------------------------

    min_variance = portfolios.loc[
        portfolios["Risk"].idxmin()
    ]

    # ------------------------------------------------------
    # Target Return Portfolio
    # ------------------------------------------------------

    target_return = portfolios[
        portfolios["Return"] >= required_return
    ]

    if len(target_return) > 0:

        target_return = target_return.loc[
            target_return["Risk"].idxmin()
        ]

    else:

        target_return = None

    # ------------------------------------------------------
    # Target Risk Portfolio
    # ------------------------------------------------------

    target_risk = portfolios[
        portfolios["Risk"] <= maximum_risk
    ]

    if len(target_risk) > 0:

        target_risk = target_risk.loc[
            target_risk["Return"].idxmax()
        ]

    else:

        target_risk = None

    # ======================================================
    # Capital Market Line
    # ======================================================

    x_cml = np.linspace(
        0,
        portfolios["Risk"].max()*1.15,
        100
    )

    slope = (

        max_sharpe["Return"] - risk_free

    ) / max_sharpe["Risk"]

    y_cml = risk_free + slope*x_cml

    # ======================================================
    # Plot
    # ======================================================

    fig = go.Figure()

    # ----------------------------------

    fig.add_trace(

        go.Scatter(

            x=portfolios["Risk"]*100,

            y=portfolios["Return"]*100,

            mode="markers",

            marker=dict(

                size=7,

                color=portfolios["Sharpe"],

                colorscale="Viridis",

                colorbar=dict(

                    title="Sharpe"

                ),

                opacity=0.75

            ),

            customdata=portfolios[assets].values,

            hovertemplate=

            "<b>Risk</b>: %{x:.2f}%<br>"+

            "<b>Return</b>: %{y:.2f}%<br>"+

            "<b>Sharpe</b>: %{marker.color:.3f}<br>"+

            "<extra></extra>",

            name="Portfolios"

        )

    )

    # ----------------------------------

    fig.add_trace(

        go.Scatter(

            x=frontier["Risk"]*100,

            y=frontier["Return"]*100,

            mode="lines",

            line=dict(

                color="black",

                width=4

            ),

            name="Efficient Frontier"

        )

    )

    # ----------------------------------

    fig.add_trace(

        go.Scatter(

            x=x_cml*100,

            y=y_cml*100,

            mode="lines",

            line=dict(

                color="orange",

                dash="dash",

                width=3

            ),

            name="Capital Market Line"

        )

    )

    # ----------------------------------

    fig.add_trace(

        go.Scatter(

            x=[max_sharpe["Risk"]*100],

            y=[max_sharpe["Return"]*100],

            mode="markers",

            marker=dict(

                size=18,

                color="red",

                symbol="star"

            ),

            name="Maximum Sharpe"

        )

    )

    # ----------------------------------

    fig.add_trace(

        go.Scatter(

            x=[min_variance["Risk"]*100],

            y=[min_variance["Return"]*100],

            mode="markers",

            marker=dict(

                size=18,

                color="green",

                symbol="diamond"

            ),

            name="Minimum Variance"

        )

    )

    # ----------------------------------

    if target_return is not None:

        fig.add_trace(

            go.Scatter(

                x=[target_return["Risk"]*100],

                y=[target_return["Return"]*100],

                mode="markers",

                marker=dict(

                    size=16,

                    color="purple",

                    symbol="square"

                ),

                name="Target Return"

            )

        )

    # ----------------------------------

    if target_risk is not None:

        fig.add_trace(

            go.Scatter(

                x=[target_risk["Risk"]*100],

                y=[target_risk["Return"]*100],

                mode="markers",

                marker=dict(

                    size=16,

                    color="brown",

                    symbol="cross"

                ),

                name="Target Risk"

            )

        )

    # ======================================================

    fig.update_layout(

        title="Portfolio Optimization",

        template="plotly_white",

        height=750,

        xaxis_title="Risk (%)",

        yaxis_title="Expected Return (%)",

        legend=dict(

            orientation="h",

            y=1.03

        )

    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="frontier_chart"
    )

# ==========================================================
# Summary Cards
# ==========================================================

st.header("Portfolio Summary")

c1,c2,c3,c4 = st.columns(4)

c1.metric(
    "Portfolios",
    f"{len(portfolios):,}"
)

c2.metric(
    "Maximum Sharpe",
    f"{max_sharpe['Sharpe']:.3f}"
)

c3.metric(
    "Minimum Risk",
    f"{min_variance['Risk']*100:.2f}%"
)

c4.metric(
    "Maximum Return",
    f"{portfolios['Return'].max()*100:.2f}%"
)

# ==========================================================
# Portfolio Comparison
# ==========================================================

summary = pd.DataFrame({

    "Portfolio":[

        "Maximum Sharpe",

        "Minimum Variance",

        "Target Return",

        "Target Risk"

    ],

    "Return (%)":[

        round(max_sharpe["Return"]*100,2),

        round(min_variance["Return"]*100,2),

        round(target_return["Return"]*100,2)
        if target_return is not None else np.nan,

        round(target_risk["Return"]*100,2)
        if target_risk is not None else np.nan

    ],

    "Risk (%)":[

        round(max_sharpe["Risk"]*100,2),

        round(min_variance["Risk"]*100,2),

        round(target_return["Risk"]*100,2)
        if target_return is not None else np.nan,

        round(target_risk["Risk"]*100,2)
        if target_risk is not None else np.nan

    ],

    "Sharpe":[

        round(max_sharpe["Sharpe"],3),

        round(min_variance["Sharpe"],3),

        round(target_return["Sharpe"],3)
        if target_return is not None else np.nan,

        round(target_risk["Sharpe"],3)
        if target_risk is not None else np.nan

    ]

})

st.dataframe(
    summary,
    use_container_width=True
)
# ==========================================================
# ADVANCED ANALYTICS
# ==========================================================

st.header("📊 Advanced Analytics")

# ----------------------------------------------------------
# Allocation Charts
# ----------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    fig = px.pie(
        values=[max_sharpe[a] for a in assets],
        names=assets,
        title="Maximum Sharpe Allocation",
        hole=0.45
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="pie_max_sharpe"
    )

with col2:

    fig = px.pie(
        values=[min_variance[a] for a in assets],
        names=assets,
        title="Minimum Variance Allocation",
        hole=0.45
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="pie_min_var"
    )

# ----------------------------------------------------------
# Allocation Bar Chart
# ----------------------------------------------------------

alloc = pd.DataFrame({

    "Asset":assets,

    "Maximum Sharpe":[
        max_sharpe[a] for a in assets
    ],

    "Minimum Variance":[
        min_variance[a] for a in assets
    ]

})

fig = go.Figure()

fig.add_bar(
    x=alloc["Asset"],
    y=alloc["Maximum Sharpe"],
    name="Maximum Sharpe"
)

fig.add_bar(
    x=alloc["Asset"],
    y=alloc["Minimum Variance"],
    name="Minimum Variance"
)

fig.update_layout(
    barmode="group",
    title="Portfolio Allocation Comparison"
)

st.plotly_chart(
    fig,
    use_container_width=True,
    key="allocation_bar"
)

# ----------------------------------------------------------
# Correlation Heatmap
# ----------------------------------------------------------

fig = px.imshow(
    correlation,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    title="Correlation Matrix"
)

st.plotly_chart(
    fig,
    use_container_width=True,
    key="corr_heatmap"
)

# ----------------------------------------------------------
# Portfolio Return Distribution
# ----------------------------------------------------------

fig = px.histogram(

    portfolios,

    x="Return",

    nbins=40,

    title="Portfolio Return Distribution"

)

fig.update_xaxes(
    tickformat=".1%"
)

st.plotly_chart(
    fig,
    use_container_width=True,
    key="return_hist"
)

# ----------------------------------------------------------
# Portfolio Risk Distribution
# ----------------------------------------------------------

fig = px.histogram(

    portfolios,

    x="Risk",

    nbins=40,

    title="Portfolio Risk Distribution"

)

fig.update_xaxes(
    tickformat=".1%"
)

st.plotly_chart(
    fig,
    use_container_width=True,
    key="risk_hist"
)

# ----------------------------------------------------------
# Risk Contribution
# ----------------------------------------------------------

weights = np.array([
    max_sharpe[a]/100
    for a in assets
])

cov = cov_matrix.values

portfolio_variance = weights.T @ cov @ weights

marginal = cov @ weights

contribution = (

    weights * marginal

) / portfolio_variance

risk_table = pd.DataFrame({

    "Asset":assets,

    "Risk Contribution (%)":

    contribution*100

})

fig = px.bar(

    risk_table,

    x="Asset",

    y="Risk Contribution (%)",

    title="Risk Contribution (Maximum Sharpe)"

)

st.plotly_chart(

    fig,

    use_container_width=True,

    key="risk_contribution"

)

# ----------------------------------------------------------
# Asset Statistics
# ----------------------------------------------------------

stats = pd.DataFrame({

    "Mean Return (%)":

    mean_returns*100,

    "Volatility (%)":

    returns.std()*100,

    "Minimum (%)":

    returns.min()*100,

    "Maximum (%)":

    returns.max()*100

})

st.subheader("Asset Statistics")

st.dataframe(
    stats.round(2),
    use_container_width=True
)

# ----------------------------------------------------------
# Growth of ₹100
# ----------------------------------------------------------

growth = (1+returns).cumprod()*100

fig = px.line(

    growth,

    x=returns_df[year_column],

    y=growth.columns,

    title="Growth of ₹100"

)

st.plotly_chart(

    fig,

    use_container_width=True,

    key="growth"

)

# ----------------------------------------------------------
# Ranking
# ----------------------------------------------------------

st.subheader("Top Portfolios")

ranking = portfolios.copy()

ranking["Return"] = (
    ranking["Return"]*100
).round(2)

ranking["Risk"] = (
    ranking["Risk"]*100
).round(2)

ranking["Sharpe"] = ranking["Sharpe"].round(3)

st.dataframe(

    ranking.sort_values(
        "Sharpe",
        ascending=False
    ).head(20),

    use_container_width=True

)

# ----------------------------------------------------------
# Download CSV
# ----------------------------------------------------------

csv = portfolios.to_csv(
    index=False
).encode()

st.download_button(

    "📥 Download Results",

    csv,

    "portfolio_results.csv",

    "text/csv"

)

# ----------------------------------------------------------
# Optional Portfolio Table
# ----------------------------------------------------------

if show_all:

    st.subheader("All Portfolios")

    st.dataframe(

        ranking,

        height=600,

        use_container_width=True

    )
# ==========================================================
# SCIPY OPTIMIZATION
# ==========================================================

st.header("🧮 Mathematical Optimization (SciPy)")

# ----------------------------------------------------------
# Sidebar
# ----------------------------------------------------------

solver = st.sidebar.selectbox(

    "Mathematical Optimizer",

    [

        "Maximum Sharpe",

        "Minimum Variance",

        "Target Return",

        "Risk Parity"

    ]

)

# ----------------------------------------------------------
# Bounds
# ----------------------------------------------------------

if allow_short:

    bounds = [(-1,1)] * len(assets)

else:

    bounds = [(0,1)] * len(assets)

constraints = [

    {

        "type":"eq",

        "fun":lambda x: np.sum(x)-1

    }

]

initial = np.ones(len(assets))/len(assets)

# ----------------------------------------------------------
# Objective Functions
# ----------------------------------------------------------

def portfolio_return(w):

    return np.dot(

        mean_returns,

        w

    )

def portfolio_variance(w):

    return np.dot(

        w.T,

        np.dot(cov_matrix,w)

    )

def portfolio_risk(w):

    return np.sqrt(

        portfolio_variance(w)

    )

def sharpe_ratio(w):

    return (

        portfolio_return(w)-risk_free

    )/portfolio_risk(w)

# ----------------------------------------------------------
# Maximum Sharpe
# ----------------------------------------------------------

def objective_sharpe(w):

    return -sharpe_ratio(w)

# ----------------------------------------------------------
# Minimum Variance
# ----------------------------------------------------------

def objective_variance(w):

    return portfolio_variance(w)

# ----------------------------------------------------------
# Target Return
# ----------------------------------------------------------

def objective_target_return(w):

    return portfolio_variance(w)

# ----------------------------------------------------------
# Risk Parity
# ----------------------------------------------------------

def objective_risk_parity(w):

    sigma = portfolio_risk(w)

    mrc = np.dot(

        cov_matrix,

        w

    ) / sigma

    rc = w * mrc

    target = np.repeat(

        sigma/len(w),

        len(w)

    )

    return np.sum(

        (rc-target)**2

    )

# ----------------------------------------------------------
# Run Solver
# ----------------------------------------------------------

if solver=="Maximum Sharpe":

    result = minimize(

        objective_sharpe,

        initial,

        method="SLSQP",

        bounds=bounds,

        constraints=constraints

    )

elif solver=="Minimum Variance":

    result = minimize(

        objective_variance,

        initial,

        method="SLSQP",

        bounds=bounds,

        constraints=constraints

    )

elif solver=="Target Return":

    cons = constraints.copy()

    cons.append(

        {

            "type":"eq",

            "fun":lambda w:

            portfolio_return(w)-required_return

        }

    )

    result = minimize(

        objective_target_return,

        initial,

        method="SLSQP",

        bounds=bounds,

        constraints=cons

    )

else:

    result = minimize(

        objective_risk_parity,

        initial,

        method="SLSQP",

        bounds=bounds,

        constraints=constraints

    )

# ----------------------------------------------------------
# Display
# ----------------------------------------------------------

if result.success:

    weights = result.x

    ret = portfolio_return(weights)

    risk = portfolio_risk(weights)

    sharpe = sharpe_ratio(weights)

    st.success("Optimization Successful")

    summary = pd.DataFrame({

        "Asset":assets,

        "Weight (%)":

        np.round(weights*100,2)

    })

    col1,col2 = st.columns(2)

    with col1:

        st.dataframe(

            summary,

            use_container_width=True

        )

    with col2:

        st.metric(

            "Expected Return",

            f"{ret*100:.2f}%"

        )

        st.metric(

            "Risk",

            f"{risk*100:.2f}%"

        )

        st.metric(

            "Sharpe",

            f"{sharpe:.3f}"

        )

    # ----------------------------------

    fig = go.Figure()

    fig.add_trace(

        go.Pie(

            labels=assets,

            values=weights,

            hole=0.45

        )

    )

    fig.update_layout(

        title="Optimal Allocation"

    )

    st.plotly_chart(

        fig,

        use_container_width=True,

        key="scipy_pie"

    )

else:

    st.error(

        "Optimization Failed."

    )
    
