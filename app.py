import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Efficient Frontier Demo",
    layout="wide"
)

st.title("Efficient Frontier using Brute Force (1% Intervals) by SRM. This optimization model covers returns of Equity, Gold and Debt for the last 26 years")

# ---------------------------------------------------------
# Load Excel directly from GitHub
# ---------------------------------------------------------

github_url = "https://raw.githubusercontent.com/srm36524/Optimization/main/Yearly%20Returns.xlsx"

try:
    df = pd.read_excel(github_url)
except Exception as e:
    st.error(f"Unable to load Excel file.\n\n{e}")
    st.stop()

st.success("Data loaded from GitHub")

# ---------------------------------------------------------
# Prepare Returns
# ---------------------------------------------------------

returns = df.iloc[:,1:].copy()

returns = returns.replace("%","",regex=True)

returns = returns.astype(float)

returns = returns/100

asset_names = returns.columns.tolist()

st.subheader("Yearly Returns")

st.dataframe(df)

# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

mean_returns = returns.mean()

cov_matrix = returns.cov()

st.subheader("Average Annual Returns")

avg = pd.DataFrame({
    "Asset":asset_names,
    "Average Return (%)":(mean_returns*100).round(2)
})

st.dataframe(avg,use_container_width=True)

st.subheader("Covariance Matrix")

st.dataframe(cov_matrix)

# ---------------------------------------------------------
# Risk Free Rate
# ---------------------------------------------------------

risk_free = st.sidebar.number_input(
    "Risk Free Rate (%)",
    value=0.0,
    step=0.5
)/100

# ---------------------------------------------------------
# Generate All Portfolios
# ---------------------------------------------------------

results=[]

progress=st.progress(0)

total=5151
count=0

for w1 in range(101):

    for w2 in range(101-w1):

        w3=100-w1-w2

        weights=np.array([w1,w2,w3])/100

        port_return=np.dot(weights,mean_returns)

        port_risk=np.sqrt(
            np.dot(
                weights.T,
                np.dot(cov_matrix,weights)
            )
        )

        sharpe=(port_return-risk_free)/port_risk if port_risk>0 else np.nan

        results.append({

            asset_names[0]:w1,

            asset_names[1]:w2,

            asset_names[2]:w3,

            "Return":port_return,

            "Risk":port_risk,

            "Sharpe":sharpe

        })

        count+=1

        progress.progress(count/total)

results=pd.DataFrame(results)

progress.empty()

st.success(f"Total Portfolios Evaluated : {len(results):,}")

# ---------------------------------------------------------
# Best Portfolios
# ---------------------------------------------------------

max_sharpe=results.loc[results["Sharpe"].idxmax()]

min_risk=results.loc[results["Risk"].idxmin()]

col1,col2=st.columns(2)

with col1:

    st.subheader("Maximum Sharpe Portfolio")

    st.dataframe(max_sharpe)

with col2:

    st.subheader("Minimum Risk Portfolio")

    st.dataframe(min_risk)

# ---------------------------------------------------------
# Efficient Frontier
# ---------------------------------------------------------

frontier=[]

risk_values=np.sort(results["Risk"].unique())

for r in risk_values:

    temp=results[np.isclose(results["Risk"],r)]

    frontier.append(temp.loc[temp["Return"].idxmax()])

frontier=pd.DataFrame(frontier)

# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

fig=go.Figure()

fig.add_trace(

    go.Scatter(

        x=results["Risk"],

        y=results["Return"],

        mode="markers",

        marker=dict(

            color=results["Sharpe"],

            colorscale="Viridis",

            size=6,

            showscale=True,

            colorbar=dict(title="Sharpe")

        ),

        text=results[asset_names].astype(str),

        name="All Portfolios"

    )

)

fig.add_trace(

    go.Scatter(

        x=frontier["Risk"],

        y=frontier["Return"],

        mode="lines",

        line=dict(

            color="black",

            width=4

        ),

        name="Efficient Frontier"

    )

)

fig.add_trace(

    go.Scatter(

        x=[max_sharpe["Risk"]],

        y=[max_sharpe["Return"]],

        mode="markers",

        marker=dict(

            color="red",

            size=14

        ),

        name="Maximum Sharpe"

    )

)

fig.add_trace(

    go.Scatter(

        x=[min_risk["Risk"]],

        y=[min_risk["Return"]],

        mode="markers",

        marker=dict(

            color="green",

            size=14

        ),

        name="Minimum Variance"

    )

)

fig.update_layout(

    title="Efficient Frontier",

    xaxis_title="Portfolio Risk (Standard Deviation)",

    yaxis_title="Expected Return",

    height=700

)

st.plotly_chart(fig,use_container_width=True)

# ---------------------------------------------------------
# Allocation Tables
# ---------------------------------------------------------

st.subheader("Maximum Sharpe Allocation")

alloc1=pd.DataFrame({

    "Asset":asset_names,

    "Weight (%)":[

        max_sharpe[asset_names[0]],

        max_sharpe[asset_names[1]],

        max_sharpe[asset_names[2]]

    ]

})

st.dataframe(alloc1,use_container_width=True)

st.subheader("Minimum Variance Allocation")

alloc2=pd.DataFrame({

    "Asset":asset_names,

    "Weight (%)":[

        min_risk[asset_names[0]],

        min_risk[asset_names[1]],

        min_risk[asset_names[2]]

    ]

})

st.dataframe(alloc2,use_container_width=True)

# ---------------------------------------------------------
# Download
# ---------------------------------------------------------

csv=results.to_csv(index=False).encode()

st.download_button(

    "Download All Portfolio Combinations",

    csv,

    file_name="Efficient_Frontier_Results.csv",

    mime="text/csv"

)

# ---------------------------------------------------------
# Display Results
# ---------------------------------------------------------

st.subheader("All Portfolio Combinations")

st.dataframe(results,use_container_width=True)
