import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    from plotly.subplots import make_subplots
    import marimo as mo
    import polars as pl
    import plotly.express as px
    import plotly.graph_objects as go

    return go, make_subplots, mo, pl, px


@app.cell
def _(mo):
    mo.md(r"""
    # Goal

    The goal is to predict if the customer will subscribe to a term deposit $y$.

    This is a binary classification problem.
    """)
    return


@app.cell
def _(pl):
    df = pl.scan_csv("term-deposit-marketing-2020.csv")
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    # EDA
    """)
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(mo):
    mo.md(r"""
    We see the following:

    - No nulls but there are a few unknowns found in some categorical features. We should investigate this.
    - Data types are as expected
    - Min balance is -8019 which seems quite high. But it could make sense if the spending habits are horrible. Also this is average over the year. The spending could be much larger in specific months and does not really represent the entire year.
    - The std for balance seems very large but it does make sense given that the min is -8019, while the median is 407. The max is also 102,127, which contributes to the large std. Since median < mean, it is right skewed.

    **Note**: `duration` is the length of the phone call in seconds. This is a target leakage variable because we only know the value after the call has ended. At prediction time (before calling a customer), this feature won't exist yet. It should be excluded from the final model or carefully evaluated (test with and without) otherwise it will likely inflate accuracy artificially.
    """)
    return


@app.cell
def _(df, pl):
    df.select(pl.col(pl.String) == "unknown").sum()
    return


@app.cell
def _(mo):
    mo.md(r"""
    From counting the unknowns, we see that

    - The unknowns in the job feature is negligable (235/40000 ~ 0.6%)
    - Education at ~ 3.8% is manageable and
    - Contact at ~ 32% is unsettling. It's a third of our data.

    It is quite strange that this marketing campaign has unknown contact values. For now let's check the class balance.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Class Balance
    """)
    return


@app.cell
def _(df, pl):
    df.group_by("y").len().with_columns(
        (pl.col("len") / pl.col("len").sum() * 100).alias("percentage")
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    92% of no's. Huge imbalance in the dataset. Is this a normal figure for marketing efforts? Also this variable states whether the client has subscribed to a term deposit; do we know if it is due to being ignored (not picking up calls) or because they said no (if duration > 0 but no contracts)?

    Let's continue and look at some plots.
    """)
    return


@app.cell
def _():
    # df.select("duration").filter(pl.col("duration") == 115).count()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Univariate Analysis
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    histograms for age, balance, duration, campaign. No need day because it is not meaningful
    """)
    return


@app.cell
def _(df, go, make_subplots, mo):
    fig_hst_num = make_subplots(rows=2, cols=2)

    fig_hst_num.add_trace(go.Histogram(x=df.collect()["age"]), row=1, col=1)
    fig_hst_num.add_trace(go.Histogram(x=df.collect()["balance"]), row=1, col=2)
    fig_hst_num.add_trace(go.Histogram(x=df.collect()["duration"]), row=2, col=1)
    fig_hst_num.add_trace(go.Histogram(x=df.collect()["campaign"]), row=2, col=2)

    fig_hst_num.update_xaxes(title_text="Age", row=1, col=1)
    fig_hst_num.update_xaxes(title_text="Balance", row=1, col=2)
    fig_hst_num.update_xaxes(title_text="Duration", row=2, col=1)
    fig_hst_num.update_xaxes(title_text="Campaign", row=2, col=2)

    fig_hst_num.update_yaxes(title_text="Count", row=1, col=1)
    fig_hst_num.update_yaxes(title_text="y", row=1, col=2)
    fig_hst_num.update_yaxes(title_text="y", row=2, col=1)
    fig_hst_num.update_yaxes(title_text="y", row=2, col=2)

    fig_hst_num.update_layout(
        height=800,
        width=800,
        title_text="Histograms of Numerical Features",
        showlegend=False,
    )

    # Change binnings so it's not automatic. Also figure out the y axis and try to get all axis the same.

    mo.ui.plotly(fig_hst_num)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Bivariate Analysis
    """)
    return


@app.cell
def _(df, mo, px):
    fig = px.scatter_matrix(
        df.collect(),
        dimensions=["age", "balance", "day", "duration", "campaign"],
        color="y",
    )

    mo.ui.plotly(fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    This is the initial pairwise scatter matrix of numeric features. We can understand more in our univariate and bivariate sections later.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
