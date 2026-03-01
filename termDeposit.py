import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import plotly.express as px

    return mo, pl, px


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

    It is quite strange that this marketing campaign has unknown values. For now let's check the class balance.
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
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(df, px):
    fig = px.scatter_matrix(
        df.collect(),
        dimensions=["age", "balance", "day", "duration", "campaign"],
        color="y",
    )
    return (fig,)


@app.cell
def _(fig):
    fig.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    - Check what these unknown variables are. Do we need to remove them/change the values into something more meaningful?
    - Check class balance
    - Plot distribution, check skewness etc. Histograms, KDE etc.
    - Outliers?
    - Target vs features: box plots, kde, heatmaps etc.
    """)
    return


if __name__ == "__main__":
    app.run()
