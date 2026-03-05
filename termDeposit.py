import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

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

    It is likely that the data was just not inserted after the calls.

    For now let's check the class balance.
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
    ~93% of no's. Huge imbalance in the dataset.

    Is this a normal figure for marketing efforts??

    Also this variable states whether the client has subscribed to a term deposit. Do we know if it's due to being ignored (not picking up calls) or because they said no (if duration > 0 but no contracts)?

    This imbalance seems to be the campaign's main problem and is something that warrants further investigation.

    Let's continue and look at some plots.
    """)
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
    #### Numerical Features
    """)
    return


@app.cell
def _(df):
    df_collected = df.collect()
    return (df_collected,)


@app.cell
def _(df_collected, go, make_subplots, mo):
    fig_hst_num = make_subplots(rows=2, cols=2)

    fig_hst_num.add_trace(
        go.Histogram(x=df_collected["age"], xbins=dict(size=1)), row=1, col=1
    )
    fig_hst_num.add_trace(
        go.Histogram(x=df_collected["balance"], xbins=dict(size=100)), row=1, col=2
    )
    fig_hst_num.add_trace(
        go.Histogram(x=df_collected["duration"], xbins=dict(size=10)), row=2, col=1
    )
    fig_hst_num.add_trace(
        go.Histogram(x=df_collected["campaign"], xbins=dict(size=1)), row=2, col=2
    )

    fig_hst_num.update_xaxes(title_text="Age", row=1, col=1)
    fig_hst_num.update_xaxes(title_text="Balance", row=1, col=2, range=[-800, 5000])
    fig_hst_num.update_xaxes(title_text="Duration", row=2, col=1, range=[0, 1000])
    fig_hst_num.update_xaxes(title_text="Campaign", row=2, col=2, range=[0, 15])

    fig_hst_num.update_yaxes(title_text="Count")

    fig_hst_num.update_layout(
        height=800,
        width=800,
        title_text="Histograms of Numerical Features",
        showlegend=False,
    )

    mo.ui.plotly(fig_hst_num)

    return


@app.cell
def _(df_collected, mo, pl):
    clipped = df_collected.select(
        pl.len().alias("total_rows"),
        ((pl.col("balance") < -800) | (pl.col("balance") > 5000)).sum().alias("balance"),
        (pl.col("duration") > 1000).sum().alias("duration"),
        (pl.col("campaign") > 15).sum().alias("campaign"),
    ).select(
        pl.col("balance"),
        pl.col("duration"),
        pl.col("campaign"),
        (pl.col("balance") / pl.col("total_rows") * 100).alias("pct_balance_clipped"),
        (pl.col("duration") / pl.col("total_rows") * 100).alias("pct_duration_clipped"),
        (pl.col("campaign") / pl.col("total_rows") * 100).alias("pct_campaign_clipped"),
    )

    r = clipped.row(0, named=True)


    mo.md(f"""
        Data outside clipped ranges:

        - Balance: {r["balance"]} ({r["pct_balance_clipped"]:.1f}%)
        - Duration: {r["duration"]} ({r["pct_duration_clipped"]:.1f}%)
        - Campaign: {r["campaign"]} ({r["pct_campaign_clipped"]:.1f}%)
    """)
    return


@app.cell
def _(df, mo, pl):
    skewness = df.select(
        pl.col("age").skew().alias("age"),
        pl.col("balance").skew().alias("balance"),
        pl.col("duration").skew().alias("duration"),
        pl.col("campaign").skew().alias("campaign"),
    )

    mo.vstack([mo.md("Skewness Coefficients"), skewness])
    return


@app.cell
def _(mo):
    mo.md(r"""
    - Age: Approximately symmetric (skewness ~ 0.44), unimodal, peak around ages 30-35. Most customers are working age adults. Expected.
    - Balance: Extreme right skew (skewness ~8.26). Concentrated near or at 0. Most customers have very low or no balance.
    - Duration: Heavy right skew (~3.17). Peak around 70-130 seconds. Most calls are short.
    - Campaign: Discrete variable. Also heavily right skewed (~4.7). Concentrated at 1-3 contacts but some were contacted more than 20 times (max = 60+).

    **Note**: Ranges for `balance`, `duration` and `campaign` have been adjusted to better see the distributions. The data outside the clipped ranges are less thatn 7%, so the histograms are showing the vast majority of the data.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### Categorical Features
    """)
    return


@app.cell
def _(df_collected, pl):
    cat_cols = df_collected.select(pl.col(pl.String).exclude("y")).columns

    cat_counts = {
        col: df_collected[col].value_counts().sort("count", descending=True)
        for col in cat_cols
    }

    # cat_counts
    return cat_cols, cat_counts


@app.cell
def _(cat_counts, pl):
    month_order = [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]

    month_sorted = (
        cat_counts["month"]
        .with_columns(pl.col("month").cast(pl.Enum(month_order)))
        .sort("month")
    )

    # month_sorted
    return (month_sorted,)


@app.cell
def _(cat_cols, cat_counts, go, make_subplots, mo, month_sorted, pl):
    fig_bar_cat = make_subplots(
        rows=5, cols=2, subplot_titles=[*cat_cols, "", "month (chronological)"]
    )

    for i, col in enumerate(cat_cols):
        row = i // 2 + 1  # maps idx 0,1 to row 1. Idx 2,3 to row 2, etc.
        col_idx = i % 2 + 1
        counts = cat_counts[col]
        fig_bar_cat.add_trace(
            go.Bar(x=counts[col], y=counts["count"]), row=row, col=col_idx
        )

    fig_bar_cat.add_trace(
        go.Bar(x=month_sorted["month"].cast(pl.String), y=month_sorted["count"]),
        row=5,
        col=2,
    )


    fig_bar_cat.update_yaxes(title_text="Count")

    fig_bar_cat.update_layout(
        height=1200,
        width=1000,
        title_text="Bar Charts of Categorical Features",
        showlegend=False,
    )

    mo.ui.plotly(fig_bar_cat)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - job: Blue-collar, management and technicians have the highest counts. unknown is negligable.
    - marital: Married is the majority. Makes sense for this product.
    - education: Secondary is highest. unknown is negligable.
    - default: Overwhelmingly no. Almost no variance, which likely means it won't be very useful for modeling.
    - housing: ~60/40 split.
    - loan: ~85% no. Low variance.
    - contact: Cellular is highest but unknown is second (~32%). As mentioned earlier, this is strange but likely nothing concerning and is most likely because the data is not recorded after the call.
    - month: May has a large spike. We see that the campaign slowly ramps up from January then spikes massively then drops off again. Summer months seems more active (Jun-Aug). September is completely missing.

    Let's move on to bivariate analysis.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Bivariate Analysis
    """)
    return


@app.cell
def _(df_collected, mo, px):
    fig = px.scatter_matrix(
        df_collected,
        dimensions=["age", "balance", "duration", "campaign"],
        color="y",
    )

    mo.ui.plotly(fig)
    return


if __name__ == "__main__":
    app.run()
