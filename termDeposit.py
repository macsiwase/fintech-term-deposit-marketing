import marimo

__generated_with = "0.23.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import pandas as pd
    import duckdb as db
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    from plotly.subplots import make_subplots
    from lazypredict.Supervised import LazyClassifier
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        confusion_matrix,
        precision_recall_curve,
        silhouette_score,
        adjusted_rand_score,
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek, SMOTEENN
    from imblearn.pipeline import Pipeline as ImbPipeline
    from xgboost import XGBClassifier
    from umap import UMAP

    return (
        ImbPipeline,
        KMeans,
        LazyClassifier,
        PCA,
        RandomForestClassifier,
        RandomUnderSampler,
        SMOTEENN,
        SMOTETomek,
        StandardScaler,
        StratifiedKFold,
        TSNE,
        UMAP,
        XGBClassifier,
        adjusted_rand_score,
        confusion_matrix,
        cross_val_score,
        db,
        fcluster,
        ff,
        go,
        linkage,
        make_subplots,
        mo,
        np,
        pd,
        pl,
        precision_recall_curve,
        px,
        silhouette_score,
        train_test_split,
    )


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
    - The std for balance seems very large but it does make sense given that the min is -8019, while the median is 407. The max is also 102,127, which contributes to the large std. Since median < mean, this suggests that we have a right skewed distribution. We quantify this later.

    **Note**: `duration` is the length of the phone call in seconds. This is a target leakage variable because we only know the value after the call has ended. At prediction time (before calling a customer), this feature won't exist yet. Therefore, we have to be careful about using this feature in our models as it might inflate accuracy artifically.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Unknowns/Nulls
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

    - The unknowns in the job feature is negligible (235/40000 ~ 0.6%)
    - Education at ~ 3.8% is manageable and
    - Contact at ~ 32% is unsettling. It's a third of our data.

    It is likely that the data was just not inserted after the calls (confirmed by the company).

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

    Is this a normal figure for marketing efforts?

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
def _(df, mo, pl):
    kurtosis = df.select(
        pl.col("age").kurtosis().alias("age"),
        pl.col("balance").kurtosis().alias("balance"),
        pl.col("duration").kurtosis().alias("duration"),
        pl.col("campaign").kurtosis().alias("campaign"),
    )

    mo.vstack([mo.md("Kurtosis Coefficients"), kurtosis])
    return


@app.cell
def _(mo):
    mo.md(r"""
    - Age: Approximately symmetric (skewness ~ 0.44), platykurtic (~-0.50) so slightly lower tails than normal, unimodal, peak around ages 30-35. Most customers are working age adults. Expected.
    - Balance: Extreme right skew (skewness ~8.26), extreme leptokurtic (~141.8) so there are extreme outlier balances from the center. Concentrated near or at 0. Most customers have very low or no balance. This will heavily influence any model because of the large variance.
    - Duration: Heavy right skew (~3.17). Very leptokurtic (~18.2) so heavier tails. Peak around 70-130 seconds. Most calls are short.
    - Campaign: Discrete variable. Also heavily right skewed (~4.7). Very leptokurtic (~36.2) so again heavier tails. Concentrated at 1-3 contacts but some were contacted more than 20 times (max = 60+).
    - Since balance, duration and campaign have large kurtosis. Standardization (z-scoring) uses mean and std, which are dominated by these extreme values. Therefore if linear models are used, robust scaling or log transforms would be needed. i.e. Tree based models are better suited than linear models and are naturally robust to these distributional properties.

    **Note**: Ranges for `balance`, `duration` and `campaign` have been adjusted to better see the distributions. The data outside the clipped ranges are less than 7%, so the histograms are showing the vast majority of the data.
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
    return month_order, month_sorted


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
    - job: Blue-collar, management and technicians have the highest counts. unknown is negligible.
    - marital: Married is the majority. Makes sense for this product.
    - education: Secondary is highest. unknown is negligible.
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
def _(mo):
    mo.md(r"""
    #### Categorical Features vs Target
    """)
    return


@app.cell
def _(cat_cols, df_collected, pl):
    cat_sub_rate = {
        col: df_collected.group_by([col]).agg(
            (pl.col("y") == "yes").mean().alias("subscription_rate")
        )
        for col in cat_cols
    }

    overall_sub_rate = (
        (
            df_collected.group_by("y")
            .len()
            .with_columns((pl.col("len") / pl.col("len").sum() * 100).alias("percentage"))
        )
        .filter(pl.col("y") == "yes")
        .select(pl.col("percentage"))
    )


    # cat_sub_rate
    return cat_sub_rate, overall_sub_rate


@app.cell
def _(
    cat_cols,
    cat_sub_rate,
    go,
    make_subplots,
    mo,
    month_order,
    overall_sub_rate,
    pl,
):
    month_sub_rate_sorted = (
        cat_sub_rate["month"]
        .with_columns(pl.col("month").cast(pl.Enum(month_order)))
        .sort("month")
    )

    fig_bar_cat_sub_rate = make_subplots(
        rows=5, cols=2, subplot_titles=[*cat_cols, "", "month (chronological)"]
    )

    for i_sub_rate, col_sub_rate in enumerate(cat_cols):
        row_sub_rate = i_sub_rate // 2 + 1  # maps idx 0,1 to row 1. Idx 2,3 to row 2, etc.
        col_idx_sub_rate = i_sub_rate % 2 + 1
        rates = cat_sub_rate[col_sub_rate].sort("subscription_rate", descending=True)
        fig_bar_cat_sub_rate.add_trace(
            go.Bar(x=rates[col_sub_rate], y=rates["subscription_rate"] * 100),
            row=row_sub_rate,
            col=col_idx_sub_rate,
        )

    fig_bar_cat_sub_rate.add_trace(
        go.Bar(
            x=month_sub_rate_sorted["month"].cast(pl.String),
            y=month_sub_rate_sorted["subscription_rate"] * 100,
        ),
        row=5,
        col=2,
    )

    fig_bar_cat_sub_rate.add_hline(
        y=overall_sub_rate[0, 0],
        line_dash="dot",
        line_color="gray",
        annotation_text="Overall Rate",
        annotation_position="top right",
        annotation_font_size=10,
    )


    fig_bar_cat_sub_rate.update_yaxes(title_text="Subscription rate (%)")

    fig_bar_cat_sub_rate.update_layout(
        height=1200,
        width=1000,
        title_text="Categorical Features by Subscription",
        showlegend=False,
    )

    mo.ui.plotly(fig_bar_cat_sub_rate)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - Job: students at 15.6% (~700 students) and retired at 10.5% have a much higher subscription rate than average. However we see from the previous charts that the volume of both student and retired people are not very high. Blue-collar, which represents the highest volume has a much lower conversion. Management (~7000 observations) has a pretty good conversion (high volume + high subscription rate) and technician has an average conversion. Management is in the sweet spot.
    - Education: Tertiary is the only one above average. Clear monotonic trend. Higher education implies higher conversion.
    - Marital: Single is much higher than the average. Married is much lower. Single people are more likely to have disposable income or fewer financial commitments.
    - Housing: People with no housing loans convert higher. Makes sense since fewer financial obligations means more open to term deposits.

    The above could be good signals for our ML model.

    - Loan and default: Default has virtually no variance so we can drop this candidate. Loan has low variance (~85% 'no') with a modest rate difference (~7.6% no vs ~5.5% yes), weak signal.
    - Contact and Month: Could be good signals for another ML model (keep calling target demographics. i.e. the ones more likely to subscribe). Cellular has a higher average sub rate. October (61% on ~160 observations) and March have much higher average rates (but very small sample sizes).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### Numerical Features vs Target
    """)
    return


@app.cell
def _(df_collected, go, make_subplots, mo):
    num_cols = ["age", "balance", "duration", "campaign"]

    fig_violin = make_subplots(rows=2, cols=2, subplot_titles=num_cols)

    for i_numerical, col_numerical in enumerate(num_cols):
        row_numerical = i_numerical // 2 + 1
        col_idx_numerical = i_numerical % 2 + 1

        for y_val, color in [("no", "#636EFA"), ("yes", "#EF553B")]:
            subset = df_collected.filter(df_collected["y"] == y_val)

            if (
                col_numerical == "campaign"
            ):  # box plot for campaign due to being a discrete variable
                fig_violin.add_trace(
                    go.Box(y=subset[col_numerical], name=y_val, marker_color=color),
                    row=row_numerical,
                    col=col_idx_numerical,
                )
            else:
                fig_violin.add_trace(  # Violin plot for other numerical features because they are continuous variables. Violin plots use KDE, which assumes continuity.
                    go.Violin(
                        y=subset[col_numerical],
                        name=y_val,
                        marker_color=color,
                        box_visible=True,
                        meanline_visible=True,
                    ),
                    row=row_numerical,
                    col=col_idx_numerical,
                )

    fig_violin.update_yaxes(row=1, col=2, range=[-800, 5000])
    fig_violin.update_yaxes(row=2, col=1, range=[0, 1000])
    fig_violin.update_yaxes(row=2, col=2, range=[0, 20])

    fig_violin.update_layout(
        height=800,
        width=800,
        showlegend=False,
        title_text="Numerical Features by Subscription",
    )

    mo.ui.plotly(fig_violin)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - Age: Both 'yes' and 'no' groups have similar medians (~38-40). The 'yes' group has a slightly higher median and a larger IQR.
    - Balance: 'yes' group has a higher median (~600) with a larger IQR. More balance implies more subs, which makes sense (more disposable income to pay for term deposits).
    - Duration: 'yes' group has a much larger IQR but we ignore this feature due to data leakage as mentioned before.
    - Campaign: Both groups look very similar. This feature is not a great discriminator.
    """)
    return


@app.cell
def _(df_collected, mo, px):
    fig_scatter = px.scatter_matrix(
        df_collected,
        dimensions=["age", "balance", "duration", "campaign"],
        color="y",
    )

    fig_scatter.update_layout(
        height=800,
        width=1000,
        title_text="Scatter Matrix of Numerical Features Colored by Subscription",
    )

    mo.ui.plotly(fig_scatter)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - High campaign count results in mostly no subscriptions.
    - Heavy overlap in age-balance (no visible boundary)
    - Some separation in pairs with duration and with campaign extreme values (weak pairwise separability).

    We quantify this pairwise separability using correlation analysis next.
    """)
    return


@app.cell
def _(df_collected, pl):
    df_corr = df_collected.select(
        pl.col("age", "balance", "duration", "campaign"),
        (pl.col("y") == "yes")
        .cast(pl.Int8)
        .alias("y"),  # encode target variable as binary to be used for pearson correlation.
    )

    df_corr_pearson = df_corr.corr()

    df_corr_spearman = df_corr.select(
        pl.all().rank()
    ).corr()  # Because there is no native spearman correlation method in polars, we need to implement it manually. Spearman correlation is the Pearson correlation of the rank-transformed data. So we will rank-transform each column and then compute the Pearson correlation on the ranked data.
    return df_corr, df_corr_pearson, df_corr_spearman


@app.cell
def _(df_corr, df_corr_spearman, mo, px):
    fig_corr_spearman = px.imshow(
        df_corr_spearman,
        y=df_corr.columns,
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        text_auto=".2f",
    )

    fig_corr_spearman.update_layout(
        height=800,
        width=800,
        title_text="Spearman Correlation Matrix",
    )

    mo.ui.plotly(fig_corr_spearman)
    return


@app.cell
def _(df_corr_pearson, df_corr_spearman, go, mo):
    corr_cols = df_corr_pearson.columns[:-1]

    fig_corr_bar = go.Figure(
        data=[
            go.Bar(x=corr_cols, y=df_corr_pearson.row(-1)[:-1], name="Pearson"),
            go.Bar(x=corr_cols, y=df_corr_spearman.row(-1)[:-1], name="Spearman"),
        ]
    )

    fig_corr_bar.update_layout(
        height=800,
        width=800,
        title_text="Correlation of Numerical Features with Target Variable (y)",
    )

    mo.ui.plotly(fig_corr_bar)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - Pearson measures linear associations and is sensitive to extreme values. Given the extreme skew we saw in our features earlier, Spearman is more robust here.
    - Heatmap shows no strong multicollinearity between the feature variables (all near 0). This means that the features are fairly independent. `duration`'s correlation with y is strongest (0.33) but as mentioned above, we exclude it due to data leakage.
    - The bar chart shows weak individual correlations with y (after excluding duration by the same reason as above), which implies that our ML model needs feature interactions to predict well.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Summary
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - No null values but there are unknowns: job (~0.6%, negligible), education (3.8%, manageable), contact (32%, likely unrecorded data, not a data quality issue).

    - Class balance analysis reveals that there's a predominant amount of non subscribers (~93%). This requires careful handling: maybe use stratified sampling, class weights or appropriate evaluation metrics (e.g. precision-recall, F1, AUC-ROC or recall instead of accuracy).

    - Balance, duration and campaign contain extreme values (kurtosis ~36-142). Valid according to the domain and not data errors. We should use Tree based models because they are robust to these extreme values.

    - Features with the highest variation in subscription rates: job, education, marital status and housing.

    - Weak individual correlations with the target y: all features were < 0.06 Spearman (not including duration). Feature interactions will probably be necessary and tree based ensemble methods will most likely be the best model.

    - Default is dropped because it has essentially no variance.

    Therefore the segments to prioritize are:

    - Students (15.6% conversion), retirees (10.5%)
    - Tertiary education (9.2%), single (9.4%), no housing loan (9.0%)
    - Management is the highest volume high conversion segment.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Modeling
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Since there are only ~7% of customers that have subscribed, we should focus on helping the company be more efficient and save cost (time) (i.e. not wasting time calling people who are likely not going to be customers) while retaining their most loyal subscribers.

    We propose a multi layered ML system:

    1. ML1 (Pre call filter): Use features known before a call (age, job, marital, education, balance, housing, loan) with the goal of reducing the 40,000 customers to a targeted subset.
    2. ML2 (Optimizer): Prioritize customers to keep calling by identifying features that indicate high probability of subscribing. That is, which customers are more likely to say yes to subscribing and keep calling these type of customers.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## ML 1: Pre Call Filter
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Let's prepare the data. We need to one hot encode our categorical values for our modeling
    """)
    return


@app.cell
def _(df_collected, pd, pl, train_test_split):
    ml1_features = ["age", "job", "marital", "education", "balance", "housing", "loan"]

    X = df_collected.select(ml1_features).to_pandas()
    y = (df_collected["y"] == "yes").cast(pl.Int8).to_pandas()

    X = pd.get_dummies(
        X, drop_first=True
    )  # avoids dummy variable trap by dropping the first category of each categorical variable. This prevents perfect multicollinearity in linear models, which can cause issues with model estimation and interpretation.

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )  # stratify=y ensures that the train and test sets have the same proportion of positive and negative examples as the original dataset, which is important for imbalanced classification problems.

    # X_train, X_test, y_train, y_test
    return X, X_test, X_train, y, y_test, y_train


@app.cell
def _(mo):
    mo.md(r"""
    ### LazyPredict
    """)
    return


@app.cell
def _(mo):
    run_model_fit_button = mo.ui.run_button(
        label="Run Model Fit"
    )  # The model fitting process can be time-consuming, so we add a button to allow users to choose when to run it.

    run_model_fit_button
    return (run_model_fit_button,)


@app.cell
def _(
    LazyClassifier,
    X_test,
    X_train,
    mo,
    run_model_fit_button,
    y_test,
    y_train,
):
    mo.stop(
        not run_model_fit_button.value,
        "Click the button above to run the model fitting process for lazypredict.",
    )

    top_classifiers = LazyClassifier(verbose=0, ignore_warnings=True)
    models, predictions = top_classifiers.fit(X_train, X_test, y_train, y_test)
    return (models,)


@app.cell
def _(models):
    models
    return


@app.cell
def _(mo):
    mo.md(r"""
    - Looking at Balanced Accuracy, most models are scoring the same or very similar (~0.5, so it's basically a coin toss). Since Balanced Accuracy is the unweighted average recall per class, the models are predicting the same class for all observations. Given our 93% 'no' majority, we can infer that the models are predicting no's for everything. Most of the models are not doing better than the DummyClassifier.
    - We also notice that recall has the same results as accuracy. This is because LazyPredict uses weighted average recall, which tends to accuracy when a model predicts only one class. This shows that accuracy is misleading for this imbalanced datasets.
    - The few models that actually learned something are: NearestCentroid (~0.59 balanced accuracy), DecisionTreeClassifier (~0.54 balanced accuracy) and ExtraTreeClassifier (~0.54 balanced accuracy) but they're still not great results.
    - Since LazyPredict only uses default parameters (not optimized), we select tree based models (RandomForestClassifier, ExtraTreesClassifier. RandomForest is an ensemble of DecisionTrees) for further development. These models are naturally robust to heavy tailed distributions and outliers.
    - NearestCentroid was the best but it's too simple, not tunable and does not scale well. So we can keep it as a simple baseline but it won't be our final model.

    For ML1, a false negative (missed subscriber) is more costly than a false positive (wasted call). Therefore, we prioritize recall on class 1 ("yes") instead of accuracy and try to answer: what proportion of actual subscribers did we identify?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    To improve recall on the minority class, we can try a couple of strategies:

    Resampling:

    - Random Undersampling. This downsamples the majority class to match the minority. We test multiple ratios (1:1, 2:1, 3:1).
    - Changing the class weight for models that support it (`class_weight='balanced'`)
    - SMOTETomek. This combines over and undersampling using SMOTE (oversample) and Tomek links (cleans the decision boundary).
    - SMOTEENN. Also combines both over and undersampling. Uses SMOTE and Edited Nearest Neighbours (ENN) (more aggressive cleaning).

    Threshold tuning:

    - This changes the decision boundary at inference. Instead of the default 0.5 cutoff, we find and use the optimal recall-precision tradeoff.

    We avoid using SMOTE by itself because it generates synthetic minority class samples, which can produce unrealistic samples if the minority class is not well-clustered.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Resampling
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Undersampling, Class Weight Adjustment and Random Forest Classifier
    """)
    return


@app.cell
def _():
    # Uncomment this cell and the mo.stop code below if the undersampling process is too slow and crashing your environment.

    # run_undersampling_button = mo.ui.run_button(
    #     label="Run Undersampling"
    # )  # The undersampling process can be cpu intensive, so we add a button to allow users to choose when to run it.

    # run_undersampling_button
    return


@app.cell
def _(
    RandomForestClassifier,
    RandomUnderSampler,
    SMOTEENN,
    SMOTETomek,
    X_test,
    X_train,
    confusion_matrix,
    y_test,
    y_train,
):
    # mo.stop(
    #     not run_undersampling_button.value,
    #     "Click the button above to run the undersampling process.",
    # )

    undersampled_data = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)

    rf_configs = [
        {"name": "Undersampled", "data": undersampled_data, "params": {}},
        {
            "name": "Balanced Weights",
            "data": (X_train, y_train),
            "params": {"class_weight": "balanced"},
        },
        {
            "name": "SMOTE-Tomek",
            "data": SMOTETomek(random_state=42).fit_resample(X_train, y_train),
            "params": {},
        },
        {
            "name": "SMOTE-ENN",
            "data": SMOTEENN(random_state=42).fit_resample(X_train, y_train),
            "params": {},
        },
    ]

    rf_results = {}

    for rf_config in rf_configs:
        model = RandomForestClassifier(random_state=42, **rf_config["params"])
        model.fit(*rf_config["data"])
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        rf_results[rf_config["name"]] = {"model": model, "cm": cm, "cm_norm": cm_norm}
    return rf_results, undersampled_data


@app.cell
def _(mo):
    mo.md(r"""
    #### Confusion Matrices
    """)
    return


@app.cell
def _(go, make_subplots, mo, rf_results):
    labels = ["No Subscription", "Subscription"]

    fig_rf_resampled_total = make_subplots(
        rows=2, cols=2, subplot_titles=list(rf_results.keys())
    )

    for i_resampled_total, (rf_name_total, rf_result_total) in enumerate(
        rf_results.items()
    ):
        row_resampled_total = i_resampled_total // 2 + 1
        col_resampled_total = i_resampled_total % 2 + 1
        fig_rf_resampled_total.add_trace(
            go.Heatmap(
                z=rf_result_total["cm"],
                x=labels,
                y=labels,
                text=rf_result_total["cm"],
                texttemplate="%{text:,d}",
                colorscale="Blues",
                showscale=(i_resampled_total == 0),
            ),
            row=row_resampled_total,
            col=col_resampled_total,
        )

    fig_rf_resampled_total.update_xaxes(title_text="Predicted")
    fig_rf_resampled_total.update_yaxes(title_text="True", row=1, col=1)
    fig_rf_resampled_total.update_yaxes(title_text="True", row=2, col=1)
    fig_rf_resampled_total.update_yaxes(showticklabels=False, row=1, col=2)
    fig_rf_resampled_total.update_yaxes(showticklabels=False, row=2, col=2)
    fig_rf_resampled_total.update_yaxes(autorange="reversed")

    fig_rf_resampled_total.update_layout(
        height=800,
        width=800,
        title_text="Confusion Matrices for Resampling Techniques (Random Forest) - Raw",
    )

    mo.ui.plotly(fig_rf_resampled_total)
    return (labels,)


@app.cell
def _(go, labels, make_subplots, mo, rf_results):
    fig_rf_resampled = make_subplots(rows=2, cols=2, subplot_titles=list(rf_results.keys()))

    for i_resampled, (rf_name, rf_result) in enumerate(rf_results.items()):
        row_resampled = i_resampled // 2 + 1
        col_resampled = i_resampled % 2 + 1
        fig_rf_resampled.add_trace(
            go.Heatmap(
                z=rf_result["cm_norm"],
                x=labels,
                y=labels,
                text=rf_result["cm_norm"].round(2),
                texttemplate="%{text:.2%}",
                colorscale="Blues",
                zmin=0,
                zmax=1,
                showscale=(i_resampled == 0),
            ),
            row=row_resampled,
            col=col_resampled,
        )

    fig_rf_resampled.update_xaxes(title_text="Predicted")
    fig_rf_resampled.update_yaxes(title_text="True", row=1, col=1)
    fig_rf_resampled.update_yaxes(title_text="True", row=2, col=1)
    fig_rf_resampled.update_yaxes(showticklabels=False, row=1, col=2)
    fig_rf_resampled.update_yaxes(showticklabels=False, row=2, col=2)
    fig_rf_resampled.update_yaxes(autorange="reversed")

    fig_rf_resampled.update_layout(
        height=800,
        width=800,
        title_text="Confusion Matrices for Resampling Techniques (Random Forest) - Normalized",
    )

    mo.ui.plotly(fig_rf_resampled)
    return


@app.cell
def _(mo):
    mo.md(r"""
    We see that 1:1 undersampling is the best strategy, identifying ~56% of subscribers correctly. All other strategies are mostly predicting 'no' for most observations.

    However 56% recall with 43% false positive rate is still not great.

    - Out of 8000 test customers, the model would call 3505 (predicted 'yes': 3182 + 323). Of those 323 are actual subscribers.
    - 256 subscribers were missed
    - ~4495 calls saved compared to calling everyone

    <br>

    **Note**:

    - 2:1 and 3:1 undersampling were tested and removed due to having similar results as balanced weights, SMOTE-Tomek and SMOTE-ENN. This is because there are more 'no' samples in training, which moves the model to predict more 'no's.

    - Combining undersampling and class weights gives identical results as undersampling alone (this was removed because it offered no improvements)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### Threshold Tuning: Precision-Recall Curves
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The undersampled Random Forest has a default threshold of 0.5: "predict yes if P(yes) > 0.5". Would lowering the threshold to a certain point help?

    Let's try threshold tuning to find out.
    """)
    return


@app.cell
def _(X_test, precision_recall_curve, rf_results, y_test):
    y_prob_rf = rf_results["Undersampled"]["model"].predict_proba(X_test)[:, 1]
    precisions_rf, recalls_rf, thresholds_rf = precision_recall_curve(y_test, y_prob_rf)
    return precisions_rf, recalls_rf, thresholds_rf


@app.cell
def _(go, mo, precisions_rf, recalls_rf, thresholds_rf):
    fig_precision_recall = go.Figure(
        data=[
            go.Scatter(
                x=thresholds_rf, y=precisions_rf[:-1], mode="lines", name="Precision"
            ),
            go.Scatter(x=thresholds_rf, y=recalls_rf[:-1], mode="lines", name="Recall"),
        ]
    )

    fig_precision_recall.update_layout(
        height=800,
        width=1000,
        title_text="Precision and Recall vs. Classification Threshold - Random Forest (Undersampled)",
        xaxis_title="Threshold",
        yaxis_title="Score",
    )

    mo.ui.plotly(fig_precision_recall)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The precision-recall curve shows that:

    - At threshold ~0.2, we catch ~92% of subscribers (recall ~92%) but ~93% would be predicted 'yes' are false positives (~7% precision).
    - At threshold ~0.5 (default), we catch ~53% with precision ~9%, slightly better precision.
    - Higher thresholds reduce calls but miss more subscribers.

    Random Forest with undersampling provided us with a baseline.

    We now try XGBoost, which uses gradient boosting (sequential tree building that corrects previous errors) with the `scale_pos_weight` parameter, which allows us to control emphasis on predicting subscribers.

    First we try with undersampling, then with class weight tuning (without undersample) and lastly with both.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### XGBoost
    """)
    return


@app.cell
def _(
    XGBClassifier,
    X_test,
    X_train,
    confusion_matrix,
    undersampled_data,
    y_test,
    y_train,
):
    xgb_configs = [
        {"name": "Undersampled", "params": {}, "data": undersampled_data},
        {
            "name": "Class Weighted (autoscaled)",
            "params": {
                "scale_pos_weight": len(y_train[y_train == 0]) / len(y_train[y_train == 1])
            },
            "data": (X_train, y_train),
        },
        {
            "name": "Undersampled + Weight 2",
            "params": {"scale_pos_weight": 2},
            "data": undersampled_data,
        },
        {
            "name": "Undersampled + Weight 5",
            "params": {"scale_pos_weight": 5},
            "data": undersampled_data,
        },
        {
            "name": "Undersampled + Weight 10",
            "params": {"scale_pos_weight": 10},
            "data": undersampled_data,
        },
    ]

    xgb_results = {}

    for config in xgb_configs:
        xgb_model = XGBClassifier(random_state=42, **config["params"])
        xgb_model.fit(*config["data"])
        y_pred_xgb = xgb_model.predict(X_test)
        xgb_cm = confusion_matrix(y_test, y_pred_xgb)
        xgb_cm_norm = xgb_cm.astype("float") / xgb_cm.sum(axis=1, keepdims=True)
        xgb_results[config["name"]] = {
            "model": xgb_model,
            "cm": xgb_cm,
            "cm_norm": xgb_cm_norm,
        }
    return (xgb_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Confusion Matrices
    """)
    return


@app.cell
def _(go, labels, make_subplots, mo, xgb_results):
    titles_xgb_total = list(xgb_results.keys())
    fig_xgb_cm_total = make_subplots(rows=2, cols=3, subplot_titles=titles_xgb_total)

    for i_xgb_total, (name_xgb_total, result_xgb_total) in enumerate(xgb_results.items()):
        row_xgb_total = i_xgb_total // 3 + 1
        col_xgb_total = i_xgb_total % 3 + 1
        fig_xgb_cm_total.add_trace(
            go.Heatmap(
                z=result_xgb_total["cm"],
                x=labels,
                y=labels,
                text=result_xgb_total["cm"],
                texttemplate="%{text:,d}",
                colorscale="Blues",
                showscale=(i_xgb_total == 0),
            ),
            row=row_xgb_total,
            col=col_xgb_total,
        )

    fig_xgb_cm_total.update_xaxes(title_text="Predicted")
    fig_xgb_cm_total.update_yaxes(title_text="True", row=1, col=1)
    fig_xgb_cm_total.update_yaxes(title_text="True", row=2, col=1)
    fig_xgb_cm_total.update_yaxes(showticklabels=False, row=1, col=2)
    fig_xgb_cm_total.update_yaxes(showticklabels=False, row=1, col=3)
    fig_xgb_cm_total.update_yaxes(showticklabels=False, row=2, col=2)
    fig_xgb_cm_total.update_yaxes(autorange="reversed")

    fig_xgb_cm_total.update_layout(
        height=800,
        width=1000,
        title_text="Confusion Matrices for Resampling Techniques (XGBoost) - Raw",
    )

    mo.ui.plotly(fig_xgb_cm_total)
    return


@app.cell
def _(go, labels, make_subplots, mo, xgb_results):
    titles_xgb = list(xgb_results.keys())
    fig_xgb_cm = make_subplots(rows=2, cols=3, subplot_titles=titles_xgb)

    for i_xgb, (name_xgb, result_xgb) in enumerate(xgb_results.items()):
        row_xgb = i_xgb // 3 + 1
        col_xgb = i_xgb % 3 + 1
        fig_xgb_cm.add_trace(
            go.Heatmap(
                z=result_xgb["cm_norm"],
                x=labels,
                y=labels,
                text=result_xgb["cm_norm"].round(2),
                texttemplate="%{text:.2%}",
                colorscale="Blues",
                zmin=0,
                zmax=1,
                showscale=(i_xgb == 0),
            ),
            row=row_xgb,
            col=col_xgb,
        )

    fig_xgb_cm.update_xaxes(title_text="Predicted")
    fig_xgb_cm.update_yaxes(title_text="True", row=1, col=1)
    fig_xgb_cm.update_yaxes(title_text="True", row=2, col=1)
    fig_xgb_cm.update_yaxes(showticklabels=False, row=1, col=2)
    fig_xgb_cm.update_yaxes(showticklabels=False, row=1, col=3)
    fig_xgb_cm.update_yaxes(showticklabels=False, row=2, col=2)
    fig_xgb_cm.update_yaxes(autorange="reversed")

    fig_xgb_cm.update_layout(
        height=800,
        width=1000,
        title_text="Confusion Matrices for Resampling Techniques (XGBoost) - Normalized",
    )

    mo.ui.plotly(fig_xgb_cm)
    return


@app.cell
def _(mo):
    mo.md(r"""
    A big improvement!

    - As we increase class weight, we capture more subscribers (recall) but at the cost of calling more non-subscribers (false positive rates).
    - With class weight = 2, we have ~77% recall, with a false positive rate of ~67%, which is manageable. We'd catch most subscribers while cutting the call list by ~1/3.
    - Class weight = 5, we have ~88% recall, with a false positive rate of ~84%.
    - Class weight = 10, we have ~93% recall, with a false positive rate of ~89%.
    - Class weight = autoscaled, performed worse than our baseline.

    Class weight = 2 might be the sweet spot for us.

    Let's validate these results with cross validation and check if the 77% recall is actually stable or just lucky.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Cross Validation
    """)
    return


@app.cell
def _(
    ImbPipeline,
    RandomUnderSampler,
    StratifiedKFold,
    X,
    XGBClassifier,
    cross_val_score,
    y,
):
    xgb_cv_pipeline = ImbPipeline(  # Use imbpipeline to ensure that the undersampling is done within each fold of the cross-validation, preventing data leakage.
        [
            ("undersampler", RandomUnderSampler(random_state=42)),
            ("classifier", XGBClassifier(random_state=42, scale_pos_weight=2)),
        ]
    )

    xgb_cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    xgb_recall_scores_cv = {
        metric: cross_val_score(xgb_cv_pipeline, X, y, cv=xgb_cv_folds, scoring=metric)
        for metric in ["recall", "accuracy"]
    }
    return (xgb_recall_scores_cv,)


@app.cell
def _(mo, xgb_recall_scores_cv):
    mo.md(f"""
    **5 Fold Cross Validation Scores (XGBoost Undersampled + Weight 2)**

    - Recall per fold: {[f"{s:.2%}" for s in xgb_recall_scores_cv["recall"]]}
    - Accuracy per fold: {[f"{s:.2%}" for s in xgb_recall_scores_cv["accuracy"]]}
    - Mean recall: {xgb_recall_scores_cv["recall"].mean():.2%}
    - Std: {xgb_recall_scores_cv["recall"].std():.2%}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The results are verified and we get ~76% recall consistently (accuracy is low due to optimizing for recall on minority class).

    ML1 can now identify ~3 out of 4 potential subscribers before we make any calls.

    Applied to the full 40,000 customers, this means that

    - ~2235 out of ~2895 actual subscribers would be correctly flagged for calling
    - The call list would be reduced from 40,000 to 27,195
    - ~660 subscribers would be missed

    Let's see which features were most important to our model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Feature Importance
    """)
    return


@app.cell
def _(X, pd, xgb_results):
    ml1_xgb_feature_importance = xgb_results["Undersampled + Weight 2"][
        "model"
    ].feature_importances_

    ml1_xgb_feature_importance_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": ml1_xgb_feature_importance,
        }
    ).sort_values("importance", ascending=False)

    ml1_xgb_feature_importance_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The features are quite uniform in that no single feature dominates and is what we previously saw in our EDA. This shows why ML1 struggled in performance.

    Now that we have a reduced list of customers to call, we develop ML2, which will help us identify which customers are more likely to say yes and to keep calling.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ML2: Optimizer with XGBoost
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ML2 uses all features including post call data because calls are actively happening. ML2 helps agents decide if this customer is worth another call or not due to how likely they are to subscribe.
    """)
    return


@app.cell
def _(RandomUnderSampler, df_collected, pd, train_test_split, y):
    ml2_features = [
        "age",
        "job",
        "marital",
        "education",
        "balance",
        "housing",
        "loan",
        "contact",
        "month",
        "duration",
        "campaign",
    ]


    X_ml2 = df_collected.select(ml2_features).to_pandas()

    X_ml2 = pd.get_dummies(X_ml2, drop_first=True)

    X_train_ml2, X_test_ml2, y_train_ml2, y_test_ml2 = train_test_split(
        X_ml2, y, test_size=0.2, random_state=42, stratify=y
    )

    undersampled_data_ml2 = RandomUnderSampler(random_state=42).fit_resample(
        X_train_ml2, y_train_ml2
    )
    return (
        X_ml2,
        X_test_ml2,
        X_train_ml2,
        undersampled_data_ml2,
        y_test_ml2,
        y_train_ml2,
    )


@app.cell
def _(
    XGBClassifier,
    X_test_ml2,
    X_train_ml2,
    confusion_matrix,
    undersampled_data_ml2,
    y_test_ml2,
    y_train_ml2,
):
    xgb_configs_ml2 = [
        {"name": "Undersampled", "params": {}, "data": undersampled_data_ml2},
        {
            "name": "Class Weighted (autoscaled)",
            "params": {
                "scale_pos_weight": len(y_train_ml2[y_train_ml2 == 0])
                / len(y_train_ml2[y_train_ml2 == 1])
            },
            "data": (X_train_ml2, y_train_ml2),
        },
        {
            "name": "Undersampled + Weight 2",
            "params": {"scale_pos_weight": 2},
            "data": undersampled_data_ml2,
        },
        {
            "name": "Undersampled + Weight 5",
            "params": {"scale_pos_weight": 5},
            "data": undersampled_data_ml2,
        },
        {
            "name": "Undersampled + Weight 10",
            "params": {"scale_pos_weight": 10},
            "data": undersampled_data_ml2,
        },
    ]

    xgb_results_ml2 = {}

    for config_ml2 in xgb_configs_ml2:
        xgb_model_ml2 = XGBClassifier(random_state=42, **config_ml2["params"])
        xgb_model_ml2.fit(*config_ml2["data"])
        y_pred_xgb_ml2 = xgb_model_ml2.predict(X_test_ml2)
        xgb_cm_ml2 = confusion_matrix(y_test_ml2, y_pred_xgb_ml2)
        xgb_cm_norm_ml2 = xgb_cm_ml2.astype("float") / xgb_cm_ml2.sum(axis=1, keepdims=True)
        xgb_results_ml2[config_ml2["name"]] = {
            "model": xgb_model_ml2,
            "cm": xgb_cm_ml2,
            "cm_norm": xgb_cm_norm_ml2,
        }
    return (xgb_results_ml2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Confusion Matrices
    """)
    return


@app.cell
def _(go, labels, make_subplots, mo, xgb_results_ml2):
    titles_xgb_total_ml2 = list(xgb_results_ml2.keys())
    fig_xgb_cm_total_ml2 = make_subplots(
        rows=2, cols=3, subplot_titles=titles_xgb_total_ml2
    )

    for i_xgb_total_ml2, (name_xgb_total_ml2, result_xgb_total_ml2) in enumerate(
        xgb_results_ml2.items()
    ):
        row_xgb_total_ml2 = i_xgb_total_ml2 // 3 + 1
        col_xgb_total_ml2 = i_xgb_total_ml2 % 3 + 1
        fig_xgb_cm_total_ml2.add_trace(
            go.Heatmap(
                z=result_xgb_total_ml2["cm"],
                x=labels,
                y=labels,
                text=result_xgb_total_ml2["cm"],
                texttemplate="%{text:,d}",
                colorscale="Blues",
                showscale=(i_xgb_total_ml2 == 0),
            ),
            row=row_xgb_total_ml2,
            col=col_xgb_total_ml2,
        )

    fig_xgb_cm_total_ml2.update_xaxes(title_text="Predicted")
    fig_xgb_cm_total_ml2.update_yaxes(title_text="True", row=1, col=1)
    fig_xgb_cm_total_ml2.update_yaxes(title_text="True", row=2, col=1)
    fig_xgb_cm_total_ml2.update_yaxes(showticklabels=False, row=1, col=2)
    fig_xgb_cm_total_ml2.update_yaxes(showticklabels=False, row=1, col=3)
    fig_xgb_cm_total_ml2.update_yaxes(showticklabels=False, row=2, col=2)
    fig_xgb_cm_total_ml2.update_yaxes(autorange="reversed")

    fig_xgb_cm_total_ml2.update_layout(
        height=800,
        width=1000,
        title_text="Confusion Matrices for ML2 Resampling Techniques (XGBoost) - Raw",
    )

    mo.ui.plotly(fig_xgb_cm_total_ml2)
    return


@app.cell
def _(go, labels, make_subplots, mo, xgb_results_ml2):
    titles_xgb_ml2 = list(xgb_results_ml2.keys())
    fig_xgb_cm_ml2 = make_subplots(rows=2, cols=3, subplot_titles=titles_xgb_ml2)

    for i_xgb_ml2, (name_xgb_ml2, result_xgb_ml2) in enumerate(xgb_results_ml2.items()):
        row_xgb_ml2 = i_xgb_ml2 // 3 + 1
        col_xgb_ml2 = i_xgb_ml2 % 3 + 1
        fig_xgb_cm_ml2.add_trace(
            go.Heatmap(
                z=result_xgb_ml2["cm_norm"],
                x=labels,
                y=labels,
                text=result_xgb_ml2["cm_norm"].round(2),
                texttemplate="%{text:.2%}",
                colorscale="Blues",
                zmin=0,
                zmax=1,
                showscale=(i_xgb_ml2 == 0),
            ),
            row=row_xgb_ml2,
            col=col_xgb_ml2,
        )

    fig_xgb_cm_ml2.update_xaxes(title_text="Predicted")
    fig_xgb_cm_ml2.update_yaxes(title_text="True", row=1, col=1)
    fig_xgb_cm_ml2.update_yaxes(title_text="True", row=2, col=1)
    fig_xgb_cm_ml2.update_yaxes(showticklabels=False, row=1, col=2)
    fig_xgb_cm_ml2.update_yaxes(showticklabels=False, row=1, col=3)
    fig_xgb_cm_ml2.update_yaxes(showticklabels=False, row=2, col=2)
    fig_xgb_cm_ml2.update_yaxes(autorange="reversed")

    fig_xgb_cm_ml2.update_layout(
        height=800,
        width=1000,
        title_text="Confusion Matrices for ML2 Resampling Techniques (XGBoost) - Normalized",
    )

    mo.ui.plotly(fig_xgb_cm_ml2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Cross Validation
    """)
    return


@app.cell
def _(
    ImbPipeline,
    RandomUnderSampler,
    StratifiedKFold,
    XGBClassifier,
    X_ml2,
    cross_val_score,
    y,
):
    xgb_ml2_cv_configs = [
        {
            "name": "XGBoost Undersampled",
            "model": ImbPipeline(
                [
                    ("undersampler", RandomUnderSampler(random_state=42)),
                    ("classifier", XGBClassifier(random_state=42)),
                ]
            ),
        },
        {
            "name": "XGBoost Class Weighted (autoscaled)",
            "model": XGBClassifier(
                random_state=42,
                scale_pos_weight=len(y[y == 0]) / len(y[y == 1]),
            ),
        },
    ]

    xgb_ml2_cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    xgb_ml2_cv_results = {}

    for xgb_ml2_cv_config in xgb_ml2_cv_configs:
        xgb_ml2_cv_results[xgb_ml2_cv_config["name"]] = {
            metric: cross_val_score(
                xgb_ml2_cv_config["model"], X_ml2, y, cv=xgb_ml2_cv_folds, scoring=metric
            )
            for metric in ["recall", "accuracy"]
        }
    return (xgb_ml2_cv_results,)


@app.cell
def _(mo, xgb_ml2_cv_results):
    xgb_ml2_cv_output = "**5 Fold Cross Validation Scores (ML2)**\n\n"

    for name, scores in xgb_ml2_cv_results.items():
        xgb_ml2_cv_output += f"**{name}:**\n"

        for metric, values in scores.items():
            xgb_ml2_cv_output += (
                f"- {metric.capitalize()} per fold: {[f'{s:.2%}' for s in values]}\n"
            )

        xgb_ml2_cv_output += f"- Mean recall: {scores['recall'].mean():.2%}\n"

        xgb_ml2_cv_output += f"- Std: {scores['recall'].std():.2%}\n\n"

    mo.md(xgb_ml2_cv_output)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    All variants for ML2 performed well! And we hit a ceiling of 96% recall with ML2 undersampled + weight 5 and weight 10.

    We also see that

    - Class weight (autoscaled) has the best precision (457/(457+646) ~= 41%) while maintaining ~79% recall. A strong result. This means that the team will spend less time on repeated calls to non-subscribers.
    - The undersampled version has ~90% recall (from cross validation) with a false positive rate of ~14% for non subscribers. A very strong result as well. It catches more subscribers than the autoscaled version but it also means the team will spend more time on repeated calls to non-subscribers.
    - Weights 5/10 hit 96% recall but at 84-89% false positive rates, making them impractical to use compared to the previous two models.

    Ultimately, it's a tradeoff between ML2 flagging less non-subscribers to call (more correct) but at the expense of catching less subscribers or flagging more non-subscribers to call at the expense of catching more subscribers.

    Now just as we did for ML1, let's see which features were most important to ML2.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Feature Importance
    """)
    return


@app.cell
def _(X_ml2, pd, xgb_results_ml2):
    ml2_fi_comparison = pd.DataFrame(
        {
            "feature": X_ml2.columns,
            "undersampled": xgb_results_ml2["Undersampled"]["model"].feature_importances_,
            "autoscaled": xgb_results_ml2["Class Weighted (autoscaled)"][
                "model"
            ].feature_importances_,
        }
    ).sort_values("undersampled", ascending=False)

    ml2_fi_comparison
    return (ml2_fi_comparison,)


@app.cell
def _(go, ml2_fi_comparison, mo):
    fig_fi_comparison = go.Figure(
        data=[
            go.Bar(
                name="Undersampled",
                x=ml2_fi_comparison["feature"],
                y=ml2_fi_comparison["undersampled"],
            ),
            go.Bar(
                name="Autoscaled",
                x=ml2_fi_comparison["feature"],
                y=ml2_fi_comparison["autoscaled"],
            ),
        ]
    )

    fig_fi_comparison.update_layout(
        barmode="group",
        title_text="ML2 Feature Importance: Undersampled vs Autoscaled",
        xaxis_title="Feature",
        yaxis_title="Importance",
        height=800,
        width=1000,
    )

    mo.ui.plotly(fig_fi_comparison)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Overall both models are similar in ranking with the `month_mar` feature dominating. Our earlier EDA also flagged this as having extremely high subscription rates.

    It's followed by `month_oct`, `month_jul`, `month_aug` and `duration` (`contact_unknown` is not a meaningful feature because it is not specific enough).

    `duration` is surprising because we had 0.33 Spearman correlation but lower importance compared to some of the month features, which suggests that month features are capturing some of the same information as duration (e.g. calls in Mar/Oct may be longer because people are more receptive in those months). It could be that the tree is splitting on month first, which reduces `duration`'s residual importance.

    Pre call features were weak predictors (bottom of the list).

    This shows us that the timing of the call matters more than duration or demographics.

    However, we note that due to some month features having small sample sizes, the models may be overfitting these. We've already seen from our cross validation results that we had low standard deviation across 5 folds so these models are stable and not overfitting but let's also try training without month features to see what happens.
    """)
    return


@app.cell
def _(
    RandomUnderSampler,
    XGBClassifier,
    confusion_matrix,
    df_collected,
    pd,
    train_test_split,
    y,
):
    ml2_features_no_month = [
        "age",
        "job",
        "marital",
        "education",
        "balance",
        "housing",
        "loan",
        "contact",
        "campaign",
        "duration",
    ]


    X_ml2_no_month = pd.get_dummies(
        df_collected.select(ml2_features_no_month).to_pandas(), drop_first=True
    )

    X_train_ml2_no_month, X_test_ml2_no_month, y_train_ml2_no_month, y_test_ml2_no_month = (
        train_test_split(X_ml2_no_month, y, test_size=0.2, random_state=42, stratify=y)
    )

    undersampled_data_ml2_no_month = RandomUnderSampler(random_state=42).fit_resample(
        X_train_ml2_no_month, y_train_ml2_no_month
    )

    no_month_configs = [
        {"name": "Undersampled", "data": undersampled_data_ml2_no_month, "params": {}},
        {
            "name": "Class Weighted (autoscaled)",
            "data": (X_train_ml2_no_month, y_train_ml2_no_month),
            "params": {
                "scale_pos_weight": len(y_train_ml2_no_month[y_train_ml2_no_month == 0])
                / len(y_train_ml2_no_month[y_train_ml2_no_month == 1])
            },
        },
    ]

    no_month_results = {}

    for config_no_month in no_month_configs:
        xgb_no_month_model = XGBClassifier(random_state=42, **config_no_month["params"])
        xgb_no_month_model.fit(*config_no_month["data"])
        y_pred_xgb_no_month = xgb_no_month_model.predict(X_test_ml2_no_month)
        cm_no_month = confusion_matrix(y_test_ml2_no_month, y_pred_xgb_no_month)
        cm_no_month_norm = cm_no_month.astype("float") / cm_no_month.sum(
            axis=1, keepdims=True
        )
        no_month_results[config_no_month["name"]] = {
            "model": xgb_no_month_model,
            "cm": cm_no_month,
            "cm_norm": cm_no_month_norm,
        }
    return (no_month_results,)


@app.cell
def _(go, labels, make_subplots, mo, no_month_results, xgb_results_ml2):
    cms = [
        (xgb_results_ml2["Undersampled"]["cm"], 1, 1),
        (no_month_results["Undersampled"]["cm"], 1, 2),
        (xgb_results_ml2["Undersampled"]["cm_norm"], 1, 3),
        (no_month_results["Undersampled"]["cm_norm"], 1, 4),
        (xgb_results_ml2["Class Weighted (autoscaled)"]["cm"], 2, 1),
        (no_month_results["Class Weighted (autoscaled)"]["cm"], 2, 2),
        (xgb_results_ml2["Class Weighted (autoscaled)"]["cm_norm"], 2, 3),
        (no_month_results["Class Weighted (autoscaled)"]["cm_norm"], 2, 4),
    ]

    fig_xgb_cm_ml2_no_month = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=[
            "Undersamp. w/ Month - Raw",
            "Undersamp. w/o Month - Raw",
            "Undersamp. w/ Month - Norm",
            "Undersamp. w/o Month - Norm",
            "Autoscaled w/ Month - Raw",
            "Autoscaled w/o Month - Raw",
            "Autoscaled w/ Month - Norm",
            "Autoscaled w/o Month - Norm",
        ],
    )

    for cm_data, row_ml2_cms, col_ml2_cms in cms:
        is_norm = col_ml2_cms >= 3
        fig_xgb_cm_ml2_no_month.add_trace(
            go.Heatmap(
                z=cm_data,
                x=labels,
                y=labels,
                text=cm_data.round(2) if is_norm else cm_data,
                texttemplate="%{text:.2%}" if is_norm else "%{text:,d}",
                colorscale="Blues",
                zmin=0 if is_norm else None,
                zmax=1 if is_norm else None,
                showscale=False,
            ),
            row=row_ml2_cms,
            col=col_ml2_cms,
        )


    fig_xgb_cm_ml2_no_month.update_xaxes(title_text="Predicted")
    fig_xgb_cm_ml2_no_month.update_yaxes(title_text="True", row=1, col=1)
    fig_xgb_cm_ml2_no_month.update_yaxes(title_text="True", row=2, col=1)
    fig_xgb_cm_ml2_no_month.update_yaxes(showticklabels=False, row=1, col=2)
    fig_xgb_cm_ml2_no_month.update_yaxes(showticklabels=False, row=1, col=3)
    fig_xgb_cm_ml2_no_month.update_yaxes(showticklabels=False, row=1, col=4)
    fig_xgb_cm_ml2_no_month.update_yaxes(showticklabels=False, row=2, col=2)
    fig_xgb_cm_ml2_no_month.update_yaxes(showticklabels=False, row=2, col=3)
    fig_xgb_cm_ml2_no_month.update_yaxes(showticklabels=False, row=2, col=4)
    fig_xgb_cm_ml2_no_month.update_yaxes(autorange="reversed")

    fig_xgb_cm_ml2_no_month.update_layout(
        height=800,
        width=1200,
        title_text="Confusion Matrices for XGBoost with and without Month Features",
    )

    mo.ui.plotly(fig_xgb_cm_ml2_no_month)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Removing the month feature results in:

    - Undersampled: Recall dropping from 92% to 85% (-7%) and false positive rate increasing from 14% to 18% (+4%)
    - Autoscaled: Recall dropping from 79% to 74% (-5%) and false positive rate increasing from 9% to 10% (+1%)

    Both models are meaningful without the month features showing that these features are not overfitting and that the signal is real. Duration and other features still provide a solid foundation.

    Note that the autoscaled model is more robust. It is less dependent on timing features and maintains a stable false positive rate. This means that call efficiency stays consistent even if the campaign timing changes in the future.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Timed Saved?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It's great to see the number of potential future subscribers we can catch and focus on but how many hours would we actually save?
    """)
    return


@app.cell
def _(df_collected, y_test):
    avg_duration_sec = df_collected["duration"].mean()
    avg_campaigns = df_collected["campaign"].mean()
    avg_repeat_campaigns = (
        avg_campaigns - 1
    )  # because the campaign count includes the current call, so repeat campaigns is one less than total campaigns.
    total_customers = df_collected.height
    scale = (
        total_customers / len(y_test)
    )  # scale up the test set predictions to the full dataset size to estimate total calls and time spent.

    # Baseline: call everyone
    baseline_calls = total_customers * avg_campaigns
    baseline_repeat_calls = total_customers * avg_repeat_campaigns
    baseline_hours = (baseline_calls * avg_duration_sec) / 3600
    baseline_repeat_hours = (baseline_repeat_calls * avg_duration_sec) / 3600
    return (
        avg_campaigns,
        avg_duration_sec,
        avg_repeat_campaigns,
        baseline_calls,
        baseline_hours,
        baseline_repeat_calls,
        baseline_repeat_hours,
        scale,
        total_customers,
    )


@app.cell
def _(
    avg_campaigns,
    avg_duration_sec,
    avg_repeat_campaigns,
    baseline_calls,
    baseline_hours,
    baseline_repeat_calls,
    baseline_repeat_hours,
    scale,
    xgb_results,
    xgb_results_ml2,
):
    xgb_time_rows = []

    for xgb1_time_name in ["Undersampled + Weight 2"]:
        xgb1_time_cm = xgb_results[xgb1_time_name]["cm"]
        xgb1_time_predicted_yes = (
            xgb1_time_cm[0][1] + xgb1_time_cm[1][1]
        ) * scale  # FP + TP scaled
        xgb1_time_calls_made = xgb1_time_predicted_yes * avg_campaigns
        xgb1_time_spent_hours = (xgb1_time_calls_made * avg_duration_sec) / 3600

        xgb_time_rows.append(
            {
                "model": f"ML1 ({xgb1_time_name})",
                "baseline_calls_no_ml": baseline_calls,  # 40,000 customers * average calls per campaign
                "calls_eliminated": baseline_calls - xgb1_time_calls_made,
                "total_calls_to_make": xgb1_time_calls_made,
                "customers_to_call": xgb1_time_predicted_yes,
                "pct_calls_eliminated(%)": (baseline_calls - xgb1_time_calls_made)
                / baseline_calls
                * 100,
                "total_time_spent_hours": xgb1_time_spent_hours,
                "total_time_saved_hours": baseline_hours - xgb1_time_spent_hours,
                "pct_time_saved(%)": (baseline_hours - xgb1_time_spent_hours)
                / baseline_hours
                * 100,
            }
        )

    for xgb2_time_name in ["Undersampled", "Class Weighted (autoscaled)"]:
        xgb2_time_cm = xgb_results_ml2[xgb2_time_name]["cm"]
        xgb2_time_predicted_yes = (xgb2_time_cm[0][1] + xgb2_time_cm[1][1]) * scale
        xgb2_time_calls_made = xgb2_time_predicted_yes * avg_repeat_campaigns
        xgb2_time_spent_hours = (xgb2_time_calls_made * avg_duration_sec) / 3600

        xgb_time_rows.append(
            {
                "model": f"ML2 ({xgb2_time_name})",
                "baseline_calls_no_ml": baseline_repeat_calls,  # 40,000 customers * (average calls per campaign - 1) because we are only saving repeat calls in ML2 since the first call has already been made for all customers.
                "calls_eliminated": baseline_repeat_calls - xgb2_time_calls_made,
                "total_calls_to_make": xgb2_time_calls_made,
                "customers_to_call": xgb2_time_predicted_yes,
                "pct_calls_eliminated(%)": (baseline_repeat_calls - xgb2_time_calls_made)
                / baseline_repeat_calls
                * 100,
                "total_time_spent_hours": xgb2_time_spent_hours,
                "total_time_saved_hours": baseline_repeat_hours - xgb2_time_spent_hours,
                "pct_time_saved(%)": (baseline_repeat_hours - xgb2_time_spent_hours)
                / baseline_repeat_hours
                * 100,
            }
        )
    return (xgb_time_rows,)


@app.cell
def _(pd, xgb_time_rows):
    time_savings_df = pd.DataFrame(xgb_time_rows).round(0)
    time_savings_df.sort_values("total_time_saved_hours", ascending=False)
    return (time_savings_df,)


@app.cell
def _(
    baseline_repeat_hours,
    go,
    make_subplots,
    mo,
    time_savings_df,
    total_customers,
):
    fig_savings = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Customers Call List", "Repeat Call Hours"],
    )

    fig_savings.add_trace(
        go.Bar(
            x=["Baseline", "After ML1"],
            y=[total_customers, time_savings_df.iloc[0]["customers_to_call"]],
            text=[
                f"{total_customers:,}",
                f"{time_savings_df.iloc[0]['customers_to_call']:,.0f} ({time_savings_df.iloc[0]['pct_calls_eliminated(%)']:.0f}% eliminated)",
            ],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig_savings.add_trace(
        go.Bar(
            x=["Baseline\n(All Repeat Calls)", "ML2\n(Undersampled)", "ML2\n(Autoscaled)"],
            y=[
                baseline_repeat_hours,
                time_savings_df.iloc[1]["total_time_spent_hours"],
                time_savings_df.iloc[2]["total_time_spent_hours"],
            ],
            text=[
                f"{baseline_repeat_hours:,.0f}h",
                f"{time_savings_df.iloc[1]['total_time_spent_hours']:,.0f}h ({time_savings_df.iloc[1]['pct_time_saved(%)']:.0f}% saved)",
                f"{time_savings_df.iloc[2]['total_time_spent_hours']:,.0f}h ({time_savings_df.iloc[2]['pct_time_saved(%)']:.0f}% saved)",
            ],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig_savings.update_yaxes(title_text="Customers", row=1, col=1)
    fig_savings.update_yaxes(title_text="Hours", row=1, col=2)

    fig_savings.update_layout(height=800, width=1400)

    mo.ui.plotly(fig_savings)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We see that after scaling,

    - ML1 reduces the original customers call list from 40,000 to 27,195 (~32% reduction) cutting total initial call hours from 8161 to 5548 (~32% reduction).

    ML2 further optimizes by identifying which customers are worth calling repeatedly:

    - ML2 undersampled: 7725 customers flagged for follow ups reducing repeat call hours from 5329 to 1029 (~81% reduction).
    - ML2 autoscaled: 5515 customers flagged and repeat call hours are reduced to 735 (from 5329. ~86% reduction).

    ML2 autoscaled saves the most time while maintaining ~79% recall on subscribers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Customer Segmentation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that we have our models, we can use them to identify the high probability subscribers and understand what types of customers they are.

    Clustering them allows us to create actionable customer profiles that help the team tailor their approach for each segment.

    We start by filtering out non subscribers then use KMeans and hierarchical clustering to cluster subscribers and finally compare both to see if they produce similar clusters.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Filtering Non subscribers
    """)
    return


@app.cell
def _(StandardScaler, db, pd):
    subscribers = db.sql("""
        SELECT * FROM df_collected WHERE y = 'yes'
        """).pl()

    # We use the pre call features because we're interested in understanding the characteristics of customers who subscribe before the call begins.
    cluster_features = ["age", "balance", "job", "marital", "education", "housing", "loan"]

    # We need to encode and scale the features for KMeans because it uses distance (features with larger ranges would dominate over binary features).
    subscribers_encoded = pd.get_dummies(
        subscribers.select(cluster_features).to_pandas(), drop_first=True
    )
    subscribers_scaled = StandardScaler().fit_transform(subscribers_encoded)
    return subscribers, subscribers_scaled


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### KMeans Clustering
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Elbow (Inertia) Method and Silhouette Scores
    """)
    return


@app.cell
def _(KMeans, silhouette_score, subscribers_scaled):
    # We also need to find the optimal number of clusters so we can use elbow method (plotting inertia for different k) for KMeans and silhouette scores (measures how similar each point is to its own cluster vs the nearest other cluster) for both KMeans and Hierarchical Clustering later (use dendrogram instead of elbow/inertia).

    k_range = range(
        2, 11
    )  # we need at least 2 clusters to segment customers, and we can try up to 10 clusters to see if there are more granular segments that are meaningful for marketing strategies. Having more than 10 clusters might be too impractical.
    inertias = []
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        k_labels = kmeans.fit_predict(subscribers_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(subscribers_scaled, k_labels))
    return inertias, k_range, silhouette_scores


@app.cell
def _(go, inertias, k_range, make_subplots, mo, silhouette_scores):
    fig_cluster = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Elbow Method", "Silhouette Score"],
    )

    fig_cluster.add_trace(
        go.Scatter(x=list(k_range), y=inertias, mode="lines+markers"),
        row=1,
        col=1,
    )

    fig_cluster.add_trace(
        go.Scatter(x=list(k_range), y=silhouette_scores, mode="lines+markers"),
        row=1,
        col=2,
    )

    fig_cluster.update_xaxes(title_text="k", dtick=1)
    fig_cluster.update_yaxes(title_text="Inertia", row=1, col=1)
    fig_cluster.update_yaxes(title_text="Silhouette Score", row=1, col=2)

    fig_cluster.update_layout(height=600, width=1000, showlegend=False)

    mo.ui.plotly(fig_cluster)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We see that:

    - Elbow: There's no clear elbow. Inertia decreases monotonically.
    - Silhouette: Monotonically increasing and peaks at k=10 (score ~ 0.29) suggesting that the data doesn't have well separated natural clusters and it could be beneficial to add more clusters. We should aim for a score of > 0.5 (typical good score).

    Because of the nature of customer data (usually continuous and overlapping rather than neatly grouped), we see that there are no clear distinct clusters in the feature space.

    Let's first try using k=4 because it gives the team a manageable number of segments to use.

    We can see if any meaningful clusters appear and adjust accordingly once we see the result.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Clustering with K=4
    """)
    return


@app.cell
def _(KMeans, subscribers, subscribers_scaled):
    kmeans_4 = KMeans(n_clusters=4, random_state=42, n_init=10)
    subscribers_with_clusters = subscribers.to_pandas().assign(
        cluster=kmeans_4.fit_predict(subscribers_scaled)
    )

    cluster_profiles = (
        subscribers_with_clusters.groupby("cluster")
        .agg(
            age=("age", "mean"),
            balance=("balance", "mean"),
            job=("job", lambda x: x.mode()[0]),
            marital=("marital", lambda x: x.mode()[0]),
            education=("education", lambda x: x.mode()[0]),
            housing=("housing", lambda x: x.mode()[0]),
            loan=("loan", lambda x: x.mode()[0]),
            count=("age", "size"),
        )
        .round(2)
        .sort_values("count", ascending=False)
    )
    return cluster_profiles, subscribers_with_clusters


@app.cell
def _(cluster_profiles):
    cluster_profiles
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - The largest cluster consists of blue collar married homeowners with debt (1091).
    - The second largest are educated, married people in management with no debt (754). They have one of the higher balances among all clusters.
    - Next are young singles with secondary education (736).
    - Finally we have wealthy married retirees with tertiary education and no debt (315).

    These 4 clusters naturally separate based on age, job, type, balance, marital status and housing, aligning with our EDA findings.

    Before we decide to try other k values, let's try hierarchical clustering using dendrogram first to see what the natural cut points are and how many clusters the data naturally supports.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hierarchical Clustering
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Initially, I tried to create the dendrogram with all 2900 subscribers but it kept crashing my VSCode.

    For visualization purposes, I render the dendrogram on a random subsample of 500. The actual cluster assignments (afterwards) uses the full dataset.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Dendrogram
    """)
    return


@app.cell
def _(mo):
    run_dendrogram_button = mo.ui.run_button(label="Run Dendrogram")

    run_dendrogram_button
    return (run_dendrogram_button,)


@app.cell
def _(linkage, np, subscribers_scaled):
    np.random.seed(42)

    sample_idx = np.random.choice(len(subscribers_scaled), size=500, replace=False)
    subscribers_sample = subscribers_scaled[sample_idx]

    linkage_matrix = linkage(subscribers_scaled, method="ward")
    return linkage_matrix, subscribers_sample


@app.cell
def _(ff, linkage, mo, run_dendrogram_button, subscribers_sample):
    mo.stop(
        not run_dendrogram_button.value,
        "Click the button above to run the dendrogram visualization.",
    )

    fig_dendrogram = ff.create_dendrogram(
        subscribers_sample,
        linkagefun=lambda x: linkage(x, method="ward"),
    )

    fig_dendrogram.update_layout(
        height=800,
        width=1200,
        title_text="Hierarchical Clustering: Dendrogram (Sample of 500 Subscribers)",
    )

    mo.ui.plotly(fig_dendrogram)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The dendrogram shows us that:

    - At distance ~45, we have 2 clusters
    - At distance ~35, we have 3 clusters
    - At distance ~28, we have 4 clusters. This validates our choice of k=4.

    Now let's assign some labels to the cluster like we did above with KMeans `fit_predict` (make the cluster profiles) so that we can have actionable results.

    For now, we use the same cut as we did for KMeans.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Clustering with K=4
    """)
    return


@app.cell
def _(fcluster, linkage_matrix, subscribers_with_clusters):
    hierarchical_labels = fcluster(linkage_matrix, t=4, criterion="maxclust")

    subscribers_with_hclusters = subscribers_with_clusters.assign(
        h_cluster=hierarchical_labels
        - 1  # to make hierarchical cluster labels start from 0 like KMeans clusters
    )
    return (subscribers_with_hclusters,)


@app.cell
def _(subscribers_with_hclusters):
    h_cluster_profiles = (
        subscribers_with_hclusters.groupby("h_cluster")
        .agg(
            age=("age", "mean"),
            balance=("balance", "mean"),
            job=("job", lambda x: x.mode()[0]),
            marital=("marital", lambda x: x.mode()[0]),
            education=("education", lambda x: x.mode()[0]),
            housing=("housing", lambda x: x.mode()[0]),
            loan=("loan", lambda x: x.mode()[0]),
            count=("age", "size"),
        )
        .round(2)
        .sort_values("count", ascending=False)
    )
    return (h_cluster_profiles,)


@app.cell
def _(h_cluster_profiles):
    h_cluster_profiles
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Hierarchical clustering produces similar archetypes to KMeans but with a very different size distribution:

    - 1761 blue collar married homeowners (61% of all subscribers)
    - 911 educated married management with tertiary education
    - 144 married retirees with higher balances
    - 80 young single students. Small but tightly distinct

    Compared to KMeans's more balanced split (1091/754/736/315), hierarchical clustering concentrates most subscribers into one dominant group and surfaces students as a small, well defined segment.
    """)
    return


@app.cell
def _(adjusted_rand_score, pd, subscribers_with_hclusters):
    cross_tab_clustering = pd.crosstab(
        subscribers_with_hclusters["cluster"],
        subscribers_with_hclusters["h_cluster"],
        rownames=["KMeans Cluster"],
        colnames=["Hierarchical Cluster"],
    )

    adj_rand = adjusted_rand_score(
        subscribers_with_hclusters["cluster"], subscribers_with_hclusters["h_cluster"]
    )
    return adj_rand, cross_tab_clustering


@app.cell
def _(adj_rand, cross_tab_clustering, ff, mo):
    fig_cross_tab = ff.create_annotated_heatmap(
        z=cross_tab_clustering.values,
        x=[f"H-Cluster {c}" for c in cross_tab_clustering.columns],
        y=[f"KMeans {c}" for c in cross_tab_clustering.index],
        colorscale="Blues",
        showscale=True,
    )

    fig_cross_tab.update_layout(
        title=f"KMeans vs Hierarchical Cluster Agreement (Adjusted Rand Index = {adj_rand:.3f})",
        xaxis_title="Hierarchical Cluster",
        yaxis_title="KMeans Cluster",
        height=800,
        width=800,
    )

    mo.ui.plotly(fig_cross_tab)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Our ARI of ~0.413 suggests moderate agreement where the two methods identify similar archetypes but differ in how they split certain groups.

    This is expected given that the data lacks clean natural clusters (silhouette peak < 0.3).

    The heatmap confirms this:

    - KMeans 0 (Married Retirees) splits roughly evenly between HCluster 0 and HCluster 1. The methods disagree on how to group these subscribers.
    - KMeans 1 (Wealthy Married Management) largely maps to HCluster 0 (675 points).
    - KMeans 2 (Blue Collar Homeowners) maps almost entirely to HCluster 3 (1,080 points) which is the strongest agreement between the two methods.
    - KMeans 3 (Young Singles) spreads across three hierarchical clusters showing that this is the most fragmented group.

    <br/>

    **Note**: The Adjusted Rand Index (ARI) measures how well two clusterings agree, adjusted for chance (0 = random, 1 = perfect match).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Visualizing the Segmentations
    """)
    return


@app.cell
def _(TSNE, pd, subscribers_scaled, subscribers_with_hclusters):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_embedding = tsne.fit_transform(subscribers_scaled)

    kmeans_labels_tsne = {
        "0": "Married Retirees",
        "1": "Wealthy Married Management",
        "2": "Blue Collar Homeowners",
        "3": "Young Singles",
    }

    h_labels_tsne = {
        "0": "Married Management",
        "1": "Married Retirees",
        "2": "Young Single Students",
        "3": "Blue Collar Homeowners",
    }

    tsne_df = pd.DataFrame(tsne_embedding, columns=["TSNE_1", "TSNE_2"]).assign(
        kmeans_cluster=subscribers_with_hclusters.cluster.astype(str).map(
            kmeans_labels_tsne
        ),
        h_cluster=subscribers_with_hclusters.h_cluster.astype(str).map(h_labels_tsne),
    )
    return h_labels_tsne, kmeans_labels_tsne, tsne_df


@app.cell
def _(go, h_labels_tsne, kmeans_labels_tsne, make_subplots, mo, tsne_df):
    fig_tsne = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["KMeans Clusters", "Hierarchical Clusters"],
    )

    for cluster_id, label in kmeans_labels_tsne.items():
        subset_ = tsne_df[tsne_df.kmeans_cluster == label]
        fig_tsne.add_trace(
            go.Scatter(
                x=subset_.TSNE_1,
                y=subset_.TSNE_2,
                mode="markers",
                name=f"{cluster_id}: {label}",
                legendgroup="kmeans",
                legendgrouptitle_text="KMeans",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    for cluster_id, label in h_labels_tsne.items():
        subset__ = tsne_df[tsne_df.h_cluster == label]
        fig_tsne.add_trace(
            go.Scatter(
                x=subset__.TSNE_1,
                y=subset__.TSNE_2,
                mode="markers",
                name=f"{cluster_id}: {label}",
                legendgroup="hierarchical",
                legendgrouptitle_text="Hierarchical",
                showlegend=True,
            ),
            row=1,
            col=2,
        )


    fig_tsne.update_layout(
        title="t-SNE: KMeans vs Hierarchical Clustering",
        height=600,
        width=1200,
    )

    fig_tsne.update_xaxes(title_text="t-SNE Dimension 1")
    fig_tsne.update_yaxes(title_text="t-SNE Dimension 2")

    mo.ui.plotly(fig_tsne)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We see the following:

    - Lots of tightly segmented clusters suggesting that the real underlying structure is more granular than 4 segments (k=4).
    - Hierarchical clustering shows much cleaner spatial separation where blue collared homeowners (green) covers the majority of the plot and young single students (pink) emerges as a tight isolated group showing that this group is very distinct.

    Now let's compare this result with Principal Component Analysis (PCA) and Uniform Manifold Approximation and Projection (UMAP).
    """)
    return


@app.cell
def _(
    PCA,
    UMAP,
    kmeans_labels_tsne,
    pd,
    subscribers_scaled,
    subscribers_with_hclusters,
    tsne_df,
):
    pca_embedding = PCA(n_components=2, random_state=42).fit_transform(subscribers_scaled)
    umap_embedding = UMAP(n_components=2, random_state=42).fit_transform(subscribers_scaled)

    comparison_df = pd.DataFrame(
        {
            "PCA_1": pca_embedding[:, 0],
            "PCA_2": pca_embedding[:, 1],
            "TSNE_1": tsne_df.TSNE_1,
            "TSNE_2": tsne_df.TSNE_2,
            "UMAP_1": umap_embedding[:, 0],
            "UMAP_2": umap_embedding[:, 1],
            "kmeans_cluster": subscribers_with_hclusters.cluster.astype(str).map(
                kmeans_labels_tsne
            ),
        }
    )
    (comparison_df,)
    return (comparison_df,)


@app.cell
def _(comparison_df, go, kmeans_labels_tsne, make_subplots, mo):
    color_map_dr = {
        "0": "#636EFA",
        "1": "#EF553B",
        "2": "#00CC96",
        "3": "#AB63FA",
    }

    panels = [
        ("PCA_1", "PCA_2", 1),
        ("TSNE_1", "TSNE_2", 2),
        ("UMAP_1", "UMAP_2", 3),
    ]

    fig_dr = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["PCA", "t-SNE", "UMAP"],
    )

    for x_col_dr, y_col_dr, col_idx_dr in panels:
        for cluster_id_dr, label_dr in kmeans_labels_tsne.items():
            subset_dr = comparison_df[comparison_df.kmeans_cluster == label_dr]
            fig_dr.add_trace(
                go.Scatter(
                    x=subset_dr[x_col_dr],
                    y=subset_dr[y_col_dr],
                    mode="markers",
                    name=f"{cluster_id_dr}: {label_dr}",
                    legendgroup=label_dr,
                    showlegend=(col_idx_dr == 1),
                    marker=dict(color=color_map_dr[cluster_id_dr], size=5),
                ),
                row=1,
                col=col_idx_dr,
            )

    fig_dr.update_xaxes(title_text="PCA Dimension 1", row=1, col=1)
    fig_dr.update_yaxes(title_text="PCA Dimension 2", row=1, col=1)
    fig_dr.update_xaxes(title_text="t-SNE Dimension 1", row=1, col=2)
    fig_dr.update_yaxes(title_text="t-SNE Dimension 2", row=1, col=2)
    fig_dr.update_xaxes(title_text="UMAP Dimension 1", row=1, col=3)
    fig_dr.update_yaxes(title_text="UMAP Dimension 2", row=1, col=3)

    fig_dr.update_layout(
        title="Dimensionality Reduction Comparison (colored by KMeans, k=4)",
        height=800,
        width=2000,
        legend=dict(title="KMeans Cluster"),
    )

    mo.ui.plotly(fig_dr)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Comparing the three methods, we see that:

    - PCA shows heavy overlap between clusters, confirming that linear projection can't cleanly separate the segments.
    - t-SNE and UMAP reveal many tight subclusters, showing that the real feature space structure is finer than k=4.
    - The same KMeans cluster appears as multiple spatially disconnected islands, meaning KMeans is producing semantically coherent but spatially fragmented groups.e.g. "Blue Collar Homeowners" captures multiple sub types that share a demographic profile but live in different regions of the feature space.

    **Note:** Our silhouette analysis never exceeded 0.29 at any k, well below the 0.5 threshold for well separated clusters. This means the data doesn't have clean natural clusters regardless of k. We chose k=4 as a pragmatic compromise which was statistically reasonable (validated by the dendrogram) and gives the team a manageable number of actionable segments. Going to k=10 would provide marginally better statistical separation but would be impractical for campaign targeting.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Conclusion
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We built a three part system to address the 7% subscriber rate problem:

    ML1: Pre call filter (XGBoost + undersampling + class weight 2)

    - Catches ~76% of potential subscribers using only pre call features
    - Reduces the call list from 40,000 to ~27,000 (~32% reduction)

    ML2: Call optimizer (XGBoost + undersampling)

    - Catches ~91% of subscribers with ~86% accuracy (5-fold CV), exceeding the 81% target
    - Uses all features including call duration and campaign context

    Customer Segmentation: KMeans + Hierarchical Clustering

    - Identified 4 interpretable segments: Wealthy Married Management, Blue Collar Homeowners, Young Singles, and Married Retirees
    - Validated with hierarchical clustering (dendrogram) and cross method visualization (PCA, t-SNE, UMAP)
    - Caveat: silhouette scores stayed below 0.3 at all k values meaning segments are directional rather than perfectly separated

    <br/>

    Key Findings

    - Timing matters most: March and October had the highest conversions and dominated ML2's feature importance.
    - High conversion segments: Students (15.6%) and retirees (10.5%) had the highest conversion rates. Management has the highest volume and a strong conversion rate.
    - Pre call features contribute roughly equally: no single demographic predictor dominates which explains ML1's lower recall compared to ML2.

    Final Recommendations

    1. Deploy ML1 to filter the call list before each campaign saving ~13,000 unnecessary calls.
    2. Use ML2 during campaigns to guide follow up decisions on live calls.
    3. Concentrate campaigns in March and October (the two highest converting months).
    4. Prioritize management, students, and retirees (highestconversion segments).
    """)
    return


if __name__ == "__main__":
    app.run()
