import marimo

__generated_with = "0.22.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from lazypredict.Supervised import LazyClassifier
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix, precision_recall_curve
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek, SMOTEENN
    from imblearn.pipeline import Pipeline as ImbPipeline
    from xgboost import XGBClassifier

    return (
        ImbPipeline,
        LazyClassifier,
        RandomForestClassifier,
        RandomUnderSampler,
        SMOTEENN,
        SMOTETomek,
        StratifiedKFold,
        XGBClassifier,
        confusion_matrix,
        cross_val_score,
        go,
        make_subplots,
        mo,
        pd,
        pl,
        precision_recall_curve,
        px,
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

    **Note**: `duration` is the length of the phone call in seconds. This is a target leakage variable because we only know the value after the call has ended. At prediction time (before calling a customer), this feature won't exist yet. It should be excluded from the final model or carefully evaluated (test with and without) otherwise it will likely inflate accuracy artificially.
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

    fig_violin.update_yaxes(title_text="Balance", row=1, col=2, range=[-800, 5000])
    fig_violin.update_yaxes(title_text="Duration", row=2, col=1, range=[0, 1000])
    fig_violin.update_yaxes(title_text="Campaign", row=2, col=2, range=[0, 20])

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

    - Balance, duration and campaign contain extreme values (kurtosis ~36-142). Valid according to the domain and not data errors. We will approach it more rigorously by using sensitivity analysis during our modeling phase.

    - Features with the strongest signals: job, education, marital status and housing.

    - Weak individual correlations with the target y: all features were < 0.06 Spearman (not including duration). Feature interactions will probably be necessary and tree based ensemble methods will most likely be the best model.

    - Default is dropped because it has essentially no variance.
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
    2. ML2 (Optimizer): Use features known after a call (contact, campaign, month, duration) and prioritize customers to keep calling. That is, which customers are more likely to say yes to subscribing and keep calling these type of customers.
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
    #### LazyPredict
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
    #### Resampling
    """)
    return


@app.cell
def _(
    RandomForestClassifier,
    RandomUnderSampler,
    X_test,
    X_train,
    confusion_matrix,
    y_test,
    y_train,
):
    undersampled_data = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)
    rf_undersampled = RandomForestClassifier(random_state=42)
    rf_undersampled.fit(*undersampled_data)
    y_pred_rf = rf_undersampled.predict(X_test)

    cm_rf_undersampled = confusion_matrix(y_test, y_pred_rf)
    cm_rf_undersampled_normalized = cm_rf_undersampled.astype(
        "float"
    ) / cm_rf_undersampled.sum(axis=1, keepdims=True)
    return (
        cm_rf_undersampled,
        cm_rf_undersampled_normalized,
        rf_undersampled,
        undersampled_data,
    )


@app.cell
def _(
    RandomForestClassifier,
    X_test,
    X_train,
    confusion_matrix,
    y_test,
    y_train,
):
    rf_balanced = RandomForestClassifier(class_weight="balanced", random_state=42)
    rf_balanced.fit(X_train, y_train)
    y_pred_rf_balanced = rf_balanced.predict(X_test)

    cm_rf_balanced = confusion_matrix(y_test, y_pred_rf_balanced)
    cm_rf_balanced_normalized = cm_rf_balanced.astype("float") / cm_rf_balanced.sum(
        axis=1, keepdims=True
    )
    return cm_rf_balanced, cm_rf_balanced_normalized


@app.cell
def _(
    RandomForestClassifier,
    SMOTETomek,
    X_test,
    X_train,
    confusion_matrix,
    y_test,
    y_train,
):
    smote_tomek = SMOTETomek(random_state=42)
    X_resampled_smote_tomek, y_resampled_smote_tomek = smote_tomek.fit_resample(
        X_train, y_train
    )
    rf_smote_tomek = RandomForestClassifier(random_state=42)
    rf_smote_tomek.fit(X_resampled_smote_tomek, y_resampled_smote_tomek)

    y_pred_rf_smote_tomek = rf_smote_tomek.predict(X_test)

    cm_rf_smote_tomek = confusion_matrix(y_test, y_pred_rf_smote_tomek)
    cm_rf_smote_tomek_normalized = cm_rf_smote_tomek.astype(
        "float"
    ) / cm_rf_smote_tomek.sum(axis=1, keepdims=True)
    return cm_rf_smote_tomek, cm_rf_smote_tomek_normalized


@app.cell
def _(
    RandomForestClassifier,
    SMOTEENN,
    X_test,
    X_train,
    confusion_matrix,
    y_test,
    y_train,
):
    smote_enn = SMOTEENN(random_state=42)
    X_resampled_smote_enn, y_resampled_smote_enn = smote_enn.fit_resample(X_train, y_train)
    rf_smote_enn = RandomForestClassifier(random_state=42)
    rf_smote_enn.fit(X_resampled_smote_enn, y_resampled_smote_enn)

    y_pred_rf_smote_enn = rf_smote_enn.predict(X_test)

    cm_rf_smote_enn = confusion_matrix(y_test, y_pred_rf_smote_enn)
    cm_rf_smote_enn_normalized = cm_rf_smote_enn.astype("float") / cm_rf_smote_enn.sum(
        axis=1, keepdims=True
    )
    return cm_rf_smote_enn, cm_rf_smote_enn_normalized


@app.cell
def _(mo):
    mo.md(r"""
    ##### Confusion Matrices
    """)
    return


@app.cell
def _(
    cm_rf_balanced,
    cm_rf_smote_enn,
    cm_rf_smote_tomek,
    cm_rf_undersampled,
    go,
    make_subplots,
    mo,
):
    labels = ["No Subscription", "Subscription"]

    fig_rf_resampled_total = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Undersampling",
            "Balanced Weights",
            "SMOTE-Tomek",
            "SMOTE-ENN",
        ],
    )

    for i_resampled_total, cm_norm_resampled_total in enumerate(
        [
            cm_rf_undersampled,
            cm_rf_balanced,
            cm_rf_smote_tomek,
            cm_rf_smote_enn,
        ]
    ):
        row_resampled_total = i_resampled_total // 2 + 1
        col_resampled_total = i_resampled_total % 2 + 1
        fig_rf_resampled_total.add_trace(
            go.Heatmap(
                z=cm_norm_resampled_total,
                x=labels,
                y=labels,
                text=cm_norm_resampled_total,
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
    fig_rf_resampled_total.update_yaxes(autorange="reversed")

    fig_rf_resampled_total.update_layout(
        height=1000,
        width=1200,
        title_text="Confusion Matrices for Resampling Techniques (Random Forest) - Raw",
    )

    mo.ui.plotly(fig_rf_resampled_total)
    return (labels,)


@app.cell
def _(
    cm_rf_balanced_normalized,
    cm_rf_smote_enn_normalized,
    cm_rf_smote_tomek_normalized,
    cm_rf_undersampled_normalized,
    go,
    labels,
    make_subplots,
    mo,
):
    fig_rf_resampled = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Undersampling",
            "Balanced Weights",
            "SMOTE-Tomek",
            "SMOTE-ENN",
        ],
    )

    for i_resampled, cm_norm_resampled in enumerate(
        [
            cm_rf_undersampled_normalized,
            cm_rf_balanced_normalized,
            cm_rf_smote_tomek_normalized,
            cm_rf_smote_enn_normalized,
        ]
    ):
        row_resampled = i_resampled // 2 + 1
        col_resampled = i_resampled % 2 + 1
        fig_rf_resampled.add_trace(
            go.Heatmap(
                z=cm_norm_resampled,
                x=labels,
                y=labels,
                text=cm_norm_resampled.round(2),
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
    fig_rf_resampled.update_yaxes(autorange="reversed")

    fig_rf_resampled.update_layout(
        height=1000,
        width=1200,
        title_text="Confusion Matrices for Resampling Techniques (Random Forest) - Normalized",
    )

    mo.ui.plotly(fig_rf_resampled)
    return


@app.cell
def _(mo):
    mo.md(r"""
    We see that 1:1 undersampling is the best strategy, identifying ~56% of subscribers. All other strategies are mostly predicting 'no' for most observations.

    However 56% recall with 43% false positive rate is still not great. It's almost the same as random chance (50%).

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
def _(X_test, precision_recall_curve, rf_undersampled, y_test):
    y_prob_rf = rf_undersampled.predict_proba(X_test)[:, 1]
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

    - At threshold ~0.2, we catch ~92% of subscribers (recall ~92%) but ~93% would be predicted 'yes' are false positives (~%7 precision).
    - At threshold ~0.5 (default), we catch ~53% with precision ~%9, slightly better precision.
    - Higher thresholds reduce calls but miss more subscribers.

    Random Forest with undersampling provided us with a baseline.

    We now try XGBoost, which uses gradient boosting (seqential tree building that corrects previous errors) with the `scale_pos_weight` parameter, which allows us to control emphasis on predicting subscribers.

    First we try with undersampling, then with class weight tuning (without undersample) and lastly with both.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### XGBoost
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
    fig_xgb_cm_total.update_yaxes(autorange="reversed")

    fig_xgb_cm_total.update_layout(
        height=1000,
        width=1200,
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
    fig_xgb_cm.update_yaxes(autorange="reversed")

    fig_xgb_cm.update_layout(
        height=1000,
        width=1200,
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
    - Class weight = autoscaled, performed worst than our baseline.

    Class weight = 2 might be the sweet spot for us.

    Let's validate these results with cross validation and check if the 77% recall is actually stable or just lucky.
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
    mo,
    y,
):
    xgb_cv_pipeline = ImbPipeline(  # Use imbpipeline to ensure that the undersampling is done within each fold of the cross-validation, preventing data leakage.
        [
            ("undersampler", RandomUnderSampler(random_state=42)),
            ("classifier", XGBClassifier(random_state=42, scale_pos_weight=2)),
        ]
    )

    xgb_cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    xgb_recall_scores_cv = cross_val_score(
        xgb_cv_pipeline, X, y, cv=xgb_cv_folds, scoring="recall"
    )

    mo.md(f"""
    **5 Fold Cross Validation Scores (XGBoost Undersampled + Weight 2)**

    - Recall per fold: {[f"{s:.2%}" for s in xgb_recall_scores_cv]}
    - Mean recall: {xgb_recall_scores_cv.mean():.2%}
    - Std: {xgb_recall_scores_cv.std():.2%}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The results are verified and we get ~76% recall consistently. So ML1 can now identify ~3 out of 4 potential subscribers before we make any calls.

    Applied to the full 40,000 customers, this means that

    - ~2235 out of ~2895 actual subscribers would be correctly flagged for calling
    - The call list would be reduced from 40,000 to 27,195
    - ~660 subscribers would be missed

    Now that we have a reduced list of customers to call, we develop ML2, which will help us identify which customers are more likely to say yes and to keep calling.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ML2: Optimizer
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
        "day",
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
    fig_xgb_cm_total_ml2.update_yaxes(autorange="reversed")

    fig_xgb_cm_total_ml2.update_layout(
        height=1000,
        width=1200,
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
    fig_xgb_cm_ml2.update_yaxes(autorange="reversed")

    fig_xgb_cm_ml2.update_layout(
        height=1000,
        width=1200,
        title_text="Confusion Matrices for ML2 Resampling Techniques (XGBoost) - Normalized",
    )

    mo.ui.plotly(fig_xgb_cm_ml2)
    return


@app.cell
def _(
    ImbPipeline,
    RandomUnderSampler,
    StratifiedKFold,
    XGBClassifier,
    X_ml2,
    cross_val_score,
    mo,
    y,
):
    xgb_ml2_cv_undersampled = ImbPipeline(
        [
            ("undersampler", RandomUnderSampler(random_state=42)),
            ("classifier", XGBClassifier(random_state=42)),
        ]
    )

    xgb_ml2_cv_weighted_auto = XGBClassifier(
        random_state=42,
        scale_pos_weight=len(y[y == 0]) / len(y[y == 1]),
    )

    xgb_ml2_cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    xgb_ml2_cv_scores_undersampled = cross_val_score(
        xgb_ml2_cv_undersampled, X_ml2, y, cv=xgb_ml2_cv_folds, scoring="recall"
    )

    xgb_ml2_recall_scores_weighted_cv_auto = cross_val_score(
        xgb_ml2_cv_weighted_auto, X_ml2, y, cv=xgb_ml2_cv_folds, scoring="recall"
    )

    mo.md(f"""
    5 Fold Cross Validation Scores (ML2)

    XGBoost Undersampled:
    - Recall per fold: {[f"{s:.2%}" for s in xgb_ml2_cv_scores_undersampled]}
    - Mean recall: {xgb_ml2_cv_scores_undersampled.mean():.2%}
    - Std: {xgb_ml2_cv_scores_undersampled.std():.2%}

    XGBoost Class Weighted (autoscaled without undersampling):
    - Recall per fold: {[f"{s:.2%}" for s in xgb_ml2_recall_scores_weighted_cv_auto]}
    - Mean recall: {xgb_ml2_recall_scores_weighted_cv_auto.mean():.2%}
    - Std: {xgb_ml2_recall_scores_weighted_cv_auto.std():.2%}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    All variants performed well! And we hit a ceiling of 96% recall in the undersampled + weight 5 and 10.

    We also see that

    - Class weight (auto) has the best precision (457/(457+646) ~= 41%) while maintaining ~79% recall. A strong result.
    - The undersampled version has ~90% recall with a false positive rate of ~14%. A very strong result and is our best choice as verified by cross validation. It also corresponds with ML2's goal of catching more subscribers.
    """)
    return


if __name__ == "__main__":
    app.run()
