import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

TARGET = "購入フラグ"

NUMERICAL_FEATURES = [
    "従業員数",
    "事業所数",
    "工場数",
    "店舗数",
    "資本金",
    "総資産",
    "流動資産",
    "固定資産",
    "負債",
    "短期借入金",
    "長期借入金",
    "純資産",
    "自己資本",
    "売上",
    "営業利益",
    "経常利益",
    "当期純利益",
    "営業CF",
    "減価償却費",
    "運転資本変動",
    "投資CF",
    "有形固定資産変動",
    "無形固定資産変動(ソフトウェア関連)",
    "アンケート１",
    "アンケート２",
    "アンケート３",
    "アンケート４",
    "アンケート５",
    "アンケート６",
    "アンケート７",
    "アンケート８",
    "アンケート９",
    "アンケート１０",
    "アンケート１１",
]

CATEGORICAL_FEATURES = [
    "業界",
    "上場種別",
]

TEXT_FEATURES = [

]
def create_tfidf(train_text, test_text, max_features=100):

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=None
    )

    all_text = pd.concat(
        [
            train_text.fillna(""),
            test_text.fillna("")
        ],
        axis=0
    )

    vectorizer.fit(all_text)

    train_vec = vectorizer.transform(
        train_text.fillna("")
    )

    test_vec = vectorizer.transform(
        test_text.fillna("")
    )

    train_df = pd.DataFrame(
        train_vec.toarray(),
        columns=[
            f"tfidf_{i}"
            for i in range(train_vec.shape[1])
        ],
        index=train_text.index
    )

    test_df = pd.DataFrame(
        test_vec.toarray(),
        columns=train_df.columns,
        index=test_text.index
    )

    return train_df, test_df

def prepare_data(train, test):

    y_train = train[TARGET]

    # 数値＋カテゴリ
    train_x = train[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    test_x = test[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()

    # ==========================
    # 特徴量エンジニアリング
    # ==========================

    for df in [train_x, test_x]:

        # ゼロ除算防止
        sales = df["売上"].replace(0, 1)
        assets = df["総資産"].replace(0, 1)
        equity = df["自己資本"].replace(0, 1)
        debt = df["負債"].replace(0, 1)
        employees = df["従業員数"].replace(0, 1)

        # 利益率
        df["営業利益率"] = df["営業利益"] / sales
        df["経常利益率"] = df["経常利益"] / sales
        df["純利益率"] = df["当期純利益"] / sales

        # 財務比率
        df["自己資本比率"] = df["自己資本"] / assets
        df["負債比率"] = df["負債"] / assets
        df["ROA"] = df["当期純利益"] / assets
        df["ROE"] = df["当期純利益"] / equity

        # 従業員あたり
        df["売上_従業員"] = df["売上"] / employees
        df["利益_従業員"] = df["当期純利益"] / employees

        # アンケート
        survey_cols = [
            "アンケート１","アンケート２","アンケート３",
            "アンケート４","アンケート５","アンケート６",
            "アンケート７","アンケート８","アンケート９",
            "アンケート１０","アンケート１１"
        ]

        df["アンケート平均"] = df[survey_cols].mean(axis=1)
        df["アンケート合計"] = df[survey_cols].sum(axis=1)
        df["アンケート標準偏差"] = df[survey_cols].std(axis=1)

    # ==========================
    # TF-IDF特徴量を追加
    # ==========================
    for col in TEXT_FEATURES:

        train_tfidf, test_tfidf = create_tfidf(
            train[col],
            test[col],
            max_features=30
        )

        # 列名が重複しないように変更
        train_tfidf.columns = [
            f"{col}_{c}" for c in train_tfidf.columns
        ]

        test_tfidf.columns = train_tfidf.columns

        train_x = pd.concat(
            [train_x, train_tfidf],
            axis=1
        )

        test_x = pd.concat(
            [test_x, test_tfidf],
            axis=1
        )

    # trainとtestを結合してOne-Hot Encoding
    all_data = pd.concat([train_x, test_x], axis=0)

    all_data = pd.get_dummies(
        all_data,
        columns=CATEGORICAL_FEATURES,
        drop_first=True
    )

    train_x = all_data.iloc[:len(train)].copy()
    test_x = all_data.iloc[len(train):].copy()

    # 欠損値を中央値で補完
    med = train_x.median(numeric_only=True)

    train_x = train_x.fillna(med)
    test_x = test_x.fillna(med)

    # bool型をint型へ変換
    bool_cols = train_x.select_dtypes(include="bool").columns
    train_x[bool_cols] = train_x[bool_cols].astype(int)
    test_x[bool_cols] = test_x[bool_cols].astype(int)

    # ----------------------------
    # 標準化
    # ----------------------------
    scaler = StandardScaler()

    train_x = pd.DataFrame(
        scaler.fit_transform(train_x),
        columns=train_x.columns,
        index=train_x.index
    )

    test_x = pd.DataFrame(
        scaler.transform(test_x),
        columns=test_x.columns,
        index=test_x.index
    )

    return train_x, y_train, test_x