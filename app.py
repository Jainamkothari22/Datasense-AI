from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
DB_PATH = APP_DIR / "data_lab.db"
DATA_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="DataSense AI", page_icon="◈", layout="wide", initial_sidebar_state="expanded")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        :root { --ink: #17211b; --mint: #d8f3dc; --green: #2d6a4f; --coral: #e76f51; --cream: #fbf8f1; }
        html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
        .stApp { background: #fbf8f1; color: #111111; }
        [data-testid="stSidebar"] { background: #d3d3d3 !important; }
        [data-testid="stSidebar"] * { color: #111111 !important; }
        [data-testid="stFileUploader"],
        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploaderDropzone"],
        [data-testid="stDownloadButton"] { background: transparent !important; border: 0 !important; box-shadow: none !important; }
        [data-testid="stFileUploader"] section:hover,
        [data-testid="stFileUploaderDropzone"]:hover,
        [data-testid="stDownloadButton"]:hover { background: transparent !important; }
        [data-testid="stFileUploaderDropzone"] svg,
        [data-testid="stFileUploaderDropzone"] svg path { color: #ffffff !important; fill: #ffffff !important; stroke: #ffffff !important; }
        [data-testid="stFileUploaderFile"],
        [data-testid="stFileUploaderFile"] > div,
        [data-testid="stFileUploaderFile"] section,
        [data-testid="stSidebar"] [data-testid="stButton"] { background: transparent !important; border: 0 !important; box-shadow: none !important; }
        [data-testid="stFileUploader"] * { color: #111111 !important; }
        [data-testid="stFileUploader"] button,
        [data-testid="stDownloadButton"] button { background: #bdbdbd !important; border: 1px solid #777777 !important; color: #111111 !important; }
        [data-testid="stFileUploader"] button:hover,
        [data-testid="stDownloadButton"] button:hover { background: #aaaaaa !important; color: #111111 !important; }
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] [data-baseweb="select"] > div { background: transparent !important; color: #111111 !important; border-color: #777777 !important; }
        [data-testid="stSidebar"] [data-testid="stButton"] button { background: transparent !important; color: #111111 !important; border: 1px solid #777777 !important; }
        [data-testid="stSidebar"] [data-testid="stButton"] button:hover { background: #bdbdbd !important; color: #111111 !important; }
        h1, h2, h3 { letter-spacing: 0; }
        h1 { font-size: 3rem !important; line-height: 1.02 !important; }
        .eyebrow { color: #e76f51; font-family: 'DM Mono', monospace; font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; }
        .hero { padding: 1.4rem 0 1.2rem; border-bottom: 1px solid #d9d8cc; margin-bottom: 1.2rem; }
        .hero p { color: #68736b; font-size: 1.03rem; max-width: 720px; }
        .metric { background: #e5f4e8; border: 1px solid #bddbc4; padding: 1rem; min-height: 102px; }
        .metric-label { color: #527060; font-family: 'DM Mono', monospace; font-size: .72rem; text-transform: uppercase; }
        .metric-value { color: #17211b; font-size: 1.8rem; font-weight: 700; margin-top: .35rem; }
        .note { border-left: 4px solid #e76f51; background: #fff3ed; padding: .85rem 1rem; margin: .5rem 0 1rem; }
        .stButton > button { border-radius: 2px; border: 1px solid #2d6a4f; font-weight: 600; }
        code, .mono { font-family: 'DM Mono', monospace; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def db() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS projects (id INTEGER PRIMARY KEY, name TEXT, file_name TEXT, rows INTEGER, columns INTEGER, created_at TEXT, path TEXT)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS activity (id INTEGER PRIMARY KEY, project_id INTEGER, action TEXT, details TEXT, created_at TEXT)"
    )
    connection.commit()
    return connection


def record_project(name: str, file_name: str, frame: pd.DataFrame, path: Path) -> int:
    connection = db()
    cursor = connection.execute(
        "INSERT INTO projects(name, file_name, rows, columns, created_at, path) VALUES (?, ?, ?, ?, ?, ?)",
        (name, file_name, len(frame), len(frame.columns), datetime.now().isoformat(timespec="seconds"), str(path)),
    )
    connection.commit()
    project_id = int(cursor.lastrowid)
    connection.close()
    return project_id


def record_activity(project_id: int, action: str, details: dict[str, Any]) -> None:
    connection = db()
    connection.execute(
        "INSERT INTO activity(project_id, action, details, created_at) VALUES (?, ?, ?, ?)",
        (project_id, action, json.dumps(details), datetime.now().isoformat(timespec="seconds")),
    )
    connection.commit()
    connection.close()


def list_data_tables() -> list[str]:
    connection = db()
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT IN ('projects', 'activity') ORDER BY name",
        connection,
    )["name"].tolist()
    connection.close()
    return [str(table) for table in tables]


def save_frame_as_table(frame: pd.DataFrame, table_name: str, replace_existing: bool) -> None:
    normalized_name = table_name.strip().lower().replace(" ", "_")
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,62}", normalized_name):
        raise ValueError("Use a table name starting with a letter, followed by letters, numbers, or underscores.")
    connection = db()
    frame.to_sql(normalized_name, connection, if_exists="replace" if replace_existing else "fail", index=False)
    connection.close()


def read_upload(uploaded_file: Any) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file)
    raise ValueError("Use a CSV or Excel file.")


def combine_frames(left: pd.DataFrame, right: pd.DataFrame, mode: str, left_key: str | None = None, right_key: str | None = None, join_type: str = "inner") -> pd.DataFrame:
    if mode == "append":
        return pd.concat([left, right], ignore_index=True, sort=False)
    if left_key is None or right_key is None:
        raise ValueError("Select a relationship column in both datasets.")
    return left.merge(right, left_on=left_key, right_on=right_key, how=join_type, suffixes=("_left", "_right"))


def metric(label: str, value: str) -> None:
    st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)


def profile_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "column": frame.columns,
        "type": [str(item) for item in frame.dtypes],
        "missing": frame.isna().sum().values,
        "missing_%": (frame.isna().mean().mul(100).round(1)).values,
        "unique": frame.nunique(dropna=True).values,
    })


def clean_data(frame: pd.DataFrame, options: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    result = frame.copy()
    before = len(result)
    normalized_columns: list[str] = []
    used_names: dict[str, int] = {}
    for column in result.columns:
        base_name = str(column).strip().replace(" ", "_").lower() or "column"
        occurrence = used_names.get(base_name, 0)
        used_names[base_name] = occurrence + 1
        normalized_columns.append(base_name if occurrence == 0 else f"{base_name}_{occurrence + 1}")
    result.columns = normalized_columns
    if options["strip_text"]:
        for column in result.select_dtypes(include="object").columns:
            result[column] = result[column].map(lambda value: value.strip() if isinstance(value, str) else value)
    if options["drop_duplicates"]:
        result = result.drop_duplicates()
    if options["missing_strategy"] == "drop":
        result = result.dropna()
    elif options["missing_strategy"] == "fill":
        for column in result.columns:
            if result[column].isna().any():
                if pd.api.types.is_numeric_dtype(result[column]):
                    result[column] = result[column].fillna(result[column].median())
                else:
                    mode = result[column].mode(dropna=True)
                    result[column] = result[column].fillna(mode.iloc[0] if not mode.empty else "Unknown")
    if options["clip_outliers"]:
        for column in result.select_dtypes(include=np.number).columns:
            low, high = result[column].quantile([0.01, 0.99])
            result[column] = result[column].clip(lower=low, upper=high)
    return result, {"rows_before": before, "rows_after": len(result), "duplicates_removed": before - len(result)}


def render_overview(frame: pd.DataFrame) -> None:
    st.subheader("Dataset overview")
    st.dataframe(frame.head(100), use_container_width=True, height=320)
    st.subheader("Column health")
    st.dataframe(profile_frame(frame), use_container_width=True, hide_index=True)


def render_visualize(frame: pd.DataFrame) -> None:
    numeric = list(frame.select_dtypes(include=np.number).columns)
    all_columns = list(frame.columns)
    if not all_columns:
        st.warning("No columns are available for visualization.")
        return
    if frame.empty:
        st.info("There are no rows to visualize. Adjust the cleaning or filter settings first.")
        return
    left, right = st.columns(2)
    with left:
        chart_type = st.selectbox("Chart type", ["Histogram", "Bar chart", "Line chart", "Scatter plot", "Box plot"])
        x_column = st.selectbox("X / category", all_columns)
    with right:
        y_options = numeric if numeric else all_columns
        y_column = st.selectbox("Y / value", y_options, index=0)
        limit = st.slider("Rows to plot", 10, min(1000, max(10, len(frame))), min(250, max(10, len(frame))))
    if x_column == y_column:
        chart_data = frame[[x_column]].dropna().head(limit)
    else:
        chart_data = frame[[x_column, y_column]].dropna().head(limit)
    if chart_type == "Histogram":
        histogram_data = chart_data[x_column].value_counts().sort_index()
        st.bar_chart(histogram_data)
    elif chart_type == "Bar chart":
        if not pd.api.types.is_numeric_dtype(chart_data[y_column]):
            st.info("Bar charts need a numeric Y / value column. Choose a numeric column or use Histogram for text data.")
            return
        grouped = chart_data.groupby(x_column, dropna=False)[y_column].mean().sort_values(ascending=False).head(30)
        st.bar_chart(grouped)
    elif chart_type == "Line chart":
        if not pd.api.types.is_numeric_dtype(chart_data[y_column]):
            st.info("Line charts need a numeric Y / value column.")
            return
        if x_column == y_column:
            st.line_chart(chart_data[y_column].reset_index(drop=True))
        else:
            st.line_chart(chart_data.set_index(x_column)[y_column])
    elif chart_type == "Scatter plot":
        if x_column == y_column:
            st.info("Scatter plots need two different numeric columns.")
        elif not pd.api.types.is_numeric_dtype(chart_data[x_column]) or not pd.api.types.is_numeric_dtype(chart_data[y_column]):
            st.info("Scatter plots need numeric X and Y columns.")
        else:
            st.scatter_chart(chart_data, x=x_column, y=y_column)
    else:
        if not pd.api.types.is_numeric_dtype(chart_data[y_column]):
            st.info("Box plots need a numeric Y / value column.")
            return
        st.dataframe(chart_data.groupby(x_column)[y_column].describe().round(2), use_container_width=True)
    if len(numeric) >= 2:
        st.subheader("Numeric correlation")
        st.dataframe(frame[numeric].corr().round(2), use_container_width=True)


def render_transform(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    st.caption("Transformations apply to the working copy. Download the result from the sidebar when finished.")
    action = st.selectbox("Transformation", ["Filter rows", "Rename column", "Convert type", "Sort rows", "Sample rows"])
    if action == "Filter rows":
        st.caption("Use a pandas expression such as `age > 25` or `segment == 'Enterprise'`.")
        expression = st.text_input("Filter expression")
        if expression:
            try:
                result = result.query(expression)
                st.success(f"Filtered to {len(result):,} rows.")
            except Exception as error:
                st.error(f"Could not apply filter: {error}")
    elif action == "Rename column":
        old_name = st.selectbox("Column to rename", list(result.columns))
        new_name = st.text_input("New name", value=old_name)
        if st.button("Apply rename") and new_name:
            result = result.rename(columns={old_name: new_name})
    elif action == "Convert type":
        column = st.selectbox("Column", list(result.columns))
        target = st.selectbox("Type", ["string", "numeric", "datetime"])
        if st.button("Apply conversion"):
            if target == "numeric":
                result[column] = pd.to_numeric(result[column], errors="coerce")
            elif target == "datetime":
                result[column] = pd.to_datetime(result[column], errors="coerce")
            else:
                result[column] = result[column].astype("string")
    elif action == "Sort rows":
        column = st.selectbox("Sort by", list(result.columns))
        descending = st.checkbox("Descending")
        result = result.sort_values(column, ascending=not descending)
    else:
        amount = st.number_input("Rows", min_value=1, max_value=max(1, len(result)), value=min(100, max(1, len(result))))
        if result.empty:
            st.info("There are no rows to sample. Adjust the filter first.")
        else:
            result = result.sample(min(int(amount), len(result)), random_state=42)
    st.dataframe(result.head(100), use_container_width=True, height=320)
    return result


def render_database(frame: pd.DataFrame, project_id: int) -> None:
    st.subheader("Create a data table")
    st.caption("Save the current working dataset as a reusable SQLite table inside this project.")
    table_name = st.text_input("Table name", value="cleaned_dataset", help="Use letters, numbers, and underscores. The name must start with a letter.")
    replace_existing = st.checkbox("Replace table if it already exists")
    if st.button("Create SQLite table", type="primary"):
        try:
            normalized_name = table_name.strip().lower().replace(" ", "_")
            save_frame_as_table(frame, table_name, replace_existing)
            record_activity(project_id, "create_table", {"table": normalized_name, "rows": len(frame), "columns": len(frame.columns)})
            st.success(f"Table `{normalized_name}` created with {len(frame):,} rows and {len(frame.columns):,} columns.")
        except ValueError as error:
            st.error(str(error))
        except sqlite3.OperationalError:
            st.error("That table already exists. Enable 'Replace table if it already exists' to overwrite it.")
    tables = list_data_tables()
    if tables:
        st.divider()
        st.subheader("Saved tables")
        selected_table = st.selectbox("Preview table", tables)
        connection = db()
        preview = pd.read_sql_query(f'SELECT * FROM "{selected_table}" LIMIT 100', connection)
        connection.close()
        st.dataframe(preview, use_container_width=True, height=280)
    else:
        st.info("No data tables have been created yet.")


def render_combine(sources: dict[str, pd.DataFrame], project_id: int) -> None:
    st.subheader("Combine data sources")
    st.caption("Append similar files or join related datasets through a shared business key.")
    source_names = list(sources)
    if len(source_names) < 2:
        st.info("Upload at least two files to compare or combine them.")
        return
    left_name, right_name = st.columns(2)
    with left_name:
        left_source = st.selectbox("First dataset", source_names, key="combine_left_source")
    with right_name:
        right_options = [name for name in source_names if name != left_source] or source_names
        right_source = st.selectbox("Second dataset", right_options, key="combine_right_source")
    mode = st.radio("Combine method", ["append", "join"], format_func=lambda value: "Append rows (same columns)" if value == "append" else "Join tables (related columns)", horizontal=True)
    left_frame = sources[left_source]
    right_frame = sources[right_source]
    if mode == "append":
        st.caption(f"Append combines {len(left_frame):,} rows from `{left_source}` with {len(right_frame):,} rows from `{right_source}`.")
        if st.button("Append datasets", type="primary"):
            combined = combine_frames(left_frame, right_frame, "append")
            st.session_state.frame = combined
            record_activity(project_id, "combine_append", {"sources": [left_source, right_source], "rows": len(combined)})
            st.success(f"Created a combined dataset with {len(combined):,} rows.")
            st.dataframe(combined.head(100), use_container_width=True)
    else:
        left_columns = list(left_frame.columns)
        right_columns = list(right_frame.columns)
        shared_columns = [column for column in left_columns if column in right_columns]
        default_left = shared_columns[0] if shared_columns else left_columns[0]
        default_right = default_left if default_left in right_columns else right_columns[0]
        key_left, key_right = st.columns(2)
        with key_left:
            left_key = st.selectbox("Relationship column in first dataset", left_columns, index=left_columns.index(default_left))
        with key_right:
            right_key = st.selectbox("Relationship column in second dataset", right_columns, index=right_columns.index(default_right))
        join_type = st.selectbox("Relationship type", ["inner", "left", "right", "outer"], format_func=lambda value: {"inner": "Matching records only", "left": "Keep all first-dataset records", "right": "Keep all second-dataset records", "outer": "Keep all records"}[value])
        if st.button("Join datasets", type="primary"):
            try:
                combined = combine_frames(left_frame, right_frame, "join", left_key, right_key, join_type)
                st.session_state.frame = combined
                record_activity(project_id, "combine_join", {"sources": [left_source, right_source], "keys": [left_key, right_key], "join_type": join_type, "rows": len(combined)})
                st.success(f"Joined datasets into {len(combined):,} rows.")
                st.dataframe(combined.head(100), use_container_width=True)
            except (TypeError, ValueError) as error:
                st.error(f"Could not join these datasets: {error}")


def render_ml(frame: pd.DataFrame) -> None:
    st.subheader("Predictive workbench")
    if len(frame) < 20:
        st.info("Add at least 20 rows to train a meaningful baseline model.")
        return
    target = st.selectbox("Target column", list(frame.columns))
    feature_columns = [column for column in frame.columns if column != target]
    selected = st.multiselect("Features", feature_columns, default=feature_columns[: min(8, len(feature_columns))])
    if not selected:
        st.warning("Select at least one feature.")
        return
    target_series = frame[target].dropna()
    if not pd.api.types.is_numeric_dtype(target_series) or target_series.nunique() <= 10:
        task = "classification"
    else:
        task = "regression"
    st.caption(f"Detected task: **{task}**. This baseline uses an 80/20 split and preprocessing for numeric and categorical features.")
    model_name = st.selectbox("Model", ["Random Forest", "Linear / Logistic baseline"])
    if not st.button("Train baseline model", type="primary"):
        return
    working = frame[selected + [target]].dropna(subset=[target])
    X = working[selected]
    y = working[target]
    categorical = list(X.select_dtypes(exclude=np.number).columns)
    numeric = list(X.select_dtypes(include=np.number).columns)
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    if task == "classification":
        if y.nunique() < 2:
            st.error("The target needs at least two classes.")
            return
        model = RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced") if model_name == "Random Forest" else LogisticRegression(max_iter=500)
        metric_name = "Accuracy"
    else:
        model = RandomForestRegressor(n_estimators=120, random_state=42) if model_name == "Random Forest" else LinearRegression()
        metric_name = "R2 score"
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=stratify)
    try:
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
    except ValueError as error:
        st.error(f"The model could not train on these columns: {error}")
        return
    score = accuracy_score(y_test, predictions) if task == "classification" else r2_score(y_test, predictions)
    error = None if task == "classification" else mean_absolute_error(y_test, predictions)
    left, right = st.columns(2)
    with left:
        metric(metric_name, f"{score:.1%}" if task == "classification" else f"{score:.3f}")
    with right:
        metric("Test rows", f"{len(y_test):,}")
    if error is not None:
        st.caption(f"Mean absolute error: {error:.3f}")
    st.dataframe(pd.DataFrame({"actual": y_test.values, "predicted": predictions}).head(25), use_container_width=True)
    st.success("Baseline trained. For a production model, add cross-validation, feature selection, and model monitoring.")


def main() -> None:
    inject_css()
    if "frame" not in st.session_state:
        st.session_state.frame = None
    if "project_id" not in st.session_state:
        st.session_state.project_id = None
    if "sources" not in st.session_state:
        st.session_state.sources = {}
    st.sidebar.markdown("# ◈ DataSense AI")
    st.sidebar.caption("A practical workspace for cleaning, understanding, and modeling tabular data.")
    uploaded_files = st.sidebar.file_uploader("Upload datasets", type=["csv", "xlsx", "xls"], accept_multiple_files=True)
    if uploaded_files:
        batch_hash = hashlib.md5(b"".join(upload.getvalue() for upload in uploaded_files)).hexdigest()
        if st.session_state.get("file_hash") != batch_hash:
            try:
                sources: dict[str, pd.DataFrame] = {}
                for uploaded_file in uploaded_files:
                    source_name = Path(uploaded_file.name).stem
                    if source_name in sources:
                        source_name = f"{source_name}_{len(sources) + 1}"
                    source_frame = read_upload(uploaded_file)
                    sources[source_name] = source_frame
                    save_path = DATA_DIR / f"{hashlib.md5(uploaded_file.getvalue()).hexdigest()}_{Path(uploaded_file.name).stem}.csv"
                    source_frame.to_csv(save_path, index=False)
                first_name = next(iter(sources))
                st.session_state.sources = sources
                st.session_state.frame = sources[first_name].copy()
                st.session_state.original_frame = sources[first_name].copy()
                st.session_state.file_hash = batch_hash
                st.session_state.project_id = record_project(first_name, uploaded_files[0].name, st.session_state.frame, DATA_DIR)
                record_activity(st.session_state.project_id, "upload", {"file_names": [file.name for file in uploaded_files], "file_count": len(uploaded_files)})
            except Exception as error:
                st.sidebar.error(str(error))
    if st.session_state.frame is None:
        st.markdown('<div class="hero"><div class="eyebrow">Portfolio project / analytics workspace</div><h1>Turn raw files into<br>decisions.</h1><p>Upload a dataset to profile its quality, clean it with a reproducible workflow, build clear charts, and train a baseline model when prediction is useful.</p></div>', unsafe_allow_html=True)
        st.info("Start by uploading a CSV or Excel file in the sidebar.")
        st.markdown("#### What this demonstrates")
        a, b, c = st.columns(3)
        with a: st.markdown("**Data quality**\n\nMissing values, duplicates, types, and outlier handling.")
        with b: st.markdown("**Exploration**\n\nReusable charts, distributions, grouping, and correlations.")
        with c: st.markdown("**Modeling**\n\nPreprocessing pipelines and measurable baseline models.")
        return
    frame = st.session_state.frame
    project_id = st.session_state.project_id
    st.markdown('<div class="hero"><div class="eyebrow">Active project</div><h1>DataSense AI</h1><p>Make one confident decision at a time. Your working data is preserved in this session and the project metadata is stored in SQLite.</p></div>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    with a: metric("Rows", f"{len(frame):,}")
    with b: metric("Columns", f"{len(frame.columns):,}")
    with c: metric("Missing cells", f"{int(frame.isna().sum().sum()):,}")
    with d: metric("Duplicate rows", f"{int(frame.duplicated().sum()):,}")
    tabs = st.tabs(["Overview", "Clean", "Visualize", "Transform", "Combine", "Model", "Database"])
    with tabs[0]:
        render_overview(frame)
    with tabs[1]:
        st.subheader("Cleaning recipe")
        st.caption("These operations are deterministic and logged as a project action.")
        col1, col2 = st.columns(2)
        with col1:
            drop_duplicates = st.checkbox("Remove duplicate rows", value=True)
            strip_text = st.checkbox("Trim text and normalize column names", value=True)
            clip_outliers = st.checkbox("Clip numeric outliers at 1st / 99th percentile")
        with col2:
            missing_strategy = st.radio("Missing values", ["keep", "fill", "drop"], format_func=lambda value: {"keep": "Leave unchanged", "fill": "Fill numeric with median / text with mode", "drop": "Drop incomplete rows"}[value])
        if st.button("Apply cleaning recipe", type="primary"):
            cleaned, stats = clean_data(frame, {"drop_duplicates": drop_duplicates, "strip_text": strip_text, "clip_outliers": clip_outliers, "missing_strategy": missing_strategy})
            st.session_state.frame = cleaned
            record_activity(project_id, "clean", stats)
            st.success(f"Cleaning applied: {stats['rows_before'] - stats['rows_after']:,} rows removed.")
            st.rerun()
    with tabs[2]:
        render_visualize(frame)
    with tabs[3]:
        transformed = render_transform(frame)
        if st.button("Use transformed data"):
            st.session_state.frame = transformed
            record_activity(project_id, "transform", {"rows": len(transformed), "columns": len(transformed.columns)})
            st.rerun()
    with tabs[4]:
        render_combine(st.session_state.sources, project_id)
    with tabs[5]:
        render_ml(frame)
    with tabs[6]:
        render_database(frame, project_id)
    st.sidebar.divider()
    st.sidebar.download_button("Download working CSV", frame.to_csv(index=False), file_name="data_lab_cleaned.csv", mime="text/csv", use_container_width=True)
    if st.sidebar.button("Reset to uploaded data", use_container_width=True):
        st.session_state.frame = st.session_state.original_frame.copy()
        st.rerun()
    st.sidebar.caption(f"Project #{project_id} · SQLite history enabled")


if __name__ == "__main__":
    main()
