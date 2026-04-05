import os
import re
import uuid
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI


# =========================
# Azure OpenAI 設定
# =========================
load_dotenv()
client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")


# =========================
# 画面設定
# =========================
st.set_page_config(
    page_title="KPI分析レポートジェネレーター（CarSensor）",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# KPI構造
# =========================
CONSISTENCY_THRESHOLD = 3.0

KPI_CONSISTENCY_RULES = [
    {
        "rule_id": "rule_01",
        "upstream": "純粋想起_Q1（MA）",
        "downstream": "助成想起_Q6（MA）",
        "label": "純粋想起→助成想起"
    },
    {
        "rule_id": "rule_02",
        "upstream": "助成想起_Q6（MA）",
        "downstream": "好意TOP2_Q11",
        "label": "助成想起→好意"
    },
    {
        "rule_id": "rule_03",
        "upstream": "好意TOP2_Q11",
        "downstream": "利用意向_Q10_1（MA）",
        "label": "好意→利用意向"
    },
    {
        "rule_id": "rule_04",
        "upstream": "利用意向_Q10_1（MA）",
        "downstream": "第一利用意向_Q10_2（MA）",
        "label": "利用意向→第一利用意向"
    },
    {
        "rule_id": "rule_05",
        "upstream": "第一利用意向_Q10_2（MA）",
        "downstream": "1年以内利用_Q9_2（MA）",
        "label": "第一利用意向→1年以内利用"
    },
    {
        "rule_id": "rule_06",
        "upstream": "1年以内利用_Q9_2（MA）",
        "downstream": "満足TOP2_Q12",
        "label": "1年以内利用→満足"
    },
    {
        "rule_id": "rule_07",
        "upstream": "利用経験_Q9_1（MA）",
        "downstream": "好意TOP2_Q11",
        "label": "利用経験→好意"
    },
]


# =========================
# 初期設定
# =========================
def init_session_state() -> None:
    defaults = {
        "mode": "総括",
        "raw_df": None,
        "uploaded_file_name": None,
        "kpi_change_table": None,
        "kpi_detail_clicked": None,
        "kpi_detail_table": None,
        "ad_effect_clicked": None,
        "ad_effect_table": None,
        "current_llm_comment": None,
        "pinned_comments": [],
        "final_summary": None,
        "consistency_check_results": None,
        "consistency_llm_comment": None,
        "overview_navigation_comment": None,
        "block2_navigation_comment": None,
        "block3_navigation_comment": None,
        "block4_navigation_comment": None,
        "block2_summary_comment": None,
        "block3_summary_comment": None,
        "block4_summary_comment": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "pinned_comments" not in st.session_state:
        st.session_state.pinned_comments = []



def reset_analysis_states() -> None:
    st.session_state.raw_df = None
    st.session_state.kpi_change_table = None
    st.session_state.kpi_detail_clicked = None
    st.session_state.kpi_detail_table = None
    st.session_state.ad_effect_clicked = None
    st.session_state.ad_effect_table = None
    st.session_state.current_llm_comment = None
    st.session_state.pinned_comments = []
    st.session_state.final_summary = None
    st.session_state.consistency_check_results = None
    st.session_state.consistency_llm_comment = None


init_session_state()


# =========================
# 共通関数
# =========================
def set_mode(mode_name: str) -> None:
    st.session_state.mode = mode_name


def load_data(uploaded_file) -> pd.DataFrame:
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("対応していないファイル形式です。")


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def format_dataframe_for_prompt(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "データなし"
    return df.head(max_rows).to_csv(index=False)



# =========================
# 総括モード用関数
# =========================
def normalize_str_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def get_sorted_month_options(df: pd.DataFrame) -> list:
    if "調査月" not in df.columns:
        return []

    month_values = (
        df["調査月"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    def month_sort_key(x: str):
        m = re.search(r"(\d{4})年\s*(\d{1,2})月", str(x))
        if m:
            return (int(m.group(1)), int(m.group(2)))
        return (-1, -1)

    month_values = sorted(month_values, key=month_sort_key, reverse=True)
    return month_values


def get_previous_month(selected_month: str, month_options_desc: list) -> str | None:
    """
    month_options_desc は新しい順
    """
    if selected_month not in month_options_desc:
        return None

    idx = month_options_desc.index(selected_month)
    if idx + 1 >= len(month_options_desc):
        return None
    return month_options_desc[idx + 1]


def filter_actual_value_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    実測値用。
    変数名列があり、'実測値' が存在する場合はそれを優先。
    """
    work_df = df.copy()
    if "変数名" in work_df.columns:
        vals = set(work_df["変数名"].dropna().astype(str))
        if "実測値" in vals:
            work_df = work_df[work_df["変数名"].astype(str) == "実測値"].copy()
    return work_df


def filter_lift_value_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    施策影響示唆用。
    変数名列があり、'認知・非認知ギャップ' が存在する場合はそれを優先。
    """
    work_df = df.copy()
    if "変数名" in work_df.columns:
        vals = set(work_df["変数名"].dropna().astype(str))
        if "認知・非認知ギャップ" in vals:
            work_df = work_df[work_df["変数名"].astype(str) == "認知・非認知ギャップ"].copy()
    return work_df


def get_total_mask(df: pd.DataFrame) -> pd.Series:
    if "セグメント名" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)

    seg_name = normalize_str_series(df["セグメント名"]).str.upper()
    return seg_name.eq("TOTAL")


def build_overview_metric_diff_table(
    df: pd.DataFrame,
    selected_month: str,
    previous_month: str
) -> pd.DataFrame:
    required_cols = ["調査月", "接触広告", "指標名", "指標番号", "セグメント名", "値"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"必要な列が不足しています: {', '.join(missing_cols)}")

    work_df = filter_actual_value_df(df).copy()
    work_df["調査月"] = work_df["調査月"].astype(str)
    work_df["接触広告"] = normalize_str_series(work_df["接触広告"])
    work_df["指標名"] = work_df["指標名"].astype(str)
    work_df["指標番号"] = pd.to_numeric(work_df["指標番号"], errors="coerce")
    work_df["値"] = pd.to_numeric(work_df["値"], errors="coerce")

    work_df = work_df[
        (work_df["調査月"].isin([selected_month, previous_month])) &
        (work_df["接触広告"] == "合計値") &
        (get_total_mask(work_df))
    ].copy()

    if work_df.empty:
        return pd.DataFrame()

    monthly_df = pd.pivot_table(
        work_df,
        index=["指標番号", "指標名"],
        columns="調査月",
        values="値",
        aggfunc="first",
    ).reset_index()

    if selected_month not in monthly_df.columns or previous_month not in monthly_df.columns:
        return pd.DataFrame()

    diff_col = f"{selected_month} ー {previous_month}"
    monthly_df[diff_col] = (monthly_df[selected_month] - monthly_df[previous_month]).round(1)

    result_df = monthly_df[["指標番号", "指標名", selected_month, previous_month, diff_col]].copy()
    result_df = result_df.sort_values(["指標番号", "指標名"]).reset_index(drop=True)
    return result_df


def build_segment_diff_table(
    df: pd.DataFrame,
    selected_month: str,
    previous_month: str
) -> pd.DataFrame:
    required_cols = ["調査月", "接触広告", "指標名", "指標番号", "セグメント名", "セグメント番号", "値"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"必要な列が不足しています: {', '.join(missing_cols)}")

    work_df = filter_actual_value_df(df).copy()
    work_df["調査月"] = work_df["調査月"].astype(str)
    work_df["接触広告"] = normalize_str_series(work_df["接触広告"])
    work_df["指標名"] = work_df["指標名"].astype(str)
    work_df["指標番号"] = pd.to_numeric(work_df["指標番号"], errors="coerce")
    work_df["セグメント番号"] = pd.to_numeric(work_df["セグメント番号"], errors="coerce")
    work_df["値"] = pd.to_numeric(work_df["値"], errors="coerce")

    work_df = work_df[
        (work_df["調査月"].isin([selected_month, previous_month])) &
        (work_df["接触広告"] == "合計値") &
        (~get_total_mask(work_df))
    ].copy()

    if work_df.empty:
        return pd.DataFrame()

    monthly_df = pd.pivot_table(
        work_df,
        index=["指標番号", "指標名", "セグメント番号", "セグメント名"],
        columns="調査月",
        values="値",
        aggfunc="first",
    ).reset_index()

    if selected_month not in monthly_df.columns or previous_month not in monthly_df.columns:
        return pd.DataFrame()

    diff_col = f"{selected_month} ー {previous_month}"
    monthly_df[diff_col] = (monthly_df[selected_month] - monthly_df[previous_month]).round(1)

    result_df = monthly_df[
        ["指標番号", "指標名", "セグメント番号", "セグメント名", selected_month, previous_month, diff_col]
    ].copy()

    result_df = result_df.sort_values(
        ["指標番号", "指標名", "セグメント番号", "セグメント名"]
    ).reset_index(drop=True)

    return result_df


def build_segment_top3_summary_tables(
    segment_diff_df: pd.DataFrame,
    selected_month: str,
    previous_month: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if segment_diff_df is None or segment_diff_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    diff_col = f"{selected_month} ー {previous_month}"
    work_df = segment_diff_df.copy()

    positive_rows = []
    negative_rows = []

    grouped = work_df.groupby(["指標番号", "指標名"], dropna=False)

    for (metric_no, metric_name), group in grouped:
        pos_group = group[group[diff_col] > 0].sort_values(diff_col, ascending=False).head(3)
        neg_group = group[group[diff_col] < 0].sort_values(diff_col, ascending=True).head(3)

        pos_record = {
            "指標番号": metric_no,
            "指標名": metric_name,
        }
        neg_record = {
            "指標番号": metric_no,
            "指標名": metric_name,
        }

        for idx, (_, row) in enumerate(pos_group.iterrows(), start=1):
            pos_record[f"Top{idx}"] = f"{row['セグメント名']} ({row[diff_col]:.1f})"

        for idx, (_, row) in enumerate(neg_group.iterrows(), start=1):
            neg_record[f"Top{idx}"] = f"{row['セグメント名']} ({row[diff_col]:.1f})"

        positive_rows.append(pos_record)
        negative_rows.append(neg_record)

    pos_df = pd.DataFrame(positive_rows).fillna("-")
    neg_df = pd.DataFrame(negative_rows).fillna("-")

    if not pos_df.empty:
        pos_df = pos_df.sort_values(["指標番号", "指標名"]).reset_index(drop=True)
    if not neg_df.empty:
        neg_df = neg_df.sort_values(["指標番号", "指標名"]).reset_index(drop=True)

    return pos_df, neg_df


def build_ad_impact_table_for_overview(
    df: pd.DataFrame,
    selected_month: str
) -> pd.DataFrame:
    required_cols = ["調査月", "接触広告", "指標名", "指標番号", "セグメント名", "値"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"必要な列が不足しています: {', '.join(missing_cols)}")

    work_df = filter_lift_value_df(df).copy()
    work_df["調査月"] = work_df["調査月"].astype(str)
    work_df["接触広告"] = normalize_str_series(work_df["接触広告"])
    work_df["指標名"] = work_df["指標名"].astype(str)
    work_df["指標番号"] = pd.to_numeric(work_df["指標番号"], errors="coerce")
    work_df["値"] = pd.to_numeric(work_df["値"], errors="coerce")

    work_df = work_df[
        (work_df["調査月"] == selected_month) &
        (work_df["接触広告"] != "合計値") &
        (get_total_mask(work_df))
    ].copy()

    if work_df.empty:
        return pd.DataFrame()

    result_df = work_df[["指標番号", "指標名", "接触広告", "値"]].copy()
    result_df = result_df.sort_values(["値", "指標番号", "指標名"], ascending=[False, True, True]).reset_index(drop=True)
    return result_df


def create_overview_summary_text(
    top_positive_df: pd.DataFrame,
    top_negative_df: pd.DataFrame,
    ad_impact_df: pd.DataFrame,
    selected_month: str,
    previous_month: str
) -> str:
    messages = []

    if top_positive_df is not None and not top_positive_df.empty:
        pos_name = top_positive_df.iloc[0]["指標名"]
        pos_diff = top_positive_df.iloc[0][f"{selected_month} ー {previous_month}"]
        messages.append(f"改善面では「{pos_name}」が {pos_diff:.1f} と大きく上昇しています。")

    if top_negative_df is not None and not top_negative_df.empty:
        neg_name = top_negative_df.iloc[0]["指標名"]
        neg_diff = top_negative_df.iloc[0][f"{selected_month} ー {previous_month}"]
        messages.append(f"悪化面では「{neg_name}」が {neg_diff:.1f} と大きく低下しています。")

    if ad_impact_df is not None and not ad_impact_df.empty:
        ad_metric = ad_impact_df.iloc[0]["指標名"]
        ad_name = ad_impact_df.iloc[0]["接触広告"]
        ad_value = ad_impact_df.iloc[0]["値"]
        messages.append(
            f"施策面では「{ad_metric}」で広告「{ad_name}」の値が {ad_value:.1f} と大きく、深掘り候補です。"
        )

    if not messages:
        return "選択月と前月の比較に必要なデータが不足しているため、総括を作成できません。"

    messages.append("まずは改善・悪化の大きい指標と、施策影響が強そうな指標から確認するのがおすすめです。")
    return " ".join(messages)



# =========================
# KPI変化用関数
# =========================
def normalize_str_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def get_sorted_month_options(df: pd.DataFrame) -> list:
    if "調査月" not in df.columns:
        return []

    month_values = (
        df["調査月"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    def month_sort_key(x: str):
        import re
        m = re.search(r"(\d{4})年\s*(\d{1,2})月", str(x))
        if m:
            return (int(m.group(1)), int(m.group(2)))
        return (-1, -1)

    month_values = sorted(month_values, key=month_sort_key, reverse=True)
    return month_values


def get_previous_month(selected_month: str, month_options_desc: list) -> str | None:
    """
    month_options_desc は新しい順
    """
    if selected_month not in month_options_desc:
        return None

    idx = month_options_desc.index(selected_month)
    if idx + 1 >= len(month_options_desc):
        return None
    return month_options_desc[idx + 1]

def build_kpi_change_table(
    df: pd.DataFrame,
    selected_month: str,
    previous_month: str
) -> pd.DataFrame:
    required_cols = ["調査月", "接触広告", "指標名", "指標番号", "値"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"必要な列が不足しています: {', '.join(missing_cols)}")

    work_df = df.copy()
    work_df["調査月"] = work_df["調査月"].astype(str)
    work_df["接触広告"] = work_df["接触広告"].astype(str)
    work_df["指標番号"] = pd.to_numeric(work_df["指標番号"], errors="coerce")
    work_df["値"] = pd.to_numeric(work_df["値"], errors="coerce")

    work_df = work_df[
        (work_df["接触広告"] == "合計値") &
        (work_df["指標番号"].between(2, 8)) &
        (work_df["調査月"].isin([selected_month, previous_month]))
    ].copy()

    if work_df.empty:
        return pd.DataFrame()

    monthly_df = pd.pivot_table(
        work_df,
        index=["指標番号", "指標名"],
        columns="調査月",
        values="値",
        aggfunc="first",
    ).reset_index()

    diff_col = f"{selected_month} ー{previous_month}（2期の差分）"

    if selected_month not in monthly_df.columns:
        raise ValueError(f"列 '{selected_month}' が見つかりません。")
    if previous_month not in monthly_df.columns:
        raise ValueError(f"列 '{previous_month}' が見つかりません。")

    monthly_df[diff_col] = (
        monthly_df[selected_month] - monthly_df[previous_month]
    ).round(1)

    result_df = monthly_df[
        ["指標番号", "指標名", diff_col]
    ].copy()

    result_df = result_df.sort_values(["指標番号", "指標名"]).reset_index(drop=True)
    return result_df


def build_kpi_detail_table(
    df: pd.DataFrame,
    selected_metric_no: int,
    selected_month: str,
    previous_month: str
) -> pd.DataFrame:
    required_cols = ["調査月", "接触広告", "指標名", "指標番号", "セグメント名", "セグメント番号", "値"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"必要な列が不足しています: {', '.join(missing_cols)}")

    work_df = df.copy()
    work_df["調査月"] = work_df["調査月"].astype(str)
    work_df["接触広告"] = work_df["接触広告"].astype(str)
    work_df["指標番号"] = pd.to_numeric(work_df["指標番号"], errors="coerce")
    work_df["セグメント番号"] = pd.to_numeric(work_df["セグメント番号"], errors="coerce")
    work_df["値"] = pd.to_numeric(work_df["値"], errors="coerce")

    work_df = work_df[
        (work_df["接触広告"] == "合計値") &
        (work_df["指標番号"] == selected_metric_no) &
        (work_df["セグメント番号"].between(1, 14)) &
        (work_df["調査月"].isin([selected_month, previous_month]))
    ].copy()

    if work_df.empty:
        return pd.DataFrame()

    monthly_df = pd.pivot_table(
        work_df,
        index=["セグメント番号", "セグメント名"],
        columns="調査月",
        values="値",
        aggfunc="first",
    ).reset_index()

    diff_col = f"{selected_month} ー{previous_month}（2期の差分）"

    if selected_month not in monthly_df.columns:
        raise ValueError(f"列 '{selected_month}' が見つかりません。")
    if previous_month not in monthly_df.columns:
        raise ValueError(f"列 '{previous_month}' が見つかりません。")

    monthly_df[diff_col] = (monthly_df[selected_month] - monthly_df[previous_month]).round(1)

    result_df = monthly_df[["セグメント番号", "セグメント名", diff_col]].copy()
    result_df = result_df.sort_values(["セグメント番号", "セグメント名"]).reset_index(drop=True)

    return result_df


def build_ad_effect_table(
    df: pd.DataFrame,
    selected_metric_no: int,
    selected_segment_no: int,
    selected_month: str
) -> pd.DataFrame:
    required_cols = [
        "調査月",
        "接触広告",
        "指標名",
        "指標番号",
        "セグメント名",
        "セグメント番号",
        "値",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"必要な列が不足しています: {', '.join(missing_cols)}")

    work_df = df.copy()
    work_df["調査月"] = work_df["調査月"].astype(str)
    work_df["接触広告"] = work_df["接触広告"].astype(str)
    work_df["指標番号"] = pd.to_numeric(work_df["指標番号"], errors="coerce")
    work_df["セグメント番号"] = pd.to_numeric(work_df["セグメント番号"], errors="coerce")
    work_df["値"] = pd.to_numeric(work_df["値"], errors="coerce")

    work_df = work_df[
        (work_df["調査月"] == selected_month) &
        (work_df["指標番号"] == selected_metric_no) &
        (work_df["セグメント番号"] == selected_segment_no)
    ].copy()

    if work_df.empty:
        return pd.DataFrame()

    ad_master = (
        work_df[["接触広告"]]
        .drop_duplicates()
        .assign(
            sort_order=lambda x: x["接触広告"].astype(str).apply(
                lambda v: 0 if v == "合計値" else 1
            )
        )
        .sort_values(["sort_order", "接触広告"])
    )

    ad_order = ad_master["接触広告"].astype(str).tolist()

    result_df = (
        work_df[["接触広告", "値"]]
        .drop_duplicates(subset=["接触広告"])
        .copy()
    )

    result_df["接触広告"] = pd.Categorical(
        result_df["接触広告"],
        categories=ad_order,
        ordered=True
    )
    result_df = result_df.sort_values("接触広告").reset_index(drop=True)
    result_df["値"] = pd.to_numeric(result_df["値"], errors="coerce").round(1)

    return result_df


# =========================
# 整合性チェックモード用関数
# =========================
def filter_actual_value_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    変数名列がある場合は '実測値' のみを対象にする。
    """
    work_df = df.copy()
    if "変数名" in work_df.columns:
        vals = set(work_df["変数名"].dropna().astype(str))
        if "実測値" in vals:
            work_df = work_df[work_df["変数名"].astype(str) == "実測値"].copy()
    return work_df


def get_total_mask(df: pd.DataFrame) -> pd.Series:
    if "セグメント名" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    return normalize_str_series(df["セグメント名"]).str.upper().eq("TOTAL")


def build_total_kpi_diff_table(
    df: pd.DataFrame,
    selected_month: str,
    previous_month: str
) -> pd.DataFrame:
    required_cols = ["調査月", "接触広告", "指標名", "値"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"必要な列が不足しています: {', '.join(missing_cols)}")

    work_df = filter_actual_value_df(df).copy()
    work_df["調査月"] = work_df["調査月"].astype(str)
    work_df["接触広告"] = normalize_str_series(work_df["接触広告"])
    work_df["指標名"] = work_df["指標名"].astype(str)
    work_df["値"] = pd.to_numeric(work_df["値"], errors="coerce")

    work_df = work_df[
        (work_df["調査月"].isin([selected_month, previous_month])) &
        (work_df["接触広告"] == "合計値") &
        (get_total_mask(work_df))
    ].copy()

    if work_df.empty:
        return pd.DataFrame()

    monthly_df = pd.pivot_table(
        work_df,
        index=["指標名"],
        columns="調査月",
        values="値",
        aggfunc="first",
    ).reset_index()

    if selected_month not in monthly_df.columns or previous_month not in monthly_df.columns:
        return pd.DataFrame()

    diff_col = f"{selected_month} ー {previous_month}"
    monthly_df[diff_col] = (monthly_df[selected_month] - monthly_df[previous_month]).round(1)

    return monthly_df[["指標名", selected_month, previous_month, diff_col]].copy()


def evaluate_kpi_consistency_rules(
    diff_df: pd.DataFrame,
    rules: list,
    selected_month: str,
    previous_month: str,
    threshold: float = 3.0
) -> pd.DataFrame:
    diff_col = f"{selected_month} ー {previous_month}"

    if diff_df is None or diff_df.empty:
        return pd.DataFrame(columns=[
            "rule_id", "label", "upstream", "downstream",
            "upstream_diff", "downstream_diff", "status", "reason"
        ])

    diff_map = {
        row["指標名"]: row[diff_col]
        for _, row in diff_df.iterrows()
    }

    results = []

    for rule in rules:
        upstream = rule["upstream"]
        downstream = rule["downstream"]
        label = rule["label"]
        rule_id = rule["rule_id"]

        upstream_diff = diff_map.get(upstream)
        downstream_diff = diff_map.get(downstream)

        if pd.isna(upstream_diff) or pd.isna(downstream_diff):
            status = "判定対象外"
            reason = "上流または下流の差分データが不足"
        else:
            reverse_negative_to_positive = (
                upstream_diff <= -threshold and downstream_diff >= threshold
            )
            reverse_positive_to_negative = (
                upstream_diff >= threshold and downstream_diff <= -threshold
            )

            if reverse_negative_to_positive:
                status = "整合性注意"
                reason = f"上流が -{threshold:.1f}pt 以下、下流が +{threshold:.1f}pt 以上で逆方向に変化"
            elif reverse_positive_to_negative:
                status = "整合性注意"
                reason = f"上流が +{threshold:.1f}pt 以上、下流が -{threshold:.1f}pt 以下で逆方向に変化"
            else:
                status = "問題なし"
                reason = "逆方向の大幅変化には該当せず"

        results.append({
            "rule_id": rule_id,
            "label": label,
            "upstream": upstream,
            "downstream": downstream,
            "upstream_diff": upstream_diff,
            "downstream_diff": downstream_diff,
            "status": status,
            "reason": reason,
        })

    result_df = pd.DataFrame(results)
    return result_df


def summarize_consistency_counts(result_df: pd.DataFrame) -> dict:
    if result_df is None or result_df.empty:
        return {
            "整合性注意": 0,
            "問題なし": 0,
            "判定対象外": 0,
        }

    return {
        "整合性注意": int((result_df["status"] == "整合性注意").sum()),
        "問題なし": int((result_df["status"] == "問題なし").sum()),
        "判定対象外": int((result_df["status"] == "判定対象外").sum()),
    }


# =========================
# データプレビュー用関数
# =========================
def build_cross_table(df: pd.DataFrame, selected_month: str, selected_ad: str) -> pd.DataFrame:
    required_cols = [
        "調査月",
        "接触広告",
        "指標名",
        "指標番号",
        "セグメント名",
        "セグメント番号",
        "値",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"必要な列が不足しています: {', '.join(missing_cols)}")

    work_df = df.copy()
    work_df["調査月"] = work_df["調査月"].astype(str)
    work_df["接触広告"] = work_df["接触広告"].astype(str)

    work_df = work_df[
        (work_df["調査月"] == selected_month)
        & (work_df["接触広告"] == selected_ad)
    ].copy()

    if work_df.empty:
        return pd.DataFrame()

    work_df["指標番号"] = pd.to_numeric(work_df["指標番号"], errors="coerce")
    work_df["セグメント番号"] = pd.to_numeric(work_df["セグメント番号"], errors="coerce")

    segment_master = (
        work_df[["セグメント番号", "セグメント名"]]
        .drop_duplicates()
        .sort_values(["セグメント番号", "セグメント名"])
    )
    segment_order = segment_master["セグメント名"].astype(str).tolist()

    pivot_df = pd.pivot_table(
        work_df,
        index=["指標番号", "指標名"],
        columns="セグメント名",
        values="値",
        aggfunc="first",
    ).reset_index()

    existing_columns = [col for col in segment_order if col in pivot_df.columns]
    pivot_df = pivot_df[["指標番号", "指標名"] + existing_columns]
    pivot_df = pivot_df.sort_values(["指標番号", "指標名"]).reset_index(drop=True)

    return pivot_df


# =========================
# LLM関連
# =========================
def generate_llm_comment_for_ad(
    metric_info: dict,
    segment_info: dict,
    selected_ad_name: str,
    selected_ad_value,
    total_value,
    ad_df: pd.DataFrame
) -> str:
    compare_text = format_dataframe_for_prompt(ad_df)

    system_prompt = """
あなたは市場調査分析のシニアアナリストです。
与えられたKPI変化、セグメント差分、特定広告の値、および同一条件内の広告比較情報をもとに、
選択された広告についてのみビジネス上意味のある示唆を日本語で簡潔にまとめてください。

要件:
- 3〜5文程度
- 選択された広告にフォーカスする
- 「この広告が相対的に高いか低いか」「合計値と比べてどうか」「このセグメントにおける示唆」を含める
- 他広告全体を網羅的に説明しすぎない
- 不明なことは断定しない
- レポートに転記しやすい自然な日本語にする
"""

    user_prompt = f"""
【対象指標情報】
指標番号: {metric_info.get('指標番号')}
指標名: {metric_info.get('指標名')}
全体差分: {metric_info.get('差分値')}

【対象セグメント情報】
セグメント番号: {segment_info.get('セグメント番号')}
セグメント名: {segment_info.get('セグメント名')}
セグメント差分値: {segment_info.get('セグメント差分値')}

【今回選択された広告】
広告名: {selected_ad_name}
値: {selected_ad_value}

【合計値】
合計値: {total_value}

【同条件での広告比較一覧】
{compare_text}

【依頼】
上記を踏まえて、「今回選択された広告」についてのみコメントしてください。
"""

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.3,
        max_tokens=500,
    )

    return response.choices[0].message.content.strip()


def generate_final_summary(pinned_comments: list) -> str:
    if not pinned_comments:
        return "ピン留めされたコメントがありません。"

    comment_blocks = []

    for i, item in enumerate(pinned_comments):
        if item.get("type") == "overview_summary":
            block = (
                f"【{i + 1}】\n"
                f"種類: 総括コメント\n"
                f"対象月: {item.get('selected_month', '-')}\n"
                f"前月比較: {item.get('previous_month', '-')}\n"
                f"コメント: {item.get('comment', '')}"
            )

        elif item.get("type") == "block2_summary":
            block = (
                f"【{i + 1}】\n"
                f"種類: どの指標が動いたかの要約\n"
                f"対象月: {item.get('selected_month', '-')}\n"
                f"前月比較: {item.get('previous_month', '-')}\n"
                f"コメント: {item.get('comment', '')}"
            )

        elif item.get("type") == "block3_summary":
            block = (
                f"【{i + 1}】\n"
                f"種類: 誰が動いたかの要約\n"
                f"対象月: {item.get('selected_month', '-')}\n"
                f"前月比較: {item.get('previous_month', '-')}\n"
                f"コメント: {item.get('comment', '')}"
            )

        elif item.get("type") == "block4_summary":
            block = (
                f"【{i + 1}】\n"
                f"種類: なぜ動いたかの要約\n"
                f"対象月: {item.get('selected_month', '-')}\n"
                f"前月比較: {item.get('previous_month', '-')}\n"
                f"コメント: {item.get('comment', '')}"
            )

        else:
            block = (
                f"【{i + 1}】\n"
                f"指標: {item.get('指標番号', '-')} - {item.get('指標名', '-')}\n"
                f"セグメント: {item.get('セグメント番号', '-')} - {item.get('セグメント名', '-')}\n"
                f"広告: {item.get('広告名', '-')}\n"
                f"コメント: {item.get('comment', '')}"
            )

        comment_blocks.append(block)

    comment_text = "\n\n".join(comment_blocks)

    system_prompt = """
あなたは市場調査分析のシニアアナリストです。
複数の分析コメントを統合し、最終サマリーを日本語で作成してください。

要件:
- 5〜8文程度
- 重複表現を整理する
- 重要な示唆を優先順位高くまとめる
- 「全体として何が起きているか」「どのセグメントや広告接触に注目すべきか」を含める
- レポート本文にそのまま使える自然な文章にする
- コメントの種類が異なる場合（総括コメント、指標要約、セグメント要約、施策要約、広告別コメントなど）は矛盾しないよう統合する
"""

    user_prompt = f"""
以下は分析担当者がピン留めした重要コメントです。
これらのみを材料に、全体サマリーを作成してください。

{comment_text}
"""

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.3,
        max_tokens=900,
    )

    return response.choices[0].message.content.strip()


def generate_overview_navigation_comment(
    summary_text: str,
    top_positive_df: pd.DataFrame,
    top_negative_df: pd.DataFrame,
    seg_pos_df: pd.DataFrame,
    seg_neg_df: pd.DataFrame,
    ad_impact_df: pd.DataFrame,
    selected_month: str,
    previous_month: str
) -> str:
    def df_to_text(df: pd.DataFrame, max_rows: int = 3) -> str:
        if df is None or df.empty:
            return "該当なし"
        return df.head(max_rows).to_string(index=False)

    system_prompt = """
あなたは市場調査分析のシニアアナリストです。
ブランドマネージャーが次にどこを見ればよいかを、簡潔に案内してください。

要件:
- 日本語
- 2〜3文程度
- 今月の総括コメントを踏まえて、次に確認すべきポイントを優先順で示す
- できるだけ「どのブロックを見るべきか」がわかるようにする
- 冗長にしない
- レポート調ではなく、画面上のナビゲーションとして自然な文にする
"""

    user_prompt = f"""
対象月: {selected_month}
前月: {previous_month}

【総括コメント】
{summary_text}

【改善Top3】
{df_to_text(top_positive_df)}

【悪化Top3】
{df_to_text(top_negative_df)}

【セグメント差プラス】
{df_to_text(seg_pos_df)}

【セグメント差マイナス】
{df_to_text(seg_neg_df)}

【施策影響】
{df_to_text(ad_impact_df)}

この情報を踏まえて、
ブランドマネージャーが次に確認すべきポイントを簡潔に案内してください。
"""

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.2,
        max_tokens=250,
    )

    return response.choices[0].message.content.strip()

def generate_block2_navigation_comment(
    block2_summary_comment: str,
    top_positive_df: pd.DataFrame,
    top_negative_df: pd.DataFrame,
    seg_pos_df: pd.DataFrame,
    seg_neg_df: pd.DataFrame,
    ad_impact_df: pd.DataFrame,
    selected_month: str,
    previous_month: str
) -> str:
    def df_to_text(df: pd.DataFrame, max_rows: int = 3) -> str:
        if df is None or df.empty:
            return "該当なし"
        return df.head(max_rows).to_string(index=False)

    system_prompt = """
あなたは市場調査分析のシニアアナリストです。
「どの指標が動いたか？」を確認したブランドマネージャーに対し、
次にどこを見るべきかを簡潔に案内してください。

要件:
- 日本語
- 2〜3文程度
- 次に確認すべきポイントを優先順で示す
- ブロック3（誰が動いたか）またはブロック4（なぜ動いたか）を見るべき理由がわかるようにする
- 冗長にしない
"""

    user_prompt = f"""
対象月: {selected_month}
前月: {previous_month}

【ブロック2サマリー】
{block2_summary_comment}

【改善Top3】
{df_to_text(top_positive_df)}

【悪化Top3】
{df_to_text(top_negative_df)}

【セグメント差プラス】
{df_to_text(seg_pos_df)}

【セグメント差マイナス】
{df_to_text(seg_neg_df)}

【施策影響】
{df_to_text(ad_impact_df)}

この内容を踏まえて、次にどこを見るべきかを簡潔に案内してください。
"""

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.2,
        max_tokens=250,
    )

    return response.choices[0].message.content.strip()


def generate_block3_navigation_comment(
    block3_summary_comment: str,
    seg_pos_df: pd.DataFrame,
    seg_neg_df: pd.DataFrame,
    ad_impact_df: pd.DataFrame,
    selected_month: str,
    previous_month: str
) -> str:
    def df_to_text(df: pd.DataFrame, max_rows: int = 5) -> str:
        if df is None or df.empty:
            return "該当なし"
        return df.head(max_rows).to_string(index=False)

    system_prompt = """
あなたは市場調査分析のシニアアナリストです。
「誰が動いたか？」を確認したブランドマネージャーに対し、
次にどこを見るべきかを簡潔に案内してください。

要件:
- 日本語
- 2〜3文程度
- 次に確認すべきポイントを優先順で示す
- 主にブロック4（なぜ動いたか）を見るべき理由がわかるようにする
- 冗長にしない
"""

    user_prompt = f"""
対象月: {selected_month}
前月: {previous_month}

【ブロック3サマリー】
{block3_summary_comment}

【セグメント差プラス】
{df_to_text(seg_pos_df)}

【セグメント差マイナス】
{df_to_text(seg_neg_df)}

【施策影響】
{df_to_text(ad_impact_df)}

この内容を踏まえて、次にどこを見るべきかを簡潔に案内してください。
"""

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.2,
        max_tokens=250,
    )

    return response.choices[0].message.content.strip()


def generate_block4_navigation_comment(
    block4_summary_comment: str,
    ad_impact_df: pd.DataFrame,
    selected_month: str,
    previous_month: str
) -> str:
    def df_to_text(df: pd.DataFrame, max_rows: int = 5) -> str:
        if df is None or df.empty:
            return "該当なし"
        return df.head(max_rows).to_string(index=False)

    system_prompt = """
あなたは市場調査分析のシニアアナリストです。
「なぜ動いたか？」を確認したブランドマネージャーに対し、
次に何をすべきかを簡潔に案内してください。

要件:
- 日本語
- 2〜3文程度
- ここまでの確認結果を踏まえて、報告に使うか、右カラムのピン留めコメントを整理するかを案内する
- 必要に応じて、重要な示唆を報告用に残すよう促す
- 冗長にしない
"""

    user_prompt = f"""
対象月: {selected_month}
前月: {previous_month}

【ブロック4サマリー】
{block4_summary_comment}

【施策影響】
{df_to_text(ad_impact_df)}

この内容を踏まえて、次に何をすべきかを簡潔に案内してください。
"""

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.2,
        max_tokens=250,
    )

    return response.choices[0].message.content.strip()



def generate_block2_summary_comment(
    top_positive_df: pd.DataFrame,
    top_negative_df: pd.DataFrame,
    selected_month: str,
    previous_month: str
) -> str:
    def df_to_text(df: pd.DataFrame, max_rows: int = 3) -> str:
        if df is None or df.empty:
            return "該当なし"
        return df.head(max_rows).to_string(index=False)

    system_prompt = """
あなたは市場調査分析のシニアアナリストです。
KPIの改善上位指標と悪化上位指標をまとめて、ブランドマネージャー向けに簡潔な総括コメントを作成してください。

要件:
- 日本語
- 2〜4文程度
- 改善した指標群と悪化した指標群の両方に触れる
- 「今月どの指標が動いたか」がひと目でわかる内容にする
- 表の数値をそのまま羅列するのではなく、意味が伝わる文章にする
- レポート本文に近い自然な文章にする
"""

    user_prompt = f"""
対象月: {selected_month}
前月: {previous_month}

【変化量がプラスに大きい指標 Top3】
{df_to_text(top_positive_df)}

【変化量がマイナスに大きい指標 Top3】
{df_to_text(top_negative_df)}

この情報をもとに、
「どの指標が動いたか？」に答える簡潔な総括コメントを作成してください。
"""

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.2,
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()



def generate_block3_summary_comment(
    seg_pos_df: pd.DataFrame,
    seg_neg_df: pd.DataFrame,
    selected_month: str,
    previous_month: str
) -> str:
    def df_to_text(df: pd.DataFrame, max_rows: int = 5) -> str:
        if df is None or df.empty:
            return "該当なし"
        return df.head(max_rows).to_string(index=False)

    system_prompt = """
あなたは市場調査分析のシニアアナリストです。
セグメント差が大きい指標の情報をもとに、ブランドマネージャー向けの簡潔な総括コメントを作成してください。

要件:
- 日本語
- 2〜4文程度
- セグメント差がプラスに大きい指標とマイナスに大きい指標の両方に触れる
- 「誰が動いたか」がひと目でわかる内容にする
- 表の値を並べるだけでなく、どの層・どの動きに注目すべきかが伝わる文章にする
- レポート本文に近い自然な文章にする
"""

    user_prompt = f"""
対象月: {selected_month}
前月: {previous_month}

【セグメント差がプラスに大きい指標】
{df_to_text(seg_pos_df)}

【セグメント差がマイナスに大きい指標】
{df_to_text(seg_neg_df)}

この情報をもとに、
「誰が動いたか？」に答える簡潔な総括コメントを作成してください。
"""

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.2,
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()


def generate_block4_summary_comment(
    ad_impact_df: pd.DataFrame,
    selected_month: str,
    previous_month: str
) -> str:
    def df_to_text(df: pd.DataFrame, max_rows: int = 5) -> str:
        if df is None or df.empty:
            return "該当なし"
        return df.head(max_rows).to_string(index=False)

    system_prompt = """
あなたは市場調査分析のシニアアナリストです。
施策影響が示唆される指標の情報をもとに、ブランドマネージャー向けの簡潔な総括コメントを作成してください。

要件:
- 日本語
- 2〜4文程度
- どの指標に施策影響の示唆があるかがわかる内容にする
- 接触広告との関係に触れる
- 表の値を並べるだけでなく、何を確認すべきかが伝わる文章にする
- レポート本文に近い自然な文章にする
"""

    user_prompt = f"""
対象月: {selected_month}
前月: {previous_month}

【施策影響が示唆される指標】
{df_to_text(ad_impact_df)}

この情報をもとに、
「なぜ動いたか？」に答える簡潔な総括コメントを作成してください。
"""

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.2,
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()




def pin_current_comment() -> None:
    current = st.session_state.current_llm_comment
    if current is None:
        return

    new_item = {
        "id": str(uuid.uuid4()),
        "指標番号": current["指標番号"],
        "指標名": current["指標名"],
        "セグメント番号": current["セグメント番号"],
        "セグメント名": current["セグメント名"],
        "広告名": current.get("広告名", ""),
        "comment": current["comment"],
    }
    st.session_state.pinned_comments.append(new_item)

def generate_consistency_llm_comment(
    result_df: pd.DataFrame,
    selected_month: str,
    previous_month: str
) -> str:
    if result_df is None or result_df.empty:
        return "整合性チェック対象のデータがありません。"

    alert_df = result_df[result_df["status"] == "整合性注意"].copy()

    if alert_df.empty:
        return (
            f"{selected_month} と {previous_month} の比較では、"
            "設定したKPI順序ルールに照らして大きな整合性崩れは確認されませんでした。"
            "現時点では、主要KPIの前後関係は概ね自然な範囲で推移しているとみられます。"
        )

    alert_text = "\n\n".join(
        [
            f"ルール: {row['label']}\n"
            f"上流指標: {row['upstream']} / 差分: {row['upstream_diff']}\n"
            f"下流指標: {row['downstream']} / 差分: {row['downstream_diff']}\n"
            f"判定理由: {row['reason']}"
            for _, row in alert_df.iterrows()
        ]
    )

    system_prompt = """
あなたは市場調査分析のシニアアナリストです。
KPI順序の整合性チェック結果をもとに、報告前の注意コメントを日本語で簡潔に作成してください。

要件:
- 3〜5文程度
- 整合性注意が出たKPI接続について、どこが説明しづらいかを明示する
- 因果を断定しない
- 社内報告時にどのような言い方が安全かも示す
- レポートに転記しやすい自然な文章にする
"""

    user_prompt = f"""
対象期間:
{selected_month} と {previous_month}

整合性注意の判定結果:
{alert_text}

依頼:
上記をもとに、報告前の注意コメントを作成してください。
"""

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.3,
        max_tokens=600,
    )

    return response.choices[0].message.content.strip()








# =========================
# サイドバー
# =========================
with st.sidebar:
    st.title("KPI分析ツール")
    st.markdown("### モード選択")

    mode_list = [
        "総括",
        "KPI変化",
        "報告前の整合性チェック",
        # "施策効果",
        # "セグメント動向",
        # "時系列変化",
        "データプレビュー",
    ]

    for mode in mode_list:
        button_type = "primary" if st.session_state.mode == mode else "secondary"
        if st.button(mode, use_container_width=True, type=button_type):
            set_mode(mode)

    st.markdown("---")
    st.markdown("### データ読み込み")
    uploaded_file = st.file_uploader(
        "CSVまたはExcelファイルをアップロード",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        st.success(f"読み込みファイル: {uploaded_file.name}")


# =========================
# アップロードファイル変更時の初期化
# =========================
if uploaded_file is not None:
    if st.session_state.uploaded_file_name != uploaded_file.name:
        reset_analysis_states()
        st.session_state.uploaded_file_name = uploaded_file.name


# =========================
# メイン画面
# =========================
st.title("KPI分析ダッシュボード")
st.caption("左ペインでモードを選択し、中央ペインで分析、右ペインでLLMコメントのピン留め管理を行います。")

selected_mode = st.session_state.mode
st.subheader(f"現在のモード: {selected_mode}")

main_col, pin_col = st.columns([4, 1.6], gap="large")


# =========================
# 右カラム：ピン留め欄
# =========================
with pin_col:
    st.markdown("### ピン留めコメント")

    if st.session_state.current_llm_comment is not None:
        with st.container(border=True):
            st.markdown("#### 現在のコメント")
            st.write(f"**指標**: {st.session_state.current_llm_comment['指標番号']} - {st.session_state.current_llm_comment['指標名']}")
            st.write(f"**セグメント**: {st.session_state.current_llm_comment['セグメント番号']} - {st.session_state.current_llm_comment['セグメント名']}")
            st.write(f"**広告**: {st.session_state.current_llm_comment.get('広告名', '-')}")
            st.write(st.session_state.current_llm_comment["comment"])

            if st.button("このコメントをピン留め", use_container_width=True):
                pin_current_comment()
                st.success("コメントをピン留めしました。")
                st.rerun()

    st.markdown("---")

    if not st.session_state.pinned_comments:
        st.info("ピン留めしたコメントはここに表示されます。")
    else:
        for item in st.session_state.pinned_comments:
            with st.container(border=True):
                if item.get("type") == "overview_summary":
                    st.write("**種類**: 総括コメント")
                    st.write(f"**対象月**: {item.get('selected_month', '-')}")
                    st.write(f"**前月比較**: {item.get('previous_month', '-')}")
                    st.write(item["comment"])

                elif item.get("type") == "block2_summary":
                    st.write("**種類**: ブロック2サマリー")
                    st.write(f"**対象月**: {item.get('selected_month', '-')}")
                    st.write(f"**前月比較**: {item.get('previous_month', '-')}")
                    st.write(item["comment"])

                elif item.get("type") == "block3_summary":
                    st.write("**種類**: ブロック3サマリー")
                    st.write(f"**対象月**: {item.get('selected_month', '-')}")
                    st.write(f"**前月比較**: {item.get('previous_month', '-')}")
                    st.write(item["comment"])

                elif item.get("type") == "block4_summary":
                    st.write("**種類**: ブロック4サマリー")
                    st.write(f"**対象月**: {item.get('selected_month', '-')}")
                    st.write(f"**前月比較**: {item.get('previous_month', '-')}")
                    st.write(item["comment"])

                else:
                    st.write(f"**指標**: {item['指標番号']} - {item['指標名']}")
                    st.write(f"**セグメント**: {item['セグメント番号']} - {item['セグメント名']}")
                    st.write(f"**広告**: {item.get('広告名', '-')}")
                    st.write(item["comment"])

                if st.button("削除", key=f"delete_pin_{item['id']}", use_container_width=True):
                    st.session_state.pinned_comments = [
                        x for x in st.session_state.pinned_comments if x["id"] != item["id"]
                    ]
                    st.rerun()

    st.markdown("---")

    if st.button("ピン留めコメントからサマリー生成", use_container_width=True):
        try:
            if not st.session_state.pinned_comments:
                st.warning("先にコメントをピン留めしてください。")
            else:
                with st.spinner("最終サマリーを生成中です..."):
                    st.session_state.final_summary = generate_final_summary(
                        st.session_state.pinned_comments
                    )
                st.success("サマリーを生成しました。")
        except Exception as e:
            st.error(f"サマリー生成中にエラーが発生しました: {e}")

    if st.session_state.final_summary:
        with st.container(border=True):
            st.markdown("### 最終サマリー")
            st.write(st.session_state.final_summary)

# =========================
# 中央カラム：分析エリア
# =========================
with main_col:
    result_container = st.container(border=True)

    with result_container:
        st.markdown("### レビューエリア")

        if uploaded_file is None:
            st.info("左ペイン下部のデータ読み込みポートからファイルをアップロードしてください。")
        else:
            st.write(f"現在アップロード中のファイル: **{uploaded_file.name}**")

        if selected_mode == "総括":
            st.write("総括モードの結果をここに表示します。")

            if uploaded_file is None:
                st.warning("総括の表示にはファイルアップロードが必要です。")
            else:
                try:
                    if st.session_state.raw_df is None:
                        st.session_state.raw_df = load_data(uploaded_file)

                    overview_df = st.session_state.raw_df.copy()

                    month_options = get_sorted_month_options(overview_df)

                    if not month_options:
                        st.warning("調査月の選択肢が見つかりません。")
                    else:
                        selected_month = st.selectbox(
                            "調査月を選択してください",
                            options=month_options,
                            index=0,
                            key="overview_selected_month"
                        )

                        previous_month = get_previous_month(selected_month, month_options)

                        if previous_month is None:
                            st.warning("選択月の前月データが見つからないため、総括を表示できません。")
                        else:
                            metric_diff_df = build_overview_metric_diff_table(
                                overview_df,
                                selected_month=selected_month,
                                previous_month=previous_month
                            )

                            segment_diff_df = build_segment_diff_table(
                                overview_df,
                                selected_month=selected_month,
                                previous_month=previous_month
                            )

                            seg_pos_df, seg_neg_df = build_segment_top3_summary_tables(
                                segment_diff_df,
                                selected_month=selected_month,
                                previous_month=previous_month
                            )

                            ad_impact_df = build_ad_impact_table_for_overview(
                                overview_df,
                                selected_month=selected_month
                            )

                            diff_col = f"{selected_month} ー {previous_month}"

                            top_positive_df = (
                                metric_diff_df.sort_values(diff_col, ascending=False).head(3).reset_index(drop=True)
                                if metric_diff_df is not None and not metric_diff_df.empty
                                else pd.DataFrame()
                            )

                            top_negative_df = (
                                metric_diff_df.sort_values(diff_col, ascending=True).head(3).reset_index(drop=True)
                                if metric_diff_df is not None and not metric_diff_df.empty
                                else pd.DataFrame()
                            )

                            summary_text = create_overview_summary_text(
                                top_positive_df=top_positive_df,
                                top_negative_df=top_negative_df,
                                ad_impact_df=ad_impact_df,
                                selected_month=selected_month,
                                previous_month=previous_month
                            )

                            try:
                                navigation_text = generate_overview_navigation_comment(
                                    summary_text=summary_text,
                                    top_positive_df=top_positive_df,
                                    top_negative_df=top_negative_df,
                                    seg_pos_df=seg_pos_df,
                                    seg_neg_df=seg_neg_df,
                                    ad_impact_df=ad_impact_df,
                                    selected_month=selected_month,
                                    previous_month=previous_month
                                )
                                st.session_state.overview_navigation_comment = navigation_text
                            except Exception:
                                st.session_state.overview_navigation_comment = "次に見るべきポイントの生成に失敗しました。ブロック2〜4を順に確認してください。"

                            try:
                                block2_summary_comment = generate_block2_summary_comment(
                                    top_positive_df=top_positive_df,
                                    top_negative_df=top_negative_df,
                                    selected_month=selected_month,
                                    previous_month=previous_month
                                )
                                st.session_state.block2_summary_comment = block2_summary_comment
                            except Exception:
                                st.session_state.block2_summary_comment = "主要な改善指標と悪化指標を確認し、今月どのKPIが大きく動いたかを把握してください。"

                            try:
                                block2_navigation_comment = generate_block2_navigation_comment(
                                    block2_summary_comment=st.session_state.block2_summary_comment,
                                    top_positive_df=top_positive_df,
                                    top_negative_df=top_negative_df,
                                    seg_pos_df=seg_pos_df,
                                    seg_neg_df=seg_neg_df,
                                    ad_impact_df=ad_impact_df,
                                    selected_month=selected_month,
                                    previous_month=previous_month
                                )
                                st.session_state.block2_navigation_comment = block2_navigation_comment
                            except Exception:
                                st.session_state.block2_navigation_comment = "次に、ブロック3でどのセグメントが変化を動かしたかを確認し、必要に応じてブロック4で施策との関係も見てください。"



                            try:
                                block3_summary_comment = generate_block3_summary_comment(
                                    seg_pos_df=seg_pos_df,
                                    seg_neg_df=seg_neg_df,
                                    selected_month=selected_month,
                                    previous_month=previous_month
                                )
                                st.session_state.block3_summary_comment = block3_summary_comment
                            except Exception:
                                st.session_state.block3_summary_comment = "セグメント差が大きい指標を確認し、どの層の動きが全体変化に影響しているかを把握してください。"

                            try:
                                block3_navigation_comment = generate_block3_navigation_comment(
                                    block3_summary_comment=st.session_state.block3_summary_comment,
                                    seg_pos_df=seg_pos_df,
                                    seg_neg_df=seg_neg_df,
                                    ad_impact_df=ad_impact_df,
                                    selected_month=selected_month,
                                    previous_month=previous_month
                                )
                                st.session_state.block3_navigation_comment = block3_navigation_comment
                            except Exception:
                                st.session_state.block3_navigation_comment = "次に、ブロック4で施策影響が示唆される指標を確認し、変化要因を整理してください。"



                            try:
                                block4_summary_comment = generate_block4_summary_comment(
                                    ad_impact_df=ad_impact_df,
                                    selected_month=selected_month,
                                    previous_month=previous_month
                                )
                                st.session_state.block4_summary_comment = block4_summary_comment
                            except Exception:
                                st.session_state.block4_summary_comment = "施策影響が示唆される指標を確認し、広告接触との関係から変化要因を把握してください。"

                            try:
                                block4_navigation_comment = generate_block4_navigation_comment(
                                    block4_summary_comment=st.session_state.block4_summary_comment,
                                    ad_impact_df=ad_impact_df,
                                    selected_month=selected_month,
                                    previous_month=previous_month
                                )
                                st.session_state.block4_navigation_comment = block4_navigation_comment
                            except Exception:
                                st.session_state.block4_navigation_comment = "重要な示唆は『報告用に使う』で右カラムに残し、最後にピン留めコメントからサマリーを生成してください。"


                            st.markdown(f"#### 対象月: {selected_month}")
                            st.caption(f"前月比較: {previous_month}")

                            # ==================================================
                            # ブロック1
                            # ==================================================
                            with st.container(border=True):
                                st.markdown("### ブロック1｜今月は何が起きたか？")
                                st.markdown("#### 全体の改善 / 悪化サマリー")
                                st.write(summary_text)

                                st.markdown("#### 次に確認するとよいポイント")
                                if st.session_state.overview_navigation_comment:
                                    st.info(st.session_state.overview_navigation_comment)
                                else:
                                    st.info("次に確認するとよいポイントを表示します。")

                                if st.button("報告用に使う", key="pin_overview_summary_btn", use_container_width=False):
                                    new_id = (
                                        max([x.get("id", 0) for x in st.session_state.pinned_comments], default=0) + 1
                                    )

                                    st.session_state.pinned_comments.append({
                                        "id": new_id,
                                        "type": "overview_summary",
                                        "selected_month": selected_month,
                                        "previous_month": previous_month,
                                        "comment": summary_text,
                                    })
                                    st.success("総括コメントをピン留めしました。")
                                    st.rerun()

                            # ==================================================
                            # ブロック2
                            # ==================================================
                            with st.container(border=True):
                                st.markdown("### ブロック2｜どの指標が動いたか？")

                                st.markdown("#### サマリーコメント")
                                if st.session_state.block2_summary_comment:
                                    st.write(st.session_state.block2_summary_comment)
                                else:
                                    st.info("どの指標が動いたかの要約を表示します。")

                                st.markdown("#### 次に確認するとよいポイント")
                                if st.session_state.block2_navigation_comment:
                                    st.info(st.session_state.block2_navigation_comment)
                                else:
                                    st.info("次に確認するとよいポイントを表示します。")

                                if st.button("報告用に使う", key="pin_block2_summary_btn", use_container_width=False):
                                    new_id = (
                                        max([x.get('id', 0) for x in st.session_state.pinned_comments], default=0) + 1
                                    )

                                    st.session_state.pinned_comments.append({
                                        "id": new_id,
                                        "type": "block2_summary",
                                        "selected_month": selected_month,
                                        "previous_month": previous_month,
                                        "comment": st.session_state.block2_summary_comment,
                                    })
                                    st.success("ブロック2のサマリーコメントをピン留めしました。")
                                    st.rerun()

                                row2_col1, row2_col2 = st.columns(2)

                                with row2_col1:
                                    st.markdown("#### 変化量がプラスに大きい指標 Top3")
                                    if top_positive_df.empty:
                                        st.info("該当データがありません。")
                                    else:
                                        display_df = top_positive_df[["指標番号", "指標名", diff_col]].copy()
                                        st.dataframe(
                                            display_df,
                                            use_container_width=True,
                                            hide_index=True,
                                            height=180
                                        )

                                with row2_col2:
                                    st.markdown("#### 変化量がマイナスに大きい指標 Top3")
                                    if top_negative_df.empty:
                                        st.info("該当データがありません。")
                                    else:
                                        display_df = top_negative_df[["指標番号", "指標名", diff_col]].copy()
                                        st.dataframe(
                                            display_df,
                                            use_container_width=True,
                                            hide_index=True,
                                            height=180
                                        )

                            # ==================================================
                            # ブロック3
                            # ==================================================
                            with st.container(border=True):
                                st.markdown("### ブロック3｜誰が動いたか？")

                                st.markdown("#### サマリーコメント")
                                if st.session_state.block3_summary_comment:
                                    st.write(st.session_state.block3_summary_comment)
                                else:
                                    st.info("どのセグメントが動いたかの要約を表示します。")

                                st.markdown("#### 次に確認するとよいポイント")
                                if st.session_state.block3_navigation_comment:
                                    st.info(st.session_state.block3_navigation_comment)
                                else:
                                    st.info("次に確認するとよいポイントを表示します。")

                                if st.button("報告用に使う", key="pin_block3_summary_btn", use_container_width=False):
                                    new_id = (
                                        max([x.get("id", 0) for x in st.session_state.pinned_comments], default=0) + 1
                                    )

                                    st.session_state.pinned_comments.append({
                                        "id": new_id,
                                        "type": "block3_summary",
                                        "selected_month": selected_month,
                                        "previous_month": previous_month,
                                        "comment": st.session_state.block3_summary_comment,
                                    })
                                    st.success("ブロック3のサマリーコメントをピン留めしました。")
                                    st.rerun()

                                st.markdown("#### セグメント差がプラスに大きい指標")
                                if seg_pos_df.empty:
                                    st.info("該当データがありません。")
                                else:
                                    st.dataframe(
                                        seg_pos_df,
                                        use_container_width=True,
                                        hide_index=True,
                                        height=320
                                    )

                                st.markdown("")

                                st.markdown("#### セグメント差がマイナスに大きい指標")
                                if seg_neg_df.empty:
                                    st.info("該当データがありません。")
                                else:
                                    st.dataframe(
                                        seg_neg_df,
                                        use_container_width=True,
                                        hide_index=True,
                                        height=320
                                    )

                            st.markdown("")

                            # ==================================================
                            # ブロック4
                            # ==================================================
                            with st.container(border=True):
                                st.markdown("### ブロック4｜なぜ動いたか？")

                                st.markdown("#### サマリーコメント")
                                if st.session_state.block4_summary_comment:
                                    st.write(st.session_state.block4_summary_comment)
                                else:
                                    st.info("施策影響が示唆される指標の要約を表示します。")

                                st.markdown("#### 次に確認するとよいポイント")
                                if st.session_state.block4_navigation_comment:
                                    st.info(st.session_state.block4_navigation_comment)
                                else:
                                    st.info("次に確認するとよいポイントを表示します。")

                                if st.button("報告用に使う", key="pin_block4_summary_btn", use_container_width=False):
                                    new_id = (
                                        max([x.get("id", 0) for x in st.session_state.pinned_comments], default=0) + 1
                                    )

                                    st.session_state.pinned_comments.append({
                                        "id": new_id,
                                        "type": "block4_summary",
                                        "selected_month": selected_month,
                                        "previous_month": previous_month,
                                        "comment": st.session_state.block4_summary_comment,
                                    })
                                    st.success("ブロック4のサマリーコメントをピン留めしました。")
                                    st.rerun()

                                st.markdown("#### 施策影響が示唆される指標")
                                if ad_impact_df.empty:
                                    st.info("該当データがありません。")
                                else:
                                    st.dataframe(
                                        ad_impact_df[["指標番号", "指標名", "接触広告", "値"]],
                                        use_container_width=True,
                                        hide_index=True,
                                        height=320
                                    )

                except Exception as e:
                    st.error(f"総括の表示中にエラーが発生しました: {e}")


        elif selected_mode == "KPI変化":
            st.write("KPI変化の結果をここに表示します。")

            if uploaded_file is None:
                st.warning("KPI変化の表示にはファイルアップロードが必要です。")
            else:
                try:
                    if st.session_state.raw_df is None:
                        st.session_state.raw_df = load_data(uploaded_file)

                    raw_df = st.session_state.raw_df.copy()

                    month_options = get_sorted_month_options(raw_df)

                    if not month_options:
                        st.warning("調査月の選択肢が見つかりません。")
                    else:
                        selected_month = st.selectbox(
                            "比較対象の調査月を選択してください",
                            options=month_options,
                            index=0,
                            key="kpi_change_selected_month"
                        )

                        previous_month = get_previous_month(selected_month, month_options)

                        if previous_month is None:
                            st.warning("選択月の前月データが見つからないため、KPI変化を表示できません。")
                        else:
                            diff_col = f"{selected_month} ー{previous_month}（2期の差分）"

                            st.caption(f"比較: {selected_month} vs {previous_month}")

                            kpi_df = build_kpi_change_table(
                                raw_df,
                                selected_month=selected_month,
                                previous_month=previous_month
                            )

                            if kpi_df.empty:
                                st.warning("KPI変化テーブルを作成できるデータがありません。")
                            else:
                                st.markdown("#### KPI変化一覧")

                                header_cols = st.columns([1, 4, 3, 2])
                                header_cols[0].markdown("**指標番号**")
                                header_cols[1].markdown("**指標名**")
                                header_cols[2].markdown(f"**{diff_col}**")
                                header_cols[3].markdown("**操作**")

                                for _, row in kpi_df.iterrows():
                                    row_cols = st.columns([1, 4, 3, 2])
                                    row_cols[0].write(int(row["指標番号"]) if pd.notna(row["指標番号"]) else "")
                                    row_cols[1].write(row["指標名"])
                                    row_cols[2].write(row[diff_col])

                                    metric_no = int(row["指標番号"]) if pd.notna(row["指標番号"]) else None
                                    detail_key = f"kpi_detail_btn_{selected_month}_{metric_no}"

                                    if row_cols[3].button("詳細分析", key=detail_key, use_container_width=True):
                                        st.session_state.kpi_detail_clicked = {
                                            "指標番号": metric_no,
                                            "指標名": row["指標名"],
                                            "差分値": row[diff_col],
                                            "選択月": selected_month,
                                            "前月": previous_month,
                                        }
                                        st.session_state.kpi_detail_table = build_kpi_detail_table(
                                            raw_df,
                                            selected_metric_no=metric_no,
                                            selected_month=selected_month,
                                            previous_month=previous_month
                                        )
                                        st.session_state.ad_effect_clicked = None
                                        st.session_state.ad_effect_table = None
                                        st.session_state.current_llm_comment = None
                                        st.rerun()

                                st.markdown("---")
                                st.markdown("#### 詳細分析結果表示エリア")

                                if st.session_state.kpi_detail_clicked is None:
                                    st.info("各指標の「詳細分析」ボタンを押すと結果をここに表示します。")
                                else:
                                    selected_info = st.session_state.kpi_detail_clicked
                                    st.write(f"選択中の指標: {selected_info['指標番号']} - {selected_info['指標名']}")
                                    st.write(f"全体差分: {selected_info['差分値']}")
                                    st.write(f"比較月: {selected_info['選択月']} vs {selected_info['前月']}")

                                    detail_df = st.session_state.kpi_detail_table
                                    selected_diff_col = f"{selected_info['選択月']} ー{selected_info['前月']}（2期の差分）"

                                    if detail_df is None or detail_df.empty:
                                        st.warning("詳細分析用のセグメント差分データがありません。")
                                    else:
                                        st.markdown("##### セグメント別 2期差分")

                                        header_cols = st.columns([1, 4, 3, 2])
                                        header_cols[0].markdown("**セグメント番号**")
                                        header_cols[1].markdown("**セグメント名**")
                                        header_cols[2].markdown(f"**{selected_diff_col}**")
                                        header_cols[3].markdown("**操作**")

                                        for _, seg_row in detail_df.iterrows():
                                            seg_no = int(seg_row["セグメント番号"]) if pd.notna(seg_row["セグメント番号"]) else None

                                            seg_cols = st.columns([1, 4, 3, 2])
                                            seg_cols[0].write(seg_no if seg_no is not None else "")
                                            seg_cols[1].write(seg_row["セグメント名"])
                                            seg_cols[2].write(seg_row[selected_diff_col])

                                            ad_effect_key = (
                                                f"ad_effect_btn_"
                                                f"{selected_info['選択月']}_"
                                                f"{selected_info['指標番号']}_"
                                                f"{seg_no}"
                                            )

                                            if seg_cols[3].button("広告効果分析", key=ad_effect_key, use_container_width=True):
                                                selected_metric_no = selected_info["指標番号"]
                                                selected_segment_no = seg_no

                                                st.session_state.ad_effect_clicked = {
                                                    "指標番号": selected_metric_no,
                                                    "指標名": selected_info["指標名"],
                                                    "セグメント番号": selected_segment_no,
                                                    "セグメント名": seg_row["セグメント名"],
                                                    "セグメント差分値": seg_row[selected_diff_col],
                                                    "選択月": selected_info["選択月"],
                                                    "前月": selected_info["前月"],
                                                }

                                                st.session_state.ad_effect_table = build_ad_effect_table(
                                                    raw_df,
                                                    selected_metric_no=selected_metric_no,
                                                    selected_segment_no=selected_segment_no,
                                                    selected_month=selected_info["選択月"]
                                                )
                                                st.session_state.current_llm_comment = None
                                                st.rerun()

                                        st.markdown("---")
                                        st.markdown("##### 広告接触効果分析")

                                        if st.session_state.ad_effect_clicked is None:
                                            st.info("各セグメントの「広告効果分析」ボタンを押すと結果をここに表示します。")
                                        else:
                                            ad_info = st.session_state.ad_effect_clicked
                                            st.write(f"対象指標: {ad_info['指標番号']} - {ad_info['指標名']}")
                                            st.write(f"対象セグメント: {ad_info['セグメント番号']} - {ad_info['セグメント名']}")
                                            st.write(f"対象月: {ad_info['選択月']}")

                                            ad_df = st.session_state.ad_effect_table

                                            if ad_df is None or ad_df.empty:
                                                st.warning("広告効果分析用のデータがありません。")
                                            else:
                                                st.markdown("#### 広告別一覧")

                                                header_cols = st.columns([4, 2, 2])
                                                header_cols[0].markdown("**接触広告**")
                                                header_cols[1].markdown("**値**")
                                                header_cols[2].markdown("**操作**")

                                                total_row = ad_df[ad_df["接触広告"].astype(str) == "合計値"]
                                                total_value = None
                                                if not total_row.empty:
                                                    total_value = total_row.iloc[0]["値"]

                                                for _, ad_row in ad_df.iterrows():
                                                    ad_name = str(ad_row["接触広告"])
                                                    ad_value = ad_row["値"]

                                                    row_cols = st.columns([4, 2, 2])
                                                    row_cols[0].write(ad_name)
                                                    row_cols[1].write(ad_value)

                                                    btn_key = (
                                                        f"generate_ad_comment_"
                                                        f"{ad_info['選択月']}_"
                                                        f"{ad_info['指標番号']}_"
                                                        f"{ad_info['セグメント番号']}_"
                                                        f"{ad_name}"
                                                    )

                                                    if row_cols[2].button("分析コメント生成", key=btn_key, use_container_width=True):
                                                        try:
                                                            with st.spinner(f"「{ad_name}」の分析コメントを生成中です..."):
                                                                llm_comment = generate_llm_comment_for_ad(
                                                                    metric_info=selected_info,
                                                                    segment_info=ad_info,
                                                                    selected_ad_name=ad_name,
                                                                    selected_ad_value=ad_value,
                                                                    total_value=total_value,
                                                                    ad_df=ad_df
                                                                )

                                                            st.session_state.current_llm_comment = {
                                                                "指標番号": ad_info["指標番号"],
                                                                "指標名": ad_info["指標名"],
                                                                "セグメント番号": ad_info["セグメント番号"],
                                                                "セグメント名": ad_info["セグメント名"],
                                                                "広告名": ad_name,
                                                                "comment": llm_comment,
                                                            }
                                                            st.success(f"「{ad_name}」の分析コメントを生成しました。")
                                                            st.rerun()

                                                        except Exception as e:
                                                            st.error(f"LLMコメント生成中にエラーが発生しました: {e}")

                                                st.markdown("---")
                                                st.markdown("##### LLM分析コメント")

                                                btn_col1, btn_col2 = st.columns([1, 1])

                                                with btn_col1:
                                                    if st.button("現在コメントをクリア", key="clear_current_comment_btn", use_container_width=True):
                                                        st.session_state.current_llm_comment = None
                                                        st.rerun()

                                                with btn_col2:
                                                    st.empty()

                                                if st.session_state.current_llm_comment is not None:
                                                    current_comment = st.session_state.current_llm_comment
                                                    with st.container(border=True):
                                                        st.write(f"**対象広告**: {current_comment.get('広告名', '-')}")
                                                        st.write(current_comment["comment"])
                                                else:
                                                    st.info("広告ごとの「分析コメント生成」ボタンを押すと、ここにコメントを表示します。")

                except Exception as e:
                    st.error(f"KPI変化の表示中にエラーが発生しました: {e}")


        elif selected_mode == "報告前の整合性チェック":
            st.write("KPI順序の整合性チェック結果をここに表示します。")

            if uploaded_file is None:
                st.warning("整合性チェックの表示にはファイルアップロードが必要です。")
            else:
                try:
                    if st.session_state.raw_df is None:
                        st.session_state.raw_df = load_data(uploaded_file)

                    raw_df = st.session_state.raw_df.copy()

                    month_options = get_sorted_month_options(raw_df)

                    if not month_options:
                        st.warning("調査月の選択肢が見つかりません。")
                    else:
                        selected_month = st.selectbox(
                            "整合性チェック対象の調査月を選択してください",
                            options=month_options,
                            index=0,
                            key="consistency_selected_month"
                        )

                        previous_month = get_previous_month(selected_month, month_options)

                        if previous_month is None:
                            st.warning("選択月の前月データが見つからないため、整合性チェックを表示できません。")
                        else:
                            st.caption(f"比較: {selected_month} vs {previous_month}")
                            st.caption(f"対象: 実測値 / 接触広告=合計値 / セグメント=TOTAL / 閾値=±{CONSISTENCY_THRESHOLD:.1f}pt")

                            diff_df = build_total_kpi_diff_table(
                                raw_df,
                                selected_month=selected_month,
                                previous_month=previous_month
                            )

                            result_df = evaluate_kpi_consistency_rules(
                                diff_df=diff_df,
                                rules=KPI_CONSISTENCY_RULES,
                                selected_month=selected_month,
                                previous_month=previous_month,
                                threshold=CONSISTENCY_THRESHOLD
                            )

                            st.session_state.consistency_check_results = result_df

                            counts = summarize_consistency_counts(result_df)

                            summary_col1, summary_col2, summary_col3 = st.columns(3)

                            with summary_col1:
                                with st.container(border=True):
                                    st.markdown("### 整合性注意")
                                    st.metric(label="件数", value=counts["整合性注意"])

                            with summary_col2:
                                with st.container(border=True):
                                    st.markdown("### 問題なし")
                                    st.metric(label="件数", value=counts["問題なし"])

                            with summary_col3:
                                with st.container(border=True):
                                    st.markdown("### 判定対象外")
                                    st.metric(label="件数", value=counts["判定対象外"])

                            st.markdown("### 判定結果一覧")

                            if result_df.empty:
                                st.info("整合性チェック結果がありません。")
                            else:
                                display_df = result_df.copy()
                                st.dataframe(
                                    display_df[[
                                        "label",
                                        "upstream",
                                        "upstream_diff",
                                        "downstream",
                                        "downstream_diff",
                                        "status",
                                        "reason"
                                    ]],
                                    use_container_width=True,
                                    hide_index=True,
                                    height=380
                                )

                            st.markdown("### KPI構造")
                            with st.container(border=True):
                                st.markdown("""
        - 純粋想起_Q1（MA） → 助成想起_Q6（MA）
        - 助成想起_Q6（MA） → 好意TOP2_Q11
        - 好意TOP2_Q11 → 利用意向_Q10_1（MA）
        - 利用意向_Q10_1（MA） → 第一利用意向_Q10_2（MA）
        - 第一利用意向_Q10_2（MA） → 1年以内利用_Q9_2（MA）
        - 1年以内利用_Q9_2（MA） → 満足TOP2_Q12
        - 利用経験_Q9_1（MA） → 好意TOP2_Q11
                                """)

                            st.markdown("### LLMによる解釈コメント")

                            btn_col1, btn_col2 = st.columns([1, 1])

                            with btn_col1:
                                if st.button("解釈コメントを生成", key="generate_consistency_llm_btn", use_container_width=True):
                                    with st.spinner("整合性チェックの解釈コメントを生成中です..."):
                                        st.session_state.consistency_llm_comment = generate_consistency_llm_comment(
                                            result_df=result_df,
                                            selected_month=selected_month,
                                            previous_month=previous_month
                                        )
                                    st.success("解釈コメントを生成しました。")
                                    st.rerun()

                            with btn_col2:
                                if st.button("解釈コメントをクリア", key="clear_consistency_llm_btn", use_container_width=True):
                                    st.session_state.consistency_llm_comment = None
                                    st.rerun()

                            if st.session_state.consistency_llm_comment:
                                with st.container(border=True):
                                    st.write(st.session_state.consistency_llm_comment)
                            else:
                                st.info("「解釈コメントを生成」を押すと、報告時の注意点を自然文で表示します。")

                except Exception as e:
                    st.error(f"整合性チェックの表示中にエラーが発生しました: {e}")


        # elif selected_mode == "施策効果":
        #     st.write("施策効果モードの結果をここに表示します。")

        # elif selected_mode == "セグメント動向":
        #     st.write("セグメント動向モードの結果をここに表示します。")

        # elif selected_mode == "時系列変化":
        #     st.write("時系列変化モードの結果をここに表示します。")

        elif selected_mode == "データプレビュー":
            st.write("データプレビューの結果をここに表示します。")

            if uploaded_file is None:
                st.warning("データプレビューにはファイルアップロードが必要です。")
            else:
                try:
                    df = load_data(uploaded_file)

                    st.markdown("#### 読み込みデータ情報")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"行数: {len(df):,}")
                    with col2:
                        st.write(f"列数: {len(df.columns):,}")

                    month_options = df["調査月"].dropna().astype(str).drop_duplicates().tolist()
                    ad_options = df["接触広告"].dropna().astype(str).drop_duplicates().tolist()

                    if not month_options:
                        st.error("調査月列に選択可能な値がありません。")
                    elif not ad_options:
                        st.error("接触広告列に選択可能な値がありません。")
                    else:
                        select_col1, select_col2 = st.columns(2)
                        with select_col1:
                            selected_month = st.selectbox(
                                "実施月を選択してください",
                                options=month_options,
                                index=0,
                            )
                        with select_col2:
                            selected_ad = st.selectbox(
                                "接触広告を選択してください",
                                options=ad_options,
                                index=0,
                            )

                        cross_df = build_cross_table(df, selected_month, selected_ad)

                        st.markdown("#### クロス集計表")
                        if cross_df.empty:
                            st.warning("選択した実施月・接触広告に該当するデータがありません。")
                        else:
                            st.dataframe(cross_df, use_container_width=True, height=600)

                except Exception as e:
                    st.error(f"データの読み込みまたは表示中にエラーが発生しました: {e}")


# =========================
# フッター
# =========================
st.markdown("---")
st.write("右カラムのピン留めコメント欄に、途中分析で気になった示唆を残し、最後にサマリー化できます。")