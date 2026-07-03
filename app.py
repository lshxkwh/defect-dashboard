# -*- coding: utf-8 -*-
"""
로봇 자동화 조립라인 품질 이상탐지 대시보드
로컬에서 실행: streamlit run app.py

같은 폴더에 아래 2개 파일이 있어야 합니다.
  - assembly_eda_cycle_summary_v2.csv
  - column_manifest.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

st.set_page_config(page_title="조립라인 품질 이상탐지 대시보드", layout="wide")

DATA_PATH = "assembly_eda_cycle_summary_v2.csv"
MANIFEST_PATH = "column_manifest.csv"


# ============================================================
# 파이프라인 (전처리 -> 피처 선택 -> 모델 학습) - 캐시로 1회만 실행
# ============================================================
@st.cache_resource(show_spinner="전처리 및 모델 학습 중 (최초 1회만 실행됩니다)...")
def build_pipeline():
    df = pd.read_csv(DATA_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)

    # ---- 메타/라벨 누수 컬럼 정리 ----
    leak_meta_cols = ["missing_count", "final_part_count"]
    meta_cols = [
        "cycle_run_id", "cycle_order", "start_time", "end_time", "start_time_unix",
        "target_4class", "label_block_id", "time_block_id", "duration_sec", "n_rows",
        "normal_label_source", "normal_needs_validation",
    ] + leak_meta_cols
    feature_cols = [c for c in df.columns if c not in meta_cols]

    y = (df["target_4class"] != "Normal").astype(int)

    # ---- 1단계: JointAngle 제거 ----
    joint_cols = [c for c in feature_cols if "JointAngle" in c]

    # ---- 2단계: 완전 중복 제거 ----
    exclude_redundant = ["I_R02_Gripper_Pot__min", "I_MHS_GreenRocketTray__detect_ratio"]

    # ---- 3단계: 공식 매니페스트 누수 ----
    official_leak = manifest.loc[
        manifest["group"].isin(["outcome_signal_R04", "outcome_signal_R03peak"]), "column"
    ].tolist()

    # ---- 4단계: 추가 발견 누수 ----
    r03_extra_leak = [
        "I_R03_Gripper_Load__large_change_count",
        "I_R03_Gripper_Load__diff_std",
        "I_R03_Gripper_Load__diff_abs_sum",
    ]

    exclude_all = list(set(joint_cols + exclude_redundant + official_leak + r03_extra_leak))
    safe_features = [c for c in feature_cols if c not in exclude_all]

    # ---- 5단계: 다중공선성 제거 ----
    def build_clusters(cols_subset):
        X_corr = df[cols_subset].select_dtypes(include=[np.number])
        zero_var = X_corr.columns[X_corr.std() == 0].tolist()
        X_corr = X_corr.drop(columns=zero_var)
        corr_matrix = X_corr.corr().abs().values
        cols_arr = X_corr.columns.to_numpy()
        upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        high_corr_idx = np.argwhere((corr_matrix >= 0.9) & upper)

        parent = {c: c for c in cols_arr}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            parent[find(a)] = find(b)

        for i, j in high_corr_idx:
            union(cols_arr[i], cols_arr[j])

        groups = defaultdict(list)
        for c in cols_arr:
            groups[find(c)].append(c)
        return [g for g in groups.values() if len(g) > 1], zero_var

    def pick_representative(cluster_cols):
        corrs = {c: abs(np.corrcoef(df[c], y)[0, 1]) for c in cluster_cols}
        return max(corrs, key=corrs.get)

    clusters, zero_var_early = build_clusters(safe_features)
    drop_cols = []
    for cl in clusters:
        rep = pick_representative(cl)
        drop_cols.extend([c for c in cl if c != rep])

    early_features = [c for c in safe_features if c not in drop_cols and c not in zero_var_early]

    # ---- 사후검사 트랙 (관절각/완전중복만 제외, 다중공선성 별도 적용) ----
    posthoc_exclude = list(set(joint_cols + exclude_redundant))
    posthoc_candidates = [c for c in feature_cols if c not in posthoc_exclude]
    clusters_p, zero_var_post = build_clusters(posthoc_candidates)
    drop_cols_p = []
    for cl in clusters_p:
        rep = pick_representative(cl)
        drop_cols_p.extend([c for c in cl if c != rep])
    posthoc_features = [c for c in posthoc_candidates if c not in drop_cols_p and c not in zero_var_post]

    # ---- fold_primary (참고용, 성능 리포트에 사용) ----
    fold_map = {1: 0, 6: 0, 2: 1, 7: 1, 3: 2, 8: 2, 4: 3, 9: 3, 5: 4, 10: 4}
    df["fold_primary"] = df["time_block_id"].map(fold_map)

    # ---- 예외 사이클 플래그 (cycle 10: 이미지 확인 전까지 재검증 필요) ----
    df["needs_image_review"] = (df["cycle_order"] == 10).astype(int)

    # ---- 배포용 모델 학습 (전체 데이터) ----
    early_model = RandomForestClassifier(
        n_estimators=300, max_depth=6, random_state=42, class_weight="balanced"
    )
    early_model.fit(df[early_features], y)

    # 조기탐지 단계에서 "어떤 부품이 의심되는지" 참고용 4클래스 모델
    # 주의: NoNose(부품 1개 누락) 유형은 조기탐지 신호로 구분 불가능함이 검증에서 확인됨 (신뢰도 낮음)
    early_type_model = RandomForestClassifier(
        n_estimators=300, max_depth=6, random_state=42, class_weight="balanced"
    )
    early_type_model.fit(df[early_features], df["target_4class"])

    posthoc_model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, verbose=-1
    )
    posthoc_model.fit(df[posthoc_features], y)

    return {
        "df": df,
        "early_model": early_model,
        "early_type_model": early_type_model,
        "early_features": early_features,
        "posthoc_model": posthoc_model,
        "posthoc_features": posthoc_features,
    }


def classify_cycle(row, pipeline, early_threshold=0.5, posthoc_threshold=0.5):
    """한 사이클(행)을 조기탐지 + 사후검사 두 모델로 판정"""
    early_X = pd.DataFrame([row[pipeline["early_features"]].astype(float)])
    early_proba = pipeline["early_model"].predict_proba(early_X)[0, 1]

    result = {"early_proba": early_proba, "early_alert": early_proba >= early_threshold}

    # 조기 경보 시, 어떤 부품 유형이 의심되는지 참고 정보 추가
    if result["early_alert"]:
        type_model = pipeline["early_type_model"]
        type_proba = type_model.predict_proba(early_X)[0]
        type_classes = type_model.classes_
        top_idx = np.argsort(type_proba)[::-1]
        result["suspected_type"] = type_classes[top_idx[0]]
        result["suspected_type_proba"] = type_proba[top_idx[0]]
        result["suspected_type_low_confidence"] = (type_classes[top_idx[0]] == "NoNose")

    # 사후검사는 R04 등 후단 신호가 있어야 하므로, 해당 컬럼이 다 채워져 있을 때만 계산
    posthoc_cols = pipeline["posthoc_features"]
    if row[posthoc_cols].isnull().any():
        result["posthoc_available"] = False
    else:
        posthoc_X = pd.DataFrame([row[posthoc_cols].astype(float)])
        posthoc_proba = pipeline["posthoc_model"].predict_proba(posthoc_X)[0, 1]
        result["posthoc_available"] = True
        result["posthoc_proba"] = posthoc_proba
        result["posthoc_verdict"] = "불량" if posthoc_proba >= posthoc_threshold else "정상"

    return result


def judge_risk_tier(actual_label, early_alert, posthoc_verdict, posthoc_available):
    """실제 라벨과 예측을 비교해서 위험도 등급을 매긴다 (현업 관점 - 놓친 불량이 제일 위험)"""
    actual_is_defect = (actual_label is not None) and (actual_label != "Normal")
    if actual_label is None:
        return "⚪ 실제라벨 없음", "N/A"

    if actual_is_defect:
        if posthoc_available and posthoc_verdict == "정상":
            return "🔴 치명적 놓침 (불량 유출 위험)", "danger"
        if not early_alert:
            return "🟠 조기탐지 사각지대 (최종판정에서 확인됨)", "warning"
        return "⚪ 정상 탐지", "ok"
    else:
        if early_alert or (posthoc_available and posthoc_verdict == "불량"):
            return "🟡 오탐(False Alarm)", "info"
        return "⚪ 정상 탐지", "ok"


# ============================================================
# UI
# ============================================================
st.title("조립라인 품질 이상탐지 대시보드")
st.caption("로봇 자동화 조립라인 센서 데이터 기반 · 조기탐지(R01~R03) + 최종판정(R04까지) 이중 게이트 구조")

with st.expander("⚠️ 사용 전 반드시 읽어주세요 (모델 한계 안내)", expanded=True):
    st.warning(
        """
- 이 모델은 **의도적으로 부품을 제거한 실험 데이터**로 학습되었습니다. 실제 현장의 자연 발생 불량에 대한 성능은 별도 검증이 필요합니다.
- **조기탐지 결과는 "확정 판정"이 아니라 "경보"**입니다. 최종 출하 여부는 반드시 사후검사(최종판정) 결과를 따라야 합니다.
- 조기탐지 모델은 **"부품 1개만 누락된 유형"을 구조적으로 탐지하지 못하는 것으로 확인**되었습니다 (관련 신호 부재, 여러 보완 방법 시도 후에도 해결 불가 확인됨). 이 유형은 사후검사 단계에서만 확정 가능합니다.
- 최종판정 모델의 높은 정확도는 특정 단일 변수(그립 동작 횟수)에 크게 의존합니다. 해당 변수 계산 로직에 오류가 있을 경우 함께 영향을 받을 수 있습니다.
"""
    )

pipeline = build_pipeline()
df = pipeline["df"]

with st.sidebar:
    st.header("⚙️ 판정 기준 설정")
    st.caption("기본값 50%는 데이터 분석 관점의 중립 기준입니다. 현장 리스크 성향에 맞게 조정하세요.")
    early_threshold = st.slider(
        "조기탐지 경보 기준", 0.1, 0.9, 0.5, 0.05,
        help="낮출수록 더 많은 경우를 '의심스럽다'고 잡아냅니다(경보는 늘지만 놓치는 건 줄어듦)."
    )
    posthoc_threshold = st.slider(
        "최종판정 불량 기준", 0.1, 0.9, 0.5, 0.05,
        help="낮출수록 더 엄격하게 불량으로 판정합니다(출하 기준을 더 까다롭게 하고 싶을 때 낮추세요)."
    )
    st.divider()
    st.caption("현재 설정은 이 브라우저 세션에서만 유지됩니다.")

tab0, tab1, tab2, tab3 = st.tabs(["🔴 실시간 시뮬레이션", "📋 기존 사이클 조회 (데모)", "📤 새 데이터 업로드", "📊 모델 성능 리포트"])

# ------------------------------------------------------------
# 탭 0: 실시간 시뮬레이션 (사이클 순서대로 하나씩 흘려보내기)
# ------------------------------------------------------------
with tab0:
    st.subheader("실시간 라인 시뮬레이션")
    st.caption("실제 공장 설비와 연동된 것이 아니라, 기존 데이터를 사이클 순서대로 재생하여 실시간처럼 보여주는 시뮬레이션입니다.")

    sorted_df = df.sort_values("cycle_order").reset_index(drop=True)

    if "sim_index" not in st.session_state:
        st.session_state.sim_index = 0
    if "sim_running" not in st.session_state:
        st.session_state.sim_running = False
    if "sim_log" not in st.session_state:
        st.session_state.sim_log = []
    if "sim_stats" not in st.session_state:
        st.session_state.sim_stats = {"total": 0, "correct": 0, "critical_miss": 0, "blind_spot": 0, "false_alarm": 0}
    if "sim_seed" not in st.session_state:
        st.session_state.sim_seed = 0
    if "sim_order" not in st.session_state:
        st.session_state.sim_order = "원본 순서 (시간순)"

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 2])
    with ctrl1:
        if st.button("▶ 시작", use_container_width=True):
            st.session_state.sim_running = True
    with ctrl2:
        if st.button("⏸ 정지", use_container_width=True):
            st.session_state.sim_running = False
    with ctrl3:
        if st.button("⏮ 초기화", use_container_width=True):
            st.session_state.sim_index = 0
            st.session_state.sim_running = False
            st.session_state.sim_log = []
            st.session_state.sim_stats = {"total": 0, "correct": 0, "critical_miss": 0, "blind_spot": 0, "false_alarm": 0}
            st.session_state.sim_seed += 1  # 초기화할 때마다 새로 섞이도록
    with ctrl4:
        speed_level = st.slider("재생 속도 (높을수록 빠름)", 1, 10, 4)
        speed = 3.0 / speed_level  # level 1 -> 3.0초 대기(느림), level 10 -> 0.3초 대기(빠름)

    order_col1, order_col2 = st.columns([2, 3])
    with order_col1:
        st.session_state.sim_order = st.radio(
            "재생 순서",
            ["원본 순서 (시간순)", "무작위로 섞기"],
            horizontal=True,
            index=0 if st.session_state.sim_order == "원본 순서 (시간순)" else 1,
        )
    with order_col2:
        if st.session_state.sim_order == "무작위로 섞기":
            st.caption("정상/불량이 실제 현장처럼 뒤섞인 순서로 재생됩니다 (초기화 시 다시 섞임).")
        else:
            st.caption("실험 데이터 수집 순서 그대로 재생합니다 (불량 유형별로 뭉쳐 나올 수 있음).")

    if st.session_state.sim_order == "무작위로 섞기":
        play_df = sorted_df.sample(frac=1, random_state=st.session_state.sim_seed).reset_index(drop=True)
    else:
        play_df = sorted_df

    status_placeholder = st.empty()
    metric_placeholder = st.empty()
    log_placeholder = st.empty()

    def render_current_state():
        stats = st.session_state.sim_stats
        with status_placeholder.container():
            if not st.session_state.sim_log:
                st.info("▶ 시작 버튼을 눌러 시뮬레이션을 시작하세요.")
            else:
                last = st.session_state.sim_log[0]
                if last.get("위험도등급") == "danger":
                    st.error(f"🔴 사이클 {last['cycle_order']} — 치명적 놓침! 불량인데 정상으로 최종판정됨. 즉시 확인 필요")
                elif last.get("최종판정") == "불량":
                    st.error(f"🚨 사이클 {last['cycle_order']} — 최종판정: 불량")
                elif last.get("위험도등급") == "warning":
                    st.warning(f"🟠 사이클 {last['cycle_order']} — 조기탐지는 놓쳤지만 최종판정에서 확인됨 (사각지대 발생)")
                elif last.get("조기경보") == "이상 의심":
                    msg = f"⚠️ 사이클 {last['cycle_order']} — 조기 경보 발생"
                    if last.get("의심유형") and last["의심유형"] != "-":
                        msg += f" · 의심 유형: **{last['의심유형']}**"
                    st.warning(msg)
                    if last.get("의심유형_저신뢰"):
                        st.caption("⚠️ '부품 1개 누락' 유형은 조기탐지 신호로 구분이 검증되지 않아, 이 추정의 신뢰도가 낮습니다. 최종판정을 반드시 확인하세요.")
                else:
                    st.success(f"✅ 사이클 {last['cycle_order']} — 정상 가동 중")

        with metric_placeholder.container():
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("누적 처리", stats["total"])
            c2.metric("정상 탐지", stats["correct"])
            c3.metric("🔴 치명적 놓침", stats["critical_miss"], help="불량인데 최종판정까지 정상으로 통과된 건수 — 반드시 0이어야 합니다.")
            c4.metric("🟠 조기탐지 사각지대", stats["blind_spot"], help="조기탐지는 놓쳤지만 최종판정에서 잡아낸 건수")
            c5.metric("🟡 오탐(False Alarm)", stats["false_alarm"], help="정상인데 경보/불량으로 잘못 판정된 건수")
            if stats["critical_miss"] > 0:
                st.error(f"⚠️ 이번 세션에서 {stats['critical_miss']}건의 불량이 시스템을 그대로 통과했습니다. 즉시 원인 점검이 필요합니다.")

        with log_placeholder.container():
            st.write("**최근 판정 로그** (최신순 · 위험도가 색으로 표시됩니다 · 행을 클릭하면 실제 라벨 상세를 볼 수 있습니다)")
            if st.session_state.sim_log:
                log_df = pd.DataFrame(st.session_state.sim_log[:15])
                display_cols = ["cycle_order", "위험도", "조기경보", "의심유형", "최종판정"]
                event = st.dataframe(
                    log_df[display_cols],
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="sim_log_table",
                )
                if event.selection.rows:
                    picked = log_df.iloc[event.selection.rows[0]]
                    st.info(
                        f"🔎 사이클 {picked['cycle_order']} 실제 라벨: **{picked['실제라벨']}** "
                        f"(예측 — 조기탐지: {picked['조기경보']}"
                        + (f" / 의심유형: {picked['의심유형']}" if picked["의심유형"] != "-" else "")
                        + f" · 최종판정: {picked['최종판정']})"
                    )

    render_current_state()

    if st.session_state.sim_running and st.session_state.sim_index < len(play_df):
        row = play_df.iloc[st.session_state.sim_index]
        result = classify_cycle(row, pipeline, early_threshold, posthoc_threshold)
        risk_label, risk_tier = judge_risk_tier(
            row["target_4class"], result["early_alert"],
            result.get("posthoc_verdict"), result.get("posthoc_available", False)
        )

        entry = {
            "cycle_order": row["cycle_order"],
            "실제라벨": row["target_4class"],
            "위험도": risk_label,
            "위험도등급": risk_tier,
            "조기탐지_확률": result["early_proba"],
            "조기경보": "이상 의심" if result["early_alert"] else "정상으로 보임",
            "의심유형": result.get("suspected_type", "-"),
            "의심유형_확률": result.get("suspected_type_proba", np.nan),
            "의심유형_저신뢰": result.get("suspected_type_low_confidence", False),
            "최종판정_확률": result.get("posthoc_proba", np.nan),
            "최종판정": result.get("posthoc_verdict", "N/A"),
        }
        st.session_state.sim_log.insert(0, entry)
        st.session_state.sim_stats["total"] += 1
        if risk_tier == "danger":
            st.session_state.sim_stats["critical_miss"] += 1
        elif risk_tier == "warning":
            st.session_state.sim_stats["blind_spot"] += 1
        elif risk_tier == "info":
            st.session_state.sim_stats["false_alarm"] += 1
        else:
            st.session_state.sim_stats["correct"] += 1

        st.session_state.sim_index += 1
        time.sleep(speed)
        st.rerun()
    elif st.session_state.sim_running and st.session_state.sim_index >= len(play_df):
        st.session_state.sim_running = False
        st.success("전체 사이클 재생이 끝났습니다. 초기화 후 다시 시작할 수 있습니다.")


# ------------------------------------------------------------
# 탭 1: 데모 - 기존 데이터셋에서 사이클 선택
# ------------------------------------------------------------
with tab1:
    st.subheader("데이터셋에 있는 사이클로 판정 체험해보기")
    cycle_options = df["cycle_order"].tolist()
    selected_cycle = st.selectbox("사이클 선택", cycle_options)

    row = df[df["cycle_order"] == selected_cycle].iloc[0]
    actual_label = row["target_4class"]

    result = classify_cycle(row, pipeline, early_threshold, posthoc_threshold)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("실제 라벨 (참고용)", actual_label)
    with col2:
        alert_text = "🚨 이상 의심" if result["early_alert"] else "✅ 정상으로 보임"
        st.metric("조기탐지 결과 (R03 시점)", alert_text)
    with col3:
        if result["posthoc_available"]:
            verdict_icon = "🚨 불량" if result["posthoc_verdict"] == "불량" else "✅ 정상"
            st.metric("최종판정 결과 (R04 시점)", verdict_icon)
        else:
            st.metric("최종판정 결과", "데이터 없음")

    if result["early_alert"] != (result.get("posthoc_verdict") == "불량"):
        st.info("ℹ️ 조기탐지와 최종판정 결과가 다릅니다 — 최종판정을 기준으로 출하 여부를 결정하세요.")

    if result["early_alert"] and result.get("suspected_type"):
        st.write(f"🔍 조기탐지 단계 의심 부품 유형: **{result['suspected_type']}**")
        if result.get("suspected_type_low_confidence"):
            st.caption("⚠️ '부품 1개 누락(NoNose)' 유형은 조기탐지 신호로 구분이 검증되지 않아 신뢰도가 낮습니다.")

    if row["needs_image_review"] == 1:
        st.error("🖼️ 이 사이클은 이미지 데이터 확보 후 재검증이 필요한 것으로 표시되어 있습니다 (cycle 10).")

    risk_label, risk_tier = judge_risk_tier(
        actual_label, result["early_alert"], result.get("posthoc_verdict"), result.get("posthoc_available", False)
    )
    if risk_tier == "danger":
        st.error(f"위험도 판정: {risk_label}")
    elif risk_tier == "warning":
        st.warning(f"위험도 판정: {risk_label}")
    elif risk_tier == "info":
        st.info(f"위험도 판정: {risk_label}")
    else:
        st.success(f"위험도 판정: {risk_label}")

# ------------------------------------------------------------
# 탭 2: 새 CSV 업로드해서 배치 판정
# ------------------------------------------------------------
with tab2:
    st.subheader("새 사이클 데이터 업로드")
    st.caption("`assembly_eda_cycle_summary_v2.csv`와 동일한 컬럼 구조를 가진 CSV를 업로드하세요.")
    uploaded = st.file_uploader("CSV 업로드", type="csv")

    if uploaded is not None:
        new_df = pd.read_csv(uploaded)
        missing_early = [c for c in pipeline["early_features"] if c not in new_df.columns]
        if missing_early:
            st.error(f"필요한 컬럼이 없습니다: {missing_early[:5]} 등 {len(missing_early)}개")
        else:
            results = []
            has_actual_label = "target_4class" in new_df.columns
            for _, r in new_df.iterrows():
                res = classify_cycle(r, pipeline, early_threshold, posthoc_threshold)
                row_result = {
                    "cycle_order": r.get("cycle_order", "-"),
                    "조기탐지_경보": "이상 의심" if res["early_alert"] else "정상으로 보임",
                    "의심유형": res.get("suspected_type", "-"),
                    "최종판정_결과": res.get("posthoc_verdict", "N/A"),
                }
                if has_actual_label:
                    actual = r["target_4class"]
                    risk_label, risk_tier = judge_risk_tier(
                        actual, res["early_alert"], res.get("posthoc_verdict"), res.get("posthoc_available", False)
                    )
                    row_result["실제라벨"] = actual
                    row_result["위험도"] = risk_label
                results.append(row_result)
            result_df = pd.DataFrame(results)

            if has_actual_label:
                n_critical = (result_df["위험도"].str.contains("치명적")).sum()
                if n_critical > 0:
                    st.error(f"🔴 업로드한 데이터 중 {n_critical}건이 '치명적 놓침'입니다 — 불량인데 정상으로 최종판정되었습니다.")
                else:
                    st.success("업로드한 데이터에서 '치명적 놓침'은 발견되지 않았습니다.")

            st.dataframe(result_df, use_container_width=True)
            st.caption("⚠️ '의심유형'이 NoNose(부품 1개 누락)로 나온 경우, 조기탐지 신호로는 검증되지 않은 추정치이므로 최종판정을 반드시 확인하세요.")
            st.download_button(
                "결과 CSV 다운로드",
                result_df.to_csv(index=False).encode("utf-8-sig"),
                "판정결과.csv",
                "text/csv",
            )

# ------------------------------------------------------------
# 탭 3: 모델 성능 리포트 (고정된 검증 결과 표시)
# ------------------------------------------------------------
with tab3:
    st.subheader("검증된 모델 성능 (fold_primary 5-fold 교차검증 기준)")

    perf_data = pd.DataFrame({
        "모델": ["조기탐지 (RandomForest)", "최종판정 (LightGBM)"],
        "사용 피처 수": [len(pipeline["early_features"]), len(pipeline["posthoc_features"])],
        "평균 Recall": [0.919, 1.000],
        "평균 Precision": [0.986, 1.000],
        "비고": ["부품 1개 누락 유형 탐지 불가 (구조적 한계)", "전체 유형 완벽 탐지"],
    })
    st.dataframe(perf_data, use_container_width=True)

    st.markdown("""
**한계 요약**
- 조기탐지: 특정 시간 구간(fold 4)에서 recall 0.556까지 하락 확인 — 원인은 특정 유형(NoNose)의 훈련/시험 세션 간 신호 차이
- 보완 시도(이상탐지 감시, 확신도 필터링, class_weight, 전체 396개 신호 재탐색) 4종 모두 실패 확인 → 이미지 데이터 도입이 다음 단계로 필요
""")

st.divider()
st.caption("본 대시보드는 실험 데이터 기반 프로토타입입니다. 실제 배포 전 현장 데이터 재검증이 필요합니다.")
