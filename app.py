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

    # ---- 배포용 모델 학습 (전체 데이터) ----
    early_model = RandomForestClassifier(
        n_estimators=300, max_depth=6, random_state=42, class_weight="balanced"
    )
    early_model.fit(df[early_features], y)

    posthoc_model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, verbose=-1
    )
    posthoc_model.fit(df[posthoc_features], y)

    return {
        "df": df,
        "early_model": early_model,
        "early_features": early_features,
        "posthoc_model": posthoc_model,
        "posthoc_features": posthoc_features,
    }


def classify_cycle(row, pipeline):
    """한 사이클(행)을 조기탐지 + 사후검사 두 모델로 판정"""
    early_X = pd.DataFrame([row[pipeline["early_features"]].astype(float)])
    early_proba = pipeline["early_model"].predict_proba(early_X)[0, 1]

    result = {"early_proba": early_proba, "early_alert": early_proba >= 0.5}

    # 사후검사는 R04 등 후단 신호가 있어야 하므로, 해당 컬럼이 다 채워져 있을 때만 계산
    posthoc_cols = pipeline["posthoc_features"]
    if row[posthoc_cols].isnull().any():
        result["posthoc_available"] = False
    else:
        posthoc_X = pd.DataFrame([row[posthoc_cols].astype(float)])
        posthoc_proba = pipeline["posthoc_model"].predict_proba(posthoc_X)[0, 1]
        result["posthoc_available"] = True
        result["posthoc_proba"] = posthoc_proba
        result["posthoc_verdict"] = "불량" if posthoc_proba >= 0.5 else "정상"

    return result


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

tab1, tab2, tab3 = st.tabs(["📋 기존 사이클 조회 (데모)", "📤 새 데이터 업로드", "📊 모델 성능 리포트"])

# ------------------------------------------------------------
# 탭 1: 데모 - 기존 데이터셋에서 사이클 선택
# ------------------------------------------------------------
with tab1:
    st.subheader("데이터셋에 있는 사이클로 판정 체험해보기")
    cycle_options = df["cycle_order"].tolist()
    selected_cycle = st.selectbox("사이클 선택", cycle_options)

    row = df[df["cycle_order"] == selected_cycle].iloc[0]
    actual_label = row["target_4class"]

    result = classify_cycle(row, pipeline)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("실제 라벨 (참고용)", actual_label)
    with col2:
        alert_text = "🚨 이상 의심" if result["early_alert"] else "✅ 정상으로 보임"
        st.metric("조기탐지 결과 (R03 시점)", alert_text, f"확률 {result['early_proba']:.1%}")
    with col3:
        if result["posthoc_available"]:
            verdict_icon = "🚨 불량" if result["posthoc_verdict"] == "불량" else "✅ 정상"
            st.metric("최종판정 결과 (R04 시점)", verdict_icon, f"확률 {result['posthoc_proba']:.1%}")
        else:
            st.metric("최종판정 결과", "데이터 없음")

    if result["early_alert"] != (result.get("posthoc_verdict") == "불량"):
        st.info("ℹ️ 조기탐지와 최종판정 결과가 다릅니다 — 최종판정을 기준으로 출하 여부를 결정하세요.")

    if row["needs_image_review"] == 1:
        st.error("🖼️ 이 사이클은 이미지 데이터 확보 후 재검증이 필요한 것으로 표시되어 있습니다 (cycle 10).")

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
            for _, r in new_df.iterrows():
                res = classify_cycle(r, pipeline)
                results.append({
                    "cycle_order": r.get("cycle_order", "-"),
                    "조기탐지_확률": round(res["early_proba"], 4),
                    "조기탐지_경보": "이상 의심" if res["early_alert"] else "정상으로 보임",
                    "최종판정_확률": round(res.get("posthoc_proba", np.nan), 4) if res["posthoc_available"] else "N/A",
                    "최종판정_결과": res.get("posthoc_verdict", "N/A"),
                })
            result_df = pd.DataFrame(results)
            st.dataframe(result_df, use_container_width=True)
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
