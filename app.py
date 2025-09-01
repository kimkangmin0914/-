
import time
import io
import math
import random
import pandas as pd
import numpy as np
import streamlit as st
from ortools.sat.python import cp_model

st.set_page_config(page_title="교회 매칭 프로그램 (팀 번호 + 이름만)", layout="wide")

# ------------------------------
# 가나다순 정렬 키
# ------------------------------
BASE, CHOS, JUNG = 0xAC00, 588, 28
def hangul_key(s: str):
    ks = []
    for ch in str(s):
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7A3:
            sidx = o - BASE
            cho = sidx // CHOS
            jung = (sidx % CHOS) // JUNG
            jong = sidx % JUNG
            ks.append((0, cho, jung, jong))
        else:
            ks.append((1, o))
    return tuple(ks)

# ------------------------------
# 유틸: 데이터 전처리
# ------------------------------
AGE_BANDS = ["10대","20대","30대","40대","50대","60대+"]

def age_to_band(age: int) -> str:
    try:
        a = int(age)
    except Exception:
        return None
    if a < 20:
        return "10대"
    if a < 30:
        return "20대"
    if a < 40:
        return "30대"
    if a < 50:
        return "40대"
    if a < 60:
        return "50대"
    return "60대+"

def normalize_gender(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s in ["남","남자","M","m","male","Male"]:
        return "남"
    if s in ["여","여자","F","f","female","Female"]:
        return "여"
    return None

# ------------------------------
# 그룹 크기 결정: 7명 기본, 6~8명 허용, 6/8인 조는 최대 4개
# ------------------------------
def choose_group_sizes(N: int, max_offsize: int = 4):
    best = None
    target_T = int(round(N/7))
    for x6 in range(0, max_offsize+1):
        for x8 in range(0, max_offsize - x6 + 1):
            rem = N - (6*x6 + 8*x8)
            if rem < 0:
                continue
            if rem % 7 != 0:
                continue
            x7 = rem // 7
            T = x6 + x7 + x8
            off = x6 + x8
            score = (abs(T - target_T), off, abs(x8 - x6))
            cand = (score, x6, x7, x8)
            if best is None or cand < best:
                best = cand
    if best is None:
        return None, "해결 실패: 6/7/8인 조의 조합으로 총원 {}명을 구성할 수 없습니다.".format(N)
    else:
        (_, x6, x7, x8) = best
        sizes = [6]*x6 + [7]*x7 + [8]*x8
        return sizes, None

def allowed_male_bounds(size):
    # 7인: 남 3~4 / 6인: 남 2~4 / 8인: 남 3~5
    if size == 7:
        return 3,4
    if size == 6:
        return 2,4
    if size == 8:
        return 3,5
    lo = int(math.floor(0.4*size))
    hi = int(math.ceil(0.6*size))
    return lo, hi

# ------------------------------
# OR-Tools CP-SAT 모델
# ------------------------------
def solve_assignment(df, seed=0, time_limit=10):
    people = df.to_dict('records')
    N = len(people)
    sizes, warn = choose_group_sizes(N, max_offsize=4)
    if sizes is None:
        return None, None, "조 크기 계산 실패", None
    G = len(sizes)

    males = [i for i,p in enumerate(people) if p['성별'] == '남']

    churches = sorted(df['교회 이름'].fillna("미상").astype(str).unique().tolist())
    church_members = {c: [i for i,p in enumerate(people) if str(p['교회 이름']) == c] for c in churches}
    bands = AGE_BANDS
    band_members = {b: [i for i,p in enumerate(people) if p['나이대'] == b] for b in bands}

    # 사전 타당성: 교회/나이대 인원수가 2*G 초과면 불가능
    overload = []
    for c, members in church_members.items():
        if len(members) > 2*G:
            overload.append((c, len(members), 2*G))
    if overload:
        msg = "불가능: 일부 교회 인원이 너무 많아(최대 2명/팀) 배치가 불가합니다.\n" + \
              "\n".join([f" - {c}: {cnt}명 > 허용 {cap}명" for c,cnt,cap in overload])
        return None, None, msg, None
    for b, members in band_members.items():
        if len(members) > 2*G:
            msg = "불가능: 일부 나이대 인원이 너무 많아(최대 2명/팀) 배치가 불가합니다.\n" + \
                  "\n".join([f" - {b}: {len(band_members[b])}명 > 허용 {2*G}명"])
            return None, None, msg, None

    model = cp_model.CpModel()

    x = {}
    for i in range(N):
        for g in range(G):
            x[(i,g)] = model.NewBoolVar(f"x_{i}_{g}")

    # 각 사람은 정확히 1개 팀
    for i in range(N):
        model.Add(sum(x[(i,g)] for g in range(G)) == 1)

    # 팀 크기 고정
    for g in range(G):
        model.Add(sum(x[(i,g)] for i in range(N)) == sizes[g])

    # 성비 제약(유연 슬랙 허용)
    sL = []
    sU = []
    for g in range(G):
        mc = model.NewIntVar(0, sizes[g], f"male_{g}")
        model.Add(mc == sum(x[(i,g)] for i in males))
        lo, hi = allowed_male_bounds(sizes[g])
        sl = model.NewIntVar(0, sizes[g], f"sL_{g}")
        su = model.NewIntVar(0, sizes[g], f"sU_{g}")
        model.Add(mc >= lo - sl)
        model.Add(mc <= hi + su)
        sL.append(sl)
        sU.append(su)

    # 교회/나이대: 팀당 최대 2명(하드) + 2명일 때 패널티
    church_pair_flags = []
    for g in range(G):
        for c in churches:
            members = church_members[c]
            if not members:
                continue
            cnt = model.NewIntVar(0, min(2, len(members)), f"church_{c}_{g}")
            model.Add(cnt == sum(x[(i,g)] for i in members))
            model.Add(cnt <= 2)
            is_pair = model.NewBoolVar(f"is_pair_{c}_{g}")
            model.Add(cnt == 2).OnlyEnforceIf(is_pair)
            model.Add(cnt != 2).OnlyEnforceIf(is_pair.Not())
            church_pair_flags.append(is_pair)

    age_pair_flags = []
    for g in range(G):
        for b in bands:
            members = band_members[b]
            if not members:
                continue
            cnt = model.NewIntVar(0, min(2, len(members)), f"band_{b}_{g}")
            model.Add(cnt == sum(x[(i,g)] for i in members))
            model.Add(cnt <= 2)
            is_pair = model.NewBoolVar(f"is_band_pair_{b}_{g}")
            model.Add(cnt == 2).OnlyEnforceIf(is_pair)
            model.Add(cnt != 2).OnlyEnforceIf(is_pair.Not())
            age_pair_flags.append(is_pair)

    # 목적함수: 성비 슬랙 최소 + 동일 교회/나이대 '2명' 최소 + 약한 무작위성
    rand = random.Random(int(time.time()) % (10**6))
    noise_terms = []
    for i in range(N):
        for g in range(G):
            w = rand.randint(0, 3)
            if w > 0:
                noise_terms.append(w * x[(i,g)])

    model.Minimize(
        1000 * sum(sL) + 1000 * sum(sU) +
        3 * sum(church_pair_flags) +
        2 * sum(age_pair_flags) +
        1 * sum(noise_terms)
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 8

    res = solver.Solve(model)
    if res not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, None, "해결 실패: 제약을 만족하는 해를 찾지 못했습니다. (시간 제한/분포 문제 가능)", None

    groups = []
    for g in range(G):
        members = [i for i in range(N) if solver.Value(x[(i,g)]) == 1]
        groups.append(members)

    # 성비 완화 경고
    total_slack = int(sum(solver.Value(v) for v in sL) + sum(solver.Value(v) for v in sU))
    warn_list = []
    if total_slack > 0:
        warn_list.append(f"주의: 성비 제약을 {total_slack}명만큼 완화하여 해를 구성했습니다.")

    return groups, warn_list, None, sizes

# ------------------------------
# UI
# ------------------------------
st.title("교회 매칭 프로그램 (팀 번호 + 이름만)")

with st.sidebar:
    st.header("설정")
    uploaded = st.file_uploader("엑셀 업로드 (.xlsx)", type=["xlsx"])
    time_limit = st.slider("해결 시간 제한(초)", min_value=5, max_value=30, value=10, step=1)
    run_btn = st.button("🎲 추첨 시작")

st.markdown("""
**입력 컬럼(필수)**: `이름`, `성별(남/여)`, `교회 이름`, `나이`  
- `나이대`는 자동 생성됩니다. (10/20/30/40/50/60+)
- 결과 표시/다운로드에는 **팀 번호와 이름만** 포함됩니다(이름은 팀별 가나다순).
""")

df = None
if uploaded is not None:
    try:
        df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"엑셀 읽기 오류: {e}")

if df is not None:
    required = ["이름","성별","교회 이름","나이"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"필수 컬럼 누락: {missing}")
        st.stop()

    df = df.copy()
    df["성별"] = df["성별"].apply(normalize_gender)
    if df["성별"].isna().any():
        st.error("성별 값 표준화 실패 행이 있습니다. ('남'/'여'만 허용)")
        st.dataframe(df[df["성별"].isna()])
        st.stop()
    df["나이대"] = df["나이"].apply(age_to_band)
    if df["나이대"].isna().any():
        st.error("나이 → 나이대 변환 실패 행이 있습니다. (정수 나이 필요)")
        st.dataframe(df[df["나이대"].isna()])
        st.stop()

    st.subheader("입력 데이터 미리보기")
    st.dataframe(df.head(10))

    N = len(df)
    sizes, warn = choose_group_sizes(N, max_offsize=4)
    if sizes is None:
        st.error(warn)
        st.stop()
    st.info(f"총 {N}명 → 후보 그룹 크기: " + ", ".join(map(str, sorted(sizes))))
    if warn:
        st.warning(warn)

    if run_btn:
        ph = st.empty()
        for pct in range(0, 101, 7):
            ph.progress(pct, text="무작위 배치 탐색 중...")
            time.sleep(0.05)

        groups, warn_list, err, sizes = solve_assignment(df, time_limit=time_limit)

        if err:
            st.error(err)
            st.stop()
        if warn_list:
            for w in warn_list:
                st.warning(w)

        st.success("배치 완료! 아래에서 팀을 확인하세요.")
        people = df.to_dict('records')

        # --- 화면 표시: 팀 번호 + 이름(가나다순, 세로 목록) ---
        st.subheader("팀 구성 (팀 번호 + 이름, 가나다순)")
        cols = st.columns(3)
        for idx, (g, members) in enumerate(enumerate(groups)):
            col = cols[idx % 3]
            team_members = [people[i] for i in members]
            with col:
                st.markdown(f"### 팀 {g+1}")
                names = [p['이름'] for p in team_members]
                names_sorted = sorted(names, key=hangul_key)
                st.markdown("\\n".join(f"- {name}" for name in names_sorted))

        # --- 다운로드: 팀 번호 + 이름 (팀별 가나다순) ---
        rows = []
        for g, members in enumerate(groups):
            team_names = [people[i]['이름'] for i in members]
            for name in sorted(team_names, key=hangul_key):
                rows.append({"팀": g+1, "이름": name})
        out_df = pd.DataFrame(rows)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            out_df.to_excel(writer, index=False, sheet_name="TeamsOnly")
        st.download_button("결과 엑셀 다운로드(팀+이름, 가나다순)", data=buf.getvalue(),
                           file_name="teams_names_only.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    st.info("왼쪽 사이드바에서 엑셀 파일을 업로드하시고 '🎲 추첨 시작'을 눌러주세요.")
