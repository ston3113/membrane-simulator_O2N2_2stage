import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import streamlit as st
import pandas as pd

# 실행시 >> streamlit run membrane_app_2stage.py

# ==================================================================
# 1. 기본 상수 및 단위 변환 설정
# ==================================================================
STP_MOLAR_VOLUME = 22414.0  # cm³/mol
BAR_TO_ATM = 0.986923
M3H_TO_CM3S = 1_000_000.0 / 3600.0
CM2_TO_M2 = 0.0001
M2_TO_CM2 = 10000.0
GPU_TO_STD_UNITS = 1e-6 * 76.0 

# [기본값 설정]
PROCESS_PARAMS_VOL = {
    "p_u_default": 12.00,  # 12 bar (추진력 확보)
    "p_p_default": 1.00,   # 대기압
}

# [선택도 10 설정] N2=50, O2=500
DEFAULT_L_GPU = np.array([50.0, 500.0]) 

RAW_FEED_FLUX_M3H = 100.00  
RAW_FEED_COMP = np.array([0.79, 0.21]) 

# [S2 면적 축소] 2.5 m2
AREA_LIST_M2 = [80.0, 2.5] 

# ==================================================================
# 2. MembraneStage 클래스 (물리적 모델)
# ==================================================================
class MembraneStage:
    def __init__(self, name):
        self.name = name
        self.area = 0.0
        self.stage_cut = 0.0
        self.feed_flux = 0.0
        self.feed_comp = None
        self.permeate_flux = 0.0
        self.permeate_comp = None
        self.retentate_flux = 0.0
        self.retentate_comp = None

    def _calc_yi_system(self, x, params):
        n_comp = len(x)
        L, p_u, p_p = params["L"], params["p_u"], params["p_p"]
        x_safe = np.clip(x, 1e-12, 1.0)

        def equations(yi):
            yi_safe = np.clip(yi, 1e-12, 1.0)
            eqs = []
            for i in range(n_comp - 1):
                driving_force_i = L[i] * (p_u * x_safe[i] - p_p * yi_safe[i])
                driving_force_j = L[i + 1] * (p_u * x_safe[i + 1] - p_p * yi_safe[i + 1])
                eqs.append(yi_safe[i] * driving_force_j - yi_safe[i + 1] * driving_force_i)
            eqs.append(np.sum(yi) - 1.0)
            return eqs

        yi_sol, _, ier, _ = fsolve(equations, x_safe.copy(), full_output=True)
        if ier != 1: pass
        return np.clip(yi_sol, 1e-10, 1.0)

    def _odes(self, A, y_state, params):
        n_comp = len(params["L"])
        x = y_state[:n_comp]
        Lu = y_state[n_comp]

        if Lu < 1e-9:
            return np.zeros(n_comp + 1)

        x = np.maximum(x, 0)
        sum_x = np.sum(x)
        if sum_x > 0:
            x /= sum_x

        yi = self._calc_yi_system(x, params)
        Ji = params["L"] * (params["p_u"] * x - params["p_p"] * yi)
        Ji = np.maximum(Ji, 0) 
        
        dLu_dA = -np.sum(Ji)
        dxi_dA = (x * np.sum(Ji) - Ji) / Lu

        return np.hstack((dxi_dA, dLu_dA))

    def run(self, feed_flux, feed_comp, area_target, params):
        if not np.isclose(np.sum(feed_comp), 1.0):
             feed_comp = feed_comp / np.sum(feed_comp)
        
        self.feed_flux = feed_flux
        self.feed_comp = feed_comp
        n_comp = len(feed_comp)
        y_state0 = np.hstack((feed_comp, feed_flux))

        # [수정된 부분] 함수 인자에 params를 추가해야 에러가 안 납니다!
        # solve_ivp가 args를 이벤트 함수에도 던지기 때문입니다.
        def retentate_empty(t, y, params): 
            return y[-1] - (feed_flux * 0.001)
            
        retentate_empty.terminal = True
        retentate_empty.direction = -1

        sol = solve_ivp(
            fun=self._odes,
            t_span=[0, area_target],
            y0=y_state0,
            method='RK45',
            args=(params,),
            events=retentate_empty
        )

        self.area = sol.t[-1]
        final_y_state = sol.y[:, -1]

        self.retentate_flux = max(final_y_state[n_comp], 0.0)
        self.retentate_comp = np.maximum(final_y_state[:n_comp], 0)
        if np.sum(self.retentate_comp) > 0:
            self.retentate_comp /= np.sum(self.retentate_comp)

        self.permeate_flux = self.feed_flux - self.retentate_flux
        
        if self.permeate_flux > 1e-9:
            feed_moles = self.feed_flux * self.feed_comp
            ret_moles = self.retentate_flux * self.retentate_comp
            permeate_moles = np.maximum(feed_moles - ret_moles, 0)
            
            if np.sum(permeate_moles) > 0:
                self.permeate_comp = permeate_moles / np.sum(permeate_moles)
            else:
                self.permeate_comp = np.zeros(n_comp)
        else:
            self.permeate_comp = np.zeros(n_comp)
            
        self.stage_cut = self.permeate_flux / self.feed_flux if self.feed_flux > 1e-9 else 0.0
        return True

# ==================================================================
# 3. Process 클래스
# ==================================================================
class Process2Stage:
    def __init__(self, params_list, area_list, stp_molar_volume=22414.0):
        self.params_list = params_list
        self.area_list = area_list
        self.stages = []
        self.stp_molar_volume = stp_molar_volume
        self.log_widget = st.empty()

    def _mix_streams(self, flux1, comp1, flux2, comp2):
        n_comp = len(comp1)
        moles1 = flux1 * comp1
        moles2 = flux2 * comp2 if flux2 > 0 else np.zeros(n_comp)
        total_moles = moles1 + moles2
        total_flux = np.sum(total_moles)
        if total_flux < 1e-9: return 0.0, np.zeros(n_comp)
        return total_flux, total_moles / total_flux

    def run_recycle_process(self, raw_feed_flux, raw_feed_comp, max_iterations=150, tolerance=1e-5):
        n_comp = len(raw_feed_comp)
        recycle_flux = 0.0
        recycle_comp = np.zeros(n_comp)

        log_output = "====== 2-Stage Recycle Process Start ======\n"
        self.log_widget.text(log_output)
        
        for i in range(max_iterations):
            current_stages = []
            
            # S1
            s1_feed_flux, s1_feed_comp = self._mix_streams(
                raw_feed_flux, raw_feed_comp, recycle_flux, recycle_comp
            )
            s1 = MembraneStage("Stage-1")
            s1.run(s1_feed_flux, s1_feed_comp, self.area_list[0], self.params_list[0])
            current_stages.append(s1)

            # S2
            s2_feed_flux = s1.permeate_flux
            s2_feed_comp = s1.permeate_comp
            s2 = MembraneStage("Stage-2")
            s2.run(s2_feed_flux, s2_feed_comp, self.area_list[1], self.params_list[1])
            current_stages.append(s2)

            # Convergence
            new_recycle_flux = s2.retentate_flux
            err = abs(new_recycle_flux - recycle_flux)
            
            log_output += f"Iter {i+1}: Recyc {recycle_flux:.4f} -> {new_recycle_flux:.4f} | S2_Perm_O2: {s2.permeate_comp[1]*100:.2f}%\n"
            self.log_widget.text(log_output)

            if err < tolerance:
                self.stages = current_stages
                self.log_widget.text(log_output + "\n✅ Converged!")
                return True
            
            recycle_flux = 0.5 * recycle_flux + 0.5 * new_recycle_flux
            recycle_comp = s2.retentate_comp
            
        self.log_widget.text(log_output + "\n❌ Max iterations reached.")
        return False

# ==================================================================
# 4. Streamlit UI
# ==================================================================
st.set_page_config(page_title="2-Stage Membrane Sim", layout="wide")
st.title("🧪 2-Stage (N2/O2) Membrane Simulator")

COMP_NAMES = ['N2', 'O2']

with st.sidebar:
    st.header("1. 초기 원료 (Raw Feed)")
    feed_flux_m3h = st.number_input("총 유량 (m³/h)", value=RAW_FEED_FLUX_M3H)
    col1, col2 = st.columns(2)
    with col1: comp_n2 = st.number_input("N2 몰분율", value=RAW_FEED_COMP[0], format="%.2f")
    with col2: comp_o2 = st.number_input("O2 몰분율", value=RAW_FEED_COMP[1], format="%.2f")

    st.header("2. 스테이지 설정 (GPU)")
    st.info("💡 Tip: S2 Area를 작게 해야 고순도가 나옵니다.")
    
    st.subheader("Stage 1")
    a1 = st.number_input("S1 Area (m²)", value=AREA_LIST_M2[0])
    pu1 = st.number_input("S1 Upstream (bar)", value=PROCESS_PARAMS_VOL["p_u_default"], key="pu1")
    pp1 = st.number_input("S1 Permeate (bar)", value=PROCESS_PARAMS_VOL["p_p_default"], key="pp1")
    l1_n2 = st.number_input("S1 N2 GPU", value=DEFAULT_L_GPU[0], key="l1n2")
    l1_o2 = st.number_input("S1 O2 GPU", value=DEFAULT_L_GPU[1], key="l1o2")

    st.subheader("Stage 2")
    a2 = st.number_input("S2 Area (m²)", value=AREA_LIST_M2[1]) 
    pu2 = st.number_input("S2 Upstream (bar)", value=PROCESS_PARAMS_VOL["p_u_default"], key="pu2")
    pp2 = st.number_input("S2 Permeate (bar)", value=PROCESS_PARAMS_VOL["p_p_default"], key="pp2")
    l2_n2 = st.number_input("S2 N2 GPU", value=DEFAULT_L_GPU[0], key="l2n2")
    l2_o2 = st.number_input("S2 O2 GPU", value=DEFAULT_L_GPU[1], key="l2o2")

    btn_run = st.button("실행 (Run)")

if btn_run:
    raw_feed_comp = np.array([comp_n2, comp_o2])
    if abs(sum(raw_feed_comp) - 1.0) > 1e-3:
        raw_feed_comp = raw_feed_comp / sum(raw_feed_comp)

    raw_feed_flux_mol = (feed_flux_m3h * M3H_TO_CM3S) / STP_MOLAR_VOLUME
    
    def get_params(pu, pp, gpu_n2, gpu_o2):
        gpu_arr = np.array([gpu_n2, gpu_o2])
        L_std = gpu_arr * GPU_TO_STD_UNITS
        L_mol = L_std / STP_MOLAR_VOLUME
        return {"L": L_mol, "p_u": pu * BAR_TO_ATM, "p_p": pp * BAR_TO_ATM}

    params_list = [
        get_params(pu1, pp1, l1_n2, l1_o2),
        get_params(pu2, pp2, l2_n2, l2_o2)
    ]
    area_list_cm2 = [a1 * M2_TO_CM2, a2 * M2_TO_CM2]

    proc = Process2Stage(params_list, area_list_cm2, STP_MOLAR_VOLUME)
    success = proc.run_recycle_process(raw_feed_flux_mol, raw_feed_comp)

    if success:
        st.success("계산 완료!")
        
        res = []
        vol_conv = STP_MOLAR_VOLUME * (3600/1e6) 

        for s in proc.stages:
            row = {
                "Stage": s.name,
                "Feed (m3/h)": s.feed_flux * vol_conv,
                "Feed O2%": s.feed_comp[1]*100,
                "Perm (m3/h)": s.permeate_flux * vol_conv,
                "Perm O2%": s.permeate_comp[1]*100,
                "Ret (m3/h)": s.retentate_flux * vol_conv,
                "StageCut": s.stage_cut
            }
            res.append(row)
        
        df = pd.DataFrame(res)
        df.set_index("Stage", inplace=True) 
        
        def highlight_target(val):
            if isinstance(val, float) and val > 94.0:
                return 'background-color: #d4edda; color: green; font-weight: bold'
            return ''
            
        st.dataframe(df.style.format("{:.2f}").map(highlight_target, subset=["Perm O2%"]), use_container_width=True)
        
        final_o2 = proc.stages[1].permeate_comp[1]*100
        final_flow = proc.stages[1].permeate_flux * vol_conv
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Final O2 Purity", f"{final_o2:.2f} %")
        c2.metric("Product Flow", f"{final_flow:.2f} m³/h")
        c3.metric("S2 Stage Cut", f"{proc.stages[1].stage_cut:.3f}")

        if proc.stages[1].stage_cut > 0.95:
            st.error("⚠️ 경고: Stage 2 Stage Cut이 1.0에 가깝습니다. 면적(Area)이 너무 커서 분리가 안 되고 있습니다.")
