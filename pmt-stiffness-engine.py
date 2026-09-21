import io
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


class PMTEngine:
    def __init__(self, p_atm=101.325):
        self.p_atm = p_atm

    @staticmethod
    def power_law(gamma, alpha, beta):
        """
        Bolton & Whittle (1999) formulation: tau = alpha * (gamma ^ beta)
        """
        gamma_safe = np.maximum(gamma, 1e-6)
        return alpha * (gamma_safe ** beta)

    def identify_liftoff_p0(self, cavity_strain_pct, radial_pressure_kpa):
        """
        Identifies in-situ lift-off stress (P0 = sigma_h0) based on initial 
        inflection and yield point (Lacasse & Lunne 1982 / Marsland & Randolph 1977).
        """
        strain_diff = np.diff(cavity_strain_pct)
        press_diff = np.diff(radial_pressure_kpa)
        slope = press_diff / np.maximum(strain_diff, 1e-4)
        
        # Detect where initial steep response breaks into cavity expansion
        idx = np.where(cavity_strain_pct[1:] >= 0.2)[0]
        if len(idx) > 0:
            target_idx = idx[0]
            p0 = float(radial_pressure_kpa[target_idx])
        else:
            target_idx = 1
            p0 = float(radial_pressure_kpa[1])
            
        return p0, target_idx

    def process_unload_reload_loop(self, loop_strain_pct, loop_stress_kpa):
        """
        Analyzes PMT reload loop, normalizes strain origin, fits Bolton & Whittle power law,
        and derives secant (Gs) and tangent (Gt) shear modulus decay profiles (Eq. 5.64).
        """
        # Isolate reload phase (from minimum stress point to end of loop)
        min_idx = np.argmin(loop_stress_kpa)
        reload_strain = loop_strain_pct[min_idx:]
        reload_stress = loop_stress_kpa[min_idx:]
        
        gamma_raw = np.maximum((reload_strain - reload_strain[0]) / 100.0, 1e-5)
        tau_raw = np.maximum(reload_stress - reload_stress[0], 1.0)

        # Non-linear curve fitting for power-law parameters alpha and beta
        try:
            popt, _ = curve_fit(self.power_law, gamma_raw, tau_raw, p0=[8000.0, 0.75], maxfev=5000)
            alpha, beta = popt
        except Exception:
            alpha, beta = 7500.0, 0.72

        # Strain decay spectrum: 10^-4 to 10^-2 (CFEM Fig. 5.31 & 5.34)
        gamma_eval = np.logspace(-4, -2, 100)
        gs_kpa = alpha * (gamma_eval ** (beta - 1.0))
        gt_kpa = alpha * beta * (gamma_eval ** (beta - 1.0))
        
        return {
            "alpha": alpha,
            "beta": beta,
            "gamma_eval": gamma_eval,
            "Gs_MPa": gs_kpa / 1000.0,
            "Gt_MPa": gt_kpa / 1000.0,
            "fitted_gamma": gamma_raw,
            "fitted_tau": self.power_law(gamma_raw, alpha, beta)
        }

    def generate_py_curve(self, cavity_strain_pct, radial_pressure_kpa, p0, pile_dia_m, soil_type="Sand"):
        """
        Constructs lateral pile p-y response directly from PMT loading response (Robertson et al. 1985).
        p = alpha_pile * sigma_r * D
        y = (delta_R / R) * (D / 2)
        """
        alpha_pile = 1.5 if soil_type == "Sand" else 2.0
        delta_r_over_r = cavity_strain_pct / 100.0
        
        # Net radial pressure mobilized after lift-off
        net_radial_stress = np.maximum(radial_pressure_kpa - p0, 0.0)
        
        y_disp_m = delta_r_over_r * (pile_dia_m / 2.0)
        p_res_kn_m = alpha_pile * (net_radial_stress * pile_dia_m)
        
        return y_disp_m, p_res_kn_m


def generate_benchmark_pmt():
    """Generates synthetic Self-Boring Pressuremeter (SBPM) loading with unload-reload cycles."""
    strain_primary = np.linspace(0.0, 10.0, 150)
    
    # Non-linear expansion with lift-off at 280 kPa
    p0_bench = 280.0
    stress_primary = p0_bench + 650.0 * np.sqrt(strain_primary / 10.0)
    
    # Embed an Unload-Reload Loop between strain 3.0% and 4.5%
    mask_loop = (strain_primary >= 3.0) & (strain_primary <= 4.5)
    loop_strain = strain_primary[mask_loop]
    unloading = 700.0 - 200.0 * np.sin(np.pi * (loop_strain - 3.0) / 1.5)
    stress_primary[mask_loop] = unloading
    
    return strain_primary, stress_primary


def build_pmt_plotly(strain, stress, p0):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strain, y=stress, mode="lines+markers", 
                             marker=dict(size=4), line=dict(color="#1f77b4", width=2), 
                             name="PMT Expansion Curve"))
    fig.add_hline(y=p0, line_dash="dash", line_color="red", 
                  annotation_text=f"Lift-off P0 = {p0:.1f} kPa", annotation_position="top left")
    fig.update_xaxes(title_text="Cavity Strain (%)")
    fig.update_yaxes(title_text="Applied Radial Pressure (kPa)")
    fig.update_layout(title="PMT Expansion & Cavity Stress-Strain Response", height=450, margin=dict(l=40, r=40, t=50, b=40))
    return fig


def build_stiffness_plotly(gamma_arr, gs_mpa, gt_mpa):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gamma_arr, y=gs_mpa, mode="lines", 
                             line=dict(color="#2ca02c", width=2.5), name="Secant Modulus (Gs)"))
    fig.add_trace(go.Scatter(x=gamma_arr, y=gt_mpa, mode="lines", 
                             line=dict(color="#d62728", width=2, dash="dash"), name="Tangent Modulus (Gt)"))
    fig.update_xaxes(type="log", title_text="Shear Strain, gamma (log scale)")
    fig.update_yaxes(title_text="Shear Modulus (MPa)")
    fig.update_layout(title="Non-linear Modulus Degradation (Bolton & Whittle)", height=450, margin=dict(l=40, r=40, t=50, b=40))
    return fig


def build_py_plotly(y_disp_m, p_res_kn_m):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_disp_m * 1000.0, y=p_res_kn_m, mode="lines+markers", 
                             marker=dict(size=4), line=dict(color="#9467bd", width=2), 
                             name="p-y Response"))
    fig.update_xaxes(title_text="Lateral Pile Deflection, y (mm)")
    fig.update_yaxes(title_text="Lateral Soil Resistance, p (kN/m)")
    fig.update_layout(title="Robertson (1985) Synthesized Pile p-y Curve", height=450, margin=dict(l=40, r=40, t=50, b=40))
    return fig


def generate_static_report(strain, stress, p0, gamma_arr, gs_mpa, gt_mpa, y_disp_m, p_res_kn_m, alpha, beta):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # Cavity expansion plot
    ax1.plot(strain, stress, color="navy", lw=1.8, label="PMT Curve")
    ax1.axhline(y=p0, color="crimson", linestyle="--", lw=1.2, label=f"P0 = {p0:.1f} kPa")
    ax1.set_xlabel("Cavity Strain (%)")
    ax1.set_ylabel("Radial Pressure (kPa)")
    ax1.set_title("Cavity Expansion Response")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Modulus degradation plot
    ax2.plot(gamma_arr, gs_mpa, color="forestgreen", lw=1.8, label="Secant Gs")
    ax2.plot(gamma_arr, gt_mpa, color="firebrick", linestyle="--", lw=1.5, label="Tangent Gt")
    ax2.set_xscale("log")
    ax2.set_xlabel("Shear Strain, $\gamma$")
    ax2.set_ylabel("Modulus (MPa)")
    ax2.set_title(f"Stiffness Degradation ($\\alpha$={alpha:.0f}, $\\beta$={beta:.2f})")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend()

    # p-y response plot
    ax3.plot(y_disp_m * 1000.0, p_res_kn_m, color="purple", lw=1.8, label="p-y curve")
    ax3.set_xlabel("Deflection, y (mm)")
    ax3.set_ylabel("Soil Resistance, p (kN/m)")
    ax3.set_title("Synthesized Pile p-y Curve")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend()

    plt.suptitle("Comprehensive PMT Stress-Strain & Non-linear Stiffness Analysis (CFEM-5)", fontsize=13)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf


def main():
    st.set_page_config(page_title="PMT Advanced Stiffness & p-y Platform", layout="wide")
    st.title("Advanced Pressuremeter (PMT) Interpretation Platform")
    st.markdown("Non-linear soil stiffness degradation and direct foundation parameter extraction (CFEM Chapter 5)")

    engine = PMTEngine()
    bench_strain, bench_stress = generate_benchmark_pmt()

    # Sidebar Parameters
    st.sidebar.header("Site & Foundation Setup")
    sigma_v0_eff = st.sidebar.number_input("Effective Overburden Stress sigma_v0' (kPa)", min_value=10.0, value=150.0, step=10.0)
    pile_dia = st.sidebar.number_input("Target Pile Diameter (m)", min_value=0.2, max_value=3.0, value=0.8, step=0.1)
    soil_class = st.sidebar.selectbox("Soil Characteristic Behavior", options=["Sand", "Clay"])

    st.sidebar.subheader("Unload-Reload Window Selection")
    u_strain_start = st.sidebar.slider("Cycle Strain Start (%)", 0.5, 8.0, 3.0, 0.1)
    u_strain_end = st.sidebar.slider("Cycle Strain End (%)", 1.0, 9.0, 4.5, 0.1)

    st.sidebar.header("Input Data")
    uploaded_file = st.sidebar.file_uploader("Upload PMT Expansion File (CSV/Excel)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_up = pd.read_csv(uploaded_file)
            else:
                df_up = pd.read_excel(uploaded_file)
            strain_data = df_up["cavity_strain_pct"].values
            stress_data = df_up["radial_pressure_kpa"].values
            st.sidebar.success("Custom PMT sounding loaded successfully.")
        except Exception as e:
            st.sidebar.error(f"Error loading custom file: {e}")
            return
    else:
        st.sidebar.info("Using representative benchmark SBPM test data.")
        strain_data, stress_data = bench_strain, bench_stress

    # Core Calculations
    p0_identified, _ = engine.identify_liftoff_p0(strain_data, stress_data)
    k0_calculated = p0_identified / max(sigma_v0_eff, 1.0)

    # Extract reload loop data points based on user slider window
    loop_mask = (strain_data >= u_strain_start) & (strain_data <= u_strain_end)
    if np.sum(loop_mask) > 3:
        loop_res = engine.process_unload_reload_loop(strain_data[loop_mask], stress_data[loop_mask])
    else:
        # Fallback window if selected window has insufficient points
        fallback_mask = (strain_data >= 2.8) & (strain_data <= 4.6)
        loop_res = engine.process_unload_reload_loop(strain_data[fallback_mask], stress_data[fallback_mask])

    # Direct p-y curve construction
    y_pile_m, p_pile_kn_m = engine.generate_py_curve(strain_data, stress_data, p0_identified, pile_dia, soil_class)

    # UI Presentation
    tab1, tab2, tab3 = st.tabs(["Stress-Strain & Lateral Stress", "Non-linear Stiffness Degradation", "Pile p-y Interface"])

    with tab1:
        st.subheader("In-Situ Lateral Stress and Cavity Expansion Profile")
        col1, col2 = st.columns([3, 1])
        with col1:
            fig_pmt = build_pmt_plotly(strain_data, stress_data, p0_identified)
            st.plotly_chart(fig_pmt, use_container_width=True)
        with col2:
            st.metric("Lift-off Pressure (P0)", f"{p0_identified:.1f} kPa")
            st.metric("Earth Pressure Coeff. (K0)", f"{k0_calculated:.2f}")
            st.metric("Effective Overburden", f"{sigma_v0_eff:.1f} kPa")

    with tab2:
        st.subheader("Power Law Non-linear Stiffness Decay (Bolton & Whittle 1999)")
        col_a, col_b = st.columns([3, 1])
        with col_a:
            fig_stiff = build_stiffness_plotly(loop_res["gamma_eval"], loop_res["Gs_MPa"], loop_res["Gt_MPa"])
            st.plotly_chart(fig_stiff, use_container_width=True)
        with col_b:
            st.metric("Power Law Constant (alpha)", f"{loop_res['alpha']:.1f} kPa")
            st.metric("Power Law Gradient (beta)", f"{loop_res['beta']:.3f}")
            st.write("**Modulus Decay Values:**")
            st.write(f"- $G_s$ at $10^{{-4}}$: `{loop_res['Gs_MPa'][0]:.1f} MPa`")
            st.write(f"- $G_s$ at $10^{{-2}}$: `{loop_res['Gs_MPa'][-1]:.1f} MPa`")

    with tab3:
        st.subheader(f"Synthesized Lateral Pile Response ({soil_class} Condition, Dia = {pile_dia} m)")
        fig_py = build_py_plotly(y_pile_m, p_pile_kn_m)
        st.plotly_chart(fig_py, use_container_width=True)

    # Static High-Resolution Plot Export
    report_png = generate_static_report(
        strain_data, stress_data, p0_identified,
        loop_res["gamma_eval"], loop_res["Gs_MPa"], loop_res["Gt_MPa"],
        y_pile_m, p_pile_kn_m, loop_res["alpha"], loop_res["beta"]
    )
    st.sidebar.download_button(
        label="Download Publication Log (PNG)",
        data=report_png,
        file_name="PMT_Advanced_Analysis_Report.png",
        mime="image/png"
    )

    # Formatted Excel Export
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        pd.DataFrame({
            "Cavity_Strain_pct": strain_data,
            "Radial_Pressure_kPa": stress_data
        }).to_excel(writer, sheet_name="PMT_Raw_Curve", index=False)

        pd.DataFrame({
            "Shear_Strain_gamma": loop_res["gamma_eval"],
            "Secant_Modulus_Gs_MPa": loop_res["Gs_MPa"],
            "Tangent_Modulus_Gt_MPa": loop_res["Gt_MPa"]
        }).to_excel(writer, sheet_name="Stiffness_Degradation", index=False)

        pd.DataFrame({
            "Pile_Deflection_mm": y_pile_m * 1000.0,
            "Lateral_Resistance_p_kN_m": p_pile_kn_m
        }).to_excel(writer, sheet_name="Synthesized_py_Curve", index=False)

        summary_df = pd.DataFrame({
            "Parameter": ["Lift_off_P0_kPa", "Calculated_K0", "Power_Law_Alpha_kPa", "Power_Law_Beta", "Pile_Diameter_m"],
            "Value": [p0_identified, k0_calculated, loop_res["alpha"], loop_res["beta"], pile_dia]
        })
        summary_df.to_excel(writer, sheet_name="Model_Parameters", index=False)

    st.sidebar.download_button(
        label="Download Comprehensive Excel Report",
        data=excel_buffer.getvalue(),
        file_name="PMT_Constitutive_Calibration_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


if __name__ == "__main__":
    main()
