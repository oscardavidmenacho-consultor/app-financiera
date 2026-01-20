import pandas as pd
import streamlit as st
import numpy as np
import io
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import os

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="PROFINANCE | Análisis Financiero", page_icon="📊", layout="wide")

# --- INYECCIÓN DE CSS (ESTILOS VISUALES) ---
st.markdown("""
<style>
    /* Aumentar tamaño de fuente de las pestañas */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.3rem;
        font-weight: 600;
    }
    /* Color de fondo suave para el dashboard */
    .stApp {
        background-color: #f9f9f9; 
    }
    /* Ajuste fino para alinear el título verticalmente con el banner */
    .main-header-container {
        display: flex;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# --- LAYOUT DE CABECERA (NUEVO: Título Izq, Banner Der) ---
# Creamos columnas: Izquierda (más ancha) para texto, Derecha (más estrecha) para imagen
c_head_text, c_head_img = st.columns([2.5, 1], gap="medium")

with c_head_text:
    # Título y subtítulo a la izquierda
    st.title("Análisis de Estados Financieros")
    st.markdown("### Oscar Menacho | Consultoría Financiera Corporativa")

with c_head_img:
    # Banner a la derecha, restringido por el ancho de la columna
    banner_file = "banner.jpg"
    if os.path.exists(banner_file):
        # Al quitar use_column_width, la imagen no se estira excesivamente
        st.image(banner_file) 
    else:
        st.warning(f"⚠️ Falta 'banner.jpg'")

st.divider() # Una línea separadora elegante después de la cabecera

# --- PALETA DE COLORES CORPORATIVOS ---
CORPORATE_COLORS = ['#004c70', '#5b9bd5', '#ed7d31', '#a5a5a5', '#ffc000', '#4472c4']

# --- FUNCIONES DE LÓGICA DE CÁLCULO ---

def encontrar_cuenta(df, lista_alias, hacer_abs=False):
    for alias in lista_alias:
        fila_encontrada = df[df.index.str.contains(alias, case=False, na=False)]
        if not fila_encontrada.empty:
            serie = fila_encontrada.iloc[0]
            serie = pd.to_numeric(serie, errors='coerce').fillna(0)
            return abs(serie) if hacer_abs else serie
    return pd.Series(0, index=df.columns)

def clean_column_headers(df):
    new_columns = [str(col).split('-')[0].split(' ')[0] for col in df.columns]
    df.columns = new_columns
    return df

def encontrar_nombre_pestana(nombres_pestanas, alias_buscar):
    for nombre in nombres_pestanas:
        if alias_buscar.lower() in nombre.lower():
            return nombre
    return None

def procesar_balance(df_balance):
    df = df_balance.copy()
    years = df.columns.tolist()
    last_year = years[-1]

    total_assets_last_year = encontrar_cuenta(df, ["ACTIVOS TOTALES"])[last_year]

    if total_assets_last_year != 0:
        df[f'AV_%_{last_year}'] = (df[last_year] / total_assets_last_year)
    else:
        df[f'AV_%_{last_year}'] = 0

    base_year = None
    activos_totales_row = encontrar_cuenta(df, ["ACTIVOS TOTALES"])

    if len(years) > 0 and activos_totales_row.get(years[0], 0) > 0:
        base_year = years[0]
    elif len(years) > 1 and activos_totales_row.get(years[1], 0) > 0:
        base_year = years[1]

    if base_year and base_year != last_year:
        df[f'Var_$ ({last_year} vs {base_year})'] = df[last_year] - df[base_year]
        numerador = df[last_year] - df[base_year]
        denominador = df[base_year]
        df[f'Var_% ({last_year} vs {base_year})'] = (numerador / denominador.replace(0, np.nan)).fillna(0)
    else:
        df['Var_$'] = 0
        df['Var_%'] = 0

    return df

def procesar_pyg(df_pyg):
    years = df_pyg.columns.tolist()
    ventas = encontrar_cuenta(df_pyg, ['Ingresos por ventas'])
    costo_ventas = encontrar_cuenta(df_pyg, ['Costo de explotación', 'Costo de ventas'], hacer_abs=True)
    resultado_bruto = encontrar_cuenta(df_pyg, ["RESULTADO BRUTO"])
    gastos_admin = encontrar_cuenta(df_pyg, ['Gastos administrativos'], hacer_abs=True)
    gastos_comerc = encontrar_cuenta(df_pyg, ['Gastos de comercialización'], hacer_abs=True)
    depreciacion = encontrar_cuenta(df_pyg, ['Depreciaciones y amortizaciones', 'Depreciación'], hacer_abs=True)
    ebitda = encontrar_cuenta(df_pyg, ["EBITDA"])
    resultado_op = encontrar_cuenta(df_pyg, ["RESULTADO OPERATIVO"])
    resultado_neto = encontrar_cuenta(df_pyg, ["RESULTADO DEL EJERCICIO", "Utilidad Neta"])

    indicadores = pd.DataFrame(columns=years, index=[
        "Crecimiento en Ventas", "Costo de Ventas / Ventas", "Margen Bruto",
        "Crecimiento Gastos Admin.", "Crecimiento Gastos Comerc.", "Gastos Operativos Totales / Ventas",
        "Margen EBITDA", "Margen Operativo", "Margen Neto"
    ])

    gastos_op_totales = gastos_admin + gastos_comerc + depreciacion

    for i, year in enumerate(years):
        if i > 0:
            prev_year = years[i-1]
            if ventas.get(prev_year, 0) != 0: 
                indicadores.loc["Crecimiento en Ventas", year] = (ventas[year] / ventas[prev_year] - 1)
            if gastos_admin.get(prev_year, 0) != 0: 
                indicadores.loc["Crecimiento Gastos Admin.", year] = (gastos_admin[year] / gastos_admin[prev_year] - 1)
            if gastos_comerc.get(prev_year, 0) != 0: 
                indicadores.loc["Crecimiento Gastos Comerc.", year] = (gastos_comerc[year] / gastos_comerc[prev_year] - 1)

        if ventas.get(year, 0) != 0:
            indicadores.loc["Costo de Ventas / Ventas", year] = costo_ventas[year] / ventas[year]
            indicadores.loc["Margen Bruto", year] = resultado_bruto[year] / ventas[year]
            indicadores.loc["Gastos Operativos Totales / Ventas", year] = gastos_op_totales[year] / ventas[year]
            indicadores.loc["Margen EBITDA", year] = ebitda[year] / ventas[year]
            indicadores.loc["Margen Operativo", year] = resultado_op[year] / ventas[year]
            indicadores.loc["Margen Neto", year] = resultado_neto[year] / ventas[year]

    return indicadores.astype(float).fillna(0)

def calcular_ratios(df_balance, df_pyg):
    years = df_balance.columns.tolist()

    activo_cte = encontrar_cuenta(df_balance, ['Activo Corriente'])
    pasivo_cte = encontrar_cuenta(df_balance, ['Pasivo Corriente'])
    inventario = encontrar_cuenta(df_balance, ['Inventario', 'Existencias'])
    caja = encontrar_cuenta(df_balance, ['Caja y bancos', 'Disponibilidades'])
    cxc = encontrar_cuenta(df_balance, ['Cuentas por cobrar'])
    cxp = encontrar_cuenta(df_balance, ['Cuentas por pagar'])
    activos_totales = encontrar_cuenta(df_balance, ['ACTIVOS TOTALES'])
    pasivos_totales = encontrar_cuenta(df_balance, ['PASIVOS TOTALES'])
    patrimonio = encontrar_cuenta(df_balance, ['PATRIMONIO TOTAL'])

    ventas = encontrar_cuenta(df_pyg, ['Ingresos por ventas'])
    costo = encontrar_cuenta(df_pyg, ['Costo de explotación', 'Costo de ventas'], hacer_abs=True)
    resultado_bruto = encontrar_cuenta(df_pyg, ["RESULTADO BRUTO"]) 
    resultado_ejercicio = encontrar_cuenta(df_pyg, ["RESULTADO DEL EJERCICIO", "Utilidad Neta"])
    ebitda = encontrar_cuenta(df_pyg, ["EBITDA"])
    resultado_op = encontrar_cuenta(df_pyg, ["RESULTADO OPERATIVO"])

    ratios = pd.DataFrame(columns=years)

    for year in years:
        ratios.loc['Liquidez Corriente', year] = activo_cte[year] / pasivo_cte[year] if pasivo_cte[year] else 0
        ratios.loc['Prueba Ácida', year] = (activo_cte[year] - inventario[year]) / pasivo_cte[year] if pasivo_cte[year] else 0
        ratios.loc['Liquidez Inmediata', year] = caja[year] / pasivo_cte[year] if pasivo_cte[year] else 0

        ratios.loc['Rotación CxC (días)', year] = cxc[year] / (ventas[year] / 365) if ventas[year] else 0
        ratios.loc['Rotación Inventario (días)', year] = inventario[year] / (costo[year] / 365) if costo[year] else 0
        ratios.loc['Rotación CxP (días)', year] = cxp[year] / (costo[year] / 365) if costo[year] else 0
        ratios.loc['Rotación de Activos', year] = ventas[year] / activos_totales[year] if activos_totales[year] else 0

        ratios.loc['Endeudamiento', year] = pasivos_totales[year] / activos_totales[year] if activos_totales[year] else 0
        ratios.loc['Multiplicador de Capital', year] = activos_totales[year] / patrimonio[year] if patrimonio[year] else 0
        ratios.loc['Razón Deuda/Capital', year] = pasivos_totales[year] / patrimonio[year] if patrimonio[year] else 0

        ratios.loc['ROA', year] = resultado_ejercicio[year] / activos_totales[year] if activos_totales[year] else 0
        ratios.loc['ROE', year] = resultado_ejercicio[year] / patrimonio[year] if patrimonio[year] else 0

        ratios.loc['Margen Bruto', year] = resultado_bruto[year] / ventas[year] if ventas[year] else 0
        ratios.loc['Margen EBITDA', year] = ebitda[year] / ventas[year] if ventas[year] else 0
        ratios.loc['Margen Operativo', year] = resultado_op[year] / ventas[year] if ventas[year] else 0
        ratios.loc['Margen Neto', year] = resultado_ejercicio[year] / ventas[year] if ventas[year] else 0

    return ratios.astype(float).fillna(0)

def to_excel(df_balance, df_pyg, df_indicadores, df_ratios):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_balance.to_excel(writer, sheet_name='Balance_Analizado')
        df_pyg.to_excel(writer, sheet_name='Resultados_e_Indicadores', startcol=0)
        df_indicadores.to_excel(writer, sheet_name='Resultados_e_Indicadores', startcol=len(df_pyg.columns) + 2)
        df_ratios.to_excel(writer, sheet_name='Ratios_Financieros')
    return output.getvalue()

def generar_dashboard(df_balance, df_pyg, df_indicadores, df_ratios):
    st.header("Dashboard Gráfico Interactivo")

    # --- CONFIGURACIÓN DE TAMAÑO DE FUENTES ---
    F_DATA = 16  
    F_AXIS = 18  
    F_LEG = 16   

    # --- PREPARACIÓN DE DATOS ---
    years_list = [str(c) for c in df_ratios.columns.tolist()]
    last_year = years_list[-1]
    orig_last_year = df_ratios.columns[-1]

    df_ratios = df_ratios.apply(pd.to_numeric, errors='coerce').fillna(0)
    df_indicadores = df_indicadores.apply(pd.to_numeric, errors='coerce').fillna(0)

    c_blue_dark = CORPORATE_COLORS[0]
    c_blue_light = CORPORATE_COLORS[1]
    c_yellow = CORPORATE_COLORS[2]
    c_ochre = CORPORATE_COLORS[5]
    c_brown = CORPORATE_COLORS[3]

    # --- FUNCIÓN DE LAYOUT (GENÉRICA) ---
    def update_fig_layout(fig):
        fig.update_layout(
            barmode='group',
            legend=dict(font=dict(size=F_LEG), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(size=14, color="black"), 
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig.update_xaxes(tickfont=dict(size=F_AXIS, color='black'), type='category', showgrid=False)
        fig.update_yaxes(tickfont=dict(size=F_AXIS, color='black'), showgrid=True, gridcolor='lightgray')
        st.plotly_chart(fig, use_container_width=True)

    # --- FILA 1 ---
    col1, _, col2 = st.columns([1, 0.1, 1])

    with col1:
        st.subheader(f"Estructura Patrimonial ({last_year})")
        act_cte = float(encontrar_cuenta(df_balance, ['Activo Corriente'])[orig_last_year])
        act_no_cte = float(encontrar_cuenta(df_balance, ['Activo No Corriente'])[orig_last_year])
        pas_cte = float(encontrar_cuenta(df_balance, ['Pasivo Corriente'])[orig_last_year])
        pas_no_cte = float(encontrar_cuenta(df_balance, ['Pasivo No Corriente'])[orig_last_year])
        patrimonio = float(encontrar_cuenta(df_balance, ['PATRIMONIO TOTAL'])[orig_last_year])

        fig = go.Figure()
        # Columna Activos
        fig.add_trace(go.Bar(name='Activo No Corriente', x=['Activos'], y=[act_no_cte], marker_color=c_blue_dark, text=f"{act_no_cte/1e6:.1f}M", textposition='auto', textfont=dict(size=F_DATA)))
        fig.add_trace(go.Bar(name='Activo Corriente', x=['Activos'], y=[act_cte], marker_color=c_blue_light, text=f"{act_cte/1e6:.1f}M", textposition='auto', textfont=dict(size=F_DATA)))

        # Columna Pasivos
        fig.add_trace(go.Bar(name='Patrimonio', x=['Pasivo y Patrimonio'], y=[patrimonio], marker_color=c_brown, text=f"{patrimonio/1e6:.1f}M", textposition='auto', textfont=dict(size=F_DATA)))
        fig.add_trace(go.Bar(name='Pasivo No Corriente', x=['Pasivo y Patrimonio'], y=[pas_no_cte], marker_color=c_ochre, text=f"{pas_no_cte/1e6:.1f}M", textposition='auto', textfont=dict(size=F_DATA)))
        fig.add_trace(go.Bar(name='Pasivo Corriente', x=['Pasivo y Patrimonio'], y=[pas_cte], marker_color=c_yellow, text=f"{pas_cte/1e6:.1f}M", textposition='auto', textfont=dict(size=F_DATA)))

        # CORRECCIÓN DE APILADO: Aplicamos el layout MANUALMENTE y NO llamamos a update_fig_layout
        fig.update_layout(
            barmode='stack', # ESTO ASEGURA QUE SE APILE
            yaxis_title='Valor', 
            legend=dict(font=dict(size=F_LEG), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), 
            font=dict(size=14, color="black"),
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig.update_xaxes(tickfont=dict(size=F_AXIS, color='black'))
        fig.update_yaxes(tickfont=dict(size=F_AXIS, color='black'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(f"Cascada de Resultados ({last_year})")
        v_ventas = float(encontrar_cuenta(df_pyg, ['Ingresos por ventas'])[orig_last_year])
        v_costo = float(encontrar_cuenta(df_pyg, ['Costo de explotación', 'Costo de ventas'], hacer_abs=True)[orig_last_year])
        v_g_admin = float(encontrar_cuenta(df_pyg, ['Gastos administrativos'], hacer_abs=True)[orig_last_year])
        v_g_comerc = float(encontrar_cuenta(df_pyg, ['Gastos de comercialización'], hacer_abs=True)[orig_last_year])
        v_depre = float(encontrar_cuenta(df_pyg, ['Depreciaciones'], hacer_abs=True)[orig_last_year])
        v_gastos_op = v_g_admin + v_g_comerc + v_depre
        v_gastos_fin = float(encontrar_cuenta(df_pyg, ['Gastos financieros'], hacer_abs=True)[orig_last_year])
        v_impuestos = float(encontrar_cuenta(df_pyg, ["Impuesto a la renta"], hacer_abs=True)[orig_last_year])

        fig = go.Figure(go.Waterfall(
            orientation = "v", measure = ["absolute", "relative", "relative", "total", "relative", "relative", "total"],
            x = ["Ventas", "Costo", "Gastos Op.", "Res. Operativo", "Gastos Fin.", "Impuestos", "Utilidad Neta"],
            y = [v_ventas, -v_costo, -v_gastos_op, None, -v_gastos_fin, -v_impuestos, None],
            totals = {"marker":{"color": "gray"}}, increasing = {"marker":{"color": c_blue_dark}}, decreasing = {"marker":{"color": c_yellow}},
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
            textposition='auto', textfont=dict(size=F_DATA)
        ))
        update_fig_layout(fig)

    st.divider()

    # --- FILA 2 ---
    col3, _, col4 = st.columns([1, 0.1, 1])
    with col3:
        st.subheader("Evolución de Ventas (Histórico)")
        ventas_series = encontrar_cuenta(df_pyg, ['Ingresos por ventas'])
        ventas_vals = pd.to_numeric(ventas_series, errors='coerce').fillna(0).tolist()

        fig = go.Figure()
        fig.add_trace(go.Bar(x=years_list, y=ventas_vals, text=[f"{v/1e6:.1f}M" for v in ventas_vals], textposition='auto', marker_color=c_blue_dark, textfont=dict(size=F_DATA)))
        update_fig_layout(fig)

    with col4:
        st.subheader("Capital de Trabajo (Activo Cte vs Pasivo Cte)")
        list_act_cte = encontrar_cuenta(df_balance, ['Activo Corriente']).astype(float).tolist()
        list_pas_cte = encontrar_cuenta(df_balance, ['Pasivo Corriente']).astype(float).tolist()
        list_neto = [(a - p) for a, p in zip(list_act_cte, list_pas_cte)]

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Activo Corriente', x=years_list, y=list_act_cte, marker_color=c_blue_light, text=[f"{v/1e6:.1f}M" for v in list_act_cte], textposition='auto', textfont=dict(size=F_DATA)))
        fig.add_trace(go.Bar(name='Pasivo Corriente', x=years_list, y=list_pas_cte, marker_color=c_yellow, text=[f"{v/1e6:.1f}M" for v in list_pas_cte], textposition='auto', textfont=dict(size=F_DATA)))
        fig.add_trace(go.Scatter(name='Capital de Trabajo Neto', x=years_list, y=list_neto, mode='lines+markers+text', text=[f"{v/1e6:.1f}M" for v in list_neto], textposition="top center", line=dict(color='green', width=3, dash='dot'), marker=dict(size=10, color='green'), textfont=dict(size=F_DATA)))
        update_fig_layout(fig)

    st.divider()

    # --- FILA 3 ---
    st.subheader("Evolución de Grandes Grupos del Balance")
    categories = ['Activo Corriente', 'Activo No Corriente', 'Pasivo Corriente', 'Pasivo No Corriente', 'Patrimonio']
    fig = go.Figure()
    for i, year_col in enumerate(df_balance.columns):
        v_ac = float(encontrar_cuenta(df_balance, ['Activo Corriente'])[year_col])
        v_anc = float(encontrar_cuenta(df_balance, ['Activo No Corriente'])[year_col])
        v_pc = float(encontrar_cuenta(df_balance, ['Pasivo Corriente'])[year_col])
        v_pnc = float(encontrar_cuenta(df_balance, ['Pasivo No Corriente'])[year_col])
        v_pat = float(encontrar_cuenta(df_balance, ['PATRIMONIO TOTAL'])[year_col])
        vals = [v_ac, v_anc, v_pc, v_pnc, v_pat]
        fig.add_trace(go.Bar(name=years_list[i], x=categories, y=vals, text=[f"{v/1e6:.1f}M" for v in vals], textposition='auto', textfont=dict(size=F_DATA)))
    update_fig_layout(fig)

    st.divider()

    # --- FILA 4: RATIOS ---
    col5, _, col6 = st.columns([1, 0.1, 1])

    with col5:
        st.subheader("Tendencias de Liquidez")
        liq_corr = df_ratios.loc['Liquidez Corriente'].values.tolist()
        pru_acid = df_ratios.loc['Prueba Ácida'].values.tolist()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years_list, y=liq_corr, name='Liquidez Corriente', mode='lines+markers+text', text=[f"{v:.2f}" for v in liq_corr], textposition="top center", line=dict(color=c_blue_dark), textfont=dict(size=F_DATA)))
        fig.add_trace(go.Scatter(x=years_list, y=pru_acid, name='Prueba Ácida', mode='lines+markers+text', text=[f"{v:.2f}" for v in pru_acid], textposition="top center", line=dict(color=c_blue_light), textfont=dict(size=F_DATA)))
        update_fig_layout(fig)

    with col6:
        st.subheader("Tendencias de Rendimiento (ROE y ROA)")
        roe = df_ratios.loc['ROE'].values.tolist()
        roa = df_ratios.loc['ROA'].values.tolist()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years_list, y=roe, name='ROE', mode='lines+markers+text', text=[f"{v:.1%}" for v in roe], textposition="top center", line=dict(color=c_ochre), textfont=dict(size=F_DATA)))
        fig.add_trace(go.Scatter(x=years_list, y=roa, name='ROA', mode='lines+markers+text', text=[f"{v:.1%}" for v in roa], textposition="top center", line=dict(color='gray'), textfont=dict(size=F_DATA)))
        fig.update_layout(yaxis_tickformat=".2%")
        update_fig_layout(fig)

    col7, _, col8 = st.columns([1, 0.1, 1])
    with col7:
        st.subheader("Tendencias de Rentabilidad (Márgenes)")
        m_bruto = df_ratios.loc['Margen Bruto'].values.tolist()
        m_oper = df_ratios.loc['Margen Operativo'].values.tolist()
        m_neto = df_ratios.loc['Margen Neto'].values.tolist()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years_list, y=m_bruto, name='Margen Bruto', mode='lines+markers+text', text=[f"{v:.0%}" for v in m_bruto], textposition="top center", line=dict(color=c_blue_dark), textfont=dict(size=F_DATA)))
        fig.add_trace(go.Scatter(x=years_list, y=m_oper, name='Margen Operativo', mode='lines+markers+text', text=[f"{v:.0%}" for v in m_oper], textposition="top center", line=dict(color=c_blue_light), textfont=dict(size=F_DATA)))
        fig.add_trace(go.Scatter(x=years_list, y=m_neto, name='Margen Neto', mode='lines+markers+text', text=[f"{v:.0%}" for v in m_neto], textposition="top center", line=dict(color=c_ochre), textfont=dict(size=F_DATA)))
        fig.update_layout(yaxis_tickformat=".2%")
        update_fig_layout(fig)

    with col8:
        st.subheader("Indicadores de Actividad (Días)")
        rot_cxc = df_ratios.loc['Rotación CxC (días)'].values.tolist()
        rot_inv = df_ratios.loc['Rotación Inventario (días)'].values.tolist()
        rot_cxp = df_ratios.loc['Rotación CxP (días)'].values.tolist()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=years_list, y=rot_cxc, name='Rotación CxC', marker_color=c_blue_dark, text=[f"{v:.0f}" for v in rot_cxc], textposition='auto', textfont=dict(size=F_DATA)))
        fig.add_trace(go.Bar(x=years_list, y=rot_inv, name='Rotación Inv.', marker_color=c_blue_light, text=[f"{v:.0f}" for v in rot_inv], textposition='auto', textfont=dict(size=F_DATA)))
        fig.add_trace(go.Bar(x=years_list, y=rot_cxp, name='Rotación CxP', marker_color=c_ochre, text=[f"{v:.0f}" for v in rot_cxp], textposition='auto', textfont=dict(size=F_DATA)))
        update_fig_layout(fig)

# --- INTERFAZ PRINCIPAL ---
with st.sidebar:
    st.header("📂 Carga de Datos")
    uploaded_file = st.file_uploader("Sube tu archivo Excel (.xlsx)", type="xlsx")

    st.divider()

    # --- LÓGICA DE DESCARGA DE PLANTILLA ---
    template_filename = "plantilla.xlsx"
    if os.path.exists(template_filename):
        with open(template_filename, "rb") as file:
            st.download_button(
                label="📥 Descargar Plantilla Excel",
                data=file,
                file_name="Plantilla_Analisis_Financiero.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        # Mensaje profesional para el cliente
        st.info("ℹ️ Instrucciones: Descargue este formato, complete sus datos financieros y suba el archivo arriba para generar el reporte.")
    else:
        st.warning(f"⚠️ Sube tu Excel bueno a Replit y renómbralo a '{template_filename}' para habilitar la descarga.")

if uploaded_file is not None:
    try:
        xls = pd.ExcelFile(uploaded_file)
        nombres_pestanas = xls.sheet_names
        nombre_balance = encontrar_nombre_pestana(nombres_pestanas, 'balance')
        nombre_pyg = encontrar_nombre_pestana(nombres_pestanas, 'resultado')

        if not nombre_balance or not nombre_pyg:
            st.error("Error: Faltan pestañas clave en el Excel.")
            st.stop()

        df_balance_orig = pd.read_excel(xls, sheet_name=nombre_balance, header=1, index_col=0).dropna(how='all').dropna(how='all', axis=1)
        df_pyg_orig = pd.read_excel(xls, sheet_name=nombre_pyg, header=1, index_col=0).dropna(how='all').dropna(how='all', axis=1)

        df_balance_orig = clean_column_headers(df_balance_orig)
        df_pyg_orig = clean_column_headers(df_pyg_orig)
        for df in [df_balance_orig, df_pyg_orig]:
            df.index.name = 'Cuenta'
            for col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
            df.fillna(0, inplace=True)

        df_balance_analizado = procesar_balance(df_balance_orig)
        df_indicadores_pyg = procesar_pyg(df_pyg_orig)
        df_ratios = calcular_ratios(df_balance_orig, df_pyg_orig)

        # Pestañas con nombres estilizados
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Balance General", 
            "📈 Estado de Resultados", 
            "🔢 Ratios Financieros", 
            "🖼️ DASHBOARD EJECUTIVO"
        ])

        with tab1:
            st.header("Análisis del Balance General")
            format_dict = {col: '{:,.0f}' for col in df_balance_orig.columns}
            format_dict.update({col: '{:.0%}' for col in df_balance_analizado.columns if '%' in col})
            format_dict.update({col: '{:,.0f}' for col in df_balance_analizado.columns if '$' in col})
            st.dataframe(df_balance_analizado.style.format(format_dict, na_rep="-"), use_container_width=True)
        with tab2:
            st.header("Análisis del Estado de Resultados (P&G)")
            st.subheader("Estado de Resultados Original")
            st.dataframe(df_pyg_orig.style.format("{:,.0f}"), use_container_width=True)
            st.divider()
            st.subheader("Indicadores P&G")
            st.dataframe(df_indicadores_pyg.style.format("{:.2%}"), use_container_width=True)
        with tab3:
            st.header("Ratios Financieros Clave")
            df_ratios_display = df_ratios.copy()
            for idx in df_ratios_display.index:
                if 'días' in idx: df_ratios_display.loc[idx] = df_ratios_display.loc[idx].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "-")
                elif any(kw in idx for kw in ['ROA', 'ROE', 'Margen']): df_ratios_display.loc[idx] = df_ratios_display.loc[idx].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
                else: df_ratios_display.loc[idx] = df_ratios_display.loc[idx].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "-")
            st.dataframe(df_ratios_display, use_container_width=True)
        with tab4:
            generar_dashboard(df_balance_orig, df_pyg_orig, df_indicadores_pyg, df_ratios)

        excel_data = to_excel(df_balance_analizado, df_pyg_orig, df_indicadores_pyg, df_ratios)

        st.sidebar.divider()
        st.sidebar.download_button(
            label="📥 Descargar Reporte Completo",
            data=excel_data,
            file_name=f"Reporte_Financiero_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"Error técnico: {e}")
else:
    st.info("👈 Sube tu archivo Excel en el panel lateral o descarga la plantilla para comenzar.")