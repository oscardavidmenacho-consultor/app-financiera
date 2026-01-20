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

# --- INYECCIÓN DE CSS ---
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.3rem;
        font-weight: 600;
    }
    .stApp {
        background-color: #f9f9f9; 
    }
    /* Estilo para quitar el subrayado del enlace del botón personalizado */
    a.custom-btn {
        text-decoration: none !important;
    }
    a.custom-btn:hover {
        opacity: 0.8;
    }
</style>
""", unsafe_allow_html=True)

# --- LAYOUT DE CABECERA ---
c_head_text, c_head_img = st.columns([2.5, 1], gap="medium")

with c_head_text:
    st.title("Análisis de Estados Financieros")
    st.markdown("### Oscar Menacho | Consultoría Financiera Corporativa")

with c_head_img:
    banner_file = "banner.jpg"
    if os.path.exists(banner_file):
        st.image(banner_file) 
    else:
        st.warning(f"⚠️ Falta 'banner.jpg'")

st.divider()

CORPORATE_COLORS = ['#004c70', '#5b9bd5', '#ed7d31', '#a5a5a5', '#ffc000', '#4472c4']

# --- FUNCIONES DE LÓGICA ---
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

# --- CREAR FIGURAS ---
def crear_figuras_dashboard(df_balance, df_pyg, df_indicadores, df_ratios):
    F_DATA = 16  
    F_AXIS = 18  
    F_LEG = 16   
    
    years_list = [str(c) for c in df_ratios.columns.tolist()]
    last_year = years_list[-1]
    orig_last_year = df_ratios.columns[-1]

    df_ratios = df_ratios.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    c_blue_dark = CORPORATE_COLORS[0]
    c_blue_light = CORPORATE_COLORS[1]
    c_yellow = CORPORATE_COLORS[2]
    c_ochre = CORPORATE_COLORS[5]
    c_brown = CORPORATE_COLORS[3]

    def apply_style(fig):
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
        return fig

    figs = {}

    # 1. Estructura Patrimonial
    act_cte = float(encontrar_cuenta(df_balance, ['Activo Corriente'])[orig_last_year])
    act_no_cte = float(encontrar_cuenta(df_balance, ['Activo No Corriente'])[orig_last_year])
    pas_cte = float(encontrar_cuenta(df_balance, ['Pasivo Corriente'])[orig_last_year])
    pas_no_cte = float(encontrar_cuenta(df_balance, ['Pasivo No Corriente'])[orig_last_year])
    patrimonio = float(encontrar_cuenta(df_balance, ['PATRIMONIO TOTAL'])[orig_last_year])
    
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(name='Activo No Corriente', x=['Activos'], y=[act_no_cte], marker_color=c_blue_dark, text=f"{act_no_cte/1e6:.1f}M", textposition='auto', textfont=dict(size=F_DATA)))
    fig1.add_trace(go.Bar(name='Activo Corriente', x=['Activos'], y=[act_cte], marker_color=c_blue_light, text=f"{act_cte/1e6:.1f}M", textposition='auto', textfont=dict(size=F_DATA)))
    fig1.add_trace(go.Bar(name='Patrimonio', x=['Pasivo y Patrimonio'], y=[patrimonio], marker_color=c_brown, text=f"{patrimonio/1e6:.1f}M", textposition='auto', textfont=dict(size=F_DATA)))
    fig1.add_trace(go.Bar(name='Pasivo No Corriente', x=['Pasivo y Patrimonio'], y=[pas_no_cte], marker_color=c_ochre, text=f"{pas_no_cte/1e6:.1f}M", textposition='auto', textfont=dict(size=F_DATA)))
    fig1.add_trace(go.Bar(name='Pasivo Corriente', x=['Pasivo y Patrimonio'], y=[pas_cte], marker_color=c_yellow, text=f"{pas_cte/1e6:.1f}M", textposition='auto', textfont=dict(size=F_DATA)))
    
    fig1 = apply_style(fig1)
    fig1.update_layout(barmode='stack', title=f"Estructura Patrimonial ({last_year})")
    figs['Estructura'] = fig1

    # 2. Cascada
    v_ventas = float(encontrar_cuenta(df_pyg, ['Ingresos por ventas'])[orig_last_year])
    v_costo = float(encontrar_cuenta(df_pyg, ['Costo de explotación', 'Costo de ventas'], hacer_abs=True)[orig_last_year])
    v_g_admin = float(encontrar_cuenta(df_pyg, ['Gastos administrativos'], hacer_abs=True)[orig_last_year])
    v_g_comerc = float(encontrar_cuenta(df_pyg, ['Gastos de comercialización'], hacer_abs=True)[orig_last_year])
    v_depre = float(encontrar_cuenta(df_pyg, ['Depreciaciones'], hacer_abs=True)[orig_last_year])
    v_gastos_op = v_g_admin + v_g_comerc + v_depre
    v_gastos_fin = float(encontrar_cuenta(df_pyg, ['Gastos financieros'], hacer_abs=True)[orig_last_year])
    v_impuestos = float(encontrar_cuenta(df_pyg, ["Impuesto a la renta"], hacer_abs=True)[orig_last_year])
    
    fig2 = go.Figure(go.Waterfall(
        orientation = "v", measure = ["absolute", "relative", "relative", "total", "relative", "relative", "total"],
        x = ["Ventas", "Costo", "Gastos Op.", "Res. Operativo", "Gastos Fin.", "Impuestos", "Utilidad Neta"],
        y = [v_ventas, -v_costo, -v_gastos_op, None, -v_gastos_fin, -v_impuestos, None],
        totals = {"marker":{"color": "gray"}}, increasing = {"marker":{"color": c_blue_dark}}, decreasing = {"marker":{"color": c_yellow}},
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
        textposition='auto', textfont=dict(size=F_DATA)
    ))
    fig2 = apply_style(fig2)
    fig2.update_layout(title=f"Cascada de Resultados ({last_year})")
    figs['Cascada'] = fig2

    # 3. Ventas Históricas
    ventas_series = encontrar_cuenta(df_pyg, ['Ingresos por ventas'])
    ventas_vals = pd.to_numeric(ventas_series, errors='coerce').fillna(0).tolist()
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=years_list, y=ventas_vals, text=[f"{v/1e6:.1f}M" for v in ventas_vals], textposition='auto', marker_color=c_blue_dark, textfont=dict(size=F_DATA)))
    fig3 = apply_style(fig3)
    fig3.update_layout(title="Evolución de Ventas")
    figs['Ventas'] = fig3

    # 4. Capital de Trabajo
    list_act_cte = encontrar_cuenta(df_balance, ['Activo Corriente']).astype(float).tolist()
    list_pas_cte = encontrar_cuenta(df_balance, ['Pasivo Corriente']).astype(float).tolist()
    list_neto = [(a - p) for a, p in zip(list_act_cte, list_pas_cte)]
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(name='Activo Corriente', x=years_list, y=list_act_cte, marker_color=c_blue_light, text=[f"{v/1e6:.1f}M" for v in list_act_cte], textposition='auto', textfont=dict(size=F_DATA)))
    fig4.add_trace(go.Bar(name='Pasivo Corriente', x=years_list, y=list_pas_cte, marker_color=c_yellow, text=[f"{v/1e6:.1f}M" for v in list_pas_cte], textposition='auto', textfont=dict(size=F_DATA)))
    fig4.add_trace(go.Scatter(name='Capital de Trabajo Neto', x=years_list, y=list_neto, mode='lines+markers+text', text=[f"{v/1e6:.1f}M" for v in list_neto], textposition="top center", line=dict(color='green', width=3, dash='dot'), marker=dict(size=10, color='green'), textfont=dict(size=F_DATA)))
    fig4 = apply_style(fig4)
    fig4.update_layout(title="Capital de Trabajo")
    figs['CapitalTrabajo'] = fig4

    # 5. Liquidez
    liq_corr = df_ratios.loc['Liquidez Corriente'].values.tolist()
    pru_acid = df_ratios.loc['Prueba Ácida'].values.tolist()
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=years_list, y=liq_corr, name='Liquidez Corriente', mode='lines+markers+text', text=[f"{v:.2f}" for v in liq_corr], textposition="top center", line=dict(color=c_blue_dark), textfont=dict(size=F_DATA)))
    fig5.add_trace(go.Scatter(x=years_list, y=pru_acid, name='Prueba Ácida', mode='lines+markers+text', text=[f"{v:.2f}" for v in pru_acid], textposition="top center", line=dict(color=c_blue_light), textfont=dict(size=F_DATA)))
    fig5 = apply_style(fig5)
    fig5.update_layout(title="Liquidez")
    figs['Liquidez'] = fig5

    # 6. Rentabilidad
    roe = df_ratios.loc['ROE'].values.tolist()
    roa = df_ratios.loc['ROA'].values.tolist()
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=years_list, y=roe, name='ROE', mode='lines+markers+text', text=[f"{v:.1%}" for v in roe], textposition="top center", line=dict(color=c_ochre), textfont=dict(size=F_DATA)))
    fig6.add_trace(go.Scatter(x=years_list, y=roa, name='ROA', mode='lines+markers+text', text=[f"{v:.1%}" for v in roa], textposition="top center", line=dict(color='gray'), textfont=dict(size=F_DATA)))
    fig6.update_layout(yaxis_tickformat=".2%", title="Rentabilidad (ROE/ROA)")
    fig6 = apply_style(fig6)
    figs['Rentabilidad'] = fig6

    # 7. Margenes
    m_bruto = df_ratios.loc['Margen Bruto'].values.tolist()
    m_oper = df_ratios.loc['Margen Operativo'].values.tolist()
    m_neto = df_ratios.loc['Margen Neto'].values.tolist()
    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(x=years_list, y=m_bruto, name='Margen Bruto', mode='lines+markers+text', text=[f"{v:.0%}" for v in m_bruto], textposition="top center", line=dict(color=c_blue_dark), textfont=dict(size=F_DATA)))
    fig7.add_trace(go.Scatter(x=years_list, y=m_oper, name='Margen Operativo', mode='lines+markers+text', text=[f"{v:.0%}" for v in m_oper], textposition="top center", line=dict(color=c_blue_light), textfont=dict(size=F_DATA)))
    fig7.add_trace(go.Scatter(x=years_list, y=m_neto, name='Margen Neto', mode='lines+markers+text', text=[f"{v:.0%}" for v in m_neto], textposition="top center", line=dict(color=c_ochre), textfont=dict(size=F_DATA)))
    fig7.update_layout(yaxis_tickformat=".2%", title="Márgenes")
    fig7 = apply_style(fig7)
    figs['Margenes'] = fig7

    # 8. Actividad
    rot_cxc = df_ratios.loc['Rotación CxC (días)'].values.tolist()
    rot_inv = df_ratios.loc['Rotación Inventario (días)'].values.tolist()
    rot_cxp = df_ratios.loc['Rotación CxP (días)'].values.tolist()
    fig8 = go.Figure()
    fig8.add_trace(go.Bar(x=years_list, y=rot_cxc, name='Rotación CxC', marker_color=c_blue_dark, text=[f"{v:.0f}" for v in rot_cxc], textposition='auto', textfont=dict(size=F_DATA)))
    fig8.add_trace(go.Bar(x=years_list, y=rot_inv, name='Rotación Inv.', marker_color=c_blue_light, text=[f"{v:.0f}" for v in rot_inv], textposition='auto', textfont=dict(size=F_DATA)))
    fig8.add_trace(go.Bar(x=years_list, y=rot_cxp, name='Rotación CxP', marker_color=c_ochre, text=[f"{v:.0f}" for v in rot_cxp], textposition='auto', textfont=dict(size=F_DATA)))
    fig8.update_layout(title="Ciclo de Efectivo (Días)")
    fig8 = apply_style(fig8)
    figs['Actividad'] = fig8
    
    return figs

def to_excel(df_balance, df_pyg, df_indicadores, df_ratios, figs):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # --- ESTILOS ---
        header_format = workbook.add_format({
            'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#004c70', 
            'font_color': 'white', 'border': 1, 'align': 'center'
        })
        num_format = workbook.add_format({'num_format': '#,##0', 'border': 1})
        pct_format = workbook.add_format({'num_format': '0.0%', 'border': 1})
        text_format = workbook.add_format({'border': 1, 'align': 'left'})
        
        sheets = {
            'Balance_Analizado': df_balance,
            'Resultados_P&G': df_pyg,
            'Indicadores_P&G': df_indicadores,
            'Ratios_Financieros': df_ratios
        }

        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, startrow=1, header=False)
            worksheet = writer.sheets[sheet_name]
            
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num + 1, value, header_format)
            worksheet.write(0, 0, "Cuenta / Concepto", header_format)

            worksheet.set_column('A:A', 40, text_format)
            
            for i, col in enumerate(df.columns):
                col_idx = i + 1 
                col_name = str(col).lower()
                
                if any(x in col_name for x in ['%', 'margen', 'roa', 'roe', 'crecimiento', 'var_%']):
                    worksheet.set_column(col_idx, col_idx, 15, pct_format)
                elif 'días' in col_name or 'dias' in col_name:
                     worksheet.set_column(col_idx, col_idx, 15, num_format)
                else:
                    worksheet.set_column(col_idx, col_idx, 18, num_format)
        
        worksheet_g = workbook.add_worksheet("DASHBOARD_GRAFICO")
        worksheet_g.hide_gridlines(2)
        worksheet_g.write(0, 0, "DASHBOARD EJECUTIVO - GRÁFICOS", header_format)
        
        row_pos = 2
        for key, fig in figs.items():
            img_bytes = fig.to_image(format="png", width=1000, height=500, scale=2)
            image_data = io.BytesIO(img_bytes)
            worksheet_g.insert_image(row_pos, 1, key, {'image_data': image_data, 'x_scale': 0.8, 'y_scale': 0.8})
            row_pos += 25

    return output.getvalue()

# --- INTERFAZ PRINCIPAL ---
with st.sidebar:
    st.header("📂 Carga de Datos")
    uploaded_file = st.file_uploader("Sube tu archivo Excel (.xlsx)", type="xlsx")
    st.divider()
    
    template_filename = "plantilla.xlsx"
    if os.path.exists(template_filename):
        with open(template_filename, "rb") as file:
            st.download_button("📥 Descargar Plantilla Excel", file, "Plantilla_Analisis_Financiero.xlsx")
        st.error("⚠️ IMPORTANTE: Esta App NO funciona con cualquier Excel. Usa la plantilla.")
    else:
        st.warning(f"⚠️ Falta '{template_filename}'.")
    
    st.divider()
    
    # --- BOTÓN CTA PERSONALIZADO (GRANDE Y NARANJA) ---
    st.markdown("""
    <a href="https://bento.me/oscar-menacho-consultor-financiero" target="_blank" class="custom-btn">
        <div style="
            width: 100%;
            background-color: #ed7d31; 
            color: white; 
            padding: 18px; 
            text-align: center; 
            border-radius: 12px; 
            font-weight: bold; 
            font-size: 18px;
            box-shadow: 0px 4px 6px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        ">
            🎓 Ver mis Cursos y Servicios
        </div>
    </a>
    """, unsafe_allow_html=True)

if uploaded_file is not None:
    try:
        xls = pd.ExcelFile(uploaded_file)
        nombres_pestanas = xls.sheet_names
        nombre_balance = encontrar_nombre_pestana(nombres_pestanas, 'balance')
        nombre_pyg = encontrar_nombre_pestana(nombres_pestanas, 'resultado')

        if not nombre_balance or not nombre_pyg:
            st.error("Error: Formato incorrecto.")
            st.stop()

        df_bal = pd.read_excel(xls, sheet_name=nombre_balance, header=1, index_col=0).dropna(how='all').dropna(how='all', axis=1)
        df_pyg_orig = pd.read_excel(xls, sheet_name=nombre_pyg, header=1, index_col=0).dropna(how='all').dropna(how='all', axis=1)

        df_bal = clean_column_headers(df_bal)
        df_pyg_orig = clean_column_headers(df_pyg_orig)
        for df in [df_bal, df_pyg_orig]:
            df.index.name = 'Cuenta'
            for col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
            df.fillna(0, inplace=True)
        
        df_bal_an = procesar_balance(df_bal)
        df_ind_pyg = procesar_pyg(df_pyg_orig)
        df_ratios = calcular_ratios(df_bal, df_pyg_orig)
        
        figs = crear_figuras_dashboard(df_bal, df_pyg_orig, df_ind_pyg, df_ratios)
        excel_data = to_excel(df_bal_an, df_pyg_orig, df_ind_pyg, df_ratios, figs)

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Balance", "📈 P&G", "🔢 Ratios", "🖼️ DASHBOARD"])

        with tab1: st.dataframe(df_bal_an.style.format("{:,.0f}"))
        with tab2: st.dataframe(df_ind_pyg.style.format("{:.2%}"))
        with tab3: st.dataframe(df_ratios)
        with tab4:
            st.header("Dashboard Gráfico Interactivo")
            col1, _, col2 = st.columns([1, 0.1, 1])
            with col1: st.plotly_chart(figs['Estructura'], use_container_width=True)
            with col2: st.plotly_chart(figs['Cascada'], use_container_width=True)
            st.divider()
            col3, _, col4 = st.columns([1, 0.1, 1])
            with col3: st.plotly_chart(figs['Ventas'], use_container_width=True)
            with col4: st.plotly_chart(figs['CapitalTrabajo'], use_container_width=True)
            st.divider()
            col5, _, col6 = st.columns([1, 0.1, 1])
            with col5: st.plotly_chart(figs['Liquidez'], use_container_width=True)
            with col6: st.plotly_chart(figs['Rentabilidad'], use_container_width=True)
            st.divider()
            col7, _, col8 = st.columns([1, 0.1, 1])
            with col7: st.plotly_chart(figs['Margenes'], use_container_width=True)
            with col8: st.plotly_chart(figs['Actividad'], use_container_width=True)
        
        st.sidebar.divider()
        st.sidebar.download_button("📥 Descargar Reporte Completo", excel_data, f"Reporte_Financiero_{datetime.now().strftime('%Y%m%d')}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    except Exception as e:
        st.error(f"Error técnico: {e}")
else:
    st.info("👋 ¡Hola! Para usar esta App, primero descarga la plantilla del menú izquierdo, complétala y súbela.")
