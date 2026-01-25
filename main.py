import pandas as pd
import streamlit as st
import numpy as np
import io
import urllib.parse
from datetime import datetime
import plotly.graph_objects as go
import os

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Oscar Menacho | Análisis Financiero", page_icon="📊", layout="wide")

# --- INYECCIÓN DE CSS (ESTILOS VISUALES - V61 DEFINITIVO) ---
st.markdown("""
<style>
    /* 1. FONDO APP PRINCIPAL */
    .stApp {
        background-color: #f9f9f9; 
    }
    
    /* 2. BARRA LATERAL (SIDEBAR) - FONDO CLARO */
    section[data-testid="stSidebar"] {
        background-color: #f0f2f6 !important;
    }
    
    /* 3. ARREGLO DE FLECHAS (COLLAPSE Y EXPAND) */
    /* Apuntamos a los dos botones posibles (dentro y fuera del sidebar) */
    [data-testid="stSidebarCollapseButton"] > svg,
    [data-testid="stSidebarExpandButton"] > svg {
        fill: #333333 !important;   /* Forzar relleno negro */
        stroke: #333333 !important; /* Forzar línea negra */
        color: #333333 !important;
    }
    /* Refuerzo para el "camino" (path) interno del icono */
    [data-testid="stSidebarCollapseButton"] > svg > path,
    [data-testid="stSidebarExpandButton"] > svg > path {
        fill: #333333 !important;
        stroke: #333333 !important;
    }

    /* 4. TEXTOS DEL SIDEBAR EN NEGRO (Excluyendo botones personalizados) */
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div[data-testid="stFileUploaderFileName"],
    section[data-testid="stSidebar"] div[data-testid="stFileUploaderFileStatus"] {
        color: #333333 !important;
    }
    
    /* Títulos del sidebar */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #333333 !important;
    }

    /* 5. EXCEPCIÓN CRÍTICA: BOTONES HOTMART/BENTO/MAIL */
    /* Estos botones tienen la clase .custom-btn. Forzamos su texto a BLANCO */
    section[data-testid="stSidebar"] a.custom-btn div,
    a.custom-btn div {
        color: white !important; /* Texto blanco */
        -webkit-text-fill-color: white !important; /* Asegurar en webkit */
    }

    /* 6. CAJA DE COMENTARIOS (EXPANDER) */
    div[data-testid="stExpander"] summary p,
    div[data-testid="stExpander"] summary span,
    div[data-testid="stExpander"] summary svg {
        color: #333333 !important;
        fill: #333333 !important;
    }

    /* 7. ESTILOS GENERALES (Títulos principales, pestañas, tablas) */
    h1, h2, h3, h4, h5, h6 {
        color: #333333 !important;
    }
    /* Excepción para el Título Principal Azul */
    h1 {
        color: #004c70 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
        font-weight: 600;
        color: #444444 !important;
    }
    
    a.custom-btn {
        text-decoration: none !important;
    }
    a.custom-btn:hover {
        opacity: 0.9;
    }
    
    .stDataFrame {
        font-size: 1.3rem !important;
    }
    .stAlert p {
        font-size: 1.2rem !important;
        line-height: 1.5 !important;
    }
    div[data-testid="stVerticalBlock"] div[data-testid="stDataFrame"] div[class*="ColumnHeaders"] {
        background-color: #004c70;
        color: white;
        font-size: 1.2rem !important;
    }
    div[data-testid="stMarkdownContainer"] > h3, div[data-testid="stMarkdownContainer"] > h4 {
        margin-top: 0px !important;
        padding-top: 0px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- LAYOUT DE CABECERA ---
col_ban1, col_ban2 = st.columns([2, 1], gap="large")

with col_ban1:
    # Usamos HTML directo para asegurar control total sobre el título principal
    st.markdown('<h1 style="color:#004c70 !important; font-size:3rem; margin-bottom:10px; padding-bottom:0; line-height: 1.1;">Análisis de Estados Financieros Automatizado</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#333333; font-weight:400; font-size: 1.5rem; line-height: 1.4; margin-bottom: 15px;">Automatiza los cálculos y enfócate en el diagnóstico. Tendencias + Ratios + Dashboard en segundos.</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#333333; font-weight:500; font-size: 1.4rem;">Oscar Menacho | Consultor y Docente Financiero</p>', unsafe_allow_html=True)

with col_ban2:
    banner_file = "banner.jpg"
    if os.path.exists(banner_file):
        st.image(banner_file, use_container_width=True) 
    else:
        st.warning(f"⚠️ Falta 'banner.jpg'")

st.divider()

CORPORATE_COLORS = ['#004c70', '#5b9bd5', '#ed7d31', '#a5a5a5', '#ffc000', '#4472c4']

# --- FUNCIONES DE FORMATO VISUAL (INTERFAZ) ---
def formato_latino(valor, es_porcentaje=False, decimales=0):
    """Convierte números a formato 1.000,00 (Punto mil, Coma decimal)"""
    if pd.isna(valor) or valor == 0:
        return "-"
    
    if es_porcentaje:
        return f"{valor * 100:,.{decimales}f}%".replace(".", "X").replace(",", ".").replace("X", ",")
    else:
        return f"{valor:,.{decimales}f}".replace(".", "X").replace(",", ".").replace("X", ",")

def aplicar_estilos_df(df, tipo='balance'):
    """Aplica colores a encabezados y formato visual a los datos"""
    df_visual = df.copy()
    
    if tipo == 'indicadores':
        for col in df_visual.columns:
            df_visual[col] = df_visual[col].apply(lambda x: formato_latino(x, es_porcentaje=True, decimales=1))

    elif tipo == 'ratios':
        for idx in df_visual.index:
            idx_str = str(idx).lower()
            if 'días' in idx_str or 'dias' in idx_str:
                fmt = lambda x: formato_latino(x, decimales=0)
            elif any(x in idx_str for x in ['margen', 'roa', 'roe']):
                fmt = lambda x: formato_latino(x, es_porcentaje=True, decimales=0)
            else:
                fmt = lambda x: formato_latino(x, decimales=1)
            
            for col in df_visual.columns:
                val = df_visual.loc[idx, col]
                df_visual.loc[idx, col] = fmt(val)

    else: 
        for col in df_visual.columns:
            col_lower = str(col).lower()
            if any(x in col_lower for x in ['%', 'var_%', 'av_', 'vertical_%', 'horizontal_%']):
                df_visual[col] = df_visual[col].apply(lambda x: formato_latino(x, es_porcentaje=True, decimales=1))
            else:
                df_visual[col] = df_visual[col].apply(lambda x: formato_latino(x, decimales=0))
            
    styler = df_visual.style.set_properties(**{
        'font-size': '20px', 
        'text-align': 'center',
        'border-color': 'lightgray'
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#004c70'), 
            ('color', 'white'), 
            ('font-size', '20px'),
            ('font-weight', 'bold'),
            ('text-align', 'center')
        ]}
    ])
    
    return styler

# --- FUNCIONES DE LÓGICA (CORE) ---
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
        df[f'Vertical_%_{last_year}'] = (df[last_year] / total_assets_last_year)
    else:
        df[f'Vertical_%_{last_year}'] = 0
            
    base_year = None
    activos_totales_row = encontrar_cuenta(df, ["ACTIVOS TOTALES"])
    
    if len(years) > 0 and activos_totales_row.get(years[0], 0) > 0:
        base_year = years[0]
    elif len(years) > 1 and activos_totales_row.get(years[1], 0) > 0:
        base_year = years[1]

    if base_year and base_year != last_year:
        df[f'Horizontal_$ ({last_year} vs {base_year})'] = df[last_year] - df[base_year]
        numerador = df[last_year] - df[base_year]
        denominador = df[base_year]
        df[f'Horizontal_% ({last_year} vs {base_year})'] = (numerador / denominador.replace(0, np.nan)).fillna(0)
    else:
        df['Horizontal_$'] = 0
        df['Horizontal_%'] = 0
        
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

# --- CREAR FIGURAS DASHBOARD ---
def crear_figuras_dashboard(df_balance, df_pyg, df_indicadores, df_ratios):
    F_DATA = 16  
    F_AXIS = 16  
    F_LEG = 14
    F_TITLE = 22 
    
    years_list = [str(c) for c in df_ratios.columns.tolist()]
    last_year = years_list[-1]
    orig_last_year = df_ratios.columns[-1]

    df_ratios = df_ratios.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    c_blue_dark = CORPORATE_COLORS[0]
    c_blue_light = CORPORATE_COLORS[1]
    c_yellow = CORPORATE_COLORS[2]
    c_ochre = CORPORATE_COLORS[5]
    c_brown = CORPORATE_COLORS[3]

    def apply_style(fig, title_text, max_y_val=None):
        if max_y_val and max_y_val > 0:
            fig.update_yaxes(range=[0, max_y_val * 1.15]) 
            
        fig.update_layout(
            template='plotly_white', 
            title=dict(text=title_text, font=dict(size=F_TITLE, color="black")), 
            barmode='group',
            legend=dict(font=dict(size=F_LEG, color="black"), orientation="v", yanchor="top", y=1, xanchor="left", x=1.02), 
            font=dict(size=14, color="black"), 
            margin=dict(l=50, r=20, t=80, b=40),
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
    max_val_est = max(act_cte + act_no_cte, pas_cte + pas_no_cte + patrimonio) 

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(name='Activo No Corriente', x=['Activos'], y=[act_no_cte], marker_color=c_blue_dark, text=f"{act_no_cte/1e6:.1f}M", textposition='auto', textfont=dict(size=F_DATA)))
    fig1.add_trace(go.Bar(name='Activo Corriente', x=['Activos'], y=[act_cte], marker_color=c_blue_light, text=f"{act_cte/1e6:.1f}M", textposition='auto', textfont=dict(size=F_DATA)))
    fig1.add_trace(go.Bar(name='Patrimonio', x=['Pasivo y Patrimonio'], y=[patrimonio], marker_color=c_brown, text=f"{patrimonio/1e6:.1f}M", textposition='auto', textfont=dict(size=F_DATA)))
    fig1.add_trace(go.Bar(name='Pasivo No Corriente', x=['Pasivo y Patrimonio'], y=[pas_no_cte], marker_color=c_ochre, text=f"{pas_no_cte/1e6:.1f}M", textposition='auto', textfont=dict(size=F_DATA)))
    fig1.add_trace(go.Bar(name='Pasivo Corriente', x=['Pasivo y Patrimonio'], y=[pas_cte], marker_color=c_yellow, text=f"{pas_cte/1e6:.1f}M", textposition='auto', textfont=dict(size=F_DATA)))
    
    fig1 = apply_style(fig1, f"Estructura Patrimonial ({last_year})", max_val_est)
    fig1.update_layout(barmode='stack')
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
        text=[f"{v/1e6:.1f}M" for v in [v_ventas, v_costo, v_gastos_op, v_ventas-v_costo-v_gastos_op, v_gastos_fin, v_impuestos, float(encontrar_cuenta(df_pyg, ["RESULTADO DEL EJERCICIO"])[orig_last_year])]],
        textposition='auto', textfont=dict(size=F_DATA)
    ))
    fig2 = apply_style(fig2, f"Cascada de Resultados ({last_year})")
    figs['Cascada'] = fig2

    # 3. Ventas Históricas
    ventas_series = encontrar_cuenta(df_pyg, ['Ingresos por ventas'])
    ventas_vals = pd.to_numeric(ventas_series, errors='coerce').fillna(0).tolist()
    max_ventas = max(ventas_vals) if ventas_vals else 0
    
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=years_list, y=ventas_vals, text=[f"{v/1e6:.1f}M" for v in ventas_vals], textposition='auto', marker_color=c_blue_dark, textfont=dict(size=F_DATA)))
    fig3 = apply_style(fig3, "Evolución de Ventas", max_ventas)
    figs['Ventas'] = fig3

    # 4. Capital de Trabajo
    list_act_cte = encontrar_cuenta(df_balance, ['Activo Corriente']).astype(float).tolist()
    list_pas_cte = encontrar_cuenta(df_balance, ['Pasivo Corriente']).astype(float).tolist()
    list_neto = [(a - p) for a, p in zip(list_act_cte, list_pas_cte)]
    max_cap = max(max(list_act_cte), max(list_pas_cte)) if list_act_cte else 0
    
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(name='Activo Corriente', x=years_list, y=list_act_cte, marker_color=c_blue_light, text=[f"{v/1e6:.1f}M" for v in list_act_cte], textposition='auto', textfont=dict(size=F_DATA)))
    fig4.add_trace(go.Bar(name='Pasivo Corriente', x=years_list, y=list_pas_cte, marker_color=c_yellow, text=[f"{v/1e6:.1f}M" for v in list_pas_cte], textposition='auto', textfont=dict(size=F_DATA)))
    fig4.add_trace(go.Scatter(name='Capital de Trabajo Neto', x=years_list, y=list_neto, mode='lines+markers+text', text=[f"{v/1e6:.1f}M" for v in list_neto], textposition="top center", line=dict(color='green', width=3, dash='dot'), marker=dict(size=10, color='green'), textfont=dict(size=F_DATA)))
    fig4 = apply_style(fig4, "Capital de Trabajo (AC vs PC)", max_cap)
    figs['CapitalTrabajo'] = fig4

    # 5. Grandes Grupos del Balance
    categories = ['Activo Cte', 'Activo No Cte', 'Pasivo Cte', 'Pasivo No Cte', 'Patrimonio']
    fig_grupos = go.Figure()
    max_val_grupos = 0
    colors_grupos = [c_blue_light, c_blue_dark, c_yellow, c_ochre, c_brown]
    
    for idx_cat, cat in enumerate(categories):
        vals = []
        for year_col in df_balance.columns:
             if cat == 'Activo Cte': val = float(encontrar_cuenta(df_balance, ['Activo Corriente'])[year_col])
             elif cat == 'Activo No Cte': val = float(encontrar_cuenta(df_balance, ['Activo No Corriente'])[year_col])
             elif cat == 'Pasivo Cte': val = float(encontrar_cuenta(df_balance, ['Pasivo Corriente'])[year_col])
             elif cat == 'Pasivo No Cte': val = float(encontrar_cuenta(df_balance, ['Pasivo No Corriente'])[year_col])
             else: val = float(encontrar_cuenta(df_balance, ['PATRIMONIO TOTAL'])[year_col])
             vals.append(val)
        
        if vals: max_val_grupos = max(max_val_grupos, max(vals))
        
        fig_grupos.add_trace(go.Bar(
            name=cat, x=years_list, y=vals, 
            marker_color=colors_grupos[idx_cat],
            text=[f"{v/1e6:.1f}M" for v in vals], textposition='auto', textfont=dict(size=F_DATA)
        ))

    fig_grupos = apply_style(fig_grupos, "Evolución Grandes Grupos del Balance", max_val_grupos)
    figs['GrandesGrupos'] = fig_grupos

    # 6. Liquidez
    liq_corr = df_ratios.loc['Liquidez Corriente'].values.tolist()
    pru_acid = df_ratios.loc['Prueba Ácida'].values.tolist()
    max_liq = max(max(liq_corr), max(pru_acid)) if liq_corr else 0
    
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=years_list, y=liq_corr, name='Liquidez Corriente', mode='lines+markers+text', text=[f"{v:.2f}" for v in liq_corr], textposition="top center", 
                              line=dict(color=c_blue_dark, width=4), marker=dict(size=12, symbol='square'), textfont=dict(size=F_DATA)))
    fig5.add_trace(go.Scatter(x=years_list, y=pru_acid, name='Prueba Ácida', mode='lines+markers+text', text=[f"{v:.2f}" for v in pru_acid], textposition="top center", 
                              line=dict(color=c_blue_light, width=4), marker=dict(size=12, symbol='square'), textfont=dict(size=F_DATA)))
    fig5 = apply_style(fig5, "Liquidez", max_liq)
    figs['Liquidez'] = fig5

    # 7. Rentabilidad
    roe = df_ratios.loc['ROE'].values.tolist()
    roa = df_ratios.loc['ROA'].values.tolist()
    max_rent = max(max(roe), max(roa)) if roe else 0
    
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=years_list, y=roe, name='ROE', mode='lines+markers+text', text=[f"{v:.1%}" for v in roe], textposition="top center", 
                              line=dict(color=c_ochre, width=4), marker=dict(size=12, symbol='square'), textfont=dict(size=F_DATA)))
    fig6.add_trace(go.Scatter(x=years_list, y=roa, name='ROA', mode='lines+markers+text', text=[f"{v:.1%}" for v in roa], textposition="top center", 
                              line=dict(color='gray', width=4), marker=dict(size=12, symbol='square'), textfont=dict(size=F_DATA)))
    fig6.update_layout(yaxis_tickformat=".2%")
    fig6 = apply_style(fig6, "Rentabilidad (ROE/ROA)", max_rent)
    figs['Rentabilidad'] = fig6

    # 8. Margenes
    m_bruto = df_ratios.loc['Margen Bruto'].values.tolist()
    m_oper = df_ratios.loc['Margen Operativo'].values.tolist()
    m_neto = df_ratios.loc['Margen Neto'].values.tolist()
    max_marg = max(max(m_bruto), max(m_oper)) if m_bruto else 0
    
    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(x=years_list, y=m_bruto, name='Margen Bruto', mode='lines+markers+text', text=[f"{v:.0%}" for v in m_bruto], textposition="top center", 
                              line=dict(color=c_blue_dark, width=4), marker=dict(size=12, symbol='square'), textfont=dict(size=F_DATA)))
    fig7.add_trace(go.Scatter(x=years_list, y=m_oper, name='Margen Operativo', mode='lines+markers+text', text=[f"{v:.0%}" for v in m_oper], textposition="top center", 
                              line=dict(color=c_blue_light, width=4), marker=dict(size=12, symbol='square'), textfont=dict(size=F_DATA)))
    fig7.add_trace(go.Scatter(x=years_list, y=m_neto, name='Margen Neto', mode='lines+markers+text', text=[f"{v:.0%}" for v in m_neto], textposition="top center", 
                              line=dict(color=c_ochre, width=4), marker=dict(size=12, symbol='square'), textfont=dict(size=F_DATA)))
    fig7.update_layout(yaxis_tickformat=".2%")
    fig7 = apply_style(fig7, "Márgenes", max_marg)
    figs['Margenes'] = fig7

    # 9. Actividad
    rot_cxc = df_ratios.loc['Rotación CxC (días)'].values.tolist()
    rot_inv = df_ratios.loc['Rotación Inventario (días)'].values.tolist()
    rot_cxp = df_ratios.loc['Rotación CxP (días)'].values.tolist()
    max_act = max(max(rot_cxc), max(rot_inv), max(rot_cxp)) if rot_cxc else 0
    
    fig8 = go.Figure()
    fig8.add_trace(go.Bar(x=years_list, y=rot_cxc, name='Rotación CxC', marker_color=c_blue_dark, text=[f"{v:.0f}" for v in rot_cxc], textposition='auto', textfont=dict(size=F_DATA)))
    fig8.add_trace(go.Bar(x=years_list, y=rot_inv, name='Rotación Inv.', marker_color=c_blue_light, text=[f"{v:.0f}" for v in rot_inv], textposition='auto', textfont=dict(size=F_DATA)))
    fig8.add_trace(go.Bar(x=years_list, y=rot_cxp, name='Rotación CxP', marker_color=c_ochre, text=[f"{v:.0f}" for v in rot_cxp], textposition='auto', textfont=dict(size=F_DATA)))
    fig8 = apply_style(fig8, "Indicadores de Actividad (Días)", max_act)
    figs['Actividad'] = fig8
    
    return figs

def to_excel(df_balance, df_pyg, df_indicadores, df_ratios, figs):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # --- ESTILOS EXCEL ---
        # Cabecera
        header_fmt = workbook.add_format({
            'bold': True, 'text_wrap': True, 'valign': 'vcenter', 'fg_color': '#004c70', 
            'font_color': 'white', 'border': 1, 'align': 'center'
        })
        
        # Subtotales (Azul Claro) - SIN bordes internos
        subtotal_fmt = workbook.add_format({'bold': True, 'bg_color': '#dce6f1', 'align': 'left'})
        subtotal_num_fmt = workbook.add_format({'bold': True, 'bg_color': '#dce6f1', 'num_format': '#,##0'})
        subtotal_pct_fmt = workbook.add_format({'bold': True, 'bg_color': '#dce6f1', 'num_format': '0.0%'})
        
        # Totales Mayores (Azul Oscuro) - SIN bordes internos
        total_fmt = workbook.add_format({'bold': True, 'bg_color': '#004c70', 'font_color': 'white', 'align': 'left'})
        total_num_fmt = workbook.add_format({'bold': True, 'bg_color': '#004c70', 'font_color': 'white', 'num_format': '#,##0'})
        total_pct_fmt = workbook.add_format({'bold': True, 'bg_color': '#004c70', 'font_color': 'white', 'num_format': '0.0%'})
        
        # Texto Normal - SIN bordes
        text_fmt = workbook.add_format({'align': 'left'})
        
        # Formatos Numéricos - SIN bordes
        money_fmt = workbook.add_format({'num_format': '#,##0'})
        int_fmt = workbook.add_format({'num_format': '0'})
        pct_1dec_fmt = workbook.add_format({'num_format': '0.0%'})
        pct_0dec_fmt = workbook.add_format({'num_format': '0%'})
        float_1dec_fmt = workbook.add_format({'num_format': '0.0'})
        note_fmt = workbook.add_format({'italic': True, 'font_color': 'gray', 'align': 'left'})
        
        # --- FORMATOS ESPECIALES PARA BORDE DERECHO (BALANCE) ---
        # Se aplican a la columna D (índice 3 en datos, 4 en excel)
        # Necesitamos versiones con borde derecho de todos los formatos base
        
        def add_right_border(base_format_props):
            props = base_format_props.copy()
            props['right'] = 1
            return workbook.add_format(props)

        # Diccionarios de propiedades para clonar
        p_money = {'num_format': '#,##0'}
        p_subtotal_num = {'bold': True, 'bg_color': '#dce6f1', 'num_format': '#,##0'}
        p_total_num = {'bold': True, 'bg_color': '#004c70', 'font_color': 'white', 'num_format': '#,##0'}
        
        # Formatos con borde derecho
        money_right_fmt = add_right_border(p_money)
        subtotal_num_right_fmt = add_right_border(p_subtotal_num)
        total_num_right_fmt = add_right_border(p_total_num)

        # Listas de filas especiales
        filas_total_mayor = ['ACTIVOS TOTALES', 'PASIVOS TOTALES', 'PATRIMONIO TOTAL', 'TOTAL PASIVO Y PATRIMONIO', 'RESULTADO DEL EJERCICIO']
        filas_subtotal = ['Activo Corriente', 'Activo No Corriente', 'Pasivo Corriente', 'Pasivo No Corriente', 
                          'RESULTADO BRUTO', 'EBITDA', 'RESULTADO OPERATIVO', 'RESULTADO OPERATIVO (EBIT / UAII)', 
                          'RESULTADO ANTES DEL IMPUESTO']

        # --- 1. HOJA BALANCE ---
        sheet_bal = 'Balance_Analizado'
        ws_bal = workbook.add_worksheet(sheet_bal)
        writer.sheets[sheet_bal] = ws_bal
        ws_bal.hide_gridlines(2)
        
        ws_bal.write(0, 0, "Cuenta / Concepto", header_fmt)
        for col_num, value in enumerate(df_balance.columns.values):
            ws_bal.write(0, col_num + 1, value, header_fmt)
            
        for row_num, (index, row) in enumerate(df_balance.iterrows()):
            idx_name = str(index).strip()
            # Determinar tipo de fila
            if any(x in idx_name for x in filas_total_mayor):
                lbl_fmt = total_fmt
                base_num = total_num_fmt
                base_pct = total_pct_fmt
                # Especial para borde derecho
                curr_right_fmt = total_num_right_fmt
            elif any(x in idx_name for x in filas_subtotal):
                lbl_fmt = subtotal_fmt
                base_num = subtotal_num_fmt
                base_pct = subtotal_pct_fmt
                curr_right_fmt = subtotal_num_right_fmt
            else:
                lbl_fmt = text_fmt
                base_num = money_fmt
                base_pct = pct_1dec_fmt
                curr_right_fmt = money_right_fmt

            ws_bal.write(row_num + 1, 0, index, lbl_fmt)
            
            for col_num, (col_name, value) in enumerate(row.items()):
                c_name = str(col_name).lower()
                
                # Determinar si es Columna con Borde Derecho (La 3ra columna de datos, índice 2 en enumerate si A=0? No, pandas items. Col 0=2023, 1=2024, 2=2025)
                # Ojo: df_balance tiene columnas: [2023, 2024, 2025, Vertical%, Horiz$, Horiz%]
                # Queremos borde derecho en '2025' (index 2)
                
                is_right_border_col = (col_num == 2) 

                if '%' in c_name or 'av_' in c_name or 'vertical' in c_name or 'horizontal_%' in c_name:
                    # Si es fila especial, mantener fondo pero cambiar num_format a %
                    final_fmt = workbook.add_format({'num_format': '0.0%'}) # base
                    if lbl_fmt == total_fmt:
                        final_fmt = total_pct_fmt
                    elif lbl_fmt == subtotal_fmt:
                        final_fmt = subtotal_pct_fmt
                    ws_bal.write(row_num + 1, col_num + 1, value, final_fmt)
                else:
                    # Es número normal (dinero)
                    if is_right_border_col:
                        ws_bal.write(row_num + 1, col_num + 1, value, curr_right_fmt)
                    else:
                        ws_bal.write(row_num + 1, col_num + 1, value, base_num)
        
        # Ajuste Anchos
        ws_bal.set_column('A:A', 40)
        ws_bal.set_column('B:D', 14) # Ajuste 1
        ws_bal.set_column('E:G', 10) # Ajuste 2

        # --- 2. HOJA RESULTADOS (NUEVO ORDEN) ---
        sheet_pyg = 'Resultados_e_Indicadores'
        ws_pyg = workbook.add_worksheet(sheet_pyg)
        writer.sheets[sheet_pyg] = ws_pyg
        ws_pyg.hide_gridlines(2)
        
        current_row = 0
        ws_pyg.merge_range(current_row, 0, current_row, len(df_pyg.columns), "ESTADO DE RESULTADOS", header_fmt)
        current_row += 1
        ws_pyg.write(current_row, 0, "Cuenta", header_fmt)
        for col_num, value in enumerate(df_pyg.columns.values):
            ws_pyg.write(current_row, col_num + 1, value, header_fmt)
        
        current_row += 1
        for row_num, (index, row) in enumerate(df_pyg.iterrows()):
            idx_name = str(index).strip()
            if any(x in idx_name for x in filas_total_mayor):
                lbl = total_fmt
                num = total_num_fmt
            elif any(x in idx_name for x in filas_subtotal):
                lbl = subtotal_fmt
                num = subtotal_num_fmt
            else:
                lbl = text_fmt
                num = money_fmt
                
            ws_pyg.write(current_row + row_num, 0, index, lbl)
            for col_num, value in enumerate(row):
                ws_pyg.write(current_row + row_num, col_num + 1, value, num)
        
        current_row += len(df_pyg) + 3
        # CAMBIO NOMBRE TITULO EXCEL
        ws_pyg.merge_range(current_row, 0, current_row, len(df_indicadores.columns), "Análisis combinado (vertical/horizontal) P&G", header_fmt)
        current_row += 1
        ws_pyg.write(current_row, 0, "Indicador", header_fmt)
        for col_num, value in enumerate(df_indicadores.columns.values):
            ws_pyg.write(current_row, col_num + 1, value, header_fmt)
            
        current_row += 1
        for row_num, (index, row) in enumerate(df_indicadores.iterrows()):
            ws_pyg.write(current_row + row_num, 0, index, text_fmt)
            for col_num, value in enumerate(row):
                ws_pyg.write(current_row + row_num, col_num + 1, value, pct_1dec_fmt)
                
        # Ajuste Anchos
        ws_pyg.set_column('A:A', 40)
        ws_pyg.set_column('B:D', 14) # Ajuste 1

        # --- 3. HOJA RATIOS (NUEVO ORDEN Y FORMATOS) ---
        sheet_ratios = 'Ratios_Financieros'
        ws_rat = workbook.add_worksheet(sheet_ratios)
        writer.sheets[sheet_ratios] = ws_rat
        ws_rat.hide_gridlines(2)
        
        ws_rat.write(0, 0, "Ratio Financiero", header_fmt)
        for col_num, value in enumerate(df_ratios.columns.values):
            ws_rat.write(0, col_num + 1, value, header_fmt)
            
        for row_num, (index, row) in enumerate(df_ratios.iterrows()):
            ws_rat.write(row_num + 1, 0, index, text_fmt)
            idx_str = str(index).lower()
            if 'días' in idx_str or 'dias' in idx_str:
                row_fmt = int_fmt
            elif any(x in idx_str for x in ['margen', 'roa', 'roe']):
                row_fmt = pct_0dec_fmt
            else:
                row_fmt = float_1dec_fmt
            
            for col_num, value in enumerate(row):
                ws_rat.write(row_num + 1, col_num + 1, value, row_fmt)
        
        # --- TABLA DE FÓRMULAS (LATERAL G1) ---
        # Ajuste 4: Empezar en G1 (Columna 6, Fila 0)
        start_col_form = 6 
        ws_rat.merge_range(0, start_col_form, 0, start_col_form + 1, "GUÍA DE FÓRMULAS UTILIZADAS", header_fmt)
        
        formulas = [
            ("Liquidez Corriente", "Activo Corriente / Pasivo Corriente"),
            ("Prueba Ácida", "(Activo Corriente - Inventario) / Pasivo Corriente"),
            ("Liquidez Inmediata", "Caja y Bancos / Pasivo Corriente"),
            ("Rotación CxC (días)", "Cuentas por Cobrar / (Ventas / 365)"),
            ("Rotación Inventario (días)", "Inventario / (Costo de Ventas / 365)"),
            ("Rotación CxP (días)", "Cuentas por Pagar / (Costo de Ventas / 365)"),
            ("Rotación de Activos", "Ventas / Activos Totales"),
            ("Endeudamiento", "Pasivos Totales / Activos Totales"),
            ("Multiplicador de Capital", "Activos Totales / Patrimonio"),
            ("Razón Deuda/Capital", "Pasivos Totales / Patrimonio"),
            ("ROA", "Utilidad Neta / Activos Totales"),
            ("ROE", "Utilidad Neta / Patrimonio"),
            ("Márgenes (Bruto/Op/Neto)", "Resultado Correspondiente / Ventas")
        ]
        
        ws_rat.write(1, start_col_form, "Ratio", header_fmt)
        ws_rat.write(1, start_col_form + 1, "Fórmula de Cálculo", header_fmt)
        
        for i, (ratio, formula) in enumerate(formulas):
            ws_rat.write(2 + i, start_col_form, ratio, text_fmt)
            ws_rat.write(2 + i, start_col_form + 1, formula, text_fmt)

        # Ajuste Anchos
        ws_rat.set_column('A:A', 40)
        ws_rat.set_column('B:D', 10) # Ajuste 3
        ws_rat.set_column('E:F', 2) # Espacio vacio
        ws_rat.set_column('G:G', 30) # Formula Ratio Name
        ws_rat.set_column('H:H', 50) # Formula Text

        # --- 4. HOJA DASHBOARD ---
        ws_dash = workbook.add_worksheet("DASHBOARD_GRAFICO")
        ws_dash.hide_gridlines(2)
        ws_dash.merge_range('A1:J1', "DASHBOARD EJECUTIVO - GRÁFICOS", header_fmt)
        ws_dash.merge_range('A2:J2', "Nota: Estos gráficos son imágenes estáticas. Puede seleccionarlos, moverlos y cambiar su tamaño a gusto.", note_fmt)
        
        row_pos = 3
        for key, fig in figs.items():
            img_bytes = fig.to_image(format="png", width=1000, height=500, scale=2)
            image_data = io.BytesIO(img_bytes)
            # ESCALA REDUCIDA A LA MITAD (0.4)
            ws_dash.insert_image(row_pos, 1, key, {'image_data': image_data, 'x_scale': 0.4, 'y_scale': 0.4})
            row_pos += 18 

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
    
    # --- BOTÓN HOTMART ---
    st.markdown("""
    <a href="https://pay.hotmart.com/J94144104S?off=37odx5m2&checkoutMode=10&offDiscount=JEMP25&src=appeeff" target="_blank" class="custom-btn">
        <div style="
            width: 100%;
            background-color: #004c70; 
            color: white; 
            padding: 15px; 
            text-align: center; 
            border-radius: 10px; 
            font-weight: bold; 
            font-size: 16px;
            box-shadow: 0px 3px 5px rgba(0,0,0,0.2);
            margin-bottom: 10px;
        ">
            🔥 Curso Especializado: Análisis y Proyección de EEFF - 25% OFF
        </div>
    </a>
    """, unsafe_allow_html=True)
    
    # --- BOTÓN BENTO ---
    st.markdown("""
    <a href="https://bento.me/oscar-menacho-consultor-financiero" target="_blank" class="custom-btn">
        <div style="
            width: 100%;
            background-color: #ed7d31; 
            color: white; 
            padding: 15px; 
            text-align: center; 
            border-radius: 10px; 
            font-weight: bold; 
            font-size: 16px;
            box-shadow: 0px 3px 5px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        ">
            🎯 Mas cursos y mis servicios
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
            for col in df.columns: 
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.fillna(0, inplace=True)
        
        df_bal_an = procesar_balance(df_bal)
        df_ind_pyg = procesar_pyg(df_pyg_orig)
        df_ratios = calcular_ratios(df_bal, df_pyg_orig)
        
        figs = crear_figuras_dashboard(df_bal, df_pyg_orig, df_ind_pyg, df_ratios)
        excel_data = to_excel(df_bal_an, df_pyg_orig, df_ind_pyg, df_ratios, figs)

        # --- SECCIÓN DE PESTAÑAS (RESTAURADO V53) ---
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Balance", "📈 P&G", "🔢 Ratios", "🖼️ DASHBOARD"])

        with tab1:
            st.header("Análisis del Balance General")
            st.dataframe(aplicar_estilos_df(df_bal_an, 'balance'), use_container_width=True)
            
        with tab2:
            st.header("Análisis del Estado de Resultados (P&G)")
            st.subheader("Estado de Resultados Original")
            st.dataframe(aplicar_estilos_df(df_pyg_orig, 'balance'), use_container_width=True)
            st.divider()
            # CAMBIO NOMBRE SUBTITULO INTERFAZ
            st.subheader("Análisis combinado (vertical/horizontal) P&G")
            st.dataframe(aplicar_estilos_df(df_ind_pyg, 'indicadores'), use_container_width=True)
            
        with tab3:
            st.header("Ratios Financieros Clave")
            st.dataframe(aplicar_estilos_df(df_ratios, 'ratios'), use_container_width=True)
            
        with tab4:
            st.header("Dashboard Gráfico Interactivo")
            
            # FILA 1
            col1, _, col2 = st.columns([1, 0.1, 1])
            with col1: st.plotly_chart(figs['Estructura'], use_container_width=True)
            with col2: st.plotly_chart(figs['Cascada'], use_container_width=True)
            st.divider()
            
            # FILA 2
            col3, _, col4 = st.columns([1, 0.1, 1])
            with col3: st.plotly_chart(figs['Ventas'], use_container_width=True)
            with col4: st.plotly_chart(figs['CapitalTrabajo'], use_container_width=True)
            st.divider()

            # FILA 3 (Grandes Grupos - FULL WIDTH)
            st.plotly_chart(figs['GrandesGrupos'], use_container_width=True)
            st.divider()
            
            # FILA 4
            col5, _, col6 = st.columns([1, 0.1, 1])
            with col5: st.plotly_chart(figs['Liquidez'], use_container_width=True)
            with col6: st.plotly_chart(figs['Rentabilidad'], use_container_width=True)
            st.divider()
            
            # FILA 5
            col7, _, col8 = st.columns([1, 0.1, 1])
            with col7: st.plotly_chart(figs['Margenes'], use_container_width=True)
            with col8: st.plotly_chart(figs['Actividad'], use_container_width=True)
        
        # --- BOTÓN DE DESCARGA PRINCIPAL ---
        st.divider()
        st.write("### 📥 Descarga tu Informe")
        st.download_button(
            label="DESCARGAR REPORTE EXCEL COMPLETO",
            data=excel_data,
            file_name=f"Reporte_Financiero_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary", 
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Error técnico: {e}")
else:
    st.info("""
    👋 ¡Hola! Para usar esta App, primero descarga la plantilla en el panel lateral, complétala y súbela.
    
    Después, ¡Descarga tu Reporte en Excel totalmente gratis! 🚀
    """)

# --- FEEDBACK FORM (V52/V53) ---
st.divider()
with st.expander("💬 ¿Tienes comentarios o sugerencias? Escríbenos aquí"):
    feedback = st.text_area("Tu opinión nos ayuda a mejorar:", placeholder="Escribe aquí...")
    if feedback:
        body_email = feedback.replace('\n', '%0A')
        body_encoded = urllib.parse.quote(body_email)
        # BOTON NEGRO COMO PEDISTE
        st.markdown(f'''
            <a href="mailto:oscar.david.menacho@gmail.com?subject=Feedback%20App%20Financiera&body={body_encoded}" target="_blank" class="custom-btn">
                <div style="
                    background-color: #333333; 
                    color: white; 
                    padding: 10px; 
                    text-align: center; 
                    border-radius: 5px; 
                    font-weight: bold;
                    margin-top: 10px;
                    width: 200px;
                ">
                    ✉️ Enviar Comentario
                </div>
            </a>
        ''', unsafe_allow_html=True)
