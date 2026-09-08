import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import yfinance as yf
import requests
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración de la página
st.set_page_config(page_title="Market Breadth Analyzer", layout="wide")

# ==========================================
# PASO 1: SIDEBAR (Selección de Mercado)
# ==========================================
with st.sidebar:
    st.title("⚙️ Configuración")
    st.markdown("### Selección de Índice")
    ticker = st.selectbox("Mercado a analizar:", ["QQQ","SPY","^STOXX50E","XXSC.MI], index=0)
    
    st.markdown("---")
    st.info("💡 *Los datos se cargan en caché. El primer cambio de ticker tardará unos segundos.*")

# Diccionario de URLs de BlackRock según el ticker
BLACKROCK_URLS = {
    'QQQ': 'https://www.blackrock.com/es/profesionales/productos/251896/fund/1497267045693.ajax?tab=all&fileType=json&asOfDate=20251205',
    'SPY': 'https://www.blackrock.com/es/profesionales/productos/253743/fund/1497267045693.ajax?tab=all&fileType=json&asOfDate=20251205',
    '^STOXX50E': 'https://www.blackrock.com/es/profesionales/productos/251929/fund/1497267045693.ajax?tab=all&fileType=json&asOfDate=20251205',
    'XXSC.MI':'https://www.blackrock.com/es/profesionales/productos/348766/fund/1497267045693.ajax?tab=all&fileType=json&asOfDate=20251022',
    
}

# ==========================================
# PASO 2: DESCARGA DE DATOS (Ventana 2)
# ==========================================
@st.cache_data(show_spinner="Descargando datos del ETF y sus componentes...")
def load_market_data(ticker_selected):
    # 1. Descargar datos del ETF principal
    stock = yf.download(
        ticker_selected, start='2020-01-01',
        interval='1d', progress=False, auto_adjust=True
    )
    if isinstance(stock.columns, pd.MultiIndex):
        stock.columns = stock.columns.get_level_values(0)

    # 2. Descargar componentes desde BlackRock
    url = BLACKROCK_URLS[ticker_selected]
    response = json.loads(requests.get(url).content.decode('utf-8-sig'))
    tickers_list = [entry[0] for entry in response['aaData'] if entry[3] == 'Equity']
    
    # 3. Descargar cierres de los componentes
    df_constituents = yf.download(tickers_list, start='2020-01-01', progress=False)['Close']
    
    return stock, df_constituents

# ==========================================
# PASO 3: CÁLCULO DE INDICADORES (Ventana 3)
# ==========================================
@st.cache_data(show_spinner="Calculando indicadores de amplitud...")
def calculate_indicators(df):
    # NHNL (New Highs / New Lows)
    df52wh = df.rolling(window=252).max()
    df52wl = df.rolling(window=252).min()
    
    new_highs = (df == df52wh).sum(axis=1)
    new_lows = (df == df52wl).sum(axis=1)
    SPNHNLCLOSE = pd.DataFrame({'NewHigh': new_highs, 'NewLow': new_lows}, index=df.index)

    # Advance / Decline y Osciladores
    returns = df.pct_change()[1:] * 100
    advances = (returns > 0).sum(axis=1)
    declines = (returns < 0).sum(axis=1)
    
    SPADVDEC = pd.DataFrame({'Avance': advances, 'Descenso': declines}, index=returns.index)
    SPADVDEC['LIN'] = (SPADVDEC['Avance'] - SPADVDEC['Descenso']).cumsum()
    SPADVDEC['RATIO'] = (SPADVDEC['Avance'] - SPADVDEC['Descenso']) / (SPADVDEC['Avance'] + SPADVDEC['Descenso'])
    SPADVDEC['LINADN'] = SPADVDEC['RATIO'].cumsum()
    
    # Fórmulas normalizadas
    roll_min = SPADVDEC['LINADN'].rolling(window=21).min()
    roll_max = SPADVDEC['LINADN'].rolling(window=21).max()
    SPADVDEC['FORMULA'] = ((SPADVDEC['LINADN'] - roll_min) / (roll_max - roll_min)) * 100
    SPADVDEC['ADn'] = SPADVDEC['FORMULA'].ewm(span=7).mean()
    
    # McClellan y SUMM
    SPADVDEC['MIMACD'] = SPADVDEC['LIN'].ewm(span=3).mean() - SPADVDEC['LIN'].ewm(span=8).mean()
    SPADVDEC['EMAMIMACD'] = SPADVDEC['MIMACD'].ewm(span=2, adjust=False).mean()
    SPADVDEC['SUMM_SHORT'] = (SPADVDEC['Avance'] - SPADVDEC['Descenso']).ewm(span=19).mean()
    SPADVDEC['SUMM_LONG'] = (SPADVDEC['Avance'] - SPADVDEC['Descenso']).ewm(span=39).mean()
    SPADVDEC['SUMM'] = (SPADVDEC['SUMM_SHORT'] - SPADVDEC['SUMM_LONG']).cumsum()
    
    SPADVDEC['AvanceEMA19'] = SPADVDEC['Avance'].ewm(span=19).mean()
    SPADVDEC['AvanceEMA39'] = SPADVDEC['Avance'].ewm(span=39).mean()
    SPADVDEC['McCellan'] = SPADVDEC['AvanceEMA19'] - SPADVDEC['AvanceEMA39']
    
    SPADVDEC['longMIMACD'] = SPADVDEC['LIN'].ewm(span=12).mean() - SPADVDEC['LIN'].ewm(span=26).mean()
    SPADVDEC['longEMAMIMACD'] = SPADVDEC['longMIMACD'].ewm(span=9).mean()

    return SPNHNLCLOSE, SPADVDEC

# Ejecución de carga de datos
STOCK, df_constituents = load_market_data(ticker)
SPNHNLCLOSE, SPADVDEC = calculate_indicators(df_constituents)

# ==========================================
# PASO 4: INTERFAZ DE PANELES Y SELECTORES
# ==========================================
# NUEVA ESTRUCTURA: Grupos de indicadores que se dibujan juntos en el mismo panel
INDICATOR_GROUPS = {
    'MIMACD + Señal (Corto)': [
        {'df': 'SPADVDEC', 'col': 'MIMACD', 'color': '#2962FF', 'name': 'MIMACD'},
        {'df': 'SPADVDEC', 'col': 'EMAMIMACD', 'color': '#FF6D00', 'name': 'Señal MIMACD'}
    ],
    'McClellan (Corto)': [
        {'df': 'SPADVDEC', 'col': 'McCellan', 'color': '#2962FF', 'name': 'McClellan'}
    ],
    'SUMM (Medio Plazo)': [
        {'df': 'SPADVDEC', 'col': 'SUMM', 'color': '#9C27B0', 'name': 'SUMM'}
    ],
    'longMIMACD (Medio)': [
        {'df': 'SPADVDEC', 'col': 'longMIMACD', 'color': '#2962FF', 'name': 'longMIMACD'},
        {'df': 'SPADVDEC', 'col': 'longEMAMIMACD', 'color': '#FF6D00', 'name': 'Señal longMIMACD'}
    ],
    'New Highs + New Lows': [
        {'df': 'SPNHNLCLOSE', 'col': 'NewHigh', 'color': '#00C853', 'name': 'New Highs'},
        {'df': 'SPNHNLCLOSE', 'col': 'NewLow', 'color': '#D50000', 'name': 'New Lows'}
    ],
    'Avances + Descensos': [
        {'df': 'SPADVDEC', 'col': 'Avance', 'color': '#00C853', 'name': 'Avances'},
        {'df': 'SPADVDEC', 'col': 'Descenso', 'color': '#D50000', 'name': 'Descensos'}
    ],
    'ADn': [
        {'df': 'SPADVDEC', 'col': 'ADn', 'color': '#2962FF', 'name': 'ADn'}
    ]
}

# Layout: Columna izquierda para el gráfico, derecha para controles
col_chart, col_controls = st.columns([4, 1])

with col_controls:
    st.subheader("Config. Gráfico")
    num_panels = st.selectbox("Nº de paneles inferiores:", [1, 2, 3], index=1)
    
    group_names = list(INDICATOR_GROUPS.keys())
    
    # Generar desplegables dinámicos según el número de paneles
    selected_indicators = []
    for i in range(num_panels):
        # Asignamos defaults lógicos: Panel 1 -> MIMACD, Panel 2 -> NHNL, Panel 3 -> ADn
        default_options = ['ADn','MIMACD + Señal (Corto)', 'New Highs + New Lows']
        default_idx = group_names.index(default_options[i]) if i < len(default_options) else 0
        
        ind = st.selectbox(f"Panel {i+1}:", group_names, index=default_idx, key=f"ind_{i}")
        selected_indicators.append(ind)

# ==========================================
# PASO 5: GRÁFICO DINÁMICO PLOTLY
# ==========================================
with col_chart:
    # Calcular filas y alturas dinámicamente
    total_rows = 1 + num_panels
    row_heights = [0.6] + [0.4 / num_panels] * num_panels
    
    specs = [[{"secondary_y": True}]] + [[{"secondary_y": False}]] * num_panels

    fig = make_subplots(
        rows=total_rows, cols=1, shared_xaxes=True, 
        vertical_spacing=0.03, row_heights=row_heights, specs=specs
    )

    # 1. Panel Principal (Velas)
    fig.add_trace(go.Candlestick(
        x=STOCK.index, open=STOCK['Open'], high=STOCK['High'], 
        low=STOCK['Low'], close=STOCK['Close'], name=ticker,
        increasing_line_color='#26A69A', decreasing_line_color='#EF5350'
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=STOCK.index, y=STOCK['Volume'], name='Volumen', opacity=0.2, marker_color='gray'
    ), row=1, col=1, secondary_y=True)

    # 2. Paneles de Indicadores (Iterando sobre los GRUPOS)
    for i, group_name in enumerate(selected_indicators):
        row_num = i + 2
        traces_in_group = INDICATOR_GROUPS[group_name]
        
        # Dibujar cada traza dentro del grupo en el mismo panel (row_num)
        for trace_info in traces_in_group:
            df_name = trace_info['df']
            col_name = trace_info['col']
            color = trace_info['color']
            name = trace_info['name']
            
            data_source = SPADVDEC if df_name == 'SPADVDEC' else SPNHNLCLOSE
            
            fig.add_trace(go.Scatter(
                x=data_source.index, y=data_source[col_name], 
                line=dict(color=color, width=1.5), name=name
            ), row=row_num, col=1)

    # 3. Layout general
    fig.update_layout(
        title=f"Análisis de Amplitud: {ticker}", 
        xaxis_rangeslider_visible=False, 
        height=850,
        hovermode="x",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Precio", secondary_y=False, row=1, col=1)
    fig.update_yaxes(title_text="Volumen", secondary_y=True, row=1, col=1, showgrid=False)
    
    # Poner el nombre del grupo como título del eje Y
    for i in range(num_panels):
        fig.update_yaxes(title_text=selected_indicators[i], row=i+2, col=1)
        
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    # ==========================================
    # PASO 6: INYECCIÓN DE JAVASCRIPT (Estilo TradingView)
    # ==========================================
    js_code = """
    <script>
    function _tv() { 
        var gd = document.getElementsByClassName('plotly-graph-div')[0]; 
        if(!gd) return setTimeout(_tv, 200); 
        var lbl = document.createElement('div'); 
        Object.assign(lbl.style, {
            position:'absolute', right:'45px', backgroundColor:'#2a2e39', 
            color:'#d1d4dc', padding:'3px 6px', borderRadius:'3px', 
            fontSize:'11px', zIndex:'1000', display:'none', pointerEvents:'none'
        }); 
        gd.appendChild(lbl); 
        gd.addEventListener('mousemove', (e) => { 
            var ya = gd._fullLayout.yaxis, box = gd.getBoundingClientRect(), 
            yM = e.clientY - box.top, p = ya.p2c(yM - gd._fullLayout.margin.t); 
            if (p >= ya.range[0] && p <= ya.range[1]) { 
                lbl.style.top = (yM - 10) + 'px'; 
                lbl.innerHTML = p.toFixed(2); 
                lbl.style.display = 'block'; 
            } else lbl.style.display = 'none'; 
        }); 
        gd.addEventListener('mouseleave', () => lbl.style.display = 'none'); 
    } 
    setTimeout(_tv, 500);
    </script>
    """
    
    # Renderizamos el HTML de Plotly y le añadimos nuestro script
    html_str = fig.to_html(
        include_plotlyjs=True, 
        config={
            'displayModeBar': True, 
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawrect', 'eraseshape'],
            'displaylogo': False
        }
    )
    components.html(html_str + js_code, height=870)
