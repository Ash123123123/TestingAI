import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import ta
import pyotp
import requests
from SmartApi import SmartConnect
from datetime import datetime, timedelta
import pytz  

# ---------------------------------------------------------
# Page Configurations
# ---------------------------------------------------------
st.set_page_config(page_title="Angel One XGBoost Predictor", layout="wide")
st.title("🇮🇳 Angel One Intraday XGBoost Dashboard")
st.write("This dashboard connects directly to Angel One, fetches real-time historical "
         "intervals, trains an XGBoost model, and predicts upcoming price trends.")

# ---------------------------------------------------------
# Step 1: Securely Load Background Credentials
# ---------------------------------------------------------
try:
    api_key = st.secrets["angel_one"]["api_key"]
    client_id = st.secrets["angel_one"]["client_id"]
    password = st.secrets["angel_one"]["password"]
    totp_key = st.secrets["angel_one"]["totp_key"]
except FileNotFoundError:
    st.error("❌ Missing secrets.toml file! Please ensure `.streamlit/secrets.toml` "
             "exists in your directory structure.")
    st.stop()
except KeyError as e:
    st.error(f"❌ Missing expected credential key in secrets.toml: {e}")
    st.stop()

# ---------------------------------------------------------
# Step 2: Sidebar Configuration Controls
# ---------------------------------------------------------
st.sidebar.header("📊 Strategy Settings")

exchange_input = st.sidebar.selectbox("Exchange", ["NSE", "NFO", "MCX"], index=0)

ticker_input = st.sidebar.text_input(
    "Trading Symbol (e.g., RELIANCE-EQ, NIFTY26JUNFUT)", 
    value="Nifty 50"
)

interval = st.sidebar.selectbox(
    "Candlestick Timeframe", 
    ["FIVE_MINUTE", "FIFTEEN_MINUTE", "ONE_HOUR"], 
    index=0
)
predict_ahead = st.sidebar.slider("Prediction Horizon (Candles Ahead)", min_value=1, max_value=5, value=3)

st.sidebar.header("🧠 AI Training Parameters")
st.sidebar.write("More days = more data, but takes longer to train.")
training_days = st.sidebar.slider("Historical Training Days", min_value=10, max_value=90, value=30, step=5)

# --- ANGEL ONE API SAFETY CAPS ---
limit_msg = ""
if interval == "FIVE_MINUTE" and training_days > 30:
    training_days = 30
    limit_msg = "⚠️ Angel One limits 5-Minute data to 30 days. Auto-adjusted to prevent API failure."
elif interval == "FIFTEEN_MINUTE" and training_days > 60:
    training_days = 60
    limit_msg = "⚠️ Angel One limits 15-Minute data to 60 days. Auto-adjusted to prevent API failure."

if limit_msg:
    st.sidebar.warning(limit_msg)

st.sidebar.header("🛡️ Risk Management (ATR Based)")
sl_multiplier = st.sidebar.slider("Stop Loss (ATR Multiplier)", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
tp1_multiplier = st.sidebar.slider("Target 1 (ATR Multiplier)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
tp2_multiplier = st.sidebar.slider("Target 2 (ATR Multiplier)", min_value=1.0, max_value=5.0, value=2.5, step=0.1)

@st.cache_data
def load_scrip_master():
    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    response = requests.get(url).json()
    return pd.DataFrame(response)

# ---------------------------------------------------------
# Step 3: Main Execution Flow
# ---------------------------------------------------------
if st.sidebar.button("Run Live Predictive Analytics"):
    with st.spinner(f"Pulling {training_days} days of data & training optimized AI..."):
        try:
            scrip_df = load_scrip_master()
            
            search_ticker = ticker_input.upper()
            if search_ticker == "NIFTY":
                search_ticker = "NIFTY 50"
            elif search_ticker == "BANKNIFTY":
                search_ticker = "NIFTY BANK"
                
            token_row = scrip_df[(scrip_df['symbol'].str.upper() == search_ticker) & (scrip_df['exch_seg'] == exchange_input)]
            
            if token_row.empty:
                st.error(f"Symbol '{ticker_input}' could not be matched on the {exchange_input} exchange. "
                         f"Verify the naming syntax (e.g., -EQ for NSE, Expiry format for NFO/MCX).")
                st.stop()
            
            symbol_token = token_row.iloc[0]['token']
            st.info(f"Connected to Token Mapping Reference ID: {symbol_token} on {exchange_input}")

            smart_conn = SmartConnect(api_key=api_key)
            totp_token = pyotp.TOTP(totp_key).now()
            session_data = smart_conn.generateSession(client_id, password, totp_token)
            
            if not session_data.get('status'):
                st.error(f"Authentication Failure: {session_data.get('message')}")
                st.stop()
            
            ist = pytz.timezone('Asia/Kolkata')
            current_ist_time = datetime.now(ist)
            
            to_date = current_ist_time.strftime("%Y-%m-%d %H:%M")
            from_date = (current_ist_time - timedelta(days=training_days)).strftime("%Y-%m-%d %H:%M")
            
            candle_params = {
                "exchange": exchange_input,
                "symboltoken": str(symbol_token),
                "interval": interval,
                "fromdate": from_date,
                "todate": to_date
            }
            
            history = smart_conn.getCandleData(candle_params)
            
            if history.get('status') and history.get('data'):
                raw_data = history['data']
                df = pd.DataFrame(raw_data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = pd.to_numeric(df[col])
                
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                df.set_index('Timestamp', inplace=True)
                
                df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
                macd = ta.trend.MACD(close=df['Close'])
                df['MACD'] = macd.macd()
                df['MACD_Signal'] = macd.macd_signal()
                df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
                df['SMA_9'] = ta.trend.SMAIndicator(close=df['Close'], window=9).sma_indicator()
                df['SMA_21'] = ta.trend.SMAIndicator(close=df['Close'], window=21).sma_indicator()
                df['MA_Diff'] = df['SMA_9'] - df['SMA_21']
                
                df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
                df['ADX'] = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14).adx()
                bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
                df['BB_Width'] = bb.bollinger_wband()
                df['Stoch_RSI'] = ta.momentum.StochRSIIndicator(close=df['Close'], window=14).stochrsi()
                
                df.dropna(inplace=True)
                
                # =====================================================================
                # OPTIMIZED STEP 3E: Volatility-Adjusted Target Labeling
                # =====================================================================
                df['Future_Close'] = df['Close'].shift(-predict_ahead)
                df['Price_Change'] = df['Future_Close'] - df['Close']
                df['Volatility_Threshold'] = df['ATR'] * 0.5

                # 2 = Bullish, 1 = Bearish, 0 = Neutral/Choppy No-Trade Zone
                df['Target'] = np.where(df['Price_Change'] > df['Volatility_Threshold'], 2,
                               np.where(df['Price_Change'] < -df['Volatility_Threshold'], 1, 0))

                live_row = df.iloc[[-1]].copy()
                df.dropna(inplace=True)

                features = ['RSI', 'MACD', 'MACD_Signal', 'ATR', 'MA_Diff', 'OBV', 'ADX', 'BB_Width', 'Stoch_RSI', 'Volume']
                X = df[features]
                y = df['Target']

                # =====================================================================
                # OPTIMIZED STEP 3F: Ultra-Conservative Regularized XGBoost 
                # =====================================================================
                model = xgb.XGBClassifier(
                    n_estimators=180,
                    max_depth=3,                  
                    learning_rate=0.02,           
                    subsample=0.65,               
                    colsample_bytree=0.65,        
                    min_child_weight=10,          
                    objective='multi:softprob',   
                    random_state=42
                )
                model.fit(X, y)
                
                # G. Make Prediction Metrics
                X_live = live_row[features]
                prediction = model.predict(X_live)[0]
                probabilities = model.predict_proba(X_live)[0]
                
                ltp = live_row['Close'].values[0]
                latest_atr = live_row['ATR'].values[0]
                
                # Multi-Class Target Handling
                if prediction == 2: # BULLISH
                    signal_text = "🟢 BULLISH (BUY)"
                    confidence_metric = probabilities[2]
                    sl_price = f"₹{(ltp - (latest_atr * sl_multiplier)):.2f}"
                    tp1_price = f"₹{(ltp + (latest_atr * tp1_multiplier)):.2f}"
                    tp2_price = f"₹{(ltp + (latest_atr * tp2_multiplier)):.2f}"
                elif prediction == 1: # BEARISH
                    signal_text = "🔴 BEARISH (SELL)"
                    confidence_metric = probabilities[1]
                    sl_price = f"₹{(ltp + (latest_atr * sl_multiplier)):.2f}"
                    tp1_price = f"₹{(ltp - (latest_atr * tp1_multiplier)):.2f}"
                    tp2_price = f"₹{(ltp - (latest_atr * tp2_multiplier)):.2f}"
                else: # NEUTRAL / CHOPPY
                    signal_text = "⚪ NEUTRAL (NO TRADE)"
                    confidence_metric = probabilities[0]
                    sl_price = "N/A"
                    tp1_price = "N/A"
                    tp2_price = "N/A"

                # ---------------------------------------------------------
                # Step 4: Streamlit UI Component Output
                # ---------------------------------------------------------
                st.success("Analysis Complete! Live predictions generated successfully.")
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric(label="Last Traded Price (LTP)", value=f"₹{ltp:.2f}")
                with metric_col2:
                    st.metric(label="XGBoost Algorithmic Signal", value=signal_text)
                with metric_col3:
                    st.metric(label="Model Predictive Confidence", value=f"{confidence_metric * 100:.2f}%")
                
                st.divider()
                
                st.subheader("🎯 AI Trade Setup Targets")
                st.write(f"Based on current market volatility (ATR: ₹{latest_atr:.2f})")
                
                setup_col1, setup_col2, setup_col3 = st.columns(3)
                with setup_col1:
                    st.metric(label="🛑 Stop Loss (SL)", value=sl_price)
                with setup_col2:
                    st.metric(label="🎯 Target 1 (TP1)", value=tp1_price)
                with setup_col3:
                    st.metric(label="🚀 Target 2 (TP2)", value=tp2_price)

                st.divider()
                
                st.subheader("Live Vector Attributes Table (Expanded)")
                st.dataframe(X_live.style.format("{:.4f}"))
                
                st.subheader("Historical Trajectory Visualization")
                st.line_chart(df['Close'].tail(75))
                
                smart_conn.terminateSession(client_id)
            else:
                st.error(f"Failed to fetch data payload from backend node: {history.get('message')} (Data payload was empty. Check if the market was closed or if the expiry date is valid).")
                
        except Exception as error:
            st.error(f"An exception crashed the background runtime: {error}")
