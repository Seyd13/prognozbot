import asyncio
import logging
from datetime import datetime, timedelta, timezone
from io import BytesIO

import aiohttp
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import BufferedInputFile, ReplyKeyboardMarkup, KeyboardButton
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from zoneinfo import ZoneInfo

# --- CONFIGURATION ---
TELEGRAM_TOKEN = '2122435147:AAG_52ELCHjFnXNxcAP4i5xNAal9I91xNTM'

# TIMEZONE SETTINGS
TIMEZONE_STR = 'Europe/Moscow'
LOCAL_TIMEZONE = ZoneInfo(TIMEZONE_STR)

STARTING_BALANCE = 100

logging.basicConfig(level=logging.INFO)

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# --- IN-MEMORY DATABASE ---
user_limits = defaultdict(lambda: {'balance': STARTING_BALANCE, 'last_prediction_time': None})

# --- DATA FUNCTIONS ---

async def get_market_data():
    """Fetch data from CoinGecko API with 1-minute timeframe."""
    # days=0.1 provides 1-minute candles
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=0.1'
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status == 200:
                    data = await response.json()
                    prices = data.get('prices', []) 
                    if not prices:
                        return None

                    df = pd.DataFrame(prices, columns=['timestamp', 'close'])
                    # Convert time: ms -> UTC -> Local
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(LOCAL_TIMEZONE)
                    
                    df = df.rename(columns={'timestamp': 'close_time'})
                    # Take last 50 points (50 minutes)
                    df = df.tail(50).reset_index(drop=True)
                    return df
                else:
                    logging.error(f'CoinGecko HTTP Error: {response.status}')
                    return None
    except Exception as e:
        logging.error(f'CoinGecko connection error: {e}')
        return None

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def predict_next_minute(df):
    df = df.copy()
    df['rsi'] = calculate_rsi(df['close'])
    df['change'] = df['close'].diff()
    df.dropna(inplace=True)

    if len(df) < 15:
        return None, None, None

    data = df[['close', 'rsi', 'change']].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    look_back = 10
    
    if len(scaled_data) <= look_back:
        return None, None, None

    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:i + look_back].flatten()) 
        y.append(scaled_data[i + look_back][0])

    if not X:
        return None, None, None

    X = np.array(X)
    y = np.array(y)

    model = MLPRegressor(hidden_layer_sizes=(10, 5), max_iter=500, random_state=42)
    try:
        model.fit(X, y)
    except Exception as e:
        logging.error(f'Model training error: {e}')
        return None, None, None

    last_window = scaled_data[-look_back:].flatten().reshape(1, -1)
    predicted_scaled = model.predict(last_window)
    
    dummy_array = np.zeros((1, 3))
    dummy_array[0, 0] = predicted_scaled[0]
    dummy_array[0, 1] = scaled_data[-1, 1] 
    dummy_array[0, 2] = scaled_data[-1, 2] 
    
    predicted_price_full = scaler.inverse_transform(dummy_array)
    predicted_price = predicted_price_full[0, 0]

    # --- TIME LOGIC (Real Time) ---
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(LOCAL_TIMEZONE)
    
    # Prediction for next minute
    next_time = now_local.replace(second=0, microsecond=0) + timedelta(minutes=1)

    return df, predicted_price, next_time

def create_plot(df, predicted_price, next_time):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot last 20 minutes
    plot_df = df.tail(20).copy()
    
    # Remove timezone for plotting
    plot_df['close_time_plot'] = plot_df['close_time'].dt.tz_localize(None)
    
    # Handle next_time
    if next_time.tzinfo is not None:
        next_time_plot = next_time.replace(tzinfo=None)
    else:
        next_time_plot = next_time

    ax.plot(plot_df['close_time_plot'], plot_df['close'], 
            label='History', color='cyan', marker='o', linestyle='-')

    ax.plot([plot_df['close_time_plot'].iloc[-1], next_time_plot],
            [plot_df['close'].iloc[-1], predicted_price],
            label='AI Prediction', color='lime', linestyle='--', marker='x')
    
    ax.scatter(next_time_plot, predicted_price, color='lime', s=100, zorder=5)

    for x, y in zip(plot_df['close_time_plot'], plot_df['close']):
        label = f'{y:.0f}'
        ax.annotate(label, (x, y), textcoords='offset points', xytext=(0,10), ha='center', fontsize=8, color='white')

    ax.annotate(f'AI: {predicted_price:.0f}', 
                (next_time_plot, predicted_price), textcoords='offset points', 
                xytext=(0,10), ha='center', fontsize=9, color='lime', fontweight='bold')

    ax.set_title(f'BTC/USDT AI Prediction ({TIMEZONE_STR})', color='white', fontsize=14)
    ax.set_xlabel('Time', color='gray')
    ax.set_ylabel('Price ($)', color='gray')
    
    ax.grid(True, color='gray', linestyle=':', alpha=0.5)
    ax.legend()
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.autofmt_xdate()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    buf.seek(0)
    return BufferedInputFile(buf.getvalue(), 'btc_prediction.png')

# --- BOT HANDLERS ---

main_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text='📊 Анализ BTC')],
        [KeyboardButton(text='ℹ️ Информация')],
        [KeyboardButton(text='💳 Мой баланс')]
    ],
    resize_keyboard=True,
    input_field_placeholder='Press a button...'
)

@dp.startup()
async def on_startup():
    logging.info('Bot started.')

@dp.message(Command('start'))
async def cmd_start(message: types.Message):
    user_id = message.from_user.id
    if user_id not in user_limits:
        user_limits[user_id] = {'balance': STARTING_BALANCE, 'last_prediction_time': None}
    
    await message.answer(
        '👋 Welcome to AI BTC Predictor!\n\n'
        'I analyze the market using neural networks.\n'
        f'Timezone: {TIMEZONE_STR}.\n'
        'Timeframe: 1 minute.',
        reply_markup=main_keyboard
    )

@dp.message(F.text == 'ℹ️ Информация')
async def cmd_info(message: types.Message):
    await message.answer(
        f'📊 **How it works:**\n'
        f'1. Source: CoinGecko (1 min TF).\n'
        f'2. Timezone: {TIMEZONE_STR}.\n\n'
        '⚠️ *Not financial advice.*',
        parse_mode='Markdown'
    )

@dp.message(F.text == '💳 Мой баланс')
async def cmd_balance(message: types.Message):
    user_id = message.from_user.id
    user_data = user_limits.get(user_id, {'balance': 0})
    
    await message.answer(
        f'💳 **Balance:** `{user_data["balance"]}` predictions.',
        parse_mode='Markdown'
    )

@dp.message(F.text == '📊 Анализ BTC')
async def cmd_predict(message: types.Message):
    user_id = message.from_user.id
    
    if user_limits[user_id]['balance'] <= 0:
        await message.answer('❌ No predictions left. Balance: 0.')
        return

    last_time = user_limits[user_id]['last_prediction_time']
    if last_time:
        now = datetime.now(LOCAL_TIMEZONE)
        delta = now - last_time
        if delta.total_seconds() < 60:
            remaining = int(60 - delta.total_seconds())
            await message.answer(f'⏳ Please wait {remaining} sec.')
            return

    status_msg = await message.answer('⏳ Getting data and training AI...')

    try:
        df_raw = await get_market_data()
        if df_raw is None:
            await status_msg.edit_text('❌ Error fetching data from CoinGecko.')
            return

        df_processed, pred_price, next_time = predict_next_minute(df_raw)
        
        if pred_price is None:
            await status_msg.edit_text('❌ Could not build model (not enough data).')
            return

        plot_buf = create_plot(df_processed, pred_price, next_time)

        current_price = df_processed['close'].iloc[-1]
        diff = pred_price - current_price
        emoji = 'Ⓜ️' if abs(diff) < 1 else ('📈' if diff > 0 else '📉')
        
        time_str = next_time.strftime('%H:%M')
        
        caption = (
            f'{emoji} **BTC/USDT Prediction**\n\n'
            f'Current: `{current_price:.2f}` $\n'
            f'Prediction for {time_str}: `{pred_price:.2f}` $\n\n'
            f'Change: `{diff:+.2f}` $\n'
            f'Predictions left: `{user_limits[user_id]["balance"] - 1}`'
        )

        user_limits[user_id]['balance'] -= 1
        user_limits[user_id]['last_prediction_time'] = datetime.now(LOCAL_TIMEZONE)

        await status_msg.delete()
        
        await bot.send_photo(
            chat_id=message.chat.id,
            photo=plot_buf,
            caption=caption,
            parse_mode='Markdown'
        )

    except Exception as default_error:
        logging.error(f'Error: {default_error}')
        await status_msg.edit_text('❌ An error occurred.')

async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
