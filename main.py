iimport asyncio
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

# --- КОНФИГУРАЦИЯ ---
TELEGRAM_TOKEN = "2122435147:AAG_52ELCHjFnXNxcAP4i5xNAal9I91xNTM"

# ВРЕМЯ
TIMEZONE_STR = "Europe/Moscow"
LOCAL_TIMEZONE = ZoneInfo(TIMEZONE_STR)

STARTING_BALANCE = 100

logging.basicConfig(level=logging.INFO)

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# --- БАЗА ДАННЫХ ---
user_limits = defaultdict(lambda: {'balance': STARTING_BALANCE, 'last_prediction_time': None})

# --- ФУНКЦИИ ---

async def get_market_data():
    """
    Получает данные с CoinGecko API.
    ИСПРАВЛЕНИЕ: days=1 дает данные за сутки. 
    Мы делаем ресемплирование (resample), чтобы выровнять точки ровно по 5 минут.
    """
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=1"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=20)) as response:
                if response.status == 200:
                    data = await response.json()
                    prices = data.get('prices', []) 
                    if not prices:
                        return None

                    df = pd.DataFrame(prices, columns=['timestamp', 'close'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # --- КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ ВРЕМЕНИ ---
                    # 1. Устанавливаем timestamp как индекс
                    df.set_index('timestamp', inplace=True)
                    
                    # 2. Ресемплируем в 5-минутные свечи ('5T').
                    # Это создает строгие интервалы: 13:00, 13:05, 13:10 и т.д.
                    # agg({'close': 'last'}) берет последнюю цену в 5-минутном окне.
                    df = df.resample('5T').agg({'close': 'last'})
                    
                    # 3. Удаляем пустые строки (если данных не было)
                    df.dropna(inplace=True)
                    
                    # 4. Возвращаем timestamp в колонку и конвертируем часовой пояс
                    df.reset_index(inplace=True)
                    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(LOCAL_TIMEZONE)
                    
                    df = df.rename(columns={'timestamp': 'close_time'})
                    
                    # Берем последние 60 свечей (5 часов истории)
                    df = df.tail(60).reset_index(drop=True)
                    return df
                else:
                    logging.error(f"Ошибка CoinGecko HTTP: {response.status}")
                    return None
    except asyncio.TimeoutError:
        logging.error("Таймаут соединения")
        return None
    except Exception as e:
        logging.error(f"Ошибка подключения: {e}")
        return None

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def predict_next_5min(df):
    """
    Предсказывает СЛЕДУЮЩУЮ 5-минутную свечу.
    """
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
    look_back = 10 # Смотрим на 10 свечей назад (50 минут)
    
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
        logging.error(f"Ошибка обучения модели: {e}")
        return None, None, None

    last_window = scaled_data[-look_back:].flatten().reshape(1, -1)
    predicted_scaled = model.predict(last_window)[0]
    
    # Обратное масштабирование
    dummy_array = np.zeros((1, 3))
    dummy_array[0, 0] = predicted_scaled
    dummy_array[0, 1] = scaled_data[-1, 1] 
    dummy_array[0, 2] = scaled_data[-1, 2] 
    
    predicted_price_full = scaler.inverse_transform(dummy_array)
    predicted_price = predicted_price_full[0, 0]

    # --- ЛОГИКА ВРЕМЕНИ ПРОГНОЗА ---
    # Берем время последней свечи в данных и прибавляем ровно 5 минут.
    last_time = df['close_time'].iloc[-1]
    next_time = last_time + timedelta(minutes=5)

    return df, predicted_price, next_time

def create_plot(df, predicted_price, next_time):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Берем последние 20 свечей (100 минут)
    plot_df = df.tail(20).copy()
    
    plot_df['close_time_plot'] = plot_df['close_time'].dt.tz_localize(None)
    next_time_plot = next_time.replace(tzinfo=None) if next_time.tzinfo else next_time
    
    # Линия истории
    ax.plot(plot_df['close_time_plot'], plot_df['close'], 
            color='cyan', marker='o', linestyle='-', markersize=8, zorder=2)
    
    # Линия прогноза
    ax.plot([plot_df['close_time_plot'].iloc[-1], next_time_plot],
            [plot_df['close'].iloc[-1], predicted_price],
            color='lime', linestyle='--', marker='x', markersize=10, zorder=2)
    
    # Точка прогноза
    ax.scatter(next_time_plot, predicted_price, color='lime', s=200, zorder=3, edgecolors='white')

    # --- ОТРИСОВКА ТЕКСТА ---
    
    # Исторические точки
    for x, y, time_obj in zip(plot_df['close_time_plot'], plot_df['close'], plot_df['close_time']):
        time_str = time_obj.strftime('%H:%M')
        price_str = f"{y:.0f}"
        
        ax.annotate(time_str, (x, y), textcoords="offset points", xytext=(0,12), 
                    ha='center', fontsize=9, color='yellow', fontweight='bold')
        ax.annotate(price_str, (x, y), textcoords="offset points", xytext=(0,-12), 
                    ha='center', fontsize=8, color='white')

    # Точка прогноза
    pred_time_str = next_time.strftime('%H:%M')
    pred_price_str = f"{predicted_price:.0f}"
    
    ax.annotate(pred_time_str, (next_time_plot, predicted_price), textcoords="offset points", xytext=(0,15), 
                ha='center', fontsize=10, color='lime', fontweight='bold')
    ax.annotate(pred_price_str, (next_time_plot, predicted_price), textcoords="offset points", xytext=(0,-15), 
                ha='center', fontsize=9, color='lime', fontweight='bold')

    ax.get_xaxis().set_visible(False)
    
    ax.set_title(f"BTC/USDT AI Prediction (5m TF) ({TIMEZONE_STR})", color='white', fontsize=16)
    ax.set_ylabel("Цена ($)", color='gray')
    ax.grid(True, color='gray', linestyle=':', alpha=0.3)
    
    ax.legend(['История', 'Прогноз AI'], loc='upper left')

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return BufferedInputFile(buf.getvalue(), "btc_prediction.png")

# --- ХЕНДЛЕРЫ ---

main_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="📊 Анализ BTC")],
        [KeyboardButton(text="ℹ️ Информация")],
        [KeyboardButton(text="💳 Мой баланс")]
    ],
    resize_keyboard=True,
    input_field_placeholder="Выберите действие..."
)

@dp.startup()
async def on_startup():
    logging.info("Бот запущен.")

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    user_id = message.from_user.id
    if user_id not in user_limits:
        user_limits[user_id] = {'balance': STARTING_BALANCE, 'last_prediction_time': None}
    await message.answer(
        "👋 Добро пожаловать в AI BTC Predictor!\n\n"
        "Анализ рынка с таймфреймом 5 минут.\n"
        f"Часовой пояс: {TIMEZONE_STR}.",
        reply_markup=main_keyboard
    )

@dp.message(F.text == "ℹ️ Информация")
async def cmd_info(message: types.Message):
    await message.answer(
        f"📊 **Как это работает:**\n"
        f"1. Данные выравниваются по 5-минутным свечам (13:00, 13:05...).\n"
        f"2. Прогноз дается на следующую 5-минутную свечу.\n\n"
        "⚠️ *Не финансовый совет.*",
        parse_mode="Markdown"
    )

@dp.message(F.text == "💳 Мой баланс")
async def cmd_balance(message: types.Message):
    user_data = user_limits.get(message.from_user.id, {'balance': 0})
    await message.answer(
        f"💳 **Ваш баланс:** `{user_data['balance']}` прогнозов.",
        parse_mode="Markdown"
    )

@dp.message(F.text == "📊 Анализ BTC")
async def cmd_predict(message: types.Message):
    user_id = message.from_user.id
    
    if user_limits[user_id]['balance'] <= 0:
        await message.answer("❌ У вас закончились прогнозы. Баланс: 0.")
        return

    last_time = user_limits[user_id]['last_prediction_time']
    if last_time:
        now = datetime.now(LOCAL_TIMEZONE)
        # Ограничение 5 минут
        if (now - last_time).total_seconds() < 300:
            remain = int(300 - (now - last_time).total_seconds())
            await message.answer(f"⏳ Подождите {remain} сек перед новым запросом.")
            return

    status_msg = await message.answer("⏳ Анализ 5-минутных свечей...")

    try:
        df_raw = await get_market_data()
        if df_raw is None:
            await status_msg.edit_text("❌ Ошибка получения данных.")
            return

        df_processed, pred_price, next_time = predict_next_5min(df_raw)
        if pred_price is None:
            await status_msg.edit_text("❌ Мало данных для построения модели.")
            return

        plot_buf = create_plot(df_processed, pred_price, next_time)
        current_price = df_processed['close'].iloc[-1]
        diff = pred_price - current_price
        emoji = "Ⓜ️" if abs(diff) < 1 else ("📈" if diff > 0 else "📉")
        
        time_str = next_time.strftime('%H:%M')
        
        caption = (
            f"{emoji} **Прогноз BTC/USDT (5m)**\n\n"
            f"Текущая свеча: `{current_price:.2f}` $\n"
            f"Прогноз на {time_str}: `{pred_price:.2f}` $\n\n"
            f"Изменение: `{diff:+.2f}` $\n"
            f"Осталось прогнозов: `{user_limits[user_id]['balance'] - 1}`"
        )

        user_limits[user_id]['balance'] -= 1
        user_limits[user_id]['last_prediction_time'] = datetime.now(LOCAL_TIMEZONE)

        await status_msg.delete()
        await bot.send_photo(
            chat_id=message.chat.id,
            photo=plot_buf,
            caption=caption,
            parse_mode="Markdown"
        )

    except Exception as e:
        logging.error(f"Критическая ошибка: {e}")
        await status_msg.edit_text("❌ Произошла ошибка бота.")

async def main():
    # Сброс вебхуков для устранения конфликта
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
