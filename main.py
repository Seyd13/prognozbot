import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from io import BytesIO

import aiohttp
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Используем неинтерактивный бэкенд для сервера
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import BufferedInputFile, ReplyKeyboardMarkup, KeyboardButton
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from zoneinfo import ZoneInfo  # Для работы с часовыми поясами (Python 3.9+)

# --- КОНФИГУРАЦИЯ ---
# Вставьте сюда ваш токен
TELEGRAM_TOKEN = "2122435147:AAG_52ELCHjFnXNxcAP4i5xNAal9I91xNTM"

# НАСТРОЙКА ВРЕМЕНИ
# Укажите вашу временную зону. Примеры: 'Europe/Moscow', 'Europe/Kiev', 'Asia/Almaty', 'UTC'
TIMEZONE_STR = "Europe/Moscow"
LOCAL_TIMEZONE = ZoneInfo(TIMEZONE_STR)

STARTING_BALANCE = 100

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# --- БАЗА ДАННЫХ В ПАМЯТИ ---
user_limits = defaultdict(lambda: {'balance': STARTING_BALANCE, 'last_prediction_time': None})

# --- ФУНКЦИИ ДАННЫХ И ИНДИКАТОРЫ ---

async def get_market_data():
    """Получает данные с CoinGecko API и конвертирует время в локальное."""
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=1"
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status == 200:
                    data = await response.json()
                    prices = data.get('prices', []) 
                    if not prices:
                        return None

                    df = pd.DataFrame(prices, columns=['timestamp', 'close'])
                    # Конвертация: ms -> datetime(UTC) -> datetime(Local)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(LOCAL_TIMEZONE)
                    
                    df = df.rename(columns={'timestamp': 'close_time'})
                    
                    # Берем последние 30 точек для анализа
                    df = df.tail(30).reset_index(drop=True)
                    return df
                else:
                    logging.error(f"Ошибка CoinGecko HTTP: {response.status}")
                    return None
    except Exception as e:
        logging.error(f"Ошибка подключения к CoinGecko: {e}")
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
        logging.error(f"Ошибка обучения модели: {e}")
        return None, None, None

    last_window = scaled_data[-look_back:].flatten().reshape(1, -1)
    predicted_scaled = model.predict(last_window)
    
    # Обратное масштабирование только цены (колонка 0)
    dummy_array = np.zeros((1, 3))
    dummy_array[0, 0] = predicted_scaled[0]
    dummy_array[0, 1] = scaled_data[-1, 1] 
    dummy_array[0, 2] = scaled_data[-1, 2] 
    
    predicted_price_full = scaler.inverse_transform(dummy_array)
    predicted_price = predicted_price_full[0, 0]

    last_data_time = df['close_time'].iloc[-1]
    
    # Вычисляем шаг времени
    if len(df) > 1:
        time_diffs = df['close_time'].diff().dropna()
        avg_step = time_diffs.median()
    else:
        avg_step = timedelta(minutes=1)

    # Следующая минута в локальном времени
    next_time = last_data_time + avg_step

    return df, predicted_price, next_time

def create_plot(df, predicted_price, next_time):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_df = df.tail(10).copy()
    
    # ПРАВИЛЬНАЯ ОБРАБОТКА ДАТ ДЛЯ MATPLOTLIB
    # Преобразуем datetime в числа, сохраняя информацию о часовом поясе
    times = mdates.date2num(plot_df['close_time'].to_pydatetime())
    next_time_num = mdates.date2num(next_time.to_pydatetime())
    
    # Отрисовка истории
    ax.plot(times, plot_df['close'], 
            label='История', color='cyan', marker='o', linestyle='-')

    # Отрисовка прогноза (линия от последней точки к прогнозу)
    ax.plot([times[-1], next_time_num],
            [plot_df['close'].iloc[-1], predicted_price],
            label='Прогноз AI', color='lime', linestyle='--', marker='x')
    
    # Точка прогноза
    ax.scatter(next_time_num, predicted_price, color='lime', s=100, zorder=5)

    # Подписи для исторических точек
    for x, y in zip(times, plot_df['close']):
        label = f"{y:.0f}"
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='white')

    # Подпись для точки прогноза
    ax.annotate(f"AI: {predicted_price:.0f}", 
                (next_time_num, predicted_price), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=9, color='lime', fontweight='bold')

    ax.set_title(f"BTC/USDT AI Prediction ({TIMEZONE_STR})", color='white', fontsize=14)
    ax.set_xlabel("Время", color='gray')
    ax.set_ylabel("Цена ($)", color='gray')
    
    ax.grid(True, color='gray', linestyle=':', alpha=0.5)
    ax.legend()
    
    # Форматирование оси X: показываем часы и минуты, учитывая локальную зону
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=LOCAL_TIMEZONE))
    fig.autofmt_xdate()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    buf.seek(0)
    return BufferedInputFile(buf.getvalue(), "btc_prediction.png")

# --- ХЕНДЛЕРЫ БОТА ---

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
    logging.info("Бот успешно запущен.")

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    user_id = message.from_user.id
    if user_id not in user_limits:
        user_limits[user_id] = {'balance': STARTING_BALANCE, 'last_prediction_time': None}
    
    await message.answer(
        "👋 Добро пожаловать в AI BTC Predictor!\n\n"
        "Я анализирую рынок с помощью нейросети и выдаю краткосрочный прогноз.\n"
        f"Установленная временная зона: {TIMEZONE_STR}.",
        reply_markup=main_keyboard
    )

@dp.message(F.text == "ℹ️ Информация")
async def cmd_info(message: types.Message):
    await message.answer(
        f"📊 **Как это работает:**\n"
        f"1. Данные CoinGecko (BTC/USD).\n"
        f"2. Время конвертируется в локальное ({TIMEZONE_STR}).\n"
        f"3. Нейросеть MLP делает прогноз на следующую минуту.\n\n"
        "⚠️ *Не является финансовым советом.*",
        parse_mode="Markdown"
    )

@dp.message(F.text == "💳 Мой баланс")
async def cmd_balance(message: types.Message):
    user_id = message.from_user.id
    user_data = user_limits.get(user_id, {'balance': 0})
    
    await message.answer(
        f"💳 **Ваш баланс:** `{user_data['balance']}` прогнозов.",
        parse_mode="Markdown"
    )

@dp.message(F.text == "📊 Анализ BTC")
async def cmd_predict(message: types.Message):
    user_id = message.from_user.id
    
    # Проверка баланса
    if user_limits[user_id]['balance'] <= 0:
        await message.answer("❌ У вас закончились прогнозы. Баланс: 0.")
        return

    # Проверка кулдауна (60 секунд)
    last_time = user_limits[user_id]['last_prediction_time']
    if last_time:
        now = datetime.now(LOCAL_TIMEZONE)
        delta = now - last_time
        if delta.total_seconds() < 60:
            remaining = int(60 - delta.total_seconds())
            await message.answer(f"⏳ Пожалуйста, подождите {remaining} сек. перед новым запросом.")
            return

    status_msg = await message.answer("⏳ Получаю данные и обучаю нейросеть...")

    try:
        # 1. Получение данных
        df_raw = await get_market_data()
        if df_raw is None:
            await status_msg.edit_text("❌ Ошибка получения данных от CoinGecko. Попробуйте позже.")
            return

        # 2. Предсказание
        df_processed, pred_price, next_time = predict_next_minute(df_raw)
        
        if pred_price is None:
            await status_msg.edit_text("❌ Не удалось построить модель (мало данных или ошибка).")
            return

        # 3. Создание графика
        try:
            plot_buf = create_plot(df_processed, pred_price, next_time)
        except Exception as plot_err:
            logging.error(f"Ошибка рисования графика: {plot_err}")
            await status_msg.edit_text("⚠️ Ошибка при генерации изображения.")
            return

        # 4. Формирование ответа
        current_price = df_processed['close'].iloc[-1]
        diff = pred_price - current_price
        emoji = "Ⓜ️" if abs(diff) < 1 else ("📈" if diff > 0 else "📉")
        
        time_str = next_time.strftime('%H:%M')
        
        caption = (
            f"{emoji} **Прогноз BTC/USDT**\n\n"
            f"Текущая: `{current_price:.2f}` $\n"
            f"Прогноз на {time_str}: `{pred_price:.2f}` $\n\n"
            f"Изменение: `{diff:+.2f}` $\n"
            f"Осталось прогнозов: `{user_limits[user_id]['balance'] - 1}`"
        )

        # Списание баланса и обновление времени
        user_limits[user_id]['balance'] -= 1
        user_limits[user_id]['last_prediction_time'] = datetime.now(LOCAL_TIMEZONE)

        await status_msg.delete()
        
        await bot.send_photo(
            chat_id=message.chat.id,
            photo=plot_buf,
            caption=caption,
            parse_mode="Markdown"
        )

    except Exception as default_error:
        logging.error(f"Критическая ошибка в хендлере: {default_error}")
        await status_msg.edit_text("❌ Произошла непредвиденная ошибка.")

async def main():
    # Удаляем вебхук, чтобы бот работал через Long Polling
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Бот остановлен.")
