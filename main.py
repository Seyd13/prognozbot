import asyncio
import logging
from datetime import datetime, timedelta
import aiohttp
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, BufferedInputFile, ReplyKeyboardMarkup, KeyboardButton
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict

# --- КОНФИГУРАЦИЯ ---
TELEGRAM_TOKEN = "2122435147:AAG_52ELCHjFnXNxcAP4i5xNAal9I91xNTM"

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Инициализация бота и диспетчера
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# --- БАЗА ДАННЫХ В ПАМЯТИ ---
# user_limits: {user_id: {'balance': 100, 'last_prediction_time': datetime}}
user_limits = defaultdict(lambda: {'balance': 100, 'last_prediction_time': None})
STARTING_BALANCE = 100

# --- ФУНКЦИИ ДАННЫХ И ИНДИКАТОРОВ ---

async def get_market_data():
    """Получает данные с CoinGecko API (последние 30 точек)."""
    # Запрашиваем данные за последние 2 дня с интервалом 5 мин, чтобы убедиться, что хватит данных,
    # или стандартный daily chart с почасовыми данными. Для точности "последних минут" лучше grab.
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
                    # Конвертируем timestamp в datetime (UTC по умолчанию в pandas)
                    df['close_time'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # Берем последние 30 точек (обычно это 30 минут для minutely data)
                    # CoinGecko daily chart часто возвращает данные с разным шагом. 
                    # Мы берем "хвост" данных.
                    df = df.tail(30).reset_index(drop=True)
                    return df
                else:
                    logging.error(f"Ошибка CoinGecko HTTP: {response.status}")
                    return None
    except Exception as e:
        logging.error(f"Ошибка подключения к CoinGecko: {e}")
        return None

def calculate_rsi(series, period=14):
    """Расчет RSI."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- ФУНКЦИИ МОДЕЛИ И ПРЕДСКАЗАНИЯ ---

def predict_next_minute(df):
    """Обучает модель и предсказывает следующую цену."""
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

    # MLPRegressor
    model = MLPRegressor(hidden_layer_sizes=(10, 5), max_iter=500, random_state=42)
    try:
        model.fit(X, y)
    except Exception as e:
        logging.error(f"Ошибка обучения модели: {e}")
        return None, None, None

    # Предсказание
    last_window = scaled_data[-look_back:].flatten().reshape(1, -1)
    predicted_scaled = model.predict(last_window)
    
    # Обратное масштабирование
    dummy_array = np.zeros((1, 3))
    dummy_array[0, 0] = predicted_scaled[0]
    dummy_array[0, 1] = scaled_data[-1, 1] 
    dummy_array[0, 2] = scaled_data[-1, 2] 
    
    predicted_price_full = scaler.inverse_transform(dummy_array)
    predicted_price = predicted_price_full[0, 0]

    # Логика времени
    last_data_time = df['close_time'].iloc[-1]
    
    # Нам нужно понять шаг данных (обычно 1 минута, но иногда API дает реже)
    # Если данных много, считаем медианный интервал
    if len(df) > 1:
        time_diffs = df['close_time'].diff().dropna()
        avg_step = time_diffs.median()
    else:
        avg_step = timedelta(minutes=1)

    next_time = last_data_time + avg_step

    return df, predicted_price, next_time

# --- ГЕНЕРАЦИЯ ГРАФИКА ---

def create_plot(df, predicted_price, next_time):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Рисуем последние 10 точек истории
    plot_df = df.tail(10).copy()
    
    # Линия истории
    ax.plot(plot_df['close_time'], plot_df['close'], 
            label='История', color='cyan', marker='o', linestyle='-')

    # Линия прогноза
    ax.plot([plot_df['close_time'].iloc[-1], next_time],
            [plot_df['close'].iloc[-1], predicted_price],
            label='Прогноз AI', color='lime', linestyle='--', marker='x')
    
    # Точка прогноза
    ax.scatter(next_time, predicted_price, color='lime', s=100, zorder=5)

    # Подписи
    for x, y in zip(plot_df['close_time'], plot_df['close']):
        label = f"{y:.0f}"
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='white')

    # Подпись прогноза
    ax.annotate(f"AI: {predicted_price:.0f}", 
                (next_time, predicted_price), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=9, color='lime', fontweight='bold')

    ax.set_title(f"BTC/USDT AI Prediction", color='white', fontsize=14)
    ax.set_xlabel("Время", color='gray')
    ax.set_ylabel("Цена ($)", color='gray')
    ax.grid(True, color='gray', linestyle=':', alpha=0.5)
    ax.legend()
    
    # Форматирование оси времени
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.autofmt_xdate()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    buf.seek(0)
    return BufferedInputFile(buf.getvalue(), "btc_prediction.png")

# --- ХЕНДЛЕРЫ БОТА ---

# Основная клавиатура (всегда внизу)
main_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="📊 Анализ BTC")],
        [KeyboardButton(text="ℹ️ Информация")],
        [KeyboardButton(text="💳 Мой баланс")]
    ],
    resize_keyboard=True,
    input_field_placeholder="Нажмите кнопку для действия..."
)

@dp.startup()
async def on_startup():
    logging.info("Бот запущен.")

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    user_id = message.from_user.id
    # Инициализация пользователя если его нет
    if user_id not in user_limits:
        user_limits[user_id] = {'balance': STARTING_BALANCE, 'last_prediction_time': None}
    
    await message.answer(
        "👋 Добро пожаловать в AI BTC Predictor!\n\n"
        "Я анализирую рынок с помощью нейросети и выдаю краткосрочный прогноз.\n"
        "Используйте меню ниже для навигации.",
        reply_markup=main_keyboard
    )

@dp.message(F.text == "ℹ️ Информация")
async def cmd_info(message: types.Message):
    await message.answer(
        "📊 **Как это работает:**\n"
        "1. Я беру данные CoinGecko (BTC/USD).\n"
        "2. Рассчитываю RSI и тренд.\n"
        "3. MLP нейросеть обучается на последних 30 точках.\n"
        "4. Выдаю прогноз на ближайший интервал времени.\n\n"
        "⚠️ *Важно:* Это не финансовый совет. Бот может ошибаться.",
        parse_mode="Markdown"
    )

@dp.message(F.text == "💳 Мой баланс")
async def cmd_balance(message: types.Message):
    user_id = message.from_user.id
    user_data = user_limits.get(user_id, {'balance': 0})
    
    await message.answer(
        f"💳 **Ваш баланс:** `{user_data['balance']}` прогнозов.\n\n"
        f"Лимит: 100 шт. на пользователя.",
        parse_mode="Markdown"
    )

@dp.message(F.text == "📊 Анализ BTC")
async def cmd_predict(message: types.Message):
    user_id = message.from_user.id
    
    # Проверка баланса
    if user_limits[user_id]['balance'] <= 0:
        await message.answer("❌ У вас закончились прогнозы. Баланс: 0.")
        return

    # Проверка кулдауна (1 минута)
    last_time = user_limits[user_id]['last_prediction_time']
    if last_time:
        # Проверяем, прошла ли минута с последнего запроса (опираемся на время сервера бота)
        # Используем datetime.now(datetime.timezone.utc) для точности, или просто naive time если сервер UTC
        now = datetime.now() 
        delta = now - last_time
        if delta.total_seconds() < 60:
            remaining = int(60 - delta.total_seconds())
            await message.answer(f"⏳ Пожалуйста, подождите {remaining} сек. перед новым запросом.")
            return

    # Отправляем статус
    status_msg = await message.answer("⏳ Получаю данные и обучаю нейросеть...")

    try:
        # 1. Получение данных
        df_raw = await get_market_data()
        if df_raw is None:
            await status_msg.edit_text("❌ Ошибка получения данных от CoinGecko.")
            return

        # 2. Предсказание
        df_processed, pred_price, next_time = predict_next_minute(df_raw)
        
        if pred_price is None:
            await status_msg.edit_text("❌ Не удалось построить модель (мало данных).")
            return

        # 3. График
        plot_buf = create_plot(df_processed, pred_price, next_time)

        current_price = df_processed['close'].iloc[-1]
        diff = pred_price - current_price
        emoji = "📈" if diff > 0 else "📉"
        
        # Формирование времени для текста
        time_str = next_time.strftime('%H:%M')
        
        caption = (
            f"{emoji} **Прогноз BTC/USDT**\n\n"
            f"Текущая: `{current_price:.2f}` $\n"
            f"Прогноз на {time_str}: `{pred_price:.2f}` $\n\n"
            f"Изменение: `{diff:+.2f}` $\n"
            f"Осталось прогнозов: `{user_limits[user_id]['balance'] - 1}`"
        )

        # Обновляем баланс и время
        user_limits[user_id]['balance'] -= 1
        user_limits[user_id]['last_prediction_time'] = datetime.now()

        # Удаляем сообщение "Загрузка" и отправляем результат (не удаляем старые сообщения)
        await status_msg.delete()
        
        await bot.send_photo(
            chat_id=message.chat.id,
            photo=plot_buf,
            caption=caption,
            parse_mode="Markdown"
        )

    except Exception as e:
        logging.error(f"Ошибка: {e}")
        await status_msg.edit_text("❌ Произошла критическая ошибка.")

# Точка входа
async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
