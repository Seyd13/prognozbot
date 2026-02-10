import asyncio
import logging
import os
from datetime import datetime, timedelta
import aiohttp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

# --- КОНФИГУРАЦИЯ ---
# ВАЖНО: Замените этот токен на новый после деплоя!
TELEGRAM_TOKEN = "2122435147:AAG_52ELCHjFnXNxcAP4i5xNAal9I91xNTM"

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Инициализация бота и диспетчера
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# Глобальный флаг для защиты от спама (один прогноз за раз)
is_predicting = False

# --- ФУНКЦИИ ДАННЫХ И ИНДИКАТОРОВ ---

async def get_binance_klines(interval='1m', limit=20):
    """Получает данные свечей с Binance."""
    url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval={interval}&limit={limit}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                # Преобразуем в DataFrame: [Time, Open, High, Low, Close, Volume, ...]
                df = pd.DataFrame(data, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                # Конвертируем типы
                df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
                df['close'] = pd.to_numeric(df['close'])
                return df[['close_time', 'close']]
            else:
                logging.error(f"Ошибка Binance API: {response.status}")
                return None

def calculate_rsi(series, period=14):
    """Расчет RSI."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- ФУНКЦИИ МОДЕЛИ И ПРЕДСКАЗАНИЯ ---

def predict_next_minute(df):
    """
    Готовит данные, обучает легкую нейросеть и делает прогноз.
    """
    # 1. Добавляем RSI
    df = df.copy()
    df['rsi'] = calculate_rsi(df['close'])
    df.dropna(inplace=True)

    # Если данных мало после RSI, берем простую разницу
    if len(df) < 5:
        logging.warning("Недостаточно данных после очистки NaN")
        return None, None

    # 2. Подготовка признаков (Features)
    # Используем цену и RSI как входные данные
    data = df[['close', 'rsi']].values
    
    # Нормализация (важно для нейросетей)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Формируем X (последние 10 точек) и y (следующая точка)
    # Для обучения используем скользящее окно
    X, y = [], []
    look_back = 10
    if len(scaled_data) <= look_back:
        return None, None

    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:i + look_back].flatten()) # Разворачиваем окно в вектор
        y.append(scaled_data[i + look_back][0]) # Предсказываем только цену (индекс 0)

    if not X:
        return None, None

    X = np.array(X)
    y = np.array(y)

    # 3. Обучение легкой модели (MLPRegressor)
    # Это небольшая нейросеть, которая учится прямо "на лету"
    model = MLPRegressor(hidden_layer_sizes=(10, 5), max_iter=500, random_state=42)
    try:
        model.fit(X, y)
    except Exception as e:
        logging.error(f"Ошибка обучения модели: {e}")
        return None, None

    # 4. Предсказание
    # Берем последние look_back точек для прогноза на 1 минуту вперед
    last_window = scaled_data[-look_back:].flatten().reshape(1, -1)
    predicted_scaled = model.predict(last_window)
    
    # Денормализация цены
    # Создаем фиктивный массив с 0 для RSI, чтобы scaler вернул только цену
    dummy_array = np.zeros((1, 2))
    dummy_array[0, 0] = predicted_scaled[0]
    predicted_price = scaler.inverse_transform(dummy_array)[0, 0]

    # Время следующей минуты
    last_time = df['close_time'].iloc[-1]
    next_time = last_time + timedelta(minutes=1)

    return df, predicted_price, next_time

# --- ГЕНЕРАЦИЯ ГРАФИКА ---

def create_plot(df, predicted_price, next_time):
    """
    Рисует график: 10 предыдущих минут + 1 предсказанная.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Берем только последние 10 точек для чистоты графика (или сколько есть)
    plot_df = df.tail(10).copy()
    
    # График истории
    ax.plot(plot_df['close_time'], plot_df['close'], 
            label='История', color='cyan', marker='o', linestyle='-')

    # График предсказания
    # Соединяем последнюю точку истории с предсказанием
    ax.plot([plot_df['close_time'].iloc[-1], next_time],
            [plot_df['close'].iloc[-1], predicted_price],
            label='Прогноз AI', color='lime', linestyle='--', marker='x')
    
    # Точки предсказания
    ax.scatter(next_time, predicted_price, color='lime', s=100, zorder=5)

    # Подписи точек (Цена и Время)
    for x, y in zip(plot_df['close_time'], plot_df['close']):
        label = f"{y:.0f}\n{x.strftime('%H:%M')}"
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='white')

    # Подпись предсказания
    ax.annotate(f"AI: {predicted_price:.0f}\n{next_time.strftime('%H:%M')}", 
                (next_time, predicted_price), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=9, color='lime', fontweight='bold')

    # Форматирование
    ax.set_title(f"BTC/USDT Прогноз на минуту", color='white', fontsize=14)
    ax.set_xlabel("Время", color='gray')
    ax.set_ylabel("Цена ($)", color='gray')
    ax.grid(True, color='gray', linestyle=':', alpha=0.5)
    ax.legend()
    
    # Формат оси времени
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.autofmt_xdate()

    # Сохранение в буфер
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

# --- ХЕНДЛЕРЫ БОТА ---

@dp.startup()
async def on_startup():
    logging.info("Бот запущен. Нажмите /start чтобы начать.")

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    kb = [
        [InlineKeyboardButton(text="🔮 Получить прогноз BTC", callback_data="predict_btc")],
        [InlineKeyboardButton(text="ℹ️ Как работает бот?", callback_data="help_info")]
    ]
    keyboard = InlineKeyboardMarkup(inline_keyboard=kb)
    await message.answer(
        "👋 Привет! Я AI-бот для прогнозирования цены Bitcoin.\n\n"
        "Я анализирую данные в реальном времени, использую нейросеть и RSI индикатор, "
        "чтобы предсказать цену закрытия следующей минутной свечи.\n\n"
        "Нажмите кнопку ниже, чтобы получить прогноз.",
        reply_markup=keyboard
    )

@dp.callback_query(F.data == "help_info")
async def show_help(callback: types.CallbackQuery):
    await callback.message.edit_text(
        "📊 **Как это работает:**\n"
        "1. Я получаю данные с биржи Binance (минутный график).\n"
        "2. Рассчитываю индекс RSI (Relative Strength Index).\n"
        "3. Обучаю легкую нейросеть (MLP) на последних данных.\n"
        "4. Рисую график с 10 минутами истории и 1 минутой прогноза.\n\n"
        "⚠️ Это не финансовая рекомендация, а демонстрация возможностей AI.",
        parse_mode="Markdown"
    )
    await callback.answer()

@dp.callback_query(F.data == "predict_btc")
async def process_prediction(callback: types.CallbackQuery):
    global is_predicting

    if is_predicting:
        await callback.answer("⏳ В данный момент я уже высчитываю прогноз. Пожалуйста, подождите.", show_alert=True)
        return

    is_predicting = True
    await callback.message.edit_text("⏳ Получаю данные с биржи и обучаю модель... Это займет несколько секунд.")

    try:
        # 1. Получаем данные
        df_raw = await get_binance_klines(limit=30) # Берем чуть больше для расчета RSI корректно
        if df_raw is None:
            await callback.message.edit_text("❌ Не удалось получить данные от Binance. Попробуйте позже.")
            return

        # 2. Предсказываем
        df_processed, pred_price, next_time = predict_next_minute(df_raw)
        
        if pred_price is None:
            await callback.message.edit_text("❌ Не удалось построить модель (мало данных).")
            return

        # 3. Рисуем график
        plot_buf = create_plot(df_processed, pred_price, next_time)

        # 4. Отправляем результат
        current_price = df_processed['close'].iloc[-1]
        diff = pred_price - current_price
        emoji = "📈" if diff > 0 else "📉"
        
        caption = (
            f"{emoji} **Прогноз BTC/USDT**\n\n"
            f"Текущая цена: `{current_price:.2f}` $\n"
            f"Прогноз на {next_time.strftime('%H:%M')}: `{pred_price:.2f}` $\n\n"
            f"Изменение: `{diff:.2f}` $"
        )

        # Клавиатура "Еще раз"
        kb = [[InlineKeyboardButton(text="🔄 Обновить прогноз", callback_data="predict_btc")]]
        keyboard = InlineKeyboardMarkup(inline_keyboard=kb)

        await callback.message.delete()
        await bot.send_photo(
            chat_id=callback.message.chat.id,
            photo=plot_buf,
            caption=caption,
            parse_mode="Markdown",
            reply_markup=keyboard
        )

    except Exception as e:
        logging.error(f"Ошибка в процессе прогноза: {e}")
        await callback.message.edit_text("❌ Произошла ошибка при обработке запроса.")
    finally:
        is_predicting = False

# --- ЗАЩИТА ОТ СПАМА И ЛИШНИХ СООБЩЕНИЙ ---

@dp.message()
async def handle_spam(message: types.Message):
    """
    Обрабатывает все сообщения, которые не являются командами или колбэками.
    """
    ignore_phrases = ["привет", "здравствуй", "хай", "что делаешь", "кто ты"]
    text = message.text.lower() if message.text else ""
    
    # Если это похоже на просто болтовню - игнорируем или шлем подсказку
    # Если бот сейчас занят (is_predicting), то блокируем жестче
    if is_predicting:
        return # Молчаливое игнорирование, чтобы не засорять чат во время расчетов

    # Иначе вежливо подсказываем
    await message.answer(
        "😕 Я понимаю только команды из меню.\n"
        "Пожалуйста, используйте кнопки или команду /start."
    )

# --- ЗАПУСК ---

async def main():
    # Удаляем вебхуки, чтобы бот мог работать в Long Polling (для Railway можно и вебхук, но LP проще для отладки)
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
