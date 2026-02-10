import asyncio
import logging
import os
from datetime import datetime, timedelta
import aiohttp
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Важно для Railway, чтобы не пытался открывать графическое окно
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

async def get_market_data():
    """
    Получает данные с CoinGecko API.
    CoinGecko не блокирует Railway и дает историю цены.
    """
    # Получаем историю BTC за последние 1-2 дня (чтобы хватило минутных данных)
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=1"
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status == 200:
                    data = await response.json()
                    prices = data.get('prices', []) # Формат: [[timestamp, price], ...]
                    
                    if not prices:
                        logging.error("CoinGecko вернул пустой список цен.")
                        return None

                    # Создаем DataFrame
                    df = pd.DataFrame(prices, columns=['timestamp', 'close'])
                    df['close_time'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # Берем последние 30 точек для анализа (чтобы хватило для обучения)
                    # CoinGecko может присылать данные не ровно по минуте, но мы берем то что есть
                    df = df.tail(30).reset_index(drop=True)
                    return df
                else:
                    logging.error(f"Ошибка CoinGecko HTTP: {response.status}")
                    return None
    except Exception as e:
        logging.error(f"Ошибка подключения к CoinGecko: {e}")
        return None

def calculate_rsi(series, period=14):
    """Расчет RSI (упрощенный для работы с произвольными временными рядами)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Избегаем деления на ноль
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- ФУНКЦИИ МОДЕЛИ И ПРЕДСКАЗАНИЯ ---

def predict_next_minute(df):
    """
    Готовит данные, обучает легкую нейросеть и делает прогноз.
    Адаптировано под данные CoinGecko.
    """
    df = df.copy()
    
    # 1. Добавляем индикаторы
    # RSI
    df['rsi'] = calculate_rsi(df['close'])
    # Изменение цены (важно для тренда)
    df['change'] = df['close'].diff()
    
    # Удаляем NaN, появившиеся после расчетов
    df.dropna(inplace=True)

    if len(df) < 10:
        logging.warning("Недостаточно данных для прогноза")
        return None, None

    # 2. Подготовка признаков (Features)
    # Используем цену, RSI и изменение цены
    data = df[['close', 'rsi', 'change']].values
    
    # Нормализация
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # 3. Формируем обучающие выборки
    X, y = [], []
    look_back = 10
    
    if len(scaled_data) <= look_back:
        return None, None

    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:i + look_back].flatten()) 
        y.append(scaled_data[i + look_back][0]) # Предсказываем цену (0-й индекс в массиве признаков)

    if not X:
        return None, None

    X = np.array(X)
    y = np.array(y)

    # 4. Обучение модели
    model = MLPRegressor(hidden_layer_sizes=(10, 5), max_iter=500, random_state=42)
    try:
        model.fit(X, y)
    except Exception as e:
        logging.error(f"Ошибка обучения модели: {e}")
        return None, None

    # 5. Предсказание
    last_window = scaled_data[-look_back:].flatten().reshape(1, -1)
    predicted_scaled = model.predict(last_window)
    
    # Денормализация
    # Создаем массив-заглушку, чтобы восстановить масштаб цены
    dummy_array = np.zeros((1, 3))
    dummy_array[0, 0] = predicted_scaled[0]
    # Для RSI и change берем последние известные значения, чтобы scaler не ругался
    dummy_array[0, 1] = scaled_data[-1, 1] 
    dummy_array[0, 2] = scaled_data[-1, 2] 
    
    predicted_price_full = scaler.inverse_transform(dummy_array)
    predicted_price = predicted_price_full[0, 0]

    # Время следующей минуты (относительно последней точки данных)
    last_time = df['close_time'].iloc[-1]
    next_time = last_time + timedelta(minutes=1)

    return df, predicted_price, next_time

# --- ГЕНЕРАЦИЯ ГРАФИКА ---

def create_plot(df, predicted_price, next_time):
    """
    Рисует график: 10 предыдущих точек + 1 предсказанная.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Берем последние 10 точек для красоты
    plot_df = df.tail(10).copy()
    
    # График истории
    ax.plot(plot_df['close_time'], plot_df['close'], 
            label='История', color='cyan', marker='o', linestyle='-')

    # График предсказания (пунктир)
    ax.plot([plot_df['close_time'].iloc[-1], next_time],
            [plot_df['close'].iloc[-1], predicted_price],
            label='Прогноз AI', color='lime', linestyle='--', marker='x')
    
    # Точка предсказания
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
    ax.set_title(f"BTC/USDT Прогноз (CoinGecko Data)", color='white', fontsize=14)
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
        "Я анализирую данные в реальном времени, использую нейросеть и индикаторы, "
        "чтобы предсказать движение цены на следующую минуту.\n\n"
        "Данные поступают с CoinGecko.\n\n"
        "Нажмите кнопку ниже, чтобы получить прогноз.",
        reply_markup=keyboard
    )

@dp.callback_query(F.data == "help_info")
async def show_help(callback: types.CallbackQuery):
    await callback.message.edit_text(
        "📊 **Как это работает:**\n"
        "1. Я получаю данные с CoinGecko (история цены).\n"
        "2. Рассчитываю RSI и тренд изменений цены.\n"
        "3. Обучаю легкую нейросеть (MLP) на последних данных.\n"
        "4. Рисую график с историей и прогнозом на минуту.\n\n"
        "⚠️ Это не финансовая рекомендация.",
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
    await callback.message.edit_text("⏳ Получаю данные с CoinGecko и обучаю нейросеть... Это займет несколько секунд.")

    try:
        # 1. Получаем данные
        df_raw = await get_market_data()
        if df_raw is None:
            await callback.message.edit_text("❌ Не удалось получить данные. Попробуйте позже.")
            return

        # 2. Предсказываем
        df_processed, pred_price, next_time = predict_next_minute(df_raw)
        
        if pred_price is None:
            await callback.message.edit_text("❌ Не удалось построить прогноз (мало данных или ошибка модели).")
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
    text = message.text.lower() if message.text else ""
    
    if is_predicting:
        return # Молчаливое игнорирование

    # Иначе вежливо подсказываем
    await message.answer(
        "😕 Я понимаю только команды из меню.\n"
        "Пожалуйста, используйте кнопки или команду /start."
    )

# --- ЗАПУСК ---

async def main():
    # Удаляем вебхуки перед запуском polling
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
