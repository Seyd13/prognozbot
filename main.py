import asyncio
import logging
from datetime import datetime, timedelta
from io import BytesIO
from typing import Optional, Tuple, Dict

import aiohttp
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import BufferedInputFile, ReplyKeyboardMarkup, KeyboardButton
from collections import defaultdict
from zoneinfo import ZoneInfo

# --- КОНФИГУРАЦИЯ ---
TELEGRAM_TOKEN = "2122435147:AAG_52ELCHjFnXNxcAP4i5xNAal9I91xNTM" # Вставьте свой токен

# ВРЕМЯ
TIMEZONE_STR = "Europe/Moscow"
LOCAL_TIMEZONE = ZoneInfo(TIMEZONE_STR)

# НАСТРОЙКИ СТРАТЕГИИ (LHLP Optimized)
# Подобраны для 5-минутного таймфрейма, можно менять
STRATEGY_PARAMS = {
    'sma_volume_period': 50,  # Период SMA объема (в оригинале 120, для 5м лучше меньше)
    'rsi_period': 14,         # Период RSI
    'rsi_long_threshold': 35, # Порог RSI для лонга (перепроданность) (было 30)
    'rsi_short_threshold': 70,# Порог RSI для шорта (перекупленность) (для улучшения шортов)
    'rsi_take_profit': 55     # Уровень RSI для фиксации прибыли (было 60)
}

STARTING_BALANCE = 100
COINS = {
    'BTC': {'id': 'bitcoin', 'symbol': 'BTC/USDT'},
    'ETH': {'id': 'ethereum', 'symbol': 'ETH/USDT'},
    'TON': {'id': 'the-open-network', 'symbol': 'TON/USDT'}
}

logging.basicConfig(level=logging.INFO)

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# --- БАЗА ДАННЫХ ---
user_limits = defaultdict(lambda: {'balance': STARTING_BALANCE, 'last_prediction_time': None})

# --- ФУНКЦИИ ДАННЫХ И АНАЛИЗА ---

async def get_market_data(coin_id: str):
    """
    Получает данные с CoinGecko API для конкретной монеты.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=1"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=20)) as response:
                if response.status == 200:
                    data = await response.json()
                    prices = data.get('prices', [])
                    volumes = data.get('total_volumes', []) 
                    
                    if not prices or not volumes:
                        return None

                    df_prices = pd.DataFrame(prices, columns=['timestamp', 'close'])
                    df_volumes = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                    
                    df = pd.merge(df_prices, df_volumes, on='timestamp', how='left')
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    # Ресемплирование: Цена - последняя, Объем - сумма за 5 минут
                    df = df.resample('5min').agg({
                        'close': 'last',
                        'volume': 'sum'
                    })
                    
                    df.dropna(inplace=True)
                    
                    df.reset_index(inplace=True)
                    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(LOCAL_TIMEZONE)
                    
                    df = df.rename(columns={'timestamp': 'close_time'})
                    
                    # Берем последние 100 свечей для расчета индикаторов
                    df = df.tail(100).reset_index(drop=True)
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

def calculate_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def analyze_strategy(df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, Optional[str], float]:
    """
    Анализ по стратегии LHLP Optimized.
    Возвращает DataFrame с индикаторами, сигнал и уверенность.
    """
    df = df.copy()
    
    # 1. Расчет индикаторов
    df['sma_vol'] = df['volume'].rolling(window=params['sma_volume_period']).mean()
    df['rsi'] = calculate_rsi(df['close'], params['rsi_period'])
    
    # Удаляем NaN, возникшие из-за скользящих окон
    df.dropna(inplace=True)
    
    if len(df) < 5:
        return df, None, 0.0

    # Последние значения
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    current_price = last['close']
    current_rsi = last['rsi']
    current_vol = last['volume']
    avg_vol = last['sma_vol']
    
    signal = "NEUTRAL"
    confidence = 0.0
    
    # --- ЛОГИКА LONG (Ваша стратегия) ---
    # Условие: Объем больше SMA И RSI ниже порога (перепроданность)
    is_volume_spike = current_vol > avg_vol
    is_oversold = current_rsi < params['rsi_long_threshold']
    
    # Проверяем, был ли сигнал на предыдущей свече (для уверенности) или сейчас
    long_condition = is_volume_spike and is_oversold
    
    # --- ЛОГИКА SHORT (Улучшенная) ---
    # Условие: Объем больше SMA И RSI выше порога (перекупленность)
    is_overbought = current_rsi > params['rsi_short_threshold']
    short_condition = is_volume_spike and is_overbought

    # --- Определение сигнала ---
    if long_condition:
        signal = "LONG"
        # Уверенность: насколько сильно превышен объем и насколько низко RSI
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
        rsi_dist = abs(params['rsi_long_threshold'] - current_rsi) # Чем ниже RSI, тем лучше
        confidence = min((vol_ratio - 1) * 50 + rsi_dist, 100) # Нормализация
        
    elif short_condition:
        signal = "SHORT"
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
        rsi_dist = abs(current_rsi - params['rsi_short_threshold']) # Чем выше RSI, тем лучше
        confidence = min((vol_ratio - 1) * 50 + rsi_dist, 100)
        
    else:
        # Если явного сигнала нет, смотрим тренд RSI для нейтрального прогноза цены
        signal = "NEUTRAL"
        # Простая экстраполяция для цены, если нет сигнала
        confidence = 0

    return df, signal, confidence

def predict_price_action(df: pd.DataFrame, signal: str, confidence: float, params: dict) -> Tuple[float, str]:
    """
    Генерирует прогноз цены на основе сигнала.
    Возвращает целевую цену и текстовое описание.
    """
    current_price = df['close'].iloc[-1]
    volatility = df['close'].pct_change().std() # Волатильность для расчета цели
    
    target_price = current_price
    action_text = "Флэт / Ожидание"
    
    # Коэффициент движения зависит от уверенности (0.5% - 2% движения)
    move_factor = 0.005 + (confidence / 100) * 0.015 
    
    if signal == "LONG":
        target_price = current_price * (1 + move_factor)
        action_text = f"🚀 **LONG Signal** (Уверенность: {confidence:.1f}%)"
    elif signal == "SHORT":
        target_price = current_price * (1 - move_factor)
        action_text = f"🔻 **SHORT Signal** (Уверенность: {confidence:.1f}%)"
    else:
        # Если нейтрально, предсказываем небольшое движение по тренду RSI
        rsi = df['rsi'].iloc[-1]
        if rsi > 50:
            target_price = current_price * (1 + volatility * 0.5) # Небольшой рост
            action_text = "↗️ Слабый тренд вверх (Нет явного сигнала)"
        else:
            target_price = current_price * (1 - volatility * 0.5) # Небольшое падение
            action_text = "↙️ Слабый тренд вниз (Нет явного сигнала)"

    return target_price, action_text

def create_plot(df: pd.DataFrame, target_price: float, signal: str, coin_symbol: str, params: dict):
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    plot_df = df.tail(20).copy()
    plot_df['close_time_plot'] = plot_df['close_time'].dt.tz_localize(None)
    
    # График цены
    ax1.plot(plot_df['close_time_plot'], plot_df['close'], 
            color='white', marker='o', linestyle='-', markersize=6, zorder=2, label='Цена')
    
    # Определяем цвет прогноза
    pred_color = 'gray'
    if signal == "LONG": pred_color = 'lime'
    elif signal == "SHORT": pred_color = 'red'
    
    # Точка текущая и Прогноз
    last_time = plot_df['close_time_plot'].iloc[-1]
    next_time = last_time + timedelta(minutes=5)
    current_price = plot_df['close'].iloc[-1]
    
    # Линия прогноза
    ax1.plot([last_time, next_time], [current_price, target_price],
            color=pred_color, linestyle='--', marker='x', markersize=10, zorder=3, linewidth=2)
    
    ax1.scatter(next_time, target_price, color=pred_color, s=150, zorder=4, edgecolors='white', linewidth=1.5)

    # Подписи
    ax1.set_title(f"{coin_symbol} Strategy Analysis", color='white', fontsize=16, fontweight='bold')
    ax1.grid(True, color='gray', linestyle=':', alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Подграфик Volume
    colors = ['green' if plot_df['close'].iloc[i] >= plot_df['close'].iloc[i-1] else 'red' 
              for i in range(1, len(plot_df))]
    colors.insert(0, 'gray') # Первая свеча
    
    ax2.bar(plot_df['close_time_plot'], plot_df['volume'], color=colors, alpha=0.6)
    ax2.plot(plot_df['close_time_plot'], plot_df['sma_vol'], color='yellow', linestyle='-', linewidth=1.5, label='SMA Vol')
    ax2.set_ylabel("Volume", color='gray')
    ax2.grid(True, color='gray', linestyle=':', alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Убираем даты с оси X
    ax1.get_xaxis().set_visible(False)
    ax2.tick_params(axis='x', rotation=45)

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return BufferedInputFile(buf.getvalue(), f"{coin_symbol.lower()}_prediction.png")

# --- ХЕНДЛЕРЫ ---

# Клавиатуры
main_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="📊 Анализ BTC"), KeyboardButton(text="📊 Анализ ETH")],
        [KeyboardButton(text="📊 Анализ TON")],
        [KeyboardButton(text="💹 Цена сейчас")],
        [KeyboardButton(text="ℹ️ Информация"), KeyboardButton(text="💳 Мой баланс")]
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
        "👋 Добро пожаловать в AI Strategy Bot!\n\n"
        "Используется стратегия: **LHLP Optimized** (Volume + RSI).\n"
        "Добавлены монеты: BTC, ETH, TON.\n"
        "Улучшен анализ шортов.\n"
        f"Часовой пояс: {TIMEZONE_STR}.",
        reply_markup=main_keyboard,
        parse_mode="Markdown"
    )

@dp.message(F.text == "ℹ️ Информация")
async def cmd_info(message: types.Message):
    await message.answer(
        f"📊 **Логика стратегии:**\n"
        f"1. **LONG:** Объем > SMA Vol и RSI < {STRATEGY_PARAMS['rsi_long_threshold']}.\n"
        f"2. **SHORT:** Объем > SMA Vol и RSI > {STRATEGY_PARAMS['rsi_short_threshold']}.\n"
        f"3. **Фильтр:** Анализ 5-минутных свечей.\n\n"
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

@dp.message(F.text == "💹 Цена сейчас")
async def cmd_current_price(message: types.Message):
    status_msg = await message.answer("⏳ Получение актуальных цен...")
    
    prices_text = "💹 **Актуальные цены:**\n\n"
    
    try:
        async with aiohttp.ClientSession() as session:
            # Запрашиваем цены для всех монет одним запросом для скорости
            ids = ','.join([c['id'] for c in COINS.values()])
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    for name, info in COINS.items():
                        price = data.get(info['id'], {}).get('usd', 'N/A')
                        if isinstance(price, float):
                            prices_text += f"• **{name}:** `${price:.2f}`\n"
                        else:
                            prices_text += f"• **{name}:** `Error`\n"
                else:
                    prices_text = "❌ Ошибка получения данных."

        await status_msg.edit_text(prices_text, parse_mode="Markdown")

    except Exception as e:
        logging.error(f"Ошибка цен: {e}")
        await status_msg.edit_text("❌ Ошибка при получении цен.")

# Универсальный обработчик анализа
async def process_analysis(message: types.Message, coin_name: str):
    user_id = message.from_user.id
    
    if user_limits[user_id]['balance'] <= 0:
        await message.answer("❌ У вас закончились прогнозы. Баланс: 0.")
        return

    last_time = user_limits[user_id]['last_prediction_time']
    if last_time:
        now = datetime.now(LOCAL_TIMEZONE)
        if (now - last_time).total_seconds() < 300:
            remain = int(300 - (now - last_time).total_seconds())
            await message.answer(f"⏳ Подождите {remain} сек перед новым запросом.")
            return

    status_msg = await message.answer(f"⏳ Анализ {coin_name} (Volume + RSI Strategy)...")

    try:
        coin_data = COINS[coin_name]
        df_raw = await get_market_data(coin_data['id'])
        
        if df_raw is None or len(df_raw) < 60:
            await status_msg.edit_text("❌ Ошибка получения данных или мало истории.")
            return

        # Шаг 1: Анализ стратегии
        df_processed, signal, confidence = analyze_strategy(df_raw, STRATEGY_PARAMS)
        
        # Шаг 2: Расчет целевой цены
        target_price, action_text = predict_price_action(df_processed, signal, confidence, STRATEGY_PARAMS)
        
        # Шаг 3: Генерация графика
        plot_buf = create_plot(df_processed, target_price, signal, coin_data['symbol'], STRATEGY_PARAMS)
        
        current_price = df_processed['close'].iloc[-1]
        next_time = df_processed['close_time'].iloc[-1] + timedelta(minutes=5)
        time_str = next_time.strftime('%H:%M')
        
        # Формирование сообщения
        diff = target_price - current_price
        diff_percent = (diff / current_price) * 100
        
        caption = (
            f"🎯 **Прогноз {coin_data['symbol']} (5m)**\n\n"
            f"{action_text}\n\n"
            f"Текущая: `${current_price:.2f}`\n"
            f"Цель на {time_str}: `${target_price:.2f}`\n"
            f"Изменение: `{diff_percent:+.2f}%`\n\n"
            f"💰 Баланс: `{user_limits[user_id]['balance'] - 1}`"
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
        import traceback
        traceback.print_exc()
        await status_msg.edit_text("❌ Произошла ошибка бота.")

# Привязка кнопок к функции анализа
@dp.message(F.text == "📊 Анализ BTC")
async def cmd_btc(message: types.Message):
    await process_analysis(message, "BTC")

@dp.message(F.text == "📊 Анализ ETH")
async def cmd_eth(message: types.Message):
    await process_analysis(message, "ETH")

@dp.message(F.text == "📊 Анализ TON")
async def cmd_ton(message: types.Message):
    await process_analysis(message, "TON")

async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
