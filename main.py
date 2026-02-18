import asyncio
import logging
import os
from datetime import datetime, timedelta
from io import BytesIO
from typing import Optional, Tuple

import aiohttp
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import BufferedInputFile, ReplyKeyboardMarkup, KeyboardButton
from dotenv import load_dotenv
from collections import defaultdict
from zoneinfo import ZoneInfo

# --- КОНФИГУРАЦИЯ ---
load_dotenv()
TELEGRAM_TOKEN = "2122435147:AAG_52ELCHjFnXNxcAP4i5xNAal9I91xNTM" 

# ВРЕМЯ
TIMEZONE_STR = "Europe/Moscow"
LOCAL_TIMEZONE = ZoneInfo(TIMEZONE_STR)

# --- НАСТРОЙКИ СТРАТЕГИИ (LHLP Optimized) ---
STRATEGY_CONFIG = {
    'sma_volume_period': 50,   # Период SMA объема
    'rsi_period': 14,          # Период RSI
    'rsi_long_enter': 30,      # Порог RSI для Long
    'rsi_short_enter': 70,     # Порог RSI для Short
}

STARTING_BALANCE = 100
COOLDOWN_SECONDS = 300 # 5 минут

# Монеты для анализа
COINS = {
    'BTC': {'id': 'bitcoin', 'symbol': 'BTC/USDT'},
    'ETH': {'id': 'ethereum', 'symbol': 'ETH/USDT'},
    'TON': {'id': 'the-open-network', 'symbol': 'TON/USDT'}
}

logging.basicConfig(level=logging.INFO)

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# --- БАЗА ДАННЫХ ---
def get_default_user_data():
    return {
        'balance': STARTING_BALANCE, 
        'last_prediction_time': None,
        'coins': {
            'BTC': {'balance': STARTING_BALANCE, 'last_time': None},
            'ETH': {'balance': STARTING_BALANCE, 'last_time': None},
            'TON': {'balance': STARTING_BALANCE, 'last_time': None}
        }
    }

user_limits = defaultdict(get_default_user_data)

# --- ФУНКЦИИ ---

async def get_market_data(coin_id: str):
    """Получает данные с CoinGecko API. Возвращает DataFrame или None."""
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
                    
                    if not prices or not volumes: return None

                    df_prices = pd.DataFrame(prices, columns=['timestamp', 'close'])
                    df_volumes = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                    
                    df = pd.merge(df_prices, df_volumes, on='timestamp', how='left')
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    df = df.resample('5min').agg({'close': 'last', 'volume': 'sum'})
                    df.dropna(inplace=True)
                    
                    df.reset_index(inplace=True)
                    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(LOCAL_TIMEZONE)
                    df = df.rename(columns={'timestamp': 'close_time'})
                    
                    # Берем 80 свечей для расчета истории
                    df = df.tail(80).reset_index(drop=True)
                    return df
                else:
                    logging.error(f"Ошибка CoinGecko HTTP: {response.status}")
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

def analyze_with_strategy(df: pd.DataFrame):
    """Анализ на основе стратегии LHLP Optimized."""
    df = df.copy()
    
    # Индикаторы
    df['sma_vol'] = df['volume'].rolling(window=STRATEGY_CONFIG['sma_volume_period']).mean()
    df['rsi'] = calculate_rsi(df['close'], STRATEGY_CONFIG['rsi_period'])
    
    df.dropna(inplace=True)
    
    if len(df) < 5: return df, "NO_DATA", 0, 0

    last = df.iloc[-1]
    current_price = last['close']
    current_rsi = last['rsi']
    current_vol = last['volume']
    avg_vol = last['sma_vol']
    
    signal = "FLAT" # По умолчанию боковик
    confidence = 0.0
    target_price = current_price
    
    # --- ЛОГИКА СТРАТЕГИИ ---
    volume_spike = current_vol > avg_vol
    
    # LONG: Vol > SMA & RSI < 30
    if volume_spike and (current_rsi < STRATEGY_CONFIG['rsi_long_enter']):
        signal = "LONG"
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
        confidence = min(30 + (vol_ratio * 10) + (STRATEGY_CONFIG['rsi_long_enter'] - current_rsi), 95)
        volatility = df['close'].pct_change().tail(5).std()
        target_price = current_price * (1 + volatility * (confidence/50))

    # SHORT: Vol > SMA & RSI > 70
    elif volume_spike and (current_rsi > STRATEGY_CONFIG['rsi_short_enter']):
        signal = "SHORT"
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
        confidence = min(30 + (vol_ratio * 10) + (current_rsi - STRATEGY_CONFIG['rsi_short_enter']), 95)
        volatility = df['close'].pct_change().tail(5).std()
        target_price = current_price * (1 - volatility * (confidence/50))
        
    else:
        # УЛУЧШЕНИЕ НЕЙТРАЛЬНОСТИ: Определяем микро-тренд по последним 3 свечам
        trend = df['close'].iloc[-1] - df['close'].iloc[-3]
        
        # Если тренд вверх
        if trend > 0:
             signal = "FLAT_UP" # Накопление/Рост
             target_price = current_price + trend * 0.5 # Продолжаем инерцию
        # Если тренд вниз
        elif trend < 0:
             signal = "FLAT_DOWN # Слабое снижение
             target_price = current_price + trend * 0.5
        else:
             signal = "FLAT"
        
        confidence = 0 # Для FLAT уверенность не важна, убираем путаницу

    return df, signal, target_price, confidence

def create_plot(df, target_price, signal, coin_symbol):
    """Отрисовка сочного графика с неоновой темой."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # --- ЦВЕТА (NEON THEME) ---
    bg_color = '#0a0a12'
    grid_color = '#2a2a3a'
    line_hist_color = '#00f2ff' # Яркий голубой (Cyan)
    line_pred_color = '#ff0055' # Малиновый (Pink/Red) для контраста или Лайм
    text_color = 'white'
    
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    plot_df = df.tail(20).copy()
    plot_df['close_time_plot'] = plot_df['close_time'].dt.tz_localize(None)
    
    last_time = plot_df['close_time_plot'].iloc[-1]
    next_time = last_time + timedelta(minutes=5)
    current_price = plot_df['close'].iloc[-1]
    
    # Выбор цвета прогноза
    if "LONG" in signal: 
        pred_color = '#00ff88' # Яркий зеленый (Lime)
    elif "SHORT" in signal: 
        pred_color = '#ff3333' # Яркий красный
    elif "UP" in signal: 
        pred_color = '#ffff33' # Желтый (нейтральный рост)
    elif "DOWN" in signal: 
        pred_color = '#ff9900' # Оранжевый (нейтральное падение)
    else: 
        pred_color = '#888888' # Серый

    # --- ОТРИСОВКА ---
    
    # Линия истории (жирная и яркая)
    ax.plot(plot_df['close_time_plot'], plot_df['close'], 
            color=line_hist_color, marker='o', linestyle='-', markersize=8, 
            zorder=2, linewidth=2, label='История')
    
    # Линия прогноза (пунктир)
    ax.plot([last_time, next_time], [current_price, target_price],
            color=pred_color, linestyle='--', marker='x', markersize=12, 
            zorder=3, linewidth=2.5, label=f'Прогноз: {signal.replace("_", " ")}')
    
    # Точка прогноза (светящаяся)
    ax.scatter(next_time, target_price, color=pred_color, s=250, zorder=4, 
               edgecolors='white', linewidth=2)

    # Текст на точках
    for x, y, time_obj in zip(plot_df['close_time_plot'], plot_df['close'], plot_df['close_time']):
        time_str = time_obj.strftime('%H:%M')
        price_str = f"{y:,.0f}" if y > 10 else f"{y:,.2f}"
        
        ax.annotate(time_str, (x, y), textcoords="offset points", xytext=(0,15), 
                    ha='center', fontsize=9, color='#ffff00', fontweight='bold')
        ax.annotate(price_str, (x, y), textcoords="offset points", xytext=(0,-15), 
                    ha='center', fontsize=8, color='white')

    # Точка прогноза текст
    pred_time_str = next_time.strftime('%H:%M')
    if target_price > 10:
        pred_price_str = f"{target_price:,.0f}"
    else:
        pred_price_str = f"{target_price:,.2f}"
    
    ax.annotate(pred_time_str, (next_time, target_price), textcoords="offset points", xytext=(0,18), 
                ha='center', fontsize=10, color=pred_color, fontweight='bold')
    ax.annotate(pred_price_str, (next_time, target_price), textcoords="offset points", xytext=(0,-18), 
                ha='center', fontsize=9, color='white', fontweight='bold')

    # Оформление осей
    ax.get_xaxis().set_visible(False)
    ax.set_title(f"{coin_symbol} Strategy Analysis", color='white', fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel("Цена ($)", color='white', fontsize=12)
    ax.grid(True, color=grid_color, linestyle='--', alpha=0.5)
    
    # Легенда
    ax.legend(loc='upper left', facecolor=bg_color, edgecolor=grid_color, labelcolor='white')

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor=bg_color)
    plt.close(fig)
    buf.seek(0)
    return BufferedInputFile(buf.getvalue(), f"{coin_symbol.lower()}_prediction.png")

# --- ХЕНДЛЕРЫ ---

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
    await bot.delete_webhook(drop_pending_updates=True)
    logging.info("Бот запущен.")

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    user_id = message.from_user.id
    if user_id not in user_limits:
        user_limits[user_id] = get_default_user_data()
        
    await message.answer(
        "👋 Добро пожаловать в AI Strategy Bot!\n\n"
        "🧠 **Ядро:** LHLP Optimized (Volume + RSI).\n"
        "🎨 **Обновлен дизайн:** Неоновые графики.\n"
        "📊 Раздельный баланс для BTC, ETH, TON.\n"
        f"🕐 Часовой пояс: {TIMEZONE_STR}.",
        reply_markup=main_keyboard,
        parse_mode="Markdown"
    )

@dp.message(F.text == "ℹ️ Информация")
async def cmd_info(message: types.Message):
    await message.answer(
        f"📊 **Логика стратегии:**\n"
        f"1. **LONG:** Объем > SMA и RSI < {STRATEGY_CONFIG['rsi_long_enter']}.\n"
        f"2. **SHORT:** Объем > SMA и RSI > {STRATEGY_CONFIG['rsi_short_enter']}.\n"
        f"3. **FLAT:** Если нет явного сигнала, идет инерционный прогноз.\n\n"
        "⚠️ *Не финансовый совет.*",
        parse_mode="Markdown"
    )

@dp.message(F.text == "💳 Мой баланс")
async def cmd_balance(message: types.Message):
    user_data = user_limits.get(message.from_user.id, get_default_user_data())
    balances = user_data['coins']
    
    text = (
        f"💳 **Баланс прогнозов:**\n\n"
        f" 🟡 BTC: `{balances['BTC']['balance']}`\n"
        f" 🔵 ETH: `{balances['ETH']['balance']}`\n"
        f" 🔷 TON: `{balances['TON']['balance']}`"
    )
    await message.answer(text, parse_mode="Markdown")

@dp.message(F.text == "💹 Цена сейчас")
async def cmd_current_price(message: types.Message):
    """
    Получает СВЕЖУЮ цену, запрашивая последние данные свечей.
    НЕ использует simple/price, чтобы избежать кэша.
    """
    status_msg = await message.answer("🔄 Получение актуальных цен (Direct Feed)...", parse_mode="Markdown")
    
    prices_text = "💹 **Актуальные цены сейчас:**\n\n"
    tasks = []

    # Запускаем параллельные запросы для скорости
    async def fetch_price(coin_name):
        df = await get_market_data(COINS[coin_name]['id'])
        if df is not None and not df.empty:
            price = df['close'].iloc[-1]
            last_candle_time = df['close_time'].iloc[-1]
            now = datetime.now(LOCAL_TIMEZONE)
            
            # Проверяем, насколько свежая свеча (должна быть не старше 10 мин)
            if (now - last_candle_time).total_seconds() < 600:
                return coin_name, price, "OK"
            else:
                return coin_name, price, "DELAYED"
        return coin_name, None, "ERROR"

    for name in COINS.keys():
        tasks.append(fetch_price(name))
    
    results = await asyncio.gather(*tasks)
    
    has_error = False
    for name, price, status in results:
        if price:
            p_str = f"{price:,.2f}" if price < 100 else f"{price:,.0f}"
            if status == "OK":
                prices_text += f"• **{name}:** `${p_str}`\n"
            else:
                prices_text += f"• **{name}:** `${p_str}` ⚠️ (задержка)\n"
        else:
            has_error = True
            prices_text += f"• **{name}:** `Ошибка`\n"

    if has_error:
        prices_text += "\n_Не удалось получить данные от CoinGecko._"

    await status_msg.edit_text(prices_text, parse_mode="Markdown")

# Универсальная функция анализа
async def process_analysis(message: types.Message, coin_name: str):
    user_id = message.from_user.id
    user_data = user_limits[user_id]
    coin_data = user_data['coins'][coin_name]
    
    # Проверка баланса конкретной монеты
    if coin_data['balance'] <= 0:
        await message.answer(f"❌ У вас закончились прогнозы для {coin_name}.")
        return

    # Проверка задержки конкретной монеты
    last_time = coin_data['last_time']
    now = datetime.now(LOCAL_TIMEZONE)
    
    if last_time:
        diff = (now - last_time).total_seconds()
        if diff < COOLDOWN_SECONDS:
            remain = int(COOLDOWN_SECONDS - diff)
            await message.answer(f"⏳ Подождите {remain} сек перед новым запросом {coin_name}.")
            return

    status_msg = await message.answer(f"⏳ Анализ {coin_name} (Strategy Mode)...")

    try:
        coin_info = COINS[coin_name]
        df_raw = await get_market_data(coin_info['id'])
        
        if df_raw is None:
            await status_msg.edit_text("❌ Ошибка получения данных.")
            return

        df_processed, signal, pred_price, confidence = analyze_with_strategy(df_raw)
        
        if signal == "NO_DATA":
            await status_msg.edit_text("❌ Мало данных для построения модели.")
            return

        plot_buf = create_plot(df_processed, pred_price, signal, coin_info['symbol'])
        current_price = df_processed['close'].iloc[-1]
        
        diff = pred_price - current_price
        
        # Формирование красивого ответа
        if "LONG" in signal:
            emoji = "🚀"
            status_text = f"LONG (Уверенность: {confidence:.0f}%)"
        elif "SHORT" in signal:
            emoji = "🔻"
            status_text = f"SHORT (Уверенность: {confidence:.0f}%)"
        elif "UP" in signal:
            emoji = "↗️"
            status_text = "FLAT (Слабый рост)"
        elif "DOWN" in signal:
            emoji = "↙️"
            status_text = "FLAT (Слабое падение)"
        else:
            emoji = "⏸"
            status_text = "FLAT (Боковик)"
        
        next_time = df_processed['close_time'].iloc[-1] + timedelta(minutes=5)
        time_str = next_time.strftime('%H:%M')
        
        caption = (
            f"{emoji} **Прогноз {coin_info['symbol']} (5m)**\n\n"
            f"Сигнал: **{status_text}**\n\n"
            f"Текущая: `${current_price:,.2f}`\n"
            f"Цель на {time_str}: `${pred_price:,.2f}`\n"
            f"Изменение: `{diff:+,.2f}` $\n\n"
            f"Осталось {coin_name} прогнозов: `{coin_data['balance'] - 1}`"
        )

        # Списываем баланс и ставим таймер
        user_limits[user_id]['coins'][coin_name]['balance'] -= 1
        user_limits[user_id]['coins'][coin_name]['last_time'] = datetime.now(LOCAL_TIMEZONE)

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
