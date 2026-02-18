import asyncio
import logging
import os
from datetime import datetime, timedelta
from io import BytesIO
from typing import Union

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

# --- НАСТРОЙКИ СТРАТЕГИИ ---
STRATEGY_CONFIG = {
    'sma_volume_period': 50,
    'rsi_period': 14,
    'rsi_long_enter': 30,
    'rsi_short_enter': 70,
}

STARTING_BALANCE = 100
CANDLE_INTERVAL = 5 # Минуты

# Монеты
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
        'coins': {
            'BTC': {'balance': STARTING_BALANCE, 'last_candle_time': None},
            'ETH': {'balance': STARTING_BALANCE, 'last_candle_time': None},
            'TON': {'balance': STARTING_BALANCE, 'last_candle_time': None}
        }
    }

user_limits = defaultdict(get_default_user_data)

# --- ФУНКЦИИ ДАННЫХ ---

async def get_market_data(coin_id: str) -> Union[pd.DataFrame, str, None]:
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=1"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=20)) as response:
                if response.status == 200:
                    data = await response.json()
                    prices = data.get('prices', [])
                    volumes = data.get('total_volumes', []) 
                    
                    if not prices or not volumes: return None

                    df = pd.DataFrame(prices, columns=['timestamp', 'close'])
                    df_vol = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                    df['volume'] = df_vol['volume']
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    df = df.resample('5min').agg({'close': 'last', 'volume': 'sum'})
                    df.dropna(inplace=True)
                    
                    df.reset_index(inplace=True)
                    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(LOCAL_TIMEZONE)
                    df = df.rename(columns={'timestamp': 'close_time'})
                    
                    return df.tail(80).reset_index(drop=True)
                
                elif response.status == 429:
                    return "RATE_LIMIT"
                else:
                    logging.error(f"Ошибка HTTP: {response.status}")
                    return None
    except Exception as e:
        logging.error(f"Ошибка сети: {e}")
        return None

async def get_simple_prices():
    ids = ','.join([c['id'] for c in COINS.values()])
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    return "RATE_LIMIT"
                return None
    except Exception:
        return None

# --- ЛОГИКА СТРАТЕГИИ ---

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def analyze_with_strategy(df: pd.DataFrame):
    df = df.copy()
    
    df['sma_vol'] = df['volume'].rolling(window=STRATEGY_CONFIG['sma_volume_period']).mean()
    df['rsi'] = calculate_rsi(df['close'], STRATEGY_CONFIG['rsi_period'])
    
    df.dropna(inplace=True)
    
    if len(df) < 5: return df, "NO_DATA", 0, 0

    last = df.iloc[-1]
    current_price = last['close']
    current_rsi = last['rsi']
    current_vol = last['volume']
    avg_vol = last['sma_vol']
    
    signal = "WAIT"
    confidence = 0.0
    target_price = current_price
    
    volume_spike = current_vol > avg_vol
    
    if volume_spike and (current_rsi < STRATEGY_CONFIG['rsi_long_enter']):
        signal = "LONG"
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
        confidence = min(30 + (vol_ratio * 10) + (STRATEGY_CONFIG['rsi_long_enter'] - current_rsi), 95)
        volatility = df['close'].pct_change().tail(5).std()
        target_price = current_price * (1 + volatility * (confidence/50))

    elif volume_spike and (current_rsi > STRATEGY_CONFIG['rsi_short_enter']):
        signal = "SHORT"
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
        confidence = min(30 + (vol_ratio * 10) + (current_rsi - STRATEGY_CONFIG['rsi_short_enter']), 95)
        volatility = df['close'].pct_change().tail(5).std()
        target_price = current_price * (1 - volatility * (confidence/50))
        
    return df, signal, target_price, confidence

def format_price(price: float):
    if price > 1000:
        return f"{price:,.0f}"
    elif price > 10:
        return f"{price:,.2f}"
    else:
        return f"{price:,.4f}"

def create_plot(df, target_price, signal, coin_symbol):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    plot_df = df.tail(20).copy()
    plot_df['close_time_plot'] = plot_df['close_time'].dt.tz_localize(None)
    
    last_time = plot_df['close_time_plot'].iloc[-1]
    next_time = last_time + timedelta(minutes=5)
    current_price = plot_df['close'].iloc[-1]
    
    ax.plot(plot_df['close_time_plot'], plot_df['close'], 
            color='cyan', marker='o', linestyle='-', markersize=8, zorder=2)
    
    # Точка прогноза только для LONG/SHORT
    if signal in ["LONG", "SHORT"]:
        if signal == "LONG": pred_color = 'lime'
        elif signal == "SHORT": pred_color = 'red'
        
        ax.plot([last_time, next_time], [current_price, target_price],
                color=pred_color, linestyle='--', marker='x', markersize=10, zorder=3)
        ax.scatter(next_time, target_price, color=pred_color, s=200, zorder=4, edgecolors='white')
        
        pred_time_str = next_time.strftime('%H:%M')
        pred_price_str = format_price(target_price)
        
        ax.annotate(pred_time_str, (next_time, target_price), textcoords="offset points", xytext=(0,15), 
                    ha='center', fontsize=10, color=pred_color, fontweight='bold')
        ax.annotate(pred_price_str, (next_time, target_price), textcoords="offset points", xytext=(0,-15), 
                    ha='center', fontsize=9, color=pred_color, fontweight='bold')
    
    for x, y, time_obj in zip(plot_df['close_time_plot'], plot_df['close'], plot_df['close_time']):
        time_str = time_obj.strftime('%H:%M')
        price_str = format_price(y)
        
        ax.annotate(time_str, (x, y), textcoords="offset points", xytext=(0,12), 
                    ha='center', fontsize=9, color='yellow', fontweight='bold')
        ax.annotate(price_str, (x, y), textcoords="offset points", xytext=(0,-12), 
                    ha='center', fontsize=8, color='white')

    ax.get_xaxis().set_visible(False)
    title_suffix = f" ({signal})" if signal in ["LONG", "SHORT"] else " (No Signal)"
    ax.set_title(f"{coin_symbol} Strategy Analysis{title_suffix}", color='white', fontsize=16)
    ax.set_ylabel("Цена ($)", color='gray')
    ax.grid(True, color='gray', linestyle=':', alpha=0.3)
    
    legend_labels = ['История', f'Прогноз ({signal})'] if signal in ["LONG", "SHORT"] else ['История']
    ax.legend(legend_labels, loc='upper left')

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
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
        "👋 Добро пожаловать!\n\n"
        "🧠 **Стратегия:** LHLP Optimized.\n"
        "⏳ **Лимит:** 1 прогноз на 1 свечу (5 мин).\n"
        "🔎 Показываю только LONG и SHORT.\n"
        f"🕐 Часовой пояс: {TIMEZONE_STR}.",
        reply_markup=main_keyboard,
        parse_mode="Markdown"
    )

@dp.message(F.text == "ℹ️ Информация")
async def cmd_info(message: types.Message):
    await message.answer(
        f"📊 **Логика:**\n"
        f"LONG: Vol > SMA & RSI < {STRATEGY_CONFIG['rsi_long_enter']}.\n"
        f"SHORT: Vol > SMA & RSI > {STRATEGY_CONFIG['rsi_short_enter']}.\n\n"
        "⚠️ *Не финансовый совет.*",
        parse_mode="Markdown"
    )

@dp.message(F.text == "💳 Мой баланс")
async def cmd_balance(message: types.Message):
    user_data = user_limits.get(message.from_user.id, get_default_user_data())
    balances = user_data['coins']
    
    text = (
        f"💳 **Баланс:**\n\n"
        f" BTC: `{balances['BTC']['balance']}`\n"
        f" ETH: `{balances['ETH']['balance']}`\n"
        f" TON: `{balances['TON']['balance']}`"
    )
    await message.answer(text, parse_mode="Markdown")

@dp.message(F.text == "💹 Цена сейчас")
async def cmd_current_price(message: types.Message):
    status_msg = await message.answer("⏳ Получение цен...")
    data = await get_simple_prices()
    
    if data == "RATE_LIMIT":
        await status_msg.edit_text("⚠️ Сервер перегружен. Попробуйте чуть позже.")
        return
    
    if not data:
        await status_msg.edit_text("❌ Ошибка получения данных.")
        return

    prices_text = "💹 **Актуальные цены сейчас:**\n\n"
    
    for name, info in COINS.items():
        price = data.get(info['id'], {}).get('usd', None)
        if price:
            p_str = format_price(price)
            prices_text += f"• **{name}:** `${p_str}`\n"
        else:
            prices_text += f"• **{name}:** `Ошибка`\n"

    await status_msg.edit_text(prices_text, parse_mode="Markdown")

async def process_analysis(message: types.Message, coin_name: str):
    user_id = message.from_user.id
    user_data = user_limits[user_id]
    coin_data = user_data['coins'][coin_name]
    
    # Проверка баланса
    if coin_data['balance'] <= 0:
        await message.answer(f"❌ У вас закончились попытки для {coin_name}. Баланс: 0.")
        return

    status_msg = await message.answer(f"⏳ Анализ {coin_name}...")

    try:
        coin_info = COINS[coin_name]
        result = await get_market_data(coin_info['id'])
        
        if isinstance(result, str) and result == "RATE_LIMIT":
            await status_msg.edit_text("⚠️ Сервер перегружен (429). Попробуйте через минуту.")
            return
        
        if result is None:
            await status_msg.edit_text("❌ Ошибка сети.")
            return
        
        df_raw = result
        
        # --- ПРОВЕРКА ВРЕМЕНИ СВЕЧИ (НОВАЯ ЛОГИКА) ---
        last_candle_time = df_raw['close_time'].iloc[-1]
        user_last_time = coin_data['last_candle_time']
        
        # Если пользователь уже делал прогноз на ЭТУ ЖЕ свечу
        if user_last_time is not None and user_last_time >= last_candle_time:
            next_candle_time = last_candle_time + timedelta(minutes=CANDLE_INTERVAL)
            now = datetime.now(LOCAL_TIMEZONE)
            
            # Сколько секунд осталось до следующей свечи
            remain_sec = (next_candle_time - now).total_seconds()
            
            if remain_sec > 0:
                remain_int = int(remain_sec)
                await status_msg.edit_text(
                    f"⏳ Прогноз на эту свечу уже получен.\n"
                    f"Следующая свеча через {remain_int} сек.\n"
                    f"Осталось попыток: {coin_data['balance']}"
                )
            else:
                # Если время вышло, но данные еще не обновились в API
                await status_msg.edit_text(
                    f"⏳ Ждем обновления данных (свеча закрыта)...\n"
                    f"Попробуйте через 10-15 секунд."
                )
            return
            
        # Если мы здесь -> это новая свеча, делаем прогноз
        
        df_processed, signal, pred_price, confidence = analyze_with_strategy(df_raw)
        
        if signal == "NO_DATA":
            await status_msg.edit_text("❌ Мало данных.")
            return

        current_price = df_processed['close'].iloc[-1]
        
        # Списываем попытку и обновляем время свечи
        user_limits[user_id]['coins'][coin_name]['balance'] -= 1
        user_limits[user_id]['coins'][coin_name]['last_candle_time'] = last_candle_time
        
        # --- Логика ответа ---
        
        if signal == "WAIT":
            plot_buf = create_plot(df_processed, current_price, "WAIT", coin_info['symbol'])
            
            caption = (
                f"💤 **{coin_info['symbol']}**\n\n"
                f"Сигнал: **Нет сигнала (NEUTRAL)**\n\n"
                f"Текущая: `${format_price(current_price)}`\n"
                f"Условия для входа не выполнены.\n\n"
                f"Осталось попыток: `{coin_data['balance'] - 1}`"
            )
        else:
            diff = pred_price - current_price
            
            if signal == "LONG":
                emoji = "🚀"
                status_text = f"LONG (Уверенность: {confidence:.0f}%)"
            else: # SHORT
                emoji = "🔻"
                status_text = f"SHORT (Уверенность: {confidence:.0f}%)"
            
            plot_buf = create_plot(df_processed, pred_price, signal, coin_info['symbol'])
            
            caption = (
                f"{emoji} **Прогноз {coin_info['symbol']}**\n\n"
                f"Сигнал: **{status_text}**\n\n"
                f"Текущая: `${format_price(current_price)}`\n"
                f"Цель: `${format_price(pred_price)}`\n"
                f"Изменение: `{diff:+.2f}` $\n\n"
                f"Осталось попыток: `{coin_data['balance'] - 1}`"
            )

        await status_msg.delete()
        await bot.send_photo(
            chat_id=message.chat.id,
            photo=plot_buf,
            caption=caption,
            parse_mode="Markdown"
        )

    except Exception as e:
        logging.error(f"Критическая ошибка: {e}")
        await status_msg.edit_text("❌ Ошибка бота.")

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
