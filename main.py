import asyncio
import logging
import os
from datetime import datetime, timedelta
from io import BytesIO
from typing import Union, Set

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

# --- НАСТРОЙКИ СТРАТЕГИИ (СДЕЛАНО ЧУВСТВИТЕЛЬНЕЕ) ---
STRATEGY_CONFIG = {
    'sma_volume_period': 50,
    'rsi_period': 14,
    # Было 30/70, стало 40/60 - ловим чаще
    'rsi_long_enter': 40,  
    'rsi_short_enter': 60, 
}

CANDLE_INTERVAL = 5 # Минуты

# Хранилище подписчиков
subscribers: Set[int] = set() 

# Монеты
COINS = {
    'BTC': {'id': 'bitcoin', 'symbol': 'BTC/USDT'},
    'ETH': {'id': 'ethereum', 'symbol': 'ETH/USDT'},
    'TON': {'id': 'the-open-network', 'symbol': 'TON/USDT'}
}

logging.basicConfig(level=logging.INFO)

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# --- ФУНКЦИИ ДАННЫХ ---

async def get_market_data(coin_id: str) -> Union[pd.DataFrame, None]:
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=1"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as response:
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
                else:
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
    
    # Логика LONG: если объем выше среднего И RSI меньше границы (40)
    if volume_spike and (current_rsi < STRATEGY_CONFIG['rsi_long_enter']):
        signal = "LONG"
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
        confidence = min(30 + (vol_ratio * 10) + (STRATEGY_CONFIG['rsi_long_enter'] - current_rsi), 95)
        volatility = df['close'].pct_change().tail(5).std()
        target_price = current_price * (1 + volatility * (confidence/50))

    # Логика SHORT: если объем выше среднего И RSI выше границы (60)
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

def format_diff(diff: float):
    if abs(diff) > 10:
        return f"{diff:+,.2f}"
    else:
        return f"{diff:+,.4f}"

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
    
    # График рисуем только если есть сигнал (но функция вызывается только тогда)
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
    ax.set_title(f"{coin_symbol} Strategy Analysis ({signal})", color='white', fontsize=16)
    ax.set_ylabel("Цена ($)", color='gray')
    ax.grid(True, color='gray', linestyle=':', alpha=0.3)
    
    ax.legend(['История', f'Прогноз ({signal})'], loc='upper left')

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return BufferedInputFile(buf.getvalue(), f"{coin_symbol.lower()}_prediction.png")

# --- РАССЫЛКА (SCHEDULER) ---

async def broadcast_signal(coin_name: str):
    if not subscribers:
        return

    coin_info = COINS[coin_name]
    logging.info(f"Анализ {coin_name} для {len(subscribers)} подписчиков...")
    
    result = await get_market_data(coin_info['id'])
    
    if result is None:
        logging.warning(f"Нет данных для {coin_name}")
        return
    
    df_processed, signal, pred_price, confidence = analyze_with_strategy(result)
    
    # ГЛАВНОЕ ИЗМЕНЕНИЕ: Если сигнала нет (WAIT) - просто выходим, ничего не шлем
    if signal not in ["LONG", "SHORT"]:
        logging.info(f"{coin_name}: Сигнала нет (WAIT). Пропуск рассылки.")
        return

    current_price = df_processed['close'].iloc[-1]
    
    # Генерируем график
    plot_buf = create_plot(df_processed, pred_price, signal, coin_info['symbol'])
    
    # Формируем сообщение (логика WAIT удалена, так как мы уже отсеяли выше)
    diff = pred_price - current_price
    if signal == "LONG":
        emoji = "🚀"
        status_text = f"LONG (Уверенность: {confidence:.0f}%)"
    else: # SHORT
        emoji = "🔻"
        status_text = f"SHORT (Уверенность: {confidence:.0f}%)"
    
    caption = (
        f"{emoji} **Прогноз {coin_info['symbol']}**\n\n"
        f"Сигнал: **{status_text}**\n\n"
        f"Текущая: `${format_price(current_price)}`\n"
        f"Цель: `${format_price(pred_price)}`\n"
        f"Изменение: `{format_diff(diff)}` $"
    )

    # Рассылаем всем
    tasks = []
    for user_id in subscribers:
        tasks.append(bot.send_photo(chat_id=user_id, photo=plot_buf, caption=caption, parse_mode="Markdown"))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Чистим список от заблокировавших бот
    for user_id, res in zip(list(subscribers), results):
        if isinstance(res, Exception):
            logging.warning(f"Ошибка отправки юзеру {user_id}. Удаляю.")
            subscribers.discard(user_id)

async def scheduler_loop():
    while True:
        now = datetime.now(LOCAL_TIMEZONE)
        
        # Расчет времени до следующей свечи
        seconds_to_next = CANDLE_INTERVAL * 60 - (now.minute % CANDLE_INTERVAL) * 60 - now.second
        
        if seconds_to_next > 5:
            logging.info(f"До следующей свечи {seconds_to_next} сек. Жду.")
            await asyncio.sleep(seconds_to_next)
        
        logging.info("Новая свеча! Запускаю анализ...")
        
        for coin_name in COINS.keys():
            await broadcast_signal(coin_name)
            await asyncio.sleep(5) # Пауза между монетами
        
        # Пауза чтобы не зацепить текущую минуту повторно
        await asyncio.sleep(15)

# --- ХЕНДЛЕРЫ ---

main_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="🚀 Подписаться на сигналы")],
        [KeyboardButton(text="💹 Цена сейчас")],
        [KeyboardButton(text="ℹ️ Информация")]
    ],
    resize_keyboard=True
)

@dp.startup()
async def on_startup():
    await bot.delete_webhook(drop_pending_updates=True)
    asyncio.create_task(scheduler_loop())
    logging.info("Бот запущен. Рассылка активирована.")

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "👋 Добро пожаловать!\n\n"
        "Этот бот работает в **автоматическом режиме**.\n"
        "Он анализирует рынок каждые 5 минут.\n\n"
        "Если условий для входа нет — бот **молчит**.\n"
        "Если есть сигнал (LONG/SHORT) — пришлет прогноз.\n\n"
        f"🕐 Часовой пояс: {TIMEZONE_STR}.",
        reply_markup=main_keyboard,
        parse_mode="Markdown"
    )

@dp.message(F.text == "ℹ️ Информация")
async def cmd_info(message: types.Message):
    await message.answer(
        f"📊 **Настройки стратегии:**\n"
        f"LONG: Vol > SMA & RSI < {STRATEGY_CONFIG['rsi_long_enter']}.\n"
        f"SHORT: Vol > SMA & RSI > {STRATEGY_CONFIG['rsi_short_enter']}.\n\n"
        "⚠️ *Не финансовый совет.*",
        parse_mode="Markdown"
    )

@dp.message(F.text == "🚀 Подписаться на сигналы")
async def cmd_subscribe(message: types.Message):
    user_id = message.from_user.id
    if user_id in subscribers:
        await message.answer("✅ Вы уже подписаны. Ждите сигналов!")
    else:
        subscribers.add(user_id)
        await message.answer("✅ Подписка оформлена!\nБот будет присылать сигналы, когда они появятся.")

@dp.message(F.text == "💹 Цена сейчас")
async def cmd_current_price(message: types.Message):
    status_msg = await message.answer("⏳ Получение цен...")
    data = await get_simple_prices()
    
    if not data:
        await status_msg.edit_text("⚠️ Не удалось получить данные.")
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

async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
