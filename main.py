import sys
import os
import telebot
import numpy as np
import matplotlib
# Важно для сервера: использовать Agg backend
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io
import logging
import asyncio
import websockets
import json
import threading
import time
from collections import deque
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime, timedelta, timezone
from flask import Flask

# --- КОНФИГУРАЦИЯ ---
# Ваш токен
TELEGRAM_TOKEN = "2122435147:AAG_52ELCHjFnXNxcAP4i5xNAal9I91xNTM"

# Настройка порта для Railway
PORT = int(os.environ.get('PORT', 8080))

# Инициализация бота
bot = telebot.TeleBot(TELEGRAM_TOKEN)

# --- НАСТРОЙКА ЛОГГИРОВАНИЯ ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# --- FLASK ДЛЯ RAILWAY (Keep-alive) ---
app = Flask(__name__)

@app.route('/')
def index():
    return "Bot is alive"

def run_flask():
    # Запускаем веб-сервер в отдельном потоке
    # use_reloader=False обязательно, иначе создастся 2 экземпляра бота
    app.run(host='0.0.0.0', port=PORT, use_reloader=False)

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def get_moscow_time():
    """Возвращает текущее время в Москве (UTC+3)"""
    moscow_tz = timezone(timedelta(hours=3))
    return datetime.now(moscow_tz)

# --- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ---
# Храним словари {'price': float, 'time': datetime}
chart_data_buffer = deque(maxlen=20) 
current_symbol = None
stop_websocket_flag = False
current_chat_id = None
is_busy = False

# --- КЛАВИАТУРЫ ---
main_keyboard = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
main_keyboard.add(telebot.types.KeyboardButton("Прогноз цены"))

asset_keyboard = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
asset_keyboard.row(telebot.types.KeyboardButton("BTCUSDT"))
asset_keyboard.row(telebot.types.KeyboardButton("ETHUSDT"))
asset_keyboard.row(telebot.types.KeyboardButton("BNBUSDT"))
asset_keyboard.row(telebot.types.KeyboardButton("ADAUSDT"))
asset_keyboard.row(telebot.types.KeyboardButton("SOLUSDT"))
asset_keyboard.row(telebot.types.KeyboardButton("Назад"))

# --- ЛОГИКА WEBSOCKET ---
# Это логика вашего "рабочего" кода: берем каждое сообщение, без фильтров по минутам

async def binance_websocket_logic(symbol, chat_id):
    """Основная логика подключения к WS"""
    global chart_data_buffer, current_symbol, stop_websocket_flag, is_busy
    
    current_symbol = symbol.lower()
    uri = f"wss://stream.binance.com:9443/ws/{current_symbol}@ticker"
    chart_data_buffer.clear()
    
    try:
        async with websockets.connect(uri) as ws:
            logger.info(f"Подключено к WebSocket для {symbol}")
            
            prediction_sent = False
            
            while not stop_websocket_flag:
                try:
                    # Получаем данные
                    message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    data = json.loads(message)
                    close_price = float(data['c'])
                    
                    # Сохраняем цену и текущее московское время (как вы просили)
                    current_time = get_moscow_time()
                    chart_data_buffer.append({'price': close_price, 'time': current_time})
                    
                    # АВТОМАТИЧЕСКАЯ ОТПРАВКА (как в вашем рабочем коде)
                    if not prediction_sent and len(chart_data_buffer) >= 12:
                        logger.info("Данных достаточно, отправляем прогноз...")
                        threading.Thread(target=send_prediction, args=(chat_id,)).start()
                        prediction_sent = True
                        
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("Соединение WebSocket закрыто.")
                    break
                except Exception as e:
                    logger.error(f"Ошибка WS: {e}")
                    break
    except Exception as e:
        logger.error(f"Критическая ошибка подключения: {e}")
    finally:
        logger.info("WebSocket поток завершил работу. Снятие блокировки.")
        is_busy = False

def run_websocket_thread(symbol, chat_id):
    """Функция для запуска в отдельном потоке."""
    global stop_websocket_flag
    stop_websocket_flag = False
    
    # Создаем свой Event Loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(binance_websocket_logic(symbol, chat_id))
    except Exception as e:
        logger.error(f"Ошибка в потоке WebSocket: {e}")
        is_busy = False
    finally:
        loop.close()

# --- ЛОГИКА ПРОГНОЗА ---

def predict_price(data_buffer):
    if len(data_buffer) < 10:
        return None, None
    
    # Берем последние 10 точек
    recent_data = list(data_buffer)[-10:] 
    prices = [d['price'] for d in recent_data]
    
    X = np.arange(len(prices)).reshape(-1, 1)
    y = prices
    
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Предсказываем следующую точку
    next_point = np.array([[len(prices)]])
    next_point_poly = poly.transform(next_point)
    predicted_close = model.predict(next_point_poly)[0]
    score = model.score(X_poly, y)
    
    return predicted_close, score

def create_price_chart(data_buffer, predicted_close=None):
    plt.figure(figsize=(10, 5))
    
    # Формируем данные
    last_points = list(data_buffer)[-20:] 
    prices = [d['price'] for d in last_points]
    timestamps = [d['time'] for d in last_points]
    
    # Форматируем метки (Часы:Минуты:Секунды) - ВРЕМЯ ПО МОСКВЕ
    time_labels = [t.strftime('%H:%M:%S') for t in timestamps]
    
    x_values = range(len(last_points))
    
    # График
    plt.plot(x_values, prices, 'bo-', linewidth=1.5, markersize=5, label='Цены закрытия')
    
    # Ось X (вращаем подписи)
    plt.xticks(x_values, time_labels, rotation=45, ha='right', fontsize=8)
    
    if predicted_close is not None:
        next_x = len(last_points)
        plt.plot(next_x, predicted_close, 'ro', markersize=8, label='Прогноз')
        
        # Цена прогноза
        plt.text(next_x, predicted_close, f'{predicted_close:.2f}', ha='center', va='bottom', fontsize=10, color='red', fontweight='bold')
        
        # Линия
        plt.plot([x_values[-1], next_x], [prices[-1], predicted_close], 'r--', alpha=0.5)
        
        # Время прогноза (+1 минута)
        last_time = timestamps[-1]
        pred_time = last_time + timedelta(minutes=1)
        pred_time_label = pred_time.strftime('%H:%M:%S')
        plt.xticks(list(x_values) + [next_x], time_labels + [pred_time_label], rotation=45, ha='right', fontsize=8)

    # Название BTC/USDT
    display_symbol = f"{current_symbol.upper()[:3]}/{current_symbol.upper()[3:]}" if current_symbol else "АКТИВ"
    plt.title(f'Прогноз цены {display_symbol}')
    plt.xlabel('Время (Москва)')
    plt.ylabel('Цена (USDT)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def send_prediction(chat_id):
    try:
        if len(chart_data_buffer) < 10:
            return
        predicted_close, score = predict_price(chart_data_buffer)
        if predicted_close is None:
            return
            
        chart_buffer = create_price_chart(chart_data_buffer, predicted_close)
        
        display_symbol = f"{current_symbol.upper()[:3]}/{current_symbol.upper()[3:]}" if current_symbol else "АКТИВ"
        last_price = list(chart_data_buffer)[-1]['price']
        
        direction_icon = "📈" if predicted_close > last_price else "📉"
        response_text = (
            f"{direction_icon} **Прогноз для {display_symbol}**\n\n"
            f"📊 Точность модели (R²): {score:.2%}\n"
            f"🕒 Текущая цена: {last_price:.2f}\n"
            f"🎯 Ожидаемая цена: {predicted_close:.2f}\n\n"
            f"💡 *Не финансовая рекомендация.*"
        )
        
        bot.send_photo(chat_id, chart_buffer, caption=response_text, parse_mode='Markdown', reply_markup=main_keyboard)
    except Exception as e:
        logger.error(f"Ошибка отправки: {e}")

# --- ХЕНДЛЕРЫ ---

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.send_message(message.chat.id, "🤖 Выберите монету, и я пришлю прогноз автоматически.", reply_markup=main_keyboard)

@bot.message_handler(func=lambda message: message.text == "Прогноз цены")
def ask_for_symbol(message):
    bot.send_message(message.chat.id, "Выберите актив:", reply_markup=asset_keyboard)

@bot.message_handler(func=lambda message: message.text in ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"])
def process_symbol_selection(message):
    global stop_websocket_flag, current_chat_id, is_busy
    
    if is_busy:
        bot.send_message(message.chat.id, "⏳ Подождите, я обрабатываю предыдущий запрос...", reply_markup=asset_keyboard)
        return

    symbol = message.text
    current_chat_id = message.chat.id
    
    is_busy = True
    
    # Останавливаем старый поток
    stop_websocket_flag = True
    time.sleep(0.2) 
    
    stop_websocket_flag = False
    
    # Запускаем новый поток
    ws_thread = threading.Thread(target=run_websocket_thread, args=(symbol, current_chat_id))
    ws_thread.daemon = True
    ws_thread.start()
    
    bot.send_message(message.chat.id, f"✅ Запустил анализ {symbol}.\n⏳ График придет через 10-15 сек...", reply_markup=main_keyboard)

@bot.message_handler(func=lambda message: message.text == "Назад")
def go_back_to_main(message):
    global stop_websocket_flag, current_symbol, chart_data_buffer, current_chat_id, is_busy
    
    stop_websocket_flag = True
    current_symbol = None
    chart_data_buffer.clear()
    current_chat_id = None
    is_busy = False
    
    bot.send_message(message.chat.id, "🛑 Отслеживание остановлено.", reply_markup=main_keyboard)

@bot.message_handler(func=lambda message: True)
def handle_text(message):
    if current_symbol is None:
        bot.send_message(message.chat.id, "Выберите монету через меню.", reply_markup=main_keyboard)
    elif len(chart_data_buffer) > 0:
        last_price = list(chart_data_buffer)[-1]['price']
        display_symbol = f"{current_symbol.upper()[:3]}/{current_symbol.upper()[3:]}"
        bot.send_message(message.chat.id, f"Текущая цена {display_symbol}: {last_price}", reply_markup=main_keyboard)
    else:
        bot.send_message(message.chat.id, "Собираю данные...", reply_markup=main_keyboard)

if __name__ == '__main__':
    # Запускаем Flask в отдельном потоке для работы на Railway
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True 
    flask_thread.start()
    
    logger.info("Бот запущен...")
    logger.info(f"Flask сервер запущен на порту {PORT}")
    
    try:
        bot.infinity_polling(timeout=10, long_polling_timeout=5)
    except Exception as e:
        logger.error(f"Критический сбой: {e}")
