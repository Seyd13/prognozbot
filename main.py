import sys
import os
import telebot
import numpy as np
import requests 
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io
import logging
import threading
import time
from collections import deque
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime, timedelta, timezone
from flask import Flask

# --- КОНФИГУРАЦИЯ ---
TELEGRAM_TOKEN = "2122435147:AAG_52ELCHjFnXNxcAP4i5xNAal9I91xNTM"
PORT = int(os.environ.get('PORT', 8080))

bot = telebot.TeleBot(TELEGRAM_TOKEN)

# --- НАСТРОЙКА ЛОГГИРОВАНИЯ ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# --- FLASK ДЛЯ RAILWAY ---
app = Flask(__name__)

@app.route('/')
def index():
    return "Bot is alive"

def run_flask():
    app.run(host='0.0.0.0', port=PORT, use_reloader=False)

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def get_moscow_time():
    moscow_tz = timezone(timedelta(hours=3))
    return datetime.now(moscow_tz)

# --- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ---
chart_data_buffer = deque(maxlen=20) 
current_symbol = None
stop_http_flag = False
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

# --- ЛОГИКА ПОЛУЧЕНИЯ ЦЕНЫ (HTTP) ---

def binance_http_logic(symbol, chat_id):
    global chart_data_buffer, current_symbol, stop_http_flag, is_busy
    
    current_symbol = symbol.lower()
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={current_symbol.upper()}"
    chart_data_buffer.clear()
    
    try:
        logger.info(f"Запущен HTTP мониторинг для {symbol}")
        prediction_sent = False
        
        while not stop_http_flag:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    close_price = float(data['price'])
                    
                    current_time = get_moscow_time()
                    chart_data_buffer.append({'price': close_price, 'time': current_time})
                    
                    if not prediction_sent and len(chart_data_buffer) >= 12:
                        logger.info("Данных достаточно, отправляем прогноз...")
                        threading.Thread(target=send_prediction, args=(chat_id,)).start()
                        prediction_sent = True
                
                time.sleep(1)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Ошибка запроса к Binance: {e}")
                time.sleep(5)
    except Exception as e:
        logger.error(f"Критическая ошибка потока: {e}")
    finally:
        logger.info("HTTP поток завершил работу.")
        is_busy = False

def run_http_thread(symbol, chat_id):
    global stop_http_flag
    stop_http_flag = False
    
    http_thread = threading.Thread(target=binance_http_logic, args=(symbol, chat_id))
    http_thread.daemon = True
    http_thread.start()

# --- ЛОГИКА ПРОГНОЗА ---

def predict_price(data_buffer):
    if len(data_buffer) < 10:
        return None, None
    
    recent_data = list(data_buffer)[-10:] 
    prices = [d['price'] for d in recent_data]
    
    X = np.arange(len(prices)).reshape(-1, 1)
    y = prices
    
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    
    next_point = np.array([[len(prices)]])
    next_point_poly = poly.transform(next_point)
    predicted_close = model.predict(next_point_poly)[0]
    score = model.score(X_poly, y)
    
    return predicted_close, score

def create_price_chart(data_buffer, predicted_close=None):
    plt.figure(figsize=(10, 5))
    
    last_points = list(data_buffer)[-20:] 
    prices = [d['price'] for d in last_points]
    timestamps = [d['time'] for d in last_points]
    
    time_labels = [t.strftime('%H:%M:%S') for t in timestamps]
    x_values = range(len(last_points))
    
    plt.plot(x_values, prices, 'bo-', linewidth=1.5, markersize=5, label='Цены закрытия')
    plt.xticks(x_values, time_labels, rotation=45, ha='right', fontsize=8)
    
    if predicted_close is not None:
        next_x = len(last_points)
        plt.plot(next_x, predicted_close, 'ro', markersize=8, label='Прогноз')
        plt.text(next_x, predicted_close, f'{predicted_close:.2f}', ha='center', va='bottom', fontsize=10, color='red', fontweight='bold')
        plt.plot([x_values[-1], next_x], [prices[-1], predicted_close], 'r--', alpha=0.5)
        
        last_time = timestamps[-1]
        pred_time = last_time + timedelta(minutes=1)
        pred_time_label = pred_time.strftime('%H:%M:%S')
        plt.xticks(list(x_values) + [next_x], time_labels + [pred_time_label], rotation=45, ha='right', fontsize=8)

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
    global stop_http_flag, current_chat_id, is_busy
    
    if is_busy:
        bot.send_message(message.chat.id, "⏳ Подождите, я обрабатываю предыдущий запрос...", reply_markup=asset_keyboard)
        return

    symbol = message.text
    current_chat_id = message.chat.id
    
    is_busy = True
    
    stop_http_flag = True
    time.sleep(0.5) 
    
    run_http_thread(symbol, current_chat_id)
    
    bot.send_message(message.chat.id, f"✅ Запустил анализ {symbol}.\n⏳ График придет через 10-15 сек...", reply_markup=main_keyboard)

@bot.message_handler(func=lambda message: message.text == "Назад")
def go_back_to_main(message):
    global stop_http_flag, current_symbol, chart_data_buffer, current_chat_id, is_busy
    
    stop_http_flag = True
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
    # Запускаем Flask в отдельном потоке
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True 
    flask_thread.start()
    
    # Небольшая пауза перед стартом бота, чтобы избежать конфликта 409 при быстром рестарте
    time.sleep(2) 
    
    logger.info("Бот запущен...")
    logger.info(f"Flask сервер запущен на порту {PORT}")
    
    try:
        bot.infinity_polling(timeout=10, long_polling_timeout=5)
    except Exception as e:
        logger.error(f"Критический сбой: {e}")
