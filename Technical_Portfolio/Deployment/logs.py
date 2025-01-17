import asyncio
import traceback
from telegram import Bot
import os
from unsync import unsync
from deploy import Deploy

BOT_TOKEN = '8001920272:AAGR92ZAnH_QUg2eenhKyRSInfqKGMwlNlc'
CHAT_ID = '2103748257'

bot = Bot(token=BOT_TOKEN)

@unsync
async def send_telegram_message(message):
    try:
        await bot.send_message(chat_id=CHAT_ID, text=message)
    except Exception as e:
        print(f"Error sending Telegram message: {e}")

@unsync
async def main():
    try:
        # Your script's main code here
        Deploy().upload_complete_market_data()
        Deploy().main_loop()
    except Exception as e:
        error_message = f"An error occurred:\n\n{traceback.format_exc()}"
        print(error_message)  # Print to console for local debugging
        await send_telegram_message(error_message)
    else:
        await send_telegram_message("Script completed successfully.")

main().result()  