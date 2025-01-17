import asyncio
import traceback
from telegram import Bot
from unsync import unsync
from deploy import Deploy

BOT_TOKEN = '8001920272:AAGR92ZAnH_QUg2eenhKyRSInfqKGMwlNlc'
BOT_TOKEN_LOGS = '7405668846:AAGqXKChZ54tmqwA6RoK4tbaD90yiAA3WHw'
CHAT_ID = '2103748257'

bot = Bot(token=BOT_TOKEN)
logging_bot = Bot(token=BOT_TOKEN_LOGS)

@unsync
async def send_telegram_message(message):
    try:
        await bot.send_message(chat_id=CHAT_ID, text=message)
    except Exception as e:
        print(f"Error sending Telegram message: {e}")
        
@unsync
async def send_telegram_logs(message):
    try:
        await logging_bot.send_message(chat_id=CHAT_ID, text=message)
    except Exception as e:
        print(f"Error sending Telegram message: {e}")


async def monitor_log_file(file_path):
    try:
        with open(file_path, "r") as file:
            file.seek(0, 2)  # Move to the end of the file
            while True:
                line = file.readline()
                if line:
                    await send_telegram_message(line.strip())  # Send each new log line
                else:
                    await asyncio.sleep(1)  # Wait for new log entries
    except FileNotFoundError:
        error_message = f"Log file not found: {file_path}"
        print(error_message)
        await send_telegram_message(error_message)
    except Exception as e:
        error_message = f"An error occurred while monitoring logs:\n\n{traceback.format_exc()}"
        print(error_message)
        await send_telegram_message(error_message)


@unsync
async def main():
    log_file_path = "output.log"  # Replace with the path to your log file
    
    # Start monitoring logs as a background task
    log_task = asyncio.create_task(monitor_log_file(log_file_path))
    
    try:
        # Your script's main functionality
        Deploy().upload_complete_market_data()
        Deploy().main_loop()
    except Exception as e:
        error_message = f"An error occurred:\n\n{traceback.format_exc()}"
        print(error_message)  # Print to console for local debugging
        await send_telegram_message(error_message)
    else:
        await send_telegram_message("Script completed successfully.")
    
    # Optional: Cancel the log task if no longer needed
    log_task.cancel()

main().result()
