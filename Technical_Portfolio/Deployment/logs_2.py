import asyncio
import os
from telegram import Bot

BOT_TOKEN = '7405668846:AAGqXKChZ54tmqwA6RoK4tbaD90yiAA3WHw'
BOT_TOKEN_2 = '7675688949:AAFa9zQz5zw1Q9J_082Vt7NsLNvpooGvffE'
CHAT_ID = '2103748257'

log_file_path = r"/home/yassi/Portfolio_1/Technical_Portfolio/Deployment/output.log"
bot = Bot(token=BOT_TOKEN)
bot_2 = Bot(token = BOT_TOKEN_2)

async def send_telegram_message(message):
    try:
        # Sending the message in chunks if it's too large
        max_message_length = 4096  # Telegram's max message length
        while len(message) > max_message_length:
            await bot.send_message(chat_id=CHAT_ID, text=message[:max_message_length])
            message = message[max_message_length:]
        await bot.send_message(chat_id=CHAT_ID, text=message)
    except Exception as e:
        print(f"Error sending Telegram message: {e}, sending to backup bot.")
        try:
            max_message_length = 4096  # Telegram's max message length
            while len(message) > max_message_length:
                await bot_2.send_message(chat_id=CHAT_ID, text=message[:max_message_length])
                message = message[max_message_length:]
            await bot_2.send_message(chat_id=CHAT_ID, text=message)
        except Exception as e:
            print(f"Error sending Telegram message with backup bot: {e}")

async def monitor_log_file(log_file_path):
    if not os.path.exists(log_file_path):
        print(f"Log file not found at {log_file_path}.")
        return

    # Open the log file for reading
    with open(log_file_path, "r", encoding="utf-8", errors="ignore") as log_file:
        # Read the entire file content first (for existing content)
        existing_content = log_file.read()
        if existing_content:
            print(f"Sending existing log content to Telegram:\n{existing_content}")
            await send_telegram_message(existing_content)

        # Seek to the end of the file to start reading new lines
        log_file.seek(0, os.SEEK_END)

        while True:
            line = log_file.readline()  # Read a new line
            if line:
                # Send message as soon as a new line is read
                print(f"Sending log to Telegram: {line.strip()}")
                await send_telegram_message(line.strip())
            else:
                await asyncio.sleep(1)  # Sleep a bit before checking for new lines
                
async def periodic_summary(log_file_path):
    """Send a summary of the log file every 6 hours."""
    while True:
        await asyncio.sleep(6 * 60 * 60)  # Wait 6 hours
        try:
            with open(log_file_path, "r", encoding="utf-8", errors="ignore") as log_file:
                # Optionally, read the last N lines instead of the entire file
                log_content = log_file.read()
                summary_message = f"Periodic Summary (last 6 hours):\n\n{log_content}"
                await send_telegram_message(summary_message)
        except Exception as e:
            print(f"Error sending periodic summary: {e}")

async def main():
    try:
        # Start real-time log monitoring
        asyncio.create_task(monitor_log_file(log_file_path))
        
        # Start periodic summaries
        asyncio.create_task(periodic_summary(log_file_path))
        
        # Keep the main loop alive
        while True:
            await asyncio.sleep(0)
    except Exception as e:
        error_message = f"An error occurred:\n\n{e}"
        print(error_message)
        await send_telegram_message(error_message)

# Run the asyncio event loop
asyncio.run(main())