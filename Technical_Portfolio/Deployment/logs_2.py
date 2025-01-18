import asyncio
import os
from telegram import Bot

"""
This python script is used to monitor logs and send it to a telegram bot.
"""

BOT_TOKEN = '7405668846:AAGqXKChZ54tmqwA6RoK4tbaD90yiAA3WHw'
CHAT_ID = '2103748257'

log_file_path = r"/home/yassi/Portfolio_1/Technical_Portfolio/Deployment/output.log"
bot = Bot(token=BOT_TOKEN)

async def send_telegram_message(message):
    try:
        # Sending the message in chunks if it's too large
        max_message_length = 4096  # Telegram's max message length
        while len(message) > max_message_length:
            await bot.send_message(chat_id=CHAT_ID, text=message[:max_message_length])
            message = message[max_message_length:]
        await bot.send_message(chat_id=CHAT_ID, text=message)
    except Exception as e:
        print(f"Error sending Telegram message: {e}")

async def monitor_log_file(log_file_path):
    if not os.path.exists(log_file_path):
        print(f"Log file not found at {log_file_path}.")
        return

    # Open the log file for reading
    with open(log_file_path, "r") as log_file:
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

async def main():
    try:
        # Start log monitoring as a background task
        asyncio.create_task(monitor_log_file(log_file_path))

        # Run continuously without waiting
        while True:
            await asyncio.sleep(0)  # Yield control to allow other tasks to run (no delay)

    except Exception as e:
        error_message = f"An error occurred:\n\n{e}"
        print(error_message)
        await send_telegram_message(error_message)

# Run the asyncio event loop
asyncio.run(main())