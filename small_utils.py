from datetime import datetime
def format_timestamp(timestamp):
    dt = datetime.fromtimestamp(timestamp)
    formatted_timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_timestamp