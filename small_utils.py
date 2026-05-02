from datetime import datetime
def format_timestamp(timestamp):
    dt = datetime.fromtimestamp(timestamp)
    formatted_timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_timestamp

def get_cost_time(cost_timetamp)->str:
    hours = int(cost_timetamp // 3600)  # 计算小时数
    minutes = int((cost_timetamp % 3600) // 60)  # 计算分钟数
    seconds = cost_timetamp % 60  # 计算剩余的秒数
    return f"{hours:02d}:{minutes:02d}:{seconds:02.0f}"