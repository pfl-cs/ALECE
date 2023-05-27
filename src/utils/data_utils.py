import time

def time_to_int(s):
    timestamp = time.mktime(time.strptime(s, "%Y-%m-%d %H:%M:%S"))
    timestamp = int(timestamp + 0.5)
    return timestamp
