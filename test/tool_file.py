import datetime

now = datetime.datetime.now()
delta = datetime.timedelta(days=-1763)
n_days = now + delta
print(n_days.strftime('%Y-%m-%d %H:%M:%S'))
