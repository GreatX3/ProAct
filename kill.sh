ps aux | grep python | grep -v grep | awk '{print $2}' | xargs kill -9
ps aux | grep sglang | grep -v grep | awk '{print $2}' | xargs kill -9
ray stop