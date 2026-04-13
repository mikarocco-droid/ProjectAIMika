MIN_CONF_BASE = 0.80

def get_dynamic_threshold(event):
    conf = event.get("confidence", 0.5)
    return max(0.75, 0.95 - conf)