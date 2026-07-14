import os

def warning_format(message, category, filename, lineno, line=None):
    return f"{category.__name__}: {message} ({os.path.basename(filename)}:{lineno})\n"
