def is_number(str):
    try:
        # for int, long, float and complex
        complex(str)
    except ValueError:
        return False

    return True
