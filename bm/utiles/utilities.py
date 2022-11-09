def listToString(s):
    joined_string = ",".join(s)

    return joined_string


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

