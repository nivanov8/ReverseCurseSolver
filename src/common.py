def load_from_txt(file_name, max=None, offset=0):
    with open(file_name, "r") as f:
        data = [line.strip() for line in f]
    data = data[offset:]
    if max is not None:
        data = data[:max]
    return data