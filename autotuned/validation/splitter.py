def time_series_splits(data, n_splits, train_val_ratio):
    length = len(data)
    splits = []

    for i in range(n_splits):
        train_end = int(length * train_val_ratio / (train_val_ratio + 1))
        val_start = train_end
        splits.append((data[:train_end], data[val_start:]))

    return splits
