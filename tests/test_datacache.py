import numpy as np

from utils.data_cache import DataCache


def _fake_data():
    x = np.random.rand(100, 5)
    y = np.random.rand(100, 2)
    x = {f"in_{i}": x[:, i] for i in range(x.shape[1])}
    y = {f"out_{i}": y[:, i] for i in range(y.shape[1])}
    return x, y


def test_cache(tmpdir):
    np.random.seed(42)
    x, y = _fake_data()
    cache = DataCache.from_dict(x=x, y=y)
    cache.save(f"{tmpdir}/test.nc")
    cache2 = DataCache.load(f"{tmpdir}/test.nc")
    assert cache.data.equals(cache2.data)
    train_frac = 0.8
    train, test = cache.train_test_split(frac_train=train_frac)
    # assert test size is 20% of full
    obtained_size = len(test) / len(cache)
    np.isclose(obtained_size, 1 - train_frac)
    assert cache.in_shape == (1, 5)
