import numpy as np
from Utils.interpretation_funcs import compute_cat_prob


def test_gaussian_inter():
    tolerance = 1.e-1
    mu = np.array([3., 2., 1., 0.1])
    sigma = np.array([0.1, 1., 1., 3])
    # sigma = np.array([1., 1., 1., 1.])
    cov = np.diag(sigma ** 2)
    samples_n = int(1.e4)
    categories_n = mu.shape[0]
    z = np.random.multivariate_normal(mu, cov, size=samples_n)
    from_sample = True
    inner_sample_size = int(1.e2)

    w_argmax = np.argmax(z, axis=1)
    ans = np.zeros(categories_n)
    for k in range(categories_n):
        ans[k] = np.mean(w_argmax == k)
    approx = compute_cat_prob(mu, sigma, from_sample, inner_sample_size)

    diff = np.linalg.norm(approx - ans) / np.linalg.norm(ans)
    print(f'\nDiff {diff:1.2e}')
    assert diff < tolerance


def test_gaussian_inter_uniform():
    categories_n = 5
    samples_n = int(1.e5)
    tolerance = 1.e-1

    mu = np.zeros(categories_n)
    sigma = np.ones(categories_n)
    z = np.random.multivariate_normal(mu, np.diag(sigma), size=samples_n)

    w_argmax = np.argmax(z, axis=1)
    ans = np.zeros(categories_n)
    for k in range(categories_n):
        ans[k] = np.mean(w_argmax == k)
    approx = compute_cat_prob(mu, sigma, from_sample=False)

    diff = np.linalg.norm(approx - ans) / np.linalg.norm(ans)
    print(f'\nDiff {diff:1.2e}')
    assert diff < tolerance
