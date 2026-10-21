import numpy as np

# ========== Шаг 1. Исходные данные ==========
x1 = np.array([1, 2, 1, 1])
x2 = np.array([70, 130, 65, 60])

# Корреляционная матрица
C = np.array([[1.0,      0.9922],
              [0.9922,   1.0   ]])

# ========== Шаг 2. Собственные числа и векторы C ==========
eigenvalues, eigenvectors = np.linalg.eigh(C)   # eigh — для симметричных матриц
print("Собственные числа:", eigenvalues)
print("Собственные векторы:\n", eigenvectors)

# ========== Шаг 3. Берём компоненту с максимальным λ ==========
idx_max = np.argmax(eigenvalues)
lambda_max = eigenvalues[idx_max]
v = eigenvectors[:, idx_max]              # собственный вектор (единичной длины)
print(f"\nλ_max = {lambda_max:.4f}")
print(f"Собственный вектор v = {v}")

# ========== Шаг 4. Стандартизация ==========
def standardize(x):
    return (x - x.mean()) / x.std(ddof=1)   # ddof=1 — выборочное СКО (делим на n−1)

x1_st = standardize(x1)
x2_st = standardize(x2)

print("\nx1st =", np.round(x1_st, 4))
print("x2st =", np.round(x2_st, 4))

# ========== Шаг 5. Главная компонента ==========
# Матрица стандартизованных данных: строки — наблюдения, столбцы — признаки
X_st = np.column_stack([x1_st, x2_st])
print("\nX_st:\n", np.round(X_st, 4))

# PC1 = X_st · v
pc1 = X_st @ v
print("\nPC1 (до округления):", pc1)
print("PC1 (округление до 2):", np.round(pc1, 2))