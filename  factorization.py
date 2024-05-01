import random

# Імовiрнiсний тест Соловея-Штрассена
# Перевіряє чи є число простим за допомогою Імовірнісного тесту Соловея-Штрассена.
# :param p: Число, яке треба перевірити на простоту.
# :param k: Кількість ітерацій для тесту.
# :return: 1, якщо p вважається простим; 0 , якщо p вважається складеним.


def solovay_strassen(p, k):

    if p == 2:
        return 1
    if p % 2 == 0 or p == 1:
        return 0

    for _ in range(k):
        x = random.randint(2, p - 1)
        gcd_value = gcd(x, p)
        if gcd_value != 1:
            return 0
        if not euler_pseudo_prime(p, x):
            return 0

    return 1

# Знаходить найбільший спільний дільник двох чисел.


def gcd(a, b):

    while b != 0:
        a, b = b, a % b
    return a

# Перевіряє чи є число p псевдопростим Ойлера за основою x.
# :param p: Число, яке треба перевірити.
# :param x: Основа перевірки.
# :return: True, якщо p є псевдопростим за основою x, False в іншому випадку.


def euler_pseudo_prime(p, x):

    if pow(x, (p - 1) // 2, p) == legendre_symbol(x, p) % p:
        return True
    return False

# Обчислює символ Лежандра (a/p).


def legendre_symbol(a, p):
    ls = pow(a, (p - 1) // 2, p)
    return -1 if ls == p - 1 else ls


if __name__ == "__main__":
    number_to_check = int(input("Введіть число для перевірки на простоту: "))
    iterations = int(input("Введіть кількість ітерацій тесту: "))
    result = solovay_strassen(number_to_check, iterations)
    if result == 1:
        print(f"Число {number_to_check} є простим.")
    else:
        print(f"Число {number_to_check} є складеним.")
