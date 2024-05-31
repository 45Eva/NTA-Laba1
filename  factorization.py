from decimal import Decimal
import random
from random import randrange
import math
import numpy as np
from sympy import factorint
from itertools import combinations
from datetime import datetime
from decimal import Decimal

def GCD(a,b):
    if b == 0:
        return a, 1, 0
    d, x, y = GCD(b, a % b)
    return d, y, x - (a // b) * y

def reverse(num, mod):
    gcd, x, y = GCD(num, mod)
    if gcd == 1:
        return (x % mod + mod) % mod
    else:
        return -1

def horner(num, pow, mod):
    result = 1

    while pow > 0:
        if pow & 1:
            result = (result * num) % mod
        num = (num * num) % mod
        pow >>= 1

    return result

def rand_prime(N = 256):
    iter = 0
    while (True):
        iter = iter + 1
        bin_str = "1"
        for i in range(N - 1):
            bit = random.randint(0, 1)
            bin_str = bin_str + str(bit)
        num = int(bin_str, 2)
        if Miller_Rabin(num) == True:
            return num

def Miller_Rabin(n, k=15):
    n = int(n)
    # Якщо n менше за 2, воно не є простим числом
    if n <= 1:
        return False
    # Якщо n дорівнює 2, воно є простим числом
    if n == 2:
        return True
    # Якщо n парне, воно не є простим числом (крім випадку n = 2)
    if n % 2 == 0:
        return False

    # Представимо n - 1 як (2^r) * d
    r = 0
    d = n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # k - кількість ітерацій для перевірки
    for _ in range(k):
        a = random.randint(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def f(x, n):
    return (x * x + 1) % n

def pollards_rho(n):
    n = int(n)
    if n % 2 == 0:
        return 2
    x = 7
    y = x
    d = 1
    while d == 1 or d == n:
        x = f(x, n)
        y = f(f(y, n), n)
        d = GCD((y - x) %n, n)[0]
    return d

def trial_division_with_pascal(n, B=10):
    n = int(n)
    # Функція для розкладу числа у систему числення з основою B
    def decompose(n, B):
        digits = []
        while n > 0:
            digits.append(n % B)
            n //= B
        return digits


    # Розклад числа у систему числення з основою B
    digits = decompose(n, B)
    # Перевірка подільності на малі прості числа
    for d in range(2, 48):
        if Miller_Rabin(d):
            # Обчислюємо залишок від ділення n на d
            remainder = sum(digit * (B ** i) for i, digit in enumerate(digits)) % d
            if remainder == 0:
                return d
            #else:
                #print(d) n ділиться на d, отже, n не є простим
              # n не ділиться на жодне з малих простих чисел, отже, можливо, що n просте

def leg(b ,n):
    a = n % b
    m = (b - 1) // 2
    if a == 0:
        return 0
    if pow(n, m, b) == 1:
        return 1
    return 0

def generate_prime_list(limit, n):
    """Генерація списку простих чисел до певного значення."""
    primes = [-1]
    num = 2
    while num < limit:
        if Miller_Rabin(num) and leg(num, n):
            primes.append(num)
        num += 1
    return primes


def calculate_B(n, a=1/math.sqrt(2)):
    """Обчислення значення B."""
    log_n = math.log(n)
    log_log_n = math.log(log_n)
    B = math.exp((log_n * log_log_n)**0.5)
    L = B**a
    return int(L)

def form_factor_base(n):
    """Формування факторної бази."""
    B = calculate_B(n)
    factor_base = generate_prime_list(B, n)
    return factor_base

def is_smooth(value, factor_base):
    """Перевірка на гладкість за факторною базою."""
    if value == 1:
        return False
    factors = factorint(value)
    return all(p in factor_base for p in factors)

def search_vector(n, M, factor_base):
    m = int(math.sqrt(n))
    a = m + M
    b = a**2 - n
    t = is_smooth(b, factor_base)
    if t:
        c = len(factor_base)
        factors = factorint(b)
        equation = [0] * (c)
        for i, p in enumerate(factor_base):
            if p in factors:
                equation[i] = factors[p]
        vector_mod_2 = [x % 2 for x in equation]
        return vector_mod_2, a, b

    # global check
    # check = False
    # A = []
    # B = []
    # vectors = []
    # c = len(factor_base)
    # m = int(math.sqrt(n))
    # if not check:
    #     check = True
    #     for i in range(-M, M + 1 ):
    #         a = m + i
    #         b = a**2 - n
    #         t = is_smooth(b, factor_base)
    #         if t:
    #             A.append(a)
    #             B.append(b)
    #             factors = factorint(b)
    #             equation = [0] * (c)
    #             zero_vector = [0] * (c)
    #             for i, p in enumerate(factor_base):
    #                 if p in factors:
    #                     equation[i] = factors[p]
    #             vector_mod_2 = [x % 2 for x in equation]
    #             vectors.append(vector_mod_2)
    #             if equation == zero_vector:
    #                 return [equation], [a], [b]
    #     return vectors, A, B
    # else:


def cry(matrix):
    """
    Шукає комбінацію векторів у матриці, що дають нульовий вектор за модулем 2.

    :param matrix: Список списків або numpy масив (матриця)
    :return: Комбінація векторів (індекси), що дають нульовий вектор або None, якщо такої немає
    """
    # Конвертуємо матрицю у numpy масив
    matrix = np.array(matrix)

    # Перевірка всіх можливих ненульових підмножин векторів
    for r in range(1, len(matrix) + 1):
        for indices in combinations(range(len(matrix)), r):
            subset = matrix[list(indices)]
            # Сума за модулем 2
            if np.all(np.mod(np.sum(subset, axis=0), 2) == 0):
                return indices

    return None

def FG(matrix):  # matrix with m rows and n columns
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    if num_rows < num_cols:
        print("More data needed, enter more rows.")
        return

    pivot_rows = [False] * num_rows
    column_to_row_map = {}
    
    for col in range(num_cols):
        pivot_found = False
        for row in range(num_rows):
            if matrix[row][col] == 1:
                pivot_rows[row] = True
                column_to_row_map[col] = row
                pivot_found = True
                break
        
        if pivot_found:
            for current_col in range(num_cols):
                if current_col == col:
                    continue
                if matrix[row][current_col] == 1:
                    for current_row in range(num_rows):
                        matrix[current_row][current_col] = (matrix[current_row][col] + matrix[current_row][current_col]) % 2

    return matrix, pivot_rows, column_to_row_map

def find_dependent_rows(matrix, pivot_rows, column_to_row_map):
    num_rows = len(matrix)
    num_cols = len(matrix[0])
    
    for row in range(num_rows):
        if not pivot_rows[row]:
            dependent_row = matrix[row]
            dependencies = [row]
            
            for col, value in enumerate(dependent_row):
                if value == 1:
                    dependencies.append(column_to_row_map[col])
            
            dependencies = [idx + 1 for idx in dependencies]
            dependencies.sort()
            yield dependencies

def pom(n):
    n = int(n)
    X = 1
    Y = 1
    factor_base = form_factor_base(n)
    l_base = len(factor_base)
    M = 0
    a, b, c = [], [], []
    while len(a) != l_base +1:
        temp1 = search_vector(n, M, factor_base)
        if temp1:
            a.append(temp1[0])
            b.append(temp1[1])
            c.append(temp1[2])
        if M == -M:
            M+=1
            continue
        temp2 = search_vector(n, -M, factor_base)
        if temp2:
            a.append(temp2[0])
            b.append(temp2[1])
            c.append(temp2[2])
        M += 1
    o, h, v = FG(a)
    for dependecy in find_dependent_rows(o, h, v):
        for i in dependecy:
            X = X*b[i-1]%n
            Y = Y*c[i-1]
        Y = Decimal(Y).sqrt()
        Y = int(Y)%n
        cry1 = (X+Y)%n
        cry2 = (X-Y)%n
        x1 = GCD(cry1, n)[0]
        x2 = GCD(cry2, n)[0]
        if x1 != 1 and x1 != n:
            return x1
        if x2 != 1 and x2 != n:
            return x2
        continue

def find_cannonical_decomposition(n):
    cannonical_decomposition = {}
    while n > 1:
        if Miller_Rabin(n):
            print("Число n є простим")
            n = int(n)
            if n in cannonical_decomposition:
                cannonical_decomposition[n] += 1 
            else:
                cannonical_decomposition[n] = 1
            return cannonical_decomposition
        d = trial_division_with_pascal(n)
        if d:
            print("Дільник d знайдено методом пробних ділень", datetime.now())
            n = n/d
            if d in cannonical_decomposition:
                cannonical_decomposition[d] += 1 
            else:
                cannonical_decomposition[d] = 1
            continue
            
        d = pollards_rho(n)
        if d:
            print("Дільник d знайдено методом ро Полларда", datetime.now())
            n = n/d
            if d in cannonical_decomposition:
                cannonical_decomposition[d] += 1 
            else:
                cannonical_decomposition[d] = 1
        if Miller_Rabin(n):
            n = int(n)
            print("Число n є простим")
            if n in cannonical_decomposition:
                cannonical_decomposition[n] += 1 
            else:
                cannonical_decomposition[n] = 1
            return cannonical_decomposition
        d = pom(n)
        if d:
            print("Дільник d знайдено методом Померанця", datetime.now())
            n = n/d
            if d in cannonical_decomposition:
                cannonical_decomposition[d] += 1 
            else:
                cannonical_decomposition[d] = 1
        else:
            print("я не можу знайти канонiчний розклад числа :(")
    return cannonical_decomposition

def main():
    start = datetime.now()
    a = {}
    prime = int(input("Введіть число: "))
    #while prime > 1:
    print("Перевірка простого числа за допомогою тесту Міллера-Рабіна...")
    if Miller_Rabin(prime):
        print(f"Число {prime} є простим.")
        if prime in a:
            a[prime] += 1
        else:
            a[prime] = 1
    else:
        print(f"Число {prime} не є простим.")
    prime = Decimal(prime)
    is_prime_result = pom(prime)
    stop = datetime.now()
    print("Час початку роботи", start)
    print("Час закінчення:", stop)
    print(f"Час виконання алгоритму: {stop-start}")
    print(is_prime_result)
    

if __name__ == "__main__":
    main()