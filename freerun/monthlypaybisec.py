balance = 320000
annualInterestRate = 0.2
r = annualInterestRate / 12.0
lo = balance / 12
hi = balance * (1 + r) ** 12 / 12.0
pay = (lo + hi) / 2.0
epsilon = 0.01
while True:
    bal = balance
    for i in range(12):
        bal = (bal - pay) * (1 + r)
    if bal > 0:
        lo = pay
    elif bal <0 and bal + epsilon < 0:
        hi = pay
    else:
        break
    pay = (lo + hi) / 2.0
print(round(pay,2))