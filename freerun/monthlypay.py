balance = 3329
annualInterestRate = 0.2
pay = 10
r = annualInterestRate / 12
hi = balance * (1 + annualInterestRate) / 12
while True:
    bal = balance
    for i in range(12):
        bal = (bal - pay) * (1 + r)
    if bal > 0:
        pay += 10
    else:
        break
print(pay)
