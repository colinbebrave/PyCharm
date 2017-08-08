print("Please think of a number between 0 and 100!")
low = 0
high = 100
guess = (low + high) / 2
while True:
    guess = (low + high) / 2
    print("Is your secret number " +str(guess))
    print("Enter 'h' to indicate the guess is too high. Enter 'l' to indicate the guess is too low.")
    print("Enter 'c' to indicate I guessed correctly.")
    s = input()
    if s == 'h':
        high = guess
    elif s == 'l':
        low = guess
    elif s == 'c':
        break
    else:
        print("Sorry, I did not understand your input.")
print("Your secret number is " + str(guess))

