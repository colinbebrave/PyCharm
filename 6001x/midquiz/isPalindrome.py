def isPalindrome(aString):
    '''
    aString: a string
    Write a Python function that returns True if aString is a palindrome (reads the same forwards or reversed)
    and False otherwise. Do not use Python's built-in reverse function or aString[::-1] to reverse strings.
    '''
    # Your code here
    if len(aString) == 0:
        return True
    elif len(aString) > 0 and aString[0] == aString[-1]:
        return isPalindrome(aString[1:-1])
    else:
        return False