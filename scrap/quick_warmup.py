# Alessandro Gambetti
# 40755

# Submission 1

# Factorial
def factorial(n):
    if n < 0:
        return 'The factorial of a negative number does not exist!'
    if n == 0:
        return 1
    if n == 1:
        return 1
    return n * factorial(n-1)


# Max Odd
def max_odd(x,y,z):
    nums = [x, y, z]
    nums = [num for num in nums if num %2 != 0]
    if len(nums) == 0:
        return 'No Odd Numbers in the List.'
    return max(nums)

# Find Pairs
def find_pairs(x):
    # Return A and B
    # Condition B is from 1 to 5

    pairs = []

    for i in range(1, 6):
        temp = x ** (1/i)
        if int(temp) ** i == x:
            pairs.append((int(temp), i))

    if len(pairs) == 0:
        return 'There are not pairs available.'

    return pairs

print(find_pairs(12))
