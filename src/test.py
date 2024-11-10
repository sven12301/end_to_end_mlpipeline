def calc(operation, a, b):
    if operation == 'add':
        return a + b
    elif operation == 'subtract':
        return a - b
    elif operation == 'multiply':
        return a * b
    elif operation == 'divide':
        if b != 0:
            return a / b
        else:
            return "Error: Division by zero"
    else:
        return "Error: Unsupported operation"

# Example usage:
result = calc('add', 10, 5)
print(result)  # Output: 15

"alskdjflöaskdjf ölaskdjf ölaskdjf ölaskdfj"