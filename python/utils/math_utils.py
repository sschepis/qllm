# Math utility functions for the Resonant Knowledge Model (Python version)

def digital_root(n):
    """
    Calculates the digital root of an integer n.

    The digital root is the single-digit value obtained by an iterative process
    of summing digits, on which process is repeated until a single-digit number
    is reached. The digital root of n is equivalent to n mod 9, except that
    if n is a multiple of 9, the digital root is 9 (not 0).

    Args:
        n (int): The number for which to calculate the digital root.
                 Must be a non-negative integer.

    Returns:
        int: The digital root (1-9), or 0 if the input is 0.
    """
    if not isinstance(n, int) or n < 0:
        raise ValueError("Input must be a non-negative integer.")

    if n == 0:
        return 0
    # The formula 1 + ((n - 1) % 9) correctly handles the mapping
    # of multiples of 9 to 9, and other numbers to their n mod 9 value.
    return 1 + ((n - 1) % 9)

# Example Usage (Illustrative)
# if __name__ == '__main__':
#     print(f"Digital root of 0: {digital_root(0)}")
#     print(f"Digital root of 5: {digital_root(5)}")
#     print(f"Digital root of 9: {digital_root(9)}")
#     print(f"Digital root of 10: {digital_root(10)}")
#     print(f"Digital root of 18: {digital_root(18)}")
#     print(f"Digital root of 12345: {digital_root(12345)}") # 1+2+3+4+5 = 15 -> 1+5 = 6