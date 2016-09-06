-- Lua demo.
print("Hello, Lua!")

-- Variables and Printing.
a, b = 24, "tacos"
c = 'please'
print(a, b, c, "\n")
d = b .. ', ' .. c
print(d)

-- Scalar Math
print(2*1, a^2, a%2, "\n")
print(a/7, "\n")
print(math.floor(a/7), math.ceil(a/7), "\n")
print(math.min(1, 22, 44), math.max(1, 22, 44), "\n")
print()
