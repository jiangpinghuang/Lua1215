-- Torch
-- Tensor Basics
A = torch.Tensor(3, 3)
B = torch.zeros(3, 3, 2)
C = torch.ones(3, 1, 3)
D = torch.randn(2)
E = torch.rand(1, 1, 1, 1)
F = torch.Tensor({{1,1}, {2, 2}, {3,3}})

print(A, "\n")
print(B, "\n")
print(C, "\n")
print(D, "\n")
print(E, "\n")
print(F)

A = torch.FloatTensor(3, 3)
print(A, "\n")
B = torch.LongTensor(3, 3)
print(B, "\n")

A = torch.randn(2, 3)
print(A:dim(), "\n")
print(A:size(1), "\n")
print(A:size(2), "\n")
print(A:size())
print(A:nElement(), "\n")
print(A:isContiguous())

-- Views on Tensors.
a = torch.range(1, 6)
print(a, "\n")
A = a:view(2, 3)
print(A)
B = A:view(3,2)
print(B, "\n")
B:zero()

-- Accessing Sub-Tensors.
A = torch.range(1, 6):view(2, 3)
firstRow = A[1]
print("A[1]")
print(A[1], "\n")
print(A, "\n")
print("firstRow:")
print(firstRow)
print("a:")
print(a)
firstCol = A:select(2,1)
print(firstCol)