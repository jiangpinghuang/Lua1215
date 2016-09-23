-- module = nn.Narrow(dimension, offset, length).
x = torch.rand(4, 5)
print(x)
require('nn')
print(nn.Narrow(1, 2, 3):forward(x))
print(nn.Narrow(1, 2, 2):forward(x))
print(nn.Narrow(2, 2, 3):forward(x))

--nn.CMulTable()
ii = {torch.ones(5)*2, torch.ones(5)*3, torch.ones(5)*4}
m = nn.CMulTable()
print(m:forward(ii))
m = nn.CAddTable()
print(m:forward(ii))

--nn.Identity()
mlp = nn.Identity()
print(mlp:forward(torch.ones(5, 2)))

--nn.View()
x = torch.rand(4, 4)
print(x)
print(nn.View(2, 8):forward(x))
print(nn.View(8, 2):forward(x))

--nn.Squeeze()
x = torch.rand(2, 1, 2, 1, 2)
print(x)
print(torch.squeeze(x))

--nn.JoinTable()
x = torch.randn(5, 1)
y = torch.randn(5, 1)
z = torch.randn(2, 1)
print(x)
print(y)
print(z)
print(nn.JoinTable(1):forward{x, y})
print(nn.JoinTable(2):forward{x, y})
print(nn.JoinTable(1):forward{x, z})

--nn.BatchNormalization()

--nn.gModel()

--nn.TemporalConvolution()?
inp = 8
outp = 1
kw = 1
dw = 1
mlp = nn.TemporalConvolution(inp, outp, kw, dw)
x = torch.rand(7, inp)
print(mlp:forward(x))

--nn.LookupTable()
module = nn.LookupTable(10, 3)
input = torch.Tensor{1, 2, 1, 10}
print(input)
print(module:forward(input))
print(module)
print(module:forward(torch.Tensor{1,2,3,4,5,6,7,8,9,10}))
input = torch.Tensor({{1,2,4,5},{4,3,2,10}})
print(input)
print(module:forward(input))

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
