-- Lua demo.
print("Hello, Lua!")

-- Variables and Printing.
a, b = 24, "tacos"
c = 'please'
print(a, b, c, "\n")
d = b .. ', ' .. c
print(d)

-- Scalar Math.
print(2*1, a^2, a%2, "\n")
print(a/7, "\n")
print(math.floor(a/7), math.ceil(a/7), "\n")
print(math.min(1, 22, 44), math.max(1, 22, 44), "\n")

-- Control Flow.
i = 1
while i < 3 do
  print(i)
  i = i + 1 -- no i += 1 or i++ in Lua.
end

for i = 10, 1, -4 do
  print(i)
end

val = 17
if val  == 0 then
  print("zero!")
elseif val%2 == 0 then
  print("even and nonzero!")
elseif val ~= 13 then
  print("odd and not 13!")
else
  print("everything else!")
end

for i = 1, 3 do
  if i % 2 == 0 then
    break
  end
  print(i)
end

-- Truth and Falsity.
a, b = nil, false
c, d = "taco", 0
if a or b then
  print("first!")
elseif c and d then
  print("second!")
else
  print("third!")
end

val2 = a and 1 or 2
print(val2, "\n")
print(c)
val3 = c and 3 or 4
print(val3, "\n")
print(c and 3)
print(3 or 4)

-- Functions.
var = 22
function f1()
  local var = 33
  return var + 1
end
print(f1(), "\n")
function f2()
  return var + 1
end
print(f2(), "\n")

function encodeDigits(a, b, c)
  local a = a or 0
  local b = b or 0
  local c = c or 0
  assert(a >= 0 and a < 10)
  assert(b >= 0 and b < 10)
  assert(c >= 0 and c < 10)
  return a*1+b*10+c*100
end
print(encodeDigits(1, 2, 3),"\n")
print(encodeDigits(2),"\n")
print(encodeDigits(nil, 2), "\n")
print(encodeDigits(), "\n")
print(encodeDigits(1, 2, 3, 4), "\n")

function divWithRemainder(a, b)
  return math.floor(a/b), a%b
end
d, r = divWithRemainder(10, 3)
print(d, r, "\n")
d = divWithRemainder(10, 3)
print(d)

-- Tables as Dictionaries.
t1 = {}
t1["one"] =1
t1["two"]=2
t1[3]="three"
print(t1, "\n")
t2 = {["one"] =1, ["two"]=2, [3]="three"}
print(t2, "\n")
print(t2["one"], t2[3], "\n")
print(t2.one)
for k, v in pairs(t1) do
  print(k, v)
end
t1["one"]=nil
print(t1, "\n")

-- Tables as (ordered) arrays.
arr = {}
arr[1] = "one"
arr[2] = "two"
arr[3] = "three"
print(arr, "\n")
arr2 = {"one", "two", "three"}
print(arr2)
print(#arr, "\n")
ugh = {["one"] = 1, ["two"] = 2}
print(ugh)
-- print(#ugh, "\n")

arr3 = {}
table.insert(arr3, "one")
table.insert(arr3, "two")
print(arr3)

print(arr2)
for i, el in ipairs(arr2) do
  print(i, el)
end
