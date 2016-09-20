local ok, cunn = pcall(require, 'fbcunn')

if not ok then
  ok, cunn = pcall(require, 'cunn')
  if ok then 
    print("warning: fbcunn not found. Falling back to cunn.")
    LookupTable = nn.LookupTable
  else
    print("Could not find cunn or fbcunn. Either is required.")
    os.exit()
  end
else
  deviceParams = cutorch.getDeviceProperties(1)
  cudaComputeCapability = deviceParams.major + deviceParams.minor/10
  LookupTable = nn.LookupTable
end
require('nngraph')

function g_init_gpu(args)
  local gpuidx = args
  print(args)
  print(gpuidx)
  print(gpuidx[1])
  print(0 or 1)
  gpuidx = gpuidx[1] or 1
  print(gpuidx)
  print(string.format("Using %s-th gpu", gpuidx))
  cutorch.setDevice(gpuidx)
  print(cutorch.setDevice(gpuidx))
  g_make_deterministic(1)
end

function g_make_deterministic(seed)
  torch.manualSeed(seed)
  cutorch.manualSeed(seed)
  torch.zeros(1, 1):cuda():uniform()
end

local function main()
  g_init_gpu(arg)
end

main()