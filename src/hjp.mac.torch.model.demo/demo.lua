require('nn')
require('cutorch')

local ok,cunn = pcall(require, 'fbcunn')
print(ok)
print(cunn)

if not ok then
  ok, cunn = pcall(require, 'cunn')
  if ok then
    print("Warning: fbcunn not found. Falling back to cunn!")
    LookupTable = nn.LookupTable
    
    -- display device information.
    deviceParams = cutorch.getDeviceProperties(1)
    print(deviceParams)    
    cudaComputeCapability = deviceParams.major + deviceParams.minor/10
    print(cudaComputeCapability)
    --LookupTable = nn.LookupTable
    
  else
    print("Could not find cunn or fbcunn. Either is required!")
    os.exit()
  end
else
  deviceParams = cutorch.getDeviceProperties(1)
  cudaComputeCapability = deviceParams.major + deviceParams.minor/10
  LookupTable = nn.LookupTable
end

require('nngraph')

local stringx = require('pl.stringx')
local file = require('pl.file')

local vocab_idx = 0
local vocab_map = {}

local params = {batch_size=5,
                seq_length=20,
                layers=2,
                decay=2,
                rnn_size=50,
                dropout=0,
                init_weight=0.1,
                lr=1,
                vocab_size=10000,
                max_epoch=2,
                max_max_epoch=5,
                max_grad_norm=3
                }
print(params)

local function transfer_data(x)
  return x:cuda()
end

local state_train, state_valid, state_test
local model = {}
local paramx, paramdx

function g_init_gpu(args)
  local gpuidx = args
  print(args)
  print(gpuidx)
  gpuidx = gpuidx[1] or 1
  print(gpuidx)
  print(string.format("Using %s-th gpu", gpuidx))
  cutorch.setDevice(gpuidx)
  g_make_deterministic(1)
end

function g_make_deterministic(seed)
  torch.manualSeed(seed)
  cutorch.manualSeed(seed)
  torch.zeros(1, 1):cuda():uniform()
end

local function replicate(x_inp, batch_size)
   local s = x_inp:size(1)
   print('s: ')
   print(s)
   print(torch.floor(s / batch_size))
   print('x: ')
   print(x)
   local x = torch.zeros(torch.floor(s / batch_size), batch_size)
   print(x)
   for i = 1, batch_size do
     print("(i - 1) * s / batch_size: ")
     print((i - 1) * s / batch_size)
     local start = torch.round((i - 1) * s / batch_size) + 1
     print('start: ')
     print(start)
     local finish = start + x:size(1) - 1
     print("finished: ")
     print(finish)
     print(x_inp:sub(start, finish))
     print(1, x:size(1), i, i)
     print(x:sub(1, x:size(1), i, i):copy(x_inp:sub(start, finish)))
     x:sub(1, x:size(1), i, i):copy(x_inp:sub(start, finish))
     print('x: ')
     print(x)
   end
   return x
end

local function load_data(fname)
   local data = file.read(fname)
   print('filename: ')
   print(fname)
   print('data: ')
   print(data)
   data = stringx.replace(data, '\n', '<eos>')
   print('replaced data: ')
   print(data)
   data = stringx.split(data)
   print('split data: ')
   print(data)
   print(string.format("Loading %s, size of data = %d", fname, #data))
   local x = torch.zeros(#data)
   print('x:')
   print(x)
   for i = 1, #data do
      if vocab_map[data[i]] == nil then
         print('data[i]: ')
         print(data[i])
         vocab_idx = vocab_idx + 1
         vocab_map[data[i]] = vocab_idx
      end
      x[i] = vocab_map[data[i]]
   end
   print('vocab_map: ')
   print(vocab_map)
   print('vocab_idx: ')
   print(vocab_idx)
   print('x_a: ')
   print(x)
   return x
end

local function traindataset(batch_size)
   local x = load_data("/home/hjp/Workshop/Model/data/tmp/ptb.train.txt")
   -- Total length will be split into batch_size classes by sequence cut.
   x = replicate(x, batch_size)
   return x
end

local function testdataset(batch_size)
   local x = load_data("/home/hjp/Workshop/Model/data/tmp/ptb.test.txt")
   print('x:size(1):')
   print(x:size(1))
   print('x: ')
   print(x)
   -- Copy the sequence for test.
   x = x:resize(x:size(1), 1):expand(x:size(1), batch_size)
   print('x: ')
   print(x)
   return x
end

local function validdataset(batch_size)
   local x = load_data("/home/hjp/Workshop/Model/data/tmp/ptb.valid.txt")
   x = replicate(x, batch_size)
   return x
end

local function reset_state(state)
  state.pos = 1
  print("state: ")
  print(state)
  if model ~= nil and model.start_s ~= nil then
    print('if!')
    for d = 1, 2 * params.layers do
      print('for!')
      model.start_s[d]:zero()
    end
  end
end

local function reset_ds()
  for d = 1, #model.ds do
    model.ds[d]:zero()
  end
end

local function lstm(x, prev_c, prev_h)
  -- Calculate all four gates in one go
  local i2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(x)
  local h2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(prev_h)
  local gates = nn.CAddTable()({i2h, h2h})
  
  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates =  nn.Reshape(4,params.rnn_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)
  
  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate, in_transform})
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  return next_c, next_h
end

local function create_network()
  local x                = nn.Identity()()
  print('x: ')
  print(x)
  local y                = nn.Identity()()
  local prev_s           = nn.Identity()()
  print(params.vocab_size)
  print(params.rnn_size)
  print('LookupT: ')
  print({[0]=LookupTable(10000, 50)(nn.Identity()())})
  print({[1]=LookupTable(10000, 50)(nn.Identity()())})
  print({[2]=LookupTable(10000, 50)(nn.Identity()())})
  print({[3]=LookupTable(10000, 50)(nn.Identity()())})
  local i                = {[0] = LookupTable(params.vocab_size,
                                                    params.rnn_size)(x)}
  print('i: ')
  print(i)
  print('i is ok!')
  local next_s           = {}
  print('next_s: ')
  print(next_s)
  print('split: ')
  local split            = {prev_s:split(2 * params.layers)}
  print(split)
  for layer_idx = 1, params.layers do
    local prev_c         = split[2 * layer_idx - 1]
    local prev_h         = split[2 * layer_idx]
    local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
    local next_c, next_h = lstm(dropped, prev_c, prev_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end
  local h2y              = nn.Linear(params.rnn_size, params.vocab_size)
  local dropped          = nn.Dropout(params.dropout)(i[params.layers])
  local pred             = nn.LogSoftMax()(h2y(dropped))
  local err              = nn.ClassNLLCriterion()({pred, y})
  local module           = nn.gModule({x, y, prev_s},
                                      {err, nn.Identity()(next_s)})
  module:getParameters():uniform(-params.init_weight, params.init_weight)
  return transfer_data(module)
end

local function setup()
  print("Creating a RNN LSTM network.")
  local core_network = create_network()
  print('core_network: ')
  print(core_network)
  paramx, paramdx = core_network:getParameters()
  print('paramx: ')
  --print(paramx)
  print('paramdx: ')
  --print(paramdx)
  print('model: ')
  print(model)
  model.s = {}
  model.ds = {}
  model.start_s = {}
  print('model again: ')
  print(model)
  for j = 0, params.seq_length do
    model.s[j] = {}
    for d = 1, 2 * params.layers do
      model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
  end
  for d = 1, 2 * params.layers do
    model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end
  model.core_network = core_network
  model.rnns = g_cloneManyTimes(core_network, params.seq_length)
  model.norm_dw = 0
  model.err = transfer_data(torch.zeros(params.seq_length))
end

function g_cloneManyTimes(net, T)
  local clones = {}
  local params, gradParams = net:parameters()
  local mem = torch.MemoryFile("w"):binary()
  mem:writeObject(net)
  for t = 1, T do
    -- We need to use a new reader for each clone.
    -- We don't want to use the pointers to already read objects.
    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()
    local cloneParams, cloneGradParams = clone:parameters()
    for i = 1, #params do
      cloneParams[i]:set(params[i])
      cloneGradParams[i]:set(gradParams[i])
    end
    clones[t] = clone
    collectgarbage()
  end
  mem:close()
  return clones
end

local function g_replace_table(to, from)
  assert(#to == #from)
  for i = 1, #to do
    to[i]:copy(from[i])
  end
end

local function g_f3(f)
  return string.format("%.3f", f)
end

local function g_d(f)
  return string.format("%d", torch.round(f))
end

local function g_disable_dropout(node)
  if type(node) == "table" and node.__typename == nil then
    for i = 1, #node do
      node[i]:apply(g_disable_dropout)
    end
    return
  end
  if string.match(node.__typename, "Dropout") then
    node.train = false
  end
end

local function g_enable_dropout(node)
  if type(node) == "table" and node.__typename == nil then
    for i = 1, #node do
      node[i]:apply(g_enable_dropout)
    end
    return
  end
  if string.match(node.__typename, "Dropout") then
    node.train = true
  end
end

local function run_valid()
  reset_state(state_valid)
  g_disable_dropout(model.rnns)
  local len = (state_valid.data:size(1) - 1) / (params.seq_length)
  local perp = 0
  for i = 1, len do
    perp = perp + fp(state_valid)
  end
  print("Validation set perplexity : " .. g_f3(torch.exp(perp / len)))
  g_enable_dropout(model.rnns)
end

local function run_test()
  reset_state(state_test)
  g_disable_dropout(model.rnns)
  local perp = 0
  local len = state_test.data:size(1)
  g_replace_table(model.s[0], model.start_s)
  for i = 1, (len - 1) do
    local x = state_test.data[i]
    local y = state_test.data[i + 1]
    perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
    perp = perp + perp_tmp[1]
    g_replace_table(model.s[0], model.s[1])
  end
  print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
  g_enable_dropout(model.rnns)
end

function fp(state)
  g_replace_table(model.s[0], model.start_s)
  if state.pos + params.seq_length > state.data:size(1) then
    reset_state(state)
  end
  for i = 1, params.seq_length do
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    model.err[i], model.s[i] = unpack(model.rnns[i]:forward({x, y, s}))
    state.pos = state.pos + 1
  end
  g_replace_table(model.start_s, model.s[params.seq_length])
  return model.err:mean()
end

function bp(state)
  paramdx:zero()
  reset_ds()
  for i = params.seq_length, 1, -1 do
    state.pos = state.pos - 1
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    local derr = transfer_data(torch.ones(1))
    local tmp = model.rnns[i]:backward({x, y, s},
                                       {derr, model.ds})[3]
    g_replace_table(model.ds, tmp)
    cutorch.synchronize()
  end
  state.pos = state.pos + params.seq_length
  model.norm_dw = paramdx:norm()
  if model.norm_dw > params.max_grad_norm then
    local shrink_factor = params.max_grad_norm / model.norm_dw
    paramdx:mul(shrink_factor)
  end
  paramx:add(paramdx:mul(-params.lr))
end


local function main()
  g_init_gpu(arg)
  state_train = {data=transfer_data(traindataset(params.batch_size))}
  state_valid =  {data=transfer_data(validdataset(params.batch_size))}
  state_test =  {data=transfer_data(testdataset(params.batch_size))}
  print(state_train)
  print(state_valid)
  print(state_test)
  print("Network parameters:")
  print(params)
  local states = {state_train, state_valid, state_test}
  print(states)
  for _, state in pairs(states) do
    print(state)
    reset_state(state)
  end
  setup()
  local step = 0
  local epoch = 0
  local total_cases = 0
  local beginning_time = torch.tic()
  local start_time = torch.tic()
  print("Starting training.")
  local words_per_step = params.seq_length * params.batch_size
  local epoch_size = torch.floor(state_train.data:size(1) / params.seq_length)
  local perps
  while epoch < params.max_max_epoch do
    local perp = fp(state_train)
    if perps == nil then
      perps = torch.zeros(epoch_size):add(perp)
    end
    perps[step % epoch_size + 1] = perp
    step = step + 1
    bp(state_train)
    total_cases = total_cases + params.seq_length * params.batch_size
    epoch = step / epoch_size
    if step % torch.round(epoch_size / 10) == 10 then
      local wps = torch.floor(total_cases / torch.toc(start_time))
      local since_beginning = g_d(torch.toc(beginning_time) / 60)
      print('epoch = ' .. g_f3(epoch) ..
            ', train perp. = ' .. g_f3(torch.exp(perps:mean())) ..
            ', wps = ' .. wps ..
            ', dw:norm() = ' .. g_f3(model.norm_dw) ..
            ', lr = ' ..  g_f3(params.lr) ..
            ', since beginning = ' .. since_beginning .. ' mins.')
    end
    if step % epoch_size == 0 then
      run_valid()
      if epoch > params.max_epoch then
          params.lr = params.lr / params.decay
      end
    end
    if step % 33 == 0 then
      cutorch.synchronize()
      collectgarbage()
    end
  end
  run_test()
  print("Training is over.")
end

main()
