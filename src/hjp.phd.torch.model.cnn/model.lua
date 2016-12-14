function make_cnn(input_size, kernel_width, num_kernel)
  local output
  local input = nn.Identity()()
  if opt.cudnn == 1 then
    local conv = cudnn.SpatialConvolution(1, num_kernel, input_size, kernel_width, 1, 1, 0)
    local conv_layer = conv(nn.View(1, -1, input_size):setNumInputDims(2)(input))
    output = nn.Sum(3)(nn.Max(3)(nn.Tanh()(conv_layer)))
  else
    local conv = nn.TemporalConvolution(input_size, num_kernel, kernel_width)
    local conv_layer = conv(input)
    output = nn.Max(2)(nn.Tanh()(conv_layer))
  end
  return nn.gModule({input}, {output})
end