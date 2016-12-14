require('nn')
require('nngraph')
require('hdf5')

cmd = torch.CmdLine()

-- data files
cmd:text("")
cmd:text("--data options--")
cmd:text("")
cmd:option('-train_file', 'data/demo-train.hdf5', [[path to the train file.]])
cmd:option('-valid_file', 'data/demo-valid.hdf5', [[path to the valid file.]])
cmd:option('-model_file', 'data/demo_model', [[path to the model file.]])

-- model parameters
cmd:text("")
cmd:text("--model parameters--")
cmd:text("")
cmd:option('-use_char', 1, [[use char as the model input.]])
cmd:option('-use_word', 1, [[use word as the model input.]])
cmd:option('-char_vec', 50, [[character embeddings size.]])
cmd:option('-word_vec', 300, [[word embeddings size.]])
cmd:option('-kernel_width', 6, [[convolutional filter size.]])
cmd:option('-kernel_num', 100, [[convolutional filter number.]])
cmd:option('-epoch', 10, [[training epochs.]])
cmd:option('-param_init', 0.1, [[parameters are initialized in (-param_init, param_init).]])
cmd:option('-seed', 2345, [[torch manual random number generator seed.]])
cmd:option('-gpuid', -1, [[which gpu to use, -1 = use cpu.]])
cmd:option('-cudnn', 0, [[whether to use cudnn for convolution.]])

function main()
  opt = cmd:parse(arg)
  print(opt)
  print(opt.epoch)
end

main()
