require 'bleu'
require 'gleu'

scorers = {
  bleu=get_bleu,
  gleu=get_gleu
}
