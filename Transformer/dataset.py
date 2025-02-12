import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Any

class BilingualDataset(Dataset):
     def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
          super().__init__()
          self.ds = ds
          self.tokenizer_src = tokenizer_src
          self.tokenizer_tgt = tokenizer_tgt
          self.src_lang = src_lang
          self.tgt_lang = tgt_lang
          self.seq_len = seq_len

          self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
          self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
          self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

     # get length of ds
     def __len__(self):
          return len(self.ds)

     # get method
     def __getitem__(self, index: Any) -> Any:

          #get pair then get each item
          src_tgt_pair = self.ds[index]
          src_text = src_tgt_pair['translation'][self.src_lang]
          tgt_text = src_tgt_pair['translation'][self.tgt_lang]

          # encode using tokenizer
          enc_input_tokens = self.tokenizer_src.encode(src_text).ids
          dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

          # get number of padding tokens
          enc_num_pad_tokens = self.seq_len - len(enc_input_tokens) - 2
          dec_num_pad_tokens = self.seq_len - len(dec_input_tokens) - 1

          if enc_num_pad_tokens < 0 or dec_num_pad_tokens < 0:
               raise ValueError("Sentence is too long")

          # encoder input (sos + enc_input_tokens + eos + pad)
          encoder_input = torch.cat(
               [
                    self.sos_token,
                    torch.tensor(enc_input_tokens, dtype=torch.int64),
                    self.eos_token,
                    torch.tensor([self.pad_token] * enc_num_pad_tokens, dtype=torch.int64)
               ]
          )
          
          # decoder input (sos + dec_input_tokens + pad)
          decoder_input = torch.cat(
               [
                    self.sos_token,
                    torch.tensor(dec_input_tokens, dtype=torch.int64),
                    torch.tensor([self.pad_token] * dec_num_pad_tokens, dtype=torch.int64)
               ]
          )

          # label (dec_input_tokens(for training) + eos + pad)
          label = torch.cat(
               [
                    torch.tensor(dec_input_tokens, dtype=torch.int64),
                    self.eos_token,
                    torch.tensor([self.pad_token] * dec_num_pad_tokens, dtype=torch.int64)
               ]
          )

          # check it matches seq_len
          assert encoder_input.size(0) == self.seq_len
          assert decoder_input.size(0) == self.seq_len
          assert label.size(0) == self.seq_len
          
          return {
               "encoder_input" : encoder_input, # seq_len
               "decoder_input" : decoder_input, # seq_len
               "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
               "decoder_mask" : (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, 1, seq_len)
               "label" : label, # seq_len
               "src_text" : src_text,
               "tgt_text" : tgt_text 
          }

def causal_mask(size):
     mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
     return mask == 0