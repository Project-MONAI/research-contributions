import argparse

import numpy as np

import torch
import torch.nn as nn

from text_feature.modeling_cxrbert import CXRBertModel
from transformers import AutoTokenizer

def tokenize_function(example_text, tokenizer, max_seq_length):

    return tokenizer(
        example_text,
        padding='max_length',
        truncation=True,
        max_length=max_seq_length,
        return_special_tokens_mask=True,
        return_tensors='pt'
    )
    
def main(args):

    base_model_name = 'microsoft/BiomedVLP-CXR-BERT-specialized'
    resume_model = args.text_model_path
    
    max_seq_length = 2048
    hidden_size = 768
    save_seq_len = 192

    model = CXRBertModel.from_pretrained(base_model_name)

    # extend embeddings
    old_embed = model.bert.embeddings.position_embeddings.weight.data
    tmp_dim = old_embed.shape[0]
    #print("tmp_dim:", tmp_dim)
    model.bert.embeddings.position_embeddings = nn.Embedding(max_seq_length, hidden_size)
    model.bert.embeddings.position_embeddings.weight.data[:tmp_dim, :] = old_embed
    model.bert.embeddings.register_buffer("position_ids", torch.arange(max_seq_length).expand((1, -1)))
    model.config.max_position_embeddings = max_seq_length

    ckpt = torch.load(resume_model+"/pytorch_model.bin")

    msg = model.load_state_dict(ckpt, strict=False)
    print(msg)

    model.cuda()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    example=tokenize_function(args.prompt, tokenizer, max_seq_length)
    for item in example.keys():
        example[item] = example[item].cuda()

    with torch.no_grad():
        feature = model(**example)

    feature_np = feature.hidden_states[-1][:, :save_seq_len, :].detach().cpu().numpy()

    np.save(args.save_path, feature_np)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--text_model_path', type=str)
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()

    main(args)