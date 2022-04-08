import os
import json
import numpy as np
from os import path
from tqdm import tqdm

#Image Processing Library
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import torch

import clip
import gzip
import regex as re
import ftfy
import html
import random
# from datasets import load_dataset

#Define Image Path
CC_Train_Image_Path = "/data/home/lyuchen/viltdata/large_experiments/cmd/cc/training"
CC_Val_Image_Path = "/data/home/lyuchen/viltdata/large_experiments/cmd/cc/validation"
CLIP_SCORE_IMAGE_PATH = "/home/zmykevin/fb_intern/data/mingyang_data/CC/clip_score_img"

def default_bpe():
    return os.path.join("/home/zmykevin/fb_intern/models", "bpe_simple_vocab_16e6.txt.gz")

def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

if __name__ == "__main__":
    #Load the Split Set
    with open("/home/zmykevin/fb_intern/data/mingyang_data/CC/cc_clip_test.json", "r") as f:
        clip_test = json.load(f)
        
    with open("/home/zmykevin/fb_intern/data/mingyang_data/CC/itm_picked_cap.json", "r") as f:
        itm_picked_caps = json.load(f)
    #Load the Retrieved Set
    # with open("/data/home/zmykevin/vinvl_data/CC/captions_retrieved_bertembedding_bookcorpus_sorted.json", "r") as f:
    #     cc_captions_retrieved = json.load(f)
        
    # #load the caption object data
    # with open("/data/home/zmykevin/vinvl_data/CC/bookcorpus_sentences.json", "r") as f:
    #     cc_objects_captions = json.load(f)
   
    print("Finish Loading the Data")

    #Load the model
    model = torch.jit.load("/home/zmykevin/.cache/clip/clip_model.pt")
    input_resolution = model.input_resolution.item()

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)	

    #Define Image Processing
    preprocess = Compose([Resize(input_resolution, interpolation=Image.BICUBIC), CenterCrop(input_resolution), ToTensor()])

    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()

    #Load the Image and Texts
    clip_score_results = {}
    clip_score = 0
    valid_count = 0

    for k,v in tqdm(itm_picked_caps.items()):
        try:
            cc_id = clip_test[k]["cc_id"]
            images = []
            texts = []
            
            current_image_path = os.path.join(CLIP_SCORE_IMAGE_PATH, "{}.jpg".format(cc_id))

            # print(current_image_path)
            assert path.exists(current_image_path)
            current_image_caption = v
            # print(current_image_caption)
            current_image = preprocess(Image.open(current_image_path).convert("RGB"))
            images.append(current_image)
            texts.append(current_image_caption)
            
    #         #break
    #         #print("Finish loading the images and texts")

            #Build the Features
            image_input = torch.tensor(np.stack(images)).cuda()
            image_input -= image_mean[:, None, None]
            image_input /= image_std[:, None, None]

    #         #print("Finish building the image inputs")
    # #
            tokenizer = SimpleTokenizer()
            text_tokens = [tokenizer.encode(desc) for desc in texts]
            text_input = torch.zeros(len(text_tokens), model.context_length, dtype=torch.long)
            sot_token = tokenizer.encoder['<|startoftext|>']
            eot_token = tokenizer.encoder['<|endoftext|>']

            
            for i, tokens in enumerate(text_tokens):
                tokens = [sot_token] + tokens + [eot_token]
                text_input[i, :len(tokens)] = torch.tensor(tokens)

            text_input = text_input.cuda() 
            #print("Done with text_input")
            with torch.no_grad():
                image_features = model.encode_image(image_input).float()
                text_features = model.encode_text(text_input).float()

                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
                clip_score += similarity[0][0]
                clip_score_results[k] = float(similarity[0][0])
            valid_count += 1
            
        except:
            continue
            #print(clip_score)
    print(valid_count)
    final_clip_score = clip_score/valid_count
    print(final_clip_score)
    #Save the clip_score_results
    with open("/home/zmykevin/fb_intern/data/mingyang_data/CC/cc_clip_score_itm_picked.json", "w") as f:
        json.dump(clip_score_results, f)

    # for k,v in tqdm(clip_test.items()):
    #     try:
    #         images = []
    #         texts = []
            
    #         current_image_path = os.path.join(CC_Train_Image_Path, str(v['cc_id'])) if path.exists(os.path.join(CC_Train_Image_Path, str(v['cc_id']))) else os.path.join(CC_Train_Image_Path, str(v['cc_id']))
    #         #print(current_image_path)
    #         assert path.exists(current_image_path)
    #         #current_image_caption = v['caption']
    #         retrieved_image_caption = cc_objects_captions[cc_captions_retrieved[k][0]]['caption']
    #         #retrieved_image_caption = random.choice(list(cc_objects_captions.values()))['caption']
    #         #retrieved_image_caption = random_sampled_captions[valid_count]['caption']
    #         #print(v)
    #         #retrieved_image_caption = v['objects_no_rep']
    #         current_image = preprocess(Image.open(current_image_path).convert("RGB"))
    #         images.append(current_image)
    #         texts.append(retrieved_image_caption)
            
    #         #break
    #         #print("Finish loading the images and texts")

    #         #Build the Features
    #         image_input = torch.tensor(np.stack(images)).cuda()
    #         image_input -= image_mean[:, None, None]
    #         image_input /= image_std[:, None, None]

    #         #print("Finish building the image inputs")
    # #
    #         tokenizer = SimpleTokenizer()
    #         text_tokens = [tokenizer.encode(desc) for desc in texts]
    #         text_input = torch.zeros(len(text_tokens), model.context_length, dtype=torch.long)
    #         sot_token = tokenizer.encoder['<|startoftext|>']
    #         eot_token = tokenizer.encoder['<|endoftext|>']

            
    #         for i, tokens in enumerate(text_tokens):
    #             tokens = [sot_token] + tokens + [eot_token]
    #             text_input[i, :len(tokens)] = torch.tensor(tokens)

    #         text_input = text_input.cuda() 
    #         #print("Done with text_input")
    #         with torch.no_grad():
    #             image_features = model.encode_image(image_input).float()
    #             text_features = model.encode_text(text_input).float()

    #             image_features /= image_features.norm(dim=-1, keepdim=True)
    #             text_features /= text_features.norm(dim=-1, keepdim=True)
    #             similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
    #             clip_score += similarity[0][0]
    #             clip_score_results[k] = float(similarity[0][0])
    #         valid_count += 1
            
    #     except:
    #         continue
    #         #print(clip_score)
    # print(valid_count)
    # final_clip_score = clip_score/valid_count
    # print(final_clip_score)
    # #Save the clip_score_results
    # with open("/data/home/zmykevin/vinvl_data/CC/cc_clip_score_bookcorpus.json", "w") as f:
    #     json.dump(clip_score_results, f)