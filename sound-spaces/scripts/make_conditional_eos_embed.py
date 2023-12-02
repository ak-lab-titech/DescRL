import gzip

import json
import numpy as np


with gzip.open(
        "/home/0/19B30511/av-nav/VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/embeddings.json.gz",
        'rt',
        encoding='utf-8',
    ) as f:
        glove_vec = np.array(json.load(f))

print(f"glove_vec_shape: {np.shape(glove_vec)}")
print(f"BOS: {glove_vec[-2]}")
print(f"EOS: {glove_vec[-1]}")

vec_dim = 50 - 2 # locationの分だけ引く
n_category = 21
new_embed_vec = np.random.normal(0, 1, (vec_dim, n_category)) # (output_size, input_size)
print(f"new_embed_vec:\n{new_embed_vec}")
print(f"shape of new_embed_vec: {np.shape(new_embed_vec)}")

new_embed_path = "/home/0/19B30511/av-nav/myss/sound-spaces/data/category_embed/savi_21_categorys.json.gz"
json_new_embed = json.dumps(new_embed_vec.tolist())
with gzip.open(new_embed_path, "wb") as f:
    f.write(json_new_embed.encode('utf-8'))
