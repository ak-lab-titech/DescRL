import gzip

import json
import numpy as np


def main(num_vec, embed_dim, file_path):

    embeds = np.zeros((embed_dim, num_vec))
    for i in range(num_vec):
        embed = np.random.normal(0.0, 1.0, embed_dim)
        embeds[:, i] = embed

    json_str = json.dumps(embeds.tolist())
    with gzip.open(file_path, 'wt', encoding='utf-8') as f:
        f.write(json_str)
    

if __name__=="__main__":
    main(
        num_vec=21,
        embed_dim=3582,
        file_path="/home/4/ud02274/navigation/myss/sound-spaces/data/category_embed/savi_21_categorys_3582.json.gz",
    )
