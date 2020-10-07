import numpy as np
import torch

def make_items_tensor(items_embeddings_key_dict):
    keys = list(sorted(items_embeddings_key_dict.keys()))
    key_to_id = dict(zip(keys, range(len(keys))))
    id_to_key = dict(zip(range(len(keys)), keys))

    items_embeddings_id_dict = {}
    for k in items_embeddings_key_dict.keys():
        items_embeddings_id_dict[key_to_id[k]] = items_embeddings_key_dict[k]
    items_embeddings_tensor = torch.stack(
        [items_embeddings_id_dict[i] for i in range(len(items_embeddings_id_dict))]
    )
    return items_embeddings_tensor, key_to_id, id_to_key



def batch_contstate_discaction(batch, item_embeddings_tensor, frame_size, num_items):

    """
    Embed Batch: continuous state discrete action
    """

    items_t, ratings_t, sizes_t, users_t = batch["items"], batch["ratings"], batch["sizes"], batch["users"]
    items_emb = item_embeddings_tensor[items_t.long()]
    b_size = ratings_t.size(0)

    items = items_emb[:, :-1, :].view(b_size, -1)
    next_items = items_emb[:, 1:, :].view(b_size, -1)
    ratings = ratings_t[:, :-1]
    next_ratings = ratings_t[:, 1:]

    state = torch.cat([items, ratings], 1)
    next_state = torch.cat([next_items, next_ratings], 1)
    action = items_t[:, -1]
    reward = ratings_t[:, -1]

    done = torch.zeros(b_size)
    done[torch.cumsum(sizes_t - frame_size, dim=0) - 1] = 1

    one_hot_action = torch.zeros(b_size, num_items)
    one_hot_action.scatter_(1, action.view(-1, 1), 1)

    batch = {
        "state": state,
        "action": one_hot_action,
        "reward": reward,
        "next_state": next_state,
        "done": done,
        "meta": {"users": users_t, "sizes": sizes_t},
    }
    return batch