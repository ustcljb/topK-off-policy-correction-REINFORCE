import numpy as np
import torch


# Used for returning, arguments are mutable
class DataFuncArgsMut:
    def __init__(
        self, df, base, users: List[int], user_dict: Dict[int, Dict[str, np.ndarray]]
    ):
        self.base = base
        self.users = users
        self.user_dict = user_dict
        self.df = df


def prepare_dataset(args_mut: DataFuncArgsMut, frame_size: int):

    """
    Basic prepare dataset function. Automatically makes index linear, in ml20 movie indices look like:
    [1, 34, 123, 2000], recnn makes it look like [0,1,2,3] for you.
    """

    # get args
    key_to_id = args_mut.base.key_to_id
    df = args_mut.df

    # rating range mapped from [0, 5] to [-5, 5]
    df["rating"] = try_progress_apply(df["rating"], lambda i: 2 * (i - 2.5))
    # id's tend to be inconsistent and sparse so they are remapped here
    df["movieId"] = try_progress_apply(df["movieId"], key_to_id.get)
    users = df[["userId", "movieId"]].groupby(["userId"]).size()
    users = users[users > frame_size].sort_values(ascending=False).index

    if pd.get_type() == "modin":
        df = df._to_pandas()  # pandas groupby is sync and doesnt affect performance
    ratings = (
        df.sort_values(by="timestamp")
        .set_index("userId")
        .drop("timestamp", axis=1)
        .groupby("userId")
    )

    # Groupby user
    user_dict = {}

    def app(x):
        userid = x.index[0]
        user_dict[userid] = {}
        user_dict[userid]["items"] = x["movieId"].values
        user_dict[userid]["ratings"] = x["rating"].values

    try_progress_apply(ratings, app)

    args_mut.user_dict = user_dict
    args_mut.users = users

    return args_mut



def batch_tensor_embeddings(batch, item_embeddings_tensor, frame_size):
    """
    Embed Batch: continuous state continuous action
    """

    items_t, ratings_t, sizes_t, users_t = get_irsu(batch)
    items_emb = item_embeddings_tensor[items_t.long()]
    b_size = ratings_t.size(0)

    items = items_emb[:, :-1, :].view(b_size, -1)
    next_items = items_emb[:, 1:, :].view(b_size, -1)
    ratings = ratings_t[:, :-1]
    next_ratings = ratings_t[:, 1:]

    state = torch.cat([items, ratings], 1)
    next_state = torch.cat([next_items, next_ratings], 1)
    action = items_emb[:, -1, :]
    reward = ratings_t[:, -1]

    done = torch.zeros(b_size)
    done[torch.cumsum(sizes_t - frame_size, dim=0) - 1] = 1

    batch = {
        "state": state,
        "action": action,
        "reward": reward,
        "next_state": next_state,
        "done": done,
        "meta": {"users": users_t, "sizes": sizes_t},
    }
    return batch



# Main function that is used as torch.DataLoader->collate_fn
# CollateFn docs:
# https://pytorch.org/docs/stable/data.html#working-with-collate-fn


def prepare_batch_static_size(
    batch, item_embeddings_tensor, frame_size=10, embed_batch=batch_tensor_embeddings
):
    item_t, ratings_t, sizes_t, users_t = [], [], [], []
    for i in range(len(batch)):
        item_t.append(batch[i]["items"])
        ratings_t.append(batch[i]["rates"])
        sizes_t.append(batch[i]["sizes"])
        users_t.append(batch[i]["users"])

    item_t = np.concatenate([rolling_window(i, frame_size + 1) for i in item_t], 0)
    ratings_t = np.concatenate(
        [rolling_window(i, frame_size + 1) for i in ratings_t], 0
    )

    item_t = torch.tensor(item_t)
    users_t = torch.tensor(users_t)
    ratings_t = torch.tensor(ratings_t).float()
    sizes_t = torch.tensor(sizes_t)

    batch = {"items": item_t, "users": users_t, "ratings": ratings_t, "sizes": sizes_t}

    return embed_batch(
        batch=batch,
        item_embeddings_tensor=item_embeddings_tensor,
        frame_size=frame_size,
    )


    
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



def batch_contstate_discaction(
    batch, item_embeddings_tensor, frame_size, num_items, *args, **kwargs
):

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