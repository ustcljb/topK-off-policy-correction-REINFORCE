import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os

from data_utils import batch_tensor_embeddings, make_items_tensor, batch_contstate_discaction

class UserDataset(Dataset):

    """
    Low Level API: dataset class user: [items, ratings], Instance of torch.DataSet
    """

    def __init__(self, users, user_dict):
        """

        :param users: integer list of user_id. Useful for train/test splitting
        :param user_dict: dictionary of users with user_id as key and [items, ratings] as value

        """

        self.users = users
        self.user_dict = user_dict

    def __len__(self):
        """
        useful for tqdm, consists of a single line:
        return len(self.users)
        """
        return len(self.users)

    def __getitem__(self, idx):
        """
        getitem is a function where non linear user_id maps to a linear index. For instance in the ml20m dataset,
        there are big gaps between neighbouring user_id. getitem removes these gaps, optimizing the speed.

        :param idx: index drawn from range(0, len(self.users)). User id can be not linear, idx is.
        :type idx: int

        :returns:  dict{'items': list<int>, rates:list<int>, sizes: int}
        """
        idx = self.users[idx]
        group = self.user_dict[idx]
        items = group["items"][:]
        rates = group["ratings"][:]
        size = items.shape[0]
        return {"items": items, "rates": rates, "sizes": size, "users": idx}

def sort_users_itemwise(user_dict, users):
    return (
        pd.Series(dict([(i, user_dict[i]["items"].shape[0]) for i in users]))
        .sort_values(ascending=False)
        .index
    )


class FrameEnv:
    """
    Static length user environment.
    """

    def __init__(self, embedding_path, rating_path, num_items, frame_size=10, 
                               batch_size=25, num_workers=1, test_size=0.05):

        super(FrameEnv, self).__init__()
        
        self.embedding_path = embedding_path
        self.rating_path = rating_path
        self.frame_size = frame_size
        self.num_items =num_items

        self.train_user_dataset = None
        self.test_user_dataset = None
        self.embeddings = None
        self.key_to_id = None
        self.id_to_key = None

        self.process_env()

        self.train_dataloader = DataLoader(
            self.train_user_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self.prepare_batch_wrapper
        )

        self.test_dataloader = DataLoader(
            self.test_user_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self.prepare_batch_wrapper
        )
    
    def process_env(self):
        movie_embeddings_key_dict = pickle.load(open(self.embedding_path, "rb"))
        self.embeddings, self.key_to_id, self.id_to_key = make_items_tensor(movie_embeddings_key_dict)
        ratings = pd.read_csv(self.rating_path)

        ratings["movieId"] = ratings["movieId"].map(self.key_to_id)
        users = ratings[["userId", "movieId"]].groupby(["userId"]).size()
        users = users[users > frame_size].sort_values(ascending=False).index
        ratings = ratings.sort_values(by="timestamp")
                         .set_index("userId")
                         .drop("timestamp", axis=1)
                         .groupby("userId")
        user_dict = {}

        def helper(x):
            userid = x.index[0]
            user_dict[userid] = {}
            user_dict[userid]["items"] = x["movieId"].values
            user_dict[userid]["ratings"] = x["rating"].values

        ratings.apply(helper)

        train_users, test_users = train_test_split(users, test_size=test_size)
        train_users = sort_users_itemwise(user_dict, train_users)[2:]
        test_users = sort_users_itemwise(user_dict, test_users)
        self.train_user_dataset = UserDataset(train_users, user_dict)
        self.test_user_dataset = UserDataset(test_users, user_dict)

    def prepare_batch_wrapper(self, x):
        batch = batch_contstate_discaction(
            x,
            self.embeddings,
            frame_size=self.frame_size
            num_items=self.num_items
        )
        return batch

    def train_batch(self):
        """ Get batch for training """
        return next(iter(self.train_dataloader))

    def test_batch(self):
        """ Get batch for testing """
        return next(iter(self.test_dataloader))