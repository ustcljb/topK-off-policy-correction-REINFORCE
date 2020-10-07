import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os

from data_utils import prepare_dataset, batch_tensor_embeddings, make_items_tensor, DataFuncArgsMut,
                       prepare_batch_static_size


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


class EnvBase:

    """
    Misc class used for serializing
    """

    def __init__(self):
        self.train_user_dataset = None
        self.test_user_dataset = None
        self.embeddings = None
        self.key_to_id = None
        self.id_to_key = None


class DataPath:

    def __init__(self, base, ratings, embeddings):
        self.ratings = base + ratings
        self.embeddings = base + embeddings

class Env:

    """
    Env abstract class
    """

    def __init__(self, path: DataPath, embed_batch=batch_tensor_embeddings, 
                 prepare_dataset=prepare_dataset, test_size=0.05):

        self.base = EnvBase()
        self.embed_batch = embed_batch
        self.prepare_dataset = prepare_dataset
        self.process_env(path)

    def process_env(self, path: DataPath):

        movie_embeddings_key_dict = pickle.load(open(path.embeddings, "rb"))
        (
            self.base.embeddings,
            self.base.key_to_id,
            self.base.id_to_key
        ) = make_items_tensor(movie_embeddings_key_dict)
        ratings = pd.read_csv(path.ratings)

        process_args_mut = DataFuncArgsMut(
            df=ratings,
            base=self.base,
            users=None,  # will be set later
            user_dict=None  # will be set later
        )

        self.prepare_dataset(process_args_mut)
        self.base = process_args_mut.base
        self.df = process_args_mut.df
        users = process_args_mut.users
        user_dict = process_args_mut.user_dict

        train_users, test_users = train_test_split(users, test_size=test_size)
        train_users = utils.sort_users_itemwise(user_dict, train_users)[2:]
        test_users = utils.sort_users_itemwise(user_dict, test_users)
        self.base.train_user_dataset = UserDataset(train_users, user_dict)
        self.base.test_user_dataset = UserDataset(test_users, user_dict)

    def load_env(self, where: str):
        self.base = pickle.load(open(where, "rb"))

    def save_env(self, where: str):
        pickle.dump(self.base, open(where, "wb"))


class FrameEnv(Env):
    """
    Static length user environment.
    """

    def __init__(self, path, frame_size=10, batch_size=25, num_workers=1):


        super(FrameEnv, self).__init__(path, min_seq_size=frame_size + 1)

        self.frame_size = frame_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataloader = DataLoader(
            self.base.train_user_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self.prepare_batch_wrapper
        )

        self.test_dataloader = DataLoader(
            self.base.test_user_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self.prepare_batch_wrapper
        )

    def prepare_batch_wrapper(self, x):
        batch = prepare_batch_static_size(
            x,
            self.base.embeddings,
            embed_batch=self.embed_batch,
            frame_size=self.frame_size
        )
        return batch

    def train_batch(self):
        """ Get batch for training """
        return next(iter(self.train_dataloader))

    def test_batch(self):
        """ Get batch for testing """
        return next(iter(self.test_dataloader))