from util.select import select
from model.supervised_learning import s_execute

from model.active_learning import a_execute

if __name__ == '__main__':
    base_url = "galaxies.csv"
    # train_url, val_url, test_url = "train.csv", "val.csv", "test.csv"
    # train_url = "train.csv"
    # process_url="solutions.csv"
    # combine("images","solutions.csv")
    select(base_url)
    # a_execute(train_url)
    # s_execute(train_url)
    # execute(train_url,val_url)
