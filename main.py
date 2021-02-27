from util.select import select, combine
from model.supervised_learning import execute

# from model.active_learning import execute

if __name__ == '__main__':
    base_url = "./"
    # train_url, val_url, test_url = "train.csv", "val.csv", "test.csv"
    train_url = "images.csv"
    # process_url="solutions.csv"
    combine("images","solutions.csv")
    # select(base_url)
    # execute(train_url, val_url, test_url)
    # execute(train_url)
    # execute(train_url,val_url)
