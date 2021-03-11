from util.select import select
from model.supervised_learning import s_execute
from model.active_learning import a_execute

if __name__ == '__main__':
    base_url = "galaxies.csv"
    train_url = "train.csv"

    # data shuffle and preprocess
    # select(base_url)

    # active learning
    # a_execute(train_url)

    # supervised learning
    s_execute(train_url)
