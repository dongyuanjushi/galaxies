from util.select import select
from model.decision_tree import execute
if __name__ == '__main__':
    base_url="./"
    train_url,val_url,test_url="train.csv","val.csv","test.csv"
    # select(base_url)
    execute(train_url,val_url,test_url)