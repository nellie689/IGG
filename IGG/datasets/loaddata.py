from torch.utils.data import DataLoader
from datasets.dataset import Dataset_Only_Test_DiFuS_ICLR, Dataset_Only_Train_DiFuS_ICLR


def loadTest(Config,fixed_src=-1):
    For_Testing_ICLR = True
    if For_Testing_ICLR == True:
        db_test = Dataset_Only_Test_DiFuS_ICLR(Config, fixed_src = fixed_src)
        return DataLoader(db_test, batch_size=16, shuffle=False), db_test.text_data



def loadTrain(Config):
    train_batch_size = Config["LStrategy"]["train_batch_size"] 
    For_ICLR_Pixel = True
    if For_ICLR_Pixel == True:
        db_train = Dataset_Only_Train_DiFuS_ICLR(Config, fixed_src = -1)
        return DataLoader(db_train, batch_size=train_batch_size, shuffle=False), db_train.text_data
