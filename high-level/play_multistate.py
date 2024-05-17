from skrl.utils import set_seed
from train_multistate import get_trainer

set_seed(43)

if __name__ == "__main__":
    trainer = get_trainer(is_eval=True)
    trainer.eval()