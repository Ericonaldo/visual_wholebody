from train_multi_bc_deter import get_trainer

if __name__ == "__main__":
    trainer = get_trainer(is_eval=True)
    trainer.eval()