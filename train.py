import os
import logging
import random
from pprint import pformat
from argparse import ArgumentParser
import pytorch_lightning as pl
from modules import GPT2Transformer
from dataset_wb import DialogDataModule

logger = logging.getLogger(__file__)

def main(args):
    pl.seed_everything((args.seed))
    dialog_data_module = DialogDataModule(args)
    model = GPT2Transformer(dialog_data_module, dialog_data_module.tokenizer, args)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        save_top_k=10,
        verbose=True,
        monitor="val_loss",
        filename="epoch{}",
        mode="max",
        every_n_train_steps=1000,
        save_weights_only=True
    )
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    trainer.fit(model, datamodule=dialog_data_module)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--test_only', action='store_true', help="only test model")
    parser.add_argument('--train_batch_size', default=8, type=int)
    parser.add_argument('--valid_batch_size', default=1, type=int)
    parser.add_argument('--seed', type=int, default=2020, help="used to fix random value")
    parser.add_argument('--dataset_path', default="", type=str, required=True, help="the train data directory")
    parser.add_argument('--output_dir', default=None, type=str, required=True)
    parser = GPT2Transformer.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
