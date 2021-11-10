import numpy as np
import logging
import argparse
from typing import Union, Dict
from argparse import Namespace
import pytorch_lightning as pl
from transformers import AdamW, OpenAIGPTLMHeadModel
from transformers.optimization import WarmupLinearSchedule, WarmupCosineSchedule, WarmupCosineWithHardRestartsSchedule
from decoder import ChatBot
from evaluate import eval_output

logger = logging.getLogger(__name__)

class GPT2Transformer(pl.LightningModule):

    def __init__(self, dataset_module, tokenizer, hparams: Union[Dict, argparse.Namespace]):
        self.dataset_module = dataset_module
        self.tokenizer = tokenizer
        if type(hparams) == dict:
            hparams = Namespace(**hparams)
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = OpenAIGPTLMHeadModel.from_pretrained(
            self.hparams.model_name_or_path)
        self.chat_bot = ChatBot(tokenizer=self.tokenizer, start_id=None, end_id=self.tokenizer.sep_token_id, maxlen=50)

    def forward(self, **inputs):
        outputs = self.model(**inputs)
        return outputs[0]

    def training_step(self, batch, batch_ids):
        inputs = {"input_ids":batch[0], "token_type_ids": batch[1], "attention_mask": batch[2], "labels": batch[3]}
        loss = self(**inputs)
        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        tensorboard_logs = {"loss":loss, "rate": lr_scheduler.get_last_lr()[-1]}
        return {"loss": loss, "log":tensorboard_logs}

    def validation_step(self, batch, batch_ids):
        outputs = batch[-1]
        decode_rst = self.chat_bot.response(self, batch[:-1])
        ave, f1, bleuave = eval_output(decode_rst, outputs)
        return {"f1":f1, "bleuave": bleuave, "ave": ave}

    def _eval_end(self, outputs):
        f1_mean = np.mean([x["f1"] for x in outputs])
        bleuave_mean = np.mean([x["bleuave"] for x in outputs])
        ave_mean = np.mean([x["ave"] for x in outputs])
        results = {"f1": f1_mean, "bleuave": bleuave_mean, "ave": ave_mean}
        return results

    def validation_epoch_end(self, outputs: list):
        logs = self._eval_end(outputs)
        return {"f1": logs["f1"], "log": logs, "pregress_bar":logs}

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon
        )
        self.opt = optimizer

        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]

    def get_lr_scheduler(self):
        scheduler = WarmupLinearSchedule(
            self.opt, warmup_steps=self.hparams.warmup_steps, t_total=self.total_steps()
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def total_steps(self) -> int:
        num_devices = max(1, self.hparams.gpus)  # TODO: consider num_tpu_cores
        effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices
        return (self.dataset_module.train_data_size / effective_batch_size) * self.hparams.max_epochs

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            required=True,
            help="Path to pretrained model"
        )
        parser.add_argument(
            "--config_name", default="", type=str, help="pretrained config name or path if not the same as model_name"
        )
        parser.add_argument(
            "--tokenizer_name",
            default=None,
            type=str,
            help="pretrained tokenizer name or path if not the same as model_name"
        )
        parser.add_argument(
            "--cache_dir",
            default="",
            type=str,
            help="where do you want to store the pre-trained models downloaded from huggingface.co"
        )
        parser.add_argument(
            "--encoder_layerdrop",
            type=float,
            help="Encoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--decoder_layerdrop",
            type=float,
            help="Decoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            help="Dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--attention_dropout",
            type=float,
            help="Attention dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
        parser.add_argument(
            "--lr_scheduler",
            default="linear",
            type=str,
            help="Learning rate scheduler",
        )
        parser.add_argument(
            "--gradient_accumulation_steps",
            dest="accumulate_grad_batches",
            type=int,
            default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass"
        )
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument("--num_workers", default=4, type=int, help="kwarg passed to DataLoader")
        parser.add_argument("--num_train_epochs", dest="max_epochs", default=3, type=int)
        parser.add_argument("--adafactor", action="store_true")
        return parser

