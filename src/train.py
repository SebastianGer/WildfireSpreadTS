from pytorch_lightning.utilities import rank_zero_only
import torch
from dataloader.FireSpreadDataModule import FireSpreadDataModule
from pytorch_lightning.cli import LightningCLI
# from models import SMPModel, BaseModel, ConvLSTMLightning, LogisticRegression # noqa
from models import BaseModel
import wandb
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
torch.set_float32_matmul_precision('high')


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("trainer.default_root_dir",
                              "trainer.logger.init_args.save_dir")
        parser.link_arguments("model.class_path",
                              "trainer.logger.init_args.name")
        parser.add_argument("--do_train", type=bool,
                            help="If True: skip training the model.")
        parser.add_argument("--do_predict", type=bool,
                            help="If True: compute predictions.")
        parser.add_argument("--do_test", type=bool,
                            help="If True: compute test metrics.")
        parser.add_argument("--do_validate", type=bool,
                            default=False, help="If True: compute val metrics.")
        parser.add_argument("--ckpt_path", type=str, default=None,
                            help="Path to checkpoint to load for resuming training, for testing and predicting.")

    def before_fit(self):
        self.wandb_setup()

    def before_test(self):
        self.wandb_setup()

    def before_validate(self):
        self.wandb_setup()

    @rank_zero_only
    def wandb_setup(self):
        """
        Save the config used by LightningCLI to disk, then save that file to wandb.
        Using wandb.config adds some strange formating that means we'd have to do some 
        processing to be able to use it again as CLI input.

        Also define min and max metrics in wandb, because otherwise it just reports the 
        last known values, which is not what we want.
        """
        config_file_name = os.path.join(wandb.run.dir, "cli_config.yaml")

        # Seems like dump has a different behavior than the direct self.parser.save.
        # It also contains the changes we make via commandline arguments after indicating a config file,
        # e.g. changing data.batch_size after also passing a data cfg file.
        # The resulting cfg contains everything, instead of being distributed over multiple files,
        # like the initial command. But as long as we get all of the cfg options, this is fine.
        # With self.parser.save, we would get the data cfg path, but not the change in batch_size.
        cfg_string = self.parser.dump(self.config, skip_none=False)
        with open(config_file_name, "w") as f:
            f.write(cfg_string)
        wandb.save(config_file_name, policy="now", base_path=wandb.run.dir)
        wandb.define_metric("train_loss_epoch", summary="min")
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("train_f1_epoch", summary="max")
        wandb.define_metric("val_f1", summary="max")


def main():

    cli = MyLightningCLI(BaseModel, FireSpreadDataModule, subclass_mode_model=True, save_config_kwargs={
        "overwrite": True}, parser_kwargs={"parser_mode": "yaml"}, run=False)
    cli.wandb_setup()

    if cli.config.do_train:
        # compiled_model = torch.compile(cli.model)
        compiled_model = cli.model
        cli.trainer.fit(compiled_model, cli.datamodule,
                        ckpt_path=cli.config.ckpt_path)

    # If we have trained a model, use the best checkpoint for testing and predicting.
    # Without this, the model's state at the end of the training would be used,
    # which is not necessarily the best.
    ckpt = cli.config.ckpt_path
    if cli.config.do_train:
        ckpt = "best"

    if cli.config.do_validate:
        cli.trainer.validate(cli.model, cli.datamodule, ckpt_path=ckpt)

    if cli.config.do_test:
        cli.trainer.test(cli.model, cli.datamodule, ckpt_path=ckpt)

    if cli.config.do_predict:

        prediction_output = cli.trainer.predict(
            cli.model, cli.datamodule, ckpt_path=ckpt)
        x_af = torch.cat(
            list(map(lambda tup: tup[0][:, -1, :, :].squeeze(), prediction_output)), axis=0)
        y = torch.cat(list(map(lambda tup: tup[1], prediction_output)), axis=0)
        y_hat = torch.cat(
            list(map(lambda tup: tup[2], prediction_output)), axis=0)
        fire_masks_combined = torch.cat(
            [x_af.unsqueeze(0), y_hat.unsqueeze(0), y.unsqueeze(0)], axis=0)

        predictions_file_name = os.path.join(
            cli.config.trainer.default_root_dir, f"predictions_{wandb.run.id}.pt")
        torch.save(fire_masks_combined, predictions_file_name)


if __name__ == "__main__":
    main()
