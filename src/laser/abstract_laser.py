import torch


class AbstractLaser:

    # Helper functions for matrix update
    @staticmethod
    def get_parameter(model, name):
        for n, p in model.named_parameters():
            if n == name:
                return p
        raise LookupError(name)

    @staticmethod
    def update_model(model, name, params):
        with torch.no_grad():
            AbstractLaser.get_parameter(model, name)[...] = params

    @staticmethod
    def get_edited_model(model, lname, lnum, rate, intervention="rank-reduction", logger=None, in_place=True):
        raise NotImplementedError()
