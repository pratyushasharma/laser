import torch

from copy import deepcopy
from laser.abstract_laser import AbstractLaser
from laser.matrix_utils import do_low_rank, sorted_mat, prune


class RobertaLaser(AbstractLaser):

    def __init__(self):
        pass

    @staticmethod
    def convert_name(name):

        if name == "k_proj":
            converted_name = "attention.self.key.weight"
        elif name == "q_proj":
            converted_name = "attention.self.query.weight"
        elif name == "v_proj":
            converted_name = "attention.self.value.weight"
        elif name == "out_proj":
            converted_name = "attention.output.dense.weight"
        elif name == "fc_in":
            converted_name = "intermediate.dense.weight"
        elif name == "fc_out":
            converted_name = "output.dense.weight"
        elif name == "None":
            converted_name = "None"
        else:
            raise AssertionError(f"Unhandled name {name}")

        return converted_name

    @staticmethod
    def get_edited_model(model, lname, lnum, rate, intervention="rank-reduction", logger=None, in_place=True):

        if in_place:
            model_edit = model
        else:
            model_edit = deepcopy(model)

        if lname == "dont":
            print(f"Not intervening at all")
            return model_edit

        ''' 
            For a given layer, we can modify the following type individually or all at onces
            roberta.encoder.layer.1.attention.self.query.weight
            roberta.encoder.layer.1.attention.self.key.weight
            roberta.encoder.layer.1.attention.self.value.weight
            roberta.encoder.layer.1.attention.output.dense.weight
            roberta.encoder.layer.1.intermediate.dense.weight
            roberta.encoder.layer.1.output.dense.weight
        '''

        num_update = 0
        for name, param in model.named_parameters():

            if lnum != 12 and not name.startswith(f"roberta.encoder.layer.{lnum}"):
                continue

            # 'k_proj', 'q_proj', 'v_proj', 'out_proj', 'fc_in', 'fc_out', 'None'
            converted_name = RobertaLaser.convert_name(lname)
            if lname != "None" and not name.startswith(f"roberta.encoder.layer.{lnum}.{converted_name}"):
                continue

            if logger is not None:
                logger.log(f"Updating Layer: roberta.encoder.layer.{lnum}.{converted_name}")
            print(f"Updating Layer: roberta.encoder.layer.{lnum}.{converted_name}")

            # For the sparsity analysis
            mat_analysis = param.detach().numpy().copy()
            mat_sort = sorted_mat(mat_analysis)

            mat_analysis = param.detach().numpy().copy()
            mat_analysis_tensor = deepcopy(param)

            if intervention == 'dropout':
                mat_analysis = prune(mat_analysis, mat_sort, rate)  # pruned_mat
                mat_analysis = torch.from_numpy(mat_analysis)

            elif intervention == 'rank-reduction':
                # Do rank reduction
                mat_analysis = do_low_rank(mat_analysis_tensor.type(torch.float32), (10 - rate) * 0.1, niter=20)

            elif intervention == 'zero':
                mat_analysis_tensor = deepcopy(param)
                mat_analysis = 0.0 * mat_analysis_tensor.type(torch.float32)

            else:
                raise AssertionError(f"Unhandled intervention type {intervention}")

            RobertaLaser.update_model(model_edit, name, mat_analysis)
            num_update += 1

        assert num_update == 1, f"Was supposed to make 1 update to the model but instead made {num_update} updates."

        return model_edit
