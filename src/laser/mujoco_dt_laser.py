import torch

from copy import deepcopy
from laser.abstract_laser import AbstractLaser
from laser.matrix_utils import do_low_rank, sorted_mat, prune


class MujocoDTLaser(AbstractLaser):

    def __init__(self):
        super(AbstractLaser, self).__init__()

    @staticmethod
    def convert_name(name):
        if name == "k_proj":
            converted_name = "attn.c_attn.weight"
        elif name == "out_proj":
            converted_name = "attn.c_proj.weight"
        elif name == "fc_in":
            converted_name = "mlp.c_fc.weight"
        elif name == "fc_out":
            converted_name = "mlp.c_proj.weight"
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

                encoder.h.2.ln_1.weight
                encoder.h.2.ln_1.bias
                encoder.h.2.attn.c_attn.weight      -> k_proj
                encoder.h.2.attn.c_attn.bias
                encoder.h.2.attn.c_proj.weight      -> out_proj
                encoder.h.2.attn.c_proj.bias
                encoder.h.2.ln_2.weight
                encoder.h.2.ln_2.bias
                encoder.h.2.mlp.c_fc.weight         -> fc_in
                encoder.h.2.mlp.c_fc.bias
                encoder.h.2.mlp.c_proj.weight       -> fc_out
                encoder.h.2.mlp.c_proj.bias
        '''

        num_update = 0
        for name, param in model.named_parameters():

            if lnum != -1 and not name.startswith(f"encoder.h.{lnum}"):
                continue

            # 'k_proj', 'q_proj', 'v_proj', 'out_proj', 'fc_in', 'fc_out', 'None'
            converted_name = MujocoDTLaser.convert_name(lname)
            if lname != "None" and not name.startswith(f"encoder.h.{lnum}.{converted_name}"):
                continue

            if logger is not None:
                logger.log(f"Updating Layer: encoder.h.{lnum}.{converted_name}")
            print(f"Updating Layer: encoder.h.{lnum}.{converted_name}")

            if intervention == 'dropout':
                # For the sparsity analysis
                mat_analysis = param.detach().numpy().copy()
                mat_sort = sorted_mat(mat_analysis)

                mat_analysis = prune(mat_analysis, mat_sort, rate)  # pruned_mat
                mat_analysis = torch.from_numpy(mat_analysis)

            elif intervention == 'rank-reduction':
                # Do rank reduction
                mat_analysis_tensor = deepcopy(param)
                mat_analysis = do_low_rank(mat_analysis_tensor.type(torch.float32), (10 - rate) * 0.1, niter=20)
            else:
                raise AssertionError(f"Unhandled intervention type {intervention}")

            MujocoDTLaser.update_model(model_edit, name, mat_analysis)
            num_update += 1

        assert num_update == 1, f"Was supposed to make 1 update to the model but instead made {num_update} updates."

        return model_edit
