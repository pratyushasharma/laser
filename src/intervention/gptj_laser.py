import torch

from copy import deepcopy
from laser.abstract_laser import AbstractLaser
from laser.matrix_utils import do_low_rank, sorted_mat, prune


class GPTJLaser(AbstractLaser):

    def __init__(self):
        super(AbstractLaser, self).__init__()

    @staticmethod
    def convert_name(name):
        """
        Convert a given generic transformer layer name to a model specific layer name(s)
        :param name: generic name
        :return: model specific layer name(s)
        """

        ''' 
            For a given layer, we can modify the following type individually or all at onces
    
            transformer.h.26.ln_1.weight
            transformer.h.26.ln_1.bias
            transformer.h.26.attn.k_proj.weight     -> k_proj
            transformer.h.26.attn.v_proj.weight     -> v_proj
            transformer.h.26.attn.q_proj.weight     -> q_proj
            transformer.h.26.attn.out_proj.weight   -> out_proj
            transformer.h.26.mlp.fc_in.weight       -> fc_in
            transformer.h.26.mlp.fc_out.weight      -> fc_out
        '''

        if name == "k_proj":
            converted_names = "attn.k_proj.weight"
        elif name == "q_proj":
            converted_names = "attn.q_proj.weight"
        elif name == "v_proj":
            converted_names = "attn.v_proj.weight"
        elif name == "out_proj":
            converted_names = "attn.out_proj.weight"
        elif name == "fc_in":
            converted_names = "mlp.fc_in.weight"
        elif name == "fc_out":
            converted_names = "mlp.fc_out.weight"
        elif name == "None":
            converted_names = "None"
        elif name == "mlp":
            converted_names = ["mlp.fc_in.weight", "mlp.fc_out.weight"]
        elif name == "attn":
            converted_names = ["attn.k_proj.weight", "attn.q_proj.weight", "attn.v_proj.weight", "attn.out_proj.weight"]
        elif name == "all":
            converted_names = ["attn.k_proj.weight", "attn.q_proj.weight", "attn.v_proj.weight",
                               "attn.out_proj.weight", "mlp.fc_in.weight", "mlp.fc_out.weight"]
        else:
            raise AssertionError(f"Unhandled name {name}")

        return converted_names

    @staticmethod
    def _modify_layer(name, lnum_to_modify, lname_to_modify, converted_names):

        # Check for layer number match
        # If must be either -1 meaning modify all layers, or must match the given layer number
        if lnum_to_modify != -1 and not name.startswith(f"transformer.h.{lnum_to_modify}."):
            return False

        # Check if layer type needs to be modified.
        #      'all', 'mlp', 'attn', 'k_proj', 'q_proj', 'v_proj', 'out_proj', 'fc_in', 'fc_out'
        # If all, then modify all
        # If mlp, then only MLP
        # If attn, then only attn
        # Otherwise, update a given layer type

        if type(converted_names) == list:
            modify_flag = any([name.endswith(f"{converted_name}") for converted_name in converted_names])
        elif type(converted_names) == str:
            modify_flag = name.endswith(f"{converted_names}")
        else:
            raise AssertionError(f"Type should be list or str. Found {type(converted_names)}.")

        return modify_flag

    @staticmethod
    def get_edited_model(model, lname, lnum, rate, intervention="rank-reduction", logger=None, in_place=True):

        if in_place:
            model_edit = model
        else:
            model_edit = deepcopy(model)

        if lname == "dont":
            print(f"Not intervening at all")
            return model_edit

        converted_names = GPTJLaser.convert_name(lname)
        num_update = 0

        for name, param in model.named_parameters():

            modify_flag = GPTJLaser._modify_layer(name=name,
                                                  lnum_to_modify=lnum,
                                                  lname_to_modify=lname,
                                                  converted_names=converted_names)

            if modify_flag:
                if logger is not None:
                    logger.log(f"Updating Layer: {name}")
                print(f"Updating Layer: {name}")
            else:
                continue

            if intervention == 'dropout':
                # For the sparsity analysis
                mat_analysis = param.detach().numpy().copy()
                mat_sort = sorted_mat(mat_analysis)

                mat_analysis = prune(mat_analysis, mat_sort, rate)  # pruned_mat
                mat_analysis = torch.from_numpy(mat_analysis)

            elif intervention == 'rank-reduction':
                # Do rank reduction
                mat_analysis_tensor = deepcopy(param)
                mat_analysis = do_low_rank(mat_analysis_tensor.type(torch.float32), (10 - rate) * 0.1)

            elif intervention == 'zero':
                mat_analysis_tensor = deepcopy(param)
                mat_analysis = 0.0 * mat_analysis_tensor.type(torch.float32)
            else:
                raise AssertionError(f"Unhandled intervention type {intervention}")

            GPTJLaser.update_model(model_edit, name, mat_analysis)
            num_update += 1

        assert num_update > 0, f"Must update some parameters GPTJ: {lnum}, {lname}"

        if logger is not None:
            logger.log(f"Total number of parameters updated is {num_update}")

        if lnum != -1 and lname not in ["all", "mlp", "attn"]:
            assert num_update == 1, f"Was supposed to make 1 update to the model but instead made {num_update} updates."

        return model_edit
