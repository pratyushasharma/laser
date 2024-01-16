import torch

from copy import deepcopy
from laser.abstract_laser import AbstractLaser
from laser.matrix_utils import do_low_rank, sorted_mat, prune


class LLAMA2Laser(AbstractLaser):

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

            model.layers.20.self_attn.q_proj.weight
            model.layers.20.self_attn.k_proj.weight
            model.layers.20.self_attn.v_proj.weight
            model.layers.20.self_attn.o_proj.weight

            model.layers.20.mlp.gate_proj.weight        -> fc_in
            model.layers.20.mlp.up_proj.weight          -> fc_up
            model.layers.20.mlp.down_proj.weight        -> fc_out

            model.layers.20.input_layernorm.weight
            model.layers.20.post_attention_layernorm.weight
        '''

        if name == "k_proj":
            converted_names = "self_attn.k_proj.weight"
        elif name == "q_proj":
            converted_names = "self_attn.q_proj.weight"
        elif name == "v_proj":
            converted_names = "self_attn.v_proj.weight"
        elif name == "out_proj":
            converted_names = "self_attn.o_proj.weight"
        elif name == "fc_in":
            converted_names = "mlp.gate_proj.weight"
        elif name == "fc_up":
            converted_names = "mlp.up_proj.weight"
        elif name == "fc_out":
            converted_names = "mlp.down_proj.weight"
        elif name == "None":
            converted_names = "None"
        elif name == "mlp":
            converted_names = ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
        elif name == "attn":
            converted_names = ["self_attn.k_proj.weight", "self_attn.q_proj.weight",
                               "self_attn.v_proj.weight", "self_attn.o_proj.weight"]
        elif name == "all":
            converted_names = ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
                               "self_attn.k_proj.weight", "self_attn.q_proj.weight",
                               "self_attn.v_proj.weight", "self_attn.o_proj.weight"]
        else:
            raise AssertionError(f"Unhandled name {name}")

        return converted_names

    @staticmethod
    def _modify_layer(name, lnum_to_modify, lname_to_modify, converted_names):

        # Check for layer number match
        # If must be either -1 meaning modify all layers, or must match the given layer number
        if lnum_to_modify != -1 and not name.startswith(f"model.layers.{lnum_to_modify}."):
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

        converted_names = LLAMA2Laser.convert_name(lname)
        num_update = 0

        for name, param in model.named_parameters():

            modify_flag = LLAMA2Laser._modify_layer(name=name,
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

            LLAMA2Laser.update_model(model_edit, name, mat_analysis)
            num_update += 1

        assert num_update > 0, f"Must update some parameters Llama: {lnum}, {lname}"

        if logger is not None:
            logger.log(f"Total number of parameters updated is {num_update}")

        if lnum != -1 and lname not in ["all", "mlp", "attn"]:
            assert num_update == 1, f"Was supposed to make 1 update to the model but instead made {num_update} updates."

        return model_edit
