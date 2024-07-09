import torch
from copy import deepcopy

from intervention.matrix_utils import prune, sorted_mat, do_low_rank


class AbstractIntervention:

    def __init__(self):
        pass

    @staticmethod
    def get_parameter(model, name):
        for n, p in model.named_parameters():
            if n == name:
                return p
        raise LookupError(name)

    @staticmethod
    def update_model(model, name, params):
        with torch.no_grad():
            AbstractIntervention.get_parameter(model, name)[...] = params

    def apply_intervention(self, model, in_place, layer_name_map):
        """
            Apply an intervention to a given model
            :param model: Model to be edited
            :param in_pace: If true then apply the intervention to the same model, else to a copy of the model
            :param layer_name_map: mapping from (layer_name, layer_number) to the weight matrix of an LLM. Designed to
                                  make the operations agnostic to an LLM.
        """
        raise NotImplementedError()


class Laser(AbstractIntervention):

    def __init__(self, lname, lnum, rho):
        super().__init__()
        self.lname = lname
        self.lnum = lnum
        self.rho = rho

    def apply_intervention(self, model, in_place, layer_name_map):

        if in_place:
            model_edit = model
        else:
            model_edit = deepcopy(model)

        param_name = layer_name_map[(self.lname, self.lnum)]
        param = self.get_parameter(model, param_name)

        mat_analysis_tensor = deepcopy(param)
        mat_analysis = do_low_rank(mat_analysis_tensor.type(torch.float32), (10 - rate) * 0.1)

        self.update_model(model_edit, param_name, mat_analysis)


class Pruning(AbstractIntervention):

    def __init__(self, lname, lnum, rho):
        super().__init__()
        self.lname = lname
        self.lnum = lnum
        self.rho = rho

    def apply_intervention(self, model, in_place, layer_name_map):

        if in_place:
            model_edit = model
        else:
            model_edit = deepcopy(model)

        param_name = layer_name_map[(self.lname, self.lnum)]
        param = self.get_parameter(model, param_name)

        mat_analysis = param.detach().numpy().copy()
        mat_sort = sorted_mat(mat_analysis)

        mat_analysis = prune(mat_analysis, mat_sort, rate)  # pruned_mat
        mat_analysis = torch.from_numpy(mat_analysis)

        self.update_model(model_edit, param_name, mat_analysis)


class Zero(AbstractIntervention):

    def __init__(self, lname, lnum):
        super().__init__()
        self.lname = lname
        self.lnum = lnum

    def apply_intervention(self, model, in_place, layer_name_map):

        if in_place:
            model_edit = model
        else:
            model_edit = deepcopy(model)

        param_name = layer_name_map[(self.lname, self.lnum)]
        param = self.get_parameter(model, param_name)

        mat_analysis_tensor = deepcopy(param)
        mat_analysis = 0.0 * mat_analysis_tensor.type(torch.float32)

        self.update_model(model_edit, param_name, mat_analysis)


class CompoundIntervention(AbstractIntervention):

    def __init__(self, interventions):
        super().__init__()
        self.interventions = interventions
        for intervention in self.interventions:
            assert issubclass(type(intervention), AbstractIntervention), \
                (f"Interventions must be of a subclass of {AbstractIntervention}. "
                 f"For intervention of type {type(intervention)}.")

    def apply_intervention(self, model, in_place, layer_name_map):

        if in_place:
            model_edit = model
        else:
            model_edit = deepcopy(model)

        for intervention in self.interventions:
            model_edit = intervention.apply_intervention(model_edit, in_place=True, layer_name_map=layer_name_map)

        return model
