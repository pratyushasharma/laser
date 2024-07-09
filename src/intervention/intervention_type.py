import torch
from copy import deepcopy

from intervention.matrix_utils import prune, sorted_mat, do_low_rank


class AbstractIntervention:

    def __init__(self):
        self.name = "None"

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

    def __str__(self):
        return self.name


class Laser(AbstractIntervention):

    def __init__(self, lname, lnum, rho, is_compress=True):
        super().__init__()
        self.lname = lname
        self.lnum = lnum
        self.rho = rho
        self.is_compress = is_compress
        self.name = f"LASER(lname={self.lname}, lnum={self.lnum}, rho={self.rho})"

    def apply_intervention(self, model, in_place, layer_name_map):

        if in_place:
            model_edit = model
        else:
            model_edit = deepcopy(model)

        param_name = layer_name_map[(self.lname, self.lnum)]
        param = self.get_parameter(model, param_name)

        mat_analysis_tensor = deepcopy(param)
        # TODO implement compress linear layers
        # To do this, we need to find the linear layer containing this matrix
        mat_analysis = do_low_rank(weight=mat_analysis_tensor.type(torch.float32),
                                   rho=self.rho,
                                   is_compress=self.is_compress)

        self.update_model(model_edit, param_name, mat_analysis)

    def __str__(self):
        return self.name


class Pruning(AbstractIntervention):

    def __init__(self, lname, lnum, rho):
        super().__init__()
        self.lname = lname
        self.lnum = lnum
        self.rho = rho
        self.name = f"Pruning(lname={self.lname}, lnum={self.lnum}, rho={self.rho})"

    def apply_intervention(self, model, in_place, layer_name_map):

        if in_place:
            model_edit = model
        else:
            model_edit = deepcopy(model)

        param_name = layer_name_map[(self.lname, self.lnum)]
        param = self.get_parameter(model, param_name)

        mat_analysis = param.detach().numpy().copy()
        mat_sort = sorted_mat(mat_analysis)

        mat_analysis = prune(mat_analysis, mat_sort, self.rho)  # pruned_mat
        mat_analysis = torch.from_numpy(mat_analysis)

        self.update_model(model_edit, param_name, mat_analysis)

    def __str__(self):
        return self.name


class Zero(AbstractIntervention):

    def __init__(self, lname, lnum):
        super().__init__()
        self.lname = lname
        self.lnum = lnum
        self.name = f"Zero(lname={self.lname}, lnum={self.lnum})"

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

    def __str__(self):
        return self.name


class CompoundIntervention(AbstractIntervention):

    def __init__(self, interventions):
        super().__init__()
        self.interventions = interventions

        for intervention in self.interventions:
            assert issubclass(type(intervention), AbstractIntervention), \
                (f"Interventions must be of a subclass of {AbstractIntervention}. "
                 f"For intervention of type {type(intervention)}.")

        self.name = "+".join([str(intervention) for intervention in interventions])

    def apply_intervention(self, model, in_place, layer_name_map):

        if in_place:
            model_edit = model
        else:
            model_edit = deepcopy(model)

        for intervention in self.interventions:
            model_edit = intervention.apply_intervention(model_edit, in_place=True, layer_name_map=layer_name_map)

        return model

    def __str__(self):
        return self.name
