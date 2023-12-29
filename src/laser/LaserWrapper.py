from transformers import LlamaForCausalLM
from transformers import RobertaForMaskedLM
from transformers import GPTJForCausalLM, DecisionTransformerModel

from laser.gptj_laser import GPTJLaser
from laser.llama2_laser import LLAMA2Laser
from laser.mujoco_dt_laser import MujocoDTLaser
from laser.phi1_5_laser import Phi15Laser
from laser.roberta_laser import RobertaLaser


class LaserWrapper:

    def __init__(self):
        pass

    @staticmethod
    def get_edited_model(model, lname, lnum, rate, intervention="rank-reduction", logger=None, in_place=True):

        if type(model) == LlamaForCausalLM:
            logger.log("Editing a LlamaForCausalLM Model")

            return LLAMA2Laser.get_edited_model(model=model,
                                                lname=lname,
                                                lnum=lnum,
                                                rate=rate,
                                                intervention=intervention,
                                                logger=logger,
                                                in_place=in_place)

        elif type(model) == RobertaForMaskedLM:

            logger.log("Editing a RobertaForMaskedLM Model")
            return RobertaLaser.get_edited_model(model=model,
                                                 lname=lname,
                                                 lnum=lnum,
                                                 rate=rate,
                                                 intervention=intervention,
                                                 logger=logger,
                                                 in_place=in_place)

        elif type(model) == GPTJForCausalLM:

            logger.log("Editing a GPTJForCausalLM Model")
            return GPTJLaser.get_edited_model(model=model,
                                              lname=lname,
                                              lnum=lnum,
                                              rate=rate,
                                              intervention=intervention,
                                              logger=logger,
                                              in_place=in_place)

        elif type(model) == DecisionTransformerModel:

            logger.log("Editing a DecisionTransformer Model")
            return MujocoDTLaser.get_edited_model(model=model,
                                                  lname=lname,
                                                  lnum=lnum,
                                                  rate=rate,
                                                  intervention=intervention,
                                                  logger=logger,
                                                  in_place=in_place)

        elif "modeling_phi.PhiForCausalLM" in str(type(model)):

            logger.log("Editing a Phi1-5 CausalLM Model")
            return Phi15Laser.get_edited_model(model=model,
                                              lname=lname,
                                              lnum=lnum,
                                              rate=rate,
                                              intervention=intervention,
                                              logger=logger,
                                              in_place=in_place)

        else:
            raise AssertionError(f"Unhandled model of type {type(model)}.")
