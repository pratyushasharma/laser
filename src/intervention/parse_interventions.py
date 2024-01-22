from intervention.intervention_type import Laser, CompoundIntervention, Pruning, Zero


class InterventionParser:

    def __init__(self, setup):
        self.args = setup.args

    def create_grid_search_interventions(self):

        interventions = []

        if self.args.intervention == "laser":

            lnames = [lname.strip() for lname in self.args.lname.split(",")]
            lnums = [int(lnum.strip()) for lnum in self.args.lnum.split(",")]
            rhos = [float(rho.strip()) for rho in self.args.rho.split(",")]

            intervention_list = []
            for lname in lnames:
                for lnum in lnums:
                    for rho in rhos:
                        intervention = Laser(lname=lname, lnum=lnum, rho=rho)
                        intervention_list.append(intervention)

            if self.args.compose:
                # Can only compose if equal number of hyperparameters
                assert len(lnames) == len(lnums) == len(rhos), \
                    f"To compose, we must provide the same number of lnames, lnum, and rhos. Provided: " \
                    f"lnames={len(lnames)}, lnum={len(lnums)}, rhos={len(rhos)}."

        elif self.args.intervention == "zero":

            lnames = [lname.strip() for lname in self.args.lname.split(",")]
            lnums = [int(lnum.strip()) for lnum in self.args.lnum.split(",")]

            intervention_list = []
            for lname in lnames:
                for lnum in lnums:
                    intervention = Zero(lname=lname, lnum=lnum)
                    intervention_list.append(intervention)

            if self.args.compose:
                # Can only compose if equal number of hyperparameters
                assert len(lnames) == len(lnums), \
                    f"To compose, we must provide the same number of lnames and lnums. Provided: " \
                    f"lnames={len(lnames)}, and lnum={len(lnums)}."

        elif self.args.intervention == "prune":

            lnames = [lname.strip() for lname in self.args.lname.split(",")]
            lnums = [int(lnum.strip()) for lnum in self.args.lnum.split(",")]
            rhos = [float(rho.strip()) for rho in self.args.rho.split(",")]

            intervention_list = []
            for lname in lnames:
                for lnum in lnums:
                    for rho in rhos:
                        intervention = Pruning(lname=lname, lnum=lnum, rho=rho)
                        intervention_list.append(intervention)

            if self.args.compose:
                # Can only compose if equal number of hyperparameters
                assert len(lnames) == len(lnums) == len(rhos), \
                    f"To compose, we must provide the same number of lnames, lnum, and rhos. Provided: " \
                    f"lnames={len(lnames)}, lnum={len(lnums)}, rhos={len(rhos)}."

        else:
            raise NotImplementedError(f"Intervention {self.args.intervention} not supported.")

        if self.args.compose:
            # Compose them
            interventions = [CompoundIntervention(interventions=intervention_list)]
        else:
            # Set grid search
            interventions = intervention_list

        return interventions
