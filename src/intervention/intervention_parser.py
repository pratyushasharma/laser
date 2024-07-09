from intervention.intervention_type import Laser, CompoundIntervention, Pruning, Zero


class InterventionParser:

    def __init__(self, setup):
        self.args = setup.args

    def parse_interventions(self):

        interventions = []

        if self.args.intervention == "laser":

            lnames = [lname.strip() for lname in self.args.lname.split(",")]
            lnums = [int(lnum.strip()) for lnum in self.args.lnum.split(",")]
            rhos = [float(rho.strip()) for rho in self.args.rho.split(",")]

            if self.args.intervention_read == "sequential":
                # Can only combine interventions together if we have equal number of hyperparameters
                assert len(lnames) == len(lnums) == len(rhos), \
                    f"To compose, we must provide the same number of lnames, lnum, and rhos. Provided: " \
                    f"lnames={len(lnames)}, lnum={len(lnums)}, rhos={len(rhos)}."

                for i in range(len(lnames)):
                    intervention = Laser(lname=lnames[i], lnum=lnums[i], rho=rhos[i])
                    interventions.append(intervention)

            else:
                for lname in lnames:
                    for lnum in lnums:
                        for rho in rhos:
                            intervention = Laser(lname=lname, lnum=lnum, rho=rho)
                            interventions.append(intervention)

        elif self.args.intervention == "zero":

            lnames = [lname.strip() for lname in self.args.lname.split(",")]
            lnums = [int(lnum.strip()) for lnum in self.args.lnum.split(",")]

            intervention_list = []
            for lname in lnames:
                for lnum in lnums:
                    intervention = Zero(lname=lname, lnum=lnum)
                    intervention_list.append(intervention)

            if self.args.intervention_read == "sequential":
                # Can only combine interventions together if we have equal number of hyperparameters
                assert len(lnames) == len(lnums), \
                    f"To compose, we must provide the same number of lname and lnum. Provided: " \
                    f"lnames={len(lnames)}, lnum={len(lnums)}."

                for i in range(len(lnames)):
                    intervention = Zero(lname=lnames[i], lnum=lnums[i])
                    interventions.append(intervention)

            else:
                for lname in lnames:
                    for lnum in lnums:
                        intervention = Zero(lname=lname, lnum=lnum)
                        interventions.append(intervention)

        elif self.args.intervention == "prune":

            lnames = [lname.strip() for lname in self.args.lname.split(",")]
            lnums = [int(lnum.strip()) for lnum in self.args.lnum.split(",")]
            rhos = [float(rho.strip()) for rho in self.args.rho.split(",")]

            if self.args.intervention_read == "sequential":
                # Can only combine interventions together if we have equal number of hyperparameters
                assert len(lnames) == len(lnums) == len(rhos), \
                    f"To compose, we must provide the same number of lnames, lnum, and rhos. Provided: " \
                    f"lnames={len(lnames)}, lnum={len(lnums)}, rhos={len(rhos)}."

                for i in range(len(lnames)):
                    intervention = Pruning(lname=lnames[i], lnum=lnums[i], rho=rhos[i])
                    interventions.append(intervention)

            else:
                for lname in lnames:
                    for lnum in lnums:
                        for rho in rhos:
                            intervention = Laser(lname=lname, lnum=lnum, rho=rho)
                            interventions.append(intervention)

        else:
            raise NotImplementedError(f"Intervention {self.args.intervention} not supported.")

        if self.args.combination == "together":
            # Apply all the interventions together
            interventions = [CompoundIntervention(interventions=interventions)]
        else:
            # Apply all interventions separately
            interventions = intervention_list

        return interventions
