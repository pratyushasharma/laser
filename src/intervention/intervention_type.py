class Laser:

    def __init__(self, lname, lnum, rho):
        self.lname = lname
        self.lnum = lnum
        self.rho = rho


class Pruning:

    def __init__(self, lname, lnum, rho):
        self.lname = lname
        self.lnum = lnum
        self.rho = rho


class Zero:

    def __init__(self, lname, lnum):
        self.lname = lname
        self.lnum = lnum


class CompoundIntervention:

    def __init__(self, interventions):
        self.interventions = interventions
