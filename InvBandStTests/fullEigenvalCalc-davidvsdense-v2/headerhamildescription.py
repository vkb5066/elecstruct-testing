#Contains a class that is useful for yielding the stuff necessary for building / acting on
#the hamiltonian

class hamil:
    ##kinetic energy
    tmul = None ##hbar^2/2m^*, the multiplictive factor in front of |k + G|^2 for kinetic energy
                ##(the m^* can optionally include local effects, if its = m then tmul ~ 3.81 eV A^2)

    ##potential energy
    vlsizes = [None, None, None] ##len h, len k, len l
    vllog2sizes = [None, None, None] ##log_2(vlsizes) -- useful for fft algos
    vl = [] ##local potential grid

    def __init__(self):
        self.tmul = None

        self.vlsizes = [None, None, None]
        self.vllog2sizes = [None, None, None]
        self.vl = []

    def prep(self):
        pass
