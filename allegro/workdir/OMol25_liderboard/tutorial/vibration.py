

from ase import Atoms
from ase.optimize import BFGS
from fairchem.core import FAIRChemCalculator, pretrained_mlip
from ase.build import add_adsorbate, fcc111
from ase.optimize import BFGS


predictor = pretrained_mlip.get_predict_unit("uma-s-1")
calc = FAIRChemCalculator(predictor, task_name="omol")

from ase.vibrations import Vibrations

n2 = Atoms("N2", [(0, 0, 0), (0, 0, 1.1)])
n2.info.update({"spin": 1, "charge": 0})
n2.calc = calc

BFGS(n2).run(fmax=0.01)


vib = Vibrations(n2)
vib.run()
vib.summary()
