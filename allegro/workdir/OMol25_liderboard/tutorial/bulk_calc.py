from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from fairchem.core import FAIRChemCalculator, pretrained_mlip

predictor = pretrained_mlip.get_predict_unit("uma-s-1")
calc = FAIRChemCalculator(predictor, task_name="omat")

atoms = bulk("Fe")
atoms.calc = calc

opt = FIRE(FrechetCellFilter(atoms))
opt.run(0.05, 100)

print(atoms.get_stress())  # !!!! We get stress now!
