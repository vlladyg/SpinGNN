from ase.atoms import Atom, Atoms
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from fairchem.core import FAIRChemCalculator, pretrained_mlip

predictor = pretrained_mlip.get_predict_unit("uma-s-1")

cu = Atoms(
    [Atom("Cu", [0.000, 0.000, 0.000])],
    cell=[[1.818, 0.000, 1.818], [1.818, 1.818, 0.000], [0.000, 1.818, 1.818]],
    pbc=True,
)
cu.calc = FAIRChemCalculator(predictor, task_name="omat")

opt = FIRE(FrechetCellFilter(cu))
opt.run(0.05, 100)

print(cu.get_potential_energy())

pd = Atoms(
    [Atom("Pd", [0.000, 0.000, 0.000])],
    cell=[[1.978, 0.000, 1.978], [1.978, 1.978, 0.000], [0.000, 1.978, 1.978]],
    pbc=True,
)
pd.calc = FAIRChemCalculator(predictor, task_name="omat")

opt = FIRE(FrechetCellFilter(pd))
opt.run(0.05, 100)

print(pd.get_potential_energy())


cupd1 = Atoms(
    [Atom("Cu", [0.000, 0.000, 0.000]), Atom("Pd", [-1.652, 0.000, 2.039])],
    cell=[[0.000, -2.039, 2.039], [0.000, 2.039, 2.039], [-3.303, 0.000, 0.000]],
    pbc=True,
)  # Note pbc=True is important, it is not the default and OMAT

cupd1.calc = FAIRChemCalculator(predictor, task_name="omat")

opt = FIRE(FrechetCellFilter(cupd1))
opt.run(0.05, 100)

print(cupd1.get_potential_energy())


cupd2 = Atoms(
    [
        Atom("Cu", [-0.049, 0.049, 0.049]),
        Atom("Cu", [-11.170, 11.170, 11.170]),
        Atom("Pd", [-7.415, 7.415, 7.415]),
        Atom("Pd", [-3.804, 3.804, 3.804]),
    ],
    cell=[[-5.629, 3.701, 5.629], [-3.701, 5.629, 5.629], [-5.629, 5.629, 3.701]],
    pbc=True,
)
cupd2.calc = FAIRChemCalculator(predictor, task_name="omat")

opt = FIRE(FrechetCellFilter(cupd2))
opt.run(0.05, 100)

print(cupd2.get_potential_energy())


# Delta Hf cupd-1 = -0.11 eV/atom
hf1 = cupd1.get_potential_energy() - cu.get_potential_energy() - pd.get_potential_energy()
print(hf1)
hf2 = (
    cupd2.get_potential_energy()
    - 2 * cu.get_potential_energy()
    - 2 * pd.get_potential_energy()
)
print(hf2)
