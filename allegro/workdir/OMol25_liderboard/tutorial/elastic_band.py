from ase.build import add_adsorbate, fcc111, molecule
from ase.optimize import LBFGS
from fairchem.core import FAIRChemCalculator, pretrained_mlip

predictor = pretrained_mlip.get_predict_unit("uma-s-1")
calc = FAIRChemCalculator(predictor, task_name="oc20")

# Set up your system as an ASE atoms object
initial = fcc111("Pt", (3, 3, 3), vacuum=8, periodic=True)

adsorbate = molecule("O")
add_adsorbate(initial, adsorbate, 2.0, "fcc")
initial.calc = calc

# Set up LBFGS dynamics object
opt = LBFGS(initial)
opt.run(0.05, 100)
print(initial.get_potential_energy())

# Set up your system as an ASE atoms object
final = fcc111("Pt", (3, 3, 3), vacuum=8, periodic=True)

adsorbate = molecule("O")
add_adsorbate(final, adsorbate, 2.0, "hcp")
final.calc = FAIRChemCalculator(predictor, task_name="oc20")

# Set up LBFGS dynamics object
opt = LBFGS(final)
opt.run(0.05, 100)
print(final.get_potential_energy())

from ase.mep import NEB

images = [initial]
for i in range(3):
    image = initial.copy()
    image.calc = FAIRChemCalculator(predictor, task_name="oc20")
    images.append(image)

images.append(final)


neb = NEB(images)
neb.interpolate()

opt = LBFGS(neb, trajectory="neb.traj")
opt.run(0.05, 100)
