import matplotlib.pyplot as plt

from ase import units
from ase.build import molecule
from ase.io import Trajectory
from ase.md.langevin import Langevin
from fairchem.core import FAIRChemCalculator, pretrained_mlip

predictor = pretrained_mlip.get_predict_unit("uma-s-1")
calc = FAIRChemCalculator(predictor, task_name="omol")

atoms = molecule("H2O")
atoms.info.update(charge=0, spin=1)  # For omol

atoms.calc = calc

dyn = Langevin(
    atoms,
    timestep=0.1 * units.fs,
    temperature_K=400,
    friction=0.001 / units.fs,
)

trajectory = Trajectory("my_md.traj", "w", atoms)
dyn.attach(trajectory.write, interval=1)
dyn.run(steps=50)

# See some results - not paper ready!
traj = Trajectory("my_md.traj")
plt.plot(
    [i * 0.1 * units.fs for i in range(len(traj))],
    [a.get_potential_energy() for a in traj],
)
plt.xlabel("Time (fs)")
plt.ylabel("Energy (eV)");
