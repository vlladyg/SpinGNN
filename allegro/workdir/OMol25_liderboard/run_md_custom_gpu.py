from ase import units
from ase.md.langevin import Langevin
from fairchem.core import pretrained_mlip, FAIRChemCalculator
import time

from fairchem.core.datasets.common_structures import get_fcc_carbon_xtal

predictor = pretrained_mlip.get_predict_unit(
    "uma-s-1p1", inference_settings="turbo", device="cuda", workers=1
)
calc = FAIRChemCalculator(predictor, task_name="omat")

atoms = get_fcc_carbon_xtal(8000)
atoms.calc = calc

dyn = Langevin(
    atoms,
    timestep=0.1 * units.fs,
    temperature_K=400,
    friction=0.001 / units.fs,
)
# warmup 10 steps
dyn.run(steps=10)
start_time = time.time()
dyn.attach(
    lambda: print(
        f"Step: {dyn.get_number_of_steps()}, E: {atoms.get_potential_energy():.3f} eV, "
        f"QPS: {dyn.get_number_of_steps()/(time.time()-start_time):.2f}"
    ),
    interval=1,
)
dyn.run(steps=1000)
