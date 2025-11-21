from ase.build import bulk
from ase.phonons import Phonons
from fairchem.core import FAIRChemCalculator, pretrained_mlip

predictor = pretrained_mlip.get_predict_unit("uma-s-1")
calc = FAIRChemCalculator(predictor, task_name="omat")

# Setup crystal
atoms = bulk("Al", "fcc", a=4.05)

# Phonon calculator
N = 7
ph = Phonons(atoms, calc, supercell=(N, N, N), delta=0.05)
ph.run()

# Read forces and assemble the dynamical matrix
ph.read(acoustic=True)
ph.clean()

path = atoms.cell.bandpath("GXULGK", npoints=100)
bs = ph.get_band_structure(path)

dos = ph.get_dos(kpts=(20, 20, 20)).sample_grid(npts=100, width=1e-3)

# Plot the band structure and DOS:
import matplotlib.pyplot as plt  # noqa

fig = plt.figure(figsize=(7, 4))
ax = fig.add_axes([0.12, 0.07, 0.67, 0.85])

emax = 0.04
bs.plot(ax=ax, emin=0.0, emax=emax)

dosax = fig.add_axes([0.8, 0.07, 0.17, 0.85])
dosax.fill_between(
    dos.get_weights(),
    dos.get_energies(),
    y2=0,
    color="grey",
    edgecolor="k",
    lw=1,
)

dosax.set_ylim(0, emax)
dosax.set_yticks([])
dosax.set_xticks([])
dosax.set_xlabel("DOS", fontsize=18);
