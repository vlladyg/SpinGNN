from ase.build import bulk, molecule
from fairchem.core import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

atoms_list = [bulk("Pt"), bulk("Cu"), bulk("NaCl", crystalstructure="rocksalt", a=2.0)]

# you need to assign the task_name desired
atomic_data_list = [
    AtomicData.from_ase(atoms, task_name="omat") for atoms in atoms_list
]
batch = atomicdata_list_to_batch(atomic_data_list)

predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
preds = predictor.predict(batch)

# energy of the first system in the batch
print(preds["energy"][0])

# forces of the first system in the batch
print(preds["forces"][batch.batch == 0])

from ase.build import bulk, molecule, fcc100, add_adsorbate
from fairchem.core import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

# a molecule
h2o = molecule("H2O")
h2o.info.update({"charge": 0, "spin": 1})

# a bulk
pt = bulk("Pt")

# a catalytic surface
slab = fcc100("Cu", (3, 3, 3), vacuum=8, periodic=True)
adsorbate = molecule("CO")
add_adsorbate(slab, adsorbate, 2.0, "bridge")

atomic_data_list = [
    # note that we put the molecule in a large box
    AtomicData.from_ase(
        h2o, task_name="omol", r_data_keys=["spin", "charge"], molecule_cell_size=12
    ),
    AtomicData.from_ase(pt, task_name="omat"),
    AtomicData.from_ase(slab, task_name="oc20"),
]
batch = atomicdata_list_to_batch(atomic_data_list)

predictions = predictor.predict(batch)

print(predictions)
