from fairchem.core import FAIRChemCalculator, pretrained_mlip

predictor = pretrained_mlip.get_predict_unit("uma-s-1")

from ase.build import molecule

#  singlet CH2
singlet = molecule("CH2_s1A1d")
singlet.info.update({"spin": 1, "charge": 0})
singlet.calc = FAIRChemCalculator(predictor, task_name="omol")

#  triplet CH2
triplet = molecule("CH2_s3B1d")
triplet.info.update({"spin": 3, "charge": 0})
triplet.calc = FAIRChemCalculator(predictor, task_name="omol")

print(triplet.get_potential_energy() - singlet.get_potential_energy())

