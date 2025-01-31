import json
from pymatgen.core import Structure
import ase
import pymatgen
from ase.calculators.singlepoint import SinglePointCalculator
from os import listdir
from os.path import isfile, join

mypath = './benchmark_data/snap/NbMoTaW/training'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f != "README.md" and f[-10:] != '_Test.json']

structure_final = []


for file in onlyfiles:
    print(f"{mypath}/{file}")
    with open(f"{mypath}/{file}", "r") as f:
        data = json.load(f)
        #ase.io.read(f)
    
    
    
    for i in range(len(data)):
        structure = Structure.from_dict(data[i]['structure'])
        structure_ASE = ase.Atoms(structure.to_ase_atoms())
    
        forces = data[i]['outputs']['forces']
        energy = data[i]['outputs']['energy']
        stress = data[i]['outputs']['stress']
    
        structure_ASE.forces = forces
        structure_ASE.stress = stress
        structure_ASE.potential_energy = energy
    
        structure_ASE.calc = SinglePointCalculator(
            structure_ASE,
            energy=data[i]['outputs']['energy'],
            forces=data[i]['outputs']['forces'],
            stress = data[i]['outputs']['stress']
        )
        structure_final.append(structure_ASE)
    

ase.io.write('NbMoTaW.xyz', structure_final)