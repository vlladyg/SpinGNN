from __future__ import annotations

from ase.build import bulk
from quacc.recipes.mlp.elastic import elastic_tensor_flow

# Make an Atoms object of a bulk Cu structure
atoms = bulk("Cu")

# Run an elastic property calculation with our favorite MLP potential
result = elastic_tensor_flow(
    atoms,
    job_params={
        "all": dict(
            method="fairchem",
            name_or_path="uma-s-1p1",
            task_name="omat",
        ),
    },
)

print(result)
