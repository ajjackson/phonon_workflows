#! /usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
from typing import List, Sequence

import ase.io
from ase.optimize import LBFGS
from ase.spacegroup.symmetrize import check_symmetry, FixSymmetry, refine_symmetry
import numpy as np
from phonopy.structure.atoms import PhonopyAtoms
from tblite.ase import TBLite


def main():
    args = get_parser().parse_args()

    atoms = ase.io.read(args.structure)

    refine_symmetry(atoms, verbose=True)

    atoms.calc = TBLite(method="GFN2-xTB", accuracy=args.accuracy)
    optimize_atoms(atoms, fmax=args.fmax)

    supercell_matrix = get_supercell_matrix(args)
    phonon_driver = setup_phonopy(
        atoms,
        supercell_matrix,
        distance=args.distance)

    supercells = phonon_driver.supercells_with_displacements

    phonon_driver.forces = get_displacement_forces(supercells)
    phonon_driver.produce_force_constants()

    phonon_driver.save(settings={'force_constants': True})

def get_displacement_forces(supercells: Sequence[PhonopyAtoms]) -> np.ndarray:
    all_forces = []
    from ase import Atoms

    # Set up structure/calculator using first displacement; this
    # will be mutated to perform other displements efficiently
    atoms = Atoms(supercells[0].symbols,
                  cell=supercells[0].cell,
                  scaled_positions=supercells[0].scaled_positions,
                  pbc=True)
    atoms.calc = TBLite(method="GFN2-xTB", accuracy=0.1)

    for i, displacement in enumerate(supercells):
        print(f"Calculating displacement {i} / {len(supercells)}...")
        atoms.set_scaled_positions(displacement.scaled_positions)
        forces = atoms.get_forces()
        all_forces.append(forces.tolist())

    all_forces = np.asarray(all_forces, dtype=float)
    print(f"Calculated all displacements.")
    return all_forces

def get_supercell_matrix(args):
    if len(args.dim) == 1:
        return np.eye(3) * args.dim[0]
    elif len(args.dim) == 3:
        return np.eye(3) * args.dim
    elif len(args.dim) == 9:
        return np.asarray(args.dim).reshape([3, 3])
    else:
        raise IndexError("--dim must have 1, 3, or 9 values")

def setup_phonopy(atoms, supercell_matrix, guess_primitive=True, distance=0.01):
    from phonopy import Phonopy
    from phonopy.structure.cells import guess_primitive_matrix

    unitcell = PhonopyAtoms(symbols=atoms.get_chemical_symbols(),
                            cell=atoms.cell.array,
                            scaled_positions=atoms.get_scaled_positions()
        )

    if guess_primitive:
        primitive_matrix = guess_primitive_matrix(unitcell)
    else:
        primitive_matrix = np.eye(3)

    phonon = Phonopy(unitcell,
                     supercell_matrix=supercell_matrix,
                     primitive_matrix=primitive_matrix)
    phonon.generate_displacements(distance=distance)
    return phonon

def optimize_atoms(atoms, fmax=1e-4):
    atoms.set_constraint(FixSymmetry(atoms))
    dyn = LBFGS(atoms)
    dyn.run(fmax=fmax)

    atoms.write('optimized.extxyz')


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('structure', type=Path)
    parser.add_argument('--dim', nargs='+', type=int, default=[2, 2, 2],
                        help='Supercell shape, expressed as 1, 3 or 9 integers')
    parser.add_argument('--accuracy', type=float, default=0.1,
                        help='Accuracy parameter for tblite')
    parser.add_argument('-d', '--distance', type=float, default=0.01,
                        help='Finite displacement size')
    parser.add_argument('--fmax', type=float, default=1e-4,
                        help='Force convergence target for geometry optimisation')
    return parser

if __name__ == '__main__':
    main()
