#! /usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
from typing import List, Sequence

from ase import Atoms
from ase.build import make_supercell
import ase.io
# from ase.constraints import UnitCellFilter
from ase.geometry import get_duplicate_atoms
from ase.optimize import LBFGS
from ase.spacegroup.symmetrize import check_symmetry, FixSymmetry, refine_symmetry
import numpy as np
from phonopy.structure.atoms import PhonopyAtoms
from tblite.ase import TBLite


def main():
    args = get_parser().parse_args()
    tblite_kwargs = dict(method=args.method, accuracy=args.accuracy)

    atoms = ase.io.read(args.structure)

    refine_symmetry(atoms, verbose=True)

    atoms.calc = TBLite(**tblite_kwargs)
    optimize_atoms(atoms, fmax=args.fmax)

    supercell_matrix = get_supercell_matrix(args)

    if args.opt_sc:
        supercell = make_supercell(atoms, supercell_matrix)
        supercell.calc = TBLite(**tblite_kwargs)
        optimize_atoms(supercell, fmax=args.fmax, write=False, adjust_cell=False)
        optimize_atoms(supercell, fmax=args.fmax, write=False, adjust_cell=True)

        # Fold supercell back onto unit cell
        opt_cell = np.linalg.solve(supercell_matrix, supercell.cell.array)
        folded_atoms = supercell.copy()
        folded_atoms.set_constraint()  # Remove symmetry constraint before deleting things
        folded_atoms.set_cell(opt_cell, scale_atoms=False)
        folded_atoms.wrap()

        # Take average positions
        duplicates = get_duplicate_atoms(folded_atoms, delete=False)
        while duplicates.any():
            active_index = duplicates[0, 0]
            images = duplicates[duplicates[:, 0]==active_index][:, 1].tolist()

            folded_atoms[active_index].position = np.mean(folded_atoms.positions[images + [active_index]], axis=0)
            print("Removing duplicate atoms", images)

            del folded_atoms[images]

            duplicates = get_duplicate_atoms(folded_atoms, delete=False)

        atoms = folded_atoms

    phonon_driver = setup_phonopy(
        atoms,
        supercell_matrix,
        distance=args.distance)

    supercells = phonon_driver.supercells_with_displacements

    phonon_driver.forces = get_displacement_forces(
        supercells,
        accuracy=args.accuracy)
    phonon_driver.produce_force_constants()

    phonon_driver.save(settings={'force_constants': True})

def get_displacement_forces(supercells: Sequence[PhonopyAtoms],
                            **tblite_kwargs) -> np.ndarray:
    all_forces = []

    # Set up structure/calculator using first displacement; this
    # will be mutated to perform other displements efficiently
    atoms = Atoms(supercells[0].symbols,
                  cell=supercells[0].cell,
                  scaled_positions=supercells[0].scaled_positions,
                  pbc=True)
    atoms.calc = TBLite(**tblite_kwargs)

    for i, displacement in enumerate(supercells):
        print(f"Calculating displacement {i+1} / {len(supercells)} ...")
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

def optimize_atoms(atoms: Atoms,
                   fmax: float = 1e-4,
                   write: bool = True,
                   adjust_cell: bool = True) -> None:
    atoms.set_constraint(FixSymmetry(atoms, adjust_cell=adjust_cell))
    dyn = LBFGS(atoms)
    dyn.run(fmax=fmax)

    if write:
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
    parser.add_argument('--method', type=str, default='GFN2-xTB',
                        help='Force convergence target for geometry optimisation')
    parser.add_argument('--opt-sc', action='store_true', dest='opt_sc',
                        help='Geometry-optimise supercell and fold to primitive')
    return parser

if __name__ == '__main__':
    main()
