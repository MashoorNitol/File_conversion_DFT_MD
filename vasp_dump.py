"""
This code converts VASP output files (CONTCAR and OSZICAR) into a single LAMMPS dump file.

The code can process multiple CONTCAR and OSZICAR files and merge them into a single output file.
For example, if you have files named CONTCAR_1, CONTCAR_2, and OSZICAR_1, OSZICAR_2 in a folder,
this code will create an 'out.dump' file that combines the data from all the CONTCAR and OSZICAR files in LAMMPS dump format.
"""

import numpy as np
import glob
import os


# Rest of the code


class ContcarParser:
    def __init__(self, filename):
        self.filename = filename
        self.atom_data = {}
        self.positions = []
        self.basevect = None
        self.total_sum = 0

    def parse(self):
        with open(self.filename, 'r') as file:
            lines = file.readlines()

            total_atoms = lines[6].strip().split()
            self.total_sum = sum(int(atom) for atom in total_atoms)
            scaling = float(lines[1].strip())
            self.basevect = scaling * np.array([list(map(float, lines[i].split())) for i in range(2, 5)])

            # Parsing atom types
            atom_types = lines[5].split()
            num_atoms = [int(x) for x in total_atoms]

            positions_start = 8
            atom_type_index = 1
            for i in range(len(atom_types)):
                atom_type = "type " + str(atom_type_index)
                atom_type_index += 1
                num_atom = num_atoms[i]
                atom_positions = []
                for j in range(positions_start, positions_start + num_atom):
                    position = lines[j].split()
                    atom_positions.append([float(position[0]), float(position[1]), float(position[2])])
                    self.positions.append([float(position[0]), float(position[1]), float(position[2])])
                self.atom_data[atom_type] = np.array(atom_positions)
                positions_start += num_atom

        all_positions = np.array(self.positions)
        cart_coords = np.matmul(all_positions, self.basevect)

        return self.atom_data, cart_coords, self.basevect, self.total_sum


class OszicarReader:
    def __init__(self, filename):
        self.filename = filename

    def read(self):
        with open(self.filename, 'r') as f:
            lines = f.readlines()
        total_energy = float(lines[-1].split()[2])
        return total_energy
    

class OutputWriter:
    def __init__(self, output_file):
        self.output_file = output_file

    def write(self, atom_data, cart_coords, basevect, total_sum, total_energy, fileno):
        lx = np.linalg.norm(basevect[0])
        xy = np.dot(basevect[0], basevect[1]) / lx
        xz = np.dot(basevect[0], basevect[2]) / lx
        ly = np.sqrt(np.linalg.norm(basevect[1]) ** 2 - xy ** 2)
        yz = (np.dot(basevect[1], basevect[2]) - xy * xz) / ly
        lz = np.sqrt(np.linalg.norm(basevect[2]) ** 2 - xz ** 2 - yz ** 2)

        with open(self.output_file, 'a') as f:
            f.write("ITEM: TIMESTEP energy, energy_weight, force_weight, nsims\n")
            f.write(f"{fileno} {total_energy}    1    1   1\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{total_sum}\n")
            f.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
            f.write(f"{min(0.0, xy, xz, xy + xz)} {lx + max(0.0, xy, xz, xy + xz)} {xy}\n")
            f.write(f"{min(0.0, yz)} {ly + max(0.0, yz)} {xz}\n")
            f.write(f"0.0 {lz} {yz}\n")
            f.write("ITEM: ATOMS id type x y z\n")
            index = 1
            for atom_type, positions in atom_data.items():
                for position in positions:
                    line = f"{index} {atom_type.split()[1]} {' '.join(str(coord) for coord in cart_coords[index - 1])}"
                    f.write(line + "\n")
                    index += 1


def process_files(contcar_files, oszicar_files, output_file):
    if os.path.exists(output_file):
        os.remove(output_file)

    all_atom_data = {}
    all_cart_coords = []
    all_basevect = []
    all_total_sum = []
    all_total_energy = []

    for contcar_file in contcar_files:
        parser = ContcarParser(contcar_file)
        atom_data, cart_coords, basevect, total_sum = parser.parse()
        all_atom_data.update(atom_data)
        all_cart_coords.append(cart_coords)
        all_basevect.append(basevect)
        all_total_sum.append(total_sum)

    for oszicar_file in oszicar_files:
        reader = OszicarReader(oszicar_file)
        total_energy = reader.read()
        all_total_energy.append(total_energy)

    writer = OutputWriter(output_file)
    fileno = 1
    for i in range(len(contcar_files)):
        writer.write(all_atom_data, all_cart_coords[i], all_basevect[i],
                     all_total_sum[i], all_total_energy[i], fileno)
        fileno += 1


contcar_files = glob.glob('CONTCAR*')
oszicar_files = glob.glob('OSZICAR*')
output_file = 'out.dump'
process_files(contcar_files, oszicar_files, output_file)
