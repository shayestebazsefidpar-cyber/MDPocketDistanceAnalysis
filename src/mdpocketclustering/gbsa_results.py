import re

def parse_residue_energies(file_path: Path) -> list[ResidueEnergy]:
    """Parse residue energy log files (decompose*.log).
    Returns a list of residues"""
    residues = []
    pattern = re.compile(r'^\s*([A-Z]{3})-(\d+)\s+([-+]?\d*\.\d+)\s+([-+]?\d*\.\d+)')

    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.match(line)
            if match:
                residues.append(
                    ResidueEnergy(
                        residue=match.group(1),
                        index=int(match.group(2)),
                        energy=float(match.group(3)),
                        std=float(match.group(4))
                    )
                )
    return residues
