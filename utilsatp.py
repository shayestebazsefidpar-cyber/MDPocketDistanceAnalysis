import re
from pathlib import Path
from typing import List

# ------------------------------
# Data classes
# ------------------------------


@dataclass
class SimulationResult:
    mutation: str
    replicate: str
    has_mg: bool
    has_atp: bool
    binding_energy: float  # can be None
    residues: list
    path: Path

    def __repr__(self):
        be = f"{self.binding_energy:.3f}" if self.binding_energy is not None else "None"
        return (
            f"SimulationResult("
            f"mutation={self.mutation}, "
            f"replicate={self.replicate}, "
            f"ATP={self.has_atp}, MG={self.has_mg}, "
            f"binding_energy={be}, "
            f"n_residues={len(self.residues)})"
        )

    def get_trajectory(self):
        return self.path / "md_nopbc_center_fit_no_water.xtc"

    def get_topology(self):
        return self.path / "md-sin-water.tpr"


# ------------------------------
# Parsers
# ------------------------------
def parse_residue_energies(file_path: Path):
    """Parse residue energy log files (decompose*.log)."""
    residues = []
    pattern = re.compile(r"^\s*([A-Z]{3})-(\d+)\s+([-+]?\d*\.\d+)\s+([-+]?\d*\.\d+)")

    with open(file_path, "r") as f:
        for line in f:
            match = pattern.match(line)
            if match:
                residues.append(
                    ResidueEnergy(
                        residue=match.group(1),
                        index=int(match.group(2)),
                        energy=float(match.group(3)),
                        std=float(match.group(4)),
                    )
                )
    return residues


def get_binding_energy(dat_file: Path):
    """Extract binding energy from summary_energy.dat file."""
    if not dat_file or not dat_file.exists():
        return None
    with open(dat_file, "r") as f:
        for line in f:
            if "Binding energy" in line:
                return float(line.split("=")[1].split()[0])
    return None


# ------------------------------
# Metadata extraction
# ------------------------------
def parse_metadata(sim_path: Path):
    """
    Extract metadata from folder structure:
    mutation/ligand/mg/replicate
    """
    mutation = sim_path.parts[-4]  # e.g., y516c
    ligand = sim_path.parts[-3]  # atp/adp
    mg_state = sim_path.parts[-2]  # mg/nomg
    replicate = sim_path.parts[-1]  # rep1, rep2, ...

    has_atp = ligand.lower() == "atp"
    has_mg = mg_state.lower() == "mg"

    return mutation, replicate, has_mg, has_atp


# ------------------------------
# Simulation parser
# ------------------------------
def parse_simulation(sim_path: Path) -> SimulationResult:
    """
    Automatically detect GBSA files and create SimulationResult.
    Expects the new organized structure.
    """
    mutation, replicate, has_mg, has_atp = parse_metadata(sim_path)

    # GBSA files
    dat_file = next(sim_path.glob("summary_energy*.dat"), None)
    log_file = next(sim_path.glob("decompose*.log"), None)

    binding_energy = get_binding_energy(dat_file)
    residues = parse_residue_energies(log_file) if log_file else []

    return SimulationResult(
        mutation=mutation,
        replicate=replicate,
        has_mg=has_mg,
        has_atp=has_atp,
        binding_energy=binding_energy,
        residues=residues,
        path=sim_path,
    )


def parse_all_simulations(root_dir: Path) -> List[SimulationResult]:
    """
    Recursively parse all simulations in organized folder structure:
    mutation/ligand/mg/replicate
    Returns a list of SimulationResult objects.
    """
    results = []

    # iterate over mutation folders
    for mutation_dir in root_dir.iterdir():
        if not mutation_dir.is_dir():
            continue

        # iterate over ligand (atp/adp)
        for ligand_dir in mutation_dir.iterdir():
            if not ligand_dir.is_dir():
                continue

            # iterate over mg state (mg/nomg)
            for mg_dir in ligand_dir.iterdir():
                if not mg_dir.is_dir():
                    continue

                # iterate over replicates
                for rep_dir in mg_dir.iterdir():
                    if not rep_dir.is_dir():
                        continue

                    # parse simulation
                    sim_result = parse_simulation(rep_dir)
                    results.append(sim_result)

    return results
