import pandas as pd
import numpy as np
from Bio.PDB import MMCIFParser, NeighborSearch
from Bio.PDB.vectors import Vector
import warnings
warnings.filterwarnings('ignore')

def analyze_chain_interactions(cif_file_path, chain_a='A', chain_b='B'):
    """
    Analyze interactions between two chains in a CIF structure file.
    
    Parameters:
    -----------
    cif_file_path : str
        Path to the .cif file
    chain_a : str
        Chain ID for first chain (default: 'A')
    chain_b : str
        Chain ID for second chain (default: 'B')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing interaction information between chains
    """
    
    # Define residue types for different interactions
    CHARGED_POSITIVE = {'ARG', 'LYS', 'HIS'}
    CHARGED_NEGATIVE = {'ASP', 'GLU'}
    POLAR_RESIDUES = {'SER', 'THR', 'ASN', 'GLN', 'TYR', 'ASP', 'GLU', 
                      'ARG', 'LYS', 'HIS', 'TRP'}
    HYDROPHOBIC_RESIDUES = {'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'MET', 'PRO'}
    
    # Distance cutoffs (in Angstroms)
    ELECTROSTATIC_CUTOFF = 5.0
    HBOND_CUTOFF = 3.5
    HYDROPHOBIC_CUTOFF = 4.5
    
    try:
        # Parse the CIF file
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('structure', cif_file_path)
        
        # Get the first model (assuming single model structure)
        model = structure[0]
        
        # Extract chains
        if chain_a not in model or chain_b not in model:
            raise ValueError(f"Chains {chain_a} or {chain_b} not found in structure")
        
        chain_A = model[chain_a]
        chain_B = model[chain_b]
        
        # Get all atoms from both chains for distance calculations
        atoms_A = [atom for residue in chain_A for atom in residue if atom.element != 'H']
        atoms_B = [atom for residue in chain_B for atom in residue if atom.element != 'H']
        
        # Create neighbor search object for efficient distance calculations
        all_atoms = atoms_A + atoms_B
        neighbor_search = NeighborSearch(all_atoms)
        
        # Store interaction data
        interactions = []
        
        # Iterate through all residue pairs between chains
        for res_A in chain_A:
            if not res_A.get_id()[0] == ' ':  # Skip heteroatoms
                continue
                
            for res_B in chain_B:
                if not res_B.get_id()[0] == ' ':  # Skip heteroatoms
                    continue
                
                # Get residue information
                res_A_name = res_A.get_resname()
                res_A_id = res_A.get_id()[1]
                res_B_name = res_B.get_resname()
                res_B_id = res_B.get_id()[1]
                
                # Calculate minimum distance between residues
                min_distance = float('inf')
                closest_atoms = (None, None)
                
                for atom_A in res_A:
                    if atom_A.element == 'H':  # Skip hydrogens
                        continue
                    for atom_B in res_B:
                        if atom_B.element == 'H':  # Skip hydrogens
                            continue
                        distance = atom_A - atom_B
                        if distance < min_distance:
                            min_distance = distance
                            closest_atoms = (atom_A.get_name(), atom_B.get_name())
                
                # Classify interactions based on distance and residue types
                interaction_types = []
                
                # Check for electrostatic interactions
                if (((res_A_name in CHARGED_POSITIVE and res_B_name in CHARGED_NEGATIVE) or
                     (res_A_name in CHARGED_NEGATIVE and res_B_name in CHARGED_POSITIVE)) and
                    min_distance <= ELECTROSTATIC_CUTOFF):
                    interaction_types.append('Electrostatic')
                
                # Check for hydrogen bonding
                if (res_A_name in POLAR_RESIDUES and res_B_name in POLAR_RESIDUES and
                    min_distance <= HBOND_CUTOFF):
                    interaction_types.append('Hydrogen_Bond')
                
                # Check for hydrophobic interactions
                if (res_A_name in HYDROPHOBIC_RESIDUES and res_B_name in HYDROPHOBIC_RESIDUES and
                    min_distance <= HYDROPHOBIC_CUTOFF):
                    interaction_types.append('Hydrophobic')
                
                # Only include residue pairs with interactions within reasonable distance
                if interaction_types or min_distance <= 6.0:  # 6A cutoff for any potential interaction
                    interactions.append({
                        'Chain_A_Residue': res_A_name,
                        'Chain_A_Position': res_A_id,
                        'Chain_B_Residue': res_B_name,
                        'Chain_B_Position': res_B_id,
                        'Min_Distance_A': min_distance,
                        'Closest_Atoms': f"{closest_atoms[0]}-{closest_atoms[1]}",
                        'Interaction_Types': ';'.join(interaction_types) if interaction_types else 'None',
                        'Has_Electrostatic': 'Electrostatic' in interaction_types,
                        'Has_Hydrogen_Bond': 'Hydrogen_Bond' in interaction_types,
                        'Has_Hydrophobic': 'Hydrophobic' in interaction_types,
                        'Chain_A_Type': classify_residue_type(res_A_name),
                        'Chain_B_Type': classify_residue_type(res_B_name)
                    })
        
        # Create DataFrame
        df = pd.DataFrame(interactions)
        
        if df.empty:
            print("No interactions found between the specified chains.")
            return df
        
        # Sort by distance
        df = df.sort_values('Min_Distance_A').reset_index(drop=True)
        
        # Add summary statistics
        print("\nInteraction Summary:")
        print(f"Total residue pairs analyzed: {len(df)}")
        print(f"Pairs with interactions: {len(df[df['Interaction_Types'] != 'None'])}")
        print(f"Electrostatic interactions: {df['Has_Electrostatic'].sum()}")
        print(f"Hydrogen bond interactions: {df['Has_Hydrogen_Bond'].sum()}")
        print(f"Hydrophobic interactions: {df['Has_Hydrophobic'].sum()}")
        
        return df
        
    except Exception as e:
        print(f"Error processing CIF file: {str(e)}")
        return pd.DataFrame()

def classify_residue_type(residue_name):
    """
    Classify residue type for easier analysis.
    
    Parameters:
    -----------
    residue_name : str
        Three-letter residue code
    
    Returns:
    --------
    str
        Residue classification
    """
    CHARGED_POSITIVE = {'ARG', 'LYS', 'HIS'}
    CHARGED_NEGATIVE = {'ASP', 'GLU'}
    POLAR_RESIDUES = {'SER', 'THR', 'ASN', 'GLN', 'TYR'}
    HYDROPHOBIC_RESIDUES = {'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'MET', 'PRO'}
    AROMATIC_RESIDUES = {'PHE', 'TYR', 'TRP', 'HIS'}
    
    if residue_name in CHARGED_POSITIVE:
        return 'Positive'
    elif residue_name in CHARGED_NEGATIVE:
        return 'Negative'
    elif residue_name in AROMATIC_RESIDUES:
        return 'Aromatic'
    elif residue_name in POLAR_RESIDUES:
        return 'Polar'
    elif residue_name in HYDROPHOBIC_RESIDUES:
        return 'Hydrophobic'
    elif residue_name == 'GLY':
        return 'Flexible'
    elif residue_name == 'CYS':
        return 'Sulfur'
    else:
        return 'Other'

def filter_interactions(
    df: pd.DataFrame, 
    interaction_type=None, 
    max_distance=None
) -> pd.DataFrame:
    """
    Filter the interaction dataframe based on criteria.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame from analyze_chain_interactions
    interaction_type : str
        Filter by interaction type ('Electrostatic', 'Hydrogen_Bond', 'Hydrophobic')
    max_distance : float
        Maximum distance cutoff in Angstroms
    
    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    if max_distance is not None:
        filtered_df = filtered_df[filtered_df['Min_Distance_A'] <= max_distance]
    
    if interaction_type is not None:
        if interaction_type == 'Electrostatic':
            filtered_df = filtered_df[filtered_df['Has_Electrostatic']]
        elif interaction_type == 'Hydrogen_Bond':
            filtered_df = filtered_df[filtered_df['Has_Hydrogen_Bond']]
        elif interaction_type == 'Hydrophobic':
            filtered_df = filtered_df[filtered_df['Has_Hydrophobic']]
    
    return filtered_df









def analyze_protein_ligand_interactions(cif_file_path, protein_chain='A', ligand_name=None, exclude_waters=True):
    """
    Analyze interactions between a protein chain and small molecule ligand(s).
    
    Parameters:
    -----------
    cif_file_path : str
        Path to the .cif file
    protein_chain : str
        Chain ID for protein chain (default: 'A')
    ligand_name : str or None
        Specific ligand residue name (e.g., 'ATP', 'HEM'). If None, analyzes all heteroatoms
    exclude_waters : bool
        Whether to exclude water molecules from analysis (default: True)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing interaction information between protein and ligand
    """
    
    # Define atom types for interaction classification
    HYDROGEN_BOND_DONORS = {'N', 'O', 'S'}  # Simplified - atoms that can donate H-bonds
    HYDROGEN_BOND_ACCEPTORS = {'N', 'O', 'S', 'F'}  # Atoms that can accept H-bonds
    HYDROPHOBIC_ATOMS = {'C'}  # Carbon atoms for hydrophobic interactions
    CHARGED_ATOMS = {'N', 'O', 'S', 'P'}  # Atoms that can carry charge
    
    # Define charged/polar residues for protein
    CHARGED_POSITIVE_RES = {'ARG', 'LYS', 'HIS'}
    CHARGED_NEGATIVE_RES = {'ASP', 'GLU'}
    POLAR_RESIDUES = {'SER', 'THR', 'ASN', 'GLN', 'TYR', 'ASP', 'GLU', 
                      'ARG', 'LYS', 'HIS', 'TRP', 'CYS'}
    HYDROPHOBIC_RESIDUES = {'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'MET', 'PRO'}
    
    # Distance cutoffs (in Angstroms)
    ELECTROSTATIC_CUTOFF = 5.0
    HBOND_CUTOFF = 3.5
    HYDROPHOBIC_CUTOFF = 4.5
    VDW_CUTOFF = 6.0  # General interaction cutoff
    
    # Common non-ligand heteroatoms to potentially exclude
    COMMON_NON_LIGANDS = {'HOH', 'WAT', 'NA', 'CL', 'MG', 'CA', 'K', 'ZN', 'FE', 'MN', 'SO4', 'PO4'}
    
    try:
        # Parse the CIF file
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('structure', cif_file_path)
        
        # Get the first model
        model = structure[0]
        
        # Extract protein chain
        if protein_chain not in model:
            raise ValueError(f"Protein chain {protein_chain} not found in structure")
        
        protein = model[protein_chain]
        
        # Find ligand residues (heteroatoms)
        ligand_residues = []
        ligand_info = []
        
        for chain in model:
            for residue in chain:
                res_id = residue.get_id()
                res_name = residue.get_resname()
                
                # Check if it's a heteroatom (not standard amino acid)
                if res_id[0] != ' ':  # Heteroatom
                    # Apply filters
                    if exclude_waters and res_name in {'HOH', 'WAT'}:
                        continue
                    
                    if ligand_name is not None and res_name != ligand_name:
                        continue
                    
                    # Skip common ions/cofactors if no specific ligand requested
                    if ligand_name is None and res_name in COMMON_NON_LIGANDS:
                        continue
                    
                    ligand_residues.append(residue)
                    ligand_info.append({
                        'ligand_name': res_name,
                        'ligand_chain': chain.get_id(),
                        'ligand_position': res_id[1],
                        'ligand_icode': res_id[2] if res_id[2] != ' ' else ''
                    })
        
        if not ligand_residues:
            print("No suitable ligand residues found in the structure.")
            print("Available heteroatoms:", [res.get_resname() for chain in model for res in chain if res.get_id()[0] != ' '])
            return pd.DataFrame()
        
        print(f"Found {len(ligand_residues)} ligand residue(s): {[res.get_resname() for res in ligand_residues]}")
        
        # Store interaction data
        interactions = []
        
        # Analyze interactions between protein residues and ligands
        for i, ligand_res in enumerate(ligand_residues):
            ligand_data = ligand_info[i]
            
            for protein_res in protein:
                if protein_res.get_id()[0] != ' ':  # Skip heteroatoms in protein chain
                    continue
                
                # Get residue information
                protein_res_name = protein_res.get_resname()
                protein_res_id = protein_res.get_id()[1]
                protein_res_icode = protein_res.get_id()[2] if protein_res.get_id()[2] != ' ' else ''
                
                # Calculate minimum distance between protein residue and ligand
                min_distance = float('inf')
                closest_atoms = (None, None)
                all_distances = []
                
                for protein_atom in protein_res:
                    if protein_atom.element == 'H':  # Skip hydrogens
                        continue
                    for ligand_atom in ligand_res:
                        if ligand_atom.element == 'H':  # Skip hydrogens
                            continue
                        distance = protein_atom - ligand_atom
                        all_distances.append(distance)
                        if distance < min_distance:
                            min_distance = distance
                            closest_atoms = (protein_atom.get_name(), ligand_atom.get_name())
                
                # Only analyze interactions within reasonable distance
                if min_distance > VDW_CUTOFF:
                    continue
                
                # Classify interactions
                interaction_types = []
                
                # Check for electrostatic interactions
                electrostatic_score = calculate_electrostatic_potential(protein_res, ligand_res, min_distance)
                if electrostatic_score > 0 and min_distance <= ELECTROSTATIC_CUTOFF:
                    interaction_types.append('Electrostatic')
                
                # Check for hydrogen bonding potential
                hbond_score = calculate_hbond_potential(protein_res, ligand_res, min_distance)
                if hbond_score > 0 and min_distance <= HBOND_CUTOFF:
                    interaction_types.append('Hydrogen_Bond')
                
                # Check for hydrophobic interactions
                hydrophobic_score = calculate_hydrophobic_potential(protein_res, ligand_res, min_distance)
                if hydrophobic_score > 0 and min_distance <= HYDROPHOBIC_CUTOFF:
                    interaction_types.append('Hydrophobic')
                
                # Van der Waals interactions (close contacts)
                if min_distance <= 4.0:
                    interaction_types.append('Van_der_Waals')
                
                # Calculate interaction strength score
                interaction_strength = calculate_interaction_strength(min_distance, interaction_types)
                
                interactions.append({
                    'Protein_Residue': protein_res_name,
                    'Protein_Position': protein_res_id,
                    'Protein_ICode': protein_res_icode,
                    'Ligand_Name': ligand_data['ligand_name'],
                    'Ligand_Chain': ligand_data['ligand_chain'],
                    'Ligand_Position': ligand_data['ligand_position'],
                    'Ligand_ICode': ligand_data['ligand_icode'],
                    'Min_Distance_A': round(min_distance, 2),
                    'Closest_Atoms': f"{closest_atoms[0]}-{closest_atoms[1]}",
                    'Interaction_Types': ';'.join(interaction_types) if interaction_types else 'Van_der_Waals',
                    'Interaction_Strength': round(interaction_strength, 2),
                    'Has_Electrostatic': 'Electrostatic' in interaction_types,
                    'Has_Hydrogen_Bond': 'Hydrogen_Bond' in interaction_types,
                    'Has_Hydrophobic': 'Hydrophobic' in interaction_types,
                    'Has_VdW': 'Van_der_Waals' in interaction_types,
                    'Protein_Type': classify_residue_type(protein_res_name),
                    'Electrostatic_Score': round(electrostatic_score, 2),
                    'HBond_Score': round(hbond_score, 2),
                    'Hydrophobic_Score': round(hydrophobic_score, 2),
                    'Avg_Distance': round(np.mean(all_distances), 2),
                    'Contact_Count': len([d for d in all_distances if d <= 4.5])
                })
        
        # Create DataFrame
        df = pd.DataFrame(interactions)
        
        if df.empty:
            print("No interactions found between protein and ligand.")
            return df
        
        # Sort by interaction strength and distance
        df = df.sort_values(['Interaction_Strength', 'Min_Distance_A'], ascending=[False, True]).reset_index(drop=True)
        
        # Print summary
        print_interaction_summary(df)
        
        return df
        
    except Exception as e:
        print(f"Error processing CIF file: {str(e)}")
        return pd.DataFrame()

def calculate_electrostatic_potential(protein_res, ligand_res, distance):
    """Calculate electrostatic interaction potential."""
    CHARGED_POSITIVE_RES = {'ARG', 'LYS', 'HIS'}
    CHARGED_NEGATIVE_RES = {'ASP', 'GLU'}
    CHARGED_ATOMS = {'N', 'O', 'S', 'P'}
    
    protein_name = protein_res.get_resname()
    
    # Count charged atoms in ligand
    ligand_charged_atoms = sum(1 for atom in ligand_res if atom.element in CHARGED_ATOMS)
    
    if ligand_charged_atoms == 0:
        return 0
    
    # Score based on protein residue charge and distance
    if protein_name in CHARGED_POSITIVE_RES or protein_name in CHARGED_NEGATIVE_RES:
        return max(0, (6.0 - distance) / 6.0) * ligand_charged_atoms
    
    return 0

def calculate_hbond_potential(protein_res, ligand_res, distance):
    """Calculate hydrogen bonding potential."""
    POLAR_RESIDUES = {'SER', 'THR', 'ASN', 'GLN', 'TYR', 'ASP', 'GLU', 
                      'ARG', 'LYS', 'HIS', 'TRP', 'CYS'}
    HBOND_ATOMS = {'N', 'O', 'S'}
    
    protein_name = protein_res.get_resname()
    
    if protein_name not in POLAR_RESIDUES:
        return 0
    
    # Count potential H-bond atoms in ligand
    ligand_hbond_atoms = sum(1 for atom in ligand_res if atom.element in HBOND_ATOMS)
    
    if ligand_hbond_atoms == 0:
        return 0
    
    # Score based on distance and number of potential H-bond atoms
    return max(0, (4.0 - distance) / 4.0) * ligand_hbond_atoms

def calculate_hydrophobic_potential(protein_res, ligand_res, distance):
    """Calculate hydrophobic interaction potential."""
    HYDROPHOBIC_RESIDUES = {'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'MET', 'PRO'}
    
    protein_name = protein_res.get_resname()
    
    if protein_name not in HYDROPHOBIC_RESIDUES:
        return 0
    
    # Count carbon atoms in ligand (proxy for hydrophobicity)
    ligand_carbon_atoms = sum(1 for atom in ligand_res if atom.element == 'C')
    
    if ligand_carbon_atoms == 0:
        return 0
    
    # Score based on distance and hydrophobic surface area
    return max(0, (5.0 - distance) / 5.0) * min(ligand_carbon_atoms / 5.0, 1.0)

def calculate_interaction_strength(distance, interaction_types):
    """Calculate overall interaction strength score."""
    if not interaction_types:
        return 0
    
    # Base score inversely related to distance
    base_score = max(0, (6.0 - distance) / 6.0)
    
    # Weight different interaction types
    weights = {
        'Electrostatic': 3.0,
        'Hydrogen_Bond': 2.0,
        'Hydrophobic': 1.5,
        'Van_der_Waals': 1.0
    }
    
    total_weight = sum(weights.get(itype, 1.0) for itype in interaction_types)
    return base_score * total_weight

# def classify_residue_type(residue_name):
#     """Classify protein residue type."""
#     CHARGED_POSITIVE = {'ARG', 'LYS', 'HIS'}
#     CHARGED_NEGATIVE = {'ASP', 'GLU'}
#     POLAR_RESIDUES = {'SER', 'THR', 'ASN', 'GLN', 'TYR'}
#     HYDROPHOBIC_RESIDUES = {'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'MET', 'PRO'}
#     AROMATIC_RESIDUES = {'PHE', 'TYR', 'TRP', 'HIS'}
    
#     if residue_name in CHARGED_POSITIVE:
#         return 'Positive'
#     elif residue_name in CHARGED_NEGATIVE:
#         return 'Negative'
#     elif residue_name in AROMATIC_RESIDUES:
#         return 'Aromatic'
#     elif residue_name in POLAR_RESIDUES:
#         return 'Polar'
#     elif residue_name in HYDROPHOBIC_RESIDUES:
#         return 'Hydrophobic'
#     elif residue_name == 'GLY':
#         return 'Flexible'
#     elif residue_name == 'CYS':
#         return 'Sulfur'
#     else:
#         return 'Other'

def print_interaction_summary(df):
    """Print summary statistics for the interactions."""
    print("\nProtein-Ligand Interaction Summary:")
    print(f"Total interacting residues: {len(df)}")
    print(f"Electrostatic interactions: {df['Has_Electrostatic'].sum()}")
    print(f"Hydrogen bond interactions: {df['Has_Hydrogen_Bond'].sum()}")
    print(f"Hydrophobic interactions: {df['Has_Hydrophobic'].sum()}")
    print(f"Van der Waals contacts: {df['Has_VdW'].sum()}")
    print(f"Average interaction distance: {df['Min_Distance_A'].mean():.2f} Å")
    print(f"Closest interaction: {df['Min_Distance_A'].min():.2f} Å")
    
    # Top interacting residues
    top_interactions = df.nlargest(5, 'Interaction_Strength')
    print("\nTop 5 strongest interactions:")
    for _, row in top_interactions.iterrows():
        print(f"{row['Protein_Residue']}{row['Protein_Position']} - {row['Ligand_Name']}: "
              f"{row['Min_Distance_A']:.2f}Å ({row['Interaction_Types']})")

def filter_protein_ligand_interactions(df, interaction_type=None, max_distance=None, min_strength=None):
    """Filter the protein-ligand interaction dataframe."""
    filtered_df = df.copy()
    
    if max_distance is not None:
        filtered_df = filtered_df[filtered_df['Min_Distance_A'] <= max_distance]
    
    if min_strength is not None:
        filtered_df = filtered_df[filtered_df['Interaction_Strength'] >= min_strength]
    
    if interaction_type is not None:
        if interaction_type == 'Electrostatic':
            filtered_df = filtered_df[filtered_df['Has_Electrostatic']]
        elif interaction_type == 'Hydrogen_Bond':
            filtered_df = filtered_df[filtered_df['Has_Hydrogen_Bond']]
        elif interaction_type == 'Hydrophobic':
            filtered_df = filtered_df[filtered_df['Has_Hydrophobic']]
        elif interaction_type == 'Van_der_Waals':
            filtered_df = filtered_df[filtered_df['Has_VdW']]
    
    return filtered_df