"""
Protein Network Analyzer
========================
A comprehensive tool for analyzing protein-protein interaction networks,
determining degrees of separation, and identifying functional relationships.

This module integrates with multiple biological databases to:
- Build protein interaction networks
- Calculate shortest paths between proteins
- Identify activators and inhibitors
- Visualize protein interaction networks

Author: Gabriel Navarro
Date: October 2025



# Example usage and test function
if __name__ == "__main__":
    # Example: Analyze connections to TP53 (tumor suppressor protein)
    example_proteins = [
        'MDM2',      # Known direct inhibitor of TP53
        'ATM',       # Activator of TP53 in DNA damage response
        'BRCA1',     # Works with TP53 in DNA repair
        'EGFR',      # Receptor tyrosine kinase
        'MYC',       # Oncogene
        'BCL2',      # Anti-apoptotic protein
        'PTEN',      # Tumor suppressor
        'AKT1'       # Kinase in PI3K pathway
    ]
    
    results = analyze_protein_network(
        target_protein='TP53',
        protein_list=example_proteins,
        species=9606,  # Human
        interaction_threshold=700,  # High confidence interactions
        max_depth=3
    )
    
    print("\nDetailed Results:")
    print(results.to_string(index=False))
"""

import requests
import networkx as nx
from typing import List, Dict, Tuple, Optional, Set
import pandas as pd
import time
from collections import defaultdict
import json


class ProteinNetworkAnalyzer:
    """
    A class for analyzing protein-protein interaction networks and determining
    functional relationships between proteins.
    
    Attributes:
        target_protein (str): The target protein of interest
        species (int): NCBI taxonomy ID (default: 9606 for Homo sapiens)
        interaction_threshold (int): Minimum confidence score (0-1000) for interactions
        graph (nx.Graph): NetworkX graph representing the protein interaction network
        activation_inhibition_data (dict): Stores functional relationship data
    """
    
    def __init__(self, target_protein: str, species: int = 9606, 
                 interaction_threshold: int = 400):
        """
        Initialize the ProteinNetworkAnalyzer.
        
        Parameters:
            target_protein (str): Target protein identifier (gene name or UniProt ID)
            species (int): NCBI taxonomy ID (default: 9606 for Homo sapiens)
            interaction_threshold (int): Minimum STRING confidence score (0-1000)
        """
        self.target_protein = target_protein.upper()
        self.species = species
        self.interaction_threshold = interaction_threshold
        self.graph = nx.Graph()
        self.activation_inhibition_data = defaultdict(dict)
        self.string_api_url = "https://string-db.org/api"
        self.protein_mapping = {}  # Maps user identifiers to STRING IDs
        
    def get_string_id(self, protein_name: str) -> Optional[str]:
        """
        Map protein name/identifier to STRING database ID.
        
        Parameters:
            protein_name (str): Protein identifier (gene name or UniProt ID)
            
        Returns:
            str: STRING protein ID or None if not found
        """
        url = f"{self.string_api_url}/json/get_string_ids"
        params = {
            'identifiers': protein_name,
            'species': self.species,
            'limit': 1
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data and len(data) > 0:
                string_id = data[0]['stringId']
                self.protein_mapping[protein_name.upper()] = string_id
                return string_id
            else:
                print(f"Warning: No STRING ID found for {protein_name}")
                return None
                
        except Exception as e:
            print(f"Error mapping {protein_name} to STRING ID: {e}")
            return None
    
    def fetch_interactions(self, protein: str, depth: int = 2) -> Dict:
        """
        Fetch protein-protein interactions from STRING database.
        
        Parameters:
            protein (str): Protein identifier
            depth (int): Network depth to retrieve (1 or 2)
            
        Returns:
            dict: Interaction data including edges and functional annotations
        """
        string_id = self.get_string_id(protein)
        if not string_id:
            return {}
        
        url = f"{self.string_api_url}/json/interaction_partners"
        params = {
            'identifiers': string_id,
            'species': self.species,
            'required_score': self.interaction_threshold,
            'limit': 500  # Reasonable limit to avoid overwhelming the API
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            time.sleep(0.5)  # Rate limiting - be respectful to the API
            return response.json()
            
        except Exception as e:
            print(f"Error fetching interactions for {protein}: {e}")
            return []       # type: ignore
    
    def build_network(self, protein_list: List[str], max_depth: int = 3) -> None:
        """
        Build a protein interaction network starting from target protein.
        
        Parameters:
            protein_list (List[str]): List of proteins to analyze
            max_depth (int): Maximum network depth to explore
        """
        # Start with target protein
        all_proteins = set([self.target_protein] + [p.upper() for p in protein_list])
        visited = set()
        current_depth = 0
        to_explore = {self.target_protein}
        
        print(f"Building network for {self.target_protein}...")
        print(f"Analyzing connections to {len(protein_list)} proteins...")
        
        while to_explore and current_depth < max_depth:
            next_level = set()
            
            for protein in to_explore:
                if protein in visited:
                    continue
                    
                visited.add(protein)
                interactions = self.fetch_interactions(protein)
                
                for interaction in interactions:
                    # Extract partner protein
                    partner_string_id = interaction.get('stringId_B', '')
                    partner_name = interaction.get('preferredName_B', '').upper()
                    score = interaction.get('score', 0)
                    
                    # Add to mapping
                    if partner_name:
                        self.protein_mapping[partner_name] = partner_string_id
                    
                    # Add edge to graph
                    if partner_name and score >= self.interaction_threshold / 1000.0:
                        self.graph.add_edge(
                            protein, 
                            partner_name, 
                            weight=score,
                            confidence=score
                        )
                        
                        # Store functional annotations if available
                        if 'mode' in interaction:
                            mode = interaction['mode']
                            if mode in ['activation', 'inhibition']:
                                self.activation_inhibition_data[protein][partner_name] = mode
                        
                        # Add to next exploration level if it's in our target list
                        if partner_name in all_proteins and partner_name not in visited:
                            next_level.add(partner_name)
            
            to_explore = next_level
            current_depth += 1
            print(f"  Depth {current_depth}: Explored {len(visited)} proteins, "
                  f"Network size: {self.graph.number_of_edges()} edges")
    
    def get_functional_relationship_with_evidence(self, source: str, target: str) -> Dict:
        """
        Get detailed evidence for functional relationship from all databases.
        This provides transparency about which databases support the relationship.
        
        Parameters:
            source (str): Source protein
            target (str): Target protein
            
        Returns:
            dict: Dictionary containing:
                - relationship: 'activator', 'inhibitor', or None
                - confidence: score based on number of supporting databases
                - evidence: dict with results from each database
                - sources: list of databases that found the relationship
        """
        evidence = {
            'STRING': self._check_string_relationship(source, target),
            'SIGNOR': self._check_signor_relationship(source, target),
            'OmniPath': self._check_omnipath_relationship(source, target),
            'Reactome': self._check_reactome_relationship(source, target)
        }
        
        # Count supporting evidence
        activator_sources = [db for db, rel in evidence.items() if rel == 'activator']
        inhibitor_sources = [db for db, rel in evidence.items() if rel == 'inhibitor']
        
        # Determine relationship with weighted consensus
        weights = {'STRING': 1.0, 'SIGNOR': 2.0, 'OmniPath': 1.5, 'Reactome': 1.0}
        activator_score = sum(weights[db] for db in activator_sources)
        inhibitor_score = sum(weights[db] for db in inhibitor_sources)
        
        if activator_score > inhibitor_score:
            relationship = 'activator'
            confidence = activator_score
            supporting_sources = activator_sources
        elif inhibitor_score > activator_score:
            relationship = 'inhibitor'
            confidence = inhibitor_score
            supporting_sources = inhibitor_sources
        else:
            relationship = None
            confidence = 0
            supporting_sources = []
        
        return {
            'relationship': relationship,
            'confidence': confidence,
            'evidence': evidence,
            'supporting_sources': supporting_sources,
            'conflicting': len(activator_sources) > 0 and len(inhibitor_sources) > 0
        }
    
    def get_functional_relationship(self, source: str, target: str) -> Optional[str]:
        """
        Determine if source protein activates or inhibits target protein.
        Uses multiple biological databases for comprehensive relationship detection:
        - STRING database (protein-protein interactions with actions)
        - SIGNOR (SIGnaling Network Open Resource - causal interactions)
        - OmniPath (aggregated signaling database)
        - Reactome (curated pathway database)
        
        Implements a consensus mechanism: if multiple sources agree, returns that
        relationship. If sources conflict, prioritizes high-confidence databases.
        
        Parameters:
            source (str): Source protein
            target (str): Target protein
            
        Returns:
            str: 'activator', 'inhibitor', or None
        """
        # Check cached data first
        if source in self.activation_inhibition_data:
            if target in self.activation_inhibition_data[source]:
                return self.activation_inhibition_data[source][target]
        
        # Collect evidence from multiple sources
        evidence = {
            'activator': 0,
            'inhibitor': 0,
            'sources': []
        }
        
        # Source 1: STRING Database
        string_result = self._check_string_relationship(source, target)
        if string_result:
            evidence[string_result] += 1
            evidence['sources'].append(f"STRING:{string_result}")
        
        # Source 2: SIGNOR Database (high quality causal interactions)
        signor_result = self._check_signor_relationship(source, target)
        if signor_result:
            evidence[signor_result] += 2  # Weight SIGNOR higher (specialized in causality)
            evidence['sources'].append(f"SIGNOR:{signor_result}")
        
        # Source 3: OmniPath (aggregates multiple signaling databases)
        omnipath_result = self._check_omnipath_relationship(source, target)
        if omnipath_result:
            evidence[omnipath_result] += 1.5  # Weight OmniPath moderately high
            evidence['sources'].append(f"OmniPath:{omnipath_result}")
        
        # Source 4: Reactome Pathways
        reactome_result = self._check_reactome_relationship(source, target)
        if reactome_result:
            evidence[reactome_result] += 1
            evidence['sources'].append(f"Reactome:{reactome_result}")
        
        # Determine consensus
        if evidence['activator'] > 0 and evidence['inhibitor'] == 0:
            result = 'activator'
        elif evidence['inhibitor'] > 0 and evidence['activator'] == 0:
            result = 'inhibitor'
        elif evidence['activator'] > evidence['inhibitor']:
            result = 'activator'
        elif evidence['inhibitor'] > evidence['activator']:
            result = 'inhibitor'
        else:
            result = None
        
        # Cache the result with source information
        if result:
            if source not in self.activation_inhibition_data:
                self.activation_inhibition_data[source] = {}
            self.activation_inhibition_data[source][target] = result
        
        return result
    
    def _check_string_relationship(self, source: str, target: str) -> Optional[str]:
        """
        Check STRING database for functional relationship.
        
        Parameters:
            source (str): Source protein
            target (str): Target protein
            
        Returns:
            str: 'activator', 'inhibitor', or None
        """
        source_id = self.protein_mapping.get(source.upper())
        target_id = self.protein_mapping.get(target.upper())
        
        if not source_id or not target_id:
            return None
        
        url = f"{self.string_api_url}/json/network"
        params = {
            'identifiers': f"{source_id}%0d{target_id}",
            'species': self.species,
            'required_score': self.interaction_threshold
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            time.sleep(0.3)  # Rate limiting
            
            for interaction in data:
                # Check for action/mode information
                if 'action' in interaction:
                    action = str(interaction['action']).lower()
                    if any(term in action for term in ['activation', 'expression', 'stimulation']):
                        return 'activator'
                    elif any(term in action for term in ['inhibition', 'repression', 'suppression']):
                        return 'inhibitor'
                
                if 'mode' in interaction:
                    mode = str(interaction['mode']).lower()
                    if mode in ['activation', 'expression']:
                        return 'activator'
                    elif mode in ['inhibition', 'repression']:
                        return 'inhibitor'
                        
        except Exception as e:
            pass  # Silently fail and try other sources
        
        return None
    
    def _check_signor_relationship(self, source: str, target: str) -> Optional[str]:
        """
        Check SIGNOR database for causal relationships.
        SIGNOR specializes in signaling information with high-quality manual curation.
        
        Parameters:
            source (str): Source protein
            target (str): Target protein
            
        Returns:
            str: 'activator', 'inhibitor', or None
        """
        # SIGNOR 3.0 API endpoint
        url = "https://signor.uniroma2.it/getData.php"
        
        try:
            # Query for relationship
            params = {
                'format': 'json',
                'entityA': source.upper(),
                'entityB': target.upper(),
                'organism': self.species
            }
            
            response = requests.get(url, params=params, timeout=10)
            time.sleep(0.5)  # Rate limiting
            
            if response.status_code == 200:
                data = response.json()
                
                if isinstance(data, list) and len(data) > 0:
                    for interaction in data:
                        effect = str(interaction.get('EFFECT', '')).lower()
                        mechanism = str(interaction.get('MECHANISM', '')).lower()
                        
                        # Check for activation signals
                        activation_terms = [
                            'up-regulates', 'activates', 'stimulates', 
                            'phosphorylates', 'induces', 'promotes'
                        ]
                        inhibition_terms = [
                            'down-regulates', 'inhibits', 'suppresses', 
                            'represses', 'blocks', 'dephosphorylates'
                        ]
                        
                        if any(term in effect or term in mechanism for term in activation_terms):
                            return 'activator'
                        elif any(term in effect or term in mechanism for term in inhibition_terms):
                            return 'inhibitor'
                            
        except Exception as e:
            pass  # Silently fail and try other sources
        
        return None
    
    def _check_omnipath_relationship(self, source: str, target: str) -> Optional[str]:
        """
        Check OmniPath database for functional relationships.
        OmniPath aggregates data from multiple signaling databases.
        
        Parameters:
            source (str): Source protein
            target (str): Target protein
            
        Returns:
            str: 'activator', 'inhibitor', or None
        """
        # OmniPath interactions endpoint
        url = "https://omnipathdb.org/interactions"
        
        try:
            params = {
                'sources': source.upper(),
                'targets': target.upper(),
                'organisms': self.species,
                'fields': 'sources,references,curation_effort,type,is_stimulation,is_inhibition'
            }
            
            response = requests.get(url, params=params, timeout=10)
            time.sleep(0.4)  # Rate limiting
            
            if response.status_code == 200:
                # OmniPath returns TSV format
                lines = response.text.strip().split('\n')
                
                if len(lines) > 1:  # Has data beyond header
                    # Parse the response
                    for line in lines[1:]:  # Skip header
                        fields = line.split('\t')
                        if len(fields) >= 8:
                            is_stimulation = fields[7] if len(fields) > 7 else ''
                            is_inhibition = fields[8] if len(fields) > 8 else ''
                            
                            if is_stimulation == '1' or is_stimulation.lower() == 'true':
                                return 'activator'
                            elif is_inhibition == '1' or is_inhibition.lower() == 'true':
                                return 'inhibitor'
                            
        except Exception as e:
            pass  # Silently fail and try other sources
        
        return None
    
    def _check_reactome_relationship(self, source: str, target: str) -> Optional[str]:
        """
        Check Reactome pathway database for functional relationships.
        Reactome provides highly curated pathway information.
        
        Parameters:
            source (str): Source protein
            target (str): Target protein
            
        Returns:
            str: 'activator', 'inhibitor', or None
        """
        # Reactome Content Service API
        base_url = "https://reactome.org/ContentService"
        
        try:
            # First, get entity ID for source protein
            search_url = f"{base_url}/data/query/{source.upper()}/enhanced"
            response = requests.get(search_url, timeout=10)
            time.sleep(0.4)
            
            if response.status_code == 200:
                source_data = response.json()
                
                if source_data and len(source_data) > 0:
                    source_id = source_data[0].get('stId') or source_data[0].get('dbId')
                    
                    if source_id:
                        # Get pathways and interactions
                        pathway_url = f"{base_url}/data/pathways/low/entity/{source_id}"
                        pathway_response = requests.get(pathway_url, timeout=10)
                        time.sleep(0.4)
                        
                        if pathway_response.status_code == 200:
                            pathways = pathway_response.json()
                            
                            # Check if target is in related pathways
                            # This is a simplified check - Reactome API is complex
                            for pathway in pathways:
                                pathway_name = pathway.get('displayName', '').lower()
                                
                                # Heuristic: if pathway name contains both proteins and 
                                # certain keywords, infer relationship
                                if target.lower() in pathway_name:
                                    if any(term in pathway_name for term in ['activation', 'positive', 'stimulation']):
                                        return 'activator'
                                    elif any(term in pathway_name for term in ['inhibition', 'negative', 'repression']):
                                        return 'inhibitor'
                        
        except Exception as e:
            pass  # Silently fail - Reactome can be complex
        
        return None
    
    def analyze_protein_distances(self, protein_list: List[str]) -> pd.DataFrame:
        """
        Calculate shortest path distances from target protein to each protein in list.
        
        Parameters:
            protein_list (List[str]): List of proteins to analyze
            
        Returns:
            pd.DataFrame: Results including distance and functional relationships
        """
        results = []
        
        for protein in protein_list:
            protein_upper = protein.upper()
            result = {
                'protein': protein,
                'distance': None,
                'path': None,
                'functional_relationship': None,
                'in_network': False
            }
            
            # Check if protein is in the network
            if protein_upper in self.graph.nodes():
                result['in_network'] = True
                
                try:
                    # Calculate shortest path
                    if nx.has_path(self.graph, self.target_protein, protein_upper):
                        path = nx.shortest_path(
                            self.graph, 
                            self.target_protein, 
                            protein_upper
                        )
                        distance = len(path) - 1
                        
                        result['distance'] = distance
                        result['path'] = ' → '.join(path)
                        
                        # For proteins within 2 degrees, check functional relationship
                        if distance <= 2:
                            if distance == 1:
                                # Direct interaction
                                func_rel = self.get_functional_relationship(
                                    self.target_protein, 
                                    protein_upper
                                )
                                result['functional_relationship'] = func_rel
                            else:
                                # Indirect interaction through one intermediate
                                intermediate = path[1]
                                func_rel = self.get_functional_relationship(
                                    intermediate, 
                                    protein_upper
                                )
                                if func_rel:
                                    result['functional_relationship'] = (
                                        f"Indirect {func_rel} (via {intermediate})"
                                    )
                    else:
                        result['distance'] = float('inf')
                        result['path'] = 'No path found'
                        
                except nx.NetworkXNoPath:
                    result['distance'] = float('inf')
                    result['path'] = 'No path found'
            else:
                result['distance'] = -1
                result['path'] = 'Protein not found in interaction network'
            
            results.append(result)
        
        # Create DataFrame and sort by distance
        df = pd.DataFrame(results)
        df = df.sort_values(
            'distance', 
            key=lambda x: x.replace(
                [None, float('inf'), 'Not in network'], # type: ignore
                [999, 998, 1000]
            )
        )

        # Clean distance to make all numeric where possible
        df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
        return df
    
    def get_network_statistics(self) -> Dict:
        """
        Calculate network statistics.
        
        Returns:
            dict: Network statistics including centrality measures
        """
        if self.graph.number_of_nodes() == 0:
            return {"error": "Network is empty"}
        
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_connected(self.graph)
        }
        
        # Calculate degree centrality for target protein
        if self.target_protein in self.graph.nodes():
            degree_centrality = nx.degree_centrality(self.graph)
            stats['target_degree_centrality'] = degree_centrality[self.target_protein]
            stats['target_num_connections'] = self.graph.degree(self.target_protein)    # type: ignore
        
        return stats

    def export_results(self, results_df: pd.DataFrame, filename: str = None) -> None:  # type: ignore
        """
        Export results to CSV file.
        
        Parameters:
            results_df (pd.DataFrame): Results dataframe
            filename (str): Output filename (default: auto-generated)
        """
        if filename is None:
            filename = f"{self.target_protein}_network_analysis.csv"
        
        results_df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")


def analyze_protein_network(target_protein: str, 
                           protein_list: List[str],
                           species: int = 9606,
                           interaction_threshold: int = 400,
                           max_depth: int = 3,
                           export_csv: bool = True) -> pd.DataFrame:
    """
    Main function to analyze protein network connections and functional relationships.
    
    This function determines:
    1. How many connections away each protein is from the target protein
    2. Whether proteins within 2 degrees act as activators or inhibitors
    
    Parameters:
        target_protein (str): Target protein identifier (gene name or UniProt ID)
        protein_list (List[str]): List of proteins to analyze
        species (int): NCBI taxonomy ID (default: 9606 for Homo sapiens)
        interaction_threshold (int): Minimum STRING confidence score (0-1000)
        max_depth (int): Maximum network depth to explore (default: 3)
        export_csv (bool): Whether to export results to CSV
        
    Returns:
        pd.DataFrame: Analysis results with columns:
            - protein: Protein identifier
            - distance: Number of connections from target (degrees of separation)
            - path: Shortest path from target to protein
            - functional_relationship: Activator/inhibitor status (if distance ≤ 2)
            - in_network: Whether protein was found in interaction network
    
    Example:
        >>> results = analyze_protein_network(
        ...     target_protein='TP53',
        ...     protein_list=['MDM2', 'ATM', 'BRCA1', 'EGFR', 'MYC'],
        ...     species=9606,
        ...     interaction_threshold=700
        ... )
        >>> print(results)
    """
    print("=" * 70)
    print("PROTEIN NETWORK ANALYZER")
    print("=" * 70)
    print(f"Target protein: {target_protein}")
    print(f"Analyzing {len(protein_list)} proteins")
    print(f"Species: {species} (9606=Human, 10090=Mouse, 10116=Rat)")
    print(f"Interaction threshold: {interaction_threshold}/1000")
    print("=" * 70)
    print()
    
    # Initialize analyzer
    analyzer = ProteinNetworkAnalyzer(
        target_protein=target_protein,
        species=species,
        interaction_threshold=interaction_threshold
    )
    
    # Build the network
    analyzer.build_network(protein_list, max_depth=max_depth)
    
    # Get network statistics
    print("\nNetwork Statistics:")
    stats = analyzer.get_network_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Analyze distances
    print("Analyzing protein distances and relationships...")
    results = analyzer.analyze_protein_distances(protein_list)
    
    # Export if requested
    if export_csv:
        analyzer.export_results(results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    # Count proteins by distance
    close_proteins = results[results['distance'] <= 2]
    if len(close_proteins) > 0:
        print(f"\nProteins within 2 degrees of {target_protein}:")
        for _, row in close_proteins.iterrows():
            func_info = row['functional_relationship'] if row['functional_relationship'] else 'Unknown'
            print(f"  • {row['protein']}: {row['distance']} degree(s) - {func_info}")
    
    # Count activators and inhibitors
    activators = results[results['functional_relationship'].astype(str).str.contains('activat', case=False, na=False)]
    inhibitors = results[results['functional_relationship'].astype(str).str.contains('inhibit', case=False, na=False)]
    print(f"\nFunctional Relationships (≤2 degrees):")
    print(f"  Activators: {len(activators)}")
    print(f"  Inhibitors: {len(inhibitors)}")
    print(f"  Unknown: {len(close_proteins) - len(activators) - len(inhibitors)}")
    
    print("\n" + "=" * 70)
    
    return results
