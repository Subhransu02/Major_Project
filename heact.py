import math
import random
import numpy as np
import matplotlib.pyplot as plt
import copy

# --- Simulation Parameters (Consistent across protocols) ---
NETWORK_SIZE_X = 100
NETWORK_SIZE_Y = 100
BS_POS = (50, 150)
NUM_NODES = 100
INITIAL_ENERGY = 0.5
PACKET_SIZE_DATA = 4000
PACKET_SIZE_CTRL = 200

# Energy Model Parameters
E_ELEC = 50e-9
E_FS = 10e-12
E_MP = 0.0013e-12
E_DA = 5e-9
D_THRESHOLD = math.sqrt(E_FS / E_MP)

# LEACH Specific Parameters
LEACH_P = 0.1 # Desired percentage of cluster heads

# HEACT Specific Parameters
HEACT_P_CH = 0.1 # Target percentage of CHs
HEACT_RECLUSTER_INTERVAL = 20 # How often to reform clusters and CH tree
HEACT_MIN_ENERGY_FOR_CH_CANDIDACY = INITIAL_ENERGY * 0.05
HEACT_MIN_ENERGY_FOR_CH_RELAY = INITIAL_ENERGY * 0.1

# --- Helper Functions (Common) ---
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def calculate_transmit_energy(m_bits, distance):
    if distance <= 0: return 0
    # Ensure distance is positive if very small non-zero
    distance = max(distance, 1e-6)
    if distance < D_THRESHOLD:
        energy = E_ELEC * m_bits + E_FS * m_bits * (distance ** 2)
    else:
        energy = E_ELEC * m_bits + E_MP * m_bits * (distance ** 4)
    return max(0, energy) # Ensure non-negative

def calculate_receive_energy(m_bits):
    return E_ELEC * m_bits

def calculate_aggregate_energy(m_bits):
    return E_DA * m_bits

# --- Node Class Definitions ---

class SensorNodeEETB:
    def __init__(self, node_id, x, y, energy=INITIAL_ENERGY):
        self.id = node_id
        self.x = x
        self.y = y
        self.initial_energy = energy
        self.energy = energy
        self.status = "alive"
        self.dist_to_bs = calculate_distance(self.x, self.y, BS_POS[0], BS_POS[1])
        # EETB specific
        self.parent = None
        self.children = set()
        self.is_relay = False
        self.path_cost_sq = float('inf')
        self.branch_root_id = -1

    def is_alive(self):
        return self.status == "alive" and self.energy > 0

    def reset_tree_links(self): # Renamed from reset_for_new_round for clarity
        self.parent = None
        self.children = set()
        self.is_relay = False
        self.path_cost_sq = float('inf')
        self.branch_root_id = -1

    def reset_for_simulation(self):
        self.energy = self.initial_energy
        self.status = "alive"
        self.reset_tree_links()

    def __repr__(self):
        state = f"id={self.id}, E={self.energy:.3f}, St={self.status}"
        if self.parent: state += f", Parent={self.parent}"
        return f"NodeEETB({state})"

class SensorNodeLEACH:
    def __init__(self, node_id, x, y, energy=INITIAL_ENERGY):
        self.id = node_id
        self.x = x
        self.y = y
        self.initial_energy = energy
        self.energy = energy
        self.status = "alive"
        self.dist_to_bs = calculate_distance(self.x, self.y, BS_POS[0], BS_POS[1])
        # LEACH specific
        self.is_ch = False
        self.cluster_id = -1 # ID of the CH it belongs to
        self.role = 'member' # 'member', 'CH'
        self.leach_epoch = -1 # Last round it was CH

    def is_alive(self):
        return self.status == "alive" and self.energy > 0

    def reset_leach_round_state(self): # Renamed
        self.is_ch = False
        self.cluster_id = -1
        self.role = 'member'

    def reset_for_simulation(self):
        self.energy = self.initial_energy
        self.status = "alive"
        self.leach_epoch = -1
        self.reset_leach_round_state()

    def __repr__(self):
        state = f"id={self.id}, E={self.energy:.3f}, St={self.status}"
        if self.role == 'CH': state += ", Role=CH"
        elif self.cluster_id != -1: state += f", Clust={self.cluster_id}"
        return f"NodeLEACH({state})"

class SensorNodeHEACT:
    def __init__(self, node_id, x, y, energy=INITIAL_ENERGY):
        self.id = node_id
        self.x = x
        self.y = y
        self.initial_energy = energy
        self.energy = energy
        self.status = "alive"
        self.dist_to_bs = calculate_distance(self.x, self.y, BS_POS[0], BS_POS[1])
        # HEACT Cluster State
        self.role = 'member' # 'member', 'CH'
        self.is_ch = False
        self.cluster_id = -1 # ID of the CH this node belongs to
        # HEACT Inter-CH Tree State
        self.parent_ch_id = None
        self.children_ch_ids = set()
        self.is_relay_ch = False
        self.path_cost_sq_ch = float('inf')

    def is_alive(self):
        return self.status == "alive" and self.energy > 0

    def reset_heact_round_state(self): # Renamed
        self.role = 'member'
        self.is_ch = False
        self.cluster_id = -1
        self.parent_ch_id = None
        self.children_ch_ids = set()
        self.is_relay_ch = False
        self.path_cost_sq_ch = float('inf')

    def reset_for_simulation(self):
        self.energy = self.initial_energy
        self.status = "alive"
        self.reset_heact_round_state()

    def __repr__(self):
        state = f"id={self.id}, E={self.energy:.3f}, St={self.status}"
        if self.is_ch: state += f", Role=CH"
        elif self.cluster_id != -1: state += f", Clust={self.cluster_id}"
        if self.parent_ch_id: state += f", ParentCH={self.parent_ch_id}"
        return f"NodeHEACT({state})"


# --- EETB Algorithm Implementation ---

def calculate_optimal_branches_eetb(nodes, L=NETWORK_SIZE_X):
    alive_nodes_list = [n for n in nodes if n.is_alive()]
    N_alive = len(alive_nodes_list)
    if N_alive == 0: return 1
    avg_dist_to_bs = np.mean([n.dist_to_bs for n in alive_nodes_list]) if alive_nodes_list else 1.0

    try:
        term1 = N_alive * E_FS * (L**2) / 3.0
        term2_fs = E_FS * (avg_dist_to_bs**2) - E_ELEC - E_DA
        term3_fs = E_FS * (L**2) / 6.0
        term2 = term2_fs
        term3 = term3_fs

        if term2 <= term3 or term1 <=0:
             h_opt_calc = max(1, int(math.sqrt(N_alive)))
        else:
             try: h_opt_calc = (term1 / (term2 - term3))**(1/3)
             except ValueError: h_opt_calc = math.sqrt(N_alive)

        h_opt = max(1, min(h_opt_calc, N_alive))
        return max(1, int(round(h_opt)))
    except (ValueError, ZeroDivisionError, OverflowError) as e:
        return max(1, int(math.sqrt(N_alive)))

def build_routing_tree_eetb(nodes, bs_pos, h_opt, energy_threshold=0.0):
    alive_nodes = [n for n in nodes if n.is_alive()]
    if not alive_nodes: return
    for node in nodes: node.reset_tree_links() # Use EETB specific reset

    sorted_nodes = sorted(alive_nodes, key=lambda n: n.dist_to_bs)
    visited_nodes_ids = set()
    potential_relays = [n for n in alive_nodes if n.energy >= energy_threshold]

    branch_roots = []
    nodes_considered_for_root = [n for n in sorted_nodes if n in potential_relays]

    for node in nodes_considered_for_root:
        if len(branch_roots) < h_opt:
            node.parent = "BS"; node.path_cost_sq = node.dist_to_bs**2
            node.is_relay = True; node.branch_root_id = node.id
            branch_roots.append(node); visited_nodes_ids.add(node.id)
        else: break

    if not branch_roots and potential_relays:
        root_node = potential_relays[0]
        root_node.parent = "BS"; root_node.path_cost_sq = root_node.dist_to_bs**2
        root_node.is_relay = True; root_node.branch_root_id = root_node.id
        branch_roots.append(root_node); visited_nodes_ids.add(root_node.id)

    if not branch_roots: return

    unvisited_nodes = [n for n in alive_nodes if n.id not in visited_nodes_ids]
    current_tree_nodes_map = {n.id: n for n in branch_roots}

    while unvisited_nodes:
        best_next_node = None; best_parent_node_id = None; min_cost = float('inf')
        possible_parent_nodes = [p for p in current_tree_nodes_map.values() if p.is_relay and p.energy >= energy_threshold]
        if not possible_parent_nodes: possible_parent_nodes = branch_roots
        if not possible_parent_nodes: break

        parent_found_for_iteration = False; q_node_to_remove = None
        for q_node in unvisited_nodes:
            for i_node in possible_parent_nodes:
                dist_qi = calculate_distance(q_node.x, q_node.y, i_node.x, i_node.y)
                current_cost = (dist_qi**2) + i_node.path_cost_sq
                if current_cost < min_cost:
                    min_cost = current_cost; best_next_node = q_node
                    best_parent_node_id = i_node.id; parent_found_for_iteration = True

        if parent_found_for_iteration and best_next_node and best_parent_node_id:
            q = best_next_node; i_node = current_tree_nodes_map[best_parent_node_id]
            q.parent = i_node.id; i_node.children.add(q.id); q.path_cost_sq = min_cost
            q.branch_root_id = i_node.branch_root_id
            q.is_relay = q.energy >= energy_threshold
            visited_nodes_ids.add(q.id); current_tree_nodes_map[q.id] = q
            q_node_to_remove = q
            min_cost = float('inf'); best_next_node = None; best_parent_node_id = None # Reset for next search
        else: break # Cannot connect remaining

        if q_node_to_remove: unvisited_nodes.remove(q_node_to_remove)

def get_traversal_order_eetb(nodes):
    alive_nodes_map = {n.id: n for n in nodes if n.is_alive()}
    if not alive_nodes_map: return []
    parent_map = {n.id: n.parent for n in alive_nodes_map.values() if n.parent is not None}
    children_map = {n.id: set(c for c in n.children if c in alive_nodes_map) for n in alive_nodes_map.values()}
    order = []; processed = set()
    leaves = [nid for nid, node in alive_nodes_map.items() if not children_map.get(nid)]
    queue = sorted(list(leaves))
    while queue:
        node_id = queue.pop(0)
        if node_id in processed: continue
        order.append(node_id); processed.add(node_id)
        parent_id = parent_map.get(node_id)
        if parent_id and parent_id != "BS" and parent_id in alive_nodes_map:
            all_children_done = all(child_id in processed for child_id in children_map.get(parent_id, set()))
            if all_children_done and parent_id not in processed and parent_id not in queue:
                 queue.append(parent_id); queue.sort()
    remaining = sorted([nid for nid in alive_nodes_map if nid not in processed])
    order.extend(remaining)
    return order

def simulate_eetb_round(nodes, bs_pos):
    energy_consumed_round = {n.id: 0.0 for n in nodes}
    packets_to_bs = 0
    nodes_in_order = get_traversal_order_eetb(nodes)
    aggregated_data_count = {n.id: 0 for n in nodes}
    nodes_map = {n.id: n for n in nodes}

    for node_id in nodes_in_order:
        node = nodes_map.get(node_id)
        if not node or not node.is_alive(): continue
        # Receive Phase
        bits_received_total = 0
        if node.is_relay and node.children:
            for child_id in list(node.children):
                 child_node = nodes_map.get(child_id)
                 if child_node and child_node.is_alive():
                      bits_transmitted_by_child = PACKET_SIZE_DATA * (aggregated_data_count[child_id] + 1)
                      e_rx = calculate_receive_energy(bits_transmitted_by_child)
                      if node.energy >= e_rx:
                           node.energy -= e_rx; energy_consumed_round[node.id] += e_rx
                           bits_received_total += bits_transmitted_by_child
                           aggregated_data_count[node.id] += (aggregated_data_count[child_id] + 1)
                      else: node.energy = 0; node.status = "dead"; energy_consumed_round[node.id] += node.energy; continue
        # Aggregate Phase
        num_packets_received = aggregated_data_count[node.id] # Actual received count
        if node.is_relay and num_packets_received > 0:
             e_da = calculate_aggregate_energy(PACKET_SIZE_DATA * num_packets_received)
             if node.energy >= e_da: node.energy -= e_da; energy_consumed_round[node.id] += e_da
             else: node.energy = 0; node.status = "dead"; energy_consumed_round[node.id] += node.energy; continue
        # Transmit Phase
        if node.parent is not None and node.is_alive():
             bits_to_transmit = PACKET_SIZE_DATA * (aggregated_data_count[node.id] + 1)
             parent_is_alive = False; distance = 0
             if node.parent == "BS": distance = node.dist_to_bs; parent_is_alive = True
             else:
                 parent_node = nodes_map.get(node.parent)
                 if parent_node and parent_node.is_alive():
                      distance = calculate_distance(node.x, node.y, parent_node.x, parent_node.y); parent_is_alive = True
             if parent_is_alive:
                  e_tx = calculate_transmit_energy(bits_to_transmit, distance)
                  if node.energy >= e_tx:
                      node.energy -= e_tx; energy_consumed_round[node.id] += e_tx
                      if node.parent == "BS": packets_to_bs += (aggregated_data_count[node.id] + 1)
                  else: node.energy = 0; node.status = "dead"; energy_consumed_round[node.id] += node.energy; continue
        if node.energy <= 0: node.status = "dead"; node.energy = 0
    return energy_consumed_round, packets_to_bs

def calculate_energy_threshold_and_interval_eetb(nodes, energy_consumed_prev_round):
    alive_nodes = [n for n in nodes if n.is_alive()]; N_alive = len(alive_nodes)
    if N_alive == 0: return 0.0, 1
    ec_round = {nid: energy_consumed_prev_round.get(nid, 0.0) for nid in [n.id for n in alive_nodes] if energy_consumed_prev_round.get(nid, 0.0) > 1e-12}
    if not ec_round: return np.mean([n.energy for n in alive_nodes] or [0]) * 0.1, 10
    ec_values = list(ec_round.values()); ec_max_j = max(ec_values) if ec_values else 0
    ec_avg_j = sum(ec_values) / N_alive if N_alive > 0 else 0
    alpha_j = max(0, min(1, (ec_max_j - ec_avg_j) / ec_max_j)) if ec_max_j > 0 else 0
    avg_remaining_energy = np.mean([n.energy for n in alive_nodes])
    e_th = avg_remaining_energy * alpha_j
    min_rounds_left = float('inf'); lambda_values = {}
    alive_nodes_map = {n.id: n for n in alive_nodes}
    for node_id, ec_node in ec_round.items():
         node = alive_nodes_map.get(node_id)
         if node and ec_node > 1e-12:
             rounds_left = node.energy / ec_node; min_rounds_left = min(min_rounds_left, rounds_left)
             if ec_avg_j > 1e-12: lambda_values[node_id] = ec_node / ec_avg_j
    min_lambda = min(lambda_values.values()) if lambda_values else 1.0
    if min_rounds_left != float('inf') and min_lambda > 1e-9: r_dyit = (min_rounds_left / min_lambda) / 2
    elif min_rounds_left != float('inf'): r_dyit = min_rounds_left / 2
    else: r_dyit = 10
    r_dyit = max(1, int(round(r_dyit)))
    return e_th, r_dyit
# --- END EETB Code ---


# --- LEACH Algorithm Implementation ---

def select_leach_chs(nodes, p, current_round):
    alive_nodes = [n for n in nodes if n.is_alive()]
    candidate_chs = []
    last_ch_round_limit = current_round - (1 / p if p > 0 else float('inf'))
    for node in alive_nodes:
        node.reset_leach_round_state() # Reset roles first
        if node.leach_epoch < last_ch_round_limit:
            threshold = p / (1 - p * (current_round % (1 / p))) if p > 0 and (1 - p * (current_round % (1 / p))) > 0 else p
            if random.random() < threshold:
                node.is_ch = True; node.role = 'CH'; node.leach_epoch = current_round
                node.cluster_id = node.id; candidate_chs.append(node)
    # Ensure at least one CH if possible
    if not candidate_chs and alive_nodes:
         highest_energy_node = max(alive_nodes, key=lambda n: n.energy)
         if highest_energy_node.leach_epoch < last_ch_round_limit : # Check eligibility
             highest_energy_node.is_ch = True; highest_energy_node.role = 'CH'
             highest_energy_node.leach_epoch = current_round; highest_energy_node.cluster_id = highest_energy_node.id
             candidate_chs.append(highest_energy_node)
             print(f"LEACH Warning: No CH elected probabilistically, selected node {highest_energy_node.id}")

    return candidate_chs

def form_leach_clusters(nodes, cluster_heads):
    alive_members = [n for n in nodes if n.is_alive() and not n.is_ch]
    ch_map = {ch.id: ch for ch in cluster_heads if ch.is_alive()}
    clusters = {ch_id: [] for ch_id in ch_map}
    energy_consumed = {n.id: 0.0 for n in nodes}
    if not ch_map: return clusters, energy_consumed # No live CHs

    # Simplified CH Ad cost (optional)
    # for ch in ch_map.values(): e_tx_ad = calculate_transmit_energy(...); ch.energy -= e_tx_ad; ...

    for member in alive_members:
        member.cluster_id = -1; best_ch_id = -1; min_dist_sq = float('inf')
        for ch_id, ch_node in ch_map.items():
            dist_sq = (member.x - ch_node.x)**2 + (member.y - ch_node.y)**2
            if dist_sq < min_dist_sq: min_dist_sq = dist_sq; best_ch_id = ch_id
        if best_ch_id != -1:
            member.cluster_id = best_ch_id
            ch_node = ch_map[best_ch_id]
            dist_to_ch = math.sqrt(min_dist_sq)
            e_tx_join = calculate_transmit_energy(PACKET_SIZE_CTRL, dist_to_ch)
            if member.energy >= e_tx_join:
                member.energy -= e_tx_join; energy_consumed[member.id] += e_tx_join
                e_rx_join = calculate_receive_energy(PACKET_SIZE_CTRL)
                if ch_node.energy >= e_rx_join:
                    ch_node.energy -= e_rx_join; energy_consumed[ch_node.id] += e_rx_join
                    clusters[best_ch_id].append(member.id)
                else: ch_node.energy=0; ch_node.status='dead'; energy_consumed[ch_node.id]+=ch_node.energy; member.cluster_id = -1
            else: member.energy=0; member.status='dead'; energy_consumed[member.id]+=member.energy
    return clusters, energy_consumed

def simulate_leach_steady_state(nodes, cluster_heads, clusters):
    energy_consumed = {n.id: 0.0 for n in nodes}; packets_to_bs = 0
    ch_map = {ch.id: ch for ch in cluster_heads if ch.is_alive()}
    nodes_map = {n.id: n for n in nodes}
    for ch_id, member_ids in clusters.items():
        ch_node = ch_map.get(ch_id);
        if not ch_node or not ch_node.is_alive(): continue
        members_sent_count = 0
        for member_id in member_ids:
            member_node = nodes_map.get(member_id)
            if member_node and member_node.is_alive() and member_node.cluster_id == ch_id:
                dist_to_ch = calculate_distance(member_node.x, member_node.y, ch_node.x, ch_node.y)
                e_tx = calculate_transmit_energy(PACKET_SIZE_DATA, dist_to_ch)
                if member_node.energy >= e_tx:
                    member_node.energy -= e_tx; energy_consumed[member_id] += e_tx
                    e_rx = calculate_receive_energy(PACKET_SIZE_DATA)
                    if ch_node.energy >= e_rx:
                        ch_node.energy -= e_rx; energy_consumed[ch_id] += e_rx
                        members_sent_count += 1
                    else: ch_node.energy=0; ch_node.status='dead'; energy_consumed[ch_id]+=ch_node.energy; break
                else: member_node.energy=0; member_node.status='dead'; energy_consumed[member_id]+=member_node.energy
        # CH aggregates and transmits
        if ch_node.is_alive():
             num_packets_to_aggregate = members_sent_count
             if num_packets_to_aggregate > 0:
                  e_da = calculate_aggregate_energy(PACKET_SIZE_DATA * num_packets_to_aggregate)
                  if ch_node.energy >= e_da: ch_node.energy -= e_da; energy_consumed[ch_id] += e_da
                  else: ch_node.energy=0; ch_node.status='dead'; energy_consumed[ch_id]+=ch_node.energy
             if ch_node.is_alive(): # Check again after aggregation
                  bits_to_transmit = PACKET_SIZE_DATA * (members_sent_count + 1)
                  dist_to_bs = ch_node.dist_to_bs
                  e_tx_bs = calculate_transmit_energy(bits_to_transmit, dist_to_bs)
                  if ch_node.energy >= e_tx_bs:
                      ch_node.energy -= e_tx_bs; energy_consumed[ch_id] += e_tx_bs
                      packets_to_bs += (members_sent_count + 1)
                  else: ch_node.energy=0; ch_node.status='dead'; energy_consumed[ch_id]+=ch_node.energy
    # Final death check for all nodes
    for node in nodes:
        if node.energy <= 0: node.status = "dead"; node.energy = 0
    return energy_consumed, packets_to_bs
# --- END LEACH Code ---


# --- HEACT Algorithm Implementation ---
def select_heact_chs(nodes, p_ch, min_energy_threshold):
    alive_nodes = [n for n in nodes if n.is_alive()]
    cluster_heads = []
    for node in alive_nodes:
        node.reset_heact_round_state() # Reset roles first
        if node.energy < min_energy_threshold: continue
        energy_factor = node.energy / node.initial_energy
        threshold = p_ch * energy_factor
        if random.random() < threshold:
            node.is_ch = True; node.role = 'CH'; node.cluster_id = node.id
            cluster_heads.append(node)
    if not cluster_heads and alive_nodes:
         highest_energy_node = max(alive_nodes, key=lambda n: n.energy)
         if highest_energy_node.energy >= min_energy_threshold:
             highest_energy_node.is_ch = True; highest_energy_node.role = 'CH'
             highest_energy_node.cluster_id = highest_energy_node.id
             cluster_heads.append(highest_energy_node)
             print(f"HEACT Warning: No CH selected probabilistically, selected node {highest_energy_node.id}")
    return cluster_heads

def form_heact_clusters(nodes, cluster_heads):
    # This logic is identical to form_leach_clusters, just uses HEACT node state
    alive_members = [n for n in nodes if n.is_alive() and not n.is_ch]
    ch_map = {ch.id: ch for ch in cluster_heads if ch.is_alive()}
    clusters = {ch_id: [] for ch_id in ch_map}
    energy_consumed = {n.id: 0.0 for n in nodes}
    if not ch_map: return clusters, energy_consumed
    for member in alive_members:
        member.cluster_id = -1; best_ch_id = -1; min_dist_sq = float('inf')
        for ch_id, ch_node in ch_map.items():
            dist_sq = (member.x - ch_node.x)**2 + (member.y - ch_node.y)**2
            if dist_sq < min_dist_sq: min_dist_sq = dist_sq; best_ch_id = ch_id
        if best_ch_id != -1:
            member.cluster_id = best_ch_id; ch_node = ch_map[best_ch_id]
            dist_to_ch = math.sqrt(min_dist_sq)
            e_tx_join = calculate_transmit_energy(PACKET_SIZE_CTRL, dist_to_ch)
            if member.energy >= e_tx_join:
                member.energy -= e_tx_join; energy_consumed[member.id] += e_tx_join
                e_rx_join = calculate_receive_energy(PACKET_SIZE_CTRL)
                if ch_node.energy >= e_rx_join:
                    ch_node.energy -= e_rx_join; energy_consumed[ch_node.id] += e_rx_join
                    clusters[best_ch_id].append(member.id)
                else: ch_node.energy=0; ch_node.status='dead'; energy_consumed[ch_node.id]+=ch_node.energy; member.cluster_id = -1
            else: member.energy=0; member.status='dead'; energy_consumed[member.id]+=member.energy
    return clusters, energy_consumed

def build_inter_cluster_tree_heact(cluster_heads, bs_pos, ch_relay_energy_threshold):
    live_chs = [ch for ch in cluster_heads if ch.is_alive()]
    if not live_chs: return
    for ch in live_chs: # Reset tree state
        ch.parent_ch_id = None; ch.children_ch_ids = set(); ch.is_relay_ch = False; ch.path_cost_sq_ch = float('inf')
    sorted_chs = sorted(live_chs, key=lambda ch: ch.dist_to_bs)
    visited_ch_ids = set(); root_chs = []
    potential_relay_chs = [ch for ch in sorted_chs if ch.energy >= ch_relay_energy_threshold]
    primary_root = next((ch for ch in potential_relay_chs), None) # Find first eligible root
    if primary_root:
        primary_root.parent_ch_id = "BS"; primary_root.path_cost_sq_ch = primary_root.dist_to_bs**2
        primary_root.is_relay_ch = True; root_chs.append(primary_root); visited_ch_ids.add(primary_root.id)
    elif live_chs: # Fallback: connect closest non-eligible CH
         closest_ch = sorted_chs[0]
         closest_ch.parent_ch_id = "BS"; closest_ch.path_cost_sq_ch = closest_ch.dist_to_bs**2
         closest_ch.is_relay_ch = False; root_chs.append(closest_ch); visited_ch_ids.add(closest_ch.id)
    if not root_chs: return
    unvisited_chs = [ch for ch in live_chs if ch.id not in visited_ch_ids]
    current_tree_potential_parents = {ch.id: ch for ch in root_chs if ch.is_relay_ch}
    while unvisited_chs:
        best_next_ch = None; best_parent_ch_id = None; min_cost = float('inf')
        possible_parents = list(current_tree_potential_parents.values())
        if not possible_parents: break
        parent_found_for_iteration = False; ch_to_remove = None
        for q_ch in unvisited_chs:
            for i_ch in possible_parents:
                dist_qi = calculate_distance(q_ch.x, q_ch.y, i_ch.x, i_ch.y)
                current_cost = (dist_qi**2) + i_ch.path_cost_sq_ch
                if current_cost < min_cost:
                    min_cost = current_cost; best_next_ch = q_ch
                    best_parent_ch_id = i_ch.id; parent_found_for_iteration = True
        if parent_found_for_iteration and best_next_ch and best_parent_ch_id:
            q = best_next_ch; i_parent_ch = current_tree_potential_parents[best_parent_ch_id]
            q.parent_ch_id = i_parent_ch.id; i_parent_ch.children_ch_ids.add(q.id); q.path_cost_sq_ch = min_cost
            q.is_relay_ch = q.energy >= ch_relay_energy_threshold
            if q.is_relay_ch: current_tree_potential_parents[q.id] = q # Add if it can relay
            visited_ch_ids.add(q.id); ch_to_remove = q
            min_cost = float('inf'); best_next_ch = None; best_parent_ch_id = None # Reset
        else: break
        if ch_to_remove: unvisited_chs.remove(ch_to_remove)

def get_ch_traversal_order_heact(cluster_heads):
    live_chs_map = {ch.id: ch for ch in cluster_heads if ch.is_alive() and ch.parent_ch_id is not None}
    if not live_chs_map: return []
    parent_map = {ch.id: ch.parent_ch_id for ch in live_chs_map.values()}
    children_map = {ch_id: set() for ch_id in live_chs_map}
    for ch_id, parent_id in parent_map.items():
        if parent_id and parent_id != "BS" and parent_id in live_chs_map: children_map[parent_id].add(ch_id)
    order = []; processed = set()
    leaves = [ch_id for ch_id, ch in live_chs_map.items() if not children_map.get(ch_id)]
    queue = sorted(list(leaves))
    while queue:
        ch_id = queue.pop(0)
        if ch_id in processed: continue
        order.append(ch_id); processed.add(ch_id)
        parent_id = parent_map.get(ch_id)
        if parent_id and parent_id != "BS" and parent_id in live_chs_map:
            all_children_done = all(child_id in processed for child_id in children_map.get(parent_id, set()))
            if all_children_done and parent_id not in processed and parent_id not in queue:
                 queue.append(parent_id); queue.sort()
    remaining = sorted([ch_id for ch_id in live_chs_map if ch_id not in processed])
    order.extend(remaining)
    return order

def simulate_heact_steady_state(nodes, cluster_heads, clusters):
    energy_consumed = {n.id: 0.0 for n in nodes}; packets_to_bs = 0
    nodes_map = {n.id: n for n in nodes}; ch_map = {ch.id: ch for ch in cluster_heads if ch.is_alive()}
    ch_packets_aggregated = {ch_id: 0 for ch_id in ch_map}
    # Phase 1: Members to CH
    for ch_id, member_ids in clusters.items():
        ch_node = ch_map.get(ch_id);
        if not ch_node or not ch_node.is_alive(): continue
        members_sent_count = 0
        for member_id in member_ids:
            member_node = nodes_map.get(member_id)
            if member_node and member_node.is_alive() and member_node.cluster_id == ch_id:
                dist_to_ch = calculate_distance(member_node.x, member_node.y, ch_node.x, ch_node.y)
                e_tx = calculate_transmit_energy(PACKET_SIZE_DATA, dist_to_ch)
                if member_node.energy >= e_tx:
                    member_node.energy -= e_tx; energy_consumed[member_id] += e_tx
                    e_rx = calculate_receive_energy(PACKET_SIZE_DATA)
                    if ch_node.energy >= e_rx:
                        ch_node.energy -= e_rx; energy_consumed[ch_id] += e_rx; members_sent_count += 1
                    else: ch_node.energy=0; ch_node.status='dead'; energy_consumed[ch_id]+=ch_node.energy; break
                else: member_node.energy=0; member_node.status='dead'; energy_consumed[member_id]+=member_node.energy
        if ch_node.is_alive(): ch_packets_aggregated[ch_id] += members_sent_count
    # Phase 2: CH Aggregation
    for ch_id, ch_node in ch_map.items():
         if not ch_node.is_alive(): continue
         num_packets_from_members = ch_packets_aggregated[ch_id]
         if num_packets_from_members > 0:
              e_da = calculate_aggregate_energy(PACKET_SIZE_DATA * num_packets_from_members)
              if ch_node.energy >= e_da: ch_node.energy -= e_da; energy_consumed[ch_id] += e_da
              else: ch_node.energy = 0; ch_node.status = 'dead'; energy_consumed[ch_id] += ch_node.energy
    # Phase 3: Inter-CH Transmission
    ch_traversal_order = get_ch_traversal_order_heact(cluster_heads)
    for ch_id in ch_map: # Add CH's own packet before traversal
         if ch_map[ch_id].is_alive(): ch_packets_aggregated[ch_id] += 1
    for ch_id in ch_traversal_order:
        ch_node = ch_map.get(ch_id);
        if not ch_node or not ch_node.is_alive(): continue
        parent_id = ch_node.parent_ch_id;
        if parent_id is None: continue
        total_packets_to_send = ch_packets_aggregated[ch_id]
        if total_packets_to_send == 0: continue
        bits_to_transmit = PACKET_SIZE_DATA * total_packets_to_send
        parent_is_alive = False; parent_node = None; distance = 0
        if parent_id == "BS": distance = ch_node.dist_to_bs; parent_is_alive = True
        else:
            parent_node = ch_map.get(parent_id)
            if parent_node and parent_node.is_alive():
                distance = calculate_distance(ch_node.x, ch_node.y, parent_node.x, parent_node.y); parent_is_alive = True
        if parent_is_alive:
            e_tx = calculate_transmit_energy(bits_to_transmit, distance)
            if ch_node.energy >= e_tx:
                ch_node.energy -= e_tx; energy_consumed[ch_id] += e_tx
                if parent_id == "BS": packets_to_bs += total_packets_to_send
                elif parent_node:
                    e_rx = calculate_receive_energy(bits_to_transmit)
                    if parent_node.energy >= e_rx:
                        parent_node.energy -= e_rx; energy_consumed[parent_id] += e_rx
                        ch_packets_aggregated[parent_id] += total_packets_to_send # Crucial: update parent count
                    else: parent_node.energy = 0; parent_node.status = 'dead'; energy_consumed[parent_id] += parent_node.energy
            else: ch_node.energy = 0; ch_node.status = 'dead'; energy_consumed[ch_id] += ch_node.energy
    # Final death check
    for node in nodes:
        if node.energy <= 0: node.status = "dead"; node.energy = 0
    return energy_consumed, packets_to_bs
# --- END HEACT Code ---


# --- Generic Simulation Runner ---
def run_simulation_protocol(protocol_name, initial_node_positions, NodeClass, max_rounds=2000):
    """Runs simulation for a specific protocol."""
    nodes = [NodeClass(i, pos[0], pos[1]) for i, pos in enumerate(initial_node_positions)]
    for node in nodes: node.reset_for_simulation() # Ensure clean state

    print(f"\n--- Running Simulation for: {protocol_name} ---")

    stats = {'alive_nodes': [], 'packets_to_bs': [], 'total_energy': [],
             'first_dead': -1, 'all_dead': -1, 'protocol_name': protocol_name}
    round_num = 0

    # Protocol-specific state initialization
    eetb_h_opt = 1; eetb_threshold = 0.0; eetb_dyn_interval = 10
    eetb_rounds_since_update = 0; eetb_energy_consumed_last_round = {n.id: 0.0 for n in nodes}
    leach_clusters = {}; leach_cluster_heads = []
    heact_clusters = {}; heact_cluster_heads = []

    if protocol_name == "EETB":
        eetb_h_opt = calculate_optimal_branches_eetb(nodes)
        build_routing_tree_eetb(nodes, BS_POS, eetb_h_opt, eetb_threshold)
        eetb_rounds_since_update = 0

    while round_num < max_rounds:
        round_num += 1
        num_alive_before = len([n for n in nodes if n.is_alive()])
        if num_alive_before == 0:
            if stats['all_dead'] == -1: stats['all_dead'] = round_num - 1
            break

        round_packets = 0
        round_energy_consumed = {n.id: 0.0 for n in nodes}

        # --- Protocol Logic Execution ---
        if protocol_name == "EETB":
            eetb_rounds_since_update += 1
            sim_energy, round_packets = simulate_eetb_round(nodes, BS_POS)
            eetb_energy_consumed_last_round = sim_energy
            for nid, eng in sim_energy.items(): round_energy_consumed[nid] = round_energy_consumed.get(nid, 0) + eng

            # Check for update
            needs_update = False
            num_alive_now = len([n for n in nodes if n.is_alive()])
            if num_alive_now == 0: break # Stop if all died mid-round
            current_threshold, dynamic_interval = calculate_energy_threshold_and_interval_eetb(nodes, eetb_energy_consumed_last_round)
            if eetb_rounds_since_update >= dynamic_interval:
                 if num_alive_now < num_alive_before: needs_update = True
                 else:
                      below_threshold = any(n.energy < current_threshold for n in nodes if n.is_alive())
                      if below_threshold: needs_update = True
            if needs_update:
                 # print(f"[EETB] Updating tree at round {round_num}")
                 eetb_h_opt = calculate_optimal_branches_eetb(nodes)
                 build_routing_tree_eetb(nodes, BS_POS, eetb_h_opt, current_threshold)
                 eetb_rounds_since_update = 0

        elif protocol_name == "LEACH":
            leach_cluster_heads = select_leach_chs(nodes, LEACH_P, round_num)
            leach_clusters, setup_energy = form_leach_clusters(nodes, leach_cluster_heads)
            for nid, eng in setup_energy.items(): round_energy_consumed[nid] = round_energy_consumed.get(nid, 0) + eng
            if any(n.is_alive() for n in leach_cluster_heads):
                 steady_energy, round_packets = simulate_leach_steady_state(nodes, leach_cluster_heads, leach_clusters)
                 for nid, eng in steady_energy.items(): round_energy_consumed[nid] = round_energy_consumed.get(nid, 0) + eng

        elif protocol_name == "HEACT":
            if (round_num - 1) % HEACT_RECLUSTER_INTERVAL == 0:
                 # print(f"[HEACT] Round {round_num}: Reconfiguring...")
                 heact_cluster_heads = select_heact_chs(nodes, HEACT_P_CH, HEACT_MIN_ENERGY_FOR_CH_CANDIDACY)
                 heact_clusters, setup_energy = form_heact_clusters(nodes, heact_cluster_heads)
                 for nid, eng in setup_energy.items(): round_energy_consumed[nid] = round_energy_consumed.get(nid, 0) + eng
                 build_inter_cluster_tree_heact(heact_cluster_heads, BS_POS, HEACT_MIN_ENERGY_FOR_CH_RELAY)
            # Steady State
            if any(ch.is_alive() for ch in heact_cluster_heads):
                 steady_energy, round_packets = simulate_heact_steady_state(nodes, heact_cluster_heads, heact_clusters)
                 for nid, eng in steady_energy.items(): round_energy_consumed[nid] = round_energy_consumed.get(nid, 0) + eng

        # --- Stats Update ---
        num_alive_now = len([n for n in nodes if n.is_alive()])
        stats['alive_nodes'].append(num_alive_now)
        stats['packets_to_bs'].append(round_packets)
        stats['total_energy'].append(sum(n.energy for n in nodes if n.is_alive()))
        if num_alive_now < NUM_NODES and stats['first_dead'] == -1: stats['first_dead'] = round_num
        if num_alive_now == 0 and stats['all_dead'] == -1: stats['all_dead'] = round_num

        if round_num % 200 == 0:
            print(f"[{protocol_name}] Round: {round_num}, Alive: {num_alive_now}, Pkts: {round_packets}")

    # Final cleanup if loop finished before all nodes died
    if stats['all_dead'] == -1: stats['all_dead'] = max_rounds
    print(f"[{protocol_name}] Simulation finished. First dead: {stats['first_dead']}, Last alive round: {stats['all_dead']}")
    return stats


# --- Plotting and Summary ---
def plot_comparison_results(all_results, max_rounds):
    protocols_to_plot = list(all_results.keys())
    plt.figure(figsize=(15, 10))
    # Plot 1: Alive Nodes
    plt.subplot(2, 2, 1)
    for name in protocols_to_plot:
        results = all_results[name]
        rounds = list(range(1, len(results['alive_nodes']) + 1))
        nl_round = results['all_dead'] if results['all_dead'] != -1 else max_rounds
        plt.plot(rounds, results['alive_nodes'], label=f"{name} (FDD={results['first_dead']}, NLD={nl_round})")
    plt.xlabel("Rounds"); plt.ylabel("Number of Alive Nodes"); plt.title("Network Stability Comparison")
    plt.legend(); plt.grid(True); plt.ylim(bottom=0)
    # Plot 2: Throughput
    plt.subplot(2, 2, 2)
    for name in protocols_to_plot:
        results = all_results[name]
        rounds = list(range(1, len(results['packets_to_bs']) + 1))
        cumulative_packets = np.cumsum(results['packets_to_bs'])
        total_packets = cumulative_packets[-1] if len(cumulative_packets)>0 else 0
        plt.plot(rounds, cumulative_packets, label=f"{name} (Total={total_packets})")
    plt.xlabel("Rounds"); plt.ylabel("Cumulative Packets Received by BS"); plt.title("Network Throughput Comparison")
    plt.legend(loc='lower right'); plt.grid(True); plt.ylim(bottom=0)
    # Plot 3: Total Energy
    plt.subplot(2, 2, 3)
    for name in protocols_to_plot:
        results = all_results[name]
        rounds = list(range(1, len(results['total_energy']) + 1))
        plt.plot(rounds, results['total_energy'], label=name)
    plt.xlabel("Rounds"); plt.ylabel("Total Remaining Network Energy (J)"); plt.title("Network Energy Consumption")
    plt.legend(); plt.grid(True); plt.ylim(bottom=0)
    # Plot 4: Average Energy
    plt.subplot(2, 2, 4)
    for name in protocols_to_plot:
        results = all_results[name]
        rounds = list(range(1, len(results['total_energy']) + 1))
        avg_energy = [(results['total_energy'][i] / results['alive_nodes'][i]) if results['alive_nodes'][i] > 0 else 0 for i in range(len(rounds))]
        plt.plot(rounds, avg_energy, label=name)
    plt.xlabel("Rounds"); plt.ylabel("Average Energy per Alive Node (J)"); plt.title("Average Node Energy")
    plt.legend(); plt.grid(True); plt.ylim(bottom=0)
    plt.tight_layout(); plt.show()

def print_summary_table(all_results, max_rounds):
    print("\n--- Simulation Summary (Single Run) ---")
    print(f"{'Protocol':<10} | {'FDN (Round)':<12} | {'NL (Round)':<12} | {'Total Packets':<15}")
    print("-" * 55)
    for name, results in all_results.items():
         nl_round = results['all_dead'] if results['all_dead'] != -1 else max_rounds
         total_pkts = sum(results['packets_to_bs'])
         print(f"{name:<10} | {results['first_dead']:<12} | {nl_round:<12} | {total_pkts:<15}")


# --- Main Execution Block ---
if __name__ == "__main__":
    # Generate initial node positions once
    initial_node_positions = [(random.uniform(0, NETWORK_SIZE_X), random.uniform(0, NETWORK_SIZE_Y))
                              for _ in range(NUM_NODES)]

    protocols_to_run = {
        "EETB": SensorNodeEETB,
        "LEACH": SensorNodeLEACH,
        "HEACT": SensorNodeHEACT
    }
    all_results = {}
    max_sim_rounds = 2000 # Adjust as needed

    for proto_name, NodeClass in protocols_to_run.items():
        all_results[proto_name] = run_simulation_protocol(
            proto_name, initial_node_positions, NodeClass, max_rounds=max_sim_rounds
        )

    # --- Generate Outputs ---
    plot_comparison_results(all_results, max_sim_rounds)
    print_summary_table(all_results, max_sim_rounds)