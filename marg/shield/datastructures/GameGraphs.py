from pprint import pprint, pformat


class TwoPlayerGameGraph:
    def __init__(self, vertices, priorities, ownership, outgoing_edges, incoming_edges, initial_state, edge_labels=None, vertex_labels=None):
        self.initial_state = initial_state
        self.vertices = vertices
        self.priorities = priorities
        self.ownership = ownership
        self.outgoing_edges = outgoing_edges
        self.incoming_edges = incoming_edges
        self.edge_labels = edge_labels
        self.vertex_labels = vertex_labels

    @classmethod
    def from_edges_list(cls, edges_list, priorities, ownership, initial_state, edge_labels=None, vertex_labels=None):
        vertices = set()
        outgoing_edges = {}
        incoming_edges = {}
        for state, action, next_state in edges_list:
            vertices.add(state)
            vertices.add(next_state)
            if state not in outgoing_edges:
                outgoing_edges[state] = {}
            outgoing_edges[state].add((action, next_state))
            if next_state not in incoming_edges:
                incoming_edges[next_state] = {}
            incoming_edges[next_state].add((action, state))
        return cls(vertices, priorities, ownership, outgoing_edges, incoming_edges, initial_state, edge_labels, vertex_labels)
    
    def get_player_vertices(self, player):
        return [state for state in self.vertices if self.ownership[state] == player]

    def get_vertices_with_priority(self, index, priority):
        return [state for state in self.vertices if self.priorities[state][index] == priority]

    def get_maximum_priority(self, index):
        return max([priority[index] for priority in self.priorities.values()])
    
    def get_maximum_vertex_label(self):
        return max(self.vertex_labels.values())
    
    def assign_owner(self, state, owner):
        self.ownership[state] = owner
    
    def set_initial_state(self, state):
        self.initial_state = state
    
    def add_vertex(self, state, priority, owner):
        self.vertices.add(state)
        self.priorities[state] = priority
        self.ownership[state] = owner
        self.outgoing_edges[state] = {}
        self.incoming_edges[state] = {}

    def add_edge(self, state, action, next_state):
        self.vertices.add(state)
        self.vertices.add(next_state)
        if state not in self.outgoing_edges:
            self.outgoing_edges[state] = {}
        self.outgoing_edges[state].add((action, next_state))
        if next_state not in self.incoming_edges:
            self.incoming_edges[next_state] = {}
        self.incoming_edges[next_state].add((action, state))

    def delete_edge(self, state, action, next_state):
        self.outgoing_edges[state].remove((action, next_state))
        self.incoming_edges[next_state].remove((action, state))

    def get_actions(self, state):
        return [action for action, _ in self.outgoing_edges[state]]

    def get_next_states(self, state):
        return [next_state for _, next_state in self.outgoing_edges[state]]

    def get_next_states_with_actions(self, state):
        return self.outgoing_edges[state]
    
    def get_state(self, state, action):
        for a, next_state in self.outgoing_edges[state]:
            if a == action:
                return next_state
        return None

    def is_terminal(self, state):
        return self.outgoing_edges[state] == {}
    
    def subgame(self, region):
        vertices = region
        priorities = {state: self.priorities[state] for state in region}
        ownership = {state: self.ownership[state] for state in region}
        outgoing_edges = {state: self.restrict_edge_to_vertex_set(self.outgoing_edges[state], vertices) for state in region}
        incoming_edges = {state: self.restrict_edge_to_vertex_set(self.incoming_edges[state], vertices) for state in region}
        initial_state = self.initial_state if self.initial_state in region else None
        edge_labels = {state: self.edge_labels[state] for state in region} if self.edge_labels is not None else None
        vertex_labels = {state: self.vertex_labels[state] for state in region} if self.vertex_labels is not None else None
        return TwoPlayerGameGraph(vertices, priorities, ownership, outgoing_edges, incoming_edges, initial_state, edge_labels, vertex_labels)
    
    def restrict_edge_to_vertex_set(self, outgoing_edges, vertex_set):
        new_outgoing_edges = []
        for action, target_state in outgoing_edges:
            if target_state in vertex_set:
                new_outgoing_edges.append((action, target_state))
        return new_outgoing_edges
    
    def __str__(self):
        return f"Vertices: {self.vertices}\nPriorities: {self.priorities}\nOwnership: {self.ownership}\nOutgoing edges: {self.outgoing_edges}\nIncoming edges: {self.incoming_edges}\nInitial state: {self.initial_state}\nEdge labels: {self.edge_labels}\nVertex labels: {self.vertex_labels}"
    

    class UnsafeEdges:
        def __init__(self, unsafe_outgoing_edges):
            self.unsafe_outgoing_edges = unsafe_outgoing_edges

        def __str__(self):
            return pformat(self.unsafe_outgoing_edges)
        
        def get_unsafe_actions(self, state):
            return [action for action, _ in self.unsafe_outgoing_edges[state]]
        
        def get_unsafe_next_states(self, state):
            return [next_state for _, next_state in self.unsafe_outgoing_edges[state]]
        
        def get_unsafe_next_states_with_actions(self, state):
            return self.unsafe_outgoing_edges[state]
        
        def add_unsafe_edge(self, state, action, next_state):
            if state not in self.unsafe_outgoing_edges:
                self.unsafe_outgoing_edges[state] = {}
            self.unsafe_outgoing_edges[state].add((action, next_state))

        def delete_unsafe_edge(self, state, action, next_state):
            self.unsafe_outgoing_edges[state].remove((action, next_state))
        
        def get_sub_template(self, region):
            for (state, edges) in self.unsafe_outgoing_edges.items():
                if state not in region:
                    del self.unsafe_outgoing_edges[state]
                else:
                    self.unsafe_outgoing_edges[state] = {(action, next_state) for action, next_state in edges if next_state in region}

    class ColiveEdges:
        def __init__(self, colive_outgoing_edges):
            self.colive_outgoing_edges = colive_outgoing_edges

        def __str__(self):
            return pformat(self.colive_outgoing_edges)

        def get_colive_states(self):
            return [state for state in self.colive_outgoing_edges]
        
        def get_colive_actions(self, state):
            return [action for action, _ in self.colive_outgoing_edges[state]]
        
        def get_colive_next_states(self, state):
            return [next_state for _, next_state in self.colive_outgoing_edges[state]]
        
        def get_colive_next_states_with_actions(self, state):
            return self.colive_outgoing_edges[state]
        
        def add_colive_edge(self, state, action, next_state):
            if state not in self.colive_outgoing_edges:
                self.colive_outgoing_edges[state] = {}
            self.colive_outgoing_edges[state].add((action, next_state))

        def delete_colive_edge(self, state, action, next_state):
            self.colive_outgoing_edges[state].remove((action, next_state))

        def add_edges_between_regions_one_player(self, regionA, regionB, outgoing_edges):
            for state in regionA:
                for action, next_state in outgoing_edges[state]:
                    if next_state in regionB:
                        self.add_colive_edge(state, action, next_state)
        
        def get_sub_template(self, region):
            for (state, edges) in self.colive_outgoing_edges.items():
                if state not in region:
                    del self.colive_outgoing_edges[state]
                else:
                    self.colive_outgoing_edges[state] = {(action, next_state) for action, next_state in edges if next_state in region}
        
    class LiveGroup:
        def __init__(self, live_group_edges):
            self.live_group_edges = live_group_edges

        def __str__(self):
            return pformat(self.live_group_edges)
        
        def get_live_states(self):
            return [state for state, edges in self.live_group_edges.items() if edges != []]

        def get_live_actions(self, state):
            return [action for action, _ in self.live_group_edges[state]]
        
        def get_live_next_states(self, state):
            return [next_state for _, next_state in self.live_group_edges[state]]
        
        def get_live_next_states_with_actions(self, state):
            return self.live_group_edges[state]
        
        def add_live_edge(self, state, action, next_state):
            if state not in self.live_group_edges:
                self.live_group_edges[state] = {}
            self.live_group_edges[state].add((action, next_state))

        def delete_live_edge(self, state, action, next_state):
            self.live_group_edges[state].remove((action, next_state))

        def add_edges_to_region(self, game_graph, state, region):
            for action, next_state in game_graph.outgoing_edges[state]:
                if next_state in region:
                    self.add_live_edge(state, action, next_state)

        def isempty(self):
            is_empty = True
            for state, edges in self.live_group_edges.items():
                if edges != set():
                    is_empty = False
                    break
            return is_empty

        def get_sub_template(self, region):
            for (state, edges) in self.live_group_edges.items():
                if state not in region:
                    del self.live_group_edges[state]
                else:
                    self.live_group_edges[state] = {(action, next_state) for action, next_state in edges if next_state in region}

        def __eq__(self, value):
            if not isinstance(value, TwoPlayerGameGraph.LiveGroup):
                return False
            return self.live_group_edges == value.live_group_edges
        


    class LiveGroups:
        def __init__(self, live_groups_list):
            self.live_groups_list = live_groups_list

        def __len__(self):
            return len(self.live_groups_list)
        
        def __str__(self):
            return pformat([str(live_group) for live_group in self.live_groups_list])
        
        def get_live_group_by_index(self, index):
            return self.live_groups_list[index]
        
        def get_live_states(self):
            return set([state for live_group in self.live_groups_list for state in live_group.get_live_states()])

        def add_live_group(self, live_group):
            self.live_groups_list.append(live_group)
        
        def delete_live_group(self, live_group):
            self.live_groups_list.remove(live_group)

        def add_live_groups_to_reach_one_player(self, region, incoming_edges):
            old_region = set()
            current_region = region.copy()
            while current_region != old_region:
                old_region = current_region.copy()
                live_group = TwoPlayerGameGraph.LiveGroup({state: set() for state in incoming_edges})
                for state in old_region:
                    for action, prev_state in incoming_edges[state]:
                        if prev_state not in old_region:
                            live_group.add_live_edge(prev_state, action, state)
                            current_region.add(prev_state)
                if not live_group.isempty():
                    self.add_live_group(live_group)

        def get_sub_template(self, region):
            for live_group in self.live_groups_list:
                live_group.get_sub_template(region)
                
        def add_live_groups_to_reach(self, game_graph, region):
            old_region = set()
            current_region = region.copy()
            while current_region != old_region:
                old_region = current_region.copy()
                live_group = TwoPlayerGameGraph.LiveGroup({state: set() for state in game_graph.incoming_edges})
                for state in old_region:
                    for _, prev_state in game_graph.incoming_edges[state]:
                        if prev_state not in old_region:
                            if game_graph.ownership[prev_state] == 0:
                                live_group.add_edges_to_region(game_graph, prev_state, old_region)
                                current_region.add(prev_state)
                            else:
                                if set(game_graph.get_next_states(prev_state)).issubset(old_region):
                                    current_region.add(prev_state)
                        
                if not live_group.isempty():
                    self.add_live_group(live_group)
            return current_region
                

