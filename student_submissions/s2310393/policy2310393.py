import gurobipy as grb
from gurobipy import GRB
import numpy as np
import random
from policy import Policy

class Policy2310393(Policy):
    def __init__(self):
        self.stock_width = 10
        self.stock_height = 10
        self.starting_cut_patterns = None  # Placeholder for initial patterns
        self.required_quantities = None

    def get_action(self, observation, info):
        items = observation["products"]
        stockpile = observation["stocks"]

        self._prepare_data_(items)

        # Solve with column generation
        cut_solution = self._process_column_generation_(stockpile)

        # Choose an action based on the solution
        move = self._determine_action_(stockpile, cut_solution)
        return move

    def _prepare_data_(self, items):
        self.required_quantities = np.array([prod["quantity"] for prod in items])
        self.item_dimensions = np.array([prod["size"] for prod in items])

        # Create trivial patterns for initialization
        total_items = len(self.item_dimensions)
        self.starting_cut_patterns = np.eye(total_items, dtype = int)

    def _process_column_generation_(self, stockpile):
        is_new_pattern = True
        next_pattern = None
        active_patterns = self.starting_cut_patterns

        while is_new_pattern:
            if next_pattern is not None:
                active_patterns = np.column_stack((active_patterns, next_pattern))

            dual_values = self._solve_lp_relaxation_(active_patterns)
            is_new_pattern, next_pattern = self._find_new_pattern_(dual_values, stockpile)

        optimal_stock_count, optimal_solution = self._solve_ip_master_(active_patterns)
        return {"cut_patterns": active_patterns, "minimal_stock": optimal_stock_count, "optimal_numbers": optimal_solution}

    def _solve_lp_relaxation_(self, active_patterns):
        model = grb.Model()
        model.setParam('OutputFlag', 0)
        allocation_vars = model.addMVar(shape = active_patterns.shape[1], lb = 0, vtype = GRB.CONTINUOUS)
        model.addConstr(active_patterns @ allocation_vars >= self.required_quantities)
        model.setObjective(allocation_vars.sum(), GRB.MINIMIZE)
        model.optimize()
        return np.array(model.getAttr("Pi", model.getConstrs()))

    def _solve_ip_master_(self, active_patterns):
        model = grb.Model()
        model.setParam('OutputFlag', 0)
        allocation_vars = model.addMVar(shape = active_patterns.shape[1], lb = 0, vtype = GRB.INTEGER)
        model.addConstr(active_patterns @ allocation_vars >= self.required_quantities)
        model.setObjective(allocation_vars.sum(), GRB.MINIMIZE)
        model.optimize()
        return model.objVal, np.array(model.getAttr("X"))

    def _find_new_pattern_(self, dual_values, stockpile):
        model = grb.Model()
        model.setParam('OutputFlag', 0)
        decision_vars = model.addMVar(shape = len(self.item_dimensions), lb = 0, vtype = GRB.INTEGER)

        model.addConstr((self.item_dimensions[:, 0] @ decision_vars) <= self.stock_width)
        model.addConstr((self.item_dimensions[:, 1] @ decision_vars) <= self.stock_height)

        model.setObjective(1 - dual_values @ decision_vars, GRB.MINIMIZE)
        model.optimize()

        if model.objVal < 0:
            return True, np.array(model.getAttr("X"))
        else:
            return False, None

    def _determine_action_(self, stockpile, cut_solution):
        stock_idx = 0
        while stock_idx < len(stockpile):
            stock = stockpile[stock_idx]
            stock_w, stock_h = self._get_stock_size_(stock)
            for pattern, qty in zip(cut_solution["cut_patterns"].T, cut_solution["optimal_numbers"]):
                if qty > 0:
                    for prod_idx, prod_count in enumerate(pattern):
                        if prod_count > 0:
                            prod_dim = self.item_dimensions[prod_idx]
                            if stock_w >= prod_dim[0] and stock_h >= prod_dim[1]:
                                attempts = 0
                                while attempts < 100:
                                    x_pos = random.randint(0, stock_w - prod_dim[0])
                                    y_pos = random.randint(0, stock_h - prod_dim[1])
                                    if self._can_place_(stock, (x_pos, y_pos), prod_dim):
                                        return {"stock_idx": stock_idx, "size": prod_dim, "position": (x_pos, y_pos)}
                                    attempts += 1
            stock_idx += 1
        return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}