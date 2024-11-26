import gurobipy as grb
from gurobipy import GRB
import numpy as np
import random
from policy import Policy

class Policy2310393(Policy):
    def __init__(self):
        self.stock_width = 20
        self.stock_height = 20
        self.initial_pattern = None
        self.required_quantities = None

    def get_action(self, observation, info):
        items = observation["products"]
        stocks = observation["stocks"]

        # Initialize item data
        self.required_quantities = np.array([prod["quantity"] for prod in items])
        self.item_dimensions = np.array([prod["size"] for prod in items])
        total_items = len(self.item_dimensions)
        self.initial_pattern = np.eye(total_items, dtype = int)

        # Column generation
        cut_solution = self.column_generation(stocks)

        # Choose an action
        move = self.select_action(stocks, cut_solution)
        return move

    def column_generation(self, stocks):
        is_new_pattern = True
        new_pattern = None
        active_patterns = self.initial_pattern

        while is_new_pattern:
            if new_pattern is not None:
                active_patterns = np.column_stack((active_patterns, new_pattern))

            dual_values = self.solve_lp(active_patterns)
            is_new_pattern, new_pattern = self.find_new_pattern(dual_values, stocks)

        optimal_stock_count, optimal_solution = self.solve_ip(active_patterns)
        return {"cut_patterns": active_patterns, "minimal_stock": optimal_stock_count, "optimal_numbers": optimal_solution}

    def solve_lp(self, active_patterns):
        model = grb.Model()
        model.setParam('OutputFlag', 0)
        allocation_vars = model.addMVar(shape = active_patterns.shape[1], lb = 0, vtype = GRB.CONTINUOUS)
        model.addConstr(active_patterns @ allocation_vars >= self.required_quantities)
        model.setObjective(allocation_vars.sum(), GRB.MINIMIZE)
        model.optimize()
        return np.array(model.getAttr("Pi", model.getConstrs()))

    def solve_ip(self, active_patterns):
        model = grb.Model()
        model.setParam('OutputFlag', 0)
        allocation_vars = model.addMVar(shape = active_patterns.shape[1], lb = 0, vtype = GRB.INTEGER)
        model.addConstr(active_patterns @ allocation_vars >= self.required_quantities)
        model.setObjective(allocation_vars.sum(), GRB.MINIMIZE)
        model.optimize()
        return model.objVal, np.array(model.getAttr("X"))

    def find_new_pattern(self, dual_values, stocks):
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

    def select_action(self, stockpile, cut_solution):
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
                                # First-fit
                                for x_pos in range(stock_w - prod_dim[0] + 1):
                                    for y_pos in range(stock_h - prod_dim[1] + 1):
                                        if self._can_place_(stock, (x_pos, y_pos), prod_dim):
                                            return {"stock_idx": stock_idx, "size": prod_dim, "position": (x_pos, y_pos)}
            stock_idx += 1
        return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}