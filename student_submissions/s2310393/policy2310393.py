import random
import numpy as np
from scipy.optimize import linprog
from policy import Policy

class Policy2310393(Policy):
    def __init__(self):
        self.stock_width = 25
        self.stock_height = 25
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
        self.starting_cut_patterns = np.eye(total_items, dtype=int)

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
        num_vars = active_patterns.shape[1]

        c = np.ones(num_vars)

        A = -active_patterns
        b = -self.required_quantities

        bounds = [(0, None) for _ in range(num_vars)]
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

        if res.success:
            return res.slack  # Or modify as needed for interpreting dual values
        else:
            raise ValueError("Linear programming problem could not be solved.")


    def _solve_ip_master_(self, active_patterns):
        num_vars = active_patterns.shape[1]
        
        c = np.ones(num_vars)
        A = -active_patterns
        b = -self.required_quantities
        
        bounds = [(0, None) for _ in range(num_vars)]
        
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        
        if res.success:
            allocation_vars = np.round(res.x).astype(int)  # Round to integers
            obj_val = allocation_vars.sum()
            return obj_val, allocation_vars
        else:
            raise ValueError("Integer programming problem could not be solved.")
    
    
    def _find_new_pattern_(self, dual_values, stockpile):
        num_vars = len(self.item_dimensions)

        c = dual_values - 1  
        A = [
            self.item_dimensions[:, 0],  
            self.item_dimensions[:, 1]  
        ]
        b = [self.stock_width, self.stock_height]
        

        bounds = [(0, None) for _ in range(num_vars)]
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        
        if res.success:
            decision_vars = np.round(res.x).astype(int)  # Round to integers
            obj_val = 1 - dual_values @ decision_vars
            if obj_val < 0:
                return True, decision_vars
            else:
                return False, None
        else:
            raise ValueError("Pattern finding problem could not be solved.")
    

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
                                for x_pos in range(stock_w - prod_dim[0] + 1):
                                    for y_pos in range(stock_h - prod_dim[1] + 1):
                                        if self._can_place_(stock, (x_pos, y_pos), prod_dim):
                                            return {"stock_idx": stock_idx, "size": prod_dim, "position": (x_pos, y_pos)}
            stock_idx += 1
        return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}
