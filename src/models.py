import numpy as np
from scipy.optimize import minimize, differential_evolution
from pymoo.core.problem import ElementwiseProblem

class RiskBudgetingBRKGA(ElementwiseProblem):
    def __init__(self, cov_matrix, k_cardinality, formulation='convex', solver_method='LBFGSB', solver_tol=1e-6, solver_maxiter=100):
        self.cov_matrix = cov_matrix
        self.k = k_cardinality
        self.formulation = formulation
        self.solver_method = solver_method
        self.solver_tol = solver_tol
        self.solver_maxiter = solver_maxiter
        self.n_assets = cov_matrix.shape[0]
        self.b_target = np.ones(self.k) / self.k
        super().__init__(n_var=self.n_assets, n_obj=1, xl=0.0, xu=1.0)

    def _decode(self, x):
        return np.argsort(x)[::-1][:self.k]

    def _evaluate(self, x, out, *args, **kwargs):
        selected_assets = self._decode(x)
        sub_cov = self.cov_matrix[np.ix_(selected_assets, selected_assets)]
        
        if self.formulation == 'convex':
            y0 = np.ones(self.k) / self.k
            bounds = tuple((1e-8, None) for _ in range(self.k))
            res = minimize(self._obj_convex, y0, args=(sub_cov, self.b_target), 
                           method='L-BFGS-B', bounds=bounds, 
                           tol=self.solver_tol, options={'maxiter': self.solver_maxiter})
            pesos = res.x / np.sum(res.x)
            out["F"] = np.sqrt(pesos.T @ sub_cov @ pesos)
        else:
            bounds = tuple((0.0, 1.0) for _ in range(self.k))
            if self.solver_method == 'SLSQP':
                x0 = np.ones(self.k) / self.k
                constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
                res = minimize(self._obj_non_convex, x0, args=(sub_cov, self.b_target), 
                               method='SLSQP', bounds=bounds, constraints=constraints, 
                               tol=self.solver_tol, options={'maxiter': self.solver_maxiter})
                wn = res.x / np.sum(res.x)
            else: # DE
                res = differential_evolution(self._obj_non_convex, bounds, args=(sub_cov, self.b_target), 
                                             tol=self.solver_tol, maxiter=self.solver_maxiter, popsize=5)
                wn = res.x / np.sum(res.x)
            out["F"] = np.sqrt(wn.T @ sub_cov @ wn)

    def _obj_convex(self, y, cov, b):
        return 0.5 * (y.T @ cov @ y) - np.sum(b * np.log(y))

    def _obj_non_convex(self, w, cov, b):
        w_sum = np.sum(w)
        if w_sum <= 1e-10: return float('inf')
        wn = w / w_sum
        risk = wn.T @ cov @ wn
        if risk <= 1e-10: return float('inf')
        rc = wn * (cov @ wn)
        return np.sum(((rc / risk) - b)**2) * 1e6

class MaximumSharpeBRKGA(ElementwiseProblem):
    def __init__(self, retornos_medios, cov_matrix, rf, k_cardinality, solver_tol=1e-6, solver_maxiter=100):
        self.ret_medios = retornos_medios
        self.cov_matrix = cov_matrix
        self.rf = rf
        self.k = k_cardinality
        self.solver_tol = solver_tol
        self.solver_maxiter = solver_maxiter
        self.n_assets = cov_matrix.shape[0]
        super().__init__(n_var=self.n_assets, n_obj=1, xl=0.0, xu=1.0)

    def _decode(self, x):
        return np.argsort(x)[::-1][:self.k]

    def _evaluate(self, x, out, *args, **kwargs):
        selected = self._decode(x)
        sub_cov = self.cov_matrix[np.ix_(selected, selected)]
        sub_ret = self.ret_medios[selected]
        
        x0 = np.ones(self.k) / self.k
        bounds = tuple((0.0, 1.0) for _ in range(self.k))
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        
        # Minimiza o negativo do Sharpe
        res = minimize(self._neg_sharpe, x0, args=(sub_ret, sub_cov, self.rf), 
                       method='SLSQP', bounds=bounds, constraints=constraints, 
                       tol=self.solver_tol, options={'maxiter': self.solver_maxiter})
        
        out["F"] = res.fun # Fitness é o negativo do Sharpe (pymoo sempre minimiza)

    def _neg_sharpe(self, w, ret, cov, rf):
        port_ret = np.dot(w, ret)
        port_vol = np.sqrt(w.T @ cov @ w)
        if port_vol <= 1e-10: return float('inf')
        return - (port_ret - rf) / port_vol

def naive_1_k_allocation(retornos_medios, cov_matrix, rf, k_cardinality):
    volatilidades = np.sqrt(np.diag(cov_matrix))
    # Sharpe individual
    sharpes = np.where(volatilidades > 0, (retornos_medios - rf) / volatilidades, -np.inf)
    
    # Seleciona os K melhores índices
    top_k_indices = np.argsort(sharpes)[::-1][:k_cardinality]
    
    pesos = np.zeros(len(retornos_medios))
    pesos[top_k_indices] = 1.0 / k_cardinality
    return top_k_indices, pesos
