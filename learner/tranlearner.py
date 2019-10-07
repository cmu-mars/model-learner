
class TranLearner:

    def __init__(self, budget, offline_func, online_func, cost_ratio):

        self.offline_budget_ratio = 0.2
        self.offline_budget = int(budget * self.offline_budget_ratio) 
        self.online_budget  = budget - self.offline_budget

        self.offline_func   = offline_func
        self.online_func    = online_func
        # cost_ratio: offline_func / online_func
        self.cost_ratio     = cost_ratio

        self.used_budget    = 0
        self.measurement    = None
        # try to use an appropriate number
        self.online_budget_per_call = min(200, int(self.online_budget*0.1))

        # Note: Ensure suggorate_model has a predict function
        #       Check update_config_files() in learner.py 
        self.suggorate_model = None

    def offline_learning(self):

        # ToDo: offline learning process
        #       Update self.suggorate_model

        # Update budget information
        self.used_budget    = int(self.offline_budget * self.cost_ratio)
        self.offline_budget = 0
        
        # Update self.measurements

        return self.suggorate_model

    def online_learning(self):
        budget = self.online_budget_per_call
        if self.online_budget < self.online_budget_per_call:
            budget = self.online_budget

        # ToDo: online learning process
        #       Update self.suggorate_model

        # Update budget information
        self.used_budget    += budget
        self.online_budget  -= budget
        
        # Update self.measurements

        return self.suggorate_model

    def get_pareto_frontier(self, Xs, Ys, maxX=True, maxY=True):
        # Sort the list in either ascending or descending order of X
        myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
        idx_sorted = sorted(range(len(Xs)), key=lambda k: Xs[k])
        # Start the Pareto frontier with the first value in the sorted list
        p_front = [myList[0]]
        i = 0
        pareto_idx = [idx_sorted[i]]
        # Loop through the sorted list
        for pair in myList[1:]:
            i += 1
            if maxY:
                if pair[1] >= p_front[-1][1]:
                    p_front.append(pair)
                    pareto_idx.append(idx_sorted[i])
            else:
                if pair[1] <= p_front[-1][1]:
                    p_front.append(pair)
                    pareto_idx.append(idx_sorted[i])
        p_frontX = [pair[0] for pair in p_front]
        p_frontY = [pair[1] for pair in p_front]
        return pareto_idx, p_frontX, p_frontY


