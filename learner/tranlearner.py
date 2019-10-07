
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
