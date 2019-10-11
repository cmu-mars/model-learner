import GPy
import GPyOpt
import numpy as np

from  maxUncerntainty import AcquisitionMU


class TranLearner:

    def __init__(self, budget, num_dims, power_model_filepath):

        self.offline_budget_ratio = 0.5
        self.offline_budget = int(budget * self.offline_budget_ratio)
        if self.offline_budget > 200:
            self.offline_budget = 200

        self.online_budget  = budget - self.offline_budget
        self.used_budget    = 0

        self.num_dims = num_dims
        self.opIDs = {}
        for i in range(self.num_dims):
            self.opIDs['o'+str(i)]=i


        self.powerModel = self.loadPowerModel(power_model_filepath)
        self.domain = self.createDomain(self.num_dims, "discrete"  , (0, 1))

        self.bo = None
        # Note: Ensure suggorate_model has a predict function
        #       Check update_config_files() in learner.py 
        self.suggorate_model = None


    # parameter:
    #   domain_type: continuous or discrete
    #   domain_1d  : (lower, upper) or (num1, num2, ..., numN)
    def createDomain(self, num_dims, domain_type, domain_1d):
        domain = []
        for i in range(num_dims):
            domain.append({'name':'x_'+str(i+1), 'type':domain_type, 'domain': domain_1d})
        return domain

    def uniSampleDomain(self, domain, num_points):
        num_dims = len(domain)
        X = np.zeros((num_points, num_dims))
        for dim_idx in range(num_dims):
            d = domain[dim_idx]
            if d['type'] == 'continuous':
                X[:, dim_idx] = np.random.uniform(d['domain'][0], d['domain'][1], size=num_points)
            elif d['type'] == 'discrete':
                X[:, dim_idx] = np.random.choice(d['domain'], size=num_points)
            else:
                raise ValueError("Unsupported variable type: {}".format(d['type'])) 
        return X



    # functionality: load power model from text file
    #   output:
    #       model: a dict where each element is a pair of (key=a polynomal term represented in a tuple, weight)
    #              The constant term       : model[(-1)] = weight1
    #              Single option term      : model[(opID)] = weight2
    #              Interacting options term: model[(opID1, opID2,...)] = weightt3
    def loadPowerModel(self, filepath):
        model={tuple([-1]):0}
        with open(filepath) as f:
            model_txt = f.read(); # there is only one line in the model file
            terms=model_txt.replace(" ", "").rstrip().split("+")
            for term in terms:
                parts=term.split("*")
                if len(parts)==1: # It is a constant. Assume there are at most 1 constant
                    model[tuple([-1])] = float(parts[0])
                else:
                    weight=parts[0]
                    opList=parts[1:]
                    opListNumericalIDs=[]
                    for op in opList:
                         opListNumericalIDs.append(self.opIDs[op])
                    model[tuple(opListNumericalIDs)]=float(weight)
        return model


    # functionality: get the performance of the power model under the given configuration
    # input:
    #   config: 2D numpy array where each row is an array of 20 elements, which represets a point
    # output:
    #   the performance of the power model
    def measurePM(self, configs):
        num_points = configs.shape[0]
        perfs = np.zeros((num_points, 1))

        for p_idx in range(num_points):
            config = configs[p_idx, :]
            perf = 0
            for key, value in self.powerModel.items():
                if -1 not in key: # if the key is not the constant term
                    for ID in key:
                        if config[ID] == 0:
                            value = 0
                            break
                perf += value
            perfs[p_idx][0] = perf
        return perfs

    def create_bo(self, model_update_interval, X_init, Y_init):
        # Initialize Bayesian optimization


        # --- feasible region
        space = GPyOpt.Design_space(self.domain)

        # --- CHOOSE the objective
        objective = GPyOpt.core.task.SingleObjective(self.measurePM)

        # --- CHOOSE the model type
        model = GPyOpt.models.GPModel(exact_feval=True,optimize_restarts=10,verbose=False)

        # --- CHOOSE the acquisition optimizer
        aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space)

        # --- CHOOSE the type of acquisition
        acquisition = AcquisitionMU(model, space, optimizer=aquisition_optimizer)

        # --- CHOOSE a collection method
        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

        # Bo object
        bo = GPyOpt.methods.ModularBayesianOptimization(
                model,
                space,
                objective,
                acquisition,
                evaluator,
                X_init,
                Y_init = Y_init,
                normalize_Y = False,
                model_update_interval = model_update_interval)

        return bo

    def offline_learning(self):

        # Map the actual budget that can be used in the offline learning
        # Because the units between offline learning and online learning are different
        # Offline learning cost is consider cheaper than the online learning cost.
       
        num_init = 0
        model_update_interval = 1
        budget = 0
        if self.offline_budget < 2:
            raise ValueError(f"The offline budget ({self.offline_budget}) < 2.")
        elif self.offline_budget <= 10:
            num_init = 2
        else:
            num_init = 10
            interval = (self.offline_budget - 10) // 10
            if interval > 1:
                model_update_interval = interval
        budget = self.offline_budget - num_init

        print(f"Offline learning budget: {self.offline_budget}")
        print(f"----Run Bayesian Optimization with {num_init} initial points and {budget} iterations")

        X_init = self.uniSampleDomain(self.domain, num_init)
        Y_init = self.measurePM(X_init)

        self.bo = self.create_bo(model_update_interval, X_init, Y_init)
        self.bo.run_optimization(budget)
        self.suggorate_model = self.bo.model

        # Update budget information
        self.used_budget    += self.offline_budget
        self.offline_budget = 0
        
        return self.suggorate_model

    def online_learning(self):
        budget = 25
        if self.online_budget < 25:
            budget  = self.online_budget
        model_update_interval = budget

        print(f"online learning started with the budget: {budget}")

        self.bo = self.create_bo(
                model_update_interval,
                self.bo.X,
                self.bo.Y)
        self.bo.run_optimization(budget)
        self.suggorate_model = self.bo.model

        # Update budget information
        self.used_budget    += budget
        self.online_budget  -= budget

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


