
from learner.learn import Learn
from learner.constants import AdaptationLevel

model_learner = Learn()
model_learner.get_true_model()
model_learner.start_learning()
if model_learner.ready.get_baseline() == AdaptationLevel.BASELINE_C:
    model_learner.dump_learned_model()

model_learner.update_config_files()

#if model_learner.ready.get_baseline() == AdaptationLevel.BASELINE_D:
#    model_learner.start_online_learning()
#    model_learner.update_config_files()


