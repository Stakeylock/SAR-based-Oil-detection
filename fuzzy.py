import numpy as np
from skfuzzy import control as ctrl
from skfuzzy import membership as mf

def setup_fuzzy_logic():
    area = ctrl.Antecedent(np.arange(0, 10000, 1), 'area')
    confidence = ctrl.Antecedent(np.arange(0, 1, 0.01), 'confidence')
    spill_likelihood = ctrl.Consequent(np.arange(0, 1, 0.01), 'spill_likelihood')

    area['small'] = mf.trapmf(area.universe, [0, 0, 500, 1000])
    area['medium'] = mf.trimf(area.universe, [500, 1500, 3000])
    area['large'] = mf.trapmf(area.universe, [2000, 3000, 5000, 10000])

    confidence['low'] = mf.trapmf(confidence.universe, [0, 0, 0.3, 0.5])
    confidence['medium'] = mf.trimf(confidence.universe, [0.3, 0.5, 0.7])
    confidence['high'] = mf.trapmf(confidence.universe, [0.5, 0.7, 0.9, 1])

    spill_likelihood['unlikely'] = mf.trimf(spill_likelihood.universe, [0, 0, 0.5])
    spill_likelihood['likely'] = mf.trimf(spill_likelihood.universe, [0, 0.5, 1])
    spill_likelihood['very likely'] = mf.trimf(spill_likelihood.universe, [0.5, 0.9, 1])

    rule1 = ctrl.Rule(area['small'] & confidence['low'], spill_likelihood['unlikely'])
    rule2 = ctrl.Rule(area['small'] & confidence['medium'], spill_likelihood['unlikely'])
    rule3 = ctrl.Rule(area['small'] & confidence['high'], spill_likelihood['likely'])
    rule4 = ctrl.Rule(area['medium'] & confidence['low'], spill_likelihood['unlikely'])
    rule5 = ctrl.Rule(area['medium'] & confidence['medium'], spill_likelihood['likely'])
    rule6 = ctrl.Rule(area['medium'] & confidence['high'], spill_likelihood['very likely'])
    rule7 = ctrl.Rule(area['large'] & confidence['low'], spill_likelihood['likely'])
    rule8 = ctrl.Rule(area['large'] & confidence['medium'], spill_likelihood['very likely'])
    rule9 = ctrl.Rule(area['large'] & confidence['high'], spill_likelihood['very likely'])

    spill_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    spill = ctrl.ControlSystemSimulation(spill_ctrl)

    return spill

def calculate_fuzzy_likelihood(area, confidence, fuzzy_ctrl):
    area = min(max(area, 0), 10000)
    confidence = min(max(confidence, 0), 1)

    fuzzy_ctrl.input['area'] = area
    fuzzy_ctrl.input['confidence'] = confidence
    fuzzy_ctrl.compute()

    return fuzzy_ctrl.output['spill_likelihood']