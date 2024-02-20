# I cant rememberb this one: 
#if torch.max(Prediction[time_idx,sti_idx]) > 60.:              
#if torch.max(Prediction[time_idx,sti_idx]) < 20.: #aLEJANDRO SUGGEST. 40.hz #10
"""coef_derivative = 1E-6 
hyperactvity_penalty_coef= 1E-8
laziness_penalty = 0.01
be_positive_coef  =  10
sparse_coding_coef = 1E-5 
tau_s = 1 
distance_coef = 1E-2
coef_norm = 1"""


#Works but too extrem 7/2 afternoon 
"""#torch.max(Prediction[time_idx,sti_idx]) > 60.:     
#torch.max(Prediction[time_idx,sti_idx]) < 20
coef_derivative = 1E-6
hyperactvity_penalty_coef= 1E-8
laziness_penalty = (laziness_penalty.sum() * 0.01)
sparse_coding_coef = 1E-4
tau_s = 1
distance_coef = 1E-1
coef_norm = 100
be_positive_coef = 100
"""


# works quite well! stable over 800 epochs: 8/2 morning
"""
if torch.max(Prediction[time_idx,sti_idx]) > 60.: #60
if torch.max(Prediction[time_idx,sti_idx]) < 20.: #aLEJANDRO SUGGEST. 40.hz #10
"""
coef_derivative = 1E-6 
hyperactvity_penalty_coef= 1E-8
laziness_penalty_coef =  0.01
sparse_coding_coef = 2E-2
distance_coef = 1E-1
coef_norm = 100