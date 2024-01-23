def custom_loss_(re, Target, derivativeE, derivativeI):
  Cost = torch.tensor(0.0)
  
  #1/ Proba term and neg log likelihood
  Pred = make_it_proba(re)
  loss_proba = torch.zeros((Pred.shape))
  for t in range(Pred.shape[0]):
    target_pop_long = Target.select(0,t).long() 
    log_pred_pop = torch.log(Pred.select(0,t)) 
    # Calculate loss by comparing the distribution to the expected probabilities
    loss_proba[t,:] = F.nll_loss(log_pred_pop, target_pop_long) #neg log likelihood

  # Sum over time step
  total_loss_proba = torch.sum(loss_proba)
  
  
  #2/ derivative good 
  stimu_pop= torch.argmax(Target).item()
  ### for excitatory neurons # targe = -derivative 
  tensor_for_cost_derivative_E = derivativeE.clone() 
  tensor_for_cost_derivative_E[:,stimu_pop] = torch.neg(derivativeE.select(1, stimu_pop)) #we want it to be negative for the stimulated population

  ### for inhibitory neurons
  tensor_for_cost_derivative_I = derivativeI.clone()
  tensor_for_cost_derivative_I = torch.neg(derivativeI)
  tensor_for_cost_derivative_I[:, stimu_pop] = torch.neg(derivativeI.select(1, stimu_pop)) #[:,stimu_pop]) #tensor_for_cost_derivativeI = [i for i in derivativeI if i not in target_arr]

  """loss_derivative = torch.add(torch.sum(torch.mean(F.softplus(tensor_for_cost_derivative_E)**2, axis = 1)), #mean over popululation #then sum over every time step
                              torch.sum(torch.mean(F.softplus(tensor_for_cost_derivative_I)**2, axis = 1)))"""
  #mean instead of sum over time step
  loss_derivative = torch.add(torch.mean(torch.mean(F.softplus(tensor_for_cost_derivative_E)**2, axis = 1)), #mean over popululation #then sum over every time step
                              torch.mean(torch.mean(F.softplus(tensor_for_cost_derivative_I)**2, axis = 1)))
  


  #3/ L2 regu term
  l2_reg_coef= 0.001# 0.0001
  l2_reg = 0.0
  for param in mymodel.parameters():
      l2_reg += (param**2)
  
  #4/ High value of re term
  high_activity_penalty_coef=0.01#0.0001
  high_activity_penalty = torch.sum(torch.sum(re[:, stimu_pop]**2))

  #5/ Combining everything
  Cost = torch.add(loss_derivative, total_loss_proba) 
  Cost_f = Cost + (l2_reg * l2_reg_coef)
  Cost_f = Cost_f + (high_activity_penalty * high_activity_penalty_coef)
  return Cost_f



print(custom_loss_(r_e, Y_target, dredt, dridt))