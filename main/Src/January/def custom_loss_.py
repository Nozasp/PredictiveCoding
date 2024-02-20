def custom_loss_(Prediction, Target, derivativeE):
  #1/ Proba term
  criterion = torch.nn.NLLLoss()
  P2 = make_it_proba(Prediction) #
  loss_norm = criterion(P2, Target.long())

  #2/ derivative good 
  eps = torch.FloatTensor([sys.float_info.epsilon]) 
  Target_idx_np = (Target).numpy().astype(int)
  loss_derivative = torch.zeros_like(Target)
  
  hyperactvity_penalty = torch.zeros(train_IN.shape[0])
  laziness_penalty = torch.zeros(train_IN.shape[0])
  
  count = 0
  i2 = 0
  start = 0
  stop = (len_sim)
  for i, sti_idx in enumerate(Target_idx_np):
      count +=1
      if count == stop:
          time_idx = slice(start, stop)
          
          
          if torch.max(Prediction[time_idx,sti_idx]) > 60.: #60
              hyperactvity_penalty[i2] = torch.sum((Prediction[time_idx, sti_idx]**2)) #torch.clamp(r_e[:, 7], max=10.0
          else:
              hyperactvity_penalty[i2] = 0
              
          if torch.max(Prediction[time_idx,sti_idx]) < 10.: #60
              laziness_penalty[i2] = torch.sum(1 / torch.clamp(Prediction[time_idx, sti_idx]**2, min = eps)) #torch.clamp(r_e[:, 7], max=10.0

          else:
              laziness_penalty[i2] = 0
 
          i2 += 1
          start = stop
          stop += len_sim


  len_sim_test = len_sim
  count = 0
  start = 0
  for i, sti_idx in enumerate(Target_idx_np):
    count += 1
    if count == len_sim_test:
        loss_derivative[i] = - F.softplus(derivativeE[start:len_sim_test,sti_idx] ).sum() + F.softplus(derivativeE[start:len_sim_test,:sti_idx]).sum() + F.softplus(derivativeE[start:len_sim_test,(sti_idx+1):]).sum()
        #ic(loss_derivative[i])
        start = count
        len_sim_test += len_sim_test


  loss_derivative_term = loss_derivative.mean()
  coef_derivative = 1E-6 #1E-6 #0.00001
  loss_derivative_term = coef_derivative * loss_derivative_term
  
  hyperactvity_penalty_term = hyperactvity_penalty.mean()
  hyperactvity_penalty_coef= 1E-7#1E-6
  hyperactvity_penalty_term = hyperactvity_penalty_term * hyperactvity_penalty_coef
  laziness_penalty = (laziness_penalty.sum() * 0.01)
  #4/ Constraint ws
  wie = list(model.parameters())[0]
  wei = list(model.parameters())[1]
  win = list(model.parameters())[2]
  
  be_positive = 0

  #be_positive = torch.sum(torch.nn.functional.relu(-model.wie))
  be_positive = torch.sum(F.relu(-wie)) +torch.sum(F.relu(- wei)) + torch.sum(F.relu(- win))#""""""


  Cost = loss_norm + loss_derivative_term + be_positive + hyperactvity_penalty_term + laziness_penalty #+ activity_regularization
  #ic(loss_norm, loss_derivative_term, be_positive,hyperactvity_penalty_term, laziness_penalty)#loss_derivative_term, hyperactvity_penalty_term)
  return Cost