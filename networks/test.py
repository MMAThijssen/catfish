#from learning_curve import draw_learning_curves

t_score = [0.53085935, 0.58493304]
v_score = [0.53446553446553, 0.7353646353646354]
t1_score = [0.53085935, 0.58493304]
v1_score = [0.53446553446553, 0.7353646353646354]
sizes = [1000, 10000]

samples = []
samples.append(t_score)
samples.append(t1_score)
print(samples)

#for s in range(len(sizes)):
#    mean_training_score = train_error[s].mean()
#    mean_validation_score = val_error[s].mean() 
# 
min_list = []
max_list = []   
mean_list = []


for s in range(len(sizes)):
    total = 0
    for i in range(len(samples)):
        total += samples[i][s]
    mean = total / len(samples)
    mean_list.append(mean)
    
print(mean_list)
        