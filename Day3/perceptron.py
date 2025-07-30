#Single Layer Perceptron

x_input=[0.1,0.3,0.4]
weights=[0.4,0.2,0.3]
thres=0.5

def step_func(weighted_sum):
    if weighted_sum > thres:
        return 1
    else:
        return 0
    

def perceptron():
    weighted_sum=0
    for x,w in zip(x_input,weights):
      weighted_sum += x*w
    print(weighted_sum)
    return step_func(weighted_sum)

output=perceptron()
print(output)


#Multilayer Perceptron



