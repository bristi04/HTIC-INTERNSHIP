def sigmoid(x):
  res=1/(1+np.exp(-x))
  return res

y_true=[1,0,0,1,1,0]
y_pred=np.array([1,1,1,0,0,0])

def cross_entropy(y_true,y_pred):
  y_pred=sigmoid(y_pred)
  loss=np.zeros(len(y_pred))
  for i in range(len(y_pred)):
      loss[i]=-1*(y_true[i]*np.log(y_pred[i])+(1-y_true[i])*np.log(1-y_pred[i]))/len(y_pred)
  return loss

res=cross_entropy(y_true,y_pred)
print(res)

x=np.linspace(0,len(res)-1,len(res))
plt.plot(x,res)
