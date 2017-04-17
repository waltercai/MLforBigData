import math
import numpy as np
import matplotlib.pyplot as plt

from analysis.DataSet import DataSet
from util.EvalUtil import EvalUtil

# This class represents the weights in the logistic regression model.
class Weights:
  def __init__(self):
    self.w0 = self.w_age = self.w_gender = self.w_depth = self.w_position = 0
    # token feature weights
    self.w_tokens = {}
    # to keep track of the access timestamp of feature weights.
    #   use this to do delayed regularization.
    self.access_time = {}
    
  def __str__(self):
    formatter = "{0:.2f}"
    string = ""
    string += "Intercept: " + formatter.format(self.w0) + "\n"
    string += "Depth: " + formatter.format(self.w_depth) + "\n"
    string += "Position: " + formatter.format(self.w_position) + "\n"
    string += "Gender: " + formatter.format(self.w_gender) + "\n"
    string += "Age: " + formatter.format(self.w_age) + "\n"
    string += "Tokens: " + str(self.w_tokens) + "\n"
    return string
  
  # @return {Double} the l2 norm of this weight vector
  def l2_norm(self):
    l2 = self.w0 * self.w0 +\
          self.w_age * self.w_age +\
          self.w_gender * self.w_gender +\
          self.w_depth * self.w_depth +\
          self.w_position * self.w_position
    for w in self.w_tokens.values():
      l2 += w * w
    return math.sqrt(l2)
  
  # @return {Int} the l2 norm of this weight vector
  def l0_norm(self):
    return 4 + len(self.w_tokens)


class LogisticRegression:
  # ==========================
  # Helper function to compute inner product w^Tx.
  # @param weights {Weights}
  # @param instance {DataInstance}
  # @return {Double}
  # ==========================
  def compute_weight_feature_product(self, weights, instance):
    # TODO: Fill in your code here
    wtx = 0.0 \
          + instance.depth * weights.w_depth \
          + instance.position * weights.w_position \
          + instance.age * weights.w_age \
          + instance.gender * weights.w_gender \
          + sum([weights.w_tokens.get(t, 0.0) for t in instance.tokens])
    return wtx
  
  # ==========================
  # Apply delayed regularization to the weights corresponding to the given
  # tokens.
  # @param tokens {[Int]} list of tokens
  # @param weights {Weights}
  # @param now {Int} current iteration
  # @param step {Double} step size
  # @param lambduh {Double} lambda
  # ==========================
  def perform_delayed_regularization(self, tokens, weights, now, step, lambduh):
    # TODO: Fill in your code here
    scale = (1 - step * lambduh)
    for t in tokens:
      past_access = weights.access_time.get(t,0)
      
      if t in weights.w_tokens:
        weights.w_tokens[t] *= scale**(now - past_access - 1)
      else:
        weights.w_tokens[t] = 0.0

      weights.access_time[t] = now

    return

  # ==========================
  # Train the logistic regression model using the training data and the
  # hyperparameters. Return the weights, and record the cumulative loss.
  # @return {Weights} the final trained weights.
  # ==========================
  def train(self, dataset, lambduh, step, avg_loss):
    color = {0.001:'b',0.01:'g',0.05:'r'}

    weights = Weights()
    count = 0
    loss = 0.0
    loss_data = []
    # TODO: Fill in your code here. The structure should look like:
    # For each data point:
    while dataset.hasNext():
      instance = dataset.nextInstance()
    
      # Your code: perform delayed regularization
      self.perform_delayed_regularization(tokens=instance.tokens,
                                          weights=weights,
                                          now=count,
                                          step=step,
                                          lambduh=lambduh)

      # Your code: predict the label, record the loss
      # Your code: compute w0 + <w, x>, and gradient
      pyx = 1.0 - 1.0 / (1 + np.exp(weights.w0 + self.compute_weight_feature_product(weights, instance)))
      ympyx = instance.clicked - pyx
      loss += (ympyx)**2
      if count % 100 == 0 and count > 0:
        loss_data.append(loss / count)
        
      # Your code: update weights along the negative gradient
      for t in instance.tokens:
        w = weights.w_tokens.get(t,0.0)
        weights.w_tokens[t] = w + step * ympyx - w * lambduh * step

      weights.w0 += step * ympyx
      
      weights.w_depth += step * (instance.depth * ympyx - weights.w_depth * lambduh)
      weights.w_position += step * (instance.position * ympyx - weights.w_position * lambduh)
      weights.w_age += step * (instance.age * ympyx - weights.w_age * lambduh)
      weights.w_gender += step * (instance.gender * ympyx - weights.w_gender * lambduh)

      update_frac = 5
      if count % (dataset.size/update_frac) == 0:
        print("processed {}% of training data".format((count * 100)/dataset.size))
      count += 1


    ###########################
    self.perform_delayed_regularization(tokens=weights.w_tokens.keys(),
                                        weights=weights,
                                        now=count,
                                        step=step,
                                        lambduh=lambduh)
    
    dataset.reset()

    plt.plot(range(100,count, 100), loss_data, color[step], label="eta = {}".format(step))
    # plt.savefig('average_loss_{}_unreg.png'.format(step))
    return weights

  # ==========================
  # Use the weights to predict the CTR for a dataset.
  # @param weights {Weights}
  # @param dataset {DataSet}
  # ==========================
  def predict(self, weights, dataset):
    # TODO: Fill in your code here
    ctr_data = []
    count = 0
    while dataset.hasNext():

      instance = dataset.nextInstance()
      pyx = 1.0 - 1.0 / (1 + np.exp(weights.w0 + self.compute_weight_feature_product(weights, instance)))
      # ctr_sum += pyx
      ctr_data.append(pyx)

      update_frac = 5
      if count % (dataset.size/update_frac) == 0:
        print("processed {}% of testing data".format((count * 100)/dataset.size))
      count += 1
    
    dataset.reset()

    return ctr_data
  
  
if __name__ == '__main__':
  # TODO: Fill in your code here
  print("Training Logistic Regression...")
  lg = LogisticRegression()
  train_size = DataSet.TRAININGSIZE
  test_size = DataSet.TESTINGSIZE

  training = DataSet("../data/train.txt", True, train_size)
  testing = DataSet("../data/test.txt", False, test_size)

  # plt.axis((0,2500000,0.04,0.10))

  # # UNREGULARIZED

  # for step in [0.001,0.01,0.05]:
  #   weights = lg.train(dataset=training, lambduh=0.0, step=step, avg_loss=0.0)
  #   ctr_predict = lg.predict(weights=weights, dataset=testing)
  #   print("L2 Norm: {}".format(weights.l2_norm()))
  #   print("depth weight: {}".format(weights.w_depth))
  #   print("position weight: {}".format(weights.w_position))
  #   print("age weight: {}".format(weights.w_age))
  #   print("gender weight: {}".format(weights.w_gender))
  #   print("")
  #   print("Average CTR testing prediction: {}".format(np.mean(ctr_predict)))
  #   rmse = EvalUtil.eval_baseline("../data/test_label.txt", np.mean(ctr_predict))
  #   print("Testing RMSE : {}".format(rmse))

  #   print("------------------------------------------")

  # plt.title("Average Loss over Step-Count")
  # plt.xlabel('Step Count')
  # plt.ylabel('Loss')
  # plt.legend(loc='upper right')
  # plt.savefig('average_loss_unreg.png'.format(step))

  # REGULARIZED

  step = 0.05
  lambduh_range = np.arange(0.002, 0.015, 0.002)
  l2_norms = []
  rmses = []
  for lambduh in lambduh_range:
    weights = lg.train(dataset=training, lambduh=lambduh, step=step, avg_loss=0.0)
    
    l2_norm = weights.l2_norm()
    l2_norms.append(l2_norm)

    ctr_predict = lg.predict(weights=weights, dataset=testing)
    rmse = EvalUtil.eval(path_to_sol="../data/test_label.txt", ctr_predictions=ctr_predict)
    rmses.append(rmse)
    print("{}, {}".format(lambduh, l2_norm))
    print("------------------------------------------")

  print("lambda range: {}".format(lambduh_range))
  print("l2 norms: {}".format(l2_norms))
  print("RMSE: {}".format(rmses))

  print "==================="

  plt.plot(lambduh_range, l2_norms)
  plt.title("Weights L2 Norm over Lambda")
  plt.xlabel('Lambda')
  plt.ylabel('L2 Norm')
  plt.savefig('l2_norm_lambda_reg.png')

  plt.clf()

  plt.plot(lambduh_range, rmses)
  plt.title("Testing RMSE over Lambda")
  plt.xlabel('Lambda')
  plt.ylabel('RMSE')
  plt.savefig('rmse_lambda_reg.png')


