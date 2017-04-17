import math
import numpy as np

from analysis.DataSet import DataSet
from util.EvalUtil import EvalUtil

class Weights:
  def __init__(self, featuredim):
    self.featuredim = featuredim
    self.w0 = self.w_age = self.w_gender = self.w_depth = self.w_position = 0
    # hashed feature weights
    self.w_hashed_features = [0.0 for _ in range(featuredim)]
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
    string += "Hashed Feature: "
    string += " ".join([str(val) for val in self.w_hashed_features])
    string += "\n"
    return string

  # @return {Double} the l2 norm of this weight vector
  def l2_norm(self):
    l2 = self.w0 * self.w0 +\
          self.w_age * self.w_age +\
          self.w_gender * self.w_gender +\
          self.w_depth * self.w_depth +\
          self.w_position * self.w_position
    for w in self.w_hashed_features:
      l2 += w * w
    return math.sqrt(l2)


class LogisticRegressionWithHashing:
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
          + sum([weights.w_hashed_features[f] * instance.hashed_text_feature[f] for f in instance.hashed_text_feature])
    return wtx
  
  # ==========================
  # Apply delayed regularization to the weights corresponding to the given
  # features.
  # @param featureids {[Int]} list of feature ids
  # @param weights {Weights}
  # @param now {Int} current iteration
  # @param step {Double} step size
  # @param lambduh {Double} lambda
  # ==========================
  def perform_delayed_regularization(self, featureids, weights, now, step, lambduh):
    # TODO: Fill in your code here
    scale = (1 - step * lambduh)
    for f in featureids:
      past_access = weights.access_time.get(f,0)
      
      weights.w_hashed_features[f] *= scale**(now - past_access - 1)

      weights.access_time[f] = now

    return
  
  # ==========================
  # Train the logistic regression model using the training data and the
  # hyperparameters. Return the weights, and record the cumulative loss.
  # @return {Weights} the final trained weights.
  # ==========================
  def train(self, dataset, dim, lambduh, step, avg_loss, personalized):
    weights = Weights(featuredim=dim)
    count = 0
    # TODO: Fill in your code here. The structure should look like:
    # For each data point:
    while dataset.hasNext():
      instance = dataset.nextHashedInstance(featuredim=dim, personal=personalized)
    
      # Your code: perform delayed regularization
      self.perform_delayed_regularization(featureids=instance.hashed_text_feature,
                                          weights=weights,
                                          now=count,
                                          step=step,
                                          lambduh=lambduh)

      # Your code: predict the label, record the loss
      pyx = 1.0 - 1.0 / (1 + np.exp(weights.w0 + self.compute_weight_feature_product(weights, instance)))
      ympyx = instance.clicked - pyx

      # Your code: compute w0 + <w, x>, and gradient
        
      # Your code: update weights along the negative gradient
      for f in instance.hashed_text_feature:
        weights.w_hashed_features[f] += step * (instance.hashed_text_feature[f] * ympyx - lambduh * weights.w_hashed_features[f])

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
    self.perform_delayed_regularization(featureids=range(dim),
                                        weights=weights,
                                        now=count,###########################
                                        step=step,
                                        lambduh=lambduh)
    
    dataset.reset()

    return weights

  # ==========================
  # Use the weights to predict the CTR for a dataset.
  # @param weights {Weights}
  # @param dataset {DataSet}
  # @param personalized {Boolean}
  # ==========================
  def predict(self, weights, dataset, personalized):
    # TODO: Fill in your code here
    ctr_data = []
    count = 0
    dim = len(weights.w_hashed_features)

    while dataset.hasNext():
      instance = dataset.nextHashedInstance(featuredim=dim, personal=personalized)
      pyx = 1.0 - 1.0 / (1 + np.exp(weights.w0 + self.compute_weight_feature_product(weights, instance)))
      ctr_data.append(pyx)

      update_frac = 5
      if count % (dataset.size/update_frac) == 0:
        print("processed {}% of testing data".format((count * 100)/dataset.size))
      count += 1
    
    dataset.reset()

    return ctr_data
  
  
if __name__ == '__main__':
  # TODO: Fill in your code here
  print "Training Logistic Regression with Hashed Features..."
  lg = LogisticRegressionWithHashing()
  train_size = DataSet.TRAININGSIZE
  test_size = DataSet.TESTINGSIZE

  training = DataSet("../data/train.txt", True, train_size)
  testing = DataSet("../data/test.txt", False, test_size)

  hash_sizes = [101, 12277, 1573549]
  lambduh = 0.001
  step = 0.01
  avg_ctrs = []
  rmses = []

  for m in hash_sizes:
    weights = lg.train(dataset=training, dim=m, lambduh=lambduh, step=step, avg_loss=0.0, personalized=False)

    ctr_predict = lg.predict(weights=weights, dataset=testing, personalized=False)
    avg_ctr = np.mean(ctr_predict)
    avg_ctrs.append(avg_ctr)
    rmse = EvalUtil.eval(path_to_sol="../data/test_label.txt", ctr_predictions=ctr_predict)
    rmses.append(rmse)

  print("m, avg CTR, RMSE")
  for i in range(len(hash_sizes)):
    print("{}, {}, {}".format(hash_sizes[i], avg_ctrs[i], rmses[i]))





