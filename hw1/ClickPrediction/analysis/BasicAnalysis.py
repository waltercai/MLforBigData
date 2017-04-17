from analysis.DataSet import DataSet
from util.EvalUtil import EvalUtil

class BasicAnalysis:
  # ==========================
  # @param dataset {DataSet}
  # @return [{Int}] the unique tokens in the dataset
  # ==========================
  def uniq_tokens(self, dataset):
    # TODO: Fill in your code here
    tokens = set()
    
    count = 0
    while dataset.hasNext():
      instance = dataset.nextInstance()
      count += 1
      map(lambda x: tokens.add(x), instance.tokens)
    if count < dataset.size:
      print("Warning: the real size of the data is less than the input size: {} < {}}".format(dataset.size, count))
    
    dataset.reset()    
    return tokens
  
  # ==========================
  # @param dataset {DataSet}
  # @return [{Int}] the unique user ids in the dataset
  # ==========================
  def uniq_users(self, dataset):
    # TODO: Fill in your code here
    users = set()
    
    count = 0
    while dataset.hasNext():
      instance = dataset.nextInstance()
      count += 1
      users.add(instance.userid)
    if count < dataset.size:
      print("Warning: the real size of the data is less than the input size: {} < {}}".format(dataset.size, count))
    
    dataset.reset()    
    return users

  # ==========================
  # @param dataset {DataSet}
  # @return {Int: [{Int}]} a mapping from age group to unique users ids
  #                        in the dataset
  # ==========================
  def uniq_users_per_age_group(self, dataset):
    # TODO: Fill in your code here
    users_by_age = {0: set(), 1: set(), 2: set(), 3: set(), 4: set(), 5: set(), 6: set()}
    
    count = 0
    while dataset.hasNext():
      instance = dataset.nextInstance()
      count += 1
      users_by_age[instance.age].add(instance.userid)
    if count < dataset.size:
      print("Warning: the real size of the data is less than the input size: {} < {}}".format(dataset.size, count))
    
    dataset.reset()    
    return users_by_age

  # ==========================
  # @param dataset {DataSet}
  # @return {Double} the average CTR for a dataset
  # ==========================
  def average_ctr(self, dataset):
    # TODO: Fill in your code here
    count = 0
    ct_count = 0
    
    while dataset.hasNext():
      instance = dataset.nextInstance()
      count += 1
      if instance.clicked:
        ct_count += 1
    if count < dataset.size:
      print("Warning: the real size of the data is less than the input size: {} < {}}".format(dataset.size, count))
    ctr = ct_count/(dataset.size + 0.0)    
    
    dataset.reset()
    return ctr

  def do_all(self, dataset):
    count = 0
    tokens = set()
    users_by_age = {0: set(), 1: set(), 2: set(), 3: set(), 4: set(), 5: set(), 6: set()}
    
    while dataset.hasNext():
      instance = dataset.nextInstance()
      count += 1
      map(lambda x: tokens.add(x), instance.tokens)
      users_by_age[instance.age].add(instance.userid)

    if count < dataset.size:
      print("Warning: the real size of the data is less than the input size: {} < {}}".format(dataset.size, count))
    ctr = ct_count/(dataset.size + 0.0)
        
    dataset.reset()
    return [ctr, tokens, users_by_age]


if __name__ == '__main__':
  # TODO: Fill in your code here
  loader = BasicAnalysis()
  train_size = DataSet.TRAININGSIZE
  test_size = DataSet.TESTINGSIZE

  training = DataSet("../data/train.txt", True, train_size)
  testing = DataSet("../data/test.txt", False, test_size)

  # derives ctr
  ctr = loader.average_ctr(training)
  print("CTR: {}".format(ctr))

  print("baseline RMSE: {}".format(EvalUtil.eval_baseline(path_to_sol="../data/test_label.txt", average_ctr=ctr)))

  # do all the heavy lifting
  # [training_tokens, training_users_by_age] = loader.do_all(training)
  # [testing_tokens, testing_users_by_age] = loader.do_all(testing)

  # derives number of unique tokens in training set
  training_tokens = loader.uniq_tokens(training)
  print("Unique token count in training: {}".format(len(training_tokens)))

  # derives number of unique tokens in testing set
  testing_tokens = loader.uniq_tokens(testing)
  print("Unique token count in testing: {}".format(len(testing_tokens)))

  # derive number of unique tokens appearing in both training and testing
  print("Unique token count appearing in both: {}".format(len(training_tokens.intersection(testing_tokens))))

  # derive number of unique users by age group for training
  print("Users by age group in training:")
  training_users_by_age = loader.uniq_users_per_age_group(training)
  for age in training_users_by_age:
    print("group {}: {}".format(age, len(training_users_by_age[age])))

  # derive number of unique users by age group for testing
  print("Users by age group in testing:")
  testing_users_by_age = loader.uniq_users_per_age_group(testing)
  for age in testing_users_by_age:
    print("group {}: {}".format(age, len(testing_users_by_age[age])))















