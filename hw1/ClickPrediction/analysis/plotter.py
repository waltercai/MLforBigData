import math
import numpy as np
import matplotlib.pyplot as plt

lambduh_range = [ 0.002,  0.004,  0.006,  0.008,  0.01,   0.012,  0.014]
l2_norms = [3.9270140507337503, 3.5386191682512362, 3.3971869697219135, 3.3276409047158064, 3.2891912376387555, 3.2669275878022446, 3.254007608138171]
rmses = [0.17290959465263367, 0.17288911053850783, 0.17289197877613732, 0.1729019852058484, 0.17291445550947251, 0.17292747373984896, 0.1729403340079834]

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


