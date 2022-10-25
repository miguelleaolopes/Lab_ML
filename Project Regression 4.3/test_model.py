from initialization import *
from model1 import *
from model2 import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# m1 = model1()
# m1.create_model()
# print('\n\n\n##### End of tensorflow rant ##########\n\n\n')
# m1.summary()
# m1.compile()
# m1.show_acc_plt()
# m1.show_acc_val()


m2 = model2()
m2.create_model()
print('\n\n\n##### End of tensorflow rant ##########\n\n\n')
m2.summary()
m2.compile()
m2.show_acc_plt()
m2.show_acc_val()