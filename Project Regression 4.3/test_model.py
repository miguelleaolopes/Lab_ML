from initialization import *
from model1 import *
from model2 import *


m1 = model2()
m1.create_model()
print('\n\n\n##### End of tensorflow rant ##########\n\n\n')
m1.summary()
m1.compile()
m1.show_acc_plt()
m1.show_acc_val()

