
from Method6_dffits import *


# m1 = method2_cooks_distance()
# m1.remove_outliers()
# m1.test_method()

# m2 = method2_cooks_distance()
# m2.remove_outliers_cyclical()
# m2.test_method()



m3 = method5_huber_theil()
m3.find_epsilon()



# A vantagem é podermos ter acesso a todas as variaveis que guardamos com "self."

#print(m1.out_list())