
from method1_se_removal import *
from method2_cook_distance import *
from method3_isolation_forest import *
from method4_ransac import *
from method5_huber_theil import *
from method6_dffits import *



# m1 = method2_cooks_distance(show_plt=True)
# m1.remove_outliers()
# m1.test_method()

# m2 = method2_cooks_distance()
# m2.remove_outliers_cyclical()
# m2.test_method()



m3 = method5_huber_theil()
m3.find_epsilon()
m3.test_method()



# A vantagem é podermos ter acesso a todas as variaveis que guardamos com "self."

#print(m1.out_list())