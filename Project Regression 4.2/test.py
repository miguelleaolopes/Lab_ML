
from method1_se_removal import *
from method1_1_se_removal_cyclical import *

m1 = method1_se_removal()
m1.remove_outliers()
m1.test_method()

m2 = method1_1_se_removal_cyclical()
m2.remove_outliers_cyclical()
m2.test_method()





# m3 = method1_1_cyclical_se_comparisson()
# m3.remove_outliers_cyclical()
# m3.test_method()
# print(np.shape(m3.x_import_wo))


# A vantagem é podermos ter acesso a todas as variaveis que guardamos com "self."

#print(m1.out_list())