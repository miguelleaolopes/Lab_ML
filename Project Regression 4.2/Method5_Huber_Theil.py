from initialization import *
from sklearn.linear_model import TheilSenRegressor, HuberRegressor
import statsmodels.api as sm


class method5_huber_theil:

    def __init__(self,silent=True,N_val=200,alpha_list = np.linspace(0.001,2,100)):
        self.silent = silent
        self.outliers_removed = False
        self.N_val = N_val
        self.alpha_list = alpha_list

    def remove_outliers(self):
        print("\n\n\nMethod 5: Remove outliers with Huber Theil method all at once")

        Huber = HuberRegressor(epsilon=1.8, max_iter=10000,tol=1e-05)
        Huber.fit(x_import, y_import)
        Huber.score(x_import, y_import)

        self.outlier_mask = Huber.outliers_
        self.all_mask = np.where(self.outlier_mask == False, True, False)
  

        self.out_list = []
        for i in range(len(self.all_mask)):
            if self.all_mask[i] == False: self.out_list.append(i)
        
        
        print("Total datapoints were : ", len(self.all_mask))
        print('There are',len(self.out_list),'outliers.\n',self.out_list)

        self.x_import_wo = x_import[self.all_mask,:]
        self.y_import_wo = y_import[self.all_mask,:]

        if not self.silent:
            print("Shape of Xinlier:", self.x_import_wo.shape)
            print("Shape of Yinlier:", self.y_import_wo.shape)

            # Testing SSE and MSE directly from all data (before cv)
            huber_linear_model = linear_model(x_import,y_import)
            print("SSE for Huber before CV:", (np.linalg.norm(self.y_import_wo-huber_linear_model.predict(self.x_import_wo)))**2)
            print("MSE for Huber before CV:", mean_squared_error(self.y_import_wo, huber_linear_model.predict(self.x_import_wo)))

        self.outliers_removed = True

    def find_epsilon(self):
        print("\n\n\nRemove outliers with Huber Theil and find the best epsilon")
        global x_train,y_train,x_test,y_test, x_train_s, x_test_s

        mse_mean = []
        eps_list=[1 + 0.001*i for i in range(3000)]

        for i in trange(len(eps_list)):
            # Initializing the model
            Huber = HuberRegressor(epsilon=eps_list[i], max_iter=10000,tol=1e-05)
            Huber.fit(x_import, y_import)
            Huber.score(x_import, y_import)


            outlier_mask = Huber.outliers_
            all_mask = np.where(outlier_mask == False, True, False)
            out_list = []

            for i in range(len(all_mask)):
                if all_mask[i] == False: out_list.append(i)

            x_import_wo = x_import[all_mask,:]
            y_import_wo = y_import[all_mask,:]

            N_validation = 500
            split_data(x_import_wo,y_import_wo,N_validation,0.2)
            

            mse_list = []
            for i in range(N_validation):
                lin_model = linear_model(x_train[i],y_train[i]) #For linear we do not center
                y_pred_lin = lin_model.predict(x_test[i])
                mse_list.append(mean_squared_error(y_pred_lin,y_test[i]))

            mse_mean.append(np.mean(mse_list))
            
        plt.plot(eps_list,mse_mean, color='blue', marker='o')
        plt.title('MSE vs Epsilon value for Huber Theil', fontsize=14)
        plt.ylabel('Epsilon', fontsize=14)
        plt.xlabel('MSE', fontsize=14)
        plt.grid(True)
        plt.show()

    def test_method(self):
    
        if self.outliers_removed:
            self.models, self.best_alphas, self.best_index = determine_best_model(self.x_import_wo, self.y_import_wo,self.N_val,self.alpha_list)
        else: print('Outliers not removed, please remove outliers first with self.remove_outliers()!')