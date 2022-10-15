from initialization import *


alpha = 0.00456514816802045
# alpha = 0.004563713924050698 

outliers = [15, 18, 24, 29, 30, 33, 34, 36, 47, 48, 62, 63, 65, 70, 71, 72, 83, 88, 93, 95]

print('There is', len(outliers),'outliers')

x_import_wo = np.delete(x_import, outliers, axis=0)
y_import_wo = np.delete(y_import, outliers, axis=0)

model = lasso_model(x_import_wo,y_import_wo,alpha,fit_intercept=True)
X_TEST = np.load("data/Xtest_Regression2.npy")
y_pred = model.predict(X_TEST)

np.save('y_predictions.npy', y_pred)
print('Y predictions shape:', np.shape(y_pred))
