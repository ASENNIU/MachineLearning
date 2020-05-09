import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, classification_report

def selct_threshold(X, Xval, yval):
    """
    use CV data to find the best epsilon
    :return:
    e: best epsilon with the highest f-score
    f-score: such best f-score
    """
    #create multivariate model using training data

    mu = X.mean(axis=0)
    cov = np.cov(X.T)
    multi_normal = stats.multivariate_normal(mu, cov)

    #this is key, use CV data for fine tuning hyper parameters
    pval = multi_normal.pdf(Xval)


    #set up epsilon candidates
    epsilon = np.linspace(np.min(pval), np.max(pval), num=10000)

    #calculate f-score
    fs = []
    for e in epsilon:
        y_pred = (pval <= e).astype('int')
        fs.append(f1_score(yval, y_pred))

    #find the best f-score
    argmax_fs = np.argmax(fs)

    return epsilon[argmax_fs], fs[argmax_fs]

def predict(X, Xval, e, Xtest, ytest):
   Xdata = np.concatenate((X, Xval), axis=0)

   mu = Xdata.mean(axis=0)
   cov = np.cov(Xdata.T)
   multi_normal = stats.multivariate_normal(mu, cov)

   #calculate probability of test data
   pval = multi_normal.pdf(Xtest)
   y_pred = (pval <= e).astype('int')

   print(classification_report(ytest, y_pred))

   return multi_normal, y_pred