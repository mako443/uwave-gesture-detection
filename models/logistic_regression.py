import numpy as np

class LogisticRegression:
    """Logistic regression classifier implemented in pure numpy.
    """

    def __init__(self, lr=0.01, reg=0.001, max_iter=1000, tol=1e-4):
        self.lr = lr
        self.reg = reg
        self.max_iter = max_iter
        self.tol = tol

        self.W = None

    def fit(self, X, y, verbose=False):
        assert len(np.unique(y)) == np.max(y) + 1, "Labels y are not (0, ..., num_classes-1)"

        num_samples, dim = X.shape
        num_classes = len(np.unique(y))        
        W = np.zeros((dim, num_classes))

        for i_iter in range(self.max_iter):
            loss = 0.0

            scores = X @ W
            scores -= np.max(scores) # Improve numerical stability

            scores_exp = np.exp(scores)
            scores_exp_rowsum = np.sum(scores_exp, axis=1)
            scores_exp_rowsum_log = np.log(scores_exp_rowsum)

            loss += np.sum(scores_exp_rowsum_log)
            loss -= np.sum(scores[np.arange(num_samples), y])

            loss /= num_samples
            loss += 0.5 * self.reg * np.sum(np.square(W))

            d_scores_exp_rowsum_log = 1/scores_exp_rowsum
            d_scores_exp_rowsum = 1*d_scores_exp_rowsum_log
            d_scores_exp = d_scores_exp_rowsum*scores_exp.T

            d_scores = 1*d_scores_exp

            d_scores[y,np.arange(num_samples)] -= 1 #correct

            d_W = np.dot(X.T,d_scores.T)
            d_W /= num_samples
            d_W += 2 * 0.5 * self.reg * W        

            W -= self.lr * d_W

            if loss < self.tol:
                if verbose:
                    print(f'Finished after {i_iter} iterations.')
                break

            if verbose and i_iter % 100 == 0:
                print(f'iter: {i_iter:04.0f}, loss: {loss:0.2f}')

        self.W = W

    def predict(self, X):
        if self.W is None:
            raise Exception("Model is not trained yet.")

        Y = X @ self.W # [num_samples, num_classes]
        y = np.argmax(Y, axis=-1)
        return y