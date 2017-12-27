import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score, best_model = float("inf"), None
        n_features = self.X.shape[1]
        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                model = GaussianHMM(n_components = n, covariance_type = "diag", n_iter = 1000, random_state = self.random_state, verbose = False).fit(self.X, self.lengths)
                logL = model.score(self.X, self.lengths)
                
                n_params = n*n + 2*n*n_features -1
                logN = np.log(len(self.X))
                bic = -2*logL + n_params*logN

            except:
                bic = float("inf")
                model = None
            if bic<best_score:
                best_model = model
                best_score = bic

        # TODO implement model selection based on BIC scores
        return  best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_model = float("inf"), None
        best_score = float("-inf")
        M = len((self.words).keys())

        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                hmm_model = GaussianHMM(n_components = n, covariance_type = "diag", n_iter = 1000, random_state = self.random_state, verbose = False).fit(self.X, self.lengths)

                logl = hmm_model.score(self.X, self.lengths)
            except:

                logl = float("-inf")
            sum1 = 0

            for word in self.hwords.keys():
                if word!= self.this_word:
                    Xi, length_wordi = self.hwords[word] 
                    try:
                        sum1 += hmm_model.score(Xi, length_wordi)
                    except:
                        sum1 += 0

            dic_score = logl - (1/(M-1))* (sum1 - (0 if logl == float("-inf") else logl))

            if dic_score> best_score:
                best_score = dic_score
                best_model = hmm_model
        return best_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        best_score = float("-inf")

        best_model = None
        if len(self.sequences)<2:
            return None

        split_method = KFold(n_splits = 2)

        for n in range(self.min_n_components, self.max_n_components+1):
            sum1 = 0
            count = 0

            for cv_train, cv_test in split_method.split(self.sequences):
                X_train, lengths_train = combine_sequences(cv_train, self.sequences)
                X_test, lengths_test = combine_sequences(cv_test, self.sequences)

                try:
                    hmm_model = GaussianHMM(n_components= n, covariance_type = "diag", n_iter = 1000,
                        random_state = self.random_state, verbose = False).fit(X_train, lengths_train)

                    logl = hmm_model.score(X_test, lengths_test)
                    count += 1
                except:
                    logl = 0

                sum1 += logl

            cv_score = sum1/(1 if count ==0 else count)

            if cv_score > best_score:
                best_score = cv_score
                best_model = hmm_model

        return best_model