from functools import reduce
import numpy as np
import re
from .helpers import drop_term


class Confidence:
    """A class for assessing expanding input terms to match the model space

    :param known: pandas DataFrame of points with gathered data (input and optional responses).
    :param model: OPTIONAL list of model terms to use for calculating I-Optimality based on column names in known.
                  If not provided, model is automatically calculated as highest order possible tier-3 RSM.
    :param responses: OPTIONAL list of column names in the known pandas DataFrame of
                      responses in model used to filter matrix to non-missing values.
    """

    def __init__(self, known, model=None, responses=None):
        self.model = model
        self.known = known.copy()
        if responses is None:
            responses = []
        self.responses = responses.copy()

        self._check_for_missing()
        self.x_columns = self._extract_x_columns()

        self.xpxi = None

        if model is None:
            self.model = self._auto_model(known)

        self.expansion_functions, self.non_linear_terms, self.linear_terms = self._parse_model()
        self._check_columns()

        self.xpxi = self.gen_xpxi()
        self.confidence_thresholds = self._calculate_confidence_threshold()

    def _extract_x_columns(self):
        all_columns = self.known.columns.to_list()
        x_columns = [x for x in all_columns if x not in self.responses]
        return x_columns

    def _check_for_missing(self):
        tmp = self.known.copy()
        for c in self.responses:
            try:
                del tmp[c]
            except KeyError:
                continue
        if np.any(tmp.isna().values):
            raise Exception('Missing values present in known values, impossible to process.')

    def _check_columns(self):
        col_names = [x for x in self.known.columns.to_list() if x not in self.responses]
        extra_cols = [x for x in col_names if x not in self.linear_terms]
        if len(extra_cols) > 0:
            print(extra_cols)
            raise Exception('All non-response columns in known data set must '
                            'be included as linear terms in the model.')

        missing_res = [x for x in self.responses if x not in self.known.columns.to_list()]
        if len(missing_res) > 0:
            print(missing_res)
            raise Exception('All provided responses must be present in the known data set.')

    def expand_x(self, pd_to_expand):
        """
        Method to expand an input set of x values to the full model
        :param pd_to_expand: pandas DataFrame containing the original column names used to generate this which
                             should be expanded
        :return: expanded pandas DataFrame
        """
        pdf = pd_to_expand.copy()
        for i in range(len(self.non_linear_terms)):
            pdf[self.non_linear_terms[i]] = self.expansion_functions[i](pdf)
        cols = [c for c in pdf.columns.to_list() if c != 'Intercept']
        cols.insert(0, 'Intercept')
        return pdf[cols]

    @staticmethod
    def _gen_term_function(term):
        terms = term.split('*')

        def output(df):
            columns = [df[tr.strip()] for tr in terms]
            return reduce(lambda y, z: z*y, columns)
        return output

    def _parse_model(self, intercept=True):
        terms = self.model.split('+')
        functions = []
        terms_out = []
        linear_out = []
        if intercept:
            def intercept(row):
                return 1

            functions.append(intercept)
            terms_out.append('Intercept')
        for term in terms:
            if re.search(r'\*', term):
                functions.append(self._gen_term_function(term))
                terms_out.append(term.strip())
            else:
                linear_out.append(term.strip())
        return functions, terms_out, linear_out

    @staticmethod
    def _check_degrees_of_freedom(shape):
        df = shape[0] - 1
        linear = shape[1]
        two_fi = (linear * (linear - 1)) / 2
        x2 = linear
        three_fi = (linear * (linear - 1) * (linear - 2)) / 6
        x3 = linear
        df -= linear
        linear = df >= 0
        df -= two_fi
        two_fi = df >= 0
        df -= x2
        x2 = df >= 0
        df -= three_fi
        three_fi = df >= 0
        df -= x3
        x3 = df >= 0
        return linear, two_fi, x2, three_fi, x3

    def _auto_model(self, df):
        linearterms = [x for x in df.columns.to_list() if x not in self.responses]
        linear, two_fi, x2, three_fi, x3 = self._check_degrees_of_freedom(df.shape)
        grid2 = np.meshgrid(linearterms, linearterms, indexing='ij')
        grid3 = np.meshgrid(linearterms, linearterms, linearterms, indexing='ij')
        if two_fi:
            two_fi = [list(a) for a in list(np.transpose(np.array([term.flatten() for term in grid2])))
                      if a[0] <= a[1]]
        else:
            two_fi = []
        if three_fi:
            three_fi = [list(a) for a in list(np.transpose(np.array([term.flatten() for term in grid3])))
                        if a[0] <= a[1] <= a[2]]
        else:
            three_fi = []
        modelterms = [[lin_term] for lin_term in linearterms] + two_fi + three_fi
        to_drop = [drop_term(term, x2, x3) for term in modelterms]
        modelterms = [modelterms[m] for m in range(len(modelterms)) if not to_drop[m]]
        unique_modelterms = []
        for m in range(len(modelterms)):
            modelterms[m].sort()
            if modelterms[m] not in unique_modelterms:
                unique_modelterms.append('*'.join(modelterms[m]))
        return '+'.join(unique_modelterms)

    def gen_xpxi(self, known=None, responses=None):
        """
        Calculate the 'X-prime, X, inverse' matrix for each response and the full dataset
        :param known: pandas DataFrame of points with gathered data.
        :param responses: OPTIONAL list of column names in known corresponding to response variables
        :return:
        """
        if known is None:
            known = self.known
        if responses is None:
            responses = self.responses

        vals = [x for x in range(known.shape[0])]

        if len(responses) == 0 or 'Full' not in responses:
            responses.insert(0, 'Full')
            known['Full'] = vals

        design = self.expand_x(known.copy())

        output = {}

        for col in responses:
            f_design = design.copy()
            f_design = f_design[f_design[col].notnull()]
            for c in responses:
                del f_design[c]
            f_design = f_design.values
            xpxi = np.linalg.inv(np.dot(f_design.transpose(), f_design))
            output[col] = xpxi
        return output

    def _calculate_confidence_threshold(self):
        tmp = self.known.copy()
        for c in self.responses:
            try:
                del tmp[c]
            except KeyError:
                continue
        values = self.calc_pred_var(x_k=tmp)
        output = {}
        for v in values:
            output[v] = [np.percentile(values[v], q=90), np.max(values[v])]
        return output

    @staticmethod
    def _apply_confidence_label(threshold, value):
        output = 'High'
        if value > threshold[1]:
            output = 'Low'
        elif value > threshold[0]:
            output = 'Mid'
        return output

    def _score_confidence(self, response, values):
        thresholds = self.confidence_thresholds[response]
        output = [self._apply_confidence_label(thresholds, v) for v in values]
        return output

    def calc_pred_var(self, x_k=None):
        """
        Take a new point x_k and calculate the prediction variance associated with it
        :param x_k: pandas DataFrame of new point
        :return: unscaled prediction variance
        """
        missing_x = [x for x in self.x_columns if x not in x_k.columns.to_list()]
        if len(missing_x) > 0:
            print(missing_x)
            raise Exception('Provided x does not match original known data')
        x_k = x_k.copy()[self.x_columns]
        full_x_k = self.expand_x(x_k).values
        output = {}
        for r in self.xpxi:
            output[r] = list(np.sqrt(np.matmul(np.matmul(full_x_k, self.xpxi[r]), full_x_k.transpose()).diagonal()))
        return output

    def assess_x(self, x_k=None):
        """
        Take a new point x_k and assess the confidence associated with predictions made on the point
        :param x_k: pandas DataFrame of new point
        :return: unscaled prediction variance, confidence score
        """
        pred_var = self.calc_pred_var(x_k=x_k)
        score = {}
        for r in pred_var:
            score[r] = self._score_confidence(response=r, values=pred_var[r])
        return pred_var, score
