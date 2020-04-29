import pytest
from mlpce import Confidence
import pandas as pd


def append_expectation(test, expectation):
    return test[0], test[1], test[2], test[3], expectation


pd_x = pd.DataFrame(data=[[-1, -0.5, 0.5, -1, 1, 1], [1, -1, 1, -1, -1, -1], [-0.5, 0.5, 1, -0.5, 0, 1],
                          [0.5, 1, 1, 0.5, -1, -1], [-0.5, 0.5, -0.5, 1, -1, 0.5], [-0.5, 0.5, -1, -0.5, 0.5, 1],
                          [1, 1, -1, -1, -1, 0.5], [1, -1, -1, -0.5, 1, 0.5], [1, 0.5, -1, 1, 0.5, 0],
                          [0, -0.5, 0.5, -0.5, -0.5, 0.5], [1, 1, 1, 1, 1, -0.5], [0.5, 1, -0.5, 0.5, -0.5, 1],
                          [0.5, -0.5, -0.5, -0.5, 0.5, -0.5], [1, -1, 1, -1, 0.5, 1], [-1, 1, 0, 1, 1, 1],
                          [1, 1, 0.5, -1, 1, 1], [-0.5, -0.5, -1, -1, 0.5, -1], [1, -1, -1, 0.5, 1, -1],
                          [0.5, -1, -1, -1, -0.5, -0.5], [-1, -1, 0, -0.5, -1, -1], [1, -0.5, 1, 0.5, 1, 0],
                          [0.5, -1, 0.5, 1, 0, -0.5], [1, 0.5, 0.5, -0.5, -0.5, -0.5], [1, -1, 1, 0.5, -1, 1],
                          [0.5, 0.5, -0.5, -1, 1, -1], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, -0.5, 0, 1, 1, 1],
                          [-0.5, -0.5, 1, 0.5, -1, -0.5], [-1, 1, 0, -0.5, 1, 0], [1, 1, -0.5, -1, -0.5, -1],
                          [0.5, 0.5, -1, 1, -1, -0.5], [0.5, 1, 1, -1, -1, 0.5], [1, -1, -1, 1, -1, 0.5],
                          [-0.5, -1, -0.5, 0.5, 1, 0], [1, -0.5, -0.5, -1, -1, 1], [-1, -0.5, -1, 1, -0.5, -1],
                          [-1, 1, -1, 1, 0.5, -1], [-0.5, -1, -1, -0.5, -1, 1], [-1, 0, -0.5, -1, -0.5, 0.5],
                          [1, -1, 0.5, -1, 1, -1], [-1, 0.5, -1, -0.5, -1, -1], [1, 1, 1, 1, -1, 1],
                          [1, -1, -0.5, 0.5, -1, -1], [-1, 0.5, 1, 1, -1, -1], [-1, -1, 1, -0.5, 1, -0.5],
                          [-1, -0.5, -1, 0.5, 0, 1], [-1, -1, 1, -1, -1, 1], [-1, 0, 0.5, 1, 1, -1],
                          [0.5, 1, 1, -1, 0.5, -1], [-0.5, 0.5, 1, -1, -1, -1], [-1, 0, 1, 1, -1, 1],
                          [-1, 1, 0.5, -0.5, -1, 1], [-0.5, 1, 0.5, 0.5, 0, -0.5], [-1, -1, 1, 1, 0.5, 0.5]],
                    columns=['a', 'b', 'c', 'd', 'e', 'f'])
pd_x_wy = pd_x.copy()
pd_x_wy['y'] = [1 for i in range(pd_x.shape[0])]
responses = ['y']
pd_x_k = pd.DataFrame(data=[[0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2]],
                      columns=['a', 'b', 'c', 'd', 'e', 'f'])

testdata = [
    (pd_x, pd_x_k, None, None),
    (pd_x.iloc[:-2, :], pd_x_k, None, None),
    (pd_x, pd_x_k, 'a+b+c+d+e+f', None),
    (pd_x_wy, pd_x_k, None, responses),
    (pd_x_wy, pd_x_k, 'a+b+c+d+e+f', responses)
]

confidence_thresholds = [
    [1.000000000000001, 1.0000000000000022],
    [0.9958477274376679, 0.998747858743115],
    [0.1674941152962534, 0.17424535876006517],
    [1.000000000000001, 1.0000000000000022],
    [0.1674941152962534, 0.17424535876006517]
]
testdata_conf_thresh = [append_expectation(testdata[i], confidence_thresholds[i]) for i in range(len(testdata))]


@pytest.mark.parametrize("known,new,model,response,expected", testdata_conf_thresh)
def test_mlpce_confidence_thresholds(known, new, model, response, expected):
    emm = Confidence(known=known, model=model, responses=response)
    assert emm.confidence_thresholds['Full'] == expected


confidence_length = [
    1,
    1,
    1,
    2,
    2
]
testdata_conf_thresh = [append_expectation(testdata[i], confidence_length[i]) for i in range(len(testdata))]


@pytest.mark.parametrize("known,new,model,response,expected", testdata_conf_thresh)
def test_mlpce_confidence_thresholds(known, new, model, response, expected):
    emm = Confidence(known=known, model=model, responses=response)
    assert len(emm.confidence_thresholds) == expected


models = [
    'a+b+c+d+e+f+a*a+a*b+a*c+a*d+a*e+a*f+b*b+b*c+b*d+b*e+b*f+c*c+c*d+c*e+c*f+d*d+d*e+d*f+e*e+e*f+f*f+a*a*a+a*b*c+a*b*d+'
    'a*b*e+a*b*f+a*c*d+a*c*e+a*c*f+a*d*e+a*d*f+a*e*f+b*b*b+b*c*d+b*c*e+b*c*f+b*d*e+b*d*f+b*e*f+c*c*c+c*d*e+c*d*f+c*e*f+'
    'd*d*d+d*e*f+e*e*e+f*f*f',
    'a+b+c+d+e+f+a*a+a*b+a*c+a*d+a*e+a*f+b*b+b*c+b*d+b*e+b*f+c*c+c*d+c*e+c*f+d*d+d*e+d*f+e*e+e*f+f*f+a*b*c+a*b*d+a*b*e+'
    'a*b*f+a*c*d+a*c*e+a*c*f+a*d*e+a*d*f+a*e*f+b*c*d+b*c*e+b*c*f+b*d*e+b*d*f+b*e*f+c*d*e+c*d*f+c*e*f+d*e*f',
    'a+b+c+d+e+f',
    'a+b+c+d+e+f+a*a+a*b+a*c+a*d+a*e+a*f+b*b+b*c+b*d+b*e+b*f+c*c+c*d+c*e+c*f+d*d+d*e+d*f+e*e+e*f+f*f+a*a*a+a*b*c+a*b*d+'
    'a*b*e+a*b*f+a*c*d+a*c*e+a*c*f+a*d*e+a*d*f+a*e*f+b*b*b+b*c*d+b*c*e+b*c*f+b*d*e+b*d*f+b*e*f+c*c*c+c*d*e+c*d*f+c*e*f+'
    'd*d*d+d*e*f+e*e*e+f*f*f',
    'a+b+c+d+e+f'
]
testdata_model = [append_expectation(testdata[i], models[i]) for i in range(len(testdata))]


@pytest.mark.parametrize("known,new,model,response,expected", testdata_model)
def test_mlpce_confidence_thresholds(known, new, model, response, expected):
    emm = Confidence(known=known, model=model, responses=response)
    assert emm.model == expected


conf_results = [
    ['High', 'Low'],
    ['High', 'Low'],
    ['High', 'Low'],
    ['High', 'Low'],
    ['High', 'Low']
]
testdata_confidence = [append_expectation(testdata[i], conf_results[i]) for i in range(len(testdata))]


@pytest.mark.parametrize("known,new,model,response,expected", testdata_confidence)
def test_mlpce_confidence_thresholds(known, new, model, response, expected):
    print(response)
    emm = Confidence(known=known, model=model, responses=response)
    var, con = emm.assess_x(new)
    assert con['Full'] == expected
