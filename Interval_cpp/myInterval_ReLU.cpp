#include <boost/multiprecision/mpfi.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <iostream>
#include <math.h>
#include <vector>
#include <string>
#include <tuple>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "myUtility.h"

using namespace std;
using namespace boost::multiprecision;

tuple<mpfi_float_1000, int> myInterval(vector<vector<mpfi_float_1000>> I_X0, int Pred_X0, vector<vector<float>> W1, vector<vector<float>> W2, vector<float> b1, int inputs_size, int hidden_neural_num, int output_size)
{

    // convert to interval
    vector<vector<mpfi_float_1000>> I_W1 = convert_2D_matrix(W1);
    vector<vector<mpfi_float_1000>> I_W2 = convert_2D_matrix(W2);
    vector<mpfi_float_1000> I_b1 = convert_array(b1);

    // init
    vector<mpfi_float_1000> Y1, norm;
    vector<int> index;
    vector<vector<mpfi_float_1000>> result;

    // forward
    for (int i = 0; i < hidden_neural_num; i++)
    {
        mpfi_float_1000 tmp = convert(0);
        for (int j = 0; j < inputs_size; j++)
        {
            tmp += I_W1[i][j] * I_X0[0][j];
        }
        tmp += I_b1[i];
        Y1.push_back(tmp);
    }

    // backward
    for (int m = 0; m < output_size; m++)
    {
        vector<mpfi_float_1000> Y2;
        Y2.clear();
        for (int n = 0; n < inputs_size; n++)
        {
            mpfi_float_1000 tmp = convert(0);
            for (int k = 0; k < hidden_neural_num; k++)
            {
                tmp += I_W2[m][k] * relu_d(Y1[k]) * I_W1[k][n];
            }
            Y2.push_back(tmp);
        }
        result.push_back(Y2);
    }

    // norm
    for (int i = 0; i < output_size; i++)
    {
        if (i == Pred_X0)
        {
            continue;
        }
        vector<mpfi_float_1000> grads;
        grads.clear();
        for (int m = 0; m < inputs_size; m++)
        {
            grads.push_back(result[Pred_X0][m] - result[i][m]);
        }
        mpfi_float_1000 tmp = convert(0);
        for (int n = 0; n < inputs_size; n++)
        {
            tmp += abs(grads[n]);
        }
        norm.push_back(tmp);
        index.push_back(i);
    }

    // find max
    tuple<mpfi_float_1000, int> result_norm = myMax(norm);
    mpfi_float_1000 max_grad = get<0>(result_norm);
    int max_grad_index = get<1>(result_norm);

    auto return_result = make_tuple(max_grad, index[max_grad_index]);

    return return_result;
}

tuple<vector<vector<mpfi_float_1000>>, mpfi_float_1000> bisection(vector<vector<mpfi_float_1000>> I_X0, int Pred_X0, vector<vector<float>> W1, vector<vector<float>> W2, vector<float> b1, int inputs_size, int hidden_neural_num, int output_size, int count)
{

    // get bi-section list
    vector<vector<mpfi_float_1000>> new_X0;

    for (int i = 0; i < I_X0.size(); i++)
    {
        vector<vector<mpfi_float_1000>> tmp = getBisection(I_X0[i]);
        for (int j = 0; j < tmp.size(); j++)
        {
            new_X0.push_back(tmp[j]);
        }
    }

    // calculate interval
    vector<vector<mpfi_float_1000>> wrapper;
    vector<mpfi_float_1000> low;
    vector<tuple<mpfi_float_1000, int>> up;
    for (int i = 0; i < new_X0.size(); i++)
    {
        wrapper.clear();
        wrapper.push_back(new_X0[i]);
        tuple<mpfi_float_1000, int> tmp = myInterval(wrapper, Pred_X0, W1, W2, b1, inputs_size, hidden_neural_num, output_size);
        tuple<mpfi_float_1000, mpfi_float_1000> result_tmp = getLowerUpper(tmp);
        auto tumple_upper = make_tuple(get<1>(result_tmp), i);
        low.push_back(get<0>(result_tmp));
        up.push_back(tumple_upper);
    }

    // sort
    vector<mpfi_float_1000> low_sorted = getSort(low);
    mpfi_float_1000 low_max = low_sorted[low_sorted.size() - 1];
    cout << "\nIn Bisection " << count << ": " << low_max << endl;

    // drop useless intervals
    vector<vector<mpfi_float_1000>> result;
    for (int i = 0; i < up.size(); i++)
    {
        tuple<mpfi_float_1000, int> tmp = up[i];
        mpfi_float_1000 tmp_upper = get<0>(tmp);
        int index = get<1>(tmp);
        if (tmp_upper > low_max)
        {
            result.push_back(new_X0[index]);
        }
    }

    cout << "New Size: " << result.size() << endl;

    auto result_tuple = make_tuple(result, low_max);

    return result_tuple;
}

bool conditions(int count, mpfi_float_1000 low_max, bool comparedToCLEVER, int maxIterationNum, float minGap, mpfi_float_1000 up_max, int boxes_size, int max_boxes)
{
    bool result = true;
    mpfi_float_1000 gap = up_max - low_max;

    // Condition 00
    if (gap < convert(minGap))
    {
        cout << "\nExit with condition 00: reach the min gap of interval number" << endl;
        result = false;
    }

    // Condition 01
    if (comparedToCLEVER)
    {
        float CLEVER = readParameters("parameters/CLEVER.csv")[0][0];
        if (convert(CLEVER) < low_max)
        {
            cout << "\nExit with condition 01: low_max > CLEVER" << endl;
            result = false;
        }
    }

    // Condition 02
    if (boxes_size > max_boxes)
    {
        cout << "\nExit with condition 02: reach max boxes number" << endl;
        result = false;
    }

    // Condition 03
    if (count >= maxIterationNum)
    {
        cout << "\nExit with condition 03: reach max iteration number" << endl;
        result = false;
    }

    cout << "Gap is " << gap << ". up_max is " << up_max << ". low_max is " << low_max << "." << endl;

    return result;
}

tuple<vector<vector<string>>, string> get_interval_Lipschitz_CPP(string file_index, float radius, int inputs_size, int hidden_neural_num, int output_size, bool comparedToCLEVER, int maxIterationNum, float minGap, int max_boxes)
{
    // read parameters
    string path = "parameters/E" + file_index;
    vector<vector<float>> X0 = readParameters(path + "/X0.csv");
    vector<vector<float>> W1 = readParameters(path + "/W1.csv");
    vector<vector<float>> W2 = readParameters(path + "/W2.csv");
    vector<float> b1 = readParameters(path + "/b1.csv")[0];
    float Pred_X0_tmp = readParameters(path + "/Pred_X0.csv")[0][0];
    int Pred_X0 = (int)round(Pred_X0_tmp);

    // convert to interval
    vector<vector<mpfi_float_1000>> I_X0 = expand_2D_matrix(X0, radius);

    tuple<mpfi_float_1000, int> results = myInterval(I_X0, Pred_X0, W1, W2, b1, inputs_size, hidden_neural_num, output_size);

    cout << "Interval result in C++: " << get<0>(results) << endl;

    vector<vector<mpfi_float_1000>> boxes = I_X0;
    mpfi_float_1000 low_max = lower(get<0>(results)).convert_to<mpfi_float_1000>();
    mpfi_float_1000 up_max = upper(get<0>(results)).convert_to<mpfi_float_1000>();
    int boxes_size = boxes.size();

    int count = 0;
    while (conditions(count, low_max, comparedToCLEVER, maxIterationNum, minGap, up_max, boxes_size, max_boxes))
    {
        count++;
        tuple<vector<vector<mpfi_float_1000>>, mpfi_float_1000> tmp = bisection(boxes, Pred_X0, W1, W2, b1, inputs_size, hidden_neural_num, output_size, count);
        boxes = get<0>(tmp);
        low_max = get<1>(tmp);
        boxes_size = boxes.size();
    }

    vector<vector<string>> boxes_return = convert_to_string_2D_array(boxes);

    cout << "\nBi-section times: " << count << endl;

    ofstream out("tmp/nB.txt", ios::app);
    if (out.is_open())
    {
        out << count << "\n";
        out.close();
    }

    float upper_bound = upper(get<0>(results)).convert_to<float>();
    float lower_bound = low_max.convert_to<float>();
    string newResult = to_string(lower_bound) + "," + to_string(upper_bound);
    cout << "\nNew interval result in C++: " << newResult << endl;

    ofstream out2("tmp/lower.txt", ios::app);
    if (out2.is_open())
    {
        out2 << lower_bound << "\n";
        out2.close();
    }

    ofstream out3("tmp/upper.txt", ios::app);
    if (out3.is_open())
    {
        out3 << upper_bound << "\n";
        out3.close();
    }

    ofstream out4("tmp/toghter.txt", ios::app);
    if (out4.is_open())
    {
        out4 << "[" << lower_bound << "," << upper_bound << "]\n";
        out4.close();
    }

    auto result_tuple = make_tuple(boxes_return, newResult);
    return result_tuple;
}

PYBIND11_MODULE(IntervalCPP_ReLU, m)
{
    m.def("get_interval_Lipschitz_CPP", &get_interval_Lipschitz_CPP);
}

// int main() {
//     get_interval_Lipschitz_CPP(4, 10, 3, false, 20, 0.0001);
//     return 0;
// }