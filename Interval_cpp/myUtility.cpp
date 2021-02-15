#include <boost/multiprecision/mpfi.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/tokenizer.hpp>
#include <boost/regex.hpp>
#include <math.h>
#include <vector>
#include <tuple>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <iterator>
#include "myUtility.h"

using namespace std;
using namespace boost;
using namespace boost::multiprecision;

// expand
mpfi_float_1000 expand(float x, float r)
{
    mpfi_float_1000 X = {x - r, x + r};
    return X;
}

vector<mpfi_float_1000> expand_array(vector<float> arr, float r)
{
    vector<mpfi_float_1000> result;
    for (int i = 0; i < arr.size(); i++)
    {
        mpfi_float_1000 tmp = expand(arr[i], r);
        result.push_back(expand(arr[i], r));
    }
    return result;
}

vector<vector<mpfi_float_1000>> expand_2D_matrix(vector<vector<float>> matrix, float r)
{
    vector<vector<mpfi_float_1000>> result;
    for (int i = 0; i < matrix.size(); i++)
    {
        result.push_back(expand_array(matrix[i], r));
    }
    return result;
}

// convert
mpfi_float_1000 convert(float x)
{
    mpfi_float_1000 X = {x};
    return X;
}

vector<mpfi_float_1000> convert_array(vector<float> arr)
{
    vector<mpfi_float_1000> result;
    for (int i = 0; i < arr.size(); i++)
    {
        result.push_back(convert(arr[i]));
    }
    return result;
}

vector<vector<mpfi_float_1000>> convert_2D_matrix(vector<vector<float>> matrix)
{
    vector<vector<mpfi_float_1000>> result;
    for (int i = 0; i < matrix.size(); i++)
    {
        result.push_back(convert_array(matrix[i]));
    }
    return result;
}

// relu
mpfi_float_1000 relu(mpfi_float_1000 x)
{
    mpfi_float_1000 zero = {0};
    mpfi_float_1000 lo = lower(x).convert_to<mpfi_float_1000>();
    mpfi_float_1000 up = upper(x).convert_to<mpfi_float_1000>();
    if (lo > zero){
        return x;
    } else if(up < zero){
        return zero;
    } else {
        return hull(zero, up);
    }
}

// relu derivative
mpfi_float_1000 relu_d(mpfi_float_1000 x) 
{
    mpfi_float_1000 lo = lower(x).convert_to<mpfi_float_1000>();
    mpfi_float_1000 up = upper(x).convert_to<mpfi_float_1000>();
    mpfi_float_1000 one = {1};
    mpfi_float_1000 zero = {0};
    mpfi_float_1000 between = expand(0.5, 0.5);
    if (lo > zero){
        return one;
    } else if(up < zero) {
        return zero;
    } else {
        return between;
    }
}

// sigmoid
mpfi_float_1000 sigmoid(mpfi_float_1000 x)
{
    mpfi_float_1000 one = {1};
    return (one / (one + exp(-x)));
}

// max
tuple<mpfi_float_1000, int> myMax(vector<mpfi_float_1000> arr)
{
    mpfi_float_1000 max = arr[0];
    for (int i = 1; i < arr.size(); i++)
    {
        mpfi_float_1000 lo = lower(max).convert_to<mpfi_float_1000>();
        mpfi_float_1000 tmp = arr[i];
        mpfi_float_1000 tmp_lo = lower(tmp).convert_to<mpfi_float_1000>();
        if (tmp_lo > lo) {
            max = arr[i];
        }
    }
    auto result = make_tuple(max, 0);
    return result;
}

// convert string to float / helper function
float toFloat(string x)
{
    return stof(x);
}

// helper function: to repeat vector
vector<vector<mpfi_float_1000>> repeatVector(vector<vector<mpfi_float_1000>> X0)
{

    vector<vector<mpfi_float_1000>> result;

    for (int i = 0; i < X0.size(); i++)
    {
        result.push_back(X0[i]);
        result.push_back(X0[i]);
    }

    return result;
}

vector<vector<mpfi_float_1000>> getBisection(vector<mpfi_float_1000> I_X0)
{

    vector<vector<mpfi_float_1000>> new_X0;

    for (int i = 0; i < I_X0.size(); i++)
    {

        mpfi_float_1000 tmp = I_X0[i];
        mpfi_float_1000 med = median(tmp).convert_to<mpfi_float_1000>();
        mpfi_float_1000 lo = lower(tmp).convert_to<mpfi_float_1000>();
        mpfi_float_1000 up = upper(tmp).convert_to<mpfi_float_1000>();

        mpfi_float_1000 part01 = hull(lo, med);
        mpfi_float_1000 part02 = hull(med, up);

        if (new_X0.empty())
        {
            vector<mpfi_float_1000> tmp_2, tmp_3;
            tmp_2.push_back(part01);
            tmp_3.push_back(part02);
            new_X0.push_back(tmp_2);
            new_X0.push_back(tmp_3);
        }
        else
        {
            new_X0 = repeatVector(new_X0);
            for (int j = 0; j < new_X0.size(); j += 2)
            {
                new_X0[j].push_back(part01);
                new_X0[j + 1].push_back(part02);
            }
        }
    }
    return new_X0;
}


// print vector<vector<mpfi_float_1000>>
void print2DIntervalVector(vector<vector<mpfi_float_1000>> X) {
    for(int i=0; i<X.size(); i++) {
        cout << "[ ";
        for(int j=0; j<X[i].size(); j++) {
            cout << X[i][j] << " ";
        }
        cout << "]" << endl;
    }
}

// print vector<vector<float>>
void print2DFloatVector(vector<vector<float>> X) {
    for(int i=0; i<X.size(); i++) {
        cout << "[ ";
        for(int j=0; j<X[i].size(); j++) {
            cout << X[i][j] << " ";
        }
        cout << "]" << endl;
    }
}

// print vector<vector<string>>
void print2DStringVector(vector<vector<string>> X) {
    for(int i=0; i<X.size(); i++) {
        cout << "[ ";
        for(int j=0; j<X[i].size(); j++) {
            cout << X[i][j] << " ";
        }
        cout << "]" << endl;
    }
}

// read parameters
typedef tokenizer< escaped_list_separator<char> > Tokenizer;

vector<vector<float>> readParameters(string filename) {
    ifstream in(filename.c_str());
    if(!in.is_open()) {
        cout << "Error: Cannot read file!" << endl;
        cout << filename << endl;
        exit(1);
    }

    vector<string> vec;
    string line;
    vector<float> tmp;
    vector<vector<float>> result;

    while(getline(in, line)) {
        tmp.clear();
        Tokenizer tok(line);
        vec.assign(tok.begin(), tok.end());

        vector<string>::iterator iter;
        for(iter = vec.begin(); iter != vec.end(); iter++) {
            tmp.push_back(stof(*iter));
        }
        result.push_back(tmp);
    }
    
    return result;
}

tuple<mpfi_float_1000, mpfi_float_1000> getLowerUpper(tuple<mpfi_float_1000, int> X) {
    mpfi_float_1000 tmp = get<0>(X);
    mpfi_float_1000 lo = lower(tmp).convert_to<mpfi_float_1000>();
    mpfi_float_1000 up = upper(tmp).convert_to<mpfi_float_1000>();

    auto results = make_tuple(lo, up);

    return results; 
}

// sort
vector<mpfi_float_1000> getSort(vector<mpfi_float_1000> X) {
    sort(X.begin(), X.end());
    return X;
}

bool compare_tuple (tuple<mpfi_float_1000, int> a, tuple<mpfi_float_1000, int> b) {
    mpfi_float_1000 tmp_a = get<0>(a);
    mpfi_float_1000 tmp_b = get<0>(b);
    return tmp_a < tmp_b;
}

vector<tuple<mpfi_float_1000, int>> getSort_tuple(vector<tuple<mpfi_float_1000, int>> X) {
    sort(X.begin(), X.end(), compare_tuple);
    return X;
}


vector<vector<string>> convert_to_string_2D_array(vector<vector<mpfi_float_1000>> array) {
    vector<vector<string>> result;
    vector<string> tmp;

    for(int i=0; i<array.size(); i++) {
        tmp.clear();
        for(int j=0; j<array[i].size(); j++) {
            mpfi_float_1000 tmp2 = array[i][j];
            string lower_bound = lower(tmp2).convert_to<string>();
            string upper_bound = upper(tmp2).convert_to<string>();
            string back = "(" + lower_bound + "," + upper_bound + ")";
            tmp.push_back(back);
        }
        result.push_back(tmp);
    }
    return(result);
}
