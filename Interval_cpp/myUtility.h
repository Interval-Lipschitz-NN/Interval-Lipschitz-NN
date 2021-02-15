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

using namespace std;
using namespace boost;
using namespace boost::multiprecision;

mpfi_float_1000 expand(float x, float r);

vector<mpfi_float_1000> expand_array(vector<float> arr, float r);

vector<vector<mpfi_float_1000>> expand_2D_matrix(vector<vector<float>> matrix, float r);

mpfi_float_1000 convert(float x);

vector<mpfi_float_1000> convert_array(vector<float> arr);

vector<vector<mpfi_float_1000>> convert_2D_matrix(vector<vector<float>> matrix);

mpfi_float_1000 relu(mpfi_float_1000 x);

mpfi_float_1000 relu_d(mpfi_float_1000 x);

mpfi_float_1000 sigmoid(mpfi_float_1000 x);

tuple<mpfi_float_1000, int> myMax(vector<mpfi_float_1000> arr);

vector<vector<mpfi_float_1000>> repeatVector(vector<vector<mpfi_float_1000>> X0);

vector<vector<mpfi_float_1000>> getBisection(vector<mpfi_float_1000> I_X0);

void print2DIntervalVector(vector<vector<mpfi_float_1000>> X);

void print2DFloatVector(vector<vector<float>> X);

void print2DStringVector(vector<vector<string>> X);

vector<vector<float>> readParameters(string filename);

tuple<mpfi_float_1000, mpfi_float_1000> getLowerUpper(tuple<mpfi_float_1000, int> X);

vector<mpfi_float_1000> getSort(vector<mpfi_float_1000> X);

vector<tuple<mpfi_float_1000, int>> getSort_tuple(vector<tuple<mpfi_float_1000, int>> X);

vector<vector<string>> convert_to_string_2D_array(vector<vector<mpfi_float_1000>> array);