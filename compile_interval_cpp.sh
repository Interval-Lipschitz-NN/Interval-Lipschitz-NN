#!/bin/bash

c++ -O3 -w -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` Interval_cpp/myInterval_ReLU.cpp Interval_cpp/myUtility.cpp Interval_cpp/myUtility.h -o IntervalCPP_ReLU`python3-config --extension-suffix` -lmpfi -lmpfr -lgmp -lm

c++ -O3 -w -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` Interval_cpp/myInterval_Sigmoid.cpp Interval_cpp/myUtility.cpp Interval_cpp/myUtility.h -o IntervalCPP_Sigmoid`python3-config --extension-suffix` -lmpfi -lmpfr -lgmp -lm

c++ -O3 -w -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` Interval_cpp/myInterval_PDE.cpp Interval_cpp/myUtility.cpp Interval_cpp/myUtility.h -o IntervalCPP_PDE`python3-config --extension-suffix` -lmpfi -lmpfr -lgmp -lm