#pragma once

/// <summary>
/// 
/// </summary>
#include <stdio.h>
#include <tchar.h>
#include <ctime>
#include <iostream>
#include <vector>
#include <memory>


/// <summary>
/// 
/// </summary>
#include "Eigen/Eigen"


/// <summary>
/// 
/// </summary>
#include "mnist/mnist_reader_less.hpp"

template<typename T>
using sptr = std::shared_ptr < T > ;

template<typename T>
using List = std::vector < T > ;

template<typename T, typename R>
using Pair = std::pair < T, R > ;

typedef float PixelType;
typedef int AnswerType;
typedef Eigen::MatrixXf ActiveType;

typedef List<Pair<ActiveType, AnswerType> > DataType;


typedef List<ActiveType> BiasType;
typedef List<ActiveType> WeightType;

