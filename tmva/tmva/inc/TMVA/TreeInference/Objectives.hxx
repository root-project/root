/**********************************************************************************
 * Project: ROOT - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *                                                                                *
 * Authors:                                                                       *
 *      Stefan Wunsch (stefan.wunsch@cern.ch)                                     *
 *      Luca Zampieri (luca.zampieri@alumni.epfl.ch)                              *
 *                                                                                *
 * Copyright (c) 2019:                                                            *
 *      CERN, Switzerland                                                         *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef TMVA_TREEINFERENCE_OBJECTIVES
#define TMVA_TREEINFERENCE_OBJECTIVES

#include <string>
#include <stdexcept>
#include <cmath> // std::exp
#include <functional> // std::function

namespace TMVA {
namespace Experimental {
namespace Objectives {

/// Logistic function f(x) = 1 / (1 + exp(-x))
template <typename T>
inline T Logistic(T value)
{
   return 1.0 / (1.0 + std::exp(-1.0 * value));
}

/// Identity function f(x) = x
template <typename T>
inline T Identity(T value)
{
   return value;
}

/// Natural exponential function f(x) = exp(x)
///
/// This objective is used for the softmax objective in the multiclass
/// case with the formula exp(x)/sum(exp(x)) and the vector x.
template <typename T>
inline T Exponential(T value)
{
   return std::exp(value);
}

/// Get function pointer to implementation from name given as string
template <typename T>
std::function<T(T)> GetFunction(const std::string &name)
{
   if (name.compare("identity") == 0)
      return std::function<T(T)>(Identity<T>);
   else if (name.compare("logistic") == 0)
      return std::function<T(T)>(Logistic<T>);
   else if (name.compare("softmax") == 0)
      return std::function<T(T)>(Exponential<T>);
   else
      throw std::runtime_error("Objective function with name \"" + name + "\" is not implemented.");
}

} // namespace Objectives
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_TREEINFERENCE_OBJECTIVES
