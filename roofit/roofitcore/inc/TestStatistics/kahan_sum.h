// Author: Patrick Bos, Netherlands eScience Center / NIKHEF 2021

/*****************************************************************************
 * RooFit
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2021, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

// --- kahan summation templates ---

#ifndef ROOT_ROOFIT_kahan_sum
#define ROOT_ROOFIT_kahan_sum

#include <map>

namespace RooFit {

template <typename C>
typename C::value_type sum_kahan(const C& container) {
   using ValueType = typename C::value_type;
   ValueType sum = 0, carry = 0;
   for (auto element : container) {
      ValueType y = element - carry;
      ValueType t = sum + y;
      carry = (t - sum) - y;
      sum = t;
   }
   return sum;
}

template <typename IndexType, typename ValueType>
ValueType sum_kahan(const std::map<IndexType, ValueType>& map) {
   ValueType sum = 0, carry = 0;
   for (auto const& element : map) {
      ValueType y = element.second - carry;
      ValueType t = sum + y;
      carry = (t - sum) - y;
      sum = t;
   }
   return sum;
}

template <typename C>
std::pair<typename C::value_type, typename C::value_type> sum_of_kahan_sums(const C& sum_values, const C& sum_carrys) {
   using ValueType = typename C::value_type;
   ValueType sum = 0, carry = 0;
   for (std::size_t ix = 0; ix < sum_values.size(); ++ix) {
      ValueType y = sum_values[ix];
      carry += sum_carrys[ix];
      y -= carry;
      const ValueType t = sum + y;
      carry = (t - sum) - y;
      sum = t;
   }
   return std::pair<ValueType, ValueType>(sum, carry);
}


template <typename IndexType, typename ValueType>
std::pair<ValueType, ValueType> sum_of_kahan_sums(const std::map<IndexType, ValueType>& sum_values, const std::map<IndexType, ValueType>& sum_carrys) {
   ValueType sum = 0, carry = 0;
   assert(sum_values.size() == sum_carrys.size());
   auto it_values = sum_values.cbegin();
   auto it_carrys = sum_carrys.cbegin();
   for (; it_values != sum_values.cend(); ++it_values, ++it_carrys) {
      ValueType y = it_values->second;
      carry += it_carrys->second;
      y -= carry;
      const ValueType t = sum + y;
      carry = (t - sum) - y;
      sum = t;
   }
   return std::pair<ValueType, ValueType>(sum, carry);
}

std::tuple<double, double> kahan_add(double sum, double additive, double carry);

}

#endif // ROOT_ROOFIT_kahan_sum
