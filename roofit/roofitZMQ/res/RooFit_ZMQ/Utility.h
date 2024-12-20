/*
 * Project: RooFit
 * Authors:
 *   RA, Roel Aaij, NIKHEF
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef SERIALIZE_UTILITY_H
#define SERIALIZE_UTILITY_H 1

#include <type_traits>

namespace ZMQ {
namespace Detail {

template <class T>
using simple_object = std::is_trivially_copyable<T>;

// is trivial
template <class T>
struct is_trivial
   : std::conditional<simple_object<typename std::decay<T>::type>::value, std::true_type, std::false_type>::type {
};

} // namespace Detail
} // namespace ZMQ

#endif // SERIALIZE_UTILITY_H
