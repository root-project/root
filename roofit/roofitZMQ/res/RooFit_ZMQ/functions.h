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

#ifndef ZEROMQ_FUNCTIONS_H
#define ZEROMQ_FUNCTIONS_H 1

#include <cstddef> // std::size_t

namespace ZMQ {

template <class T>
std::size_t defaultSizeOf(const T &)
{
   return sizeof(T);
}

std::size_t stringLength(const char &cs);

} // namespace ZMQ

#endif // ZEROMQ_FUNCTIONS_H
