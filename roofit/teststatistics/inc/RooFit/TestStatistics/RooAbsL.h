/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *
 * Copyright (c) 2016-2020, Netherlands eScience Center
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
#ifndef ROOT_ROOFIT_TESTSTATISTICS_RooAbsL
#define ROOT_ROOFIT_TESTSTATISTICS_RooAbsL

#include <cstddef>  // std::size_t

namespace RooFit {
namespace TestStatistics {

// forward declarations
class RooAbsPdf;
class RooAbsData;

class RooAbsL {
public:
   virtual double evaluate_partition(std::size_t, std::size_t, std::size_t) = 0;

private:
   virtual void optimize_pdf();
   RooAbsPdf *pdf;
   RooAbsData *data;
   //   RooPDFOptimizer optimizer;
};

}
}

#endif // ROOT_ROOFIT_TESTSTATISTICS_RooAbsL
