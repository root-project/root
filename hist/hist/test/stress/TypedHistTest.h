// @(#)root/test:$Id$
// Authors: David Gonzalez Maline November 2008
//          Martin Storø Nyfløtt  June 2017

#ifndef ROOT_TYPEDHISTTEST_H
#define ROOT_TYPEDHISTTEST_H

#include "gtest/gtest.h"

using namespace std;

template <typename HIST>
class HistTest : public ::testing::Test
{
public:
   HIST* ProduceHist(const char* name, const char* title, Int_t dim,
                    const Int_t* nBins, const Double_t* xMin = 0,
                    const Double_t* xMax = 0)
   {
      return new HIST(name, title, dim, nBins, xMin, xMax);
   }

   const char* GetClassName()
   {
      return HIST::Class()->GetName();
   }
};

#endif //ROOT_TYPEDHISTTEST_H
