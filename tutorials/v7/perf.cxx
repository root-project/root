/// \file
/// \ingroup tutorial_v7
///
/// \macro_code
///
/// \date 2015-07-08
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!
/// \author Axel Naumann <axel@cern.ch>

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RHist.hxx"
#include "ROOT/RFit.hxx"
#include "ROOT/RHistBufferedFill.hxx"

#include <chrono>
#include <iostream>
#include <type_traits>

using namespace ROOT;

long createNew(int count)
{
   long ret = 1;
   for (int i = 0; i < count; ++i) {
      Experimental::RH2D hist({{{100, 0., 1.}, {{0., 1., 2., 3., 10.}}}});
      ret ^= (long)&hist;
   }
   return ret;
}

long fillNew(int count)
{
   Experimental::RH2D hist({{{100, 0., 1.}, {{0., 1., 2., 3., 10.}}}});
   for (int i = 0; i < count; ++i)
      hist.Fill({0.611, 0.611});
   return hist.GetNDim();
}

long fillN(int count)
{
   Experimental::RH2D hist({{{100, 0., 1.}, {{0., 1., 2., 3., 10.}}}});
   std::vector<Experimental::Hist::RCoordArray<2>> v(count);
   for (int i = 0; i < count; ++i)
      v[i] = {0.611, 0.611};
   hist.FillN(v);
   return hist.GetNDim();
}

long fillBufferedNew(int count)
{
   Experimental::RH2D hist({{{100, 0., 1.}, {{0., 1., 2., 3., 10.}}}});
   Experimental::RHistBufferedFill<Experimental::RH2D> filler(hist);
   for (int i = 0; i < count; ++i)
      filler.Fill({0.611, 0.611});
   return hist.GetNDim();
}

using timefunc_t = std::add_pointer_t<long(int)>;

void time1(timefunc_t run, int count, const std::string &name)
{
   using namespace std::chrono;
   auto start = high_resolution_clock::now();
   run(count);
   auto end = high_resolution_clock::now();
   duration<double> time_span = duration_cast<duration<double>>(end - start);

   std::cout << count << " * " << name << ": " << time_span.count() << "seconds \n";
}

void time(timefunc_t r7, int count, const std::string &name)
{
   time1(r7, count, name + " (ROOT7)");
}

void perf()
{
   time(createNew, 1000000, "create 2D hists");
   time(fillNew, 100000000, "2D fills");
   time(fillBufferedNew, 100000000, "2D fills (buffered)");
}
