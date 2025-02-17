// Author: Alvaro Tolosa-Delgado CERN 07/2023
// Author: Jorge Agramunt Ros IFIC(Valencia,Spain) 07/2023

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// This file declares a class, with some members and method
// The definition of the method is done in the corresponding .cpp file

#ifndef __data2Tree__
#define __data2Tree__

#include <vector>

class myDetectorData {
public:
   //-- Example of method...
   void clear();

   //-- Class members
   //-- initialized by construction, C++11
   double time = 1;
   double energy = 2;
   int detectorID = 3;
   std::vector<double> correlatedDetectors_v = {1, 2, 3};
};

#endif
