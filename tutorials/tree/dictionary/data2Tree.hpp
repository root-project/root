// ##########################################################################
// # Alvaro Tolosa Delgado @ IFIC(Valencia,Spain)  alvaro.tolosa@ific.uv.es #
// # Copyright (c) 2018 Alvaro Tolosa. All rights reserved.		 #
// ##########################################################################

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
