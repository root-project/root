// ##########################################################################
// # Alvaro Tolosa Delgado @ IFIC(Valencia,Spain)  alvaro.tolosa@ific.uv.es #
// # Copyright (c) 2018 Alvaro Tolosa. All rights reserved.		 #
// ##########################################################################

// This file contains the definition of the class method
// The class method was declared in the corresponding .hpp file

#include "data2Tree.hpp"

void myDetectorData::clear()
{
   time = 0;
   energy = 0;
   detectorID = 0;
   correlatedDetectors_v.clear();
   return;
}
