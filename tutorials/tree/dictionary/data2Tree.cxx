// Author: Alvaro Tolosa-Delgado CERN 07/2023
// Author: Jorge Agramunt Ros IFIC(Valencia,Spain) 07/2023

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// This file contains the definition of the class method
// The class method was declared in the corresponding .hpp file

#include "data2Tree.hxx"

void myDetectorData::clear()
{
   time = 0;
   energy = 0;
   detectorID = 0;
   correlatedDetectors_v.clear();
}
