// Author: Jakob Blomer CERN 10/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

// The RNTuple and RFileNTupleAnchor class versions must match
#pragma link C++ options=version(3) class ROOT::Experimental::Internal::RFileNTupleAnchor + ;
#pragma link C++ class ROOT::Experimental::RNTuple - ;

#endif
