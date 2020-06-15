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

#pragma link C++ nestedtypedefs;
#pragma link C++ nestedclasses;

// Support for auto-loading in the RNTuple tutorials
#pragma link C++ class ROOT::Experimental::Detail::RFieldBase-;
#pragma link C++ class ROOT::Experimental::Detail::RFieldBase::RSchemaIterator-;
#pragma link C++ class ROOT::Experimental::RVectorField-;
#pragma link C++ class ROOT::Experimental::RNTupleReader-;
#pragma link C++ class ROOT::Experimental::RNTupleWriter-;
#pragma link C++ class ROOT::Experimental::RNTupleModel-;

#pragma link C++ class ROOT::Experimental::RNTuple+;

#endif
