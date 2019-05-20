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

// Support for auto-loading in the RForest tutorials
#pragma link C++ class ROOT::Experimental::Detail::RFieldBase-;
#pragma link C++ class ROOT::Experimental::Detail::RFieldBase::RIterator-;
#pragma link C++ class ROOT::Experimental::RFieldVector-;
#pragma link C++ class ROOT::Experimental::RInputForest-;
#pragma link C++ class ROOT::Experimental::ROutputForest-;
#pragma link C++ class ROOT::Experimental::RNTupleModel-;

#pragma link C++ class ROOT::Experimental::Internal::RForestHeader+;
#pragma link C++ class ROOT::Experimental::Internal::RForestFooter+;
#pragma link C++ class ROOT::Experimental::Internal::RFieldHeader+;
#pragma link C++ class ROOT::Experimental::Internal::RColumnHeader+;
#pragma link C++ class ROOT::Experimental::Internal::RClusterFooter+;
#pragma link C++ class ROOT::Experimental::Internal::RPageInfo+;
#pragma link C++ class ROOT::Experimental::Internal::RPagePayload+;

#endif
