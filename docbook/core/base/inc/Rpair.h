// @(#)root/base:$Id$
// Author: Philippe Canal    12/3/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun, Fons Rademakers, and al.          *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_Rpair
#define ROOT_Rpair

// Include the definition of pairs
#include <utility>

// Import pairs (and string) into the global namespace to satisfy
// the current CINT implementation of dictionary generation.
#if defined(R__SOLARIS)
using std::pair;
using std::string;
#else
namespace std {}
using namespace std;
#endif


#endif


