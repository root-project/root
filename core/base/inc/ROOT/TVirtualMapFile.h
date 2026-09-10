// @(#)root/io:$Id$
// Author: Philippe Canal March 2024

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVirtualMapFile
#define ROOT_TVirtualMapFile

#include "TObject.h"

/**
\class TVirtualMapFile
\ingroup Base

Abstract base class for TMapFile

This allows Core to handle TMapFile which is implemented in RIO

*/

class TVirtualMapFile : public TObject {
public:
   virtual void Close(Option_t *option = "") = 0;

   ClassDefOverride(TVirtualMapFile, 0); // Base of TMapFile
};

#endif
