// @(#)root/roostats:$Id$
// Author: L. Moneta
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include "TH1.h"
#include "RooStats/HistFactory/HistRef.h"

namespace RooStats{
namespace HistFactory {

   TH1 * HistRef::CopyObject(TH1 * h) { 
      // implementation of method copying the contained pointer
      // (just use Clone)
      if (!h) return 0; 
      return (TH1*) h->Clone(); 
   } 

   void HistRef::DeleteObject(TH1 * h) {
      if (h) delete h;
   } 

}
}

   
