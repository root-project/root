// @(#)root/table:$Id$
// Author: Valery Fine(fine@bnl.gov)   13/03/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TColumnView
#define ROOT_TColumnView
 
#include "TChair.h"
 
//______________________________________________________________________________
//
//  TColumnView - helper class to browse the TTable objects via ROOT TBrowser
//______________________________________________________________________________
class TColumnView : public TChair {

public:
   TColumnView(const char *colName="", TTable *table=0); 
   virtual        ~TColumnView();
   virtual void    Browse(TBrowser *b);
   TH1            *Histogram(const char *selection=""); // *MENU*
   virtual Bool_t  IsFolder() const;
   ClassDef(TColumnView,0) // Helper to represent one TTable column
};

#endif
