// @(#)root/feynman:$Id$
// Author: Advait Dhingra and Oliver Couet    12/04/2021

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
 //--------------------------------------------------------------------------

 #ifndef ROOT_TFeynman
 #define ROOT_TFeynman

#include "TFeynmanEntry.h"
#include "TAttLine.h"
#include "TList.h"
#include "TVirtualPad.h"

class TFeynman : public TAttLine, public TObject {

public:

   // The constructor
   TFeynman();

   TFeynmanEntry *AddItem(const char* particleName, Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelX, Double_t labelY, const char* label);
   TFeynmanEntry *AddPair(const char *particleLabel, Double_t x, Double_t y, Double_t radius);

   virtual void   Draw( Option_t* option = "" );
   virtual void   Paint( Option_t* option = "" );

protected:

   TList *fPrimitives; ///< List of TFeynman entries

   ClassDef(TFeynman,1) // Feynman diagram

};
#endif
