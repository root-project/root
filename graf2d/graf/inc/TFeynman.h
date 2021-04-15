// @(#)root/feynman:$Id$
// Author: Advait Dhingra    12/04/2021

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

 #ifndef ROOT_TFeynman
 #define ROOT_TFeynman



#include "TFeynmanEntry.h"
#include "TAttLine.h"
#include "TList.h"
#include "TVirtualPad.h"
#include "TPad.h"

class TFeynman : public TAttLine, public TObject {

	public:

		// The constructor
		TFeynman(Double_t canvasWidth, Double_t canvasHeight);

    TFeynmanEntry *AddItem(const char* particleName, Double_t x1, Double_t y1, Double_t x2, Double_t y2);


    virtual void Draw();
		virtual void Paint();

  protected:
    TList *fPrimitives; ///< List of TFeynman entries
};
#endif
