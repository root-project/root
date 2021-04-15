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

<<<<<<< HEAD

class TFeynman : public TAttLine, public TObject {
=======
class TFeynman : public TAttLine {
>>>>>>> 7aad3df60e43a80be989dd65e792c4e4b670adc7

	public:

		// The constructor
		TFeynman(Double_t canvasWidth, Double_t canvasHeight);

<<<<<<< HEAD
    TFeynmanEntry *AddItem(const char* particleName, Double_t x1, Double_t y1, Double_t x2, Double_t y2);
=======
    TFeynmanEntry* AddItem(const char* particleName, Double_t x1, Double_t y1, Double_t x2, Double_t y2);
>>>>>>> 7aad3df60e43a80be989dd65e792c4e4b670adc7


    virtual void Draw();
		virtual void Paint();

  protected:
    TList *fPrimitives; ///< List of TFeynman entries
};
#endif
