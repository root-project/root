// @(#)root/feynman:$Id$
// Author: Advait Dhingra    12/04/2021

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <iostream>
#include "TMath.h"
#include "TVector3.h"
#include "TCurlyLine.h"
#include "TFeynmanEntry.h"
#include "TObject.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TLine.h"
#include "TPolyLine.h"
#include "TMarker.h"
#include "TList.h"
#include "TVirtualPad.h"
#include "TMath.h"
#include "TROOT.h"
#include "TMultiGraph.h"
#include "TGraph.h"
#include "TH1.h"
#include "THStack.h"
#include "TCurlyLine.h"

#include "../src/TFeynmanEntry.cxx"

class TFeynman {

	public:

		// The constructor
		TFeynman(Double_t canvasWidth, Double_t canvasHeight);

    TFeynmanEntry* Add(const TObject *particle);


    virtual void Draw();
		virtual void Paint();

  protected:
    TList *fPrimitives; ///< List of TFeynman entries
};


class Boson : public TFeynman {
public:

	Boson(const char* name, Double_t x1, Double_t y1, Double_t x2, Double_t y2) {
			TCurlyLine *boson = new TCurlyLine(x1, y1, x2, y2);

			Add((TObject*)boson);
	}

	TObject* ToObject(TCurlyLine* boson);


	virtual void Paint();

	virtual void SetX1(Double_t value) {fx1 = value;};
	virtual void SetY1(Double_t value) {fy1 = value;};
	virtual void SetX2(Double_t value) {fx2 = value;};
	virtual void SetY2(Double_t value) {fy2 = value;};
protected:
	Double_t fx1;
	Double_t fy1;
	Double_t fx2;
	Double_t fy2;
}
