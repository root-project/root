// @(#)root/graf:$Id$
// Author: Advait Dhingra and Oliver Couet 13/04/21

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
//--------------------------------------------------------------------------

#ifndef ROOT_TFeynmanEntry
#define ROOT_TFeynmanEntry

#include "TArrow.h"
#include "TVirtualPad.h"
#include "TLatex.h"

class TFeynmanEntry : public TObject {
    public:
        TFeynmanEntry(const char* particleName, Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelX, Double_t labelY, const char* label);
        virtual const char   *GetParticleName() const { return fParticle.Data(); }
        virtual Double_t GetX1()  {return fX1;}
        virtual Double_t GetY1()  {return fY1;}
        virtual Double_t GetX2()  {return fX2;}
        virtual Double_t GetY2() const {return fY2;}
        virtual void     Paint( Option_t* option = "" );
    protected:
        TString      fParticle; ///< Name of the particle (label)
        Double_t     fX1;
        Double_t     fX2; ///< Starting Point
        Double_t     fY1;
        Double_t     fY2; ///<Stopping Point
        Double_t     fLabelX;
        Double_t     fLabelY; ///< Position of the label
        TString      fLabel; ///< Label to be displayed
};
#endif
