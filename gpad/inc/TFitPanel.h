// @(#)root/gpad:$Name$:$Id$
// Author: Rene Brun   24/11/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFitPanel
#define ROOT_TFitPanel


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFitPanel                                                            //
//                                                                      //
// Class used to control histograms fit panel                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////



#ifndef ROOT_TDialogCanvas
#include "TDialogCanvas.h"
#endif

class TSlider;
class TH1;

class TFitPanel : public TDialogCanvas {

protected:
        TString     fOption;     //Fitting options
        TString     fFunction;   //Function to fit
        TString     fSame;       //graphics option to superimpose new fit on existing picture
        TSlider     *fSlider;    //Pointer to fitpanel slider
        TObject     *fObjectFit; //Pointer to object to fit

public:
        TFitPanel();
        TFitPanel(const char *name, const char *title, UInt_t ww=300, UInt_t wh=400);
        virtual ~TFitPanel();
        virtual void  AddOption(Option_t *option);
        virtual void  Apply(const char *action="");
        virtual void  BuildStandardButtons();
        virtual void  ExecuteEvent(Int_t event, Int_t px, Int_t py);
        TObject       *GetObjectFit() {return fObjectFit;}
        virtual void  SavePrimitive(ofstream &out, Option_t *option);
        virtual void  SetDefaults();
        virtual void  SetFunction(const char *function);
        virtual void  SetSame();

        ClassDef(TFitPanel,1)  //Class used to control histograms fit panel
};

#endif

