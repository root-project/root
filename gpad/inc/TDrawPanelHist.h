// @(#)root/gpad:$Name$:$Id$
// Author: Rene Brun   26/11/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDrawPanelHist
#define ROOT_TDrawPanelHist


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TDrawPanelHist                                                       //
//                                                                      //
// Class used to control histogram drawing options                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////



#ifndef ROOT_TDialogCanvas
#include "TDialogCanvas.h"
#endif

class TSlider;
class TH1;

class TDrawPanelHist : public TDialogCanvas {

protected:
        TString     fOption;     //Fitting options
        TSlider     *fSlider;    //Pointer to fitpanel slider
        TObject     *fHistogram; //Pointer to histogram

public:
        TDrawPanelHist();
        TDrawPanelHist(const char *name, const char *title, UInt_t ww=500, UInt_t wh=600);
        virtual ~TDrawPanelHist();
        virtual void  AddOption(Option_t *option);
        virtual void  Apply(const char *action="");
        virtual void  BuildStandardButtons();
        virtual void  ExecuteEvent(Int_t event, Int_t px, Int_t py);
        TObject       *GetHistogram() {return fHistogram;}
        virtual void  SavePrimitive(ofstream &out, Option_t *option);
        virtual void  SetDefaults();
        virtual void  SetSame();

        ClassDef(TDrawPanelHist,1)  //Class used to control histogram drawing options
};

#endif

