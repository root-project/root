// @(#)root/gpad:$Name:  $:$Id: TDrawPanelHist.h,v 1.5 2005/11/23 11:03:12 couet Exp $
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
   TDrawPanelHist(const char *name, const char *title, UInt_t ww, UInt_t wh, const TVirtualPad *pad, const TObject *obj);
   virtual ~TDrawPanelHist();
   virtual void  AddOption(Option_t *option);
   virtual void  Apply(const char *action="");
   virtual void  BuildStandardButtons();
   virtual void  ExecuteEvent(Int_t event, Int_t px, Int_t py);
   TObject       *GetHistogram() const {return fHistogram;}
   virtual void  RecursiveRemove(TObject *obj);
   virtual void  SavePrimitive(ostream &out, Option_t *option = "");
   virtual void  SetDefaults();
   virtual void  SetSame();

   ClassDef(TDrawPanelHist,1)  //Class used to control histogram drawing options
};

#endif

