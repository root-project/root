// @(#)root/gui:$Name:  $:$Id: TGedPropertyFrame.h,v 1.0 2003/06/24 13:41:59 rdm Exp $
// Author: Marek Biskup, Ilka Antcheva   15/07/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGedPropertyFrame
#define ROOT_TGedPropertyFrame

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGedPropertyFrame                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

class TPad;
class TGDoubleSlider;
class TGRadioButton;
class TGedAttFrame;
class TVirtualPad;
class TCanvas;


class TGedPropertyFrame : public TGCompositeFrame {

protected:
   TGedAttFrame  *fAttFrame[5];
   TObject       *fModel;
   TVirtualPad   *fPad;
   
   void   Build(void);

public:
   TGedPropertyFrame(const TGWindow *p, TCanvas* canvas = 0);
   virtual ~TGedPropertyFrame();

   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t);
   virtual void   ConnectToCanvas(TCanvas *c);
   virtual void   SetModel(TObject *t, TVirtualPad* pad);

   ClassDef(TGedPropertyFrame,0) //property frame
};

#endif
