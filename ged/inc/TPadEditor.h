// @(#)root/ged:$Name:  TPadEditor.h
// Author: Ilka  Antcheva 24/06/04

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPadEditor
#define ROOT_TPadEditor

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TPadEditor                                                          //
//                                                                      //
//  Editor of pad/canvas objects.                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGButton
#include "TGWidget.h"
#endif
#ifndef ROOT_TGedFrame
#include "TGedFrame.h"
#endif

class TGColorSelect;
class TGCheckButton;
class TGRadioButton;
class TGLineWidthComboBox;
class TPad;


class TPadEditor : public TGedFrame {

protected:
   TPad                *fPadPointer;       // TPad object
   TGCheckButton       *fEditable;         // set pad editable
   TGCheckButton       *fCrosshair;        // set crosshair   
   TGCheckButton       *fFixedAR;          // set fixed aspect ratio
   TGCheckButton       *fGridX;            // set grid on X
   TGCheckButton       *fGridY;            // set grid on Y
   TGCheckButton       *fLogX;             // set log scale on X
   TGCheckButton       *fLogY;             // set log scale on Y
   TGCheckButton       *fLogZ;             // set log scale on Z
   TGCheckButton       *fTickX;            // set ticks on X
   TGCheckButton       *fTickY;            // set ticks on Y
   TGRadioButton       *fBmode;            // set sinken pad border mode
   TGRadioButton       *fBmode0;           // set no pad border
   TGRadioButton       *fBmode1;           // set raised pad border mode
   TGLineWidthComboBox *fBsize;            // set pad border size
   Bool_t               fInit;             // init flag 
   TGCompositeFrame    *f7;                // container frame;  
   
   virtual void ConnectSignals2Slots();
 
public:
   TPadEditor(const TGWindow *p, Int_t id,
              Int_t width = 140, Int_t height = 30,
              UInt_t options = kChildFrame,
              Pixel_t back = GetDefaultFrameBackground());
   virtual ~TPadEditor(); 

   virtual void   SetModel(TVirtualPad *pad, TObject *obj, Int_t event);
   virtual void   DoEditable(Bool_t on);
   virtual void   DoCrosshair(Bool_t on);
   virtual void   DoFixedAspectRatio(Bool_t on);
   virtual void   DoGridX(Bool_t on);
   virtual void   DoGridY(Bool_t on);
   virtual void   DoLogX(Bool_t on);
   virtual void   DoLogY(Bool_t on);
   virtual void   DoLogZ(Bool_t on);
   virtual void   DoTickX(Bool_t on);
   virtual void   DoTickY(Bool_t on);
   virtual void   DoBorderMode();
   virtual void   DoBorderSize(Int_t size);
           
   ClassDef(TPadEditor,0)  //editor of TPad objects
};

#endif
