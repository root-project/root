// @(#)root/ged:$Name:  $:$Id: TGedAttFrame.h,v 1.2 2004/02/19 15:36:45 brun Exp $
// Author: Marek Biskup ,Ilka  Antcheva 28/07/03
// ***It nedds more fixes ***
//
/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGedAttFrame
#define ROOT_TGedAttFrame

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TGedAttFrame, TGedAttNameFrame, TGedAttFillFrame,                   //
//  TGedAttLineFrame, TGedAttTextFrame, TGedAttMarkerFrame              //
//                                                                      //
//  Frames with object attributes, just like on TAttCanvases.           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGButton
#include "TGWidget.h"
#endif
#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

class TPad;
class TCanvas;
class TGColorSelect;
class TGedPatternSelect;
class TGedMarkerSelect;
class TGLabel;
class TGLineStyleComboBox;
class TGLineWidthComboBox;
class TGComboBox;
class TGFontTypeComboBox;


class TGedAttFrame : public TGCompositeFrame, public TGWidget {

protected:
   TObject        *fModel;       // selected object, if exists
   TCanvas        *fCanvas;      // selected canvas, if exists
   TPad           *fPad;         // selected pad, if exists

   long    ExecuteInt(TObject *obj, const char *method, const char *params);
   Float_t ExecuteFloat(TObject *obj, const char *method, const char *params);
   virtual TGCompositeFrame *MakeTitle(char *c);

public:
   TGedAttFrame(const TGWindow *p, Int_t id,
                Int_t width = 140, Int_t height = 30,
                UInt_t options = kChildFrame,
                Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGedAttFrame() { }

   virtual void SetActive(Bool_t active = true);
   virtual void SetModel(TPad *pad, TObject *obj, Int_t event) = 0;
   virtual void Refresh();
   virtual void ConnectToCanvas(TCanvas *c);

   ClassDef(TGedAttFrame, 0); //attribute frame
};


class TGedAttNameFrame : public TGedAttFrame {

protected:
   TGLabel        *fLabel;

public:
   TGedAttNameFrame(const TGWindow *p, Int_t id,
                    Int_t width = 140, Int_t height = 30,
                    UInt_t options = kChildFrame,
                    Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGedAttNameFrame() { Cleanup(); }

   virtual void  SetModel(TPad *pad, TObject *obj, Int_t event);

   ClassDef(TGedAttNameFrame,0)  //name attribute farme
};


class TGedAttFillFrame : public TGedAttFrame {

protected:
   TGColorSelect       *fColorSelect;
   TGedPatternSelect   *fPatternSelect;

public:
   TGedAttFillFrame(const TGWindow *p, Int_t id,
                    Int_t width = 140, Int_t height = 30,
                    UInt_t options = kChildFrame,
                    Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGedAttFillFrame() { Cleanup(); }

   virtual void   SetModel(TPad *pad, TObject *obj, Int_t event);
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

   ClassDef(TGedAttFillFrame,0)  //fill attribute frame
};


class TGedAttLineFrame : public TGedAttFrame {

protected:
   TGLineStyleComboBox  *fStyleCombo;
   TGLineWidthComboBox  *fWidthCombo;
   TGColorSelect        *fColorSelect;

public:
   TGedAttLineFrame(const TGWindow *p, Int_t id,
                    Int_t width = 140, Int_t height = 30,
                    UInt_t options = kChildFrame,
                    Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGedAttLineFrame() { Cleanup(); }

   virtual void   SetModel(TPad *pad, TObject *obj, Int_t event);
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

   ClassDef(TGedAttLineFrame,0)  // line attribute frame
};


class TGedAttTextFrame : public TGedAttFrame {

protected:
   TGFontTypeComboBox  *fTypeCombo;
   TGComboBox          *fSizeCombo;
   TGComboBox          *fAlignCombo;
   TGColorSelect       *fColorSelect;

   static TGComboBox *BuildFontSizeComboBox(TGFrame *parent, Int_t id);
   static TGComboBox *BuildTextAlignComboBox(TGFrame *parent, Int_t id);

public:
   TGedAttTextFrame(const TGWindow *p, Int_t id,
                    Int_t width = 140, Int_t height = 30,
                    UInt_t options = kChildFrame,
                    Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGedAttTextFrame() { Cleanup(); }

   virtual void   SetModel(TPad *pad, TObject *obj, Int_t event);
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

   ClassDef(TGedAttTextFrame,0)  //text attribute frame
};


class TGedAttMarkerFrame : public TGedAttFrame {

protected:
   TGFontTypeComboBox  *fTypeCombo;
   TGComboBox          *fSizeCombo;
   TGColorSelect       *fColorSelect;
   TGedMarkerSelect    *fMarkerSelect;

   static TGComboBox *BuildMarkerSizeComboBox(TGFrame *parent, Int_t id);

public:
   TGedAttMarkerFrame(const TGWindow *p, Int_t id,
                      Int_t width = 140, Int_t height = 30,
                      UInt_t options = kChildFrame,
                      Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGedAttMarkerFrame() { Cleanup(); }

   virtual void     SetModel(TPad *pad, TObject *obj, Int_t event);
   virtual Bool_t   ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

   ClassDef(TGedAttMarkerFrame,0)  //marker attribute farme
};

#endif
