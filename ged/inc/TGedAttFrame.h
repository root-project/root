// @(#)root/ged:$Name:  $:$Id: TGedAttFrame.h,v 1.5 2004/04/15 10:09:01 brun Exp $
// Author: Marek Biskup, Ilka  Antcheva 28/07/03

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
class TGLabel;
class TGComboBox;
class TGNumberEntry;
class TGCheckButton;
class TGRadioButton;
class TGTextEntry;
class TGColorSelect;
class TGedPatternSelect;
class TGedMarkerSelect;
class TGLineStyleComboBox;
class TGLineWidthComboBox;
class TGFontTypeComboBox;


class TGedAttFrame : public TGCompositeFrame, public TGWidget {

protected:
   TObject      *fModel;       // selected object, if exists
   TCanvas      *fCanvas;      // selected canvas, if exists
   TPad         *fPad;         // selected pad, if exists

   Long_t     ExecuteInt(TObject *obj, const char *method, const char *params);
   char      *ExecuteChar(TObject *obj, const char *method, const char *params);
   Float_t    ExecuteFloat(TObject *obj, const char *method, const char *params);
   virtual    TGCompositeFrame *MakeTitle(const char *c);

public:
   TGedAttFrame(const TGWindow *p, Int_t id,
                Int_t width = 140, Int_t height = 30,
                UInt_t options = kChildFrame,
                Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGedAttFrame() { }

   virtual void SetActive(Bool_t active = true);
   virtual void SetModel(TPad *pad, TObject *obj, Int_t event) = 0;
   virtual void Refresh();
   virtual void Update();
   virtual void ConnectToCanvas(TCanvas *c);

   ClassDef(TGedAttFrame, 0); //attribute frame
};


class TGedAttNameFrame : public TGedAttFrame {

protected:
   TGLabel        *fLabel;      //label of attribute frame

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
   TGLineStyleComboBox  *fStyleCombo;       // line style combo box
   TGLineWidthComboBox  *fWidthCombo;       // line width combo box
   TGColorSelect        *fColorSelect;      // color selection widget

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
   TGFontTypeComboBox  *fTypeCombo;       // font style combo box
   TGComboBox          *fSizeCombo;       // font size combo box
   TGComboBox          *fAlignCombo;      // font aligh combo box
   TGColorSelect       *fColorSelect;     // color selection widget

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
   TGFontTypeComboBox  *fTypeCombo;       // font style combo box
   TGComboBox          *fSizeCombo;       // font size combo box
   TGColorSelect       *fColorSelect;     // color selection widget
   TGedMarkerSelect    *fMarkerSelect;    // marker selection widget

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

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TGedAttAxis, TGedAttAxisTitle, TGedAttAxisLabel                     //
//                                                                      //
//  Frames with axis, axis title and axis label attributes              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGedAttAxisFrame : public TGedAttFrame {

protected:
   TGColorSelect    *fAxisColor;     // color selection widget
   TGCheckButton    *fLogAxis;       // logarithmic check box    
   TGNumberEntry    *fTickLength;    // tick length number entry
   TGNumberEntry    *fDiv1;          // primary axis division number entry
   TGNumberEntry    *fDiv2;          // secondary axis division number entry
   TGNumberEntry    *fDiv3;          // tertiary axis division number entry
   TGCheckButton    *fOptimize;      // tick optimization check box
   TGCheckButton    *fTicksBoth;     // check box setting ticks on both axis sides
   TGCheckButton    *fMoreLog;       // more logarithmic labels check box
   Int_t             fTicksFlag;     // positive/negative ticks' flag

public:
   TGedAttAxisFrame(const TGWindow *p, Int_t id,
                    Int_t width = 140, Int_t height = 30,
                    UInt_t options = kChildFrame,
                    Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGedAttAxisFrame() { Cleanup(); }

   virtual void   SetModel(TPad *pad, TObject *obj, Int_t event);
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
   virtual void   DoTickLength();
   virtual void   DoTicks();
   virtual void   DoDivisions();
   virtual void   DoLogAxis();
   virtual void   DoMoreLog();
   
   ClassDef(TGedAttAxisFrame,0)  // axis attribute frame
};

class TGedAttAxisTitle : public TGedAttFrame {

protected:
   TGColorSelect       *fTitleColor;   // color selection widget
   TGFontTypeComboBox  *fTitleFont;    // title font combo box
   Int_t                fPrecision;    // font precision level
   TGNumberEntry       *fTitleSize;    // title size number entry
   TGNumberEntry       *fTitleOffset;  // title offset number entry
   TGCheckButton       *fCentered;     // check button for centered title
   TGCheckButton       *fRotated;      // check button for rotated title

public:
   TGedAttAxisTitle(const TGWindow *p, Int_t id,
                    Int_t width = 140, Int_t height = 30,
                    UInt_t options = kChildFrame,
                    Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGedAttAxisTitle() { Cleanup(); }

   virtual void   SetModel(TPad *pad, TObject *obj, Int_t event);
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
   virtual void   DoTitleSize();
   virtual void   DoTitleFont();
   virtual void   DoTitleOffset();
   virtual void   DoTitleCentered();
   virtual void   DoTitleRotated();
   
   ClassDef(TGedAttAxisTitle,0)  // axis title frame
};

class TGedAttAxisLabel : public TGedAttFrame {

protected:
   TGColorSelect       *fLabelColor;   // color selection widget
   TGFontTypeComboBox  *fLabelFont;    // label font combo box
   Int_t                fPrecision;    // font precision level
   TGNumberEntry       *fLabelSize;    // label size number entry
   TGNumberEntry       *fLabelOffset;  // label offset number entry
   TGCheckButton       *fNoExponent;   // check box for No exponent choice

public:
   TGedAttAxisLabel(const TGWindow *p, Int_t id,
                    Int_t width = 140, Int_t height = 30,
                    UInt_t options = kChildFrame,
                    Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGedAttAxisLabel() { Cleanup(); }

   virtual void     SetModel(TPad *pad, TObject *obj, Int_t event);
   virtual Bool_t   ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
   virtual void     DoLabelSize();
   virtual void     DoLabelFont();
   virtual void     DoLabelOffset();
   virtual void     DoNoExponent();
   
   ClassDef(TGedAttAxisLabel,0)  // axis label frame
};
#endif
