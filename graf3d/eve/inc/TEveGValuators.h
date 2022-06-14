// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveGValuators
#define ROOT_TEveGValuators

#include "TGNumberEntry.h"

class TGLabel;
class TGHSlider;
class TGDoubleHSlider;

class TEveGValuatorBase: public TGCompositeFrame, public TGWidget
{
   TEveGValuatorBase(const TEveGValuatorBase&);            // Not implemented
   TEveGValuatorBase& operator=(const TEveGValuatorBase&); // Not implemented

protected:
   UInt_t      fLabelWidth;
   Bool_t      fAlignRight;
   Bool_t      fShowSlider;

   Int_t       fNELength; // Number-entry length (in characters)
   Int_t       fNEHeight; // Number-entry height (in pixels)

   TGLabel*    fLabel;

public:
   TEveGValuatorBase(const TGWindow *p, const char* title, UInt_t w, UInt_t h, Int_t widgetId=-1);
   virtual ~TEveGValuatorBase() {}

   virtual void Build(Bool_t connect=kTRUE) = 0;

   void SetLabelWidth(Int_t w)        { fLabelWidth = w; }
   void SetAlignRight(Bool_t a)       { fAlignRight = a; }
   void SetShowSlider(Bool_t s=kTRUE) { fShowSlider = s; }

   void SetNELength(Int_t l)          { fNELength = l; }
   void SetNEHeight(Int_t h)          { fNEHeight = h; }

   TGLabel* GetLabel() {return fLabel;}

   ClassDef(TEveGValuatorBase, 0); // Base class for composite GUI elements for setting of numeric values.
};


/******************************************************************************/

class TEveGValuator: public TEveGValuatorBase
{
   TEveGValuator(const TEveGValuator&);            // Not implemented
   TEveGValuator& operator=(const TEveGValuator&); // Not implemented

protected:
   Float_t        fValue;
   Float_t        fMin;
   Float_t        fMax;

   Bool_t         fSliderNewLine;
   Int_t          fSliderDivs;
   TGNumberEntry* fEntry;
   TGHSlider*     fSlider;

   Int_t CalcSliderPos(Float_t v);

public:
   TEveGValuator(const TGWindow *p, const char* title, UInt_t w, UInt_t h, Int_t widgetId=-1);
   virtual ~TEveGValuator() {}

   virtual void Build(Bool_t connect=kTRUE);

   Float_t GetValue() const { return fValue; }
   virtual void SetValue(Float_t v, Bool_t emit=kFALSE);

   void SliderCallback();
   void EntryCallback();
   void ValueSet(Double_t); //*SIGNAL*

   TGHSlider*     GetSlider() { return fSlider; }
   TGNumberEntry* GetEntry()  { return fEntry; }

   void SetSliderNewLine(Bool_t nl) { fSliderNewLine = nl; }

   void GetLimits(Float_t& min, Float_t& max) const { min = fMin; max = fMax; }
   Float_t GetLimitMin() const { return fMin; }
   Float_t GetLimitMax() const { return fMax; }
   void SetLimits(Int_t min, Int_t max);
   void SetLimits(Float_t min, Float_t max, Int_t npos,
                  TGNumberFormat::EStyle nef=TGNumberFormat::kNESRealTwo);

   void SetToolTip(const char* tip);
   void SetEnabled(Bool_t state);

   ClassDef(TEveGValuator, 0); // Composite GUI element for single value selection (supports label, number-entry and slider).
};


/******************************************************************************/

class TEveGDoubleValuator: public TEveGValuatorBase
{
   TEveGDoubleValuator(const TEveGDoubleValuator&);            // Not implemented
   TEveGDoubleValuator& operator=(const TEveGDoubleValuator&); // Not implemented

protected:
   TGNumberEntry*    fMinEntry;
   TGNumberEntry*    fMaxEntry;
   TGDoubleHSlider*  fSlider;

public:
   TEveGDoubleValuator(const TGWindow *p, const char* title, UInt_t w, UInt_t h, Int_t widgetId=-1);
   virtual ~TEveGDoubleValuator() {}

   virtual void Build(Bool_t connect=kTRUE);

   void MinEntryCallback();
   void MaxEntryCallback();
   void SliderCallback();
   void ValueSet(); //*SIGNAL*

   TGDoubleHSlider* GetSlider()   { return fSlider; }
   TGNumberEntry*   GetMinEntry() { return fMinEntry; }
   TGNumberEntry*   GetMaxEntry() { return fMaxEntry; }

   void SetLimits(Int_t min, Int_t max);
   void SetLimits(Float_t min, Float_t max, TGNumberFormat::EStyle nef=TGNumberFormat::kNESRealTwo);
   void SetValues(Float_t min, Float_t max, Bool_t emit=kFALSE);

   void GetValues(Float_t& min, Float_t& max) const
   { min = fMinEntry->GetNumber(); max = fMaxEntry->GetNumber(); }
   Float_t GetMin() const { return fMinEntry->GetNumber(); }
   Float_t GetMax() const { return fMaxEntry->GetNumber(); }
   Float_t GetLimitMin() const { return fMinEntry->GetNumMin(); }
   Float_t GetLimitMax() const { return fMaxEntry->GetNumMax(); }

   ClassDef(TEveGDoubleValuator, 0); // Composite GUI element for selection of range (label, two number-entries and double-slider).
};


/******************************************************************************/

class TEveGTriVecValuator : public TGCompositeFrame, public TGWidget
{
   TEveGTriVecValuator(const TEveGTriVecValuator&);            // Not implemented
   TEveGTriVecValuator& operator=(const TEveGTriVecValuator&); // Not implemented

protected:
   TEveGValuator* fVal[3];

   // Weed-size vars from TEveGValuator; copied.
   UInt_t      fLabelWidth;
   Int_t       fNELength; // Number-entry length (in characters)
   Int_t       fNEHeight; // Number-entry height (in pixels)

public:
   TEveGTriVecValuator(const TGWindow *p, const char* name, UInt_t w, UInt_t h, Int_t widgetId=-1);
   virtual ~TEveGTriVecValuator() {}

   void Build(Bool_t vertical, const char* lab0, const char* lab1, const char* lab2);

   TEveGValuator* GetValuator(Int_t i) const { return fVal[i]; }

   Float_t GetValue(Int_t i) const   { return fVal[i]->GetValue(); }
   void SetValue(Int_t i, Float_t v) { fVal[i]->SetValue(v); }

   void GetValues(Float_t& v0, Float_t& v1, Float_t& v2) const
   { v0 = GetValue(0); v1 = GetValue(1); v2 = GetValue(2); }
   void GetValues(Float_t v[3]) const
   { v[0] = GetValue(0); v[1] = GetValue(1); v[2] = GetValue(2); }
   void GetValues(Double_t v[3]) const
   { v[0] = GetValue(0); v[1] = GetValue(1); v[2] = GetValue(2); }

   void SetValues(Float_t v0, Float_t v1, Float_t v2)
   { SetValue(0, v0); SetValue(1, v1); SetValue(2, v2); }
   void SetValues(Float_t v[3])
   { SetValue(0, v[0]); SetValue(1, v[1]); SetValue(2, v[2]); }
   void SetValues(Double_t v[3])
   { SetValue(0, v[0]); SetValue(1, v[1]); SetValue(2, v[2]); }

   void ValueSet(); //*SIGNAL*

                      // Weed-size vars from TEveGValuator; copied.
   void SetLabelWidth(Int_t w) { fLabelWidth = w; }
   void SetNELength(Int_t l)   { fNELength   = l; }
   void SetNEHeight(Int_t h)   { fNEHeight   = h; }

   void SetLimits(Int_t min, Int_t max);
   void SetLimits(Float_t min, Float_t max,
                  TGNumberFormat::EStyle nef=TGNumberFormat::kNESRealTwo);

   ClassDef(TEveGTriVecValuator, 0); // Composite GUI element for setting three numerical values (label, three number-entries).
};

#endif
