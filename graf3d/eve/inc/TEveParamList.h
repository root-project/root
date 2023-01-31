// @(#)root/eve:$Id$
// Author: Dmytro Kovalskyi, 28.2.2008

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveParamList
#define ROOT_TEveParamList

#include "TEveElement.h"
#include "TQObject.h"
#include <vector>

//==============================================================================
//==============================================================================
// TEveParamList
//==============================================================================

class TEveParamList : public TEveElement,
                      public TNamed,
                      public TQObject
{
   friend class TEveParamListEditor;

public:
   struct FloatConfig_t
   {
      Float_t   fValue, fMin, fMax;
      TString   fName;
      Bool_t    fSelector;

      FloatConfig_t(TString name, Double_t value, Double_t min, Double_t max, Bool_t selector = kFALSE):
         fValue(value), fMin(min), fMax(max), fName(name), fSelector(selector) {}
      FloatConfig_t(): fValue(0), fMin(0), fMax(0), fName(""), fSelector(kFALSE) {}
   };
   typedef std::vector<FloatConfig_t>       FloatConfigVec_t;
   typedef FloatConfigVec_t::iterator       FloatConfigVec_i;
   typedef FloatConfigVec_t::const_iterator FloatConfigVec_ci;

   struct IntConfig_t
   {
      Int_t     fValue, fMin, fMax;
      TString   fName;
      Bool_t    fSelector;

      IntConfig_t(TString name, Int_t value, Int_t min, Int_t max, Bool_t selector=kFALSE) :
         fValue(value), fMin(min), fMax(max), fName(name), fSelector(selector) {}
      IntConfig_t() : fValue(0), fMin(0), fMax(0), fName(""), fSelector(kFALSE) {}
   };
   typedef std::vector<IntConfig_t>       IntConfigVec_t;
   typedef IntConfigVec_t::iterator       IntConfigVec_i;
   typedef IntConfigVec_t::const_iterator IntConfigVec_ci;

   struct BoolConfig_t
   {
      Bool_t    fValue;
      TString   fName;

      BoolConfig_t(TString name, Bool_t value): fValue(value), fName(name) {}
      BoolConfig_t() : fValue(kFALSE), fName("") {}
   };
   typedef std::vector<BoolConfig_t>       BoolConfigVec_t;
   typedef BoolConfigVec_t::iterator       BoolConfigVec_i;
   typedef BoolConfigVec_t::const_iterator BoolConfigVec_ci;

private:
   TEveParamList(const TEveParamList&);            // Not implemented
   TEveParamList& operator=(const TEveParamList&); // Not implemented

protected:
   Color_t              fColor;
   FloatConfigVec_t     fFloatParameters;
   IntConfigVec_t       fIntParameters;
   BoolConfigVec_t      fBoolParameters;

public:
   TEveParamList(const char* n="TEveParamList", const char* t="", Bool_t doColor=kFALSE);
   virtual ~TEveParamList() {}

   void AddParameter(const FloatConfig_t& parameter) { fFloatParameters.push_back(parameter); }
   void AddParameter(const IntConfig_t& parameter)   { fIntParameters.push_back(parameter); }
   void AddParameter(const BoolConfig_t& parameter)  { fBoolParameters.push_back(parameter); }

   const FloatConfigVec_t&  GetFloatParameters() { return fFloatParameters; }
   const IntConfigVec_t&    GetIntParameters()   { return fIntParameters; }
   const BoolConfigVec_t&   GetBoolParameters()  { return fBoolParameters; }

   FloatConfig_t    GetFloatParameter(const TString& name);
   IntConfig_t      GetIntParameter  (const TString& name);
   Bool_t           GetBoolParameter (const TString& name);

   void ParamChanged(const char* name); // *SIGNAL*

   ClassDef(TEveParamList, 0); // Eve element to store generic configuration information.
};


//==============================================================================
//==============================================================================
// TEveParamListEditor
//==============================================================================

#include "TGedFrame.h"

class TGButton;
class TGCheckButton;
class TGNumberEntry;
class TGColorSelect;

class TEveGValuator;
class TEveGDoubleValuator;

class TEveParamList;

class TGNumberEntry;

class TEveParamListEditor : public TGedFrame
{
private:
   TEveParamListEditor(const TEveParamListEditor&);            // Not implemented
   TEveParamListEditor& operator=(const TEveParamListEditor&); // Not implemented

protected:
   TEveParamList                 *fM; // Model object.
   TGVerticalFrame               *fParamFrame;
   std::vector<TGNumberEntry*>    fIntParameters;
   std::vector<TGNumberEntry*>    fFloatParameters;
   std::vector<TGCheckButton*>    fBoolParameters;

   virtual void InitModel(TObject* obj);

public:
   TEveParamListEditor(const TGWindow *p = nullptr, Int_t width=170, Int_t height=30,
         UInt_t options=kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   virtual ~TEveParamListEditor() {}

   virtual void SetModel(TObject* obj);

   // Declare callback/slot methods
   void DoIntUpdate();
   void DoFloatUpdate();
   void DoBoolUpdate();

   ClassDef(TEveParamListEditor, 0); // GUI editor for TEveParamList.
};
#endif
