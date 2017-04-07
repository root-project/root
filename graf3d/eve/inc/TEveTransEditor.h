// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveTransEditor
#define ROOT_TEveTransEditor

#include "TGedFrame.h"

class TGCheckButton;
class TGTextButton;

class TEveTrans;
class TEveGTriVecValuator;


class TEveTransSubEditor : public TGVerticalFrame
{
   TEveTransSubEditor(const TEveTransSubEditor&);            // Not implemented
   TEveTransSubEditor& operator=(const TEveTransSubEditor&); // Not implemented

protected:
   TEveTrans            *fTrans;

   TGHorizontalFrame    *fTopHorFrame;

   TGCheckButton        *fUseTrans;
   TGCheckButton        *fEditTrans;

   TGVerticalFrame      *fEditTransFrame;

   TEveGTriVecValuator  *fPos;
   TEveGTriVecValuator  *fRot;
   TEveGTriVecValuator  *fScale;

   TGCheckButton        *fAutoUpdate;
   TGTextButton         *fUpdate;

public:
   TEveTransSubEditor(TGWindow* p);
   virtual ~TEveTransSubEditor() {}

   void SetModel(TEveTrans* t);
   void SetTransFromData();

   void UseTrans();     //*SIGNAL*
   void TransChanged(); //*SIGNAL*

   void DoUseTrans();
   void DoEditTrans();
   void DoTransChanged();

   TEveGTriVecValuator*  GetPosValuator(){ return fPos;}
   TEveGTriVecValuator*  GetRotValuator(){ return fRot;}
   TEveGTriVecValuator*  GetScaleValuator(){ return fScale;}

   ClassDef(TEveTransSubEditor, 0); // Sub-editor for TEveTrans class.
};


class TEveTransEditor : public TGedFrame
{
private:
   TEveTransEditor(const TEveTransEditor&);            // Not implemented
   TEveTransEditor& operator=(const TEveTransEditor&); // Not implemented

protected:
   TEveTrans          *fM;  // Model object.
   TEveTransSubEditor *fSE; // Actual editor widget.

public:
   TEveTransEditor(const TGWindow* p=0, Int_t width=170, Int_t height=30, UInt_t options = kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   virtual ~TEveTransEditor() {}

   virtual void SetModel(TObject* obj);

   ClassDef(TEveTransEditor, 0); // Editor for TEveTrans class.
};

#endif
