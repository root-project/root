// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveStraightLineSetEditor
#define ROOT_TEveStraightLineSetEditor

#include "TGedFrame.h"

class TGCheckButton;
class TGNumberEntry;
class TGColorSelect;

class TEveStraightLineSet;

class TEveStraightLineSetEditor : public TGedFrame
{
private:
   TEveStraightLineSetEditor(const TEveStraightLineSetEditor&);            // Not implemented
   TEveStraightLineSetEditor& operator=(const TEveStraightLineSetEditor&); // Not implemented

protected:
   TEveStraightLineSet* fM; // fModel dynamic-casted to TEveStraightLineSetEditor

   // Declare widgets
   TGCheckButton*     fRnrMarkers;
   TGCheckButton*     fRnrLines;

public:
   TEveStraightLineSetEditor(const TGWindow *p = nullptr, Int_t width=170, Int_t height=30, UInt_t options = kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   virtual ~TEveStraightLineSetEditor() {}

   virtual void SetModel(TObject* obj);

   // Declare callback/slot methods
   void DoRnrMarkers();
   void DoRnrLines();

   ClassDef(TEveStraightLineSetEditor, 0); // Editor for TEveStraightLineSet class.
};

#endif
