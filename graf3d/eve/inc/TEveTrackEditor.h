// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveTrackEditors
#define ROOT_TEveTrackEditors

#include "TGedFrame.h"

class TGButton;
class TGCheckButton;
class TGNumberEntry;
class TGColorSelect;

class TEveGValuator;
class TEveGDoubleValuator;
class TEveTrackPropagatorSubEditor;
class TEveTrack;
class TEveTrackList;


/******************************************************************************/
// TEveTrackEditor
/******************************************************************************/

class TEveTrackEditor : public TGedFrame
{
private:
   TEveTrackEditor(const TEveTrackEditor&);            // Not implemented
   TEveTrackEditor& operator=(const TEveTrackEditor&); // Not implemented

protected:
   TEveTrack                      *fM;
   TGTextButton                   *fRSEditor;
public:
   TEveTrackEditor(const TGWindow *p = nullptr, Int_t width=170, Int_t height=30,
                   UInt_t options=kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   virtual ~TEveTrackEditor() {}

   virtual void SetModel(TObject* obj);
   void DoEditPropagator();

   ClassDef(TEveTrackEditor, 0); // Editor for TEveTrack class.
};


/******************************************************************************/
// TEveTrackListEditor
/******************************************************************************/

class TEveTrackListEditor : public TGedFrame
{
private:
   TEveTrackListEditor(const TEveTrackListEditor&);            // Not implemented
   TEveTrackListEditor& operator=(const TEveTrackListEditor&); // Not implemented

   void CreateRefTab();
protected:
   TGVerticalFrame                 *fRefs;

   TEveTrackList                   *fTC; // fModel dynamic-casted to TEveTrackListEditor

   TGCheckButton                   *fRnrLine;
   TGCheckButton                   *fRnrPoints;

   TEveGDoubleValuator             *fPtRange;
   TEveGDoubleValuator             *fPRange;

   TEveTrackPropagatorSubEditor      *fRSSubEditor;

public:
   TEveTrackListEditor(const TGWindow *p = nullptr, Int_t width=170, Int_t height=30,
                       UInt_t options=kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   virtual ~TEveTrackListEditor() {}

   void CreateRefsTab();
   virtual void SetModel(TObject* obj);

   void DoRnrLine();
   void DoRnrPoints();

   void DoPtRange();
   void DoPRange();

   ClassDef(TEveTrackListEditor, 0); // Editor for TEveTrackList class.
};

#endif
