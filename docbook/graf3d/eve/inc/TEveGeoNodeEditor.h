// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveGeoNodeEditor
#define ROOT_TEveGeoNodeEditor

#include "TGedFrame.h"

class TGCheckButton;
class TGNumberEntry;
class TGColorSelect;

class TEveGeoNode;
class TEveGeoTopNode;

class TEveGValuator;

class TEveGeoNodeEditor : public TGedFrame
{
   TEveGeoNodeEditor(const TEveGeoNodeEditor&);            // Not implemented
   TEveGeoNodeEditor& operator=(const TEveGeoNodeEditor&); // Not implemented

protected:
   TEveGeoNode*    fNodeRE;

   TGCheckButton*  fVizNode;
   TGCheckButton*  fVizNodeDaughters;
   TGCheckButton*  fVizVolume;
   TGCheckButton*  fVizVolumeDaughters;

public:
   TEveGeoNodeEditor(const TGWindow* p=0, Int_t width=170, Int_t height=30,
                     UInt_t options=kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   virtual ~TEveGeoNodeEditor() {}

   virtual void SetModel(TObject* obj);

   void DoVizNode();
   void DoVizNodeDaughters();
   void DoVizVolume();
   void DoVizVolumeDaughters();

   ClassDef(TEveGeoNodeEditor, 0); // Editor for TEveGeoNode class.
};

/******************************************************************************/

class TEveGeoTopNodeEditor : public TGedFrame
{
   TEveGeoTopNodeEditor(const TEveGeoTopNodeEditor&);            // Not implemented
   TEveGeoTopNodeEditor& operator=(const TEveGeoTopNodeEditor&); // Not implemented

protected:
   TEveGeoTopNode* fTopNodeRE;

   TEveGValuator*  fVisOption;
   TEveGValuator*  fVisLevel;
   TEveGValuator*  fMaxVisNodes;

public:
   TEveGeoTopNodeEditor(const TGWindow* p=0, Int_t width=170, Int_t height=30,
                        UInt_t options=kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   virtual ~TEveGeoTopNodeEditor() {}

   virtual void SetModel(TObject* obj);

   void DoVisOption();
   void DoVisLevel();
   void DoMaxVisNodes();

   ClassDef(TEveGeoTopNodeEditor, 0); // Editor for TEveGeoTopNode class.
};

#endif
