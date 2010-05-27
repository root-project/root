// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveTrackPropagatorEditor
#define ROOT_TEveTrackPropagatorEditor

#include "TGedFrame.h"

class TGButton;
class TGCheckButton;
class TGNumberEntry;
class TGColorSelect;
class TGComboBox;
class TGLineWidthComboBox;
class TGLineStyleComboBox;

class TAttMarkerEditor;

class TEveTrackPropagator;

class TEveGValuator;
class TEveGDoubleValuator;
class TEveTrackPropagatorSubEditor;

class TEveTrackPropagatorSubEditor : public TGVerticalFrame
{
   friend class TEveTrackPropagatorEditor;
   friend class TEveTrackListEditor;

private:
   TEveTrackPropagatorSubEditor(const TEveTrackPropagatorSubEditor&);            // Not implemented
   TEveTrackPropagatorSubEditor& operator=(const TEveTrackPropagatorSubEditor&); // Not implemented

protected:
   TEveTrackPropagator *fM;

   TEveGValuator      *fMaxR;
   TEveGValuator      *fMaxZ;
   TEveGValuator      *fMaxOrbits;
   TEveGValuator      *fMaxAng;
   TEveGValuator      *fDelta;

   TGCompositeFrame   *fRefsCont;
   TGCompositeFrame   *fPMFrame;

   TGButton           *fFitDaughters;
   TGButton           *fFitReferences;
   TGButton           *fFitDecay;
   TGButton           *fFitCluster2Ds;
   TGButton           *fRnrDaughters;
   TGButton           *fRnrReferences;
   TGButton           *fRnrDecay;
   TGButton           *fRnrCluster2Ds;
   TGButton           *fRnrFV;

   TAttMarkerEditor   *fPMAtt;
   TAttMarkerEditor   *fFVAtt;

   TGComboBox         *fProjTrackBreaking;
   TGButton           *fRnrPTBMarkers;
   TAttMarkerEditor   *fPTBAtt;

public:
   TEveTrackPropagatorSubEditor(const TGWindow* p);
   virtual ~TEveTrackPropagatorSubEditor() {}

   void SetModel(TEveTrackPropagator* m);

   void Changed(); //*SIGNAL*

   void DoMaxR();
   void DoMaxZ();
   void DoMaxOrbits();
   void DoMaxAng();
   void DoDelta();

   void DoFitPM();
   void DoRnrPM();

   void DoRnrFV();

   void DoModePTB(UChar_t mode);
   void DoRnrPTB();

   void CreateRefsContainer(TGVerticalFrame* p);

   ClassDef(TEveTrackPropagatorSubEditor, 0); // Sub-editor for TEveTrackPropagator class.
};

/******************************************************************************/
// TEveTrackPropagatorEditor
/******************************************************************************/

class TEveTrackPropagatorEditor : public TGedFrame
{
private:
   TEveTrackPropagatorEditor(const TEveTrackPropagatorEditor&);            // Not implemented
   TEveTrackPropagatorEditor& operator=(const TEveTrackPropagatorEditor&); // Not implemented

   void CreateRefTab();
protected:
   TEveTrackPropagator           *fM;           // Model object.
   TEveTrackPropagatorSubEditor  *fRSSubEditor; // Render-style sub-editor.

public:
   TEveTrackPropagatorEditor(const TGWindow* p=0, Int_t width=170, Int_t height=30,
                             UInt_t options=kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   virtual ~TEveTrackPropagatorEditor() {}

   virtual void SetModel(TObject* obj);

   ClassDef(TEveTrackPropagatorEditor, 1); // Editor for TEveTrackPropagator class.
};

#endif
