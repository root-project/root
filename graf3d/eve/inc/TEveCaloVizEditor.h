// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveCaloVizEditor
#define ROOT_TEveCaloVizEditor

#include "TGedFrame.h"

class TEveCaloViz;
class TGDoubleHSlider;
class TEveGValuator;
class TEveGDoubleValuator;
class TEveRGBAPaletteSubEditor;
class TGCheckButton;
class TGRadioButton;

class TGVerticalFrame;

class TEveCaloVizEditor : public TGedFrame
{
private:
   TEveCaloVizEditor(const TEveCaloVizEditor&);            // Not implemented
   TEveCaloVizEditor& operator=(const TEveCaloVizEditor&); // Not implemented

protected:
   TEveCaloViz               *fM; // Model object.

   TGRadioButton             *fPlotE;
   TGRadioButton             *fPlotEt;

   TGCheckButton             *fScaleAbs;
   TEveGValuator             *fMaxValAbs;
   TEveGValuator             *fMaxTowerH;

   TEveGDoubleValuator       *fEtaRng;
   TEveGValuator             *fPhi;
   TEveGValuator             *fPhiOffset;

   TGVerticalFrame            *fTower;
   TEveRGBAPaletteSubEditor   *fPalette;        // Palette sub-editor.x

public:
   TEveCaloVizEditor(const TGWindow* p=0, Int_t width=170, Int_t height=30,
                     UInt_t options=kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   virtual ~TEveCaloVizEditor() {}

   virtual void SetModel(TObject* obj);

   void DoMaxTowerH();
   void DoScaleAbs();
   void DoMaxValAbs();

   void DoPlot();

   void DoEtaRange();
   void DoPhi();

   void DoPalette();


   ClassDef(TEveCaloVizEditor, 0); // GUI editor for TEveCaloVizEditor.
};

#endif
