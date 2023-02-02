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
class TEveCalo3D;
class TGDoubleHSlider;
class TEveGValuator;
class TEveGDoubleValuator;
class TGCheckButton;
class TGRadioButton;
class TGNumberEntry;

class TGVerticalFrame;

class TEveCaloVizEditor : public TGedFrame
{
private:
   TEveCaloVizEditor(const TEveCaloVizEditor&);            // Not implemented
   TEveCaloVizEditor& operator=(const TEveCaloVizEditor&); // Not implemented
   void MakeSliceInfo();

protected:
   TEveCaloViz               *fM; // Model object.

   TGNumberEntry             *fFrameTransparency;

   TGRadioButton             *fPlotE;
   TGRadioButton             *fPlotEt;

   TGCheckButton             *fScaleAbs;
   TEveGValuator             *fMaxValAbs;
   TEveGValuator             *fMaxTowerH;

   TEveGDoubleValuator       *fEtaRng;
   TEveGValuator             *fPhi;
   TEveGValuator             *fPhiOffset;

   TGVerticalFrame           *fDataFrame;
   TGVerticalFrame           *fSliceFrame;

public:
   TEveCaloVizEditor(const TGWindow *p = nullptr, Int_t width=170, Int_t height=30,
                     UInt_t options=kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   virtual ~TEveCaloVizEditor() {}

   virtual void SetModel(TObject* obj);

   void DoMaxTowerH();
   void DoScaleAbs();
   void DoMaxValAbs();

   void DoPlot();

   void DoEtaRange();
   void DoPhi();

   void DoSliceThreshold();
   void DoSliceColor(Pixel_t color);
   void DoSliceTransparency(Long_t transp);

   ClassDef(TEveCaloVizEditor, 0); // GUI editor for TEveCaloVizEditor.
};

/**************************************************************************/

class TEveCalo3DEditor : public TGedFrame
{
private:
   TEveCalo3DEditor(const TEveCalo3DEditor&);            // Not implemented
   TEveCalo3DEditor& operator=(const TEveCalo3DEditor&); // Not implemented

protected:
   TEveCalo3D          *fM; // Model object.
   TGNumberEntry       *fFrameTransparency;

public:
   TEveCalo3DEditor(const TGWindow *p = nullptr, Int_t width=170, Int_t height=30,
                     UInt_t options=kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   virtual ~TEveCalo3DEditor() {}

   virtual void SetModel(TObject* obj);
   void    DoFrameTransparency();

   ClassDef(TEveCalo3DEditor, 0); // GUI editor for TEveCalo3DEditor.
};

#endif
