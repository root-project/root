// @(#)root/gl:$Id:$
// Author: Bertrand Bellenot 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGL5DDataSetEditor
#define ROOT_TGL5DDataSetEditor

#ifndef ROOT_TGedFrame
#include "TGedFrame.h"
#endif
#ifndef ROOT_TGLUtil
#include "TGLUtil.h"
#endif

class TGCheckButton;
class TGNumberEntry;
class TGDoubleSlider;

class TGL5DPainter;

class TGL5DDataSetEditor : public TGedFrame
{
private:
   TGCheckButton    *fShowBoxCut;
   TGNumberEntry    *fNumberOfPlanes;
   TGCheckButton    *fLogScale;
   TGDoubleSlider   *fSlideRange;

   //Model
   TGL5DPainter     *fPainter;

   void ConnectSignals2Slots();

   TGL5DDataSetEditor(const TGL5DDataSetEditor &);
   TGL5DDataSetEditor &operator = (const TGL5DDataSetEditor &);

   void CreateStyleTab();

public:
   TGL5DDataSetEditor(const TGWindow *p=0, Int_t width=140, Int_t height=30,
               UInt_t options=kChildFrame, Pixel_t back=GetDefaultFrameBackground());
   ~TGL5DDataSetEditor();

   virtual void   SetModel(TObject* obj);

   void           DoLogScale();
   void           DoNumberOfPlanes();
   void           DoShowBoxCut();
   void           DoSliderRangeMoved();

   ClassDef(TGL5DDataSetEditor, 0); //GUI for editing OpenGL 5D Viewer attributes
};

#endif
