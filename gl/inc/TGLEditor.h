// @(#)root/gl:$Name:  $:$Id: TArcBall.h,v 1.4 2004/09/03 12:52:42 brun Exp $
// Author:  Timur Pocheptsov  03/08/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLEditor
#define ROOT_TGLEditor

#ifndef ROOT_RQ_OBJECT
#include "RQ_OBJECT.h"
#endif
#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif


class TGLayoutHints;
class TGLMatView;
class TGHSlider;
class TGButton;
class TGCanvas;
class TGLabel;

class TGLEditor : public TGCompositeFrame {
   friend class TGLMatView;
private:
   TGHSlider     *fRedSlider;
   TGHSlider     *fGreenSlider;
   TGHSlider     *fBlueSlider;
   TGHSlider     *fAlphaSlider;
   TGButton      *fApplyButton;
   TGLayoutHints *fLayout;
   TGLayoutHints *fLabelLayout;
   TGLayoutHints *fViewLayout;

   TGCanvas      *fViewCanvas;
   TGLMatView    *fMatView;
   Bool_t        fIsActive;

   Color_t       fRGBA[4];
   TGLabel       *fInfo[4];
   Window_t      fGLWin;
   ULong_t       fCtx;
public:
   TGLEditor(const TGWindow *parent, Int_t r = 100, Int_t g = 100, Int_t b = 100, Int_t a = 100);
   ~TGLEditor();
   void SetRGBA(Color_t r, Color_t g, Color_t b, Color_t a);
   void GetRGBA(Color_t &r, Color_t &g, Color_t &b, Color_t &a)const;
   void DoSlider(Int_t val);

   TGButton *GetButton()const
   {
      return fApplyButton;
   }
   void Stop()
   {
      fIsActive = kFALSE;
   }

private:
   Bool_t HandleContainerNotify(Event_t *event);
   Bool_t HandleContainerExpose(Event_t *event);
   void DrawSphere()const;
   void SwapBuffers()const;
   void MakeCurrent()const;
   //Non-copyable class
   TGLEditor(const TGLEditor &);
   TGLEditor & operator = (const TGLEditor &);

   ClassDef(TGLEditor, 0)
};

#endif
