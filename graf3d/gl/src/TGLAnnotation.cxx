// @(#)root/gl:$Id$
// Author:  Matevz and Alja Tadel  20/02/2009

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLAnnotation.h"

#include "TGLIncludes.h"
#include "TColor.h"
#include "TGLUtil.h"
#include "TGLCamera.h"
#include "TGLRnrCtx.h"
#include "TGLSelectRecord.h"
#include "TGLViewerBase.h"
#include "TObjString.h"
#include "TGFrame.h"
#include "TGTextEdit.h"
#include "TGButton.h"
#include "TGLViewer.h"

#include <KeySymbols.h>

//______________________________________________________________________________
//
//
// GL-overaly annotation.
//
//

ClassImp(TGLAnnotation);

//______________________________________________________________________________
TGLAnnotation::TGLAnnotation(TGLViewerBase *parent, const char *text, Float_t posx, Float_t posy, TGLVector3 ref) :
   TGLOverlayElement(TGLOverlayElement::kAnnotation),
   fMainFrame(0), fTextEdit(0),
   fParent(0),
   fText(text),
   fLabelFontSize(0.02),
   fBackColor(0x4872fa),
   fBackHighColor(0x488ffa),
   fTextColor(0xfbbf84),
   fTextHighColor(0xf1da44),
   fAlpha(0.6),
   fPosX(posx), fPosY(posy),
   fMouseX(0),  fMouseY(0),
   fInDrag(kFALSE),
   fActive(kFALSE)
{
   // Constructor.

   fPointer = ref;
   parent->AddOverlayElement(this);
   fParent = (TGLViewer*)parent;
}

//______________________________________________________________________________
TGLAnnotation::~TGLAnnotation()
{
   // Destructor.

   fParent->RemoveOverlayElement(this);
   delete fMainFrame;
}

//______________________________________________________________________
Bool_t TGLAnnotation::Handle(TGLRnrCtx&          rnrCtx,
                             TGLOvlSelectRecord& selRec,
                             Event_t*            event)
{
   // Handle overlay event.
   // Return TRUE if event was handled.

   if (selRec.GetN() < 2) return kFALSE;
   Int_t recID = selRec.GetItem(1);

   switch (event->fType)
   {
      case kButtonPress:
      {
         // Chech selRec ... if pressed in 'X', 'E'
         if (recID == 2)
         {
            delete this;
         }
         else if (recID == 3)
         {
            MakeEditor();
         }
         else
         {
            fMouseX = event->fX;
            fMouseY = event->fY;
            fInDrag = kTRUE;
         }
         return kTRUE;
      }
      case kButtonRelease:
      {
         fInDrag = kFALSE;
         return kTRUE;
      }
      case kMotionNotify:
      {
         if (fInDrag)
         {
            const TGLRect& vp = rnrCtx.RefCamera().RefViewport();
            fPosX += (Float_t)(event->fX - fMouseX) / vp.Width();
            fPosY -= (Float_t)(event->fY - fMouseY) / vp.Height();
            fMouseX = event->fX;
            fMouseY = event->fY;
         }
         return kTRUE;
      }
      case kGKeyPress:
      {
         switch (rnrCtx.GetEventKeySym())
         {
            case kKey_E: case kKey_e:
               MakeEditor();
               return kTRUE;
            case kKey_X: case kKey_x:
               delete this;
               return kTRUE;
            default:
               return kFALSE;
         }
      }
      default:
      {
         return kFALSE;
      }
   }
}

//______________________________________________________________________________
Bool_t TGLAnnotation::MouseEnter(TGLOvlSelectRecord& /*rec*/)
{
   // Mouse has entered overlay area.

   fActive = kTRUE;
   return kTRUE;
}

//______________________________________________________________________
void TGLAnnotation::MouseLeave()
{
   // Mouse has left overlay area.

   fActive = kFALSE;
}

/**************************************************************************/
void TGLAnnotation::Render(TGLRnrCtx& rnrCtx)
{
   // Render the annotation.

   glDisable(GL_LIGHTING);

   glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_POINT_BIT);
   Float_t r, g, b;
   // button
   //
   {
      glMatrixMode(GL_PROJECTION);
      glPushMatrix();
      glLoadIdentity();
      if (rnrCtx.Selection())
      {
         TGLRect rect(*rnrCtx.GetPickRectangle());
         rnrCtx.GetCamera()->WindowToViewport(rect);
         gluPickMatrix(rect.X(), rect.Y(), rect.Width(), rect.Height(),
                       (Int_t*) rnrCtx.GetCamera()->RefViewport().CArr());
      }
      const TGLRect& vp = rnrCtx.RefCamera().RefViewport();
      glOrtho(vp.X(), vp.Width(), vp.Y(), vp.Height(), 0, 1);
      glMatrixMode(GL_MODELVIEW);
      glPushMatrix();
      glLoadIdentity();

      TGLCapabilitySwitch lights_off(GL_LIGHTING, kFALSE);
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
      glDisable(GL_CULL_FACE);
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      glShadeModel(GL_FLAT);
      glClearColor(0.0, 0.0, 0.0, 0.0);

      Float_t posX = vp.Width()  * fPosX;
      Float_t posY = vp.Height() * fPosY;

      // Text rendering
      Float_t cfs = fLabelFontSize*vp.Width();
      Int_t fs = TGLFontManager::GetFontSize(cfs);
      if (fLabelFont.GetMode() == TGLFont::kUndef)
      {
         rnrCtx.RegisterFont(fs, "arial",  TGLFont::kPixmap, fLabelFont);
      }
      else if (fLabelFont.GetSize() != fs)
      {
         rnrCtx.ReleaseFont(fLabelFont);
         rnrCtx.RegisterFont(fs, "arial",  TGLFont::kPixmap, fLabelFont);
      }

      // move to picked location
      glTranslatef(posX, posY, -0.99);

      glEnable(GL_POLYGON_OFFSET_FILL);
      glPolygonOffset(1, 1);

      glLineWidth(1);

      // get size of bg area
      Float_t ascent, descent, line_height;
      fLabelFont.MeasureBaseLineParams(ascent, descent, line_height);

      Float_t llx, lly, llz, urx, ury, urz;

      TObjArray* lines = fText.Tokenize("\n");
      Float_t width  = 0;
      Float_t height = 0;
      TIter  lit(lines);
      TObjString* osl;
      while ((osl = (TObjString*) lit()) != 0)
      {
         fLabelFont.BBox(osl->GetString().Data(), llx, lly, llz, urx, ury, urz);
         width = TMath::Max(width, urx);
         height += line_height + descent;
      }
      width  += 2 * descent;
      height += 2 * descent;

      // polygon background
      Float_t padT =  2;
      Int_t   padF = 10;
      Float_t padM = padF + 2 * padT;

      {
         glPushName(0);

         TColor::Pixel2RGB(fActive ? fBackHighColor : fBackColor, r, g, b);
         TGLUtil::Color4f(r, g, b, fAlpha);

         // bg plain
         glLoadName(1);
         glBegin(GL_QUADS);
         glVertex2f(0, 0);
         glVertex2f(0, height);
         glVertex2f(width, height);
         glVertex2f(width, 0);
         glEnd();

         // outline
         TColor::Pixel2RGB(fActive?fTextHighColor:fTextColor, r, g, b);
         TGLUtil::Color4f(r, g, b, fAlpha);

         glBegin(GL_LINE_LOOP);
         glVertex2f(0, 0);
         glVertex2f(0, height);
         glVertex2f(width, height);
         glVertex2f(width, 0);
         glEnd();

         // edit area
         if (fActive)
         {
            Float_t y = height;
            Float_t x = 0;
            TColor::Pixel2RGB(fBackHighColor, r, g, b);
            TGLUtil::Color4f(r, g, b, fAlpha);

            // edit button
            glLoadName(2);
            glBegin(GL_QUADS);
            glVertex2f(x + padM, y);
            glVertex2f(x,        y);
            glVertex2f(x,        y + padM);
            glVertex2f(x + padM, y + padM);
            glEnd();
            // close button
            glLoadName(3);
            x = padM;
            TColor::Pixel2RGB(fBackHighColor, r, g, b);
            TGLUtil::Color4f(r, g, b, fAlpha);
            glBegin(GL_QUADS);
            glVertex2f(x + padM, y);
            glVertex2f(x,        y);
            glVertex2f(x,        y + padM);
            glVertex2f(x + padM, y + padM);
            glEnd();

            // outlines
            TColor::Pixel2RGB(fTextHighColor, r, g, b);
            TGLUtil::Color4f(r, g, b, fAlpha);
            // left
            x = 0;
            glBegin(GL_LINE_LOOP);
            glVertex2f(x + padM, y);
            glVertex2f(x,        y);
            glVertex2f(x,        y + padM);
            glVertex2f(x + padM, y + padM);
            glEnd();
            // right
            x = padM;
            glBegin(GL_LINE_LOOP);
            glVertex2f(x + padM, y);
            glVertex2f(x,        y);
            glVertex2f(x,        y + padM);
            glVertex2f(x + padM, y + padM);
            glEnd();
         }
         glPopName();
      }

      glDisable(GL_POLYGON_OFFSET_FILL);

      // labels
      fLabelFont.PreRender();
      TColor::Pixel2RGB(fActive?fTextHighColor:fTextColor, r, g, b);
      TGLUtil::Color3f(r, g, b);
      TIter  next_base(lines);
      TObjString* os;
      glPushMatrix();
      glTranslatef(descent, height, 0);
      while ((os = (TObjString*) next_base()) != 0)
      {
         glTranslatef(0, -(line_height + descent), 0);
         fLabelFont.BBox(os->GetString().Data(), llx, lly, llz, urx, ury, urz);
         glRasterPos2i(0, 0);
         glBitmap(0, 0, 0, 0, 0, 0, 0);
         fLabelFont.Render(os->GetString().Data());
      }
      glPopMatrix();
      fLabelFont.PostRender();

      // menu
      if (fMenuFont.GetMode() == TGLFont::kUndef)
      {
         rnrCtx.RegisterFont(padF, "arial",  TGLFont::kPixmap, fMenuFont);
      }

      if (fActive)
      {
         TColor::Pixel2RGB(fTextHighColor, r, g, b);
         TGLUtil::Color3f(r, g, b);
         Float_t x = padT;
         Float_t y = height + padT + 0.5*padF;
         fMenuFont.PreRender();
         fMenuFont.RenderBitmap("X", x, y, 0, TGLFont::kLeft);
         x += padM + padT;
         fMenuFont.RenderBitmap("E", x, y, 0, TGLFont::kLeft);
         fMenuFont.PostRender();
      }

      glMatrixMode(GL_PROJECTION);
      glPopMatrix();
      glMatrixMode(GL_MODELVIEW);
      glPopMatrix();
   }

   // line
   //
   glLineWidth(2);
   TColor::Pixel2RGB(fActive ?fBackHighColor : fBackColor, r, g, b);
   TGLUtil::Color4f(r, g, b, fAlpha);
   glBegin(GL_LINES);
   TGLRect& vp = rnrCtx.RefCamera().RefViewport();
   TGLVertex3 v = rnrCtx.RefCamera().ViewportToWorld(TGLVertex3(fPosX*vp.Width(), fPosY*vp.Height(), 0));

   glVertex3dv(v.Arr());
   glVertex3dv(fPointer.Arr());
   glEnd();
   glPopAttrib();
}

//______________________________________________________________________________
void TGLAnnotation::MakeEditor()
{
   // Show the annotation editor.

   if (fMainFrame == 0)
   {
      fMainFrame = new TGMainFrame(gClient->GetRoot(), 1000, 1000);
      fMainFrame->SetWindowName("Annotation Editor");

      TGVerticalFrame* vf = new TGVerticalFrame(fMainFrame);

      fTextEdit = new TGTextEdit(vf,  1000, 1000, kSunkenFrame);
      vf->AddFrame(fTextEdit,  new TGLayoutHints(kLHintsExpandX|kLHintsExpandY));

      TGHorizontalFrame* hf = new TGHorizontalFrame(vf);

      TGTextButton* btt1 = new TGTextButton(hf, "OK");
      hf->AddFrame(btt1, new TGLayoutHints(kLHintsExpandX, 2, 2, 2, 2));

      TGTextButton* btt2 = new TGTextButton(hf, "Cancel");
      hf->AddFrame(btt2, new TGLayoutHints(kLHintsExpandX, 2, 2, 2, 2));

      btt1->Connect("Clicked()", "TGLAnnotation", this, "UpdateText()");
      btt2->Connect("Clicked()", "TGLAnnotation", this, "CloseEditor()");

      vf->AddFrame(hf, new TGLayoutHints(kLHintsBottom | kLHintsRight | kLHintsExpandX, 2, 2, 5, 1));

      fMainFrame->AddFrame(vf,  new TGLayoutHints(kLHintsExpandX|kLHintsExpandY));
      fMainFrame->SetCleanup(kDeepCleanup);
      fMainFrame->MapSubwindows();
   }

   TGText *tgt = new TGText();
   tgt->LoadBuffer(fText.Data());
   fTextEdit->SetText(tgt);

   Int_t nrow = tgt->RowCount();
   Int_t h = nrow*20;
   Int_t w = fTextEdit->ReturnLongestLineWidth();
   fMainFrame->Resize(TMath::Max(100, w+30), TMath::Max(100, h+40));

   fMainFrame->Layout();
   fMainFrame->MapWindow();
}

//______________________________________________________________________________
void TGLAnnotation::CloseEditor()
{
   // Close the annotation editor.

   fMainFrame->UnmapWindow();
}

//______________________________________________________________________________
void TGLAnnotation::UpdateText()
{
   // Modify the annotation text from the text-edit widget.

   fText = fTextEdit->GetText()->AsString();
   fParent->RequestDraw();
}
