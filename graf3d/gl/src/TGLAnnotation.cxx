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
#include "TROOT.h"
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

#include "TMath.h"

#include <KeySymbols.h>

//______________________________________________________________________________
//
//
// GL-overaly annotation.
//
//

ClassImp(TGLAnnotation);

Color_t  TGLAnnotation::fgBackColor = kAzure + 10;
Color_t  TGLAnnotation::fgTextColor = kOrange;

//______________________________________________________________________________
TGLAnnotation::TGLAnnotation(TGLViewerBase *parent, const char *text, Float_t posx, Float_t posy) :
   TGLOverlayElement(TGLOverlayElement::kAnnotation),

   fPosX(posx), fPosY(posy),
   fMouseX(0),  fMouseY(0),
   fDrag(kNone),
   fActive(kFALSE),
   fMainFrame(0), fTextEdit(0),

   fParent(0),

   fText(text),
   fTextSize(0.03),
   fTextAlign(TGLFont::kLeft),
   fBackColor(fgBackColor),
   fTextColor(fgTextColor),
   fTransparency(100),
   fDrawRefLine(kFALSE),
   fUseColorSet(kTRUE)
{
   // Constructor.
   // Create annotation as plain text

   parent->AddOverlayElement(this);
   fParent = (TGLViewer*)parent;
}

//______________________________________________________________________________
TGLAnnotation::TGLAnnotation(TGLViewerBase *parent, const char *text, Float_t posx, Float_t posy, TGLVector3 ref) :
   TGLOverlayElement(TGLOverlayElement::kAnnotation),
   fPosX(posx), fPosY(posy),
   fMouseX(0),  fMouseY(0),
   fDrag(kNone),
   fActive(kFALSE),
   fMainFrame(0), fTextEdit(0),

   fParent(0),

   fText(text),
   fTextSize(0.03),
   fTextAlign(TGLFont::kLeft),
   fBackColor(fgBackColor),
   fTextColor(fgTextColor),
   fTransparency(40),
   fDrawRefLine(kTRUE),
   fUseColorSet(kFALSE)
{
   // Constructor.
   // Create annotaton by picking an object.

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
         fMouseX = event->fX;
         fMouseY = event->fY;
         fDrag = (recID == kResizeID) ? kResize : kMove;
         return kTRUE;
      }
      case kButtonRelease:
      {
         fDrag = kNone;
         if (recID == kDeleteID)
         {
            TGLViewer *v = fParent;
            delete this;
            v->RequestDraw(rnrCtx.ViewerLOD());
         }
         else if (recID == kEditID)
            MakeEditor();
      }
      case kMotionNotify:
      {
         const TGLRect& vp = rnrCtx.RefCamera().RefViewport();
         if (vp.Width() == 0 || vp.Height() == 0) return false;

         if (fDrag != kNone)
         {
            if (fDrag == kMove)
            {
               fPosX += (Float_t)(event->fX - fMouseX) / vp.Width();
               fPosY -= (Float_t)(event->fY - fMouseY) / vp.Height();
               fMouseX = event->fX;
               fMouseY = event->fY;
               // Make sure we don't go offscreen (use fDraw variables set in draw)
               if (fPosX < 0)
                  fPosX = 0;
               else if (fPosX +fDrawW > 1.0f)
                  fPosX = 1.0f - fDrawW;
               if (fPosY < fDrawH)
                  fPosY = fDrawH;
               else if (fPosY > 1.0f)
                  fPosY = 1.0f;

            }
            else
            {
               fMouseX = event->fX;
               fMouseY = event->fY;
               Float_t dX = TMath::Min((Float_t)(fMouseX) / vp.Width(), 1.f) - (fPosX+fDrawW);
               // in transalte from X11 to local GL coordinate system
               Float_t my = 1  - TMath::Min((Float_t)(fMouseY) / vp.Height(), 1.f); 
               Float_t cy = (fPosY-fDrawH);
               // printf("mouseY %f,  coord %f \n", my, cy);
               Float_t dY = cy -my;

               Float_t rx = dX/fDrawW;
               Float_t ry = dY/fDrawH;

               Float_t sd = (TMath::Abs(rx) > TMath::Abs(ry)) ? ry : rx;
               fTextSize *= (1 + sd);
               fTextSize = TMath::Max(fTextSize, 0.01f); // down limit text size
            }

         }
         return kTRUE;
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

   const TGLRect& vp = rnrCtx.RefCamera().RefViewport();
   if (vp.Width() == 0 && vp.Height() == 0)
      return;

   Float_t old_depth_range[2];
   glGetFloatv(GL_DEPTH_RANGE, old_depth_range);
   glDepthRange(0, 0.001);


   glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_POLYGON_BIT );
   TGLCapabilitySwitch lights_off(GL_LIGHTING, kFALSE);
   glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
   glDisable(GL_CULL_FACE);
   glEnable(GL_BLEND);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

   // prepare colors
   Color_t bgCol, fgCol;
   if (fUseColorSet)
   {
      fgCol = rnrCtx.ColorSet().Markup().GetColorIndex();

      TColor* c1 = gROOT->GetColor(rnrCtx.ColorSet().Markup().GetColorIndex());
      TColor* c2 = gROOT->GetColor(rnrCtx.ColorSet().Background().GetColorIndex());
      Float_t f1 = 0.5, f2 = 0.5;
      bgCol = TColor::GetColor(c1->GetRed()  *f1  + c2->GetRed()  *f2,
                               c1->GetGreen()*f1  + c2->GetGreen()*f2,
                               c1->GetBlue() *f1  + c2->GetBlue() *f2);
   }
   else {
      fgCol = fTextColor;
      bgCol = fBackColor;
   }

   if (fDrawRefLine)
   {
      TGLUtil::ColorTransparency(bgCol, fTransparency);
      TGLUtil::LineWidth(2);
      glBegin(GL_LINES);
      TGLVertex3 v = rnrCtx.RefCamera().ViewportToWorld(TGLVertex3(fPosX*vp.Width(), fPosY*vp.Height(), 0));
      glVertex3dv(v.Arr());
      glVertex3dv(fPointer.Arr());
      glEnd();
   }

   // reset matrix
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
   glMatrixMode(GL_MODELVIEW);
   glPushMatrix();
   glLoadIdentity();

   // set ortho camera to [0,1] [0.1]
   glLoadIdentity();
   glTranslatef(-1, -1, 0);
   glScalef(2, 2, 1);

   glEnable(GL_POLYGON_OFFSET_FILL);
   glPolygonOffset(0.1, 1);

   TGLUtil::LineWidth(1);

   // move to pos
   glTranslatef(fPosX, fPosY, 0);

   // get unscaled text size
   Int_t fs = TGLFontManager::GetFontSize(TMath::Nint(vp.Height()*fTextSize), 12, 64);
   rnrCtx.RegisterFontNoScale(fs, "arial",  TGLFont::kTexture, fFont);
   Float_t ascent, descent, line_height;
   fFont.MeasureBaseLineParams(ascent, descent, line_height);
   TObjArray* lines = fText.Tokenize("\n");
   Float_t widthTxt  = 0;
   Float_t heightTxt = 0;
   TIter  lit(lines);
   TObjString* osl;
   Float_t llx, lly, llz, urx, ury, urz;
   while ((osl = (TObjString*) lit()) != 0)
   {
      fFont.BBox(osl->GetString().Data(), llx, lly, llz, urx, ury, urz);
      widthTxt = TMath::Max(widthTxt, urx);
      heightTxt += (line_height + descent);
   }
   widthTxt  += 2 * descent;
   heightTxt += 2 * descent;

   // keep proportions
   Float_t sy = fTextSize/(line_height+descent);
   Float_t sx = sy/vp.Aspect();
   fDrawW = sx*widthTxt;
   fDrawH = sy*heightTxt;

   glScalef(sx, sy, 1.);


   glPushName(kMoveID);

   Float_t x1, x2, y1, y2;
   Float_t z3 = 0;     // main background
   Float_t z2 = -0.01; // outlines and text
   Float_t z1 = -0.02; // button on top of text
   Float_t z0 = -0.03; // button on top of text

   // main background
   glLoadName(kMoveID);
   x1 = 0;
   x2 = fDrawW/sx;
   y1 = -fDrawH/sy;
   y2 = 0;
   TGLUtil::ColorTransparency(bgCol, fTransparency);
   glBegin(GL_QUADS);
   glVertex3f(x1, y1, z3);
   glVertex3f(x2, y1, z3);
   glVertex3f(x2, y2, z3);
   glVertex3f(x1, y2, z3);
   glEnd();
   // main polygon outline
   TGLUtil::ColorTransparency(fgCol, GetLineTransparency());
   glBegin(GL_LINE_LOOP);
   glVertex3f(x1, y1, z2);
   glVertex3f(x2, y1, z2);
   glVertex3f(x2, y2, z2);
   glVertex3f(x1, y2, z2);
   glEnd();

   // annotation text
   TGLUtil::Color(fgCol);
   TIter  next_base(lines);
   TObjString* os;
   fFont.PreRender();
   glPushMatrix();
   Float_t tx = 0;
   while ((os = (TObjString*) next_base()) != 0)
   {
      if (fTextAlign == TGLFont::kLeft) {
         tx = 0;
      }
      else if  (fTextAlign == TGLFont::kCenterH) {
         tx = 0.5 * widthTxt - descent ;
      }
      else {
         tx = widthTxt - 2*descent;
      }
      glTranslatef(0, -(line_height + descent), 0);
      fFont.Render(os->GetString(), tx+descent, 0, z2, fTextAlign, TGLFont::kTop) ;
   }
   glPopMatrix();
   fFont.PostRender();

   // buttons
   if (fActive)
   {
      Float_t bbox[6];
      fFont.PreRender();
      glPushMatrix();
      fFont.BBox("X", bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);
      glLoadName(kDeleteID);
      fFont.Render("X", descent, descent, z2, fTextAlign, TGLFont::kTop);
      x2 = bbox[3]+ descent;
      glLoadName(kEditID);
      fFont.Render("E", descent+x2, descent, z2, fTextAlign, TGLFont::kTop);
      fFont.PostRender();
      glPopMatrix();

      x2 = line_height;
      x1 = 0;
      y1 = 0;
      y2 = line_height+ descent;
      {
         // close button
         glLoadName(kDeleteID);
         // polygon
         TGLUtil::ColorTransparency(bgCol, fTransparency);
         glBegin(GL_QUADS);
         glVertex3f(x1, y1, z3);
         glVertex3f(x2, y1, z3);
         glVertex3f(x2, y2, z3);
         glVertex3f(x1, y2, z3);
         glEnd();
         //  outline
         TGLUtil::ColorTransparency(fgCol, GetLineTransparency());
         glBegin(GL_LINE_LOOP);
         glVertex3f(x1, y1, z0);
         glVertex3f(x2, y1, z0);
         glVertex3f(x2, y2, z0);
         glVertex3f(x1, y2, z0);
         glEnd();
      }
      {
         // edit button
         x1 += x2;
         x2 += x2;
         glLoadName(kEditID);
         // polygon
         TGLUtil::ColorTransparency(bgCol, fTransparency);
         glBegin(GL_QUADS);
         glVertex3f(x1, y1, z3);
         glVertex3f(x2, y1, z3);
         glVertex3f(x2, y2, z3);
         glVertex3f(x1, y2, z3);
         glEnd();
         //  outline
         TGLUtil::ColorTransparency(fgCol, GetLineTransparency());
         glBegin(GL_LINE_LOOP);
         glVertex3f(x1, y1, z0);
         glVertex3f(x2, y1, z0);
         glVertex3f(x2, y2, z0);
         glVertex3f(x1, y2, z0);
         glEnd();
      }
      {
         // resize button
         glLoadName(kResizeID);
         // polygon
         x1 = fDrawW/sx - line_height;
         x2 = fDrawW/sx;
         y1 = -fDrawH/sy;
         y2 = -fDrawH/sy + line_height;
         TGLUtil::ColorTransparency(bgCol, fTransparency);
         glBegin(GL_QUADS);
         glVertex3f(x1, y1, z1);
         glVertex3f(x2, y1, z1);
         glVertex3f(x2, y2, z1);
         glVertex3f(x1, y2, z1);
         glEnd();
         // draw resize corner lines
         TGLUtil::Color(kRed);
         glBegin(GL_LINES);
         TGLUtil::ColorTransparency(fgCol, GetLineTransparency());
         Float_t aOff = 0.25*line_height;
         glVertex3f(x1+aOff, y1+aOff, z0);
         glVertex3f(x2-aOff, y1+aOff, z0);
         glVertex3f(x2-aOff, y1+aOff, z0);
         glVertex3f(x2-aOff, y2-aOff, z0);
         glEnd();
      }
   }

   glPopName();

   glMatrixMode(GL_PROJECTION);
   glPopMatrix();
   glMatrixMode(GL_MODELVIEW);
   glPopMatrix();

   glDepthRange(old_depth_range[0], old_depth_range[1]);
   glPopAttrib();
}

//______________________________________________________________________________
Char_t TGLAnnotation::GetLineTransparency() const
{
   // Returns transparecy of annotation outline.
   // If annotation is selected enforce visiblity of outline.

   if (fActive)
      return TMath::Min(70, fTransparency);
   else
      return fTransparency;
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
   fMainFrame->UnmapWindow();
   fParent->RequestDraw();
}
