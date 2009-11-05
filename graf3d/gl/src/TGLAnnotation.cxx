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
   fInDrag(kFALSE),
   fActive(kFALSE),
   fMainFrame(0), fTextEdit(0),

   fParent(0),

   fText(text),
   fTextSize(0.02),
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
   fInDrag(kFALSE),
   fActive(kFALSE),
   fMainFrame(0), fTextEdit(0),

   fParent(0),

   fText(text),
   fTextSize(0.02),
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
         fInDrag = kTRUE;

         return kTRUE;
      }
      case kButtonRelease:
      {
         fInDrag = kFALSE;

         if (recID == 2)
         {
            delete this;
            fParent->RequestDraw(rnrCtx.ViewerLOD());
         }
         else if (recID == 3)
         {
            MakeEditor();
         }

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
            // Make sure we don't go offscreen (use fDraw variables set in draw)
            if (fPosX < 0)
               fPosX = 0;
            else if (fPosX + fDrawW > 1.0f)
               fPosX = 1.0f - fDrawW;
            if (fPosY - fDrawH + fDrawY < 0)
               fPosY = fDrawH - fDrawY;
            else if (fPosY + fDrawY > 1.0f)
               fPosY = 1.0f - fDrawY;
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

   Float_t old_depth_range[2];
   glGetFloatv(GL_DEPTH_RANGE, old_depth_range);
   glDepthRange(0, 0.001);

   glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_POLYGON_BIT );
   TGLCapabilitySwitch lights_off(GL_LIGHTING, kFALSE);
   glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
   glDisable(GL_CULL_FACE);
   glEnable(GL_BLEND);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

   const TGLRect& vp = rnrCtx.RefCamera().RefViewport();

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
   glOrtho(vp.X(), vp.Width(), vp.Y(), vp.Height(), 0, 1);
   glMatrixMode(GL_MODELVIEW);
   glPushMatrix();
   glLoadIdentity();

   glEnable(GL_POLYGON_OFFSET_FILL);
   glPolygonOffset(0.1, 1);

   TGLUtil::LineWidth(1);

   // move to pos
   Float_t posX = vp.Width()  * fPosX;
   Float_t posY = vp.Height() * fPosY;
   glTranslatef(posX, posY, -0.99);


   // get size of bg area, look at font attributes
   rnrCtx.RegisterFontNoScale(TMath::Nint(fTextSize*vp.Width()), "arial",  TGLFont::kPixmap, fFont);
   Float_t ascent, descent, line_height;
   fFont.MeasureBaseLineParams(ascent, descent, line_height);
   TObjArray* lines = fText.Tokenize("\n");
   Float_t width  = 0;
   Float_t height = 0;
   TIter  lit(lines);
   TObjString* osl;
   Float_t llx, lly, llz, urx, ury, urz;
   while ((osl = (TObjString*) lit()) != 0)
   {
      fFont.BBox(osl->GetString().Data(), llx, lly, llz, urx, ury, urz);
      width = TMath::Max(width, urx);
      height -= (line_height + descent);
   }
   width  += 2 * descent;
   height -= 2 * descent;

   // Store variables needed for border check when box is dragged.
   fDrawW = (Float_t) width / vp.Width();
   fDrawH = (Float_t) - height / vp.Height();
   fDrawY = line_height / vp.Height();

   // polygon background
   Float_t padT =  2;
   Int_t   padF = 10;
   Float_t padM = padF + 2 * padT;

   glPushName(0);

   // bg plain
   Float_t y = line_height;
   Float_t x = 0;
   glLoadName(1);
   TGLUtil::ColorTransparency(bgCol, fTransparency);
   glBegin(GL_QUADS);
   glVertex2f(x, y);
   glVertex2f(x, y + height);
   glVertex2f(x+width, y + height);
   glVertex2f(x+width, y);
   glEnd();

   // outline
   TGLUtil::ColorTransparency(fgCol, fTransparency);
   glBegin(GL_LINE_LOOP);
   glVertex2f(x, y);
   glVertex2f(x, y + height);
   glVertex2f(x+width, y + height);
   glVertex2f(x+width, y);
   glEnd();

   if (fActive && fTransparency < 100)
   {  // edit area

      TGLUtil::ColorTransparency(bgCol, fTransparency);
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
      glBegin(GL_QUADS);
      glVertex2f(x + padM, y);
      glVertex2f(x,        y);
      glVertex2f(x,        y + padM);
      glVertex2f(x + padM, y + padM);
      glEnd();

      // outlines
      TGLUtil::ColorTransparency(fgCol, fTransparency);
      x = 0; // left
      glBegin(GL_LINE_LOOP);
      glVertex2f(x + padM, y);
      glVertex2f(x,        y);
      glVertex2f(x,        y + padM);
      glVertex2f(x + padM, y + padM);
      glEnd(); // right
      x = padM;
      glBegin(GL_LINE_LOOP);
      glVertex2f(x + padM, y);
      glVertex2f(x,        y);
      glVertex2f(x,        y + padM);
      glVertex2f(x + padM, y + padM);
      glEnd();
   }
   glPopName();

   // text
   Float_t zOff = 0.2; // more than 0, else not rendered
   fFont.PreRender();
   TGLUtil::Color(fgCol);
   TIter  next_base(lines);
   TObjString* os;
   glPushMatrix();
   glTranslatef(descent, line_height, zOff);
   Float_t tx = 0;
   while ((os = (TObjString*) next_base()) != 0)
   {
      glTranslatef(0, -(line_height + descent), 0);
      if (fTextAlign == TGLFont::kLeft) {
         tx = 0;
      }
      else if  (fTextAlign == TGLFont::kCenterH) {
         tx = 0.5 * width - descent ;
      }
      else {
         tx = width - 2*descent;
      }
      fFont.Render(os->GetString(), tx, 0, 0, fTextAlign, TGLFont::kTop);
   }
   glPopMatrix();
   fFont.PostRender();

   // menu

   if (fActive && fTransparency < 100)
   {
      x = padT;
      y = padT + 0.5*padF + line_height;
      rnrCtx.RegisterFontNoScale(padF, "arial",  TGLFont::kPixmap, fMenuFont);
      fMenuFont.PreRender();
      fMenuFont.Render("X", x, y, zOff, TGLFont::kLeft, TGLFont::kCenterV);
      x += padM + padT;
      fMenuFont.Render("E", x, y, zOff, TGLFont::kLeft, TGLFont::kCenterV);
      fMenuFont.PostRender();
   }

   glMatrixMode(GL_PROJECTION);
   glPopMatrix();
   glMatrixMode(GL_MODELVIEW);
   glPopMatrix();

   glDepthRange(old_depth_range[0], old_depth_range[1]);
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
   fMainFrame->UnmapWindow();
   fParent->RequestDraw();
}
