#include "TPad.h"
#include "TPoint.h"
#include "TPadPainter.h"
#include "TVirtualX.h"
#include "TImage.h"

// Local scratch buffer for screen points, faster than allocating buffer on heap
const Int_t kPXY = 1002;
static TPoint gPXY[kPXY];


ClassImp(TPadPainter)


//______________________________________________________________________________
TPadPainter::TPadPainter()
{
   //Empty ctor. Here only because of explicit copy ctor.
}

/*
Line/fill/etc. attributes can be set inside TPad, but not only where:
many of them are set by base sub-objects of 2d primitives
(2d primitives usually inherit TAttLine or TAttFill etc.).  And these sub-objects
call gVirtualX->SetLineWidth ... etc. So, if I save some attributes in my painter,
it will be mess - at any moment I do not know, where to take line attribute - from
gVirtualX or from my own member. So! All attributed, _ALL_ go to/from gVirtualX.
*/


//______________________________________________________________________________
Color_t TPadPainter::GetLineColor() const
{
   // Delegate to gVirtualX.

   return gVirtualX->GetLineColor();
}


//______________________________________________________________________________
Style_t TPadPainter::GetLineStyle() const
{
   // Delegate to gVirtualX.

   return gVirtualX->GetLineStyle();
}


//______________________________________________________________________________
Width_t TPadPainter::GetLineWidth() const
{
   // Delegate to gVirtualX.

   return gVirtualX->GetLineWidth();
}


//______________________________________________________________________________
void TPadPainter::SetLineColor(Color_t lcolor)
{
   // Delegate to gVirtualX.

   gVirtualX->SetLineColor(lcolor);
}


//______________________________________________________________________________
void TPadPainter::SetLineStyle(Style_t lstyle)
{
   // Delegate to gVirtualX.

   gVirtualX->SetLineStyle(lstyle);
}


//______________________________________________________________________________
void TPadPainter::SetLineWidth(Width_t lwidth)
{
   // Delegate to gVirtualX.

   gVirtualX->SetLineWidth(lwidth);
}


//______________________________________________________________________________
Color_t TPadPainter::GetFillColor() const
{
   // Delegate to gVirtualX.

   return gVirtualX->GetFillColor();
}


//______________________________________________________________________________
Style_t TPadPainter::GetFillStyle() const
{
   // Delegate to gVirtualX.

   return gVirtualX->GetFillStyle();
}


//______________________________________________________________________________
Bool_t TPadPainter::IsTransparent() const
{
   // Delegate to gVirtualX.

   //IsTransparent is implemented as inline function in TAttFill.
   return gVirtualX->IsTransparent();
}


//______________________________________________________________________________
void TPadPainter::SetFillColor(Color_t fcolor)
{
   // Delegate to gVirtualX.

   gVirtualX->SetFillColor(fcolor);
}


//______________________________________________________________________________
void TPadPainter::SetFillStyle(Style_t fstyle)
{
   // Delegate to gVirtualX.

   gVirtualX->SetFillStyle(fstyle);
}


//______________________________________________________________________________
void TPadPainter::SetOpacity(Int_t percent)
{
   // Delegate to gVirtualX.

   gVirtualX->SetOpacity(percent);
}


//______________________________________________________________________________
Short_t TPadPainter::GetTextAlign() const
{
   // Delegate to gVirtualX.

   return gVirtualX->GetTextAlign();
}


//______________________________________________________________________________
Float_t TPadPainter::GetTextAngle() const
{
   // Delegate to gVirtualX.

   return gVirtualX->GetTextAngle();
}


//______________________________________________________________________________
Color_t TPadPainter::GetTextColor() const
{
   // Delegate to gVirtualX.

   return gVirtualX->GetTextColor();
}


//______________________________________________________________________________
Font_t TPadPainter::GetTextFont() const
{
   // Delegate to gVirtualX.

   return gVirtualX->GetTextFont();
}


//______________________________________________________________________________
Float_t TPadPainter::GetTextSize() const
{
   // Delegate to gVirtualX.

   return gVirtualX->GetTextSize();
}


//______________________________________________________________________________
Float_t TPadPainter::GetTextMagnitude() const
{
   // Delegate to gVirtualX.

   return gVirtualX->GetTextMagnitude();
}


//______________________________________________________________________________
void TPadPainter::SetTextAlign(Short_t align)
{
   // Delegate to gVirtualX.

   gVirtualX->SetTextAlign(align);
}


//______________________________________________________________________________
void TPadPainter::SetTextAngle(Float_t tangle)
{
   // Delegate to gVirtualX.

   gVirtualX->SetTextAngle(tangle);
}


//______________________________________________________________________________
void TPadPainter::SetTextColor(Color_t tcolor)
{
   // Delegate to gVirtualX.

   gVirtualX->SetTextColor(tcolor);
}


//______________________________________________________________________________
void TPadPainter::SetTextFont(Font_t tfont)
{
   // Delegate to gVirtualX.

   gVirtualX->SetTextFont(tfont);
}


//______________________________________________________________________________
void TPadPainter::SetTextSize(Float_t tsize)
{
   // Delegate to gVirtualX.
   
   gVirtualX->SetTextSize(tsize);
}


//______________________________________________________________________________
void TPadPainter::SetTextSizePixels(Int_t npixels)
{
   // Delegate to gVirtualX.

   gVirtualX->SetTextSizePixels(npixels);
}


//______________________________________________________________________________
Int_t TPadPainter::CreateDrawable(UInt_t w, UInt_t h)
{
   // Create a gVirtualX Pixmap.

   return gVirtualX->OpenPixmap(Int_t(w), Int_t(h));
}


//______________________________________________________________________________
void TPadPainter::ClearDrawable()
{
   // Clear the current gVirtualX window.

   gVirtualX->ClearWindow();
}


//______________________________________________________________________________
void TPadPainter::CopyDrawable(Int_t id, Int_t px, Int_t py)
{
   // Copy a gVirtualX pixmap.

   gVirtualX->CopyPixmap(id, px, py);
}


//______________________________________________________________________________
void TPadPainter::DestroyDrawable()
{
   // Close the current gVirtualX pixmap.

   gVirtualX->ClosePixmap();
}


//______________________________________________________________________________
void TPadPainter::SelectDrawable(Int_t device)
{
   // Select the window in which the graphics will go.

   gVirtualX->SelectWindow(device);
}


//______________________________________________________________________________
void TPadPainter::DrawLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2)
{
   // Paint a simple line.

   Int_t px1 = gPad->XtoPixel(x1);
   Int_t px2 = gPad->XtoPixel(x2);
   Int_t py1 = gPad->YtoPixel(y1);
   Int_t py2 = gPad->YtoPixel(y2);

   gVirtualX->DrawLine(px1, py1, px2, py2);
}


//______________________________________________________________________________
void TPadPainter::DrawLineNDC(Double_t u1, Double_t v1, Double_t u2, Double_t v2)
{
   // Paint a simple line in normalized coordinates.

   Int_t px1 = gPad->UtoPixel(u1);
   Int_t py1 = gPad->VtoPixel(v1);
   Int_t px2 = gPad->UtoPixel(u2);
   Int_t py2 = gPad->VtoPixel(v2);
   gVirtualX->DrawLine(px1, py1, px2, py2);
}


//______________________________________________________________________________
void TPadPainter::DrawBox(Double_t x1, Double_t y1, Double_t x2, Double_t y2, EBoxMode mode)
{
   // Paint a simple box.

   Int_t px1 = gPad->XtoPixel(x1);
   Int_t px2 = gPad->XtoPixel(x2);
   Int_t py1 = gPad->YtoPixel(y1);
   Int_t py2 = gPad->YtoPixel(y2);

   // Box width must be at least one pixel
   if (TMath::Abs(px2-px1) < 1) px2 = px1+1;
   if (TMath::Abs(py1-py2) < 1) py1 = py2+1;

   gVirtualX->DrawBox(px1,py1,px2,py2,(TVirtualX::EBoxMode)mode);
}


//______________________________________________________________________________
void TPadPainter::DrawFillArea(Int_t n, const Double_t *x, const Double_t *y)
{
   // Paint filled area.

   TPoint *pxy = &gPXY[0];
   if (n >= kPXY) pxy = new TPoint[n+1]; if (!pxy) return;
   Int_t fillstyle = gVirtualX->GetFillStyle();
   for (Int_t i=0; i<n; i++) {
      pxy[i].fX = gPad->XtoPixel(x[i]);
      pxy[i].fY = gPad->YtoPixel(y[i]);
   }
   if (fillstyle == 0) {
      pxy[n].fX = pxy[0].fX;
      pxy[n].fY = pxy[0].fY;
      gVirtualX->DrawFillArea(n+1,pxy);
   } else {
      gVirtualX->DrawFillArea(n,pxy);
   }
   if (n >= kPXY) delete [] pxy;
}


//______________________________________________________________________________
void TPadPainter::DrawFillArea(Int_t n, const Float_t *x, const Float_t *y)
{
   // Paint filled area.

   TPoint *pxy = &gPXY[0];
   if (n >= kPXY) pxy = new TPoint[n+1]; if (!pxy) return;
   Int_t fillstyle = gVirtualX->GetFillStyle();
   for (Int_t i=0; i<n; i++) {
      pxy[i].fX = gPad->XtoPixel(x[i]);
      pxy[i].fY = gPad->YtoPixel(y[i]);
   }
   if (fillstyle == 0) {
      pxy[n].fX = pxy[0].fX;
      pxy[n].fY = pxy[0].fY;
      gVirtualX->DrawFillArea(n+1,pxy);
   } else {
      gVirtualX->DrawFillArea(n,pxy);
   }
   if (n >= kPXY) delete [] pxy;
}


//______________________________________________________________________________
void TPadPainter::DrawPolyLine(Int_t n, const Double_t *x, const Double_t *y)
{
   // Paint polyline.

   TPoint *pxy = &gPXY[0];
   if (n >= kPXY) pxy = new TPoint[n+1]; if (!pxy) return;
   for (Int_t i=0; i<n; i++) {
      pxy[i].fX = gPad->XtoPixel(x[i]);
      pxy[i].fY = gPad->YtoPixel(y[i]);
   }
   gVirtualX->DrawPolyLine(n,pxy);
   if (n >= kPXY) delete [] pxy;
}


//______________________________________________________________________________
void TPadPainter::DrawPolyLine(Int_t n, const Float_t *x, const Float_t *y)
{
   // Paint polyline.

   TPoint *pxy = &gPXY[0];
   if (n >= kPXY) pxy = new TPoint[n+1]; if (!pxy) return;
   for (Int_t i=0; i<n; i++) {
      pxy[i].fX = gPad->XtoPixel(x[i]);
      pxy[i].fY = gPad->YtoPixel(y[i]);
   }
   gVirtualX->DrawPolyLine(n,pxy);
   if (n >= kPXY) delete [] pxy;
}


//______________________________________________________________________________
void TPadPainter::DrawPolyLineNDC(Int_t n, const Double_t *u, const Double_t *v)
{
   // Paint polyline in normalized coordinates.

   TPoint *pxy = &gPXY[0];
   if (n >= kPXY) pxy = new TPoint[n+1]; if (!pxy) return;
   for (Int_t i=0; i<n; i++) {
      pxy[i].fX = gPad->UtoPixel(u[i]);
      pxy[i].fY = gPad->VtoPixel(v[i]);
   }
   gVirtualX->DrawPolyLine(n,pxy);
   if (n >= kPXY) delete [] pxy;
}


//______________________________________________________________________________
void TPadPainter::DrawPolyMarker(Int_t n, const Double_t *x, const Double_t *y)
{
   // Paint polymarker.

   TPoint *pxy = &gPXY[0];
   if (n >= kPXY) pxy = new TPoint[n+1]; if (!pxy) return;
   for (Int_t i=0; i<n; i++) {
      pxy[i].fX = gPad->XtoPixel(x[i]);
      pxy[i].fY = gPad->YtoPixel(y[i]);
   }
   gVirtualX->DrawPolyMarker(n,pxy);
   if (n >= kPXY)   delete [] pxy;
}


//______________________________________________________________________________
void TPadPainter::DrawPolyMarker(Int_t n, const Float_t *x, const Float_t *y)
{
   // Paint polymarker.

   TPoint *pxy = &gPXY[0];
   if (n >= kPXY) pxy = new TPoint[n+1]; if (!pxy) return;
   for (Int_t i=0; i<n; i++) {
      pxy[i].fX = gPad->XtoPixel(x[i]);
      pxy[i].fY = gPad->YtoPixel(y[i]);
   }
   gVirtualX->DrawPolyMarker(n,pxy);
   if (n >= kPXY)   delete [] pxy;
}


//______________________________________________________________________________
void TPadPainter::DrawText(Double_t x, Double_t y, const char *text, ETextMode mode)
{
   // Paint text.

   Int_t px = gPad->XtoPixel(x);
   Int_t py = gPad->YtoPixel(y);
   Double_t angle = GetTextAngle();
   Double_t mgn = GetTextMagnitude();
   gVirtualX->DrawText(px, py, angle, mgn, text, (TVirtualX::ETextMode)mode);
}


//______________________________________________________________________________
void TPadPainter::DrawTextNDC(Double_t u, Double_t v, const char *text, ETextMode mode)
{
   // Paint text in normalized coordinates.

   Int_t px = gPad->UtoPixel(u);
   Int_t py = gPad->VtoPixel(v);
   Double_t angle = GetTextAngle();
   Double_t mgn = GetTextMagnitude();
   gVirtualX->DrawText(px, py, angle, mgn, text, (TVirtualX::ETextMode)mode);
}


//______________________________________________________________________________
void TPadPainter::SaveImage(TVirtualPad *pad, const char *fileName, Int_t type) const
{
   // Save the image displayed in the canvas pointed by "pad" into a 
   // binary file.

   if (type == TImage::kGif) {
      gVirtualX->WriteGIF((char*)fileName);
   } else {
      TImage *img = TImage::Create();
      img->FromPad(pad);
      img->WriteImage(fileName, (TImage::EImageFileTypes)type);
      delete img;
   }
}
