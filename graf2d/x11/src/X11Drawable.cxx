// @(#)root/x11:$Id$
// Author: Sergey Linev   23/11/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include "X11Drawable.h"

#include "TGX11.h"
#include "TStyle.h"
#include "RStipples.h"
#include "TObjArray.h"
#include "TError.h"
#include "TPoint.h"
#include "TMath.h"

#include "gifencode.h"
#include "rotated.h"

#include <algorithm>


struct RXGCValues:XGCValues{};
struct RXColor:XColor{};
struct RXImage:XImage{};
struct RXPoint:XPoint{};
struct RXVisualInfo:XVisualInfo{};
struct RVisual:Visual{};


const Int_t kBIGGEST_RGB_VALUE = 65535;


static int  gMarkerLineStyle = LineSolid;
static int  gMarkerCapStyle  = CapRound;
static int  gMarkerJoinStyle = JoinRound;

//
// Keep style values for line GC
//
static int  gCapStyle  = CapButt;
static int  gJoinStyle = JoinMiter;


X11Drawable::X11Drawable(Int_t id, TGX11  &x11) : XDrawable(id), fX11(x11)
{

}

X11Drawable::~X11Drawable()
{

}

////////////////////////////////////////////////////////////////////////////////
/// Check if drawable is empty

Bool_t X11Drawable::IsNull() const
{
   return !fOpen;
}

////////////////////////////////////////////////////////////////////////////////
/// Set fill attributes

void X11Drawable::SetAttFill(const TAttFill &att)
{
   Int_t cindex = att.GetFillColor();
   if (!gStyle->GetFillColor() && cindex > 1)
      cindex = 0;
   if (cindex >= 0)
      fX11.SetColor(this, &fGClist[kGCfill], Int_t(cindex));
   fAttFill.SetFillColor(cindex);

   Int_t style = att.GetFillStyle() / 1000;
   Int_t fasi  = att.GetFillStyle() % 1000;
   Int_t stn = (fasi >= 1 && fasi <=25) ? fasi : 2;
   fAttFill.SetFillStyle(style * 1000 + fasi);

   switch (style) {
      case 1:         // solid
         fillHollow = 0;
         XSetFillStyle((Display*)fX11.fDisplay, fGClist[kGCfill], FillSolid);
         break;

      case 2:         // pattern
         fillHollow = 1;
         break;

      case 3:         // hatch
         fillHollow = 0;
         XSetFillStyle((Display*)fX11.fDisplay, fGClist[kGCfill], FillStippled);

         if (stn != fillFasi) {
            if (fillPattern != 0)
               XFreePixmap((Display*)fX11.fDisplay, fillPattern);

            fillPattern = XCreateBitmapFromData((Display*)fX11.fDisplay, fX11.fRootWin,
                                                      (const char*)gStipples[stn], 16, 16);

            XSetStipple((Display*)fX11.fDisplay, fGClist[kGCfill], fillPattern);
            fillFasi = stn;
         }
         break;

      default:
         fillHollow = 1;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set line attributes

void X11Drawable::SetAttLine(const TAttLine &att)
{
   if (fAttLine.GetLineStyle() != att.GetLineStyle()) { //set style index only if different
      if (att.GetLineStyle() <= 1)
         dashList.clear();
      else if (att.GetLineStyle() == 2)
         dashList = { 3, 3 };
      else if (att.GetLineStyle() == 3)
         dashList = { 1, 2 };
      else if (att.GetLineStyle() == 4) {
         dashList = { 3, 4, 1, 4} ;
      } else {
         TString st = (TString)gStyle->GetLineStyleString(att.GetLineStyle());
         auto tokens = st.Tokenize(" ");
         Int_t nt = tokens->GetEntries();
         dashList.resize(nt);
         for (Int_t j = 0; j < nt; ++j) {
            Int_t it;
            sscanf(tokens->At(j)->GetName(), "%d", &it);
            dashList[j] = (Int_t) (it/4);
         }
         delete tokens;
      }
      dashLength = 0;
      for (auto elem : dashList)
         dashLength += elem;
      dashOffset = 0;
      lineStyle = dashList.size() == 0 ? LineSolid : LineOnOffDash;
   }

   lineWidth = att.GetLineWidth();
   if (lineStyle == LineSolid) {
      if (lineWidth == 1)
         lineWidth = 0;
   } else {
      if (lineWidth == 0)
         lineWidth = 1;
   }

  if (lineWidth >= 0) {
      XSetLineAttributes((Display*)fX11.fDisplay, fGClist[kGCline], lineWidth,
                          lineStyle, gCapStyle, gJoinStyle);
      if (lineStyle == LineOnOffDash)
         XSetLineAttributes((Display*)fX11.fDisplay, fGClist[kGCdash], lineWidth,
                             lineStyle, gCapStyle, gJoinStyle);
   }

   if (att.GetLineColor() >= 0) {
      fX11.SetColor(this, &fGClist[kGCline], (Int_t) att.GetLineColor());
      fX11.SetColor(this, &fGClist[kGCdash], (Int_t) att.GetLineColor());
   }

   fAttLine = att;
}

////////////////////////////////////////////////////////////////////////////////
/// Set marker attributes for specified window

void X11Drawable::SetAttMarker(const TAttMarker &att)
{
   fX11.SetColor(this, &fGClist[kGCmark], att.GetMarkerColor());

   Bool_t changed = (att.GetMarkerSize() != fAttMarker.GetMarkerSize()) ||
                    (att.GetMarkerStyle() != fAttMarker.GetMarkerStyle());

   fAttMarker = att;

   if (!changed)
      return;

   markerLineWidth = TAttMarker::GetMarkerLineWidth(att.GetMarkerStyle());
   markerType = att.GetMarkerShape(markerSize, markerShape, 1., TAttMarker::kDotAsLines);

   // The fast dots and simple lines do not  need join properties
   if (markerType == TAttMarker::kShapeSegments) {
       XSetLineAttributes((Display*)fX11.fDisplay, fGClist[kGCmark], markerLineWidth, LineSolid, CapButt, JoinMiter);
   } else {
       XSetLineAttributes((Display*)fX11.fDisplay, fGClist[kGCmark], markerLineWidth,
                          gMarkerLineStyle, gMarkerCapStyle, gMarkerJoinStyle);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set text attributes for specified window

void X11Drawable::SetAttText(const TAttText &att)
{
   Int_t txalh = att.GetTextAlign() / 10;
   Int_t txalv = att.GetTextAlign() % 10;

   textAlign = kAlignNone;

   switch (txalh) {
      case 0 :
      case 1 :
         switch (txalv) {  //left
            case 1 :
               textAlign = kBLeft;   //bottom
               break;
            case 2 :
               textAlign = kMLeft;   //middle
               break;
            case 3 :
               textAlign = kTLeft;   //top
               break;
         }
         break;
      case 2 :
         switch (txalv) { //center
            case 1 :
               textAlign = kBCenter;   //bottom
               break;
            case 2 :
               textAlign = kMCenter;   //middle
               break;
            case 3 :
               textAlign = kTCenter;   //top
               break;
         }
         break;
      case 3 :
         switch (txalv) {  //right
            case 1 :
               textAlign = kBRight;   //bottom
               break;
            case 2 :
               textAlign = kMRight;   //center
               break;
            case 3 :
               textAlign = kTRight;   //top
               break;
         }
         break;
   }

   fX11.SetColor(this, &fGClist[kGCtext], att.GetTextColor());

   XGCValues values;
   if (XGetGCValues((Display*)fX11.fDisplay, fGClist[kGCtext], GCForeground | GCBackground, &values)) {
      XSetForeground( (Display*)fX11.fDisplay, fGClist[kGCinvt], values.background );
      XSetBackground( (Display*)fX11.fDisplay, fGClist[kGCinvt], values.foreground );
   } else {
      ::Error("SetAttText", "cannot get GC values");
   }
   XSetBackground((Display*)fX11.fDisplay, fGClist[kGCtext], fX11.GetColor(0).fPixel);

   fX11.SetCurrentFont(this);

   fAttText = att;
}

////////////////////////////////////////////////////////////////////////////////
/// Set window draw mode

void X11Drawable::SetDrawMode(TVirtualX::EDrawMode mode)
{
   if (!fX11.fDisplay)
      return;

   auto gxmode = GXcopy;
   if (mode == TVirtualX::kXor)
      gxmode = GXxor;
   else if (mode == TVirtualX::kInvert)
      gxmode = GXinvert;
   for (int i = 0; i < kMAXGC; i++)
      XSetFunction((Display*)fX11.fDisplay, fGClist[i], gxmode);

   drawMode = mode;
}

////////////////////////////////////////////////////////////////////////////////
/// Clear window

void X11Drawable::ClearWindow()
{
   if (!fIsPixmap && !fDoubleBuffer) {
      XSetWindowBackground((Display*)fX11.fDisplay, fDrawing, fX11.GetColor(0).fPixel);
      XClearWindow((Display*)fX11.fDisplay, fDrawing);
      XFlush((Display*)fX11.fDisplay);
   } else {
      fX11.SetColor(this, &fGClist[kGCpxmp], 0);
      XFillRectangle((Display*)fX11.fDisplay, fDrawing, fGClist[kGCpxmp],
                     0, 0, fWidth, fHeight);
      fX11.SetColor(this, &fGClist[kGCpxmp], 1);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Close window

void X11Drawable::CloseWindow()
{
   if (fBuffer)
      XFreePixmap((Display*)fX11.fDisplay, fBuffer);

   if (!fNewColors.empty()) {
      if (fX11.fRedDiv == -1)
         XFreeColors((Display*)fX11.fDisplay, fX11.fColormap, fNewColors.data(), fNewColors.size(), 0);
      fNewColors.clear();
   }

   if (!fShared) { // if not QT window
      if (fIsPixmap)
         XFreePixmap((Display*)fX11.fDisplay, fWindow);
      else
         XDestroyWindow((Display*)fX11.fDisplay, fWindow);

      XFlush((Display*)fX11.fDisplay);
   }

   for (int i = 0; i < kMAXGC; ++i)
      XFreeGC((Display*)fX11.fDisplay, fGClist[i]);

   if (fillPattern != 0) {
      XFreePixmap((Display*)fX11.fDisplay, fillPattern);
      fillPattern = 0;
   }

   fOpen = 0;

   fX11.RemoveWindowHandle(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Update display for specified window.
///
///  \param [in] mode : (1) update (0) sync
///
/// Synchronise client and server once (not permanent).
/// Copy the pixmap gCws->fDrawing on the window gCws->fWindow
/// if the double buffer is on.

void X11Drawable::UpdateWindow(Int_t mode)
{
   if (fDoubleBuffer) {
      XCopyArea((Display*)fX11.fDisplay, fDrawing, fWindow,
                fGClist[kGCpxmp], 0, 0, fWidth, fHeight, 0, 0);
   }
   if (mode == 1) {
      XFlush((Display*)fX11.fDisplay);
   } else {
      XSync((Display*)fX11.fDisplay, False);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set opacity. This image manipulation routine works
/// by adding to a percent amount of neutral to each pixels RGB.
/// Since it requires quite some additional color map entries is it
/// only supported on displays with more than > 8 color planes (> 256
/// colors).

void X11Drawable::SetOpacity(Int_t percent)
{
   if ((fX11.fDepth <= 8) || (percent <= 0)) return;
   if (percent > 100) percent = 100;

   // get pixmap from server as image
   XImage *image = XGetImage((Display*)fX11.fDisplay, fDrawing, 0, 0, fWidth,
                             fHeight, AllPlanes, ZPixmap);
   if (!image) return;

   // collect different image colors
   std::vector<ULong_t> orgcolors;
   for (UInt_t y = 0; y < fHeight; y++) {
      for (UInt_t x = 0; x < fWidth; x++) {
         ULong_t pixel = XGetPixel(image, x, y);
         if (std::find(orgcolors.begin(), orgcolors.end(), pixel) == orgcolors.end())
            orgcolors.emplace_back(pixel);
      }
   }
   if (orgcolors.empty()) {
      XDestroyImage(image);
      return;
   }

   // clean up old colors
   if (!fNewColors.empty()) {
      if (fX11.fRedDiv == -1)
         XFreeColors((Display*)fX11.fDisplay, fX11.fColormap, fNewColors.data(), fNewColors.size(), 0);
      fNewColors.clear();
   }

   std::vector<RXColor> xcol(orgcolors.size());

   for (std::size_t i = 0; i < orgcolors.size(); i++) {
      xcol[i].pixel = orgcolors[i];
      xcol[i].red   = xcol[i].green = xcol[i].blue = 0;
      xcol[i].flags = DoRed | DoGreen | DoBlue;
   }
   fX11.QueryColors(fX11.fColormap, xcol.data(), orgcolors.size());

   // create new colors mixing:  "100-percent" of old color and "percent" of new background color
   XColor_t &bkgr = fX11.GetColor(fAttFill.GetFillColor());

   for (std::size_t i = 0; i < orgcolors.size(); i++) {
      xcol[i].red   = (UShort_t) TMath::Min((Int_t) xcol[i].red * (100 - percent) / 100  + bkgr.fRed * percent / 100, kBIGGEST_RGB_VALUE);
      xcol[i].green = (UShort_t) TMath::Min((Int_t) xcol[i].green * (100 - percent) / 100  + bkgr.fGreen * percent / 100, kBIGGEST_RGB_VALUE);
      xcol[i].blue  = (UShort_t) TMath::Min((Int_t) xcol[i].blue * (100 - percent) / 100  + bkgr.fBlue * percent / 100, kBIGGEST_RGB_VALUE);
      if (!fX11.AllocColor(fX11.fColormap, &xcol[i]))
         ::Warning("SetOpacity", "failed to allocate color %hd, %hd, %hd",
                 xcol[i].red, xcol[i].green, xcol[i].blue);
      // assumes that in case of failure xcol[i].pixel is not changed
   }

   fNewColors.resize(orgcolors.size());

   for (std::size_t i = 0; i < orgcolors.size(); i++)
      fNewColors[i] = xcol[i].pixel;

   // put opaque colors in image
   for (UInt_t y = 0; y < fHeight; y++) {
      for (UInt_t x = 0; x < fWidth; x++) {
         ULong_t pixel = XGetPixel(image, x, y);
         auto iter = std::find(orgcolors.begin(), orgcolors.end(), pixel);
         if (iter != orgcolors.end()) {
            auto idx = iter - orgcolors.begin();
            XPutPixel(image, x, y, fNewColors[idx]);
         }
      }
   }

   // put image back in pixmap on server
   XPutImage((Display *)fX11.fDisplay, fDrawing, fGClist[kGCpxmp], image, 0, 0, 0, 0, fWidth, fHeight);
   XFlush((Display *)fX11.fDisplay);

   XDestroyImage(image);
}

////////////////////////////////////////////////////////////////////////////////
/// Copy the pixmap wid at the position xpos, ypos in the specified window.

void X11Drawable::CopyPixmap(Int_t wid, Int_t xpos, Int_t ypos)
{
   auto win = fX11.fWindows[wid];
   if (!win)
      return;

   XCopyArea((Display*)fX11.fDisplay, win->fDrawing, fDrawing, win->fGClist[kGCpxmp], 0, 0, win->fWidth,
             win->fHeight, xpos, ypos);
   XFlush((Display*)fX11.fDisplay);
}

void X11Drawable::DrawBox(Int_t x1, Int_t y1, Int_t x2, Int_t y2, TVirtualX::EBoxMode mode)
{
   Int_t x = TMath::Min(x1, x2);
   Int_t y = TMath::Min(y1, y2);
   Int_t w = TMath::Abs(x2 - x1);
   Int_t h = TMath::Abs(y2 - y1);

   switch (mode) {

      case TVirtualX::kHollow:
         XDrawRectangle((Display*)fX11.fDisplay, fDrawing, fGClist[kGCline], x, y, w, h);
         break;

      case TVirtualX::kFilled:
         XFillRectangle((Display*)fX11.fDisplay, fDrawing, fGClist[kGCfill], x, y, w, h);
         break;

      default:
         break;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Fill area described by polygon on specified window
///
///  \param [in] n     number of points
///  \param [in] xy   list of points

void X11Drawable::DrawFillArea(Int_t n, TPoint *xy)
{
   XPoint *xyp = (XPoint *)xy;

   if (fillHollow)
      XDrawLines((Display *)fX11.fDisplay, fDrawing, fGClist[kGCfill], xyp, n, CoordModeOrigin);
   else
      XFillPolygon((Display *)fX11.fDisplay, fDrawing, fGClist[kGCfill], xyp, n, Nonconvex, CoordModeOrigin);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw a line
///
///  \param [in] x1,y1        : begin of line
///  \param [in] x2,y2        : end of line

void X11Drawable::DrawLine(Int_t x1, Int_t y1, Int_t x2, Int_t y2)
{
   if (lineStyle == LineSolid)
      XDrawLine((Display*)fX11.fDisplay, fDrawing, fGClist[kGCline], x1, y1, x2, y2);
   else {
      XSetDashes((Display*)fX11.fDisplay, fGClist[kGCdash], dashOffset, dashList.data(), dashList.size());
      XDrawLine((Display*)fX11.fDisplay, fDrawing, fGClist[kGCdash], x1, y1, x2, y2);
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Draw a line through all points
///
///  \param [in] n     number of points
///  \param [in] xy   list of points

void X11Drawable::DrawPolyLine(Int_t n, TPoint *xy)
{
   XPoint *xyp = (XPoint*)xy;

   const Int_t kMaxPoints = 1000001;

   if (n > kMaxPoints) {
      int ibeg = 0;
      int iend = kMaxPoints - 1;
      while (iend < n) {
         DrawPolyLine(kMaxPoints, &xy[ibeg]);
         ibeg = iend;
         iend += kMaxPoints - 1;
      }
      if (ibeg < n) {
         int npt = n - ibeg;
         DrawPolyLine(npt, &xy[ibeg]);
      }
   } else if (n > 1) {
      if (lineStyle == LineSolid)
         XDrawLines((Display*)fX11.fDisplay, fDrawing, fGClist[kGCline], xyp, n, CoordModeOrigin);
      else {
         XSetDashes((Display*)fX11.fDisplay, fGClist[kGCdash], dashOffset, dashList.data(), dashList.size());
         XDrawLines((Display*)fX11.fDisplay, fDrawing, fGClist[kGCdash], xyp, n, CoordModeOrigin);

         // calculate length of line to update dash offset
         for (int i = 1; i < n; i++) {
            int dx = xyp[i].x - xyp[i-1].x;
            int dy = xyp[i].y - xyp[i-1].y;
            if (dx < 0) dx = - dx;
            if (dy < 0) dy = - dy;
            dashOffset += dx > dy ? dx : dy;
         }
         dashOffset %= dashLength;
      }
   } else {
      int px = xyp[0].x;
      int py = xyp[0].y;
      XDrawPoint((Display*)fX11.fDisplay, fDrawing,
                 lineStyle == LineSolid ? fGClist[kGCline] : fGClist[kGCdash], px, py);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Draws N segments between provided points
///
/// \param [in] n    number of segements
/// \param [in] xy   list of points, size 2*n

void X11Drawable::DrawLinesSegments(Int_t n, TPoint *xy)
{
   const Int_t kMaxSegments = 500000;
   if (n > kMaxSegments) {
      Int_t ibeg = 0;
      Int_t iend = kMaxSegments;
      while (ibeg < n) {
         DrawLinesSegments(iend - ibeg, &xy[ibeg*2]);
         ibeg = iend;
         iend = TMath::Min(n, iend + kMaxSegments);
      }
   } else if (n > 0) {
      if (lineStyle == LineSolid)
         XDrawSegments((Display*)fX11.fDisplay, fDrawing, fGClist[kGCline], (XSegment *) xy, n);
      else {
         XSetDashes((Display*)fX11.fDisplay, fGClist[kGCdash], dashOffset, dashList.data(), dashList.size());
         XDrawSegments((Display*)fX11.fDisplay, fDrawing, fGClist[kGCdash], (XSegment *) xy, n);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Draw n markers with the current attributes at position x, y
///
///  \param [in] n     number of markers to draw
///  \param [in] xy   x,y coordinates of markers

void X11Drawable::DrawPolyMarker(Int_t n, TPoint *xy)
{
   XPoint *xyp = (XPoint *) xy;

   if (markerType == TAttMarker::kShapeDot) {
      const int kNMAX = 1000000;
      int nt = n/kNMAX;
      for (int it = 0; it <= nt; it++) {
         if (it < nt) {
            XDrawPoints((Display*)fX11.fDisplay, fDrawing, fGClist[kGCmark], &xyp[it*kNMAX], kNMAX, CoordModeOrigin);
         } else {
            XDrawPoints((Display*)fX11.fDisplay, fDrawing, fGClist[kGCmark], &xyp[it*kNMAX], n-it*kNMAX, CoordModeOrigin);
         }
      }
   } else {
      int r = markerSize / 2;
      auto &shape = markerShape;

      for (int m = 0; m < n; m++) {
         for (auto &pnt : shape) {
            pnt.fX += xyp[m].x;
            pnt.fY += xyp[m].y;
         }
         switch(markerType) {
            case TAttMarker::kShapeDot:
               // dot - handled before
               break;
            case TAttMarker::kShapeCircle:
               // hollow circle
               XDrawArc((Display*)fX11.fDisplay, fDrawing, fGClist[kGCmark],
                         xyp[m].x - r, xyp[m].y - r, markerSize, markerSize, 0, 360*64);
               break;
            case TAttMarker::kShapeFilledCircle:
               // filled circle
               XFillArc((Display*)fX11.fDisplay, fDrawing, fGClist[kGCmark],
                        xyp[m].x - r, xyp[m].y - r, markerSize, markerSize, 0, 360*64);
               break;
            case TAttMarker::kShapePolyLine:
               // hollow polygon
               XDrawLines((Display*)fX11.fDisplay, fDrawing, fGClist[kGCmark],
                          (XPoint *) shape.data(), shape.size(), CoordModeOrigin);
               break;
            case TAttMarker::kShapeFilledArea:
               // filled polygon
               XFillPolygon((Display*)fX11.fDisplay, fDrawing, fGClist[kGCmark],
                            (XPoint *) shape.data(), shape.size(), Nonconvex, CoordModeOrigin);
               break;
            case TAttMarker::kShapeSegments:
               // segmented line
               XDrawSegments((Display*)fX11.fDisplay, fDrawing, fGClist[kGCmark],
                             (XSegment *) shape.data(), shape.size()/2);
               break;
            case TAttMarker::kShapeTriangles:
               // filled triangles
               for (std::size_t t = 0; t < shape.size(); t += 3) {
                  XFillPolygon((Display*)fX11.fDisplay, fDrawing, fGClist[kGCmark],
                               (XPoint *) shape.data() + t, 3, Nonconvex, CoordModeOrigin);
               }
               break;
         }
         for (auto &pnt : shape) {
            pnt.fX -= xyp[m].x;
            pnt.fY -= xyp[m].y;
         }

      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Draw a text string using current font on specified window.
///
///  \param [in] mode       : drawing mode
///              - mode=0     : the background is not drawn (kClear)
///              - mode=1     : the background is drawn (kOpaque)
///  \param [in] x,y        : text position
///  \param [in] angle      : text angle
///  \param [in] mgn        : magnification factor
///  \param [in] text       : text string

void X11Drawable::DrawText(Int_t x, Int_t y, Float_t angle, Float_t mgn,
                         const char *text, TVirtualX::ETextMode mode)
{
   if (!text || !textFont || fAttText.GetTextSize() < 0)
      return;

   XRotSetMagnification(mgn);

   switch (mode) {

      case TVirtualX::kClear:
         XRotDrawAlignedString((Display*)fX11.fDisplay, textFont, angle,
                      fDrawing, fGClist[kGCtext], x, y, (char*)text, textAlign);
         break;

      case TVirtualX::kOpaque:
         XRotDrawAlignedImageString((Display*)fX11.fDisplay, textFont, angle,
                      fDrawing, fGClist[kGCtext], x, y, (char*)text, textAlign);
         break;

      default:
         break;
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Helper class to extract pixel information from XImage for store in the GIF

class TX11GifEncode : public TGifEncode {
   private:
      XImage *fXimage = nullptr;
   protected:
      void get_scline(int y, int width, unsigned char *buf) override
      {
         for (int x = 0; x < width; x++) {
            ULong_t pixel = XGetPixel(fXimage, x, y);
            buf[x] = 0;

            auto iter = std::find(orgcolors.begin(), orgcolors.end(), pixel);
            if (iter != orgcolors.end()) {
               auto idx = iter - orgcolors.begin();
               if (idx < 256)
                  buf[x] = (unsigned char) idx;
            }
         }
      }
   public:
      std::vector<ULong_t> orgcolors; // all unique pixels

      TX11GifEncode(XImage *image) : fXimage(image) {}
};


////////////////////////////////////////////////////////////////////////////////
/// Writes the specified window into GIF file. Returns 1 in case of success,
/// 0 otherwise.

Int_t X11Drawable::WriteGIF(const char *name)
{
   XImage *image = XGetImage((Display*)fX11.fDisplay, fDrawing, 0, 0,
                             fWidth, fHeight,
                             AllPlanes, ZPixmap);

   if (!image) {
      Error("WriteGIFW", "Cannot create image for writing GIF. Try in batch mode.");
      return 0;
   }

   /// Collect R G B colors of the palette used by the image.
   /// The image pixels are changed to index values in these R G B arrays.
   /// This produces a colormap with only the used colors (so even on displays
   /// with more than 8 planes we will be able to create GIF's when the image
   /// contains no more than 256 different colors). If it does contain more
   /// colors we will have to use GIFquantize to reduce the number of colors.

   TX11GifEncode gif(image);

   // collect different image colors
   for (UInt_t x = 0; x < fWidth; x++) {
      for (UInt_t y = 0; y < fHeight; y++) {
         ULong_t pixel = XGetPixel(image, x, y);
         if (std::find(gif.orgcolors.begin(), gif.orgcolors.end(), pixel) == gif.orgcolors.end())
            gif.orgcolors.emplace_back(pixel);
      }
   }

   auto ncolors = gif.orgcolors.size();

   if (ncolors > 256) {
      Error("WriteGIFW", "Cannot create GIF of image containing more than 256 colors. Try in batch mode.");
      XDestroyImage(image);
      return 0;
   }

   // get RGB values belonging to pixels
   std::vector<RXColor> xcol(ncolors);

   for (std::size_t i = 0; i < ncolors; i++) {
      xcol[i].pixel = gif.orgcolors[i];
      xcol[i].red   = xcol[i].green = xcol[i].blue = 0;
      xcol[i].flags = DoRed | DoGreen | DoBlue;
   }

   fX11.QueryColors(fX11.fColormap, xcol.data(), ncolors);

   UShort_t maxcol = 0;
   for (std::size_t i = 0; i < ncolors; i++) {
      maxcol = TMath::Max(maxcol, xcol[i].red);
      maxcol = TMath::Max(maxcol, xcol[i].green);
      maxcol = TMath::Max(maxcol, xcol[i].blue);
   }
   if (maxcol == 0)
      maxcol = 255;

   std::vector<unsigned char> r(ncolors), b(ncolors), g(ncolors);

   for (std::size_t i = 0; i < ncolors; i++) {
      r[i] = (unsigned char) (xcol[i].red * 255 / maxcol);
      g[i] = (unsigned char) (xcol[i].green * 255 / maxcol);
      b[i] = (unsigned char) (xcol[i].blue * 255 / maxcol);
   }

   Int_t ret = 0;

   if (gif.OpenFile(name)) {
      auto len = gif.GIFencode(fWidth, fHeight, ncolors, r.data(), g.data(), b.data());
      if (len > 0)
         ret = 1;
      gif.CloseFile();
   } else {
      ::Error("WriteGIF", "cannot write file: %s", name);
   }

   // cleanup image at the end
   XDestroyImage(image);

   return ret;
}
