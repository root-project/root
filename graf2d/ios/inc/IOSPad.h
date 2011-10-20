// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov, 14/8/2011

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_IOSPad
#define ROOT_IOSPad

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// IOSPad                                                               //
//                                                                      //
// This class is a pad for iOS.                                         //
// I use this class instead of original TPad/TCanvas to                 //
// keep it as simple and clean as possible, and to be able              //
// to extend this class for iOS only.                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <utility>
#include <vector>
#include <string>

#include <CoreGraphics/CGContext.h>

#ifndef ROOT_TLine
#include "TList.h"
#endif

#ifndef ROOT_IOSTextOperations
#include "IOSTextOperations.h"
#endif

#ifndef ROOT_IOSPainter
#include "IOSPainter.h"
#endif

#ifndef ROOT_IOSPadStub
#include "IOSPadStub.h"
#endif

namespace ROOT {
namespace iOS {

// TODO: class needs verification and clean-up.
// Some parts from original TPad (coordinate conversions/painting algorithms)
// were copy & pasted at the beginning, will be re-written and fixed.

class Pad : public PadStub {
public:
   // Pad status bits
   enum EStatusBits {
      kFraming      = BIT(6),
      kHori         = BIT(9),
      kClipFrame    = BIT(10),
      kPrintingPS   = BIT(11),
      kCannotMove   = BIT(12),
      kClearAfterCR = BIT(14)
   };

   void GetTextExtent(UInt_t &w, UInt_t &h, const char *text);
   void SetContext(CGContextRef ctx);

   Pad(UInt_t w, UInt_t h);
   ~Pad();
   
   //Final-overriders for TVirtualPad class.
   const char *GetName() const;
   const char *GetTitle() const;
   
   void Clear(Option_t * = "");
   
   //Coordinates, conversions, ranges.
   void AbsCoordinates(Bool_t set) {fAbsCoord = set;}
   
   Double_t AbsPixeltoX(Int_t px) {return fAbsPixeltoXk + px * fPixeltoX;}
   Double_t AbsPixeltoY(Int_t py) {return fAbsPixeltoYk + py * fPixeltoY;}
   
   Double_t GetXlowNDC() const {return fXlowNDC;}
   Double_t GetYlowNDC() const {return fYlowNDC;}
   
   Double_t GetWNDC() const {return fWNDC;}
   Double_t GetHNDC() const {return fHNDC;}

   void SetViewWH(UInt_t w, UInt_t h);

   
   UInt_t GetWw() const;
   UInt_t GetWh() const;
   
   Double_t GetAbsXlowNDC() const {return fAbsXlowNDC;}
   Double_t GetAbsYlowNDC() const {return fAbsYlowNDC;}
   
   Double_t GetAbsWNDC() const {return fAbsWNDC;}
   Double_t GetAbsHNDC() const {return fAbsHNDC;}
   
   Double_t GetAspectRatio() const { return fAspectRatio; }
   void SetFixedAspectRatio(Bool_t fixed = kTRUE);
   Bool_t HasFixedAspectRatio() const {return fFixedAspectRatio;}
      
   Double_t GetUxmin() const {return fUxmin;}
   Double_t GetUymin() const {return fUymin;}
   Double_t GetUxmax() const {return fUxmax;}
   Double_t GetUymax() const {return fUymax;}
   
   void SetGrid(Int_t valuex = 1, Int_t valuey = 1) {fGridx = valuex; fGridy = valuey; }
      
   void SetGridx(Int_t value = 1) {fGridx = value;}
   Bool_t GetGridx() const {return fGridx;}
   
   void SetGridy(Int_t value = 1) {fGridy = value;}
   Bool_t GetGridy() const {return fGridy;}
   
   void SetTicks(Int_t valuex = 1, Int_t valuey = 1) {fTickx = valuex; fTicky = valuey;}
   
   void SetTickx(Int_t value = 1) {fTickx = value;}
   Int_t GetTickx() const {return fTickx;}
   
   void SetTicky(Int_t value = 1) {fTicky = value;}
   Int_t GetTicky() const {return fTicky;}
   
   Double_t GetX1() const { return fX1; }
   Double_t GetX2() const { return fX2; }
   Double_t GetY1() const { return fY1; }
   Double_t GetY2() const { return fY2; }

   Double_t PadtoX(Double_t x) const;
   Double_t PadtoY(Double_t y) const;
   Double_t XtoPad(Double_t x) const;
   Double_t YtoPad(Double_t y) const;

   Int_t UtoAbsPixel(Double_t u) const {return Int_t(fUtoAbsPixelk + u * fUtoPixel);}
   Int_t VtoAbsPixel(Double_t v) const {return Int_t(fVtoAbsPixelk + v * fVtoPixel);}
   
   Int_t UtoPixel(Double_t u) const;
   Int_t VtoPixel(Double_t v) const;
   
   Int_t XtoAbsPixel(Double_t x) const;
   Int_t YtoAbsPixel(Double_t y) const;
   
   Int_t XtoPixel(Double_t x) const;
   Int_t YtoPixel(Double_t y) const;

   Double_t PixeltoX(Int_t px);
   Double_t PixeltoY(Int_t py);

   void SetLogx(Int_t value = 1);
   Int_t GetLogx() const {return fLogx;}
   
   void SetLogy(Int_t value = 1);
   Int_t GetLogy() const {return fLogy;}
   
   void SetLogz(Int_t value = 1);
   Int_t GetLogz() const {return fLogz;}
   
   void SetPad(const char *name, const char *title,
               Double_t xlow, Double_t ylow, Double_t xup,
               Double_t yup, Color_t color=35,
               Short_t bordersize=5, Short_t bordermode=-1);
   void SetPad(Double_t xlow, Double_t ylow, Double_t xup, Double_t yup);
   
   void SetTheta(Double_t theta=30) {fTheta = theta;}
   Double_t GetTheta() const {return fTheta;}

   void SetPhi(Double_t phi=30) {fPhi = phi;}
   Double_t GetPhi() const   {return fPhi;}
   
   void SetVertical(Bool_t vert=kTRUE);

   void GetRange(Double_t &x1, Double_t &y1, Double_t &x2, Double_t &y2);
   void Range(Double_t x1, Double_t y1, Double_t x2, Double_t y2);
   
   void GetRangeAxis(Double_t &xmin, Double_t &ymin, Double_t &xmax, Double_t &ymax);
   void RangeAxis(Double_t xmin, Double_t ymin, Double_t xmax, Double_t ymax);
   
   void GetPadPar(Double_t &xlow, Double_t &ylow, Double_t &xup, Double_t &yup);

   //Pad's properties.
   void SetFillStyle(Style_t fstyle);

   Bool_t IsBatch() const {return kFALSE;}
   
   void SetEditable(Bool_t mode = kTRUE);
   Bool_t IsEditable() const {return fEditable;}

   Bool_t IsFolder() const {return kTRUE;}
   Bool_t IsModified() const {return kFALSE;}
   Bool_t IsRetained() const {return kTRUE;}
   Bool_t IsVertical() const {return !TestBit(kHori);}

   void SetBorderMode(Short_t bordermode) {fBorderMode = bordermode; }   
   Short_t GetBorderMode() const {return fBorderMode;}
   
   void SetBorderSize(Short_t bordersize) {fBorderSize = bordersize; }
   Short_t GetBorderSize() const {return fBorderSize;}

   //Misc. pad operations.
   TVirtualPad* cd(Int_t dummyNumber = -1);
   
   void SetView(TView *view = 0);
   
   void ResizePad(Option_t *option="");
   
   TFrame *GetFrame();
   
   TList *GetListOfPrimitives() const {return &fPrimitives;}
   TList *GetListOfExecs() const {return &fExecs;}
   
   TObject *GetPrimitive(const char *name) const;
   TView *GetView() const {return fView;}

   //Graphics.
   void Paint(Option_t *option="");
   void PaintBox(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Option_t *option="");
   void PaintFillArea(Int_t n, Float_t *x, Float_t *y, Option_t *option="");
   void PaintFillArea(Int_t n, Double_t *x, Double_t *y, Option_t *option="");
 
   void PaintPadFrame(Double_t xmin, Double_t ymin, Double_t xmax, Double_t ymax);
   void PaintLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2);
   void PaintLineNDC(Double_t u1, Double_t v1,Double_t u2, Double_t v2);
   void PaintLine3D(Float_t *p1, Float_t *p2);
   void PaintLine3D(Double_t *p1, Double_t *p2);
   void PaintPolyLine(Int_t n, Float_t *x, Float_t *y, Option_t *option="");
   void PaintPolyLine(Int_t n, Double_t *x, Double_t *y, Option_t *option="");
   void PaintPolyLine3D(Int_t n, Double_t *p);
   void PaintPolyLineNDC(Int_t n, Double_t *x, Double_t *y, Option_t *option="");
   void PaintPolyMarker(Int_t n, Float_t *x, Float_t *y, Option_t *option="");
   void PaintPolyMarker(Int_t n, Double_t *x, Double_t *y, Option_t *option="");
   
   void PaintText(Double_t x, Double_t y, const char *text);
   void PaintTextNDC(Double_t u, Double_t v, const char *text);

   void RedrawAxis(Option_t *option="");
   TH1F *DrawFrame(Double_t xmin, Double_t ymin, Double_t xmax, Double_t ymax, const char *title);
   
   //
   TVirtualViewer3D *GetViewer3D(Option_t *opt);
   
   
   //
   Bool_t PadInSelectionMode() const
   {
      return fInSelectionMode;
   }
   
   Bool_t PadInHighlightMode() const
   {
      return fInHighlightMode;
   }
   
   //This is a temporary hack, will be removed when selection
   //is done correctly.
   void ExecuteRotateView(Int_t evType, Int_t px, Int_t py);
   void ExecuteEventAxis(Int_t, Int_t, Int_t, TAxis *);
   
   Bool_t SelectionIsValid()const;
   void InvalidateSelection(Bool_t invalidateBufferOnly = kFALSE);

   void PaintForSelection();
   void SetSelectionBuffer(UInt_t w, UInt_t h, unsigned char *buff);
   void Pick(Int_t px, Int_t py);
   void Unpick();
   TObject *ObjectInPoint(Int_t px, Int_t py);

   TObject *GetSelected() const;

   void PaintShadowForSelected() const;
   void PaintSelected() const;
   
   void PushTopLevelSelectable(TObject *top);

   void PushSelectableObject(TObject *obj);

   void PopTopLevelSelectable();
   
   //New special function.
   void PaintThumbnail();
   
   void SetPaintOption(const TObject *obj, const char *option);
   
private:
   //Painter based on CoreGraphics.
   mutable Painter fPainter;

   //View's width and height (from UIKit).
   UInt_t fViewW;
   UInt_t fViewH;

   //Standard pad's coordinates and coefficients.
   Double_t      fX1;              //  X of lower X coordinate
   Double_t      fY1;              //  Y of lower Y coordinate
   Double_t      fX2;              //  X of upper X coordinate
   Double_t      fY2;              //  Y of upper Y coordinate

   Double_t      fXtoAbsPixelk;    //  Conversion coefficient for X World to absolute pixel
   Double_t      fXtoPixelk;       //  Conversion coefficient for X World to pixel
   Double_t      fXtoPixel;        //    xpixel = fXtoPixelk + fXtoPixel*xworld
   Double_t      fYtoAbsPixelk;    //  Conversion coefficient for Y World to absolute pixel
   Double_t      fYtoPixelk;       //  Conversion coefficient for Y World to pixel
   Double_t      fYtoPixel;        //    ypixel = fYtoPixelk + fYtoPixel*yworld

   Double_t      fUtoAbsPixelk;    //  Conversion coefficient for U NDC to absolute pixel
   Double_t      fUtoPixelk;       //  Conversion coefficient for U NDC to pixel
   Double_t      fUtoPixel;        //    xpixel = fUtoPixelk + fUtoPixel*undc
   Double_t      fVtoAbsPixelk;    //  Conversion coefficient for V NDC to absolute pixel
   Double_t      fVtoPixelk;       //  Conversion coefficient for V NDC to pixel
   Double_t      fVtoPixel;        //    ypixel = fVtoPixelk + fVtoPixel*vndc

   Double_t      fAbsPixeltoXk;    //  Conversion coefficient for absolute pixel to X World
   Double_t      fPixeltoXk;       //  Conversion coefficient for pixel to X World
   Double_t      fPixeltoX;        //     xworld = fPixeltoXk + fPixeltoX*xpixel
   Double_t      fAbsPixeltoYk;    //  Conversion coefficient for absolute pixel to Y World
   Double_t      fPixeltoYk;       //  Conversion coefficient for pixel to Y World
   Double_t      fPixeltoY;        //     yworld = fPixeltoYk + fPixeltoY*ypixel

   Double_t      fXlowNDC;         //  X bottom left corner of pad in NDC [0,1]
   Double_t      fYlowNDC;         //  Y bottom left corner of pad in NDC [0,1]
   Double_t      fWNDC;            //  Width of pad along X in NDC
   Double_t      fHNDC;            //  Height of pad along Y in NDC

   Double_t      fAbsXlowNDC;      //  Absolute X top left corner of pad in NDC [0,1]
   Double_t      fAbsYlowNDC;      //  Absolute Y top left corner of pad in NDC [0,1]
   Double_t      fAbsWNDC;         //  Absolute Width of pad along X in NDC
   Double_t      fAbsHNDC;         //  Absolute Height of pad along Y in NDC

   Double_t      fUxmin;           //  Minimum value on the X axis
   Double_t      fUymin;           //  Minimum value on the Y axis
   Double_t      fUxmax;           //  Maximum value on the X axis
   Double_t      fUymax;           //  Maximum value on the Y axis

   Double_t      fTheta;           //  theta angle to view as lego/surface
   Double_t      fPhi;             //  phi angle   to view as lego/surface

   Double_t      fAspectRatio;     //  ratio of w/h in case of fixed ratio

   Int_t         fTickx;           //  Set to 1 if tick marks along X
   Int_t         fTicky;           //  Set to 1 if tick marks along Y
   Int_t         fLogx;            //  (=0 if X linear scale, =1 if log scale)
   Int_t         fLogy;            //  (=0 if Y linear scale, =1 if log scale)
   Int_t         fLogz;            //  (=0 if Z linear scale, =1 if log scale)
   Short_t       fBorderSize;      //  pad bordersize in pixels
   Short_t       fBorderMode;      //  Bordermode (-1=down, 0 = no border, 1=up)
   
   Bool_t        fGridx;           //  Set to true if grid along X
   Bool_t        fGridy;           //  Set to true if grid along Y
   Bool_t        fAbsCoord;        //  Use absolute coordinates
   Bool_t        fEditable;        //  True if canvas is editable
   Bool_t        fFixedAspectRatio; //  True if fixed aspect ratio

   mutable TList fPrimitives;      // List of primitives in a pad.
   mutable TList fExecs;           // List of commands to be executed when a pad event occurs (empty on iOS).

   TFrame       *fFrame;           // Pointer to 2-D frame (if one exists)
   TView        *fView;            // Pointer to 3-D view (if one exists)
   
   //
   TVirtualViewer3D *fViewer3D;

   //Selection trick now!
   Bool_t                     fSelectionIsValid;
   UInt_t                     fSelectionAreaWidth;
   std::vector<unsigned char> fSelectionBuffer;

   TObject                   *fSelected;
   TObject                   *fParentOfSelected;
   
   mutable Bool_t             fInSelectionMode;
   mutable Bool_t             fInHighlightMode;
   
   typedef std::pair<TObject *, UInt_t> Parent_t;
   std::vector<Parent_t>      fParentPainters;
   
   typedef std::pair<TObject *, TObject *> ObjectPair_t;
   std::vector<ObjectPair_t>  fSelectables;
   
   UInt_t fObjectID;
   //This requires special processing at the moment.
   Bool_t fContains3DObject;
   
   const char *GetSelectedParentDrawOption()const;
   const char *GetSelectedDrawOption()const;

   //Non-overriders.
   TObject *FindObject(const char *name) const;
   TObject *FindObject(const TObject *obj) const;
   
   Int_t Clip(Float_t *x, Float_t *y, Float_t xclipl, Float_t yclipb, Float_t xclipr, Float_t yclipt);
   Int_t Clip(Double_t *x, Double_t *y, Double_t xclipl, Double_t yclipb, Double_t xclipr, Double_t yclipt);
   Int_t ClippingCode(Double_t x, Double_t y, Double_t xcl1, Double_t ycl1, Double_t xcl2, Double_t ycl2);
   Int_t ClipPolygon(Int_t n, Double_t *x, Double_t *y, Int_t nn, Double_t *xc, Double_t *yc, Double_t xclipl, Double_t yclipb, Double_t xclipr, Double_t yclipt);

   void PaintBorder(Color_t color, Bool_t tops);
   void PaintFillAreaHatches(Int_t n, Double_t *x, Double_t *y, Int_t FillStyle);
   void PaintHatches(Double_t dy, Double_t angle, Int_t nn, Double_t *xx, Double_t *yy);

   Pad(const Pad &pad);
   Pad &operator=(const Pad &rhs);
};

}//namespace iOS
}//namespace ROOT

#endif
