// @(#)root/base:$Name$:$Id$
// Author: Rene Brun   05/12/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVirtualPad
#define ROOT_TVirtualPad


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualPad                                                          //
//                                                                      //
// Abstract base class for Pads and Canvases                            //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TAttPad
#include "TAttPad.h"
#endif

#ifndef ROOT_TVirtualX
#include "TVirtualX.h"
#endif

#ifndef ROOT_TString
#include "TString.h"
#endif

#ifndef ROOT_Buttons
#include "Buttons.h"
#endif

// forward declarations
class TObject;
class TObjLink;
class TView;
class TCanvas;
class TH1F;
class TFrame;
class TBox;
class TPadView3D;

class TVirtualPad : public TObject, public TAttLine, public TAttFill, public TAttPad {

protected:
   Bool_t       fResizing;         //!true when resizing the pad

public:
   TVirtualPad();
   TVirtualPad(const char *name, const char *title, Float_t xlow,
               Float_t ylow, Float_t xup, Float_t yup,
               Color_t color=19, Short_t bordersize=4, Short_t bordermode=1);
   virtual ~TVirtualPad();
   virtual void     AbsCoordinates(Bool_t set) = 0;
   virtual Float_t  AbsPixeltoX(Int_t px) = 0;
   virtual Float_t  AbsPixeltoY(Int_t py) = 0;
   virtual void     AddExec(const char *name, const char *command) = 0;
   virtual void     cd(Int_t subpadnumber=0) = 0;
   virtual void     Clear(Option_t *option="") = 0;
   virtual void     Close(Option_t *option="") = 0;
   virtual void     CopyPixmap() = 0;
   virtual void     CopyPixmaps() = 0;
   virtual void     DeleteExec(const char *name) = 0;
   virtual void     Divide(Int_t nx=1, Int_t ny=1, Float_t xmargin=0.01, Float_t ymargin=0.01, Int_t color=0) = 0;
   virtual void     Draw(Option_t *option="") = 0;
   virtual void     DrawClassObject(TObject *obj, Option_t *option="") = 0;
   virtual TH1F    *DrawFrame(Float_t xmin, Float_t ymin, Float_t xmax, Float_t ymax, const char *title="") = 0;
   virtual void     DrawLine(Float_t x1, Float_t y1, Float_t x2, Float_t y2) = 0;
   virtual void     DrawLineNDC(Float_t u1, Float_t v1, Float_t u2, Float_t v2) = 0;
   virtual void     DrawText(Float_t x, Float_t y, const char *text) = 0;
   virtual void     DrawTextNDC(Float_t u, Float_t v, const char *text) = 0;
   virtual Short_t  GetBorderMode() = 0;
   virtual Short_t  GetBorderSize() = 0;
   virtual Int_t    GetCanvasID() const = 0;
   virtual TCanvas  *GetCanvas() = 0;
   virtual TVirtualPad *GetVirtCanvas() = 0;
   virtual Int_t    GetEvent() const  = 0;
   virtual Int_t    GetEventX() const = 0;
   virtual Int_t    GetEventY() const = 0;
   virtual TFrame   *GetFrame() = 0;
   virtual Color_t  GetHighLightColor() const = 0;
   virtual void     GetRange(Float_t &x1, Float_t &y1, Float_t &x2, Float_t &y2) = 0;
   virtual void     GetRangeAxis(Axis_t &xmin, Axis_t &ymin, Axis_t &xmax, Axis_t &ymax) = 0;
   virtual void     GetPadPar(Float_t &xlow, Float_t &ylow, Float_t &xup, Float_t &yup) = 0;
   virtual Float_t  GetXlowNDC() = 0;
   virtual Float_t  GetYlowNDC() = 0;
   virtual Float_t  GetWNDC() = 0;
   virtual Float_t  GetHNDC() = 0;
   virtual UInt_t   GetWw() = 0;
   virtual UInt_t   GetWh() = 0;
   virtual Float_t  GetAbsXlowNDC() = 0;
   virtual Float_t  GetAbsYlowNDC() = 0;
   virtual Float_t  GetAbsWNDC() = 0;
   virtual Float_t  GetAbsHNDC() = 0;
   virtual Float_t  GetPhi() = 0;
   virtual Float_t  GetTheta() = 0;
   virtual Float_t  GetUxmin() = 0;
   virtual Float_t  GetUymin() = 0;
   virtual Float_t  GetUxmax() = 0;
   virtual Float_t  GetUymax() = 0;
   virtual Bool_t   GetGridx() = 0;
   virtual Bool_t   GetGridy() = 0;
   virtual Int_t    GetTickx() = 0;
   virtual Int_t    GetTicky() = 0;
   virtual Float_t  GetX1() const = 0;
   virtual Float_t  GetX2() const = 0;
   virtual Float_t  GetY1() const = 0;
   virtual Float_t  GetY2() const = 0;
   virtual TList    *GetListOfPrimitives() = 0;
   virtual TList    *GetListOfExecs() = 0;
   virtual TObject  *GetPrimitive(const char *name) = 0;
   virtual TObject  *GetSelected() = 0;
   virtual TObject  *GetPadPointer() = 0;
   virtual TVirtualPad  *GetPadSave() const = 0;
   virtual TVirtualPad  *GetSelectedPad() const = 0;
   virtual TView    *GetView() = 0;
   virtual Int_t    GetLogx() = 0;
   virtual Int_t    GetLogy() = 0;
   virtual Int_t    GetLogz() = 0;
   virtual TVirtualPad  *GetMother() = 0;
   virtual const char *GetName() const = 0;
   virtual const char *GetTitle() const = 0;
   virtual Int_t    GetPadPaint() = 0;
   virtual Int_t    GetPixmapID() = 0;
   virtual TPadView3D *GetView3D() = 0;
   virtual void     HighLight(Color_t col=kRed, Bool_t set=kTRUE) = 0;
   virtual Bool_t   IsBatch() = 0;
   Bool_t           IsBeingResized() const { return fResizing; }
   virtual Bool_t   IsEditable() = 0;
   virtual Bool_t   IsModified() = 0;
   virtual Bool_t   IsRetained() = 0;
   virtual void     ls(Option_t *option="") = 0;
   virtual void     Modified(Bool_t flag=1) = 0;
   virtual Bool_t   OpaqueMoving() const = 0;
   virtual Bool_t   OpaqueResizing() const = 0;
   virtual Float_t  PadtoX(Axis_t x) const = 0;
   virtual Float_t  PadtoY(Axis_t y) const = 0;
   virtual void     Paint(Option_t *option="") = 0;
   virtual void     PaintBorderPS(Float_t xl,Float_t yl,Float_t xt,Float_t yt,Int_t bmode,Int_t bsize,Int_t dark,Int_t light) = 0;
   virtual void     PaintBox(Float_t x1, Float_t y1, Float_t x2, Float_t y2, Option_t *option="") = 0;
   virtual void     PaintFillArea(Int_t n, Float_t *x, Float_t *y, Option_t *option="") = 0;
   virtual void     PaintPadFrame(Float_t xmin, Float_t ymin, Float_t xmax, Float_t ymax) = 0;
   virtual void     PaintLine(Float_t x1, Float_t y1, Float_t x2, Float_t y2) = 0;
   virtual void     PaintLineNDC(Coord_t u1, Coord_t v1,Coord_t u2, Coord_t v2) = 0;
   virtual void     PaintLine3D(Float_t *p1, Float_t *p2) = 0;
   virtual void     PaintPolyLine(Int_t n, Float_t *x, Float_t *y, Option_t *option="") = 0;
   virtual void     PaintPolyLine3D(Int_t n, Float_t *p) = 0;
   virtual void     PaintPolyLineNDC(Int_t n, Float_t *x, Float_t *y, Option_t *option="") = 0;
   virtual void     PaintPolyMarker(Int_t n, Float_t *x, Float_t *y, Option_t *option="") = 0;
   virtual void     PaintModified() = 0;
   virtual void     PaintText(Float_t x, Float_t y, const char *text) = 0;
   virtual void     PaintTextNDC(Float_t u, Float_t v, const char *text) = 0;
   virtual Float_t  PixeltoX(Int_t px) = 0;
   virtual Float_t  PixeltoY(Int_t py) = 0;
   virtual void     Pop() = 0;
   virtual void     Print(const char *filename="") = 0;
   virtual void     Print(const char *filename, Option_t *option) = 0;
   virtual void     Range(Float_t x1, Float_t y1, Float_t x2, Float_t y2) = 0;
   virtual void     RangeAxis(Axis_t xmin, Axis_t ymin, Axis_t xmax, Axis_t ymax) = 0;
   virtual void     RecursiveRemove(TObject *obj) = 0;
   virtual void     RedrawAxis(Option_t *option="") = 0;
   virtual void     ResetView3D(TPadView3D *view=0) = 0;
   virtual void     ResizePad(Option_t *option="") = 0;
   virtual void     SaveAs(const char *filename="") = 0;
   virtual void     SetBatch(Bool_t batch=kTRUE) = 0;
   virtual void     SetBorderMode(Short_t bordermode) = 0;
   virtual void     SetBorderSize(Short_t bordersize) = 0;
   virtual void     SetCanvasSize(UInt_t ww, UInt_t wh) = 0;
   virtual void     SetCursor(ECursor cursor) = 0;
   virtual void     SetDoubleBuffer(Int_t mode=1) = 0;
   virtual void     SetEditable(Bool_t mode=kTRUE) = 0;
   virtual void     SetGrid(Int_t valuex = 1, Int_t valuey = 1) = 0;
   virtual void     SetGridx(Int_t value = 1) = 0;
   virtual void     SetGridy(Int_t value = 1) = 0;
   virtual void     SetLogx(Int_t value = 1) = 0;
   virtual void     SetLogy(Int_t value = 1) = 0;
   virtual void     SetLogz(Int_t value = 1) = 0;
   virtual void     SetPad(const char *name, const char *title,
                           Float_t xlow, Float_t ylow, Float_t xup,
                           Float_t yup, Color_t color=35,
                           Short_t bordersize=5, Short_t bordermode=-1) = 0;
   virtual void     SetPad(Float_t xlow, Float_t ylow, Float_t xup, Float_t yup) = 0;
   virtual void     SetAttFillPS(Color_t color, Style_t style) = 0;
   virtual void     SetAttLinePS(Color_t color, Style_t style, Width_t lwidth) = 0;
   virtual void     SetAttMarkerPS(Color_t color, Style_t style, Size_t msize) = 0;
   virtual void     SetAttTextPS(Int_t align, Float_t angle, Color_t color, Style_t font, Float_t tsize) = 0;
   virtual void     SetName(const char *name) = 0;
   virtual void     SetSelected(TObject *obj) = 0;
   virtual void     SetTicks(Int_t valuex = 1, Int_t valuey = 1) = 0;
   virtual void     SetTickx(Int_t value = 1) = 0;
   virtual void     SetTicky(Int_t value = 1) = 0;
   virtual void     SetTitle(const char *title="") = 0;
   virtual void     SetTheta(Float_t theta=30) = 0;
   virtual void     SetPhi(Float_t phi=30) = 0;
   virtual void     SetToolTipText(const char *text, Long_t delayms = 1000) = 0;
   virtual void     SetView(TView *view) = 0;
   virtual TObject *WaitPrimitive(const char *pname="", const char *emode="") = 0;
   virtual void     Update() = 0;
   virtual Int_t    UtoAbsPixel(Float_t u) const = 0;
   virtual Int_t    VtoAbsPixel(Float_t v) const = 0;
   virtual Int_t    UtoPixel(Float_t u) const = 0;
   virtual Int_t    VtoPixel(Float_t v) const = 0;
   virtual Int_t    XtoAbsPixel(Axis_t x) const = 0;
   virtual Int_t    YtoAbsPixel(Axis_t y) const = 0;
   virtual Float_t  XtoPad(Axis_t x) const = 0;
   virtual Float_t  YtoPad(Axis_t y) const = 0;
   virtual Int_t    XtoPixel(Axis_t x) const = 0;
   virtual Int_t    YtoPixel(Axis_t y) const = 0;

   virtual TObject *CreateToolTip(const TBox *b, const char *text, Long_t delayms) = 0;
   virtual void     DeleteToolTip(TObject *tip) = 0;
   virtual void     ResetToolTip(TObject *tip) = 0;
   virtual void     CloseToolTip(TObject *tip) = 0;

   virtual void     x3d(Option_t *option="") = 0;

   static TVirtualPad *&Pad();

   ClassDef(TVirtualPad,1)  //Abstract base class for Pads and Canvases
};

#ifndef __CINT__
#define gPad (TVirtualPad::Pad())

R__EXTERN void **(*gThreadTsd)(void*,Int_t);
#endif
R__EXTERN Int_t (*gThreadXAR)(const char *xact, Int_t nb, void **ar, Int_t *iret);

#endif
