// @(#)root/gpad:$Id$
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPad
#define ROOT_TPad

#include "TVirtualPad.h"
#include "TAttBBox2D.h"
#include <vector>

class TVirtualViewer3D;
class TVirtualPadPainter;
class TBrowser;
class TBox;
class TLegend;
class TArrow;
class TPoint;

class TPad : public TVirtualPad, public TAttBBox2D {

private:
   TObject      *fTip;              ///<! tool tip associated with box

protected:
   Double_t      fX1;               ///<  X of lower X coordinate
   Double_t      fY1;               ///<  Y of lower Y coordinate
   Double_t      fX2;               ///<  X of upper X coordinate
   Double_t      fY2;               ///<  Y of upper Y coordinate

   Double_t      fXtoAbsPixelk;     ///<  Conversion coefficient for X World to absolute pixel
   Double_t      fXtoPixelk;        ///<  Conversion coefficient for X World to pixel
   Double_t      fXtoPixel;         ///<    xpixel = fXtoPixelk + fXtoPixel*xworld
   Double_t      fYtoAbsPixelk;     ///<  Conversion coefficient for Y World to absolute pixel
   Double_t      fYtoPixelk;        ///<  Conversion coefficient for Y World to pixel
   Double_t      fYtoPixel;         ///<    ypixel = fYtoPixelk + fYtoPixel*yworld

   Double_t      fUtoAbsPixelk;     ///<  Conversion coefficient for U NDC to absolute pixel
   Double_t      fUtoPixelk;        ///<  Conversion coefficient for U NDC to pixel
   Double_t      fUtoPixel;         ///<    xpixel = fUtoPixelk + fUtoPixel*undc
   Double_t      fVtoAbsPixelk;     ///<  Conversion coefficient for V NDC to absolute pixel
   Double_t      fVtoPixelk;        ///<  Conversion coefficient for V NDC to pixel
   Double_t      fVtoPixel;         ///<    ypixel = fVtoPixelk + fVtoPixel*vndc

   Double_t      fAbsPixeltoXk;     ///<  Conversion coefficient for absolute pixel to X World
   Double_t      fPixeltoXk;        ///<  Conversion coefficient for pixel to X World
   Double_t      fPixeltoX;         ///<     xworld = fPixeltoXk + fPixeltoX*xpixel
   Double_t      fAbsPixeltoYk;     ///<  Conversion coefficient for absolute pixel to Y World
   Double_t      fPixeltoYk;        ///<  Conversion coefficient for pixel to Y World
   Double_t      fPixeltoY;         ///<     yworld = fPixeltoYk + fPixeltoY*ypixel

   Double_t      fXlowNDC;          ///<  X bottom left corner of pad in NDC [0,1]
   Double_t      fYlowNDC;          ///<  Y bottom left corner of pad in NDC [0,1]
   Double_t      fXUpNDC;
   Double_t      fYUpNDC;
   Double_t      fWNDC;             ///<  Width of pad along X in Normalized Coordinates (NDC)
   Double_t      fHNDC;             ///<  Height of pad along Y in Normalized Coordinates (NDC)

   Double_t      fAbsXlowNDC;       ///<  Absolute X top left corner of pad in NDC [0,1]
   Double_t      fAbsYlowNDC;       ///<  Absolute Y top left corner of pad in NDC [0,1]
   Double_t      fAbsWNDC;          ///<  Absolute Width of pad along X in NDC
   Double_t      fAbsHNDC;          ///<  Absolute Height of pad along Y in NDC

   Double_t      fUxmin;            ///<  Minimum value on the X axis
   Double_t      fUymin;            ///<  Minimum value on the Y axis
   Double_t      fUxmax;            ///<  Maximum value on the X axis
   Double_t      fUymax;            ///<  Maximum value on the Y axis

   Double_t      fTheta;            ///<  theta angle to view as lego/surface
   Double_t      fPhi;              ///<  phi angle   to view as lego/surface

   Double_t      fAspectRatio;      ///<  ratio of w/h in case of fixed ratio

   Int_t         fPixmapID;         ///<! Off-screen pixmap identifier
   Int_t         fGLDevice;         ///<! OpenGL off-screen pixmap identifier
   Bool_t        fCopyGLDevice;     ///<!
   Bool_t        fEmbeddedGL;       ///<!
   Int_t         fNumber;           ///<  pad number identifier
   Int_t         fTickx;            ///<  Set to 1 if tick marks along X
   Int_t         fTicky;            ///<  Set to 1 if tick marks along Y
   Int_t         fLogx;             ///<  (=0 if X linear scale, =1 if log scale)
   Int_t         fLogy;             ///<  (=0 if Y linear scale, =1 if log scale)
   Int_t         fLogz;             ///<  (=0 if Z linear scale, =1 if log scale)
   Int_t         fPadPaint;         ///<  Set to 1 while painting the pad
   Int_t         fCrosshair;        ///<  Crosshair type (0 if no crosshair requested)
   Int_t         fCrosshairPos;     ///<  Position of crosshair
   Short_t       fBorderSize;       ///<  pad bordersize in pixels
   Short_t       fBorderMode;       ///<  Bordermode (-1=down, 0 = no border, 1=up)
   Bool_t        fModified;         ///<  Set to true when pad is modified
   Bool_t        fGridx;            ///<  Set to true if grid along X
   Bool_t        fGridy;            ///<  Set to true if grid along Y
   Bool_t        fAbsCoord;         ///<  Use absolute coordinates
   Bool_t        fEditable;         ///<  True if canvas is editable
   Bool_t        fFixedAspectRatio; ///<  True if fixed aspect ratio
   TPad         *fMother;           ///<! pointer to mother of the list
   TCanvas      *fCanvas;           ///<! Pointer to mother canvas
   TList        *fPrimitives;       ///<->List of primitives (subpads)
   TList        *fExecs;            ///<  List of commands to be executed when a pad event occurs
   TString       fName;             ///<  Pad name
   TString       fTitle;            ///<  Pad title
   TFrame       *fFrame;            ///<! Pointer to 2-D frame (if one exists)
   TView        *fView;             ///<! Pointer to 3-D view (if one exists)
   TObject      *fPadPointer;       ///<! free pointer
   TObject      *fPadView3D;        ///<! 3D View of this TPad
   static Int_t  fgMaxPickDistance; ///<  Maximum Pick Distance
   Int_t         fNumPaletteColor;  ///<  Number of objects with an automatic color
   Int_t         fNextPaletteColor; ///<  Next automatic color
   std::vector<Bool_t> fCollideGrid;///<! Grid used to find empty space when adding a box (Legend) in a pad
   Int_t         fCGnx;             ///<! Size of the collide grid along x
   Int_t         fCGny;             ///<! Size of the collide grid along y

   // 3D Viewer support
   TVirtualViewer3D *fViewer3D;     ///<! Current 3D viewer

   void          DestroyExternalViewer3D();
   Int_t         DistancetoPrimitive(Int_t px, Int_t py) override;
   void          ExecuteEvent(Int_t event, Int_t px, Int_t py) override;
   virtual void  HideToolTip(Int_t event);
   void          PaintBorder(Color_t color, Bool_t tops);
   void          PaintBorderPS(Double_t xl,Double_t yl,Double_t xt,Double_t yt,Int_t bmode,Int_t bsize,Int_t dark,Int_t light) override;
   void          PaintDate();
   void          SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void          SetBatch(Bool_t batch=kTRUE) override;

private:
   TPad(const TPad &pad) = delete;
   TPad &operator=(const TPad &rhs) = delete;

   void CopyBackgroundPixmap(Int_t x, Int_t y);
   void CopyBackgroundPixmaps(TPad *start, TPad *stop, Int_t x, Int_t y);
   void DrawDist(Rectangle_t aBBox, Rectangle_t bBBox, char mode);

   Bool_t            Collide(Int_t i, Int_t j, Int_t w, Int_t h);
   void              FillCollideGrid(TObject *o);
   void              FillCollideGridTBox(TObject *o);
   void              FillCollideGridTFrame(TObject *o);
   void              FillCollideGridTGraph(TObject *o);
   void              FillCollideGridTH1(TObject *o);
   void              LineNotFree(Int_t x1, Int_t x2, Int_t y1, Int_t y2);

public:
   // TPad status bits
   enum {
      kFraming      = BIT(6),  ///< Frame is requested
      kHori         = BIT(9),  ///< Pad is horizontal
      kClipFrame    = BIT(10), ///< Clip on frame
      kPrintingPS   = BIT(11), ///< PS Printing
      kCannotMove   = BIT(12), ///< Fixed position
      kClearAfterCR = BIT(14)  ///< Clear after CR
   };

   TPad();
   TPad(const char *name, const char *title, Double_t xlow,
        Double_t ylow, Double_t xup, Double_t yup,
        Color_t color=-1, Short_t bordersize=-1, Short_t bordermode=-2);
   virtual ~TPad();
   void              AbsCoordinates(Bool_t set) override { fAbsCoord = set; }
   Double_t          AbsPixeltoX(Int_t px) override { return fAbsPixeltoXk + px*fPixeltoX; }
   Double_t          AbsPixeltoY(Int_t py) override { return fAbsPixeltoYk + py*fPixeltoY; }
   virtual void      AbsPixeltoXY(Int_t xpixel, Int_t ypixel, Double_t &x, Double_t &y);
   void              AddExec(const char *name, const char *command) override;
   virtual void      AutoExec();
   void              Browse(TBrowser *b) override;
   TLegend          *BuildLegend(Double_t x1=0.3, Double_t y1=0.21, Double_t x2=0.3, Double_t y2=0.21, const char *title="", Option_t *option = "") override; // *MENU*
   TVirtualPad      *cd(Int_t subpadnumber=0) override; // *MENU*
   void              Clear(Option_t *option="") override;
   virtual Int_t     Clip(Float_t *x, Float_t *y, Float_t xclipl, Float_t yclipb, Float_t xclipr, Float_t yclipt);
   Int_t             Clip(Double_t *x, Double_t *y, Double_t xclipl, Double_t yclipb, Double_t xclipr, Double_t yclipt) override;
   virtual Int_t     ClippingCode(Double_t x, Double_t y, Double_t xcl1, Double_t ycl1, Double_t xcl2, Double_t ycl2);
   virtual Int_t     ClipPolygon(Int_t n, Double_t *x, Double_t *y, Int_t nn, Double_t *xc, Double_t *yc, Double_t xclipl, Double_t yclipb, Double_t xclipr, Double_t yclipt);
   void              Close(Option_t *option="") override;
   virtual void      Closed() { Emit("Closed()"); } // *SIGNAL*
   void              CopyPixmap() override;
   void              CopyPixmaps() override;
   void              DeleteExec(const char *name) override;
   void              Divide(Int_t nx=1, Int_t ny=1, Float_t xmargin=0.01, Float_t ymargin=0.01, Int_t color=0) override; // *MENU*
   virtual void      DivideSquare(Int_t n, Float_t xmargin=0.01, Float_t ymargin=0.01, Int_t color=0);
   void              Draw(Option_t *option="") override;
   void              DrawClassObject(const TObject *obj, Option_t *option="") override;
   static  void      DrawColorTable();
   virtual void      DrawCrosshair();
   TH1F             *DrawFrame(Double_t xmin, Double_t ymin, Double_t xmax, Double_t ymax, const char *title="") override;
   void              ExecuteEventAxis(Int_t event, Int_t px, Int_t py, TAxis *axis) override;
   TObject          *FindObject(const char *name) const override;
   TObject          *FindObject(const TObject *obj) const override;
   void              UseCurrentStyle() override;  // *MENU*
   Short_t           GetBorderMode() const override { return fBorderMode;}
   Short_t           GetBorderSize() const override { return fBorderSize;}
   Int_t             GetCrosshair() const;
   Int_t             GetCanvasID() const override;
   TCanvasImp       *GetCanvasImp() const override;
   TFrame           *GetFrame() override;
   Int_t             GetEvent() const override;
   Int_t             GetEventX() const override;
   Int_t             GetEventY() const override;
   Color_t           GetHighLightColor() const override;
   void              GetRange(Double_t &x1, Double_t &y1, Double_t &x2, Double_t &y2) override;
   void              GetRangeAxis(Double_t &xmin, Double_t &ymin, Double_t &xmax, Double_t &ymax) override;
   void              GetPadPar(Double_t &xlow, Double_t &ylow, Double_t &xup, Double_t &yup) override;
   Double_t          GetXlowNDC() const override { return fXlowNDC; }
   Double_t          GetYlowNDC() const override { return fYlowNDC; }
   /// Get width of pad along X in Normalized Coordinates (NDC)
   Double_t          GetWNDC() const override { return fWNDC; }
   /// Get height of pad along Y in Normalized Coordinates (NDC)
   Double_t          GetHNDC() const override { return fHNDC; }
   UInt_t            GetWw() const override;
   UInt_t            GetWh() const override;
   Double_t          GetAbsXlowNDC() const override { return fAbsXlowNDC; }
   Double_t          GetAbsYlowNDC() const override { return fAbsYlowNDC; }
   Double_t          GetAbsWNDC() const override { return fAbsWNDC; }
   Double_t          GetAbsHNDC() const override { return fAbsHNDC; }
   Double_t          GetAspectRatio() const override { return fAspectRatio; }
   Double_t          GetPhi() const override { return fPhi; }
   Double_t          GetTheta() const override { return fTheta; }
   ///Returns the minimum x-coordinate value visible on the pad. If log axis the returned value is in decades.
   Double_t          GetUxmin() const override { return fUxmin; }
   ///Returns the minimum y-coordinate value visible on the pad. If log axis the returned value is in decades.
   Double_t          GetUymin() const override { return fUymin; }
   ///Returns the maximum x-coordinate value visible on the pad. If log axis the returned value is in decades.
   Double_t          GetUxmax() const override { return fUxmax; }
   ///Returns the maximum y-coordinate value visible on the pad. If log axis the returned value is in decades.
   Double_t          GetUymax() const override { return fUymax; }
   Bool_t            GetGridx() const override { return fGridx; }
   Bool_t            GetGridy() const override { return fGridy; }
   Int_t             GetNumber() const override { return fNumber; }
   Int_t             GetTickx() const override { return fTickx; }
   Int_t             GetTicky() const override { return fTicky; }
   Double_t          GetX1() const override { return fX1; }
   Double_t          GetX2() const override { return fX2; }
   Double_t          GetY1() const override { return fY1; }
   Double_t          GetY2() const override { return fY2; }
   static Int_t      GetMaxPickDistance();
   TList            *GetListOfPrimitives() const override { return fPrimitives; }
   TList            *GetListOfExecs() const override { return fExecs; }
   TObject          *GetPrimitive(const char *name) const override;  //obsolete, use FindObject instead
   TObject          *GetSelected() const override;
   TVirtualPad      *GetPad(Int_t subpadnumber) const override;
   TObject          *GetPadPointer() const override { return fPadPointer; }
   TVirtualPad      *GetPadSave() const override;
   TVirtualPad      *GetSelectedPad() const override;
   Int_t             GetGLDevice() override;
   TView            *GetView() const override { return fView; }
   TObject          *GetView3D() const override { return fPadView3D; }// Return 3D View of this TPad
   Int_t             GetLogx() const override { return fLogx; }
   Int_t             GetLogy() const override { return fLogy; }
   Int_t             GetLogz() const override { return fLogz; }
   TVirtualPad      *GetMother() const override { return fMother; }
   const char       *GetName() const override { return fName.Data(); }
   const char       *GetTitle() const override { return fTitle.Data(); }
   TCanvas          *GetCanvas() const override { return fCanvas; }
   TVirtualPad      *GetVirtCanvas() const override;
   TVirtualPadPainter *GetPainter() override;
   Int_t             GetPadPaint() const override { return fPadPaint; }
   Int_t             GetPixmapID() const override { return fPixmapID; }
   ULong_t           Hash() const override { return fName.Hash(); }
   Bool_t            HasCrosshair() const override;
   void              HighLight(Color_t col=kRed, Bool_t set=kTRUE) override;
   Bool_t            HasFixedAspectRatio() const override { return fFixedAspectRatio; }
   Bool_t            IsBatch() const override;
   virtual Bool_t    IsEditable() const override { return fEditable; }
   Bool_t            IsFolder() const override { return kTRUE; }
   Bool_t            IsModified() const override { return fModified; }
   Bool_t            IsRetained() const override;
   Bool_t            IsVertical() const override { return !TestBit(kHori); }
   void              ls(Option_t *option="") const override;
   void              Modified(Bool_t flag=1) override;  // *SIGNAL*
   Bool_t            OpaqueMoving() const override;
   Bool_t            OpaqueResizing() const override;
   Double_t          PadtoX(Double_t x) const override;
   Double_t          PadtoY(Double_t y) const override;
   void              Paint(Option_t *option="") override;
   void              PaintBox(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Option_t *option="") override;
   void              PaintFillArea(Int_t n, Float_t *x, Float_t *y, Option_t *option="") override; // Obsolete
   void              PaintFillArea(Int_t n, Double_t *x, Double_t *y, Option_t *option="") override;
   void              PaintFillAreaNDC(Int_t n, Double_t *x, Double_t *y, Option_t *option="") override;
   void              PaintFillAreaHatches(Int_t n, Double_t *x, Double_t *y, Int_t FillStyle);
   void              PaintHatches(Double_t dy, Double_t angle, Int_t nn, Double_t *xx, Double_t *yy);
   void              PaintPadFrame(Double_t xmin, Double_t ymin, Double_t xmax, Double_t ymax) override;
   void              PaintLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2) override;
   void              PaintLineNDC(Double_t u1, Double_t v1,Double_t u2, Double_t v2) override;
   void              PaintLine3D(Float_t *p1, Float_t *p2) override;
   void              PaintLine3D(Double_t *p1, Double_t *p2) override;
   void              PaintPolyLine(Int_t n, Float_t *x, Float_t *y, Option_t *option="") override;
   void              PaintPolyLine(Int_t n, Double_t *x, Double_t *y, Option_t *option="") override;
   void              PaintPolyLine3D(Int_t n, Double_t *p) override;
   void              PaintPolyLineNDC(Int_t n, Double_t *x, Double_t *y, Option_t *option="") override;
   void              PaintPolyMarker(Int_t n, Float_t *x, Float_t *y, Option_t *option="") override;
   void              PaintPolyMarker(Int_t n, Double_t *x, Double_t *y, Option_t *option="") override;
   void              PaintModified() override;
   void              PaintText(Double_t x, Double_t y, const char *text) override;
   void              PaintText(Double_t x, Double_t y, const wchar_t *text) override;
   void              PaintTextNDC(Double_t u, Double_t v, const char *text) override;
   void              PaintTextNDC(Double_t u, Double_t v, const wchar_t *text) override;
   virtual TPad     *Pick(Int_t px, Int_t py, TObjLink *&pickobj);
   Double_t          PixeltoX(Int_t px) override;
   Double_t          PixeltoY(Int_t py) override;
   virtual void      PixeltoXY(Int_t xpixel, Int_t ypixel, Double_t &x, Double_t &y);
   void              Pop() override;  // *MENU*
   void              Print(const char *filename="") const override;
   void              Print(const char *filename, Option_t *option) override;
   void              Range(Double_t x1, Double_t y1, Double_t x2, Double_t y2) override; // *MENU* *ARGS={x1=>fX1,y1=>fY1,x2=>fX2,y2=>fY2}
   virtual void      RangeChanged() { Emit("RangeChanged()"); } // *SIGNAL*
   void              RangeAxis(Double_t xmin, Double_t ymin, Double_t xmax, Double_t ymax) override;
   void              RecursiveRemove(TObject *obj) override;
   void              RedrawAxis(Option_t *option="") override;
   void              ResetView3D(TObject *view=nullptr) override { fPadView3D=view; }
   void              ResizePad(Option_t *option="") override;
   virtual void      Resized() { Emit("Resized()"); } // *SIGNAL*
   void              SaveAs(const char *filename="",Option_t *option="") const override; // *MENU*
   void              SetBorderMode(Short_t bordermode) override { fBorderMode = bordermode; Modified(); } // *MENU*
   void              SetBorderSize(Short_t bordersize) override { fBorderSize = bordersize; Modified(); } // *MENU*
   void              SetCanvas(TCanvas *c) override { fCanvas = c; }
   void              SetCanvasSize(UInt_t ww, UInt_t wh) override;
   void              SetCrosshair(Int_t crhair=1) override; // *TOGGLE*
   void              SetCursor(ECursor cursor) override;
   void              SetDoubleBuffer(Int_t mode=1) override;
   void              SetDrawOption(Option_t *option="") override;
   void              SetEditable(Bool_t mode=kTRUE) override; // *TOGGLE*
   void              SetFixedAspectRatio(Bool_t fixed = kTRUE) override;  // *TOGGLE*
   void              SetGrid(Int_t valuex = 1, Int_t valuey = 1) override { fGridx = valuex; fGridy = valuey; Modified(); }
   void              SetGridx(Int_t value = 1) override { fGridx = value; Modified(); } // *TOGGLE*
   void              SetGridy(Int_t value = 1) override { fGridy = value; Modified(); } // *TOGGLE*
   void              SetFillStyle(Style_t fstyle) override;
   void              SetLogx(Int_t value = 1) override; // *TOGGLE*
   void              SetLogy(Int_t value = 1) override; // *TOGGLE*
   void              SetLogz(Int_t value = 1) override; // *TOGGLE*
   virtual void      SetNumber(Int_t number) { fNumber = number; }
   void              SetPad(const char *name, const char *title,
                           Double_t xlow, Double_t ylow, Double_t xup,
                           Double_t yup, Color_t color=35,
                           Short_t bordersize=5, Short_t bordermode=-1) override;
   void              SetPad(Double_t xlow, Double_t ylow, Double_t xup, Double_t yup) override;
   void              SetAttFillPS(Color_t color, Style_t style) override;
   void              SetAttLinePS(Color_t color, Style_t style, Width_t lwidth) override;
   void              SetAttMarkerPS(Color_t color, Style_t style, Size_t msize) override;
   void              SetAttTextPS(Int_t align, Float_t angle, Color_t color, Style_t font, Float_t tsize) override;
   static  void      SetMaxPickDistance(Int_t maxPick=5);
   void              SetName(const char *name) override { fName = name; } // *MENU*
   void              SetSelected(TObject *obj) override;
   void              SetTicks(Int_t valuex = 1, Int_t valuey = 1) override { fTickx = valuex; fTicky = valuey; Modified(); }
   void              SetTickx(Int_t value = 1) override { fTickx = value; Modified(); } // *TOGGLE*
   void              SetTicky(Int_t value = 1) override { fTicky = value; Modified(); } // *TOGGLE*
   void              SetTitle(const char *title="") override { fTitle = title; }
   void              SetTheta(Double_t theta=30) override { fTheta = theta; Modified(); }
   void              SetPhi(Double_t phi=30) override { fPhi = phi; Modified(); }
   void              SetToolTipText(const char *text, Long_t delayms = 1000) override;
   void              SetVertical(Bool_t vert=kTRUE) override;
   void              SetView(TView *view = nullptr) override;
   void              SetViewer3D(TVirtualViewer3D *viewer3d) override { fViewer3D = viewer3d; }

   virtual void      SetGLDevice(Int_t dev) {fGLDevice = dev;}
   void              SetCopyGLDevice(Bool_t copy) override { fCopyGLDevice = copy; }

   void              ShowGuidelines(TObject *object, const Int_t event, const char mode = 'i', const bool cling = true) override;
   void              Update() override;
   Int_t             UtoAbsPixel(Double_t u) const override { return Int_t(fUtoAbsPixelk + u*fUtoPixel); }
   Int_t             VtoAbsPixel(Double_t v) const override { return Int_t(fVtoAbsPixelk + v*fVtoPixel); }
   Int_t             UtoPixel(Double_t u) const override;
   Int_t             VtoPixel(Double_t v) const override;
   TObject          *WaitPrimitive(const char *pname="", const char *emode="") override;
   Int_t             XtoAbsPixel(Double_t x) const override;
   Int_t             YtoAbsPixel(Double_t y) const override;
   Double_t          XtoPad(Double_t x) const override;
   Double_t          YtoPad(Double_t y) const override;
   Int_t             XtoPixel(Double_t x) const override;
   Int_t             YtoPixel(Double_t y) const override;
   virtual void      XYtoAbsPixel(Double_t x, Double_t y, Int_t &xpixel, Int_t &ypixel) const;
   virtual void      XYtoPixel(Double_t x, Double_t y, Int_t &xpixel, Int_t &ypixel) const;

   TObject          *CreateToolTip(const TBox *b, const char *text, Long_t delayms) override;
   void              DeleteToolTip(TObject *tip) override;
   void              ResetToolTip(TObject *tip) override;
   void              CloseToolTip(TObject *tip) override;

   Int_t             IncrementPaletteColor(Int_t i, TString opt) override;
   Int_t             NextPaletteColor() override;

   void              DrawCollideGrid();
   Bool_t            PlaceBox(TObject *o, Double_t w, Double_t h, Double_t &xl, Double_t &yb) override;

   virtual void      x3d(Option_t *type=""); // Depreciated

   TVirtualViewer3D *GetViewer3D(Option_t * type = "") override;
   Bool_t            HasViewer3D() const override { return fViewer3D != nullptr; }
   void              ReleaseViewer3D(Option_t * type = "") override;

   Rectangle_t       GetBBox() override;
   TPoint            GetBBoxCenter() override;
   void              SetBBoxCenter(const TPoint &p) override;
   void              SetBBoxCenterX(const Int_t x) override;
   void              SetBBoxCenterY(const Int_t y) override;
   void              SetBBoxX1(const Int_t x) override;
   void              SetBBoxX2(const Int_t x) override;
   void              SetBBoxY1(const Int_t y) override;
   void              SetBBoxY2(const Int_t y) override;

   virtual void      RecordPave(const TObject *obj);              // *SIGNAL*
   virtual void      RecordLatex(const TObject *obj);             // *SIGNAL*
   virtual void      EventPave() { Emit("EventPave()"); }         // *SIGNAL*
   virtual void      StartEditing() { Emit("StartEditing()"); }   // *SIGNAL*

   ClassDefOverride(TPad,13)  //A Graphics pad
};


//______________________________________________________________________________
inline void TPad::Modified(Bool_t flag)
{
   if (!fModified && flag) Emit("Modified()");
   fModified = flag;
}


//______________________________________________________________________________
inline void TPad::AbsPixeltoXY(Int_t xpixel, Int_t ypixel, Double_t &x, Double_t &y)
{
   x = AbsPixeltoX(xpixel);
   y = AbsPixeltoY(ypixel);
}


//______________________________________________________________________________
inline Double_t TPad::PixeltoX(Int_t px)
{
   if (fAbsCoord) return fAbsPixeltoXk + px*fPixeltoX;
   else           return fPixeltoXk    + px*fPixeltoX;
}


//______________________________________________________________________________
inline Double_t TPad::PixeltoY(Int_t py)
{
   if (fAbsCoord) return fAbsPixeltoYk + py*fPixeltoY;
   else           return fPixeltoYk    + py*fPixeltoY;
}


//______________________________________________________________________________
inline void TPad::PixeltoXY(Int_t xpixel, Int_t ypixel, Double_t &x, Double_t &y)
{
   x = PixeltoX(xpixel);
   y = PixeltoY(ypixel);
}


//______________________________________________________________________________
inline Int_t TPad::UtoPixel(Double_t u) const
{
   Double_t val;
   if (fAbsCoord) val = fUtoAbsPixelk + u*fUtoPixel;
   else           val = u*fUtoPixel;
   if (val < -kMaxPixel) return -kMaxPixel;
   if (val >  kMaxPixel) return  kMaxPixel;
   return Int_t(val);
}


//______________________________________________________________________________
inline Int_t TPad::VtoPixel(Double_t v) const
{
   Double_t val;
   if (fAbsCoord) val = fVtoAbsPixelk + v*fVtoPixel;
   else           val = fVtoPixelk    + v*fVtoPixel;
   if (val < -kMaxPixel) return -kMaxPixel;
   if (val >  kMaxPixel) return  kMaxPixel;
   return Int_t(val);
}


//______________________________________________________________________________
inline Int_t TPad::XtoAbsPixel(Double_t x) const
{
   Double_t val = fXtoAbsPixelk + x*fXtoPixel;
   if (val < -kMaxPixel) return -kMaxPixel;
   if (val >  kMaxPixel) return  kMaxPixel;
   return Int_t(val);
}


//______________________________________________________________________________
inline Int_t TPad::XtoPixel(Double_t x) const
{
   Double_t val;
   if (fAbsCoord) val = fXtoAbsPixelk + x*fXtoPixel;
   else           val = fXtoPixelk    + x*fXtoPixel;
   if (val < -kMaxPixel) return -kMaxPixel;
   if (val >  kMaxPixel) return  kMaxPixel;
   return Int_t(val);
}


//______________________________________________________________________________
inline Int_t TPad::YtoAbsPixel(Double_t y) const
{
   Double_t val = fYtoAbsPixelk + y*fYtoPixel;
   if (val < -kMaxPixel) return -kMaxPixel;
   if (val >  kMaxPixel) return  kMaxPixel;
   return Int_t(val);
}


//______________________________________________________________________________
inline Int_t TPad::YtoPixel(Double_t y) const
{
   Double_t val;
   if (fAbsCoord) val = fYtoAbsPixelk + y*fYtoPixel;
   else           val = fYtoPixelk    + y*fYtoPixel;
   if (val < -kMaxPixel) return -kMaxPixel;
   if (val >  kMaxPixel) return  kMaxPixel;
   return Int_t(val);
}


//______________________________________________________________________________
inline void TPad::XYtoAbsPixel(Double_t x, Double_t y, Int_t &xpixel, Int_t &ypixel) const
{
   xpixel = XtoAbsPixel(x);
   ypixel = YtoAbsPixel(y);
}


//______________________________________________________________________________
inline void TPad::XYtoPixel(Double_t x, Double_t y, Int_t &xpixel, Int_t &ypixel) const
{
   xpixel = XtoPixel(x);
   ypixel = YtoPixel(y);
}


//______________________________________________________________________________
inline void TPad::SetDrawOption(Option_t *)
{ }

#endif

