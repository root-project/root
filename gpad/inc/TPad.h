// @(#)root/gpad:$Name$:$Id$
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


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPad                                                                 //
//                                                                      //
// A Graphics pad.                                                      //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TVirtualPad
#include "TVirtualPad.h"
#endif

class TBrowser;
class TBox;

class TPad : public TVirtualPad {

private:
   TObject     *fTip;             //!tool tip associated with box

protected:
   Float_t      fX1;              //X of lower X coordinate
   Float_t      fY1;              //Y of lower Y coordinate
   Float_t      fX2;              //X of upper X coordinate
   Float_t      fY2;              //Y of upper Y coordinate
   Short_t      fBorderSize;      //pad bordersize in pixels
   Short_t      fBorderMode;      //Bordermode (-1=down, 0 = no border, 1=up)
   Int_t        fLogx;            //(=0 if X linear scale, =1 if log scale)
   Int_t        fLogy;            //(=0 if Y linear scale, =1 if log scale)
   Int_t        fLogz;            //(=0 if Z linear scale, =1 if log scale)

   Float_t      fXtoAbsPixelk;    //Conversion coefficient for X World to absolute pixel
   Float_t      fXtoPixelk;       //Conversion coefficient for X World to pixel
   Float_t      fXtoPixel;        // xpixel = fXtoPixelk + fXtoPixel*xworld
   Float_t      fYtoAbsPixelk;    //Conversion coefficient for Y World to absolute pixel
   Float_t      fYtoPixelk;       //Conversion coefficient for Y World to pixel
   Float_t      fYtoPixel;        // ypixel = fYtoPixelk + fYtoPixel*yworld

   Float_t      fUtoAbsPixelk;    //Conversion coefficient for U NDC to absolute pixel
   Float_t      fUtoPixelk;       //Conversion coefficient for U NDC to pixel
   Float_t      fUtoPixel;        // xpixel = fUtoPixelk + fUtoPixel*undc
   Float_t      fVtoAbsPixelk;    //Conversion coefficient for Y World to absolute pixel
   Float_t      fVtoPixelk;       //Conversion coefficient for Y World to pixel
   Float_t      fVtoPixel;        // ypixel = fVtoPixelk + fVtoPixel*vndc

   Float_t      fAbsPixeltoXk;    //Conversion coefficient for absolute pixel to X World
   Float_t      fPixeltoXk;       //Conversion coefficient for pixel to X World
   Float_t      fPixeltoX;        // xworld = fPixeltoXk + fPixeltoX*xpixel
   Float_t      fAbsPixeltoYk;    //Conversion coefficient for absolute pixel to Y World
   Float_t      fPixeltoYk;       //Conversion coefficient for pixel to Y World
   Float_t      fPixeltoY;        // yworld = fPixeltoYk + fPixeltoY*ypixel

   Float_t      fXlowNDC;         //X bottom left corner of pad in NDC [0,1]
   Float_t      fYlowNDC;         //Y bottom left corner of pad in NDC [0,1]
   Float_t      fWNDC;            //Width of pad along X in NDC
   Float_t      fHNDC;            //Height of pad along Y in NDC

   Float_t      fAbsXlowNDC;      //Absolute X top left corner of pad in NDC [0,1]
   Float_t      fAbsYlowNDC;      //Absolute Y top left corner of pad in NDC [0,1]
   Float_t      fAbsWNDC;         //Absolute Width of pad along X in NDC
   Float_t      fAbsHNDC;         //Absolute Height of pad along Y in NDC

   Float_t      fUxmin;           //Minimum value on the X axis
   Float_t      fUymin;           //Minimum value on the Y axis
   Float_t      fUxmax;           //Maximum value on the X axis
   Float_t      fUymax;           //Maximum value on the Y axis

   TPad         *fMother;         //pointer to mother of the list
   TCanvas      *fCanvas;         //!Pointer to mother canvas
   Int_t        fPixmapID;        //Off-screen pixmap identifier
   TList        *fPrimitives;     //List of primitives (subpads)
   TList        *fExecs;          //List of commands to be executed when a pad event occurs
   TString      fName;            //Pad name
   TString      fTitle;           //Pad title
   Int_t        fPadPaint;        //Set to 1 while painting the pad
   Bool_t       fModified;        //Set to true when pad is modified
   Bool_t       fGridx;           //Set to true if grid along X
   Bool_t       fGridy;           //Set to true if grid along Y
   Int_t        fTickx;           //Set to 1 if tick marks along X
   Int_t        fTicky;           //Set to 1 if tick marks along Y
   TFrame       *fFrame;          //Pointer to 2-D frame (if one exists)
   TView        *fView;           //Pointer to 3-D view (if one exists)
   Float_t      fTheta;           //theta angle to view as lego/surface
   Float_t      fPhi;             //phi angle   to view as lego/surface
   TObject      *fPadPointer;     //free pointer
   Int_t        fNumber;          //pad number identifier
   Bool_t       fAbsCoord;        //Use absolute coordinates
   Bool_t       fIsEditable;      //True if canvas is editable
   TPadView3D   *fPadView3D;      //3D View of this TPad

   virtual Int_t DistancetoPrimitive(Int_t px, Int_t py);
   virtual void  ExecuteEvent(Int_t event, Int_t px, Int_t py);
   virtual void  HideToolTip(Int_t event);
   void          PaintBorder(Color_t color, Bool_t tops);
   virtual void  PaintBorderPS(Float_t xl,Float_t yl,Float_t xt,Float_t yt,Int_t bmode,Int_t bsize,Int_t dark,Int_t light);
   virtual void  SavePrimitive(ofstream &out, Option_t *option);
   virtual void  SetBatch(Bool_t batch=kTRUE);

private:
   TPad(const TPad &pad);  // cannot copy pads, use TObject::Clone()
   TPad &operator=(const TPad &rhs);  // idem

   void CopyBackgroundPixmap(Int_t x, Int_t y);
   void CopyBackgroundPixmaps(TPad *start, TPad *stop, Int_t x, Int_t y);

public:
   // TPad status bits
   enum {
      kFraming = BIT(6),
      kPrintingPS = BIT(11)
   };

   TPad();
   TPad(const char *name, const char *title, Float_t xlow,
        Float_t ylow, Float_t xup, Float_t yup,
        Color_t color=-1, Short_t bordersize=-1, Short_t bordermode=-2);
   virtual ~TPad();
   void             AbsCoordinates(Bool_t set) { fAbsCoord = set; }
   Float_t          AbsPixeltoX(Int_t px) {return fAbsPixeltoXk + px*fPixeltoX;}
   Float_t          AbsPixeltoY(Int_t py) {return fAbsPixeltoYk + py*fPixeltoY;}
   virtual void     AbsPixeltoXY(Int_t xpixel, Int_t ypixel, Axis_t &x, Axis_t &y);
   virtual void     AddExec(const char *name, const char *command);
   virtual void     AutoExec();
   virtual void     Browse(TBrowser *b);
   void             cd(Int_t subpadnumber=0);
   void             Clear(Option_t *option="");
   virtual Int_t    Clip(Float_t *x, Float_t *y, Float_t xclipl, Float_t yclipb, Float_t xclipr, Float_t yclipt);
   virtual Int_t    ClippingCode(Float_t x, Float_t y, Float_t xcl1, Float_t ycl1, Float_t xcl2, Float_t ycl2);
   virtual void     Close(Option_t *option="");
   virtual void     CopyPixmap();
   virtual void     CopyPixmaps();
   virtual void     CreateNewLine(Int_t event, Int_t px, Int_t py, Int_t mode);
   virtual void     CreateNewEllipse(Int_t event, Int_t px, Int_t py,Int_t mode);
   virtual void     CreateNewPad(Int_t event, Int_t px, Int_t py, Int_t mode);
   virtual void     CreateNewPave(Int_t event, Int_t px, Int_t py, Int_t mode);
   virtual void     CreateNewPolyLine(Int_t event, Int_t px, Int_t py, Int_t mode);
   virtual void     CreateNewText(Int_t event, Int_t px, Int_t py, Int_t mode);
   virtual void     DeleteExec(const char *name);
   virtual void     Divide(Int_t nx=1, Int_t ny=1, Float_t xmargin=0.01, Float_t ymargin=0.01, Int_t color=0); // *MENU*
   virtual void     Draw(Option_t *option="");
   virtual void     DrawClassObject(TObject *obj, Option_t *option="");
   static  void     DrawColorTable();
   TH1F            *DrawFrame(Float_t xmin, Float_t ymin, Float_t xmax, Float_t ymax, const char *title="");
   void             DrawLine(Float_t x1, Float_t y1, Float_t x2, Float_t y2);
   void             DrawLineNDC(Float_t u1, Float_t v1, Float_t u2, Float_t v2);
   void             DrawText(Float_t x, Float_t y, const char *text);
   void             DrawTextNDC(Float_t u, Float_t v, const char *text);
   virtual void     UseCurrentStyle();  // *MENU*
   virtual Short_t  GetBorderMode() { return fBorderMode;}
   virtual Short_t  GetBorderSize() { return fBorderSize;}
   virtual Int_t    GetCanvasID() const;
   TFrame           *GetFrame();
   virtual Int_t    GetEvent() const;
   virtual Int_t    GetEventX() const;
   virtual Int_t    GetEventY() const;
   virtual Color_t  GetHighLightColor() const;
   virtual void     GetRange(Float_t &x1, Float_t &y1, Float_t &x2, Float_t &y2);
   virtual void     GetRangeAxis(Axis_t &xmin, Axis_t &ymin, Axis_t &xmax, Axis_t &ymax);
   virtual void     GetPadPar(Float_t &xlow, Float_t &ylow, Float_t &xup, Float_t &yup)
                     {xlow = fXlowNDC; ylow = fYlowNDC; xup = fXlowNDC+fWNDC; yup = fYlowNDC+fHNDC;}
   Float_t          GetXlowNDC() {return fXlowNDC;}
   Float_t          GetYlowNDC() {return fYlowNDC;}
   Float_t          GetWNDC() {return fWNDC;}
   Float_t          GetHNDC() {return fHNDC;}
   virtual UInt_t   GetWw();
   virtual UInt_t   GetWh();
   Float_t          GetAbsXlowNDC() {return fAbsXlowNDC;}
   Float_t          GetAbsYlowNDC() {return fAbsYlowNDC;}
   Float_t          GetAbsWNDC() {return fAbsWNDC;}
   Float_t          GetAbsHNDC() {return fAbsHNDC;}
   Float_t          GetPhi()   {return fPhi;}
   Float_t          GetTheta() {return fTheta;}
   Float_t          GetUxmin() {return fUxmin;}
   Float_t          GetUymin() {return fUymin;}
   Float_t          GetUxmax() {return fUxmax;}
   Float_t          GetUymax() {return fUymax;}
   Bool_t           GetGridx() {return fGridx;}
   Bool_t           GetGridy() {return fGridy;}
   Int_t            GetNumber() {return fNumber;}
   Int_t            GetTickx() {return fTickx;}
   Int_t            GetTicky() {return fTicky;}
   Float_t          GetX1() const { return fX1; }
   Float_t          GetX2() const { return fX2; }
   Float_t          GetY1() const { return fY1; }
   Float_t          GetY2() const { return fY2; }
   TList            *GetListOfPrimitives() {return fPrimitives;}
   TList            *GetListOfExecs() {return fExecs;}
   virtual TObject  *GetPrimitive(const char *name);
   virtual TObject  *GetSelected();
   virtual TObject  *GetPadPointer() {return fPadPointer;}
   TVirtualPad      *GetPadSave() const;
   TVirtualPad      *GetSelectedPad() const;
   TView            *GetView() {return fView;}
   TPadView3D       *GetView3D(){return fPadView3D;}// Return 3D View of this TPad
   Int_t            GetLogx() {return fLogx;}
   Int_t            GetLogy() {return fLogy;}
   Int_t            GetLogz() {return fLogz;}
   virtual TVirtualPad *GetMother() {return fMother;}
   const char      *GetName() const {return fName.Data();}
   const char      *GetTitle() const {return fTitle.Data();}
   virtual TCanvas *GetCanvas() { return fCanvas; }
   virtual TVirtualPad *GetVirtCanvas();
   Int_t            GetPadPaint() {return fPadPaint;}
   Int_t            GetPixmapID() {return fPixmapID;}
   void             HighLight(Color_t col=kRed, Bool_t set=kTRUE);
   virtual Bool_t   IsBatch();
   virtual Bool_t   IsEditable() {return fIsEditable;}
   Bool_t           IsFolder() {return kTRUE;}
   Bool_t           IsModified() {return fModified;}
   virtual Bool_t   IsRetained();
   virtual void     ls(Option_t *option="");
   void             Modified(Bool_t flag=1) { fModified = flag; }
   virtual Bool_t   OpaqueMoving() const;
   virtual Bool_t   OpaqueResizing() const;
   Float_t          PadtoX(Float_t x) const;
   Float_t          PadtoY(Float_t y) const;
   virtual void     Paint(Option_t *option="");
   void             PaintBox(Float_t x1, Float_t y1, Float_t x2, Float_t y2, Option_t *option="");
   void             PaintFillArea(Int_t n, Float_t *x, Float_t *y, Option_t *option="");
   void             PaintPadFrame(Float_t xmin, Float_t ymin, Float_t xmax, Float_t ymax);
   void             PaintLine(Float_t x1, Float_t y1, Float_t x2, Float_t y2);
   void             PaintLineNDC(Coord_t u1, Coord_t v1,Coord_t u2, Coord_t v2);
   void             PaintLine3D(Float_t *p1, Float_t *p2);
   void             PaintPolyLine(Int_t n, Float_t *x, Float_t *y, Option_t *option="");
   void             PaintPolyLine3D(Int_t n, Float_t *p);
   void             PaintPolyLineNDC(Int_t n, Float_t *x, Float_t *y, Option_t *option="");
   void             PaintPolyMarker(Int_t n, Float_t *x, Float_t *y, Option_t *option="");
   virtual void     PaintModified();
   void             PaintText(Float_t x, Float_t y, const char *text);
   void             PaintTextNDC(Float_t u, Float_t v, const char *text);
   virtual TPad    *Pick(Int_t px, Int_t py, TObjLink *&pickobj);
   Float_t          PixeltoX(Int_t px);
   Float_t          PixeltoY(Int_t py);
   virtual void     PixeltoXY(Int_t xpixel, Int_t ypixel, Axis_t &x, Axis_t &y);
   virtual void     Pop();
   virtual void     Print(const char *filename="");
   virtual void     Print(const char *filename, Option_t *option);
   virtual void     Range(Float_t x1, Float_t y1, Float_t x2, Float_t y2); // *MENU* *ARGS={x1=>fX1,y1=>fY1,x2=>fX2,y2=>fY2}
   virtual void     RangeAxis(Axis_t xmin, Axis_t ymin, Axis_t xmax, Axis_t ymax);
   virtual void     RecursiveRemove(TObject *obj);
   virtual void     RedrawAxis(Option_t *option="");
   virtual void     ResetView3D(TPadView3D *view=0){fPadView3D=view;}
   virtual void     ResizePad(Option_t *option="");
   virtual void     SaveAs(const char *filename=""); // *MENU*
   virtual void     SetBorderMode(Short_t bordermode) {fBorderMode = bordermode;} // *MENU*
   virtual void     SetBorderSize(Short_t bordersize) {fBorderSize = bordersize;} // *MENU*
   virtual void     SetCanvasSize(UInt_t ww, UInt_t wh);
   virtual void     SetCursor(ECursor cursor);
   virtual void     SetDoubleBuffer(Int_t mode=1);
   virtual void     SetDrawOption(Option_t *option="");
   virtual void     SetEditable(Bool_t mode=kTRUE); // *TOGGLE*
   virtual void     SetGrid(Int_t valuex = 1, Int_t valuey = 1) {fGridx = valuex; fGridy = valuey;}
   virtual void     SetGridx(Int_t value = 1) {fGridx = value;} // *TOGGLE*
   virtual void     SetGridy(Int_t value = 1) {fGridy = value;} // *TOGGLE*
   virtual void     SetFillStyle(Style_t fstyle);
   virtual void     SetLogx(Int_t value = 1); // *TOGGLE*
   virtual void     SetLogy(Int_t value = 1); // *TOGGLE*
   virtual void     SetLogz(Int_t value = 1); // *TOGGLE*
   virtual void     SetNumber(Int_t number) {fNumber = number;}
   virtual void     SetPad(const char *name, const char *title,
                           Float_t xlow, Float_t ylow, Float_t xup,
                           Float_t yup, Color_t color=35,
                           Short_t bordersize=5, Short_t bordermode=-1);
   virtual void     SetPad(Float_t xlow, Float_t ylow, Float_t xup, Float_t yup);
   virtual void     SetAttFillPS(Color_t color, Style_t style);
   virtual void     SetAttLinePS(Color_t color, Style_t style, Width_t lwidth);
   virtual void     SetAttMarkerPS(Color_t color, Style_t style, Size_t msize);
   virtual void     SetAttTextPS(Int_t align, Float_t angle, Color_t color, Style_t font, Float_t tsize);
   virtual void     SetName(const char *name) {fName = name;} // *MENU*
   virtual void     SetSelected(TObject *obj);
   virtual void     SetTicks(Int_t valuex = 1, Int_t valuey = 1) {fTickx = valuex; fTicky = valuey;}
   virtual void     SetTickx(Int_t value = 1) {fTickx = value;} // *TOGGLE*
   virtual void     SetTicky(Int_t value = 1) {fTicky = value;} // *TOGGLE*
   virtual void     SetTitle(const char *title="") {fTitle = title;}
   virtual void     SetTheta(Float_t theta=30) {fTheta = theta;}
   virtual void     SetPhi(Float_t phi=30) {fPhi = phi;}
   virtual void     SetToolTipText(const char *text, Long_t delayms = 1000);
   virtual void     SetView(TView *view) {fView = view;}
   virtual void     Update();
   Int_t            UtoAbsPixel(Float_t u) const {return Int_t(fUtoAbsPixelk + u*fUtoPixel);}
   Int_t            VtoAbsPixel(Float_t v) const {return Int_t(fVtoAbsPixelk + v*fVtoPixel);}
   Int_t            UtoPixel(Float_t u) const;
   Int_t            VtoPixel(Float_t v) const;
   virtual TObject *WaitPrimitive(const char *pname="", const char *emode="");
   Int_t            XtoAbsPixel(Axis_t x) const {return Int_t(fXtoAbsPixelk + x*fXtoPixel);}
   Int_t            YtoAbsPixel(Axis_t y) const {return Int_t(fYtoAbsPixelk + y*fYtoPixel);}
   Float_t          XtoPad(Axis_t x) const;
   Float_t          YtoPad(Axis_t y) const;
   Int_t            XtoPixel(Axis_t x) const;
   Int_t            YtoPixel(Axis_t y) const;
   virtual void     XYtoAbsPixel(Axis_t x, Axis_t y, Int_t &xpixel, Int_t &ypixel) const;
   virtual void     XYtoPixel(Axis_t x, Axis_t y, Int_t &xpixel, Int_t &ypixel) const;

   virtual TObject   *CreateToolTip(const TBox *b, const char *text, Long_t delayms);
   virtual void       DeleteToolTip(TObject *tip);
   virtual void       ResetToolTip(TObject *tip);
   virtual void       CloseToolTip(TObject *tip);

   virtual void     x3d(Option_t *option=""); // *MENU*

   ClassDef(TPad,4)  //A Graphics pad
};


//---- inlines -----------------------------------------------------------------

//______________________________________________________________________________
inline void TPad::AbsPixeltoXY(Int_t xpixel, Int_t ypixel, Axis_t &x, Axis_t &y)
{
   x = AbsPixeltoX(xpixel);
   y = AbsPixeltoY(ypixel);
}

//______________________________________________________________________________
inline Float_t TPad::PixeltoX(Int_t px)
{
   if (fAbsCoord) return fAbsPixeltoXk + px*fPixeltoX;
   else           return fPixeltoXk    + px*fPixeltoX;
}

//______________________________________________________________________________
inline Float_t TPad::PixeltoY(Int_t py)
{
   if (fAbsCoord) return fAbsPixeltoYk + py*fPixeltoY;
   else           return fPixeltoYk    + py*fPixeltoY;
}

//______________________________________________________________________________
inline void TPad::PixeltoXY(Int_t xpixel, Int_t ypixel, Axis_t &x, Axis_t &y)
{
   x = PixeltoX(xpixel);
   y = PixeltoY(ypixel);
}

//______________________________________________________________________________
inline Int_t TPad::UtoPixel(Float_t u) const
{
   if (fAbsCoord) return Int_t(fUtoAbsPixelk + u*fUtoPixel);
   else           return Int_t(u*fUtoPixel);
}

//______________________________________________________________________________
inline Int_t TPad::VtoPixel(Float_t v) const
{
   if (fAbsCoord) return Int_t(fVtoAbsPixelk + v*fVtoPixel);
   else           return Int_t(fVtoPixelk    + v*fVtoPixel);
}

//______________________________________________________________________________
inline Int_t TPad::XtoPixel(Axis_t x) const
{
   if (fAbsCoord) return Int_t(fXtoAbsPixelk + x*fXtoPixel);
   else           return Int_t(fXtoPixelk    + x*fXtoPixel);
}

//______________________________________________________________________________
inline Int_t TPad::YtoPixel(Axis_t y) const
{
   if (fAbsCoord) return Int_t(fYtoAbsPixelk + y*fYtoPixel);
   else           return Int_t(fYtoPixelk    + y*fYtoPixel);
}

//______________________________________________________________________________
inline void TPad::XYtoAbsPixel(Axis_t x, Axis_t y, Int_t &xpixel, Int_t &ypixel) const
{
   xpixel = XtoAbsPixel(x);
   ypixel = YtoAbsPixel(y);
}

//______________________________________________________________________________
inline void TPad::XYtoPixel(Axis_t x, Axis_t y, Int_t &xpixel, Int_t &ypixel) const
{
   xpixel = XtoPixel(x);
   ypixel = YtoPixel(y);
}

//______________________________________________________________________________
inline void TPad::SetDrawOption(Option_t *)
{ }

#endif

