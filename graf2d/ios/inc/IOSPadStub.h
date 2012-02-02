// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov, 14/8/2011

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_IOSPadStub
#define ROOT_IOSPadStub

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// IOSPadStub                                                           //
//                                                                      //
// TVirtualPad interface is huge: ~150 public virtual member functions, //
// mixture of different interfaces in fact.                             //
// We do not need these 68 functions, but I have to implement them      //
// (they are pure virtual), so, we have a "stub" class with             //
// empty virtual functions.                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TVirtualPad
#include "TVirtualPad.h"
#endif

namespace ROOT {
namespace iOS {

class PadStub : public TVirtualPad {
public:
   TLegend *BuildLegend(Double_t x1, Double_t y1, Double_t x2, Double_t y2, const char *title);
   void Close(Option_t *);
   void AddExec(const char *, const char *);
   void CopyPixmap();
   void CopyPixmaps();
   void DeleteExec(const char *);
   void Divide(Int_t, Int_t, Float_t, Float_t, Int_t);
   void Draw(Option_t *);
   void DrawClassObject(const TObject *, Option_t *);
   void SetBatch(Bool_t batch);
   Int_t GetCanvasID() const;
   TCanvasImp *GetCanvasImp() const;
   Int_t GetEvent() const;
   Int_t GetEventX() const;
   Int_t GetEventY() const;
   Int_t GetNumber() const;
   TVirtualPad *GetPad(Int_t subpadnumber) const;
   TObject *GetPadPointer() const;
   TVirtualPad *GetPadSave() const;
   TVirtualPad *GetSelectedPad() const;
   TObject *GetView3D() const;
   void ResetView3D(TObject *);
   TCanvas *GetCanvas() const;
   TVirtualPad *GetVirtCanvas() const;
   Int_t GetPadPaint() const;
   Int_t GetPixmapID() const;
   Bool_t HasCrosshair() const;
   void SetCrosshair(Int_t);
   void SetAttFillPS(Color_t, Style_t);
   void SetAttLinePS(Color_t, Style_t, Width_t);
   void SetAttMarkerPS(Color_t, Style_t, Size_t);
   void SetAttTextPS(Int_t, Float_t, Color_t, Style_t, Float_t);
   void PaintBorderPS(Double_t, Double_t, Double_t, Double_t, Int_t, Int_t, Int_t, Int_t);
   Int_t GetGLDevice();
   void SetCopyGLDevice(Bool_t);
   void Pop();
   void Print(const char *) const;
   void Print(const char *, Option_t *);
   TVirtualPad *GetMother() const;
   TObject *CreateToolTip(const TBox *, const char *, Long_t);
   void DeleteToolTip(TObject *);
   void ResetToolTip(TObject *);
   void CloseToolTip(TObject *);
   void SetToolTipText(const char *, Long_t);
   void HighLight(Color_t, Bool_t);
   Color_t GetHighLightColor() const;
   void ls(Option_t *option) const;
   void Modified(Bool_t flag);
   Bool_t OpaqueMoving() const;
   Bool_t OpaqueResizing() const;
   void PaintModified();
   void RecursiveRemove(TObject *obj);
   void SaveAs(const char *,Option_t *) const;
   void SetCanvas(TCanvas *);
   void SetCanvasSize(UInt_t, UInt_t);
   void SetCursor(ECursor cursor);
   void SetDoubleBuffer(Int_t mode);
   void SetName(const char *);
   void SetTitle(const char *);
   void SetSelected(TObject *);
   void Update();
   TObject *WaitPrimitive(const char *, const char *);
   void ReleaseViewer3D(Option_t *);
   Bool_t HasViewer3D() const;
};

}//namespace iOS
}//namespace ROOT

#endif
