// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLAxisPainter
#define ROOT_TGLAxisPainter

#include "TAttAxis.h"
#include "TGLUtil.h"
#include "TString.h"

class TGLRnrCtx;
class TGLFont;

class TGLAxisAttrib: public TAttAxis
{
   friend class TGLAxisPainter;
public:
   enum ETextAlign_e { kCenterDown, kCenterUp, kLeft, kRight };

protected:
   TGLVector3   fDir;
   Double_t     fMin;
   Double_t     fMax;

   Float_t      fTMScale[3];
   TGLVector3   fTMOff[3];
   Int_t        fTMNDim;

   ETextAlign_e   fTextAlign;

   Bool_t       fRelativeFontSize;
   Int_t        fAbsLabelFontSize;
   Int_t        fAbsTitleFontSize;

   TString      fLabelFontName;
   TString      fTitleFontName;

   TString      fTitle;
   TGLVector3   fTitlePos;

public:
   TGLAxisAttrib();
   virtual ~TGLAxisAttrib(){}
  
   // Getters && Setters

   TGLVector3&  RefDir() { return fDir; }
   void SetRng(Double_t min, Double_t max) { fMin=min; fMax=max;}
   void GetRng(Double_t &min, Double_t &max) {min=fMin; max=fMax;}

   TGLVector3&  RefTMOff(Int_t i) { return fTMOff[i]; }
   void SetTMNDim(Int_t i) {fTMNDim=i;}
   Int_t GetTMNDim() {return fTMNDim;}

   void SetTextAlign(ETextAlign_e a) {fTextAlign=a;}
   ETextAlign_e GetTextAlign() const { return fTextAlign;}

   void SetRelativeFontSize(Bool_t x) { fRelativeFontSize=x; }
   Bool_t GetRelativeFontSize() const {return fRelativeFontSize;}

   void  SetAbsLabelFontSize(Int_t fs) {fAbsLabelFontSize=fs;}
   Int_t GetAbsLabelFontSize()const {return fAbsLabelFontSize;}
   void  SetAbsTitleFontSize(Int_t fs) {fAbsTitleFontSize=fs;}
   Int_t GetAbsTitleFontSize() const {return fAbsTitleFontSize;}

   void SetLabelFontName(const char* name) { fLabelFontName = name; }
   const char*  GetLabelFontName() const {return fLabelFontName.Data();}
   void SetTitleFontName(const char* name) { fTitleFontName = name; }
   const char*  GetTitleFontName() const {return fTitleFontName.Data();}

   void SetTitle(const char* title) {fTitle = title;}
   const char* GetTitle() const {return fTitle.Data();}
   TGLVector3& RefTitlePos() {return fTitlePos;}

   // override TAttAxis function
   virtual void	SetNdivisions(Int_t n, Bool_t /*optim*/=kTRUE) { fNdivisions =n; }

   ClassDef(TGLAxisAttrib, 0); // GL axis attributes.
};

/**************************************************************************/

class TGLAxisPainter
{
private:
   TGLAxisPainter(const TGLAxisPainter&);            // Not implemented
   TGLAxisPainter& operator=(const TGLAxisPainter&); // Not implemented

   void RnrText(const char* txt, TGLVector3 pos, TGLFont &font) const;
   void DrawTick(TGLVector3 &tv, Int_t order) const;
   const char* FormAxisValue(Float_t x) const;

   TGLAxisAttrib* fAtt;

public:
   TGLAxisPainter():fAtt(0) {}
   virtual ~TGLAxisPainter() {}

   void Paint(TGLRnrCtx& ctx, TGLAxisAttrib &atrib);

   ClassDef(TGLAxisPainter, 0); // GL axis painter.
};

#endif
