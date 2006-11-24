// @(#)root/gl:$Name:  $:$Id: TGLPlotPainter.h,v 1.8 2006/11/22 16:19:55 couet Exp $
// Author:  Timur Pocheptsov  14/06/2006
                                                                                
/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLPlotPainter
#define ROOT_TGLPlotPainter

#include <vector>

#ifndef ROOT_TVirtualGL
#include "TVirtualGL.h"
#endif
#ifndef ROOT_TGLPlotBox
#include "TGLPlotBox.h"
#endif
#ifndef ROOT_TPoint
#include "TPoint.h"
#endif
#ifndef ROOT_TGLUtil
#include "TGLUtil.h"
#endif

class TGLPlotCoordinates;
class TGLOrthoCamera;
class TString;
class TColor;
class TAxis;
class TH1;

/*
   Box cut. When attached to a plot, cuts away a part of it.
   Can be moved in a plot's own area in X/Y/Z directions.
*/

class TGLBoxCut {
private:
   enum EMovementDirection {
      kAlongX,
      kAlongY,
      kAlongZ
   };

   EMovementDirection         fDirection;
   Double_t                   fXLength;
   Double_t                   fYLength;
   Double_t                   fZLength;
   TGLVertex3                 fCenter;
   Rgl::Range_t               fXRange;
   Rgl::Range_t               fYRange;
   Rgl::Range_t               fZRange;

   const TGLPlotBox * const   fPlotBox;
   Bool_t                     fActive;
   Double_t                   fFactor;

   TPoint                     fMousePos;
   enum{kSelectionColor = 7};

public:
   TGLBoxCut(const TGLPlotBox *plotBox);
   //Class has ClassDef macro == virtual functions, so we need 
   //explicitly declared/defined dtor to supress warnings from g++
   virtual ~TGLBoxCut();

   void   TurnOnOff();
   Bool_t IsActive()const{return fActive;}
   void   SetFactor(Double_t f){fFactor = f;}

   void   DrawBox(Bool_t selectionPass, Int_t selected)const;
   void   SetDirectionX();
   void   SetDirectionY();
   void   SetDirectionZ();
   
   void   StartMovement(Int_t px, Int_t py);
   void   MoveBox(Int_t px, Int_t py);
   
   Bool_t IsInCut(Double_t xMin, Double_t xMax, Double_t yMin, Double_t yMax,
                  Double_t zMin, Double_t zMax)const;
private:
   void AdjustBox();

   ClassDef(TGLBoxCut, 0)//Cuts away part of a plot.
};

/*
   TGLPlotPainter class defines interface to different plot painters.
*/

class TGLPlotPainter : public TVirtualGLPainter {
private:
   Int_t                 fGLContext;
   const TColor         *fPadColor;

protected:
   TH1                  *fHist;
   TAxis                *fXAxis;
   TAxis                *fYAxis;
   TAxis                *fZAxis;

   TGLPlotCoordinates   *fCoord;
   TGLOrthoCamera       *fCamera;
   TGLSelectionBuffer    fSelection;
   Bool_t                fUpdateSelection;
   Bool_t                fSelectionPass;
   Int_t                 fSelectedPart;
   TPoint                fMousePosition;
   mutable Double_t      fXOZSectionPos;
   mutable Double_t      fYOZSectionPos;
   mutable Double_t      fXOYSectionPos;
   TGLPlotBox            fBackBox;
   TGLBoxCut             fBoxCut;

   std::vector<Double_t> fZLevels;
   Bool_t                fHighColor;
   
   enum ESelectionBase{
      kHighColorSelectionBase = 7,
      kTrueColorSelectionBase = 8
   };
   
   ESelectionBase        fSelectionBase;

public:
   TGLPlotPainter(TH1 *hist, TGLOrthoCamera *camera, TGLPlotCoordinates *coord, Int_t context,
                  Bool_t xoySelectable, Bool_t xozSelectable, Bool_t yozSelectable);

   virtual void     Paint();
   //Checks, if mouse cursor is above plot.
   virtual Bool_t   PlotSelected(Int_t px, Int_t py);
   //Init geometry does plot's specific initialization.
   virtual Bool_t   InitGeometry() = 0;

   virtual void     StartPan(Int_t px, Int_t py) = 0;
   //Pan function is already declared in TVirtualGLPainter.

   //Add string option, it can be a digit in "lego" or "surf".
   virtual void     AddOption(const TString &stringOption) = 0;
   //Function to process additional events (key presses, mouse clicks.)
   virtual void     ProcessEvent(Int_t event, Int_t px, Int_t py) = 0;
   //Used by GLpad
   void             SetGLContext(Int_t context);
   void             SetPadColor(const TColor *color);
   virtual void     SetFrameColor(const TColor *frameColor);
   //Camera is external to painter, if zoom was changed, or camera
   //was rotated, selection must be invalidated.
   void             InvalidateSelection();

protected:
   Int_t            GetGLContext()const;
   const TColor    *GetPadColor()const;
   Bool_t           MakeGLContextCurrent()const;
   //
   void             MoveSection(Int_t px, Int_t py);
   void             DrawSections()const;
   virtual void     DrawSectionXOZ()const = 0;
   virtual void     DrawSectionYOZ()const = 0;
   virtual void     DrawSectionXOY()const = 0;

   virtual void     InitGL()const = 0;
   virtual void     ClearBuffers()const = 0;
   virtual void     DrawPlot()const = 0;

   void             PrintPlot()const;
   //

   ClassDef(TGLPlotPainter, 0) //Base for gl plots
};

/*
   Auxiliary class, which holds different
   information about plot's current coordinate system
*/

class TGLPlotCoordinates {
private:
   EGLCoordType    fCoordType;

   Rgl::BinRange_t fXBins;
   Rgl::BinRange_t fYBins;
   Rgl::BinRange_t fZBins;

   Double_t        fXScale;
   Double_t        fYScale;
   Double_t        fZScale;

   Rgl::Range_t    fXRange;
   Rgl::Range_t    fYRange;
   Rgl::Range_t    fZRange;

   Rgl::Range_t    fXRangeScaled;
   Rgl::Range_t    fYRangeScaled;
   Rgl::Range_t    fZRangeScaled;

   Bool_t          fXLog;
   Bool_t          fYLog;
   Bool_t          fZLog;

   Bool_t          fModified;
   Double_t        fFactor;

public:
   TGLPlotCoordinates();
   virtual ~TGLPlotCoordinates();

   void         SetCoordType(EGLCoordType type);
   EGLCoordType GetCoordType()const;

   void   SetXLog(Bool_t xLog);
   Bool_t GetXLog()const;

   void   SetYLog(Bool_t yLog);
   Bool_t GetYLog()const;

   void   SetZLog(Bool_t zLog);
   Bool_t GetZLog()const;

   void   ResetModified();
   Bool_t Modified()const;

   Bool_t SetRanges(const TH1 *hist, Bool_t errors = kFALSE, Bool_t zBins = kFALSE);

   Int_t  GetNXBins()const;
   Int_t  GetNYBins()const;
   Int_t  GetNZBins()const;

   const Rgl::BinRange_t &GetXBins()const;
   const Rgl::BinRange_t &GetYBins()const;
   const Rgl::BinRange_t &GetZBins()const;

   const Rgl::Range_t    &GetXRange()const;
   Double_t               GetXLength()const;
   const Rgl::Range_t    &GetYRange()const;
   Double_t               GetYLength()const;
   const Rgl::Range_t    &GetZRange()const;
   Double_t               GetZLength()const;

   const Rgl::Range_t    &GetXRangeScaled()const;
   const Rgl::Range_t    &GetYRangeScaled()const;
   const Rgl::Range_t    &GetZRangeScaled()const;

   Double_t GetXScale()const{return fXScale;}
   Double_t GetYScale()const{return fYScale;}
   Double_t GetZScale()const{return fZScale;}

   Int_t    GetFirstXBin()const{return fXBins.first;}
   Int_t    GetLastXBin()const{return fXBins.second;}
   Int_t    GetFirstYBin()const{return fYBins.first;}
   Int_t    GetLastYBin()const{return fYBins.second;}
   Int_t    GetFirstZBin()const{return fZBins.first;}
   Int_t    GetLastZBin()const{return fZBins.second;}

   Double_t GetFactor()const;

private:
   Bool_t SetRangesCartesian(const TH1 *hist, Bool_t errors, Bool_t zBins);
   Bool_t SetRangesPolar(const TH1 *hist);
   Bool_t SetRangesCylindrical(const TH1 *hist);
   Bool_t SetRangesSpherical(const TH1 *hist);

   TGLPlotCoordinates(const TGLPlotCoordinates &);
   TGLPlotCoordinates &operator = (const TGLPlotCoordinates &);

   ClassDef(TGLPlotCoordinates, 0)//Auxilary class, holds plot dimensions.
};

#endif
