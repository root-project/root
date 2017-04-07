// @(#)root/gl:$Id$
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

#include "TVirtualGL.h"
#include "TGLPlotBox.h"
#include "TPoint.h"
#include "TGLUtil.h"
#include "TNamed.h"

class TGLPlotCoordinates;
class TGLPlotCamera;
class TGL5DDataSet;
class TString;
class TColor;
class TAxis;
class TH1;
class TH3;
class TF3;

/*
   Box cut. When attached to a plot, cuts away a part of it.
   Can be moved in a plot's own area in X/Y/Z directions.
*/

class TGLBoxCut {
private:
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

public:
   TGLBoxCut(const TGLPlotBox *plotBox);
   virtual ~TGLBoxCut();

   void   TurnOnOff();
   Bool_t IsActive()const{return fActive;}
   void   SetActive(Bool_t a);

   void   ResetBoxGeometry();

   void   SetFactor(Double_t f){fFactor = f;}

   void   DrawBox(Bool_t selectionPass, Int_t selected)const;

   void   StartMovement(Int_t px, Int_t py);
   void   MoveBox(Int_t px, Int_t py, Int_t axisID);

   Bool_t IsInCut(Double_t xMin, Double_t xMax, Double_t yMin, Double_t yMax,
                  Double_t zMin, Double_t zMax)const;

   template<class V>
   Bool_t IsInCut(const V * v) const
   {
      //Check, if box defined by xmin/xmax etc. is in cut.
      if (v[0] >= fXRange.first && v[0] < fXRange.second &&
          v[1] >= fYRange.first && v[1] < fYRange.second &&
          v[2] >= fZRange.first && v[2] < fZRange.second)
         return kTRUE;
      return kFALSE;
   }

   Rgl::Range_t GetXRange()const{return fXRange;}
   Rgl::Range_t GetYRange()const{return fYRange;}
   Rgl::Range_t GetZRange()const{return fZRange;}

private:
   void AdjustBox();

   ClassDef(TGLBoxCut, 0)//Cuts away part of a plot.
};

/*
   2D contour for TH3 slicing.
*/

class TGLTH3Slice : public TNamed {
public:
   enum ESliceAxis {kXOZ, kYOZ, kXOY};

private:
   ESliceAxis                fAxisType;
   const TAxis                    *fAxis;
   mutable TGLLevelPalette   fPalette;

   const TGLPlotCoordinates *fCoord;
   const TGLPlotBox         *fBox;
   Int_t                     fSliceWidth;

   const TH3                *fHist;
   const TF3                *fF3;

   mutable TGL2DArray<Double_t> fTexCoords;

   mutable Rgl::Range_t         fMinMax;

public:
   TGLTH3Slice(const TString &sliceName,
               const TH3 *hist,
               const TGLPlotCoordinates *coord,
               const TGLPlotBox * box,
               ESliceAxis axis);
   TGLTH3Slice(const TString &sliceName,
               const TH3 *hist, const TF3 *fun,
               const TGLPlotCoordinates *coord,
               const TGLPlotBox * box,
               ESliceAxis axis);

   void   DrawSlice(Double_t pos)const;
   //SetSliceWidth must have "menu" comment.
   void   SetSliceWidth(Int_t width = 1); // *MENU*

   void   SetMinMax(const Rgl::Range_t &newRange)
   {
      fMinMax = newRange;
   }

   const TGLLevelPalette & GetPalette()const
   {
      return fPalette;
   }

private:
   void   PrepareTexCoords(Double_t pos, Int_t sliceBegin, Int_t sliceEnd)const;
   void   FindMinMax(Int_t sliceBegin, Int_t sliceEnd)const;
   Bool_t PreparePalette()const;
   void   DrawSliceTextured(Double_t pos)const;
   void   DrawSliceFrame(Int_t low, Int_t up)const;

   ClassDef(TGLTH3Slice, 0) // TH3 slice
};


/*
   TGLPlotPainter class defines interface to different plot painters.
*/

class TGLPlotPainter;

/*
Object of this class, created on stack in DrawPlot member-functions,
saves modelview matrix, moves plot to (0; 0; 0), and
restores modelview matrix in dtor.
*/

namespace Rgl {

class PlotTranslation {
public:
   PlotTranslation(const TGLPlotPainter *painter);
   ~PlotTranslation();

private:
   const TGLPlotPainter *fPainter;
};

}

class TGLPlotPainter : public TVirtualGLPainter {
   friend class Rgl::PlotTranslation;
private:
   const TColor         *fPadColor;

protected:
   const Float_t        *fPhysicalShapeColor;

   Double_t              fPadPhi;
   Double_t              fPadTheta;
   TH1                  *fHist;
   TAxis                *fXAxis;
   TAxis                *fYAxis;
   TAxis                *fZAxis;

   TGLPlotCoordinates   *fCoord;
   TGLPlotCamera        *fCamera;
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
      kTrueColorSelectionBase = 10
   };

   Int_t                 fSelectionBase;
   mutable Bool_t        fDrawPalette;
   Bool_t                fDrawAxes;

public:
/*   TGLPlotPainter(TH1 *hist, TGLPlotCamera *camera, TGLPlotCoordinates *coord, Int_t context,
                  Bool_t xoySelectable, Bool_t xozSelectable, Bool_t yozSelectable);
   TGLPlotPainter(TGLPlotCamera *camera, Int_t context);*/
   TGLPlotPainter(TH1 *hist, TGLPlotCamera *camera, TGLPlotCoordinates *coord,
                  Bool_t xoySelectable, Bool_t xozSelectable, Bool_t yozSelectable);
   TGLPlotPainter(TGL5DDataSet *data, TGLPlotCamera *camera, TGLPlotCoordinates *coord);
   TGLPlotPainter(TGLPlotCamera *camera);

   const TGLPlotBox& RefBackBox() const { return fBackBox; }
   void              SetPhysicalShapeColor(const Float_t *rgba)
   {
      fPhysicalShapeColor = rgba;
   }

   virtual void     InitGL()const = 0;
   virtual void     DeInitGL()const = 0;
   virtual void     DrawPlot()const = 0;
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
   void             SetPadColor(const TColor *color);

   virtual void     SetFrameColor(const TColor *frameColor);
   //Camera is external to painter, if zoom was changed, or camera
   //was rotated, selection must be invalidated.
   void             InvalidateSelection();

   enum ECutAxisID {
      kXAxis = 7,
      kYAxis = 8,
      kZAxis = 9
   };

   Bool_t           CutAxisSelected()const{return !fHighColor && fSelectedPart <= kZAxis && fSelectedPart >= kXAxis;}

   void SetDrawFrontBox(Bool_t b) {fBackBox.SetDrawFront(b);}
   void SetDrawBackBox(Bool_t b) {fBackBox.SetDrawBack(b);}
   void SetDrawAxes(Bool_t s) {fDrawAxes = s;}
   Bool_t GetDrawAxes() {return fDrawAxes;}

protected:
   const TColor    *GetPadColor()const;
   //
   void             MoveSection(Int_t px, Int_t py);
   void             DrawSections()const;
   virtual void     DrawSectionXOZ()const = 0;
   virtual void     DrawSectionYOZ()const = 0;
   virtual void     DrawSectionXOY()const = 0;

   virtual void     DrawPaletteAxis()const;

   virtual void     ClearBuffers()const;

   void             PrintPlot()const;

   //Attention! After one of this methods was called,
   //the GL_MATRIX_MODE could become different from what
   //you had before the call: for example, SaveModelviewMatrix will
   //change it to GL_MODELVIEW.
   void             SaveModelviewMatrix()const;
   void             SaveProjectionMatrix()const;

   void             RestoreModelviewMatrix()const;
   void             RestoreProjectionMatrix()const;

   ClassDef(TGLPlotPainter, 0) //Base for gl plots
};

/*
   Auxiliary class, which holds different
   information about plot's current coordinate system
*/

class TH2Poly;

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
   //
   Bool_t SetRanges(const TH1 *hist, Bool_t errors = kFALSE, Bool_t zBins = kFALSE);
   //
   Bool_t SetRanges(TH2Poly *hist);
   //
   Bool_t SetRanges(const TAxis *xAxis, const TAxis *yAxis, const TAxis *zAxis);

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
   Bool_t SetRangesPolar(const TH1 *hist);
   Bool_t SetRangesCylindrical(const TH1 *hist);
   Bool_t SetRangesSpherical(const TH1 *hist);

   Bool_t SetRangesCartesian(const TH1 *hist, Bool_t errors = kFALSE, Bool_t zBins = kFALSE);

   TGLPlotCoordinates(const TGLPlotCoordinates &);
   TGLPlotCoordinates &operator = (const TGLPlotCoordinates &);

   ClassDef(TGLPlotCoordinates, 0)//Auxilary class, holds plot dimensions.
};

class TGLLevelPalette;

namespace Rgl {

void DrawPalette(const TGLPlotCamera *camera, const TGLLevelPalette &palette);
void DrawPalette(const TGLPlotCamera *camera, const TGLLevelPalette &palette,
                 const std::vector<Double_t> &levels);
void DrawPaletteAxis(const TGLPlotCamera *camera, const Range_t &minMax, Bool_t logZ);

//Polygonal histogram (TH2Poly) is slightly stretched along x and y.
extern const Double_t gH2PolyScaleXY;

}

#endif
