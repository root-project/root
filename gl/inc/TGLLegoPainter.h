// @(#)root/gl:$Name:  $:$Id: TGLLegoPainter.h,v 1.4 2006/08/31 13:42:14 couet Exp $
// Author:  Timur Pocheptsov  14/06/2006
                                                                                
/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLLegoPainter
#define ROOT_TGLLegoPainter

#ifndef ROOT_TGLPlotPainter
#include "TGLPlotPainter.h"
#endif
#ifndef ROOT_TGLQuadric
#include "TGLQuadric.h"
#endif
#ifndef ROOT_TGLUtil
#include "TGLUtil.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class TGLOrthoCamera;
class TAxis;
class TH1;

/*
   TGLLegoPainter. The concrete implementation of abstract TGLPlotPainter.
*/

class TGLLegoPainter : public TGLPlotPainter {
private:
   enum ELegoType {
      kColorSimple,
      kColorLevel,
      kCylindricBars
   };
   //Bars, cylinders or textured bars.
   mutable ELegoType     fLegoType;
   Double_t              fMinZ;
   Rgl::Range_t          fMinMaxVal;//For texture coordinates generation.

   std::vector<Rgl::Range_t>  fXEdges;
   std::vector<Rgl::Range_t>  fYEdges;

   typedef std::pair<Double_t, Double_t> CosSin_t;
   std::vector<CosSin_t> fCosSinTableX;
   std::vector<CosSin_t> fCosSinTableY;
   TString               fBinInfo;
   mutable TGLQuadric    fQuadric;
   Bool_t                fDrawErrors;

   mutable TGLLevelPalette       fPalette;
   mutable std::vector<Double_t> fColorLevels;

   TGLLegoPainter(const TGLLegoPainter &);
   TGLLegoPainter &operator = (const TGLLegoPainter &);

public:
   TGLLegoPainter(TH1 *hist, TGLOrthoCamera *camera, TGLPlotCoordinates *coord, Int_t ctx = -1);

   //TGLPlotPainter's final-overriders
   char        *GetPlotInfo(Int_t px, Int_t py);
   Bool_t       InitGeometry();
   void         StartPan(Int_t px, Int_t py);
   void         Pan(Int_t px, Int_t py);
   void         AddOption(const TString &stringOption);
   void         ProcessEvent(Int_t event, Int_t px, Int_t py);

private:
   //Auxilary functions.
   Bool_t       InitGeometryCartesian();
   Bool_t       InitGeometryPolar();
   Bool_t       InitGeometryCylindrical();
   Bool_t       InitGeometrySpherical();
   //Overriders
   void         InitGL()const;
   void         DrawPlot()const;
   void         ClearBuffers()const;

   void         DrawLegoCartesian()const;
   void         DrawLegoPolar()const;
   void         DrawLegoCylindrical()const;
   void         DrawLegoSpherical()const;

   void         SetLegoColor()const;

   void         DrawSectionXOZ()const;
   void         DrawSectionYOZ()const;
   void         DrawSectionXOY()const;

   Bool_t       ClampZ(Double_t &zVal)const;
   Bool_t       PreparePalette()const;

   ClassDef(TGLLegoPainter, 0)//Lego painter
};

#endif
