// @(#)root/gl:$Id$
// Author: Timur Pocheptsov  2009
/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGL5D
#define ROOT_TGL5D

#include <utility>
#include <vector>
#include <list>

#ifndef ROOT_TGLPlotPainter
#include "TGLPlotPainter.h"
#endif
#ifndef ROOT_TGLIsoMesh
#include "TGLIsoMesh.h"
#endif
#ifndef ROOT_TGLUtil
#include "TGLUtil.h"
#endif
#ifndef ROOT_KDEFGT
#include "TKDEFGT.h"
#endif

class TGLPlotCamera;
class TH3F;

//
//This class paints set of iso-surfaces. xyz - range must be specified
//via TH3F, data (5 vectors) must be set via SetDataSources.
//

class TGL5D : public TGLPlotPainter {
public:
   typedef Rgl::Mc::TIsoMesh<Float_t>        Mesh_t;
   
   //Iso surfaces.
   struct Surf_t {
      Surf_t() : f4D(0.), fShowCloud(kFALSE), fHide(kFALSE), fColor(0)
      {
      }
      
      Mesh_t                fMesh;     //Mesh.
      Double_t              f4D;       //Iso-level.
      Double_t              fRange;    //Selection critera (f4D +- fRange).
      Bool_t                fShowCloud;//Show/Hide original cloud.
      Bool_t                fHide;     //Show/Hide surface.
      Color_t               fColor;    //Color.
      std::vector<Double_t> fPreds;    //Predictions for 5-th variable.
   };
   
   typedef std::list<Surf_t> SurfList_t;
   typedef SurfList_t::iterator SurfIter_t;
   typedef SurfList_t::const_iterator ConstSurfIter_t;

private:
   //Density estimator.
   TKDEFGT                  fKDE;  //Density estimator.
   
   const Surf_t            fDummy; //Empty surface.
   Bool_t                  fInit;  //Geometry was set.
   
   SurfList_t              fIsos;  //List of iso-surfaces.
   
   //Input data.
   Int_t                   fNP;//Size of input.
   const Double_t         *fV1;//V1.
   const Double_t         *fV2;//V2.
   const Double_t         *fV3;//V3.
   const Double_t         *fV4;//V4.
   const Double_t         *fV5;//V5.
   
   typedef std::vector<Double_t>::size_type size_type;
   
   Rgl::Range_t fV1MinMax;//V1 range.
   Rgl::Range_t fV2MinMax;//V2 range.
   Rgl::Range_t fV3MinMax;//V3 range.
   Rgl::Range_t fV4MinMax;//V4 range.
   Rgl::Range_t fV5MinMax;//V5 range.
   //
   std::vector<Double_t>   fTS;  //targets.
   std::vector<Double_t>   fDens;//densities in target points.
   //
   std::vector<Double_t>   fPtsSorted;//Packed xyz coordinates for cloud.

   //Double_t fV5SliderMin, fV5SliderMax;
   Bool_t                 fShowSlider;//For future.
public:
   TGL5D(TH3F *hist, TGLPlotCamera *camera, TGLPlotCoordinates *coord);
   
   //Interface to manipulate TGL5D object.
   void SetDataSources(Int_t size, const Double_t *v1, const Double_t *v2, const Double_t *v3,
                       const Double_t *v4, const Double_t *v5);
   
   const Rgl::Range_t &GetV1Range()const;
   const Rgl::Range_t &GetV2Range()const;
   const Rgl::Range_t &GetV3Range()const;
   const Rgl::Range_t &GetV4Range()const;
   const Rgl::Range_t &GetV5Range()const;
   
   //void       SetV5SliderMin(Double_t min);
   //Double_t   GetV5SliderMin() const {return fV5SliderMin;}
   
   //void       SetV5SliderMax(Double_t max);
   //Double_t   GetV5SliderMax() const {return fV5SliderMax;}
   
   Bool_t     ShowSlider() const {return fShowSlider;}
   void       ShowSlider(Bool_t show) {fShowSlider = show;}
   //Add new iso for selected value of v4. +- range
   SurfIter_t AddSurface(Double_t v4, Color_t ci, Double_t isoVal = 1., Double_t sigma = 1., 
                         Double_t e = 10., Double_t range = 1e-3, Int_t lowNumOfPoints = 100);
   void       SetSurfaceMode(SurfIter_t surf, Bool_t cloudOn);
   void       SetSurfaceColor(SurfIter_t surf, Color_t colorIndex);
   void       HideSurface(SurfIter_t surf);
   void       ShowSurface(SurfIter_t surf);
   void       RemoveSurface(SurfIter_t surf);
   
   //TGLPlotPainter final-overriders.
   char    *GetPlotInfo(Int_t px, Int_t py);
   Bool_t   InitGeometry();
   void     StartPan(Int_t px, Int_t py);
   void     Pan(Int_t px, Int_t py);
   void     AddOption(const TString &option);
   void     ProcessEvent(Int_t event, Int_t px, Int_t py);

private:
   //TGLPlotPainter final-overriders.
   void     InitGL()const;
   void     DeInitGL()const;
   
   void     DrawPlot()const;
   
   //Empty overriders.
   void     DrawSectionXOZ()const{}
   void     DrawSectionYOZ()const{}
   void     DrawSectionXOY()const{}
   
   //Auxiliary functions.
   void     SetSurfaceColor(Color_t index)const;
   void     DrawCloud()const;
   void     DrawSubCloud(Double_t v4, Double_t range, Color_t ci)const;
   void     DrawMesh(ConstSurfIter_t surf)const;
   
   TGL5D(const TGL5D &);
   TGL5D &operator = (const TGL5D &);

   ClassDef(TGL5D, 0)//Class to paint set of iso surfaces.
};

#endif
