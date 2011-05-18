// @(#)root/gl:$Id$
// Author:  Timur Pocheptsov  28/07/2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGL5DPainter
#define ROOT_TGL5DPainter

#include <vector>
#include <list>

#ifndef ROOT_TGLMarchingCubes
#include "TGLMarchingCubes.h"
#endif
#ifndef ROOT_TGLPlotPainter
#include "TGLPlotPainter.h"
#endif
#ifndef ROOT_TGLIsoMesh
#include "TGLIsoMesh.h"
#endif
#ifndef ROOT_TKDEFGT
#include "TKDEFGT.h"
#endif
#ifndef ROOT_TGLUtil
#include "TGLUtil.h"
#endif


class TGLPlotCamera;
class TGL5DDataSet;

//
//Painter to draw TGL5DDataSet ("gl5d" option for TTree).
//

class TGL5DPainter : public TGLPlotPainter {
public:
   enum EDefaults {
      kNContours = 4,
      kNLowPts   = 50
   };

   typedef Rgl::Mc::TIsoMesh<Float_t> Mesh_t;
   
   //Iso surface.
   struct Surf_t {
      Surf_t() 
         : f4D(0.), fRange(0.), fShowCloud(kFALSE), fHide(kFALSE), 
           fColor(0), fHighlight(kFALSE), fAlpha(100)
      {
      }
      
      Mesh_t                fMesh;     //Mesh.
      Double_t              f4D;       //Iso-level.
      Double_t              fRange;    //Selection critera (f4D +- fRange).
      Bool_t                fShowCloud;//Show/Hide original cloud.
      Bool_t                fHide;     //Show/Hide surface.
      Color_t               fColor;    //Color.
      std::vector<Double_t> fPreds;    //Predictions for 5-th variable.
      Bool_t                fHighlight;//If surface was selected via GUI - highlight it.
      Int_t                 fAlpha;    //Opacity percentage of a surface.
   };
   
   typedef std::list<Surf_t>          SurfList_t;
   typedef SurfList_t::iterator       SurfIter_t;
   typedef SurfList_t::const_iterator ConstSurfIter_t;

private:
   TKDEFGT                  fKDE;                        //Density estimator.
   Rgl::Mc::TMeshBuilder<TKDEFGT, Float_t> fMeshBuilder; //Mesh builder.

   const Surf_t             fDummy; //Empty surface (for effective insertion into list).
   Bool_t                   fInit;  //Geometry was set.
   
   SurfList_t               fIsos;  //List of iso-surfaces.
   TGL5DDataSet            *fData;  //Dataset to visualize.
   
   typedef std::vector<Double_t>::size_type size_type;

   Rgl::Range_t             fV5PredictedRange; //For future.
   Rgl::Range_t             fV5SliderRange;    //For future.
   Bool_t                   fShowSlider;       //For future.
   
   Double_t                 fAlpha;     //Parameter to define selection range.
   Int_t                    fNContours; //Number of "pre-defined" contours.
   
public:
   TGL5DPainter(TGL5DDataSet *data, TGLPlotCamera *camera, TGLPlotCoordinates *coord);

   //Add new iso for selected value of v4. +- range
   SurfIter_t AddSurface(Double_t v4, Color_t ci, Double_t isoVal = 1., Double_t sigma = 1., 
                         Double_t range = 1e-3, Int_t lowNumOfPoints = kNLowPts);

   void       AddSurface(Double_t v4);
   void       RemoveSurface(SurfIter_t surf);

   //TGLPlotPainter final-overriders.
   char      *GetPlotInfo(Int_t px, Int_t py);
   Bool_t     InitGeometry();
   void       StartPan(Int_t px, Int_t py);
   void       Pan(Int_t px, Int_t py);
   void       AddOption(const TString &option);
   void       ProcessEvent(Int_t event, Int_t px, Int_t py);

   //Methods for ged.
   void       ShowBoxCut(Bool_t show) {fBoxCut.SetActive(show);}
   Bool_t     IsBoxCutShown()const{return fBoxCut.IsActive();}
   
   void       SetAlpha(Double_t newAlpha);
   Double_t   GetAlpha()const{return fAlpha;}
   
   void       SetNContours(Int_t num);
   Int_t      GetNContours()const{return fNContours;}

   void       ResetGeometryRanges();
   
   SurfIter_t SurfacesBegin();
   SurfIter_t SurfacesEnd();

private:
   //TGLPlotPainter final-overriders.
   void       InitGL()const;
   void       DeInitGL()const;
   
   void       DrawPlot()const;
   
   //Empty overriders.
   void       DrawSectionXOZ()const{}
   void       DrawSectionYOZ()const{}
   void       DrawSectionXOY()const{}
   
   //Auxiliary functions.
   void       SetSurfaceColor(ConstSurfIter_t surf)const;
   void       DrawCloud()const;
   void       DrawSubCloud(Double_t v4, Double_t range, Color_t ci)const;
   void       DrawMesh(ConstSurfIter_t surf)const;
   
   TGL5DPainter(const TGL5DPainter &);
   TGL5DPainter &operator = (const TGL5DPainter &);
};

#endif
