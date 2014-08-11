// @(#)root/gl:$Id$
// Author:  Timur Pocheptsov  31/08/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLTF3Painter
#define ROOT_TGLTF3Painter

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

class TGLPlotCamera;
class TF3;

/*
Draw TF3 using marching cubes.
*/

class TGLTF3Painter : public TGLPlotPainter {
private:
   enum ETF3Style {
      kDefault,
      kMaple0,
      kMaple1,
      kMaple2
   };

   ETF3Style fStyle;

   Rgl::Mc::TIsoMesh<Double_t> fMesh;
   TF3 *fF3;

   TGLTH3Slice fXOZSlice;
   TGLTH3Slice fYOZSlice;
   TGLTH3Slice fXOYSlice;

public:
   TGLTF3Painter(TF3 *fun, TH1 *hist, TGLPlotCamera *camera, TGLPlotCoordinates *coord);

   char   *GetPlotInfo(Int_t px, Int_t py);
   Bool_t  InitGeometry();
   void    StartPan(Int_t px, Int_t py);
   void    Pan(Int_t px, Int_t py);
   void    AddOption(const TString &stringOption);
   void    ProcessEvent(Int_t event, Int_t px, Int_t py);

private:
   void    InitGL()const;
   void    DeInitGL()const;

   void    DrawPlot()const;
   //
   void    DrawToSelectionBuffer()const;
   void    DrawDefaultPlot()const;
   void    DrawMaplePlot()const;
   //

   void    SetSurfaceColor()const;
   Bool_t  HasSections()const;

   void    DrawSectionXOZ()const;
   void    DrawSectionYOZ()const;
   void    DrawSectionXOY()const;

   ClassDef(TGLTF3Painter, 0) // GL TF3 painter.
};

/*
   Iso painter draws iso surfaces - "gliso" option for TH3XXX::Draw.
   Can be one-level iso (as standard non-gl "iso" option),
   or multi-level iso: equidistant contours (if you only specify
   number of contours, or at user defined levels).
*/

class TGLIsoPainter : public TGLPlotPainter {
private:
   typedef Rgl::Mc::TIsoMesh<Float_t>        Mesh_t;
   typedef std::list<Mesh_t>                 MeshList_t;
   typedef std::list<Mesh_t>::iterator       MeshIter_t;
   typedef std::list<Mesh_t>::const_iterator ConstMeshIter_t;

   TGLTH3Slice           fXOZSlice;
   TGLTH3Slice           fYOZSlice;
   TGLTH3Slice           fXOYSlice;

   Mesh_t                fDummyMesh;
   //List of meshes.
   MeshList_t            fIsos;
   //Cached meshes (will be used if geometry must be rebuilt
   //after TPad::PaintModified)
   MeshList_t            fCache;
   //Min and max bin contents.
   Rgl::Range_t          fMinMax;
   //Palette. One color per iso-surface.
   TGLLevelPalette       fPalette;
   //Iso levels. Equidistant or user-defined.
   std::vector<Double_t> fColorLevels;
   //Now meshes are initialized only once.
   //To be changed in future.
   Bool_t                fInit;

public:
   TGLIsoPainter(TH1 *hist, TGLPlotCamera *camera, TGLPlotCoordinates *coord);

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
   void     DrawSectionXOZ()const;
   void     DrawSectionYOZ()const;
   void     DrawSectionXOY()const;
   //Auxiliary methods.
   Bool_t   HasSections()const;
   void     SetSurfaceColor(Int_t ind)const;
   void     SetMesh(Mesh_t &mesh, Double_t isoValue);
   void     DrawMesh(const Mesh_t &mesh, Int_t level)const;
   void     FindMinMax();

   TGLIsoPainter(const TGLIsoPainter &);
   TGLIsoPainter &operator = (const TGLIsoPainter &);

   ClassDef(TGLIsoPainter, 0) // Iso option for TH3.
};

#endif
