#ifndef ROOT_TGLTF3Painter
#define ROOT_TGLTF3Painter

#include <vector>
#include <list>

#ifndef ROOT_TGLPlotPainter
#include "TGLPlotPainter.h"
#endif
#ifndef ROOT_TGLUtil
#include "TGLUtil.h"
#endif

class TGLOrthoCamera;
class TF3;

class TGLTF3Painter : public TGLPlotPainter {
private:
   enum ETF3Style {
      kDefault,
      kMaple0,
      kMaple1,
      kMaple2
   };

   ETF3Style fStyle;

public:
   struct TriFace_t {
      TGLVertex3 fXYZ[3];
      TGLVector3 fNormals[3];
   };

private:
   std::vector<TriFace_t> fMesh;
   TF3 *fF3;

public:
   TGLTF3Painter(TF3 *fun, TH1 *hist, TGLOrthoCamera *camera, TGLPlotCoordinates *coord,
                 Int_t glContext = -1);
   
   char   *GetPlotInfo(Int_t px, Int_t py);
   Bool_t  InitGeometry();
   void    StartPan(Int_t px, Int_t py);
   void    Pan(Int_t px, Int_t py);
   void    AddOption(const TString &stringOption);
   void    ProcessEvent(Int_t event, Int_t px, Int_t py);

private:
   void    InitGL()const;
   void    DrawPlot()const;

   void    SetSurfaceColor()const;

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
public:
   //Triangle face of iso mesh.
   struct TriFace_t {
      TGLVertex3 fXYZ[3];
      TGLVector3 fNormal;//Flat normal.
      TGLVector3 fPerVertexNormals[3];//Smoothed normals for each vertex.
   };
   //Each of cubes (marching-cubes) has a
   //corresponding set of triangles in a mesh, possibly empty.
   //fFirst is the number of the first triangle,
   //fLast is the end of the range (excluded).
   struct Range_t {
      Range_t() : fFirst(-1), fLast(0)
      {
      }
      Range_t(Int_t f, Int_t l) : fFirst(f), fLast(l)
      {
      }
      Int_t fFirst;
      Int_t fLast;
   };
   struct Mesh_t {
      std::vector<Range_t>   fBoxRanges;
      std::vector<TriFace_t> fMesh;
      void Swap(Mesh_t &rhs)
      {
         std::swap(fBoxRanges, rhs.fBoxRanges);
         std::swap(fMesh,      rhs.fMesh);
      }
   };

private:
   TGLTH3Slice fXOZSlice;
   TGLTH3Slice fYOZSlice;
   TGLTH3Slice fXOYSlice;

   Mesh_t                                    fDummyMesh;
   
   typedef std::list<Mesh_t>                 MeshList_t;
   typedef std::list<Mesh_t>::iterator       MeshIter_t;
   typedef std::list<Mesh_t>::const_iterator ConstMeshIter_t;
   
   //List of meshes.
   MeshList_t                                fIsos;
   //Cheched meshes (will be used if geometry must be rebuilt
   //after TPad::PaintModified)
   MeshList_t                                fCache;
   //Min and max bin contents.
   Rgl::Range_t                              fMinMax;
   //Palette. One color per iso-surface.
   TGLLevelPalette                           fPalette;
   //Iso levels. Equidistant or user-defined.
   std::vector<Double_t>                     fColorLevels;

   //Now meshes are initialized only once.
   //To be changed in future.
   Bool_t                                    fInit;
   
public:
   TGLIsoPainter(TH1 *hist, TGLOrthoCamera *camera, TGLPlotCoordinates *coord,
                 Int_t glContext = -1);

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
   void     DrawPlot()const;
   void     DrawSectionXOZ()const;
   void     DrawSectionYOZ()const;
   void     DrawSectionXOY()const;
   //Auxiliary methods.
   Bool_t   HasSections()const;
   void     SetSurfaceColor(Int_t ind)const;
   void     SetMesh(Mesh_t &mesh, Double_t isoValue);
   void     DrawMesh(const Mesh_t &mesh, Int_t level)const;
   void     CheckBox(const std::vector<TriFace_t> &mesh, TriFace_t &face, const Range_t &box);
   void     FindMinMax();

   TGLIsoPainter(const TGLIsoPainter &);
   TGLIsoPainter &operator = (const TGLIsoPainter &);

   ClassDef(TGLIsoPainter, 0) //Iso option for TH3
};

#endif
