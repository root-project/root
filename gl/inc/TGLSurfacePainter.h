#ifndef ROOT_TGLSurfacePainter
#define ROOT_TGLSurfacePainter

#include <vector>
#include <list>

#ifndef ROOT_TGLPlotPainter
#include "TGLPlotPainter.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TGLUtil
#include "TGLUtil.h"
#endif

class TGLSurfacePainter : public TGLPlotPainter {
private:
   enum ESurfaceType {
      kSurf,
      kSurf1,
      kSurf2,
      kSurf3,
      kSurf4,
      kSurf5
   };

   mutable ESurfaceType fType;

   TGL2DArray<TGLVertex3>                         fMesh;
   mutable TGL2DArray<Double_t>                   fTexMap;
   TGL2DArray<std::pair<TGLVector3, TGLVector3> > fFaceNormals;
   TGL2DArray<TGLVector3>                         fAverageNormals;

   mutable TString                 fObjectInfo;

   struct Projection_t {
      UChar_t fRGBA[4];
      std::vector<TGLVertex3> fVertices;
      void Swap(Projection_t &rhs);
   };

   mutable Projection_t            fProj;

   mutable std::list<Projection_t> fXOZProj;
   mutable std::list<Projection_t> fYOZProj;
   mutable std::list<Projection_t> fXOYProj;

   mutable TGLLevelPalette         fPalette;
   mutable std::vector<Double_t>   fColorLevels;
   Rgl::Range_t                    fMinMaxVal;

   Bool_t                          fSectionPass;
   mutable Bool_t                  fUpdateTexMap;

public:
   TGLSurfacePainter(TH1 *hist, TGLOrthoCamera *camera, TGLPlotCoordinates *coord,
                     Int_t glContext = -1);

   //TGLPlotPainter's final-overriders.
   char  *GetPlotInfo(Int_t px, Int_t py);
   Bool_t InitGeometry();
   void   StartPan(Int_t px, Int_t py);
   void   Pan(Int_t px, Int_t py);
   void   AddOption(const TString &stringOption);
   void   ProcessEvent(Int_t event, Int_t px, Int_t py);

private:
   void   InitGL()const;
   void   DrawPlot()const;

   void   SetNormals();
   void   SetSurfaceColor()const;

   Bool_t InitGeometryCartesian();
   Bool_t InitGeometryPolar();
   Bool_t InitGeometryCylindrical();
   Bool_t InitGeometrySpherical();

   void   DrawProjections()const;
   void   DrawSectionXOZ()const;
   void   DrawSectionYOZ()const;
   void   DrawSectionXOY()const;

   void   ClampZ(Double_t &zVal)const;

   char  *WindowPointTo3DPoint(Int_t px, Int_t py)const;

   Bool_t PreparePalette()const;
   void   GenTexMap()const;
   void   DrawContoursProjection()const;

   Bool_t Textured()const;
   Bool_t HasSections()const;
   Bool_t HasProjections()const;

   ClassDef(TGLSurfacePainter, 0)//Surface painter.
};

#endif
