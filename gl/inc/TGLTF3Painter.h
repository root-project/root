#ifndef ROOT_TGLTF3Painter
#define ROOT_TGLTF3Painter

#include <vector>

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

#endif
