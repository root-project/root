#ifndef ROOT_TGLVoxelPainter
#define ROOT_TGLVoxelPainter

#include <vector>

#include "TGLPlotPainter.h"
#include "TGLQuadric.h"
#include "TString.h"
#include "TGLUtil.h"

class TGLOrthoCamera;
class TH1;
class TF1;

class TGLVoxelPainter : public TGLPlotPainter {
private:

   TString                 fPlotInfo;
   Rgl::Range_t            fMinMaxVal;

   TGLVoxelPainter(const TGLVoxelPainter &) = delete;
   TGLVoxelPainter &operator = (const TGLVoxelPainter &) = delete;

   mutable TGLLevelPalette fPalette;
   mutable std::vector<Double_t> fLevels;

public:
   TGLVoxelPainter(TH1 *hist, TGLPlotCamera *camera, TGLPlotCoordinates *coord);

   char   *GetPlotInfo(Int_t px, Int_t py) override;
   Bool_t  InitGeometry() override;
   void    StartPan(Int_t px, Int_t py) override;
   void    Pan(Int_t px, Int_t py) override;
   void    AddOption(const TString &stringOption) override;
   void    ProcessEvent(Int_t event, Int_t px, Int_t py) override;

private:
   //Overriders
   void    InitGL()const override;
   void    DeInitGL()const override;

   void    DrawPlot()const override;


   void    DrawSectionXOZ()const override;
   void    DrawSectionYOZ()const override;
   void    DrawSectionXOY()const override;

   void    DrawPalette()const;
   void    DrawPaletteAxis()const override;

   //Aux. functions.
   void    FindVoxelColor(Double_t binContent, Float_t *rgba)const;
   void    SetVoxelColor(const Float_t *rgba)const;

   Bool_t  HasSections()const;
   void    PreparePalette()const;

   TF1    *fTransferFunc;

   ClassDefOverride(TGLVoxelPainter, 0)//Voxel painter
};

#endif
