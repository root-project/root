#ifndef ROOT_TGLH2PolyPainter
#define ROOT_TGLH2PolyPainter

#include <vector>
#include <list>

#ifndef ROOT_TGLPlotPainter
#include "TGLPlotPainter.h"
#endif
#ifndef ROOT_TGLPadUtils
#include "TGLPadUtils.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class TGLH2PolyPainter : public TGLPlotPainter {
public:
   TGLH2PolyPainter(TH1 *hist, TGLPlotCamera *camera, TGLPlotCoordinates *coord);

   char        *GetPlotInfo(Int_t px, Int_t py);
   Bool_t       InitGeometry();
   void         StartPan(Int_t px, Int_t py);
   void         Pan(Int_t px, Int_t py);
   void         AddOption(const TString &stringOption);
   void         ProcessEvent(Int_t event, Int_t px, Int_t py);

private:
   //Overriders
   void         InitGL()const;
   void         DeInitGL()const;
   void         DrawPlot()const;
   //Aux. functions.
   //Draw edges of a bin.
   void         DrawExtrusion()const;
   //Draw caps for a bin.
   void         DrawCaps()const;
   //
   Bool_t       CacheGeometry();
   Bool_t       UpdateGeometry();
   //Find the color in palette using bin content.
   void         SetBinColor(Int_t bin)const;

   //Empty overriders.
   void         DrawSectionXOZ()const;
   void         DrawSectionYOZ()const;
   void         DrawSectionXOY()const;
   void         DrawPalette()const;
   void         DrawPaletteAxis()const;

   TString                            fBinInfo; //Used by GetPlotInfo.

   std::vector<Int_t>                 fBinColors;

   std::vector<Double_t>              fCap; //Temporary array for cap's vertices.
   std::list<Rgl::Pad::Tesselation_t> fCaps;//Caps for all bins.

   ClassDef(TGLH2PolyPainter, 0); //Painter class for TH2Poly.
};

#endif
