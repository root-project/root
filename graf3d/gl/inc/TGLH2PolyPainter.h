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

class TMultiGraph;
class TGraph;

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
   void         DrawExtrusion(const TGraph *polygon, Double_t zMin, Double_t zMax, Int_t nBin)const;
   void         DrawExtrusion(const TMultiGraph *polygon, Double_t zMin, Double_t zMax, Int_t nBin)const;

   //Draw caps for a bin.
   typedef std::list<Rgl::Pad::Tesselation_t>::const_iterator CIter_t;
   void         DrawCaps()const;
   void         DrawCap(CIter_t cap, Int_t bin)const;
   //
   Bool_t       CacheGeometry();
   Bool_t       BuildTesselation(Rgl::Pad::Tesselator & tess, const TGraph *g, Double_t z);
   Bool_t       BuildTesselation(Rgl::Pad::Tesselator & tess, const TMultiGraph *mg, Double_t z);
   Bool_t       UpdateGeometry();
   //Find the color in palette using bin content.
   void         SetBinColor(Int_t bin)const;

   //Empty overriders.
   void         DrawSectionXOZ()const;
   void         DrawSectionYOZ()const;
   void         DrawSectionXOY()const;
   void         DrawPalette()const;
   void         DrawPaletteAxis()const;

   //Aux. staff.
   void         FillTemporaryPolygon(const Double_t *xs, const Double_t *ys, Double_t z, Int_t n)const;
   void         MakePolygonCCW()const;
   Bool_t       ClampZ(Double_t &zVal)const;

   TString                            fBinInfo; //Used by GetPlotInfo.

   std::vector<Int_t>                 fBinColors;

   mutable std::vector<Double_t>      fPolygon; //Temporary array for polygon's vertices.
   std::list<Rgl::Pad::Tesselation_t> fCaps;//Caps for all bins.

   Bool_t                             fZLog;//Change in logZ updates only bin heights.
   Double_t                           fZMin;

   ClassDef(TGLH2PolyPainter, 0); //Painter class for TH2Poly.
};

#endif
