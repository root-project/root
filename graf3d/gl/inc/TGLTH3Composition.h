// @(#)root/gl:$Id$
// Author:  Timur Pocheptsov  07/08/2009

#ifndef ROOT_TGLTH3Composition
#define ROOT_TGLTH3Composition

#include <utility>
#include <memory>
#include <vector>

#include "TGLHistPainter.h"
#include "TGLPlotPainter.h"
#include "TGLQuadric.h"
#include "TH3.h"

//
//Composition of TH3 objects. All TH3 must have the same axis range
//and the same number of bins. If this condition is violated,
//AddTH3 will throw.
//IMPORTANT: TGLTH3Composition does not own TH3 objects
//it contains.
//This class inherits TH3 - to re-use TH3 editor.
//I use TH3C to reduce memory usage.
//Slising is not implemeted yet.
//

class TGLTH3Composition : public TH3C {
   friend class TGLTH3CompositionPainter;
public:
   TGLTH3Composition();//I need it only because of explicit private copy ctor.

   enum ETH3BinShape {
      kBox,
      kSphere
   };

   void AddTH3(const TH3 *hist, ETH3BinShape shape = kBox);

   //These are functions for TPad and
   //TPad's standard machinery (picking, painting).
   Int_t    DistancetoPrimitive(Int_t px, Int_t py) override;
   void     ExecuteEvent(Int_t event, Int_t px, Int_t py) override;
   char    *GetObjectInfo(Int_t px, Int_t py) const override;
   void     Paint(Option_t *option) override;

private:
   void CheckRanges(const TH3 *hist);

   typedef std::pair<const TH3 *, ETH3BinShape> TH3Pair_t;

   std::vector<TH3Pair_t>        fHists;
   std::unique_ptr<TGLHistPainter> fPainter;

   TGLTH3Composition(const TGLTH3Composition &) = delete;
   TGLTH3Composition &operator = (const TGLTH3Composition &) = delete;

   ClassDefOverride(TGLTH3Composition, 0)//Composition of TH3 objects.
};

//
//TGLTH3CompositionPainter class.
//
class TGLTH3CompositionPainter: public TGLPlotPainter {
public:
   TGLTH3CompositionPainter(TGLTH3Composition *data, TGLPlotCamera *camera,
                            TGLPlotCoordinates *coord);

   //TGLPlotPainter final-overriders.
   char      *GetPlotInfo(Int_t px, Int_t py) override;
   Bool_t     InitGeometry() override;
   void       StartPan(Int_t px, Int_t py) override;
   void       Pan(Int_t px, Int_t py) override;
   void       AddOption(const TString &option) override;
   void       ProcessEvent(Int_t event, Int_t px, Int_t py) override;

private:
   //TGLPlotPainter final-overriders.
   void       InitGL()const override;
   void       DeInitGL()const override;

   void       DrawPlot()const override;

   //Empty overriders.
   void       DrawSectionXOZ()const override {}
   void       DrawSectionYOZ()const override {}
   void       DrawSectionXOY()const override {}

   void       SetColor(Int_t color)const;

   TGLTH3Composition            *fData;
   std::pair<Double_t, Double_t> fMinMaxVal;

   mutable TGLQuadric            fQuadric;

   TGLTH3CompositionPainter(const TGLTH3CompositionPainter &) = delete;
   TGLTH3CompositionPainter &operator = (const TGLTH3CompositionPainter &)  = delete;

   ClassDefOverride(TGLTH3CompositionPainter, 0)//Painter to draw several TH3.
};


#endif
