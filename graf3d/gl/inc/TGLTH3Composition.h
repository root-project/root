// @(#)root/gl:$Id$
// Author:  Timur Pocheptsov  07/08/2009

#ifndef ROOT_TGLTH3Composition
#define ROOT_TGLTH3Composition

#include <utility>
#include <memory>
#include <vector>

#ifndef ROOT_TGLHistPainter
#include "TGLHistPainter.h"
#endif
#ifndef ROOT_TGLPlotPainter
#include "TGLPlotPainter.h"
#endif
#ifndef ROOT_TGLQuadric
#include "TGLQuadric.h"
#endif
#ifndef ROOT_TH3
#include "TH3.h"
#endif

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
   Int_t    DistancetoPrimitive(Int_t px, Int_t py);
   void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
   char    *GetObjectInfo(Int_t px, Int_t py) const;
   void     Paint(Option_t *option);

private:
   void CheckRanges(const TH3 *hist);

   typedef std::pair<const TH3 *, ETH3BinShape> TH3Pair_t;

   std::vector<TH3Pair_t>        fHists;
   std::auto_ptr<TGLHistPainter> fPainter;

   TGLTH3Composition(const TGLTH3Composition &rhs);
   TGLTH3Composition &operator = (const TGLTH3Composition &);

   ClassDef(TGLTH3Composition, 0)//Composition of TH3 objects.
};

//
//TGLTH3CompositionPainter class.
//
class TGLTH3CompositionPainter: public TGLPlotPainter {
public:
   TGLTH3CompositionPainter(TGLTH3Composition *data, TGLPlotCamera *camera,
                            TGLPlotCoordinates *coord);

   //TGLPlotPainter final-overriders.
   char      *GetPlotInfo(Int_t px, Int_t py);
   Bool_t     InitGeometry();
   void       StartPan(Int_t px, Int_t py);
   void       Pan(Int_t px, Int_t py);
   void       AddOption(const TString &option);
   void       ProcessEvent(Int_t event, Int_t px, Int_t py);

private:
   //TGLPlotPainter final-overriders.
   void       InitGL()const;
   void       DeInitGL()const;

   void       DrawPlot()const;

   //Empty overriders.
   void       DrawSectionXOZ()const{}
   void       DrawSectionYOZ()const{}
   void       DrawSectionXOY()const{}

   void       SetColor(Int_t color)const;

   TGLTH3Composition            *fData;
   std::pair<Double_t, Double_t> fMinMaxVal;

   mutable TGLQuadric            fQuadric;

   TGLTH3CompositionPainter(const TGLTH3CompositionPainter &rhs);
   TGLTH3CompositionPainter &operator = (const TGLTH3CompositionPainter &rhs);

   ClassDef(TGLTH3CompositionPainter, 0)//Painter to draw several TH3.
};


#endif
