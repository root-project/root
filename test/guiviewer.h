// @(#)root/test:$Id$
// Author: Brett Viren   04/15/2001

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

class TList;
class TCanvas;
class TRootEmbeddedCanvas;
class TGaxis;
class TGDoubleSlider;


class Viewer : public TGMainFrame {

private:
   TList               *fCleanup;
   TCanvas             *fCanvas;
   TRootEmbeddedCanvas *fHScaleCanvas, *fVScaleCanvas;
   TGaxis              *fHScale, *fVScale;
   TGDoubleSlider      *fHSlider;
   TGDoubleSlider      *fVSlider;

public:
   Viewer(const TGWindow *win);
   virtual ~Viewer();
   void DoButton();
   void DoSlider();
   void SetRange(Float_t xmin, Float_t ymin, Float_t xmax, Float_t ymax,
                 Bool_t move_slider = kTRUE);
   ClassDef(Viewer,0) //GUI example
};
