// @(#)root/ged:$Id$
// Author: Denis Favre-Miville   08/09/05

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TStylePreview
#define ROOT_TStylePreview

#include "TGFrame.h"

class TCanvas;
class TList;
class TRootEmbeddedCanvas;
class TStyle;
class TVirtualPad;

class TStylePreview : public TGTransientFrame {

private:
   TRootEmbeddedCanvas  *fEcan;                    ///< canvas for preview
   TVirtualPad          *fPad;                     ///< original pad previewed
   TList                *fTrashListLayout;         ///< to avoid memory leak

public:
   TStylePreview(const TGWindow *p, TStyle *style, TVirtualPad *currentPad);
   virtual ~TStylePreview();
   void Update(TStyle *style, TVirtualPad *pad);
   void MapTheWindow();
   TCanvas *GetMainCanvas();

   ClassDef(TStylePreview, 0) // Preview window used by the TStyleManager class
};

#endif
