// Author: Sergey Linev, GSI  06/08/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRaylibGuiFactory
#define ROOT_TRaylibGuiFactory

#include "TGuiFactory.h"

namespace ROOT {
namespace Experimental {

/** \class TRaylibGuiFactory
    \ingroup raylibcanvas

    Factory for ROOT GUI components using raylib as the rendering backend.
    Provides specialization for TCanvasImp class.
*/
class TRaylibGuiFactory : public TGuiFactory {

public:
   TRaylibGuiFactory(const char *name = "raylib", const char *title = "ROOT Raylib Gui");
   ~TRaylibGuiFactory() override = default;

   Bool_t UseVirtualX() const override { return kFALSE; }

   TCanvasImp *CreateCanvasImp(TCanvas *c, const char *title,
                                UInt_t width, UInt_t height) override;
   TCanvasImp *CreateCanvasImp(TCanvas *c, const char *title,
                                Int_t x, Int_t y, UInt_t width, UInt_t height) override;

   ClassDefOverride(TRaylibGuiFactory, 0) // raylib gui factory
};

} // namespace Experimental
} // namespace ROOT

#endif