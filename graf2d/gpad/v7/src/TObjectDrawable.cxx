/// \file TObjectDrawable.cxx
/// \ingroup CanvasPainter ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-05-31
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!


/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/TObjectDrawable.hxx>

#include "ROOT/TDisplayItem.hxx"
#include <ROOT/TLogger.hxx>
#include <ROOT/TVirtualCanvasPainter.hxx>

#include <exception>

void ROOT::Experimental::Internal::TObjectDrawable::Paint(TVirtualCanvasPainter& canv) {
   ROOT::Experimental::TDisplayItem *res = new TOrdinaryDisplayItem<TObject>(fObj.get());
   res->SetOption(fOpts.c_str());

   canv.AddDisplayItem(res);
}

