/// \file TMenuItem.cxx
/// \ingroup Base ROOT7
/// \author Sergey Linev
/// \date 2017-07-18
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TMenuItem.hxx"

#include "TROOT.h"
#include "TClass.h"
#include "TBufferJSON.h"

void ROOT::Experimental::TMenuItems::Cleanup()
{
   for (unsigned n = 0; n < fItems.size(); ++n) delete fItems[n];

   fItems.clear();
}

std::string ROOT::Experimental::TMenuItems::ProduceJSON()
{
   TClass *cl = gROOT->GetClass("std::vector<ROOT::Experimental::Detail::TMenuItem*>");

   // printf("Got items %d class %p %s\n", (int) fItems.size(), cl, cl->GetName());

   // FIXME: got problem with std::list<TMenuItem>, can be generic TBufferJSON
   TString res = TBufferJSON::ConvertToJSON(&fItems, cl);

   // printf("Got JSON %s\n", res.Data());

   return res.Data();
}
