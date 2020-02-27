// @(#)root/eve7:$Id$
// Author: Sergey Linev, 27.02.2020

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/REveGeoPainter.hxx>

using namespace ROOT::Experimental;

REveGeoPainter::REveGeoPainter(TGeoManager *manager) : TVirtualGeoPainter(manager)
{
   printf("DID CREATE REveGeoPainter\n");
}

REveGeoPainter::~REveGeoPainter()
{

}
