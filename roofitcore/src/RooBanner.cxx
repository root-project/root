#include "RooFitCore/RooFit.hh"

#include "Rtypes.h"
#include "Rtypes.h"
#include "Riostream.h"

// -- CLASS DESCRIPTION [AUX] --
// Print banner message when RooFit library is loaded

const char* VTAG="2.09" ;

Int_t doBanner()
{
  cout << endl
  << "\033[1mRooFit v" << VTAG << " -- Developed by Wouter Verkerke and David Kirkby\033[0m " << endl 
              << "                Copyright (C) 2000-2005 NIKHEF, University of California & Stanford University" << endl 
              << "                All rights reserved, please read http://roofit.sourceforge.net/license.txt" << endl << endl ;
  return 0 ;
}

static Int_t dummy = doBanner() ;
