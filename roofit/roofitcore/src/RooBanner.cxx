#include "RooFit.h"

#include "Rtypes.h"
#include "Rtypes.h"
#include "Riostream.h"

//////////////////////////////////////////////////////////////////////////////
// 
// BEGIN_HTML
// Print banner message when RooFit library is loaded
// END_HTML
//

const char* VTAG="3.48" ;

Int_t doBanner()

{
#ifndef __ROOFIT_NOBANNER
  cout << endl
  << "\033[1mRooFit v" << VTAG << " -- Developed by Wouter Verkerke and David Kirkby\033[0m " << endl 
              << "                Copyright (C) 2000-2011 NIKHEF, University of California & Stanford University" << endl 
              << "                All rights reserved, please read http://roofit.sourceforge.net/license.txt" << endl << endl ;
#endif
  return 0 ;
}

static Int_t dummy = doBanner() ;
