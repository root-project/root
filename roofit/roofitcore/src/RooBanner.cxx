#include "RooFit.h"

#include "Rtypes.h"
#include "Riostream.h"
#include "TEnv.h"

//////////////////////////////////////////////////////////////////////////////
// 
// BEGIN_HTML
// Print banner message when RooFit library is loaded
// END_HTML
//

using namespace std;

const char* VTAG="3.60" ;

Int_t doBanner();

static Int_t dummy = doBanner() ;

Int_t doBanner()

{
#ifndef __ROOFIT_NOBANNER
   if (gEnv->GetValue("RooFit.Banner", 1)) {
       cout << endl
      << "\033[1mRooFit v" << VTAG << " -- Developed by Wouter Verkerke and David Kirkby\033[0m " << endl
      << "                Copyright (C) 2000-2013 NIKHEF, University of California & Stanford University" << endl
      << "                All rights reserved, please read http://roofit.sourceforge.net/license.txt" << endl
      << endl ;
   }
#endif
  (void) dummy;
  return 0 ;
}

