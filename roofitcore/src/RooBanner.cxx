#include "Rtypes.h"
#include <iostream.h>

Int_t doBanner()
{
  cout << endl
       << "\033[1mRooFit -- Developed by Wouter Verkerke and David Kirkby\033[0m " << endl 
              << "          Copyright (C) 2001-2002 University of California & Stanford University" << endl 
              << "          http://www.slac.stanford.edu/BFROOT/www/Computing/Offline/ROOT/RooFit" << endl
              << "          If you've enjoyed use of this product, please send $10 (cash)" << endl 
              << "          to the authors, care of SLAC, Building 48, Room 230." << endl << endl ;    
  return 0 ;
}

static Int_t dummy = doBanner() ;
