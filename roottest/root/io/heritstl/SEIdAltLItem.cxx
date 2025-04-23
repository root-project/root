////////////////////////////////////////////////////////////////////////////
// $Id$
//
// SEIdAltLItem
//
// SEIdAltLItem is a strip-end alternative list (vector) item
//
// Author:  R. Hatcher 2001.10.22
//
////////////////////////////////////////////////////////////////////////////

#include "SEIdAltLItem.h"

#include <typeinfo>
#include <iostream>
#include <iomanip>
using namespace std;

ClassImp(SEIdAltLItem)

//______________________________________________________________________________
std::ostream& operator<<(std::ostream& os, const SEIdAltLItem& item)
{

   os << " " << item.fStripEndId 
      << " wgt=" << item.fWeight
      << " pe=" << item.fPE
      << " lin=" << item.fSigLin
      << " corr=" << item.fSigCorr
      << " time=" << item.fTime
      << " " << endl;

   return os;
}

//______________________________________________________________________________
void SEIdAltLItem::Print(Option_t *option) const
{
   switch (option[0]) {
   case 'c':
      printf("%4d | %9.4f | %6.1f | %9f | %9f | %9f ",
             fStripEndId,fWeight,fPE,fSigLin,fSigCorr,fTime);
      break;
   default:
      printf("%4d wgt=%10f pe=%6.1f lin=%10f corr=%10f time=%10f ",
             fStripEndId,fWeight,fPE,fSigLin,fSigCorr,fTime);
   }
}
//______________________________________________________________________________
