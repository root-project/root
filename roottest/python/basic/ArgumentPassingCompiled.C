/*
  File: roottest/python/basic/ArgumentPassingCompiled.C
  Author: Wim Lavrijsen@lbl.gov
  Created: 05/09/11
  Last: 07/25/11
*/

// Note that part of this is a repetition of ArgumentPassingInterpreted.C

#include "TString.h"
#include "TLorentzVector.h"

#include <iostream>

namespace CompiledTest {

TString StringValueArguments( TString arg1, int argn = 0, TString arg2 = "default" )
{
   switch (argn) {
   case 0:
      return arg1;
   case 1:
      return arg2;
   default:
      break;
   }

   return "argn invalid";
}

TString StringRefArguments(
   const TString& arg1, int argn = 0, const TString& arg2 = "default" )
{
   return StringValueArguments( arg1, argn, arg2 );
}


TLorentzVector LorentzVectorValueArguments( TLorentzVector arg1,
   int argn = 0, TLorentzVector arg2 = TLorentzVector( 1, 2, 3, 4 ) )
{
   switch (argn) {
   case 0:
      return arg1;
   case 1:
      return arg2;
   default:
      break;
   }

   return TLorentzVector( -1, -1, -1, -1 );
}

TLorentzVector LorentzVectorRefArguments( const TLorentzVector& arg1,
   int argn = 0, const TLorentzVector& arg2 = TLorentzVector( 1, 2, 3, 4 ) )
{
   return LorentzVectorValueArguments( arg1, argn, arg2 );
}

unsigned int UnsignedIntByRef( unsigned int& a ) {
   a = 3;
   return 3;
}

} // namespace CompiledTest
