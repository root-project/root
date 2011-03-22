/*
  File: roottest/python/function/GlobalFunction.C
  Author: Wim Lavrijsen@lbl.gov
  Created: 04/22/05
  Last: 02/16/11
*/

double InterpDivideByTwo( double d ) {
   return d/2.;
}

namespace InterpMyNameSpace {

   double InterpNSDivideByTwo( double d ) {
      return d/2.;
   }

}
