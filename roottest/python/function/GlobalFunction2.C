/*
  File: roottest/python/function/GlobalFunction2.C
  Author: Wim Lavrijsen@lbl.gov
  Created: 02/16/11
  Last: 02/16/11
*/

double DivideByTwo( double d ) {
   return d/2.;
}

namespace MyNameSpace {

   double NSDivideByTwo( double d ) {
      return d/2.;
   }

}
