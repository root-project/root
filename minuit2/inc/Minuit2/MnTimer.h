// @(#)root/minuit2:$Name:  $:$Id: MnTimer.h,v 1.1.6.3 2005/11/29 11:08:34 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_MnTimer
#define ROOT_Minuit2_MnTimer

namespace ROOT {

   namespace Minuit2 {


//  
//  Vincenzo's PentiumTimer, taken from COBRA and adapted
//
//   V 0.0 
//

extern "C" inline unsigned long long int  RdtscPentium() {
  unsigned long long int x;
  __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
  return x;
}

class MnTimer {

public:
  
  typedef unsigned long long int PentiumTimeType;
  typedef long long int PentiumTimeIntervalType;
  typedef PentiumTimeType TimeType;

  inline static TimeType Time() {return RdtscPentium();}

private:

};

  }  // namespace Minuit2

}  // namespace ROOT

#endif  // ROOT_Minuit2_MnTimer

