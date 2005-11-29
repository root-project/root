// @(#)root/minuit2:$Name:  $:$Id: SinParameterTransformation.h,v 1.9.6.4 2005/11/29 11:08:34 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_SinParameterTransformation
#define ROOT_Minuit2_SinParameterTransformation

namespace ROOT {

   namespace Minuit2 {


class MnMachinePrecision;

class SinParameterTransformation {

public:

  SinParameterTransformation() {}

  ~SinParameterTransformation() {}

  double Int2ext(double Value, double Upper, double Lower) const;
  double Ext2int(double Value, double Upper, double Lower, 
		 const MnMachinePrecision&) const;
  double DInt2Ext(double Value, double Upper, double Lower) const;

private:

};

  }  // namespace Minuit2

}  // namespace ROOT

#endif  // ROOT_Minuit2_SinParameterTransformation
