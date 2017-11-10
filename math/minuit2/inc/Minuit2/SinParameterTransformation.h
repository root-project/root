// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei, E.G.P. Bos   2003-2017

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 * Copyright (c) 2017 Patrick Bos, Netherlands eScience Center        *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_SinParameterTransformation
#define ROOT_Minuit2_SinParameterTransformation

namespace ROOT {

  namespace Minuit2 {


    class MnMachinePrecision;

/**
   class for the transformation for double-limited parameter
   Using a sin function one goes from a double-limited parameter range to
   an unlimited one
 */
    class SinParameterTransformation {

    public:

      SinParameterTransformation() {}

      ~SinParameterTransformation() {}

      double Int2ext(double Value, double Upper, double Lower) const;
      double Ext2int(double Value, double Upper, double Lower,
                     const MnMachinePrecision&) const;
      double DInt2Ext(double Value, double Upper, double Lower) const;

      double D2Int2Ext(double Value, double Upper, double Lower) const;
      double GStepInt2Ext(double Value, double Upper, double Lower) const;

    private:

    };

  }  // namespace Minuit2

}  // namespace ROOT

#endif  // ROOT_Minuit2_SinParameterTransformation
