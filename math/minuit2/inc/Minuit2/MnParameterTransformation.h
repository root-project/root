// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei, E.G.P. Bos   2003-2017

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

/**
   class for the transformation for double-limited parameter
   Using a sin function one goes from a double-limited parameter range to
   an unlimited one
 */
class SinParameterTransformation {
public:
   long double Int2ext(long double Value, long double Upper, long double Lower) const;
   long double Ext2int(long double Value, long double Upper, long double Lower, const MnMachinePrecision &) const;
   long double DInt2Ext(long double Value, long double Upper, long double Lower) const;
   long double DExt2Int(long double Value, long double Upper, long double Lower) const;
};

/**
 * Transformation from external to internal Parameter based on  sqrt(1 + x**2)
 *
 * This transformation applies for the case of single side Lower Parameter limits
 */

class SqrtLowParameterTransformation /* : public ParameterTransformation */ {
public:
   // transformation from internal to external
   long double Int2ext(long double Value, long double Lower) const;

   // transformation from external to internal
   long double Ext2int(long double Value, long double Lower, const MnMachinePrecision &) const;

   // derivative of transformation from internal to external
   long double DInt2Ext(long double Value, long double Lower) const;

   // derivative of transformation from external to internal
   long double DExt2Int(long double Value, long double Lower) const;
};

/**
 * Transformation from external to internal Parameter based on  sqrt(1 + x**2)
 *
 * This transformation applies for the case of single side Upper Parameter limits
 */

class SqrtUpParameterTransformation /* : public ParameterTransformation */ {
public:
   // transformation from internal to external
   long double Int2ext(long double Value, long double Upper) const;

   // transformation from external to internal
   long double Ext2int(long double Value, long double Upper, const MnMachinePrecision &) const;

   // derivative of transformation from internal to external
   long double DInt2Ext(long double Value, long double Upper) const;

   // derivative of transformation from external to internal
   long double DExt2Int(long double Value, long double Upper) const;
};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_SinParameterTransformation
