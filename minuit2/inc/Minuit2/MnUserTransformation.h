// @(#)root/minuit2:$Name:  $:$Id: MnUserTransformation.h,v 1.1 2005/11/29 14:42:18 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_MnUserTransformation
#define ROOT_Minuit2_MnUserTransformation

#ifndef ROOT_Minuit2_MnConfig
#include "Minuit2/MnConfig.h"
#endif
#ifndef ROOT_Minuit2_MnMatrix
#include "Minuit2/MnMatrix.h"
#endif
#ifndef ROOT_Minuit2_MinuitParameter
#include "Minuit2/MinuitParameter.h"
#endif
#ifndef ROOT_Minuit2_MnMachinePrecision
#include "Minuit2/MnMachinePrecision.h"
#endif
#ifndef ROOT_Minuit2_SinParameterTransformation
#include "Minuit2/SinParameterTransformation.h"
#endif
#ifndef ROOT_Minuit2_SqrtLowParameterTransformation
#include "Minuit2/SqrtLowParameterTransformation.h"
#endif
#ifndef ROOT_Minuit2_SqrtUpParameterTransformation
#include "Minuit2/SqrtUpParameterTransformation.h"
#endif

#include <vector>

namespace ROOT {

   namespace Minuit2 {


class MnUserCovariance;

// class MnMachinePrecision;

/** knows how to transform between user specified parameters (external) and
    internal parameters used for minimization
 */

class MnUserTransformation {

public:

  MnUserTransformation() : fPrecision(MnMachinePrecision()),
			   fParameters(std::vector<MinuitParameter>()),
			   fExtOfInt(std::vector<unsigned int>()),
			   fDoubleLimTrafo(SinParameterTransformation()),
			   fUpperLimTrafo(SqrtUpParameterTransformation()),
			   fLowerLimTrafo(SqrtLowParameterTransformation()),
			   fCache(std::vector<double>()) {}

  MnUserTransformation(const std::vector<double>&, const std::vector<double>&);

  ~MnUserTransformation() {}

  MnUserTransformation(const MnUserTransformation& trafo) : 
    fPrecision(trafo.fPrecision),
    fParameters(trafo.fParameters),fExtOfInt(trafo.fExtOfInt), 
    fDoubleLimTrafo(trafo.fDoubleLimTrafo), 
    fUpperLimTrafo(trafo.fUpperLimTrafo), 
    fLowerLimTrafo(trafo.fLowerLimTrafo), fCache(trafo.fCache) {}
  
  MnUserTransformation& operator=(const MnUserTransformation& trafo) {
    fPrecision = trafo.fPrecision;
    fParameters = trafo.fParameters;
    fExtOfInt = trafo.fExtOfInt;
    fDoubleLimTrafo = trafo.fDoubleLimTrafo;
    fUpperLimTrafo = trafo.fUpperLimTrafo;
    fLowerLimTrafo = trafo.fLowerLimTrafo;
    fCache = trafo.fCache;
    return *this;
  }

  const std::vector<double>& operator()(const MnAlgebraicVector&) const;

  // Index = internal Parameter
  double Int2ext(unsigned int, double) const;

  // Index = internal Parameter
  double Int2extError(unsigned int, double, double) const;

  MnUserCovariance Int2extCovariance(const MnAlgebraicVector&, const MnAlgebraicSymMatrix&) const;

  // Index = external Parameter
  double Ext2int(unsigned int, double) const;

  // Index = internal Parameter
  double DInt2Ext(unsigned int, double) const;

//   // Index = external Parameter
//   double dExt2Int(unsigned int, double) const;

  // Index = external Parameter
  unsigned int IntOfExt(unsigned int) const;

  // Index = internal Parameter
  unsigned int ExtOfInt(unsigned int internal) const { 
    assert(internal < fExtOfInt.size());
    return fExtOfInt[internal];
  }

  const std::vector<MinuitParameter>& Parameters() const {
    return fParameters;
  }

  unsigned int VariableParameters() const {return static_cast<unsigned int> ( fExtOfInt.size() );}

private:

  MnMachinePrecision fPrecision;
  std::vector<MinuitParameter> fParameters;
  std::vector<unsigned int> fExtOfInt;

  SinParameterTransformation fDoubleLimTrafo;
  SqrtUpParameterTransformation fUpperLimTrafo;
  SqrtLowParameterTransformation fLowerLimTrafo;

  mutable std::vector<double> fCache;

public:

  //forwarded interface
  const MnMachinePrecision& Precision() const {return fPrecision;}
  void SetPrecision(double eps) {fPrecision.SetPrecision(eps);}

  //access to parameters and errors in column-wise representation 
  std::vector<double> Params() const;
  std::vector<double> Errors() const;

  //access to single Parameter
  const MinuitParameter& Parameter(unsigned int) const;

  //add free Parameter
  bool Add(const char*, double, double);
  //add limited Parameter
  bool Add(const char*, double, double, double, double);
  //add const Parameter
  bool Add(const char*, double);

  //interaction via external number of Parameter
  void Fix(unsigned int);
  void Release(unsigned int);
  void SetValue(unsigned int, double);
  void SetError(unsigned int, double);
  void SetLimits(unsigned int, double, double);
  void SetUpperLimit(unsigned int, double);
  void SetLowerLimit(unsigned int, double);
  void RemoveLimits(unsigned int);

  double Value(unsigned int) const;
  double Error(unsigned int) const;
  
  //interaction via Name of Parameter
  void Fix(const char*);
  void Release(const char*);
  void SetValue(const char*, double);
  void SetError(const char*, double);
  void SetLimits(const char*, double, double);
  void SetUpperLimit(const char*, double);
  void SetLowerLimit(const char*, double);
  void RemoveLimits(const char*);

  double Value(const char*) const;
  double Error(const char*) const;
  
  //convert Name into external number of Parameter
  unsigned int Index(const char*) const;
  //convert external number into Name of Parameter
  const char* Name(unsigned int) const;

};

  }  // namespace Minuit2

}  // namespace ROOT

#endif  // ROOT_Minuit2_MnUserTransformation
