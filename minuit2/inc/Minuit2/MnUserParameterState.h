// @(#)root/minuit2:$Name:  $:$Id: MnUserParameterState.h,v 1.3.6.4 2005/11/29 11:08:34 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_MnUserParameterState
#define ROOT_Minuit2_MnUserParameterState

#include "Minuit2/MnUserParameters.h"
#include "Minuit2/MnUserCovariance.h"
#include "Minuit2/MnGlobalCorrelationCoeff.h"

namespace ROOT {

   namespace Minuit2 {


class MinimumState;

/** class which holds the external user and/or internal Minuit representation 
    of the parameters and errors; 
    transformation internal <-> external on demand;  
 */

class MnUserParameterState {

public:

  /// default constructor (invalid state)
  MnUserParameterState() : fValid(false), fCovarianceValid(false), fParameters(MnUserParameters()), fCovariance(MnUserCovariance()), fIntParameters(std::vector<double>()), fIntCovariance(MnUserCovariance()) {} 

  /// construct from user parameters (before minimization)
  MnUserParameterState(const std::vector<double>&, const std::vector<double>&);

  MnUserParameterState(const MnUserParameters&);

  /// construct from user parameters + covariance (before minimization)
  MnUserParameterState(const std::vector<double>&, const std::vector<double>&, unsigned int);

  MnUserParameterState(const std::vector<double>&, const MnUserCovariance&);

  MnUserParameterState(const MnUserParameters&, const MnUserCovariance&);

  /// construct from internal parameters (after minimization)
  MnUserParameterState(const MinimumState&, double, const MnUserTransformation&);

  ~MnUserParameterState() {}

  MnUserParameterState(const MnUserParameterState& state) : fValid(state.fValid), fCovarianceValid(state.fCovarianceValid), fGCCValid(state.fGCCValid), fFVal(state.fFVal), fEDM(state.fEDM), fNFcn(state.fNFcn), fParameters(state.fParameters), fCovariance(state.fCovariance), fGlobalCC(state.fGlobalCC), fIntParameters(state.fIntParameters), fIntCovariance(state.fIntCovariance) {}

  MnUserParameterState& operator=(const MnUserParameterState& state) {
    fValid = state.fValid;
    fCovarianceValid = state.fCovarianceValid;
    fGCCValid = state.fGCCValid;
    fFVal = state.fFVal;
    fEDM = state.fEDM;
    fNFcn = state.fNFcn;
    fParameters = state.fParameters;
    fCovariance = state.fCovariance;
    fGlobalCC = state.fGlobalCC;
    fIntParameters = state.fIntParameters;
    fIntCovariance = state.fIntCovariance;
    return *this;
  }

  //user external representation
  const MnUserParameters& Parameters() const {return fParameters;}
  const MnUserCovariance& Covariance() const {return fCovariance;}
  const MnGlobalCorrelationCoeff& GlobalCC() const {return fGlobalCC;}

  //Minuit internal representation
  const std::vector<double>& IntParameters() const {return fIntParameters;}
  const MnUserCovariance& IntCovariance() const {return fIntCovariance;}

  //transformation internal <-> external
  const MnUserTransformation& Trafo() const {return fParameters.Trafo();}

  bool IsValid() const {return fValid;}
  bool HasCovariance() const {return fCovarianceValid;}
  bool HasGlobalCC() const {return fGCCValid;}

  double Fval() const {return fFVal;}
  double Edm() const {return fEDM;}
  unsigned int NFcn() const {return fNFcn;}  

private:
  
  bool fValid;
  bool fCovarianceValid;
  bool fGCCValid;

  double fFVal;
  double fEDM;
  unsigned int fNFcn;

  MnUserParameters fParameters;
  MnUserCovariance fCovariance;
  MnGlobalCorrelationCoeff fGlobalCC;

  std::vector<double> fIntParameters;
  MnUserCovariance fIntCovariance;

public:

// facade: forward interface of MnUserParameters and MnUserTransformation

  //access to parameters (row-wise)
  const std::vector<MinuitParameter>& MinuitParameters() const;
  //access to parameters and errors in column-wise representation 
  std::vector<double> Params() const;
  std::vector<double> Errors() const;

  //access to single Parameter
  const MinuitParameter& Parameter(unsigned int i) const;

  //add free Parameter
  void Add(const char* Name, double val, double err);
  //add limited Parameter
  void Add(const char* Name, double val, double err, double , double);
  //add const Parameter
  void Add(const char*, double);

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

  // transformation internal <-> external
  double Int2ext(unsigned int, double) const;
  double Ext2int(unsigned int, double) const;
  unsigned int IntOfExt(unsigned int) const;
  unsigned int ExtOfInt(unsigned int) const;
  unsigned int VariableParameters() const;
  const MnMachinePrecision& Precision() const;
  void SetPrecision(double eps);
};

  }  // namespace Minuit2

}  // namespace ROOT

#endif  // ROOT_Minuit2_MnUserParameterState
