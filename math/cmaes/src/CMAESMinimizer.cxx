/**
 * CMA-ES, Covariance Matrix Evolution Strategy
 * Copyright (c) 2014 INRIA
 * Author: Emmanuel Benazera <emmanuel.benazera@lri.fr>
 *
 * This file is part of libcmaes.
 *
 * libcmaes is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libcmaes is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with libcmaes.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "CMAESMinimizer.h"
#include "Math/IFunctionfwd.h" // fObjFunc
#include "Math/IOptions.h"
#include "Math/Error.h"
#include "Fit/ParameterSettings.h"

#ifdef USE_ROOT_ERROR
#include "TROOT.h"
#endif

namespace ROOT
{
  namespace cmaes
  {

    TCMAESMinimizer::TCMAESMinimizer()
      :Minimizer(),fDim(0),fFreeDim(0)
    {
    }

    TCMAESMinimizer::TCMAESMinimizer(const char *type)
      :Minimizer(),fDim(0),fFreeDim(0)
    {
      //std::string algoname(type);
      // tolower() is not an  std function (Windows)
      //std::transform(algoname.begin(), algoname.end(), algoname.begin(), (int(*)(int)) tolower );
    }

    TCMAESMinimizer::TCMAESMinimizer(const TCMAESMinimizer &m)
      :Minimizer()
    {
    }

    TCMAESMinimizer& TCMAESMinimizer::operator = (const TCMAESMinimizer &rhs)
    {
      if (this == &rhs) return *this;
      return *this;
    }
    
    TCMAESMinimizer::~TCMAESMinimizer()
    {
    }

    void TCMAESMinimizer::Clear()
    {
      fCMAsols = CMASolutions();
      fDim = 0; fFreeDim = 0;
      fLBounds.clear();
      fUBounds.clear();
      fVariablesType.clear();
      fInitialX.clear();
      fInitialSigma.clear();
      fNames.clear();
      fGlobalCC.clear();
      fValues.clear();
      fErrors.clear();
    }

    void TCMAESMinimizer::SetFunction(const ROOT::Math::IMultiGenFunction &fun)
    {
      fObjFunc = &fun;
      fDim = fun.NDim();
    }
    
    bool TCMAESMinimizer::SetVariable(unsigned int ivar, const std::string & name, double val, double step)
    {
      if (ivar > fInitialX.size() ) {
	MATH_ERROR_MSG("TCMAESMinimizer::SetVariable","ivar out of range");
	return false;
      }
      if (ivar == fInitialX.size() ) { 
	fInitialX.push_back(val); 
	fNames.push_back(name);	
	fInitialSigma.push_back(step);
	fLBounds.push_back(0);
	fUBounds.push_back(0);
	if (step==0.){
	  fVariablesType.push_back(1);
	}
	else {
	  fFreeDim++;
	  fVariablesType.push_back(0);
	}
      }
      else { 
	if (step==0.) {
	  if (fInitialSigma[ivar]!=0.) { //Constraining a free variable.
	    fFreeDim--; 
	    fVariablesType[ivar] = 1;
	  }
	}
	else {
	  if (fInitialSigma[ivar]==0.) { //Freeing a constrained variable
	    fFreeDim++;
	    fVariablesType[ivar] = 0;
	  }
	}
	fInitialX[ivar] = val;
	fNames[ivar] = name;
	fInitialSigma[ivar] = step;
      } 
      return true;
    }

    bool TCMAESMinimizer::SetLowerLimitedVariable(unsigned int  ivar , const std::string & name , double val , double step , double lower )
    {
      if (lower > val) {
	MATH_WARN_MSG("TCMAESMinimizer::SetLowerLimitedVariable", "Starting point set into the unfeasible domain"); // fix with val=lower; ?
      }
      bool r = SetVariable(ivar, name, val, step);
      if (!r) return false;
      fLBounds[ivar] = lower;
      fVariablesType[ivar] = 2;
      return true;
    }

    bool TCMAESMinimizer::SetUpperLimitedVariable(unsigned int ivar , const std::string & name , double val , double step , double upper )
    {
      if (upper > val) {
	MATH_WARN_MSG("TCMAESMinimizer::SetUpperLimitedVariable", "Starting point set into the unfeasible domain");
      }
      bool r = SetVariable(ivar, name, val, step);
      if (!r) return false;
      fUBounds[ivar] = upper;
      fVariablesType[ivar] = 3;
      return true;
    }

    bool TCMAESMinimizer::SetLimitedVariable(unsigned int ivar , const std::string & name , double val , double step , double lower, double upper)
    {
      if (upper == lower) {
	MATH_WARN_MSG("TCMAESMinimizer::SetLimitedVariable","Upper bound equal to lower bound. Variable is constrained to fixed value.");
	return SetFixedVariable(ivar, name, val);
      }
      if (upper < lower) {
	MATH_WARN_MSG("TCMAESMinimizer::SetLimitedVariable","Upper bound lesser than lower bound. Bounds exchanged.");
	double temp(upper);
	upper = lower;
	lower = temp;
      }
      if (val < lower || val > upper) {
	MATH_WARN_MSG("TCMAESMinimizer::SetLimitedVariable", "Starting point set into the unfeasible domain");
      }
      bool r = SetVariable(ivar, name, val, step);
      if (!r) return false;
      fLBounds[ivar] = lower;
      fUBounds[ivar] = upper;
      fVariablesType[ivar] = 4;
      return true;
    }
    
    bool TCMAESMinimizer::SetVariableValue(unsigned int ivar, double val )
    {
      if (ivar >= fInitialX.size() ) {
	//TODO string that gives value of ivar and fInitialX.size() 
	MATH_ERROR_MSG("TCMAESMinimizer::SetVariableValue","ivar out of range");
	return false;
      }
      if (fVariablesType[ivar] == 2 || fVariablesType[ivar] == 4) {
	if (fLBounds[ivar] > val) {
	  MATH_WARN_MSG("TCMAESMinimizer::SetVariableValue", "Starting point set into the unfeasible domain");
	}
      }
      if (fVariablesType[ivar] == 3 || fVariablesType[ivar] == 4) {
	if (fUBounds[ivar] < val) {
	  MATH_WARN_MSG("TCMAESMinimizer::SetVariableValue", "Starting point set into the unfeasible domain");
	}
      }
      fInitialX[ivar] = val;
      return true;
    }
    
    bool TCMAESMinimizer::SetVariableValues(const double * x)
    {
      if (x == NULL)
	{
	  MATH_WARN_MSG("TCMAESMinimizer::SetVariableValues", "No values given, no change to the starting point.");
	  return false;
	}
      unsigned int i;
      for (i=0; i<fInitialX.size(); i++) {
	SetVariableValue(i, x[i]);
      }
      return true;
    }

    bool TCMAESMinimizer::SetVariableLowerLimit(unsigned int ivar, double lower)
    {
      if (ivar > fLBounds.size())
	return false;
      fLBounds[ivar] = lower;
      fVariablesType[ivar] = 2;
      return true;
    }

    bool TCMAESMinimizer::SetVariableUpperLimit(unsigned int ivar, double upper)
    {
      if (ivar > fUBounds.size())
	return false;
      fUBounds[ivar] = upper;
      fVariablesType[ivar] = 3;
      return true;
    }

    bool TCMAESMinimizer::SetVariableLimits(unsigned int ivar, double lower, double upper)
    {
      if (ivar >= fLBounds.size() || ivar >= fUBounds.size())
	return false;
      fLBounds[ivar] = lower;
      fUBounds[ivar] = upper;
      fVariablesType[ivar] = 4;
      return true;
    }

    bool TCMAESMinimizer::GetVariableSettings(unsigned int ivar, ROOT::Fit::ParameterSettings &varObj) const
    {
      if (ivar >= fInitialX.size())
	{
	  MATH_ERROR_MSG("TCMAESMinimizer::GetVariableSettings","wrong variable index");
	  return false;
	}
      varObj.Set(fNames.at(ivar),fInitialX.at(ivar),false); //XXX: not sure of last param type.
      if (fVariablesType.at(ivar) == 4)
	varObj.SetLimits(fLBounds.at(ivar),fUBounds.at(ivar));
      else if (fVariablesType.at(ivar) == 3)
	varObj.SetUpperLimit(fUBounds.at(ivar));
      else if (fVariablesType.at(ivar) == 2)
	varObj.SetLowerLimit(fLBounds.at(ivar));
      return true;
    }

    std::string TCMAESMinimizer::VariableName(unsigned int ivar) const
    {
      if (ivar >= fInitialX.size())
	return std::string();
      return fNames.at(ivar);
    }

    int TCMAESMinimizer::VariableIndex(const std::string &name) const
    {
      for (unsigned int i=0;i<fNames.size();i)
	if (fNames.at(i) == name)
	  return i;
      return -1;
    }
    
    bool TCMAESMinimizer::Minimize()
    {
      if (!fObjFunc) {
	MATH_ERROR_MSG("TCMAESMinimizer::Minimize","Objective function has not been set"); 
	return false;
      }
      if (!fDim) { 
	MATH_ERROR_MSG("TCMAESMinimizer::Minimize","Dimension has not been set"); 
	return false;
      }
      if (fDim > fInitialX.size()) {
	MATH_ERROR_MSG("TCMAESMinimizer::Minimize","Dimension larger than initial X size's");
	return false;
      }
      if (fDim < fInitialX.size()) {
	MATH_WARN_MSG("TCMAESMinimizer::Minimize","Dimension smaller than initial X size's");
      }

      //ROOT::Math::IOptions *cmaesOpt = ROOT::Math::MinimizerOptions::FindDefault("cmaes"); //TODO.
      
      //TODO: phenotype / genotype.
      
      CMAParameters<> cmaparams(fDim);
      //TODO: x0, sigma0, ...
      FitFunc ffit = [this](const double *x, const int N)
	{
	  return (*fObjFunc)(x);
	};
      ProgressFunc<CMAParameters<>,CMASolutions> pfunc = [](const CMAParameters<> &cmaparams, const CMASolutions &cmasols) { return 0; };
      fCMAsols = libcmaes::cmaes<>(ffit,cmaparams,pfunc);
      //fCMAsols = cmaes<>([this](const double *x, const int N){ return (*fObjFunc)(x);},cmaparams); //TODO: use bounds as needed.
      fStatus = fCMAsols._run_status; //TODO: convert so that to match that of Minuit2 ?
      return fCMAsols._run_status >= 0;
    }

    double TCMAESMinimizer::MinValue() const
    {
      return fCMAsols.best_candidate()._fvalue;
    }

    const double* TCMAESMinimizer::X() const
    {
      //TODO: return pheno x when applicable (in solution object).
      //std::cout << "X=" << fCMAsols.best_candidate()._x.transpose() << std::endl;
      fValues.clear();
      Candidate bc = fCMAsols.best_candidate();
      for (int i=0;i<fDim;i++)
	fValues.push_back(bc._x(i));
      return &fValues.front();
    }

    const double* TCMAESMinimizer::Errors() const
    {
      fErrors.clear();
      //std::cout << "diag=" << fCMAsols._cov.diagonal() << std::endl;
      const double* diag = fCMAsols._cov.diagonal().data();
      for (int i=0;i<fDim;i++)
	fErrors.push_back(std::sqrt(std::abs(diag[i]))); // abs for numerical errors that bring the sqrt below 0.
      return &fErrors.front();
    }
    
    unsigned int TCMAESMinimizer::NCalls() const
    {
      return fCMAsols._nevals;
    }

    double TCMAESMinimizer::CovMatrix(unsigned int i, unsigned int j) const
    {
      return fCMAsols._cov(i,j);
    }

    bool TCMAESMinimizer::GetCovMatrix(double *cov) const
    {
      std::copy(fCMAsols._cov.data(),fCMAsols._cov.data()+fCMAsols._cov.size(),cov);
      return true;
    }

    double TCMAESMinimizer::Correlation(unsigned int i, unsigned int j) const
    {
      return std::sqrt(std::abs(fCMAsols._cov(i,i)*fCMAsols._cov(j,j)));
    }

    double TCMAESMinimizer::GlobalCC(unsigned int i) const
    {
      // original Minuit paper says:
      // \rho_k^2 = 1 - [C_{kk}C_{kk}^{-1}]^{-1}
      if (fGlobalCC.empty()) // need to pre-compute the vector coefficient
	{
	  dMat covinv = fCMAsols._cov.inverse();
	  for (int i=0;i<covinv.rows();i++)
	    {
	      double denom = covinv(i,i)*fCMAsols._cov(i,i);
	      if (denom < 1.0 && denom > 0.0)
		fGlobalCC.push_back(0.0);
	      else fGlobalCC.push_back(std::sqrt(1.0 - 1.0/denom));
	    }
	}
      return fGlobalCC.at(i);
    }

    bool TCMAESMinimizer::GetMinosError(unsigned int i, double &errLow, double &errUp, int j)
    {
      //TODO.
      return false;
    }

    bool TCMAESMinimizer::Scan(unsigned int i, unsigned int &nstep, double *x, double *y, double xmin, double xmax)
    {
      //TODO.
      return false;
    }

    bool TCMAESMinimizer::Contour(unsigned int i, unsigned int j, unsigned int &npoints, double *xi, double *xj)
    {
      //TODO.
      return false;
    }

    void TCMAESMinimizer::PrintResults()
    {
      std::cout << "CMAESMinimizer : Valid minimum - status = " << fStatus << std::endl;
      std::cout << "FVAL  = " << MinValue() << std::endl;
      std::cout << "Nfcn  = " << NCalls() << std::endl;
      for (unsigned int i=0;i<fDim;i++)
	{
	  std::cout << fNames.at(i) << "\t  = " << X()[i] << std::endl;
	  //TODO: error bounds.
	}
    }
    
  }
}
