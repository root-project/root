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

#include "errstats.h" // libcmaes extras.

#ifdef USE_ROOT_ERROR
#include "TROOT.h"
#endif

using namespace libcmaes;

namespace ROOT
{
  // registers a default empty set of extra options.
  //ROOT::Math::IOptions &defIOptions = ROOT::Math::MinimizerOptions::Default("cmaes");
  
  namespace cmaes
  {

    TCMAESMinimizer::TCMAESMinimizer()
      :Minimizer(),fDim(0),fFreeDim(0),fWithBounds(false),fWithGradient(false)
    {
    }

    TCMAESMinimizer::TCMAESMinimizer(const char *type)
      :Minimizer(),fDim(0),fFreeDim(0),fWithBounds(false),fWithGradient(false)
    {
      std::string algoname(type);
      // tolower() is not an  std function (Windows)
      std::transform(algoname.begin(), algoname.end(), algoname.begin(), (int(*)(int)) tolower );
      if (algoname == "cmaes")
	fMinimizer = CMAES_DEFAULT;
      else if (algoname == "ipop")
	fMinimizer = IPOP_CMAES;
      else if (algoname == "bipop")
	fMinimizer = BIPOP_CMAES;
      else if (algoname == "acmaes")
	fMinimizer = aCMAES;
      else if (algoname == "aipop")
	fMinimizer = aIPOP_CMAES;
      else if (algoname == "abipop")
	fMinimizer = aBIPOP_CMAES;
      else if (algoname == "sepcmaes")
	fMinimizer = sepCMAES;
      else if (algoname == "sepipop")
	fMinimizer = sepIPOP_CMAES;
      else if (algoname == "sepbipop")
	fMinimizer = sepBIPOP_CMAES;

      
    }

    TCMAESMinimizer::TCMAESMinimizer(const TCMAESMinimizer &m)
      :Minimizer(),fDim(0),fFreeDim(0),fWithBounds(false),fWithGradient(false)
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
      fCMAparams = CMAParameters<>();
      fDim = 0; fFreeDim = 0;
      fLBounds.clear();
      fUBounds.clear();
      fVariablesType.clear();
      fInitialX.clear();
      fInitialSigma.clear();
      fFixedVariables.clear();
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

    void TCMAESMinimizer::SetFunction(const ROOT::Math::IMultiGradFunction &fun)
    {
      SetFunction(static_cast<const ::ROOT::Math::IMultiGenFunction &> (fun));
      fObjFuncGrad = &fun;
      //fDim = fun.NDim();
      fWithGradient = true;
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
	fLBounds.push_back(-std::numeric_limits<double>::max());
	fUBounds.push_back(std::numeric_limits<double>::max());
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
      fWithBounds = true;
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
      fWithBounds = true;
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
      fWithBounds = true;
      return true;
    }

    bool TCMAESMinimizer::SetFixedVariable(unsigned int ivar, const std::string &name, double val)
    {
      SetVariable(ivar,name,val,0.0);
      fFixedVariables.insert(std::pair<int,double>(ivar,val));
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

    bool TCMAESMinimizer::SetVariableStepSize(unsigned int ivar, double step)
    {
      if (ivar > fInitialX.size())
	return false;
      fInitialSigma[ivar] = step;
      return true;
    }
    
    bool TCMAESMinimizer::SetVariableLowerLimit(unsigned int ivar, double lower)
    {
      if (ivar > fLBounds.size())
	return false;
      fLBounds[ivar] = lower;
      fVariablesType[ivar] = 2;
      fWithBounds = true;
      return true;
    }

    bool TCMAESMinimizer::SetVariableUpperLimit(unsigned int ivar, double upper)
    {
      if (ivar > fUBounds.size())
	return false;
      fUBounds[ivar] = upper;
      fVariablesType[ivar] = 3;
      fWithBounds = true;
      return true;
    }

    bool TCMAESMinimizer::SetVariableLimits(unsigned int ivar, double lower, double upper)
    {
      if (ivar >= fLBounds.size() || ivar >= fUBounds.size())
	return false;
      fLBounds[ivar] = lower;
      fUBounds[ivar] = upper;
      fVariablesType[ivar] = 4;
      fWithBounds = true;
      return true;
    }

    bool TCMAESMinimizer::FixVariable(unsigned int ivar)
    {
      fFixedVariables.insert(std::pair<int,double>(ivar,fInitialX.at(ivar))); // XXX: sets initial variable.
    }

    bool TCMAESMinimizer::IsFixedVariable(unsigned int ivar) const
    {
      std::map<int,double>::const_iterator mit;
      if ((mit=fFixedVariables.find(ivar))!=fFixedVariables.end())
	return true;
      return false;
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
	std::cout << "fDim=" << fDim << " / fInitialX size=" << fInitialX.size() << std::endl;
	MATH_ERROR_MSG("TCMAESMinimizer::Minimize","Dimension larger than initial X size's");
	return false;
      }
      if (fDim < fInitialX.size()) {
	MATH_WARN_MSG("TCMAESMinimizer::Minimize","Dimension smaller than initial X size's");
      }

      ROOT::Math::IOptions *cmaesOpt = ROOT::Math::MinimizerOptions::FindDefault("cmaes");
      //std::cerr << "cmaesOpt ptr: " << cmaesOpt << std::endl;
      if (cmaesOpt)
	cmaesOpt->Print(std::cout);
      
      //TODO: phenotype / genotype.

      FitFunc ffit = [this](const double *x, const int N)
	{
	  return (*fObjFunc)(x);
	};

      // gradient function.
      std::cout << "fWithGradient=" << fWithGradient << std::endl;
      GradFunc gfit = nullptr;
      if (fWithGradient)
	{
	  gfit = [this](const double *x, const int N)
	    {
	      dVec grad(N);
	      fObjFuncGrad->Gradient(x,grad.data());
	      return grad;
	    };
	}
      
      double sigma0 = *std::min_element(fInitialSigma.begin(),fInitialSigma.end());
      int lambda = -1;
      int maxiter = fMaxIter > 0 ? fMaxIter : -1;
      int maxfevals = 100*fMaxCalls; // CMA-ES requires much more calls than Minuit.
      int noisy = 0;
      int nrestarts = -1;
      double ftarget = -1.0;
      std::string fplot;
      
      //TODO: set hyper-parameters according to IOptions object.
      if (cmaesOpt)
	{
	  cmaesOpt->GetValue("lambda",lambda);
	  cmaesOpt->GetValue("noisy",noisy);
	  cmaesOpt->GetValue("restarts",nrestarts);
	  cmaesOpt->GetValue("ftarget",ftarget);
	  cmaesOpt->GetValue("fplot",fplot);
	}
      
      if (gDebug > 0)
	{
	  std::cout << "Running CMA-ES with dim=" << fDim << " / sigma0=" << sigma0 << " / lambda=" << lambda << " / fTol=" << fTol << " / with_bounds=" << fWithBounds << " / maxiter=" << maxiter << " / maxfevals=" << maxfevals << std::endl;
	  std::cout << "x0=";
	  std::copy(fInitialX.begin(),fInitialX.end(),std::ostream_iterator<double>(std::cout," "));
	  std::cout << std::endl;
	}

      if (fWithBounds)
	{
	  Info("CMAESMinimizer","Minimizing with bounds");
	  //ProgressFunc<CMAParameters<>,CMASolutions> pfunc = [](const CMAParameters<> &cmaparams, const CMASolutions &cmasols) { return 0; };
	  GenoPheno<pwqBoundStrategy> gp(&fLBounds.front(),&fUBounds.front(),fDim);
	  CMAParameters<GenoPheno<pwqBoundStrategy>> cmaparams(fDim,&fInitialX.front(),sigma0,lambda,0,gp);
	  cmaparams._algo = fMinimizer;
	  if (gDebug > 0)
	    cmaparams._quiet = false;
	  else cmaparams._quiet = true;
	  for (auto mit=fFixedVariables.begin();mit!=fFixedVariables.end();mit++)
	    cmaparams.set_fixed_p((*mit).first,(*mit).second);
	  cmaparams.set_ftolerance(fTol);
	  cmaparams.set_max_iter(maxiter);
	  cmaparams.set_max_fevals(maxfevals);
	  if (noisy > 0)
	    cmaparams.set_noisy();
	  if (nrestarts > 0)
	    cmaparams.set_restarts(nrestarts);
	  if (ftarget > 0.0)
	    cmaparams.set_ftarget(ftarget);
	  cmaparams._fplot = fplot;
	  fCMAsols = libcmaes::cmaes<GenoPheno<pwqBoundStrategy>>(ffit,cmaparams);
	  fCMAparamsb = cmaparams;
	}
      else
	{
	  //ProgressFunc<CMAParameters<>,CMASolutions> pfunc = [](const CMAParameters<> &cmaparams, const CMASolutions &cmasols) { return 0; };
	  CMAParameters<> cmaparams(fDim,&fInitialX.front(),sigma0,lambda);
	  cmaparams._algo = fMinimizer;
	  if (gDebug > 0)
	    cmaparams._quiet = false;
	  else cmaparams._quiet = true;
	  for (auto mit=fFixedVariables.begin();mit!=fFixedVariables.end();mit++)
	    cmaparams.set_fixed_p((*mit).first,(*mit).second);
	  cmaparams.set_ftolerance(fTol);
	  cmaparams.set_max_iter(maxiter);
	  cmaparams.set_max_fevals(maxfevals);
	  if (noisy > 0)
	    cmaparams.set_noisy();
	  if (nrestarts > 0)
	    cmaparams.set_restarts(nrestarts);
	  if (ftarget > 0.0)
	    cmaparams.set_ftarget(ftarget);
	  cmaparams._fplot = fplot;
	  fCMAsols = libcmaes::cmaes<>(ffit,cmaparams);
	  fCMAparams = cmaparams;
	}
      Info("CMAESMinimizer","optimization status=%i",fCMAsols._run_status);
      if (fCMAsols._run_status >= 0)
	fStatus = 0; //TODO: convert so that to match that of Minuit2 ?
      else fStatus = 5;
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

    double TCMAESMinimizer::Edm() const
    {
      // XXX: cannot recompute it here as there's no access to the optimizer itself.
      //      instead this is returning the value computed at the end of last optimization call
      //      and stored within the solution object.
      return fCMAsols._edm;
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
      FitFunc ffit = [this](const double *x, const int N)
	{
	  return (*fObjFunc)(x);
	};
      
      // runopt is a flag which specifies if only lower or upper error needs to be run. TODO: support for one bound only in libcmaes ?
      int samplesize = 10;
      if (gDebug > 0)
	std::cerr << "Computing 'Minos' confidence interval with profile likelihood on parameter " << i << " / samplesize=" << samplesize << " / with_bounds=" << fWithBounds << std::endl;
      pli le;
      if (!fWithBounds)
	{
	  fCMAparams.set_automaxiter(true);
	  le = errstats<>::profile_likelihood(ffit,fCMAparams,fCMAsols,i,false,samplesize,fUp);
	}
      else
	{
	  fCMAparamsb.set_automaxiter(true);
	  le = errstats<GenoPheno<pwqBoundStrategy>>::profile_likelihood(ffit,fCMAparamsb,fCMAsols,i,false,samplesize);
	}
      errLow = le._errmin;
      errUp = le._errmax;
      return true;
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
