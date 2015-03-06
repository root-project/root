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

#include <fstream>

using namespace libcmaes;

namespace ROOT
{
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
      fMinimizer = algoname;
    }

    /*TCMAESMinimizer::TCMAESMinimizer(const TCMAESMinimizer &m)
      :Minimizer(),fDim(0),fFreeDim(0),fWithBounds(false),fWithGradient(false)
    {
    }*/

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
      fCMAparams = CMAParameters<GenoPheno<NoBoundStrategy,NoScalingStrategy>>();
      fCMAparams_b = CMAParameters<GenoPheno<pwqBoundStrategy,NoScalingStrategy>>();
      fCMAparams_l = CMAParameters<GenoPheno<NoBoundStrategy,linScalingStrategy>>();
      fCMAparams_lb = CMAParameters<GenoPheno<pwqBoundStrategy,linScalingStrategy>>();
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
      if (ivar >= fInitialX.size())
	return false;
      fFixedVariables.insert(std::pair<int,double>(ivar,fInitialX.at(ivar))); // XXX: sets initial variable.
      return true;
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
      for (unsigned int i=0;i<fNames.size();i++)
	if (fNames.at(i) == name)
	  return i;
      return -1;
    }

    template <class TGenoPheno>
    void TCMAESMinimizer::SetMParameters(CMAParameters<TGenoPheno> &cmaparams,
					 const int &maxiter, const int &maxfevals,
					 const int &noisy, const int &nrestarts,
					 const double &ftarget,
					 const std::string &fplot,
					 const bool &withnumgradient,
					 const bool &mtfeval,
					 const bool &quiet,
					 const int &elitist,
					 const bool &uh)
    {
      cmaparams.set_str_algo(fMinimizer);
      if (gDebug > 0 || !quiet)
	cmaparams.set_quiet(false);
      else cmaparams.set_quiet(true);
      for (auto mit=fFixedVariables.begin();mit!=fFixedVariables.end();mit++)
	cmaparams.set_fixed_p((*mit).first,(*mit).second);
      cmaparams.set_edm(true); // always activate EDM computation.
      cmaparams.set_ftolerance(Tolerance());
      cmaparams.set_max_iter(maxiter);
      cmaparams.set_max_fevals(maxfevals);
      if (noisy > 0)
	cmaparams.set_noisy();
      if (nrestarts > 0)
	cmaparams.set_restarts(nrestarts);
      if (ftarget > 0.0)
	cmaparams.set_ftarget(ftarget);
      cmaparams.set_fplot(fplot);
      cmaparams.set_gradient(withnumgradient);
      cmaparams.set_mt_feval(mtfeval);
      cmaparams.set_elitism(elitist);
      cmaparams.set_uh(uh);
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
	std::cout << "fDim=" << fDim << " / fInitialX size=" << fInitialX.size() << " / freeDim=" << fFreeDim << std::endl;
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
      
      FitFunc ffit = [this](const double *x, const int N)
	{
	  /*std::copy(x,x+N,std::ostream_iterator<double>(std::cout," "));
	    std::cout << std::endl;*/
	  (void)N;
	  return (*fObjFunc)(x);
	};
      
      // gradient function.
      //std::cout << "fWithGradient=" << fWithGradient << std::endl;
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

      //debug
      /*if (fWithBounds)
	{
	  std::cout << "bounds:\n";
	  std::copy(fLBounds.begin(),fLBounds.end(),std::ostream_iterator<double>(std::cout," "));
	  std::cout << std::endl;
	  std::copy(fUBounds.begin(),fUBounds.end(),std::ostream_iterator<double>(std::cout," "));
	  std::cout << std::endl;
	  }*/
      //debug

      double sigma0 = *std::min_element(fInitialSigma.begin(),fInitialSigma.end());
      double sigma0scaled = 1e-1; // default value.
      if (!fWithLinearScaling)
	sigma0scaled = sigma0;
      dVec vscaling = dVec::Constant(fDim,1.0);
      for (size_t i=0;i<fInitialSigma.size();i++)
	vscaling(i) /= fInitialSigma.at(i);
      //std::cerr << "lscaling=" << vscaling.transpose() << std::endl;
      dVec vshift = dVec::Constant(fDim,0.0);

      int lambda = -1;
      int maxiter = MaxIterations() > 0 ? MaxIterations() : -1;
      int maxfevals = 100*MaxFunctionCalls(); // CMA-ES requires much more calls than Minuit. //TODO: set into options...
      int noisy = 0;
      int nrestarts = -1;
      double ftarget = -1.0;
      std::string fplot;
      int withnumgradient = 0; // whether to use numerical gradient injection.
      int mtfeval = 0; // parallel execution of objective function
      int quiet = 0;
      int seed = 0;
      int elitist = 0; // elitism: forces best solution in various manners
      int uh = 0; // uncertainty handling, for noisy functions
      
      // set hyper-parameters according to IOptions object.
      if (cmaesOpt)
	{
	  cmaesOpt->GetValue("sigma",sigma0scaled);
	  cmaesOpt->GetValue("lambda",lambda);
	  cmaesOpt->GetValue("noisy",noisy);
	  cmaesOpt->GetValue("restarts",nrestarts);
	  cmaesOpt->GetValue("ftarget",ftarget);
	  cmaesOpt->GetValue("fplot",fplot);
	  cmaesOpt->GetValue("lscaling",fWithLinearScaling);
	  cmaesOpt->GetValue("numgradient",withnumgradient);
	  cmaesOpt->GetValue("mt_feval",mtfeval);
	  cmaesOpt->GetValue("quiet",quiet);
	  cmaesOpt->GetValue("seed",seed);
	  cmaesOpt->GetValue("elitist",elitist);
	  cmaesOpt->GetValue("uh",uh);
	}
      
      if (gDebug > 0)
	{
	  std::cout << "Running CMA-ES with dim=" << fDim << " / sigma0=" << sigma0scaled << " / lambda=" << lambda << " / fTol=" << Tolerance() << " / with_bounds=" << fWithBounds << " / with_gradient=" << fWithGradient << " / linear_scaling=" << fWithLinearScaling << " / maxiter=" << maxiter << " / maxfevals=" << maxfevals << " / mtfeval=" << mtfeval << std::endl;
	  std::cout << "x0=";
	  std::copy(fInitialX.begin(),fInitialX.end(),std::ostream_iterator<double>(std::cout," "));
	  std::cout << std::endl;
	}

      if (fWithLinearScaling)
	{
	  if (fWithBounds)
	    {
	      Info("CMAESMinimizer","Minimizing with bounds and linear scaling");
	      GenoPheno<pwqBoundStrategy,linScalingStrategy> gp(vscaling,vshift,&fLBounds.front(),&fUBounds.front());
	      CMAParameters<GenoPheno<pwqBoundStrategy,linScalingStrategy>> cmaparams(fDim,&fInitialX.front(),sigma0scaled,lambda,seed,gp);
	      SetMParameters(cmaparams,maxiter,maxfevals,noisy,nrestarts,ftarget,fplot,withnumgradient,mtfeval,quiet,elitist,uh);
	      fCMAsols = libcmaes::cmaes<GenoPheno<pwqBoundStrategy,linScalingStrategy>>(ffit,cmaparams,CMAStrategy<CovarianceUpdate,GenoPheno<pwqBoundStrategy,linScalingStrategy>>::_defaultPFunc,fWithGradient?gfit:nullptr);
	      fCMAparams_lb = cmaparams;
	    }
	  else
	    {
	      Info("CMAESMinimizer","Minimizing with linear scaling");
	      GenoPheno<NoBoundStrategy,linScalingStrategy> gp(vscaling,vshift);
	      CMAParameters<GenoPheno<NoBoundStrategy,linScalingStrategy>> cmaparams(fDim,&fInitialX.front(),sigma0scaled,lambda,seed,gp);
	      SetMParameters(cmaparams,maxiter,maxfevals,noisy,nrestarts,ftarget,fplot,withnumgradient,mtfeval,quiet,elitist,uh);
	      fCMAsols = libcmaes::cmaes<GenoPheno<NoBoundStrategy,linScalingStrategy>>(ffit,cmaparams,CMAStrategy<CovarianceUpdate,GenoPheno<NoBoundStrategy,linScalingStrategy>>::_defaultPFunc,fWithGradient?gfit:nullptr);
	      fCMAparams_l = cmaparams;
	    }
	}
      else
	{
	  if (fWithBounds)
	    {
	      Info("CMAESMinimizer","Minimizing with bounds");
	      GenoPheno<pwqBoundStrategy,NoScalingStrategy> gp(&fLBounds.front(),&fUBounds.front(),fDim);
	      CMAParameters<GenoPheno<pwqBoundStrategy,NoScalingStrategy>> cmaparams(fDim,&fInitialX.front(),sigma0scaled,lambda,seed,gp);
	      SetMParameters(cmaparams,maxiter,maxfevals,noisy,nrestarts,ftarget,fplot,withnumgradient,mtfeval,quiet,elitist,uh);
	      fCMAsols = libcmaes::cmaes<GenoPheno<pwqBoundStrategy,NoScalingStrategy>>(ffit,cmaparams,CMAStrategy<CovarianceUpdate,GenoPheno<pwqBoundStrategy,NoScalingStrategy>>::_defaultPFunc,fWithGradient?gfit:nullptr);
	      fCMAparams_b = cmaparams;
	    }
	  else
	    {
	      Info("CMAESMinimizer","Minimizing without bounds or linear scaling");
	      CMAParameters<GenoPheno<NoBoundStrategy,NoScalingStrategy>> cmaparams(fDim,&fInitialX.front(),sigma0scaled,lambda,seed);
	      SetMParameters(cmaparams,maxiter,maxfevals,noisy,nrestarts,ftarget,fplot,withnumgradient,mtfeval,quiet,elitist,uh);
	      fCMAsols = libcmaes::cmaes<GenoPheno<NoBoundStrategy,NoScalingStrategy>>(ffit,cmaparams,CMAStrategy<CovarianceUpdate,GenoPheno<NoBoundStrategy,NoScalingStrategy>>::_defaultPFunc,fWithGradient?gfit:nullptr);
	      fCMAparams = cmaparams;
	    }
	}
      Info("CMAESMinimizer","optimization status=%i",fCMAsols.run_status());
      if (fCMAsols.edm() > 10*Tolerance()) // XXX: max edm seems to be left to each minimizer's internal implementation...
	fStatus = 3;
      else if (fCMAsols.run_status() == 0 || fCMAsols.run_status() == 1)
	fStatus = 0;
      else if (fCMAsols.run_status() == 7 || fCMAsols.run_status() == 9)
	fStatus = 4; // reached budget limit.
      else fStatus = 5;
      return fCMAsols.run_status() >= 0; // above 0 are partial successes at worst.
    }

    double TCMAESMinimizer::MinValue() const
    {
      return fCMAsols.best_candidate().get_fvalue();
    }

    const double* TCMAESMinimizer::X() const
    {
      fValues.clear();
      Candidate bc = fCMAsols.best_candidate();

      dVec x;
      if (fWithLinearScaling)
	{
	  if (fWithBounds)
	    x = bc.get_x_pheno_dvec<GenoPheno<pwqBoundStrategy,linScalingStrategy>>(fCMAparams_lb);
	  else x = bc.get_x_pheno_dvec<GenoPheno<NoBoundStrategy,linScalingStrategy>>(fCMAparams_l);
	}
      else
	{
	  if (fWithBounds)
	    x = bc.get_x_pheno_dvec<GenoPheno<pwqBoundStrategy,NoScalingStrategy>>(fCMAparams_b);
	  else x = bc.get_x_dvec();
	}
      for (int i=0;i<(int)fDim;i++)
	fValues.push_back(x(i));
      return &fValues.front();
    }

    double TCMAESMinimizer::Edm() const
    {
      // XXX: cannot recompute it here as there's no access to the optimizer itself.
      //      instead this is returning the value computed at the end of last optimization call
      //      and stored within the solution object.
      return fCMAsols.edm();
    }
    
    const double* TCMAESMinimizer::Errors() const
    {
      fErrors.clear();
      dVec vgdiag;
      if (fWithLinearScaling)
	{
	  if (fWithBounds)
	    {
	      vgdiag = fCMAparams_lb.get_gp().pheno(dVec(fCMAsols.sigma()*fCMAsols.cov_ref().diagonal()));
	    }
	  else
	    {
	      vgdiag = fCMAparams_l.get_gp().pheno(dVec(fCMAsols.sigma()*fCMAsols.cov_ref().diagonal()));
	    }
	}
      else if (fWithBounds)
	{
	  vgdiag = fCMAparams_b.get_gp().pheno(dVec(fCMAsols.sigma()*fCMAsols.cov_ref().diagonal()));
	}
      else vgdiag = fCMAsols.sigma()*fCMAsols.cov_ref().diagonal();
      for (int i=0;i<(int)fDim;i++)
	fErrors.push_back(std::sqrt(std::abs(vgdiag(i)))); // abs for numerical errors that bring the sqrt below 0.
      return &fErrors.front();
    }
    
    unsigned int TCMAESMinimizer::NCalls() const
    {
      return fCMAsols.nevals();
    }

    double TCMAESMinimizer::CovMatrix(unsigned int i, unsigned int j) const
    {
      return fCMAsols.cov_ref()(i,j);
    }

    bool TCMAESMinimizer::GetCovMatrix(double *cov) const
    {
      std::copy(fCMAsols.cov_data(),fCMAsols.cov_data()+fCMAsols.cov_ref().size(),cov);
      return true;
    }

    double TCMAESMinimizer::Correlation(unsigned int i, unsigned int j) const
    {
      return std::sqrt(std::abs(fCMAsols.cov_ref()(i,i)*fCMAsols.cov_ref()(j,j)));
    }

    double TCMAESMinimizer::GlobalCC(unsigned int i) const
    {
      // original Minuit paper says:
      // \rho_k^2 = 1 - [C_{kk}C_{kk}^{-1}]^{-1}
      if (fGlobalCC.empty()) // need to pre-compute the vector coefficient
	{
	  dMat covinv = fCMAsols.cov_ref().inverse();
	  for (int i=0;i<covinv.rows();i++)
	    {
	      double denom = covinv(i,i)*fCMAsols.cov_ref()(i,i);
	      if (denom < 1.0 && denom > 0.0)
		fGlobalCC.push_back(0.0);
	      else fGlobalCC.push_back(std::sqrt(1.0 - 1.0/denom));
	    }
	}
      return fGlobalCC.at(i);
    }

    bool TCMAESMinimizer::GetMinosError(unsigned int i, double &errLow, double &errUp, int j)
    {
      (void)j;
      FitFunc ffit = [this](const double *x, const int N)
	{
	  (void)N;
	  return (*fObjFunc)(x);
	};
      
      // runopt is a flag which specifies if only lower or upper error needs to be run. TODO: support for one bound only in libcmaes ?
      int samplesize = 10;
      if (gDebug > 0)
	std::cerr << "Computing 'Minos' confidence interval with profile likelihood on parameter " << i << " / samplesize=" << samplesize << " / with_bounds=" << fWithBounds << std::endl;
      pli le;
      if (fWithLinearScaling)
	{
	  if (!fWithBounds)
	    {
	      le = errstats<GenoPheno<NoBoundStrategy,linScalingStrategy>>::profile_likelihood(ffit,fCMAparams_l,fCMAsols,i,false,samplesize,ErrorDef(),100);
	    }
	  else
	    {
	      le = errstats<GenoPheno<pwqBoundStrategy,linScalingStrategy>>::profile_likelihood(ffit,fCMAparams_lb,fCMAsols,i,false,samplesize);
	    }
	}
      else
	{
	  if (!fWithBounds)
	    {
	      le = errstats<GenoPheno<NoBoundStrategy,NoScalingStrategy>>::profile_likelihood(ffit,fCMAparams,fCMAsols,i,false,samplesize,ErrorDef());
	    }
	  else
	    {
	      le = errstats<GenoPheno<pwqBoundStrategy,NoScalingStrategy>>::profile_likelihood(ffit,fCMAparams_b,fCMAsols,i,false,samplesize);
	    }
	}
      errLow = le.get_err_min();
      errUp = le.get_err_max();
      return true;
    }
    
    bool TCMAESMinimizer::Scan(unsigned int i, unsigned int &nstep, double *x, double *y, double xmin, double xmax)
    {
      std::vector<std::pair<double,double>> result;
      std::vector<double> params = fValues;
      double amin = MinValue();
      result.push_back(std::pair<double,double>(params[i],amin));
      
      double low=xmin, high=xmax;
      if (low <= high && nstep-1 >= 2)
	{
	  if (low == 0 && high == 0)
	    {
	      low = fValues[i] - 2.0*fErrors.at(i);
	      high = fValues[i] + 2.0*fErrors.at(i); 
	    }
	  
	  if (low == 0 && high == 0 
	      && (fLBounds[i] > -std::numeric_limits<double>::max()
		  || fUBounds[i] < std::numeric_limits<double>::max()))
	    {
	      if (fLBounds[i] > -std::numeric_limits<double>::max())
		low = fLBounds[i];
	      if (fUBounds[i] < std::numeric_limits<double>::max())
		high = fUBounds[i];
	    }
	  
	  if (fLBounds[i] > -std::numeric_limits<double>::max()
	      || fUBounds[i] < std::numeric_limits<double>::max())
	    {
	      if (fLBounds[i] > -std::numeric_limits<double>::max())
		low = std::max(low,fLBounds[i]);
	      if (fUBounds[i] < std::numeric_limits<double>::max())
		high = std::min(high,fUBounds[i]);
	    }
	  
	  double x0 = low;
	  double stp = (high-low) / static_cast<double>(nstep-2);
	  for (unsigned int j=0;j<nstep-1;j++)
	    {
	      params[i] = x0 + j*stp;
	      double fval = (*fObjFunc)(&params.front());
	      result.push_back(std::pair<double,double>(params[i],fval));
	    }
	}
      
      for (int s=0;s<nstep;s++)
	{
	  x[s] = result[s].first;
	  y[s] = result[s].second;
	}
      return true;
    }

    bool TCMAESMinimizer::Contour(unsigned int i, unsigned int j, unsigned int &npoints, double *xi, double *xj)
    {
      FitFunc ffit = [this](const double *x, const int N)
	{
	  (void)N;
	  return (*fObjFunc)(x);
	};
      
      contour ct;
      if (fWithLinearScaling)
	{
	  if (!fWithBounds)
	    {
	      ct = errstats<GenoPheno<NoBoundStrategy,linScalingStrategy>>::contour_points(ffit,i,j,npoints,ErrorDef(),
											   fCMAparams_l,fCMAsols,0.1,100);
	    }
	  else
	    {
	      ct = errstats<GenoPheno<pwqBoundStrategy,linScalingStrategy>>::contour_points(ffit,i,j,npoints,ErrorDef(),
											    fCMAparams_lb,fCMAsols,0.1,100);
	    }
	}
      else
	{
	  if (!fWithBounds)
	    {
	      ct = errstats<GenoPheno<NoBoundStrategy,NoScalingStrategy>>::contour_points(ffit,i,j,npoints,ErrorDef(),
											  fCMAparams,fCMAsols,0.1,100);
	    }
	  else
	    {
	      ct = errstats<GenoPheno<pwqBoundStrategy,NoScalingStrategy>>::contour_points(ffit,i,j,npoints,ErrorDef(),
											   fCMAparams_b,fCMAsols,0.1,100);
	    }
	}
      for (size_t i=0;i<ct._points.size();i++)
	{
	  xi[i] = ct._points.at(i).first;
	  xj[i] = ct._points.at(i).second;
	}
      return true;
    }

    void TCMAESMinimizer::PrintResults()
    {
      std::cout << "CMAESMinimizer : Valid minimum - status = " << fStatus << std::endl;
      std::cout << "FVAL  = " << MinValue() << std::endl;
      std::cout << "Nfcn  = " << NCalls() << std::endl;
      std::cout << "Edm   = " << Edm() << std::endl;
      std::map<int,double>::const_iterator mit;
      for (unsigned int i=0;i<fDim;i++)
	{
	  std::cout << fNames.at(i) << "\t  = " << X()[i] << "\t";
	  std::cout << "+/-  " << fErrors.at(i);
	  if ((mit=fFixedVariables.find(i))!=fFixedVariables.end())
	    std::cout << "\t(fixed)";
	  else if (fVariablesType.at(i) > 1)
	    std::cout << "\t(limited)";
	  std::cout << std::endl;	  
	}
    }
    
  }
}
