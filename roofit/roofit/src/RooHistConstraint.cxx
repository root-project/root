/*
 * Project: RooFit
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

/** \class RooHistConstraint
 * \ingroup Roofit
 * The RooHistConstraint implements constraint terms for a binned PDF with statistical uncertainties.
 * Following the Barlow-Beeston method, it adds Poisson constraints for each bin that
 * constrain the statistical uncertainty of the template histogram.
 *
 * It can therefore be used to estimate the Monte Carlo uncertainty of a fit.
 *
 * Check also the tutorial rf709_BarlowBeeston.C
 *
 */

#include <RooHistConstraint.h>

#include <RooParamHistFunc.h>
#include <RooRealVar.h>

#include <Math/PdfFuncMathCore.h>


////////////////////////////////////////////////////////////////////////////////
/// Create a new RooHistConstraint.
/// \param[in] name Name of the PDF. This is used to identify it in a likelihood model.
/// \param[in] title Title for plotting etc.
/// \param[in] phfSet Set of parametrised histogram functions (RooParamHistFunc).
/// \param[in] threshold Threshold (bin content) up to which statistical uncertainties are taken into account.
RooHistConstraint::RooHistConstraint(const char *name, const char *title,
    const RooArgSet& phfSet, int threshold) :
  RooAbsPdf(name,title),
  _gamma("gamma","gamma",this),
  _nominal("nominal","nominal",this),
  _relParam(true)
{
  // Implementing constraint on sum of RooParamHists
  //
  // Step 1 - Create new gamma parameters for sum
  // Step 2 - Replace entries in gamma listproxy of components with new sum components
  // Step 3 - Implement constraints in terms of gamma sum parameters


  if (phfSet.size()==1) {

    auto phf = dynamic_cast<RooParamHistFunc*>(phfSet.first()) ;

    if (!phf) {
      coutE(InputArguments) << "RooHistConstraint::ctor(" << GetName()
                 << ") ERROR: input object must be a RooParamHistFunc" << std::endl ;
      throw std::string("RooHistConstraint::ctor ERROR incongruent input arguments") ;
    }

    // Now populate nominal with parameters
    for (int i=0 ; i<phf->_dh.numEntries() ; i++) {
      phf->_dh.get(i) ;
      if (phf->_dh.weight()<threshold && phf->_dh.weight() != 0.) {
        const char* vname = Form("%s_nominal_bin_%i",GetName(),i) ;
        auto var = std::make_unique<RooRealVar>(vname,vname,0,1.E30);
        var->setVal(phf->_dh.weight()) ;
        var->setConstant(true);

        auto gamma = static_cast<RooRealVar*>(phf->_p.at(i));
        if (var->getVal() > 0.0) {
          gamma->setConstant(false);
        }

        _nominal.addOwned(std::move(var)) ;
        _gamma.add(*gamma) ;
      }
    }

    return ;
  }



  int nbins(-1) ;
  std::vector<RooParamHistFunc*> phvec ;
  RooArgSet gammaSet ;
  std::string bin0_name ;
  for (const auto arg : phfSet) {

    auto phfComp = dynamic_cast<RooParamHistFunc*>(arg) ;
    if (phfComp) {
      phvec.push_back(phfComp) ;
      if (nbins==-1) {
        nbins = phfComp->_p.size() ;
        bin0_name = phfComp->_p.at(0)->GetName() ;
        gammaSet.add(phfComp->_p) ;
      } else {
        if (int(phfComp->_p.size())!=nbins) {
          coutE(InputArguments) << "RooHistConstraint::ctor(" << GetName()
                << ") ERROR: incongruent input arguments: all input RooParamHistFuncs should have same #bins" << std::endl ;
          throw std::string("RooHistConstraint::ctor ERROR incongruent input arguments") ;
        }
        if (bin0_name != phfComp->_p.at(0)->GetName()) {
          coutE(InputArguments) << "RooHistConstraint::ctor(" << GetName()
                << ") ERROR: incongruent input arguments: all input RooParamHistFuncs should have the same bin parameters.\n"
                << "Previously found " << bin0_name << ", now found " << phfComp->_p.at(0)->GetName() << ".\n"
                << "Check that the right RooParamHistFuncs have been passed to this RooHistConstraint." << std::endl;
          throw std::string("RooHistConstraint::ctor ERROR incongruent input arguments") ;
        }

      }
    } else {
      coutW(InputArguments) << "RooHistConstraint::ctor(" << GetName()
                 << ") WARNING: ignoring input argument " << arg->GetName() << " which is not of type RooParamHistFunc" << std::endl;
    }
  }

  _gamma.add(gammaSet) ;

  // Now populate nominal and nominalErr with parameters
  for (int i=0 ; i<nbins ; i++) {

    double sumVal(0) ;
    for (const auto phfunc : phvec) {
      sumVal += phfunc->getNominal(i);
    }

    if (sumVal<threshold && sumVal != 0.) {

      const char* vname = Form("%s_nominal_bin_%i",GetName(),i) ;
      auto var = std::make_unique<RooRealVar>(vname,vname,0,1000);

      double sumVal2(0) ;
      for(auto const& elem : phvec) {
        sumVal2 += elem->getNominal(i) ;
      }
      var->setVal(sumVal2) ;
      var->setConstant(true) ;

      vname = Form("%s_nominal_error_bin_%i",GetName(),i) ;
      //RooRealVar* vare = new RooRealVar(vname,vname,0,1000) ;

      //double sumErr2(0) ;
      //for(auto const& elem : phvec) {
        //sumErr2 += std::pow(elem->getNominalError(i),2) ;
      //}
      //vare->setVal(sqrt(sumErr2)) ;
      //vare->setConstant(true) ;

      _nominal.addOwned(std::move(var));
      //      _nominalErr.add(*vare) ;

      (static_cast<RooRealVar*>(_gamma.at(i)))->setConstant(false) ;

    }
  }
}

////////////////////////////////////////////////////////////////////////////////

 RooHistConstraint::RooHistConstraint(const RooHistConstraint& other, const char* name) :
   RooAbsPdf(other,name),
   _gamma("gamma",this,other._gamma),
   _nominal("nominal",this,other._nominal),
   _relParam(other._relParam)
 {
 }

////////////////////////////////////////////////////////////////////////////////

 double RooHistConstraint::evaluate() const
 {
   double prod(1.0);

   for (unsigned int i=0; i < _nominal.size(); ++i) {
     const auto& gamma = static_cast<const RooAbsReal&>(_gamma[i]);
     const auto& nominal = static_cast<const RooAbsReal&>(_nominal[i]);
     double gammaVal = gamma.getVal();
     const int nomVal = static_cast<int>(nominal.getVal());

     if (_relParam) {
       gammaVal *= nomVal;
     }

     if (gammaVal>0) {
       const double pois = ROOT::Math::poisson_pdf(nomVal, gammaVal);
       prod *= pois;
     } else if (nomVal > 0) {
       coutE(Eval) << "ERROR in RooHistConstraint: gamma=0 and nom>0" << std::endl;
     }
   }

   return prod;
 }

////////////////////////////////////////////////////////////////////////////////

double RooHistConstraint::getLogVal(const RooArgSet* /*set*/) const
{
   double sum = 0.;
   for (unsigned int i=0; i < _nominal.size(); ++i) {
     const auto& gamma = static_cast<const RooAbsReal&>(_gamma[i]);
     const auto& nominal = static_cast<const RooAbsReal&>(_nominal[i]);
     double gammaVal = gamma.getVal();
     const int nomVal = static_cast<int>(nominal.getVal());

     if (_relParam) {
       gammaVal *= nomVal;
     }

     if (gammaVal>0) {
       const double logPoisson = nomVal * log(gammaVal) - gammaVal - std::lgamma(nomVal + 1);
       sum += logPoisson ;
     } else if (nomVal > 0) {
       coutE(Eval) << "ERROR in RooHistConstraint: gamma=0 and nom>0" << std::endl;
     }
   }

   return sum ;
}
