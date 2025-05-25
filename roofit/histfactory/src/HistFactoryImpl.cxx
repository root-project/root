/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2023
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include <RooStats/HistFactory/Detail/HistFactoryImpl.h>

#include <RooStats/HistFactory/HistFactoryException.h>

#include <RooConstVar.h>
#include <RooGaussian.h>
#include <RooMsgService.h>
#include <RooPoisson.h>
#include <RooProduct.h>
#include <RooRealVar.h>

namespace RooStats {
namespace HistFactory {
namespace Detail {

/**
 * \brief Configure constrained gamma parameters for fitting.
 *
 * This function configures constrained gamma parameters for fitting. If a
 * given relative sigma is less than or equal to zero or below a threshold, the
 * gamma parameter is set to be constant. The function also sets reasonable
 * ranges for the gamma parameter and provides a reasonable starting point for
 * pre-fit errors.
 *
 * @param gammas    The gamma parameters to be configured.
 * @param relSigmas The relative sigma values to be used for configuring the
 *                  limits and errors.
 * @param minSigma  The minimum relative sigma threshold. If a relative sigma is
 *                  below this threshold, the gamma parameter is set to be
 *                  constant.
 */
void configureConstrainedGammas(RooArgList const &gammas, std::span<const double> relSigmas, double minSigma)
{
   assert(gammas.size() == relSigmas.size());

   for (std::size_t i = 0; i < gammas.size(); ++i) {
      auto &gamma = *static_cast<RooRealVar *>(gammas.at(i));
      double sigmaRel = relSigmas[i];

      // If the sigma is zero, the parameter might as well be constant
      if (sigmaRel <= 0) {
         gamma.setConstant(true);
         continue;
      }

      // Set reasonable ranges
      gamma.setMax(1. + 5. * sigmaRel);
      gamma.setMin(0.);
      // Set initial error too
      gamma.setError(sigmaRel);

      // Give reasonable starting point for pre-fit errors by setting it to the
      // absolute sigma Mostly useful for pre-fit plotting.
      // Note: in commit 2129c4d920 "[HF] Reduce verbosity of HistFactory."
      // from 2020, there was a check added to do this only for Gaussian
      // constrained parameters and for Poisson constrained parameters if they
      // are stat errors without any justification. In the ROOT 6.30
      // development cycle, this check got removed again to cause less surprise
      // to the user.
      gamma.setError(sigmaRel);

      // If the sigma value is less than a supplied threshold, set the variable to
      // constant
      if (sigmaRel < minSigma) {
         oocxcoutW(nullptr, HistFactory)
            << "Warning: relative sigma " << sigmaRel << " for \"" << gamma.GetName() << "\" falls below threshold of "
            << minSigma << ". Setting: " << gamma.GetName() << " to constant" << std::endl;
         gamma.setConstant(true);
      }
   }
}

// Take a RooArgList of RooAbsReals and create N constraint terms (one for
// each gamma) whose relative uncertainty is the value of the ith RooAbsReal
CreateGammaConstraintsOutput createGammaConstraints(RooArgList const &paramSet,
                                                    std::span<const double> relSigmas, double minSigma,
                                                    Constraint::Type type)
{
   CreateGammaConstraintsOutput out;

   // Check that there are N elements in the RooArgList
   if (relSigmas.size() != paramSet.size()) {
      std::cout << "Error: In createGammaConstraints, encountered bad number of relative sigmas" << std::endl;
      std::cout << "Given vector with " << relSigmas.size() << " bins,"
                << " but require exactly " << paramSet.size() << std::endl;
      throw hf_exc();
   }

   configureConstrainedGammas(paramSet, relSigmas, minSigma);

   for (std::size_t i = 0; i < paramSet.size(); ++i) {

      RooRealVar &gamma = static_cast<RooRealVar &>(paramSet[i]);

      oocxcoutI(nullptr, HistFactory)
         << "Creating constraint for: " << gamma.GetName() << ". Type of constraint: " << type << std::endl;

      const double sigmaRel = relSigmas[i];

      // If the sigma is <= 0,
      // do cont create the term
      if (sigmaRel <= 0) {
         oocxcoutI(nullptr, HistFactory)
            << "Not creating constraint term for " << gamma.GetName() << " because sigma = " << sigmaRel
            << " (sigma<=0)"
            << " (bin number = " << i << ")" << std::endl;
         continue;
      }

      // Make Constraint Term
      std::string constrName = std::string(gamma.GetName()) + "_constraint";
      std::string nomName = std::string("nom_") + gamma.GetName();

      if (type == Constraint::Gaussian) {

         // Type 1 : RooGaussian

         // Make sigma
         std::string sigmaName = std::string(gamma.GetName()) + "_sigma";
         auto constrSigma = std::make_unique<RooConstVar>(sigmaName.c_str(), sigmaName.c_str(), sigmaRel);

         // Make "observed" value
         auto constrNom = std::make_unique<RooRealVar>(nomName.c_str(), nomName.c_str(), 1.0, 0, 10);
         constrNom->setConstant(true);

         // Make the constraint:
         auto term = std::make_unique<RooGaussian>(constrName.c_str(), constrName.c_str(), *constrNom, gamma, *constrSigma);

         out.globalObservables.push_back(constrNom.get());

         term->addOwnedComponents(std::move(constrSigma));
         term->addOwnedComponents(std::move(constrNom));

         out.constraints.emplace_back(std::move(term));
      } else if (type == Constraint::Poisson) {

         // this is correct Poisson equivalent to a Gaussian with mean 1 and stdev sigma
         const double tau = 1. / (sigmaRel * sigmaRel);

         // Make nominal "observed" value
         auto constrNom = std::make_unique<RooRealVar>(nomName.c_str(), nomName.c_str(), tau);
         constrNom->setMin(0);
         constrNom->setConstant(true);

         // Make the scaling term
         std::string scalingName = std::string(gamma.GetName()) + "_tau";
         auto poissonScaling = std::make_unique<RooConstVar>(scalingName.c_str(), scalingName.c_str(), tau);

         // Make mean for scaled Poisson
         std::string poisMeanName = std::string(gamma.GetName()) + "_poisMean";
         auto constrMean = std::make_unique<RooProduct>(poisMeanName.c_str(), poisMeanName.c_str(), gamma, *poissonScaling);

         // Type 2 : RooPoisson
         auto term = std::make_unique<RooPoisson>(constrName.c_str(), constrName.c_str(), *constrNom, *constrMean);
         term->setNoRounding(true);

         out.globalObservables.push_back(constrNom.get());

         term->addOwnedComponents(std::move(poissonScaling));
         term->addOwnedComponents(std::move(constrMean));
         term->addOwnedComponents(std::move(constrNom));

         out.constraints.emplace_back(std::move(term));
      } else {

         std::cout << "Error: Did not recognize Stat Error constraint term type: " << type
                   << " for : " << gamma.GetName() << std::endl;
         throw hf_exc();
      }
   } // end loop over parameters

   return out;
}

} // namespace Detail
} // namespace HistFactory
} // namespace RooStats
