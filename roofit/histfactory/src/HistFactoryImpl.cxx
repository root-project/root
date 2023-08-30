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

#include <RooMsgService.h>
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
 * @param gammas   The gamma parameters to be configured.
 * @param sigmaRel The relative sigma values to be used for configuring the
 *                 limits and errors.
 * @param minSigma The minimum relative sigma threshold. If a relative sigma is
 *                 below this threshold, the gamma parameter is set to be
 *                 constant.
 */
void configureConstrainedGammas(RooArgList const &gammas, std::span<double> relSigmas, double minSigma)
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
         oocxcoutW(static_cast<TObject *>(nullptr), HistFactory)
            << "Warning: relative sigma " << sigmaRel << " for \"" << gamma.GetName() << "\" falls below threshold of "
            << minSigma << ". Setting: " << gamma.GetName() << " to constant" << std::endl;
         gamma.setConstant(true);
      }
   }
}

} // namespace Detail
} // namespace HistFactory
} // namespace RooStats
