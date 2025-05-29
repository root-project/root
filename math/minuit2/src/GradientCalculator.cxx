#include "Minuit2/MnPrint.h"
#include "Minuit2/GradientCalculator.h"

namespace ROOT {

namespace Minuit2 {

/**
 * Enable parallelization of gradient calculation using OpenMP.
 * This is different from the default parallel mechanism elsewhere (IMT, threads, TBB, ...).
 * It can only be used to minimise thread-safe functions in Minuit2.
 * \param doParallel true to enable, false to disable.
 * \note Enabling this function does not guarantee that the derived gradient calculator class
 * has implemented a OMP-parallelized version of the code. For example, the numeric Hessian
 * computation (HessianGradientCalculator) does not make any use of OpenMP, only
 * Numerical2PGradientCalculator makes use of OMP pragmas at the moment.
 * \note If OPENMP is not available, i.e. ROOT was built without OpenMP support (minuit2_omp),
 * and an error is printed if doParallel=true; parallelization is disabled in any case.
 * \return false if OPENMP is not available and doParallel=true, otherwise it returns true.
 */
bool GradientCalculator::SetParallelOMP(bool doParallel)
{
#ifndef _OPENMP
   if (doParallel) {
      MnPrint print("GradientCalculator");
      print.Error("Minuit 2 was built without OpenMP support! Can't enable OMP-parallel gradients.");
      fDoParallelOMP = false;
      return false;
   } else {
      fDoParallelOMP = false;
      return true;
   }
#else
   fDoParallelOMP = doParallel;
   return true;
#endif
}

} // namespace Minuit2

} // namespace ROOT
