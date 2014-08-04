// @(#)root/mathcore:$Id$
// Author: L. Moneta Tue Nov 28 10:52:47 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class FitUtil

#ifdef ROOT_FIT_PARALLEL

#include "Fit/FitUtilParallel.h"

#include "Fit/BinData.h"
#include "Fit/UnBinData.h"
#include "Fit/FitUtil.h"

#include "Math/IParamFunction.h"

int ncalls = 0;

#include <limits>
#include <cmath>
#include <numeric>

//#define DEBUG
#ifdef DEBUG
#include <iostream>
#endif

#ifdef USE_PTHREAD

#include <pthread.h>


#define NUMBER_OF_THREADS 2

#else
#include <omp.h>
// maximum number of threads for the array
#define MAX_THREAD 8

#endif

namespace ROOT {

   namespace Fit {

      namespace FitUtilParallel {

#ifdef USE_PTHREAD

class ThreadData {

public:

   ThreadData() :
      fBegin(0),fEnd(0) ,
      fData(0),
      fFunc(0)
   {}

   ThreadData(unsigned int i1, unsigned int i2, const BinData & data, IModelFunction &func) :
      fBegin(i1), fEnd(i2),
      fData(&data),
      fFunc(&func)
   {}

   void Set(unsigned int nrej, double sum) {
      fNRej = nrej;
      fSum = sum;
   }

   const BinData & Data() const { return *fData; }

   IModelFunction & Func() { return *fFunc; }

   double Sum() const { return fSum; }

   unsigned int NRej() const { return fNRej; }

   unsigned int Begin() const { return fBegin; }
   unsigned int End() const { return fEnd; }

private:

   const unsigned int fBegin;
   const unsigned int fEnd;
   const BinData * fData;
   IModelFunction * fFunc;
   double fSum;
   unsigned int fNRej;
};


// function used by the threads
void *EvaluateResidual(void * ptr) {

   ThreadData * t = (ThreadData *) ptr;

   unsigned int istart = t->Begin();
   unsigned int iend = t->End();
   double chi2 = 0;
   unsigned int nRejected = 0;
   const int nthreads = NUMBER_OF_THREADS;

   const BinData & data = t->Data();
   IModelFunction & func = t->Func();
   for (unsigned int i = istart; i < iend; i+=nthreads) {
      const double * x = data.Coords(i);
      double y = data.Value(i);
      double invError = data.InvError(i);
      double fval = 0;
      fval = func ( x );

// #ifdef DEBUG
//       std::cout << x[0] << "  " << y << "  " << 1./invError << " params : ";
//       for (int ipar = 0; ipar < func.NPar(); ++ipar)
//          std::cout << p[ipar] << "\t";
//       std::cout << "\tfval = " << fval << std::endl;
// #endif

      // avoid singularity in the function (infinity and nan ) in the chi2 sum
      // eventually add possibility of excluding some points (like singularity)
      if (fval > - std::numeric_limits<double>::max() && fval < std::numeric_limits<double>::max() ) {
         // calculat chi2 point
         double tmp = ( y -fval )* invError;
         chi2 += tmp*tmp;
      }
      else
         nRejected++;

   }

#ifdef DEBUG
   std::cout << "end loop " << istart << "  " << iend << " chi2 = " << chi2 << " nrej " << nRejected << std::endl;
#endif
   t->Set(nRejected,chi2);
   return 0;
}

double EvaluateChi2(IModelFunction & func, const BinData & data, const double * p, unsigned int & nPoints) {
   // evaluate the chi2 given a  function reference  , the data and returns the value and also in nPoints
   // the actual number of used points

   const int nthreads = NUMBER_OF_THREADS;

   unsigned int n = data.Size();

#ifdef DEBUG
   std::cout << "\n\nFit data size = " << n << std::endl;
#endif

   func.SetParameters(p);

   // start the threads
   pthread_t   thread[nthreads];
   ThreadData *  td[nthreads];
   unsigned int istart = 0;
   for (int ithread = 0; ithread < nthreads; ++ithread) {
//       int n_th = n/nthreads;
//       if (ithread == 0 ) n_th += n%nthreads;
//       int iend = istart + n_th;
      int iend = n;
      istart = ithread;
      td[ithread] = new ThreadData(istart,iend,data,func);
      pthread_create(&thread[ithread], NULL, EvaluateResidual, td[ithread]);
      //istart = iend;
   }

   for (int ithread = 0; ithread < nthreads; ++ithread)
      pthread_join(thread[ithread], NULL);

   // sum finally the results of the various threads

   double chi2 = 0;
   int nRejected = 0;
   for (int ithread = 0; ithread < nthreads; ++ithread) {
      nRejected += td[ithread]->NRej();
      chi2 += td[ithread]->Sum();
      delete td[ithread];
   }


#ifdef DEBUG
   std::cout << "chi2 = " << chi2 << " n = " << nRejected << std::endl;
#endif

   nPoints = n - nRejected;
   return chi2;

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// use open MP
#else

double EvaluateChi2(IModelFunction & func, const BinData & data, const double * p, unsigned int & nPoints) {
   return FitUtil::EvaluateChi2(func,data,p,nPoints);
}


// use openMP (log-likelihood calculation)

inline double EvalLogF(double fval) {
   // evaluate the log with a protections against negative argument to the log
   // smooth linear extrapolation below function values smaller than  epsilon
   // (better than a simple cut-off)
   const static double epsilon = 2.*std::numeric_limits<double>::min();
   if(fval<= epsilon)
      return fval/epsilon + std::log(epsilon) - 1;
   else
      return std::log(fval);
}


double EvaluateLogL(IModelFunction & func, const UnBinData & data, const double * p, unsigned int &nPoints) {
   // evaluate the LogLikelihood

   unsigned int n = data.Size();

#ifdef DEBUG
   std::cout << "\n\nFit data size = " << n << std::endl;
   //std::cout << "func pointer is " << typeid(func).name() << std::endl;
   std::cout << "\tpar = [ " << func.NPar() << " ] =  ";
   for (unsigned int ipar = 0; ipar < func.NPar(); ++ipar)
      std::cout << p[ipar] << ", ";
   std::cout <<"---------------------------\n";
#endif

   double logl = 0;
   int nRejected = 0;
//   func.SetParameters(p);

   //std::vector<double> sum( MAX_THREAD );


#pragma omp parallel
#pragma omp for reduction (+:logl,nRejected)
//#pragma omp for reduction (+:logl,nRejected) schedule (static, 10)
//#pragma omp reduction (+:nRejected)

   for (unsigned int i = 0; i < n; ++ i) {

      //int ith = omp_get_thread_num();
      //func.SetParameters(p);

      const double * x = data.Coords(i);
      double fval = func ( x, p  ); // this is thread safe passing the params

      if (fval < 0) {
         nRejected++; // reject points with negative pdf (cannot exist)
      }
      else
         //sum[ith] += EvalLogF( fval);
         logl += EvalLogF( fval);

#ifdef DEBUG
#pragma omp critical
      {     std::cout << " ==== i = " << i << " thread " << omp_get_thread_num()
                      << "fval = " << fval << " logl = " << logl << std::endl;}
//       std::cout << "x [ " << data.PointSize() << " ] = ";
//       for (unsigned int j = 0; j < data.PointSize(); ++j)
//          std::cout << x[j] << "\t";
#endif
   }

   // reset the number of fitting data points
   if (nRejected != 0)  nPoints = n - nRejected;

#ifdef DEBUG
   ncalls++;
   int pr = std::cout.precision(15);
   std::cout << "ncalls " << ncalls << " Logl = " << logl << " np = " << nPoints << std::endl;
   std::cout.precision(pr);
   assert(ncalls<3);
#endif

   // logl = std::accumulate(sum.begin(), sum.end(),0. );

   double result = - logl;
   return result;
}

#endif


      } // end namespace FitUtilParallel

   } // end namespace Fit

} // end namespace ROOT




#endif
