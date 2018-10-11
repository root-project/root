// @(#)root/mathcore:$Id$
// Author: L. Moneta Tue Aug 4 2015

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2015  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// random engines based on ROOT 

#ifndef ROOT_Math_MixMaxEngine
#define ROOT_Math_MixMaxEngine

#include <cstdint>
#include <vector>
#include <string>

#include "Math/TRandomEngine.h"


// struct rng_state_st;    /// forward declare generator state

// typedef struct rng_state_st rng_state_t;

// namespace mixmax { 
//    template<int Ndim>
//    class mixmax_engine; 
// }

namespace ROOT {

   namespace Math {
      
      template<int N>
      class MixMaxEngineImpl;

      /**
         MixMaxEngine is a wrapper class for the MIXMAX Random number generator.
         MIXMAX is a matrix-recursive random number generator introduced by
         G. Savvidy.

         The real implementation of the generator, written in C, is in the mixmax.h and mixmax.cxx files.
         This generator code is available also at hepforge: http://mixmax.hepforge.org
         The MIXMAX code has been created and developed by Konstantin Savvidy and it is 
         released under GNU Lesser General Public License v3.

         This wrapper class provides 3 different variants of MIXMAX according to the template para extra parameter N. 
         The extra parameter, `SkipNumber`, is used to perform additional iterations of the generator before returning the random numbers. 
           For example, when `SkipNumber = 2`, the generator will have two extra iterations that will be discarder. 

         *  MIXMAX with N = 240. This is a new version of  the generator (version 2.0beta)  described in the 
            <a href="http://dx.doi.org/10.1016/j.chaos.2016.05.003">2016 paper</a> (3rd reference), with 
            special number $s=487013230256099140$, $m=2^{51}+1$ and having a period of $10^{4389}$. 
          
         *  MIXMAX with N = 17, from the 2.0beta version with $s=0$ and $m=2^{36}+1$. The period of the generator is $10^{294}$. 

         *  MIXMAX with N = 256 from the 1.0 version. The period is (for `SkipNumber=0`) $10^{4682}$. 
            For this generator we recommend in ROOT using a default value of `SkipNumber=2, while for the previous two generators 
            skipping is not needed. 
 
            This table describes the properties of the MIXMAX generators. MIXMAX is a genuine 61 bit generator on the Galois field GF[p], where
           $p=2^{61}-1$ is the Mersenne prime number.
          The MIXMAX generators with these parameters pass all of the BigCrush
          tests in the <a href="http://simul.iro.umontreal.ca/testu01/tu01.html">TestU01 suite</a\>.
   
\latexonly
 \begin{table}[h]
   \centering
     \begin{tabular}{@{} rrlcrr @{}} 
     
      \toprule
      Dimension &~ Entropy & Decorrelation Time &  Iteration Time  & Relaxation Time                                 &Period  q\\
      N     &~~ $~h(T)$   &~~~ $\tau_0 = {1\over h(T) 2N }$ & t & $\tau ={1\over h(T) \ln {1\over \delta v_0}}$ &  $  \log_{10} (q)$  \\ % Crush
      \midrule
      256    & 194   & ~~~~~0.000012     & 1   & 95.00  &   4682\footnote{full
      period is not confirmed}  \\
      \hline
         8   & 220   & $~~~~~0.00028$    & 1   & 1.54   &    129  \\
        17   & 374   & ~~~~~0.000079     & 1   & 1.92   &    294  \\
       240   & 8679  & ~~~~~0.00000024   & 1   & 1.17   &   4389  \\
      \bottomrule
   \end{tabular}
     \caption{The entropy $h(T)$, decorrelation time $\tau_0$ 
       decorrelation time, relaxation time $\tau $ and period of the MIXMAX generator
       \cite{savvidy2017ex,savvidy2017cl},
       expressed in units of the iteration time $t$, which is
       normalised to 1.
       Clearly $\tau_0~ < t ~< \tau $.
}
\end{table}
\endlatexonly

         The References for MIXMAX are 

         *  G.K.Savvidy and N.G.Ter-Arutyunian, *On the Monte Carlo simulation of physical systems,
         J.Comput.Phys. 97, 566 (1991)*;
         Preprint EPI-865-16-86, Yerevan, Jan. 1986

         *  K.Savvidy, *The MIXMAX random number generator*, 
         Comp. Phys. Commun. 196 (2015), pp 161–165
         http://dx.doi.org/10.1016/j.cpc.2015.06.003

         *  K.Savvidy and G.Savvidy, *Spectrum and Entropy of C-systems MIXMAX Random Number Generator*,
         Chaos, Solitons & Fractals, Volume 91, (2016) pp. 33–38
         http://dx.doi.org/10.1016/j.chaos.2016.05.003
    

         @ingroup Random
      */

      template<int N, int SkipNumber>
      class MixMaxEngine : public TRandomEngine {

      public:

         typedef  TRandomEngine BaseType;

         // this should be changed for WINDOWS
#ifndef __LP64__
         typedef uint64_t StateInt_t;
#else
         typedef unsigned long long StateInt_t;
#endif
         typedef uint64_t Result_t;


         MixMaxEngine(uint64_t seed=1);

         virtual ~MixMaxEngine();


         /// Get the size of the generator
         static int Size();

         /// maximum integer that can be generated. For MIXMAX is 2^61-1
         static uint64_t MaxInt();

         /// minimum integer that can be generated. For MIXMAX is 0
         static uint64_t MinInt();

         /// set the generator seed
         void  SetSeed(Result_t seed);

         // generate a random number (virtual interface)
         virtual double Rndm() { return Rndm_impl(); }

         /// generate a double random number (faster interface)
         inline double operator() () { return Rndm_impl(); }

         /// generate an array of random numbers
         void RndmArray (int n, double * array);

         /// generate a 64  bit integer number
         Result_t IntRndm();

         /// get name of the generator
         static const char *Name();

      protected:
         // protected functions used for tesing the generator

         /// get the state of the generator
         void GetState(std::vector<StateInt_t> & state) const;


         ///set the full initial generator state
         void SetState(const std::vector<StateInt_t> & state);

         /// Get the counter (between 0 and Size-1)
         int Counter() const;


      private:

         /// implementation function to generate the random number
         double Rndm_impl();

         //rng_state_t * fRngState;  // mix-max generator state
         //mixmax::mixmax_engine<N> * fRng;  // mixmax internal engine class
         MixMaxEngineImpl<N> * fRng;  // mixmax internal engine class
         
      };

      typedef MixMaxEngine<240,0> MixMaxEngine240;
      typedef MixMaxEngine<256,2> MixMaxEngine256;
      typedef MixMaxEngine<17,0> MixMaxEngine17;
      
      extern template class MixMaxEngine<240,0>;
      extern template class MixMaxEngine<256,0>;
      extern template class MixMaxEngine<256,2>;
      extern template class MixMaxEngine<256,4>;
      extern template class MixMaxEngine<17,0>;
      extern template class MixMaxEngine<17,1>;
      extern template class MixMaxEngine<17,2>;

   } // end namespace Math

} // end namespace ROOT


#include "Math/MixMaxEngine.icc"

#endif /* ROOT_Math_MixMaxEngine */ 
