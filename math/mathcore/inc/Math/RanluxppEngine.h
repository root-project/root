// @(#)root/mathcore:$Id$
// Author: Jonas Hahnfeld 11/2020

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_Math_RanluxppEngine
#define ROOT_Math_RanluxppEngine

#include "Math/TRandomEngine.h"

#include <cstdint>
#include <memory>

namespace ROOT {
namespace Math {

template <int w, int p, int u = 0>
class RanluxppEngineImpl;

template <int p>
class RanluxppEngine final : public TRandomEngine {

private:
   using ImplType = RanluxppEngineImpl<48, p>;
   std::unique_ptr<ImplType> fImpl;

public:
   RanluxppEngine(uint64_t seed = 314159265);
   virtual ~RanluxppEngine();

   /// Generate a double-precision random number with 48 bits of randomness
   double Rndm() override;
   /// Generate a double-precision random number (non-virtual method)
   double operator()();
   /// Generate a random integer value with 48 bits
   uint64_t IntRndm();

   /// Initialize and seed the state of the generator
   void SetSeed(uint64_t seed);
   /// Skip `n` random numbers without generating them
   void Skip(uint64_t n);

   /// Get name of the generator
   static const char *Name() { return "RANLUX++"; }
};

using RanluxppEngine24 = RanluxppEngine<24>;
using RanluxppEngine2048 = RanluxppEngine<2048>;

extern template class RanluxppEngine<24>;
extern template class RanluxppEngine<2048>;


template <int p>
class RanluxppCompatEngineJames final : public TRandomEngine {

private:
   using ImplType = RanluxppEngineImpl<24, p>;
   std::unique_ptr<ImplType> fImpl;

public:
   RanluxppCompatEngineJames(uint64_t seed = 314159265);
   virtual ~RanluxppCompatEngineJames();

   /// Generate a floating point random number with 24 bits of randomness
   double Rndm() override;
   /// Generate a floating point random number (non-virtual method)
   double operator()();
   /// Generate a random integer value with 24 bits
   uint64_t IntRndm();

   /// Initialize and seed the state of the generator
   void SetSeed(uint64_t seed);
   /// Skip `n` random numbers without generating them
   void Skip(uint64_t n);

   /// Get name of the generator
   static const char *Name() { return "RanluxppCompatJames"; }
};

/// Compatibility engine for original RANLUX implementation by James, luxury
/// level 3 (p = 223). The sequence of numbers also matches `gsl_rng_ranlux`.
using RanluxppCompatEngineJamesP3 = RanluxppCompatEngineJames<223>;
/// Compatibility engine for original RANLUX implementation by James, luxury
/// level 4 (p = 389). The sequence of numbers also matches `gsl_rng_ranlux389`.
using RanluxppCompatEngineJamesP4 = RanluxppCompatEngineJames<389>;

extern template class RanluxppCompatEngineJames<223>;
extern template class RanluxppCompatEngineJames<389>;


/// Compatibility engine for `gsl_rng_ranlxs*` from the GNU Scientific Library.
template <int p>
class RanluxppCompatEngineGslRanlxs final : public TRandomEngine {

private:
   using ImplType = RanluxppEngineImpl<24, p>;
   std::unique_ptr<ImplType> fImpl;

public:
   RanluxppCompatEngineGslRanlxs(uint64_t seed = 1);
   virtual ~RanluxppCompatEngineGslRanlxs();

   /// Generate a floating point random number with 24 bits of randomness
   double Rndm() override;
   /// Generate a floating point random number (non-virtual method)
   double operator()();
   /// Generate a random integer value with 24 bits
   uint64_t IntRndm();

   /// Initialize and seed the state of the generator
   void SetSeed(uint64_t seed);
   /// Skip `n` random numbers without generating them
   void Skip(uint64_t n);

   /// Get name of the generator
   static const char *Name() { return "RanluxppCompatGslRanlxs"; }
};

using RanluxppCompatEngineGslRanlxs0 = RanluxppCompatEngineGslRanlxs<218>;
using RanluxppCompatEngineGslRanlxs1 = RanluxppCompatEngineGslRanlxs<404>;
using RanluxppCompatEngineGslRanlxs2 = RanluxppCompatEngineGslRanlxs<794>;

extern template class RanluxppCompatEngineGslRanlxs<218>;
extern template class RanluxppCompatEngineGslRanlxs<404>;
extern template class RanluxppCompatEngineGslRanlxs<794>;


/// Compatibility engine for `gsl_rng_ranlxd*` from the GNU Scientific Library.
template <int p>
class RanluxppCompatEngineGslRanlxd final : public TRandomEngine {

private:
   using ImplType = RanluxppEngineImpl<48, p>;
   std::unique_ptr<ImplType> fImpl;

public:
   RanluxppCompatEngineGslRanlxd(uint64_t seed = 1);
   virtual ~RanluxppCompatEngineGslRanlxd();

   /// Generate a floating point random number with 48 bits of randomness
   double Rndm() override;
   /// Generate a floating point random number (non-virtual method)
   double operator()();
   /// Generate a random integer value with 48 bits
   uint64_t IntRndm();

   /// Initialize and seed the state of the generator
   void SetSeed(uint64_t seed);
   /// Skip `n` random numbers without generating them
   void Skip(uint64_t n);

   /// Get name of the generator
   static const char *Name() { return "RanluxppCompatGslRanlxd"; }
};

using RanluxppCompatEngineGslRanlxd1 = RanluxppCompatEngineGslRanlxd<404>;
using RanluxppCompatEngineGslRanlxd2 = RanluxppCompatEngineGslRanlxd<794>;

extern template class RanluxppCompatEngineGslRanlxd<404>;
extern template class RanluxppCompatEngineGslRanlxd<794>;


template <int w, int p>
class RanluxppCompatEngineLuescherImpl;

/// Compatibility engine for Lüscher's ranlxs implementation written in C.
template <int p>
class RanluxppCompatEngineLuescherRanlxs final : public TRandomEngine {

private:
   using ImplType = RanluxppCompatEngineLuescherImpl<24, p>;
   std::unique_ptr<ImplType> fImpl;

public:
   RanluxppCompatEngineLuescherRanlxs(uint64_t seed = 314159265);
   virtual ~RanluxppCompatEngineLuescherRanlxs();

   /// Generate a floating point random number with 24 bits of randomness
   double Rndm() override;
   /// Generate a floating point random number (non-virtual method)
   double operator()();
   /// Generate a random integer value with 24 bits
   uint64_t IntRndm();

   /// Initialize and seed the state of the generator
   void SetSeed(uint64_t seed);
   /// Skip `n` random numbers without generating them
   void Skip(uint64_t n);

   /// Get name of the generator
   static const char *Name() { return "RanluxppCompatLuescherRanlxs"; }
};

using RanluxppCompatEngineLuescherRanlxs0 = RanluxppCompatEngineLuescherRanlxs<218>;
using RanluxppCompatEngineLuescherRanlxs1 = RanluxppCompatEngineLuescherRanlxs<404>;
using RanluxppCompatEngineLuescherRanlxs2 = RanluxppCompatEngineLuescherRanlxs<794>;

extern template class RanluxppCompatEngineLuescherRanlxs<218>;
extern template class RanluxppCompatEngineLuescherRanlxs<404>;
extern template class RanluxppCompatEngineLuescherRanlxs<794>;


/// Compatibility engine for Lüscher's ranlxd implementation written in C.
template <int p>
class RanluxppCompatEngineLuescherRanlxd final : public TRandomEngine {

private:
   using ImplType = RanluxppCompatEngineLuescherImpl<48, p>;
   std::unique_ptr<ImplType> fImpl;

public:
   RanluxppCompatEngineLuescherRanlxd(uint64_t seed = 314159265);
   virtual ~RanluxppCompatEngineLuescherRanlxd();

   /// Generate a floating point random number with 48 bits of randomness
   double Rndm() override;
   /// Generate a floating point random number (non-virtual method)
   double operator()();
   /// Generate a random integer value with 48 bits
   uint64_t IntRndm();

   /// Initialize and seed the state of the generator
   void SetSeed(uint64_t seed);
   /// Skip `n` random numbers without generating them
   void Skip(uint64_t n);

   /// Get name of the generator
   static const char *Name() { return "RanluxppCompatLuescherRanlxd"; }
};
using RanluxppCompatEngineLuescherRanlxd1 = RanluxppCompatEngineLuescherRanlxd<404>;
using RanluxppCompatEngineLuescherRanlxd2 = RanluxppCompatEngineLuescherRanlxd<794>;

extern template class RanluxppCompatEngineLuescherRanlxd<404>;
extern template class RanluxppCompatEngineLuescherRanlxd<794>;


/// Compatibility engine for `std::ranlux24` from the C++ standard.
class RanluxppCompatEngineStdRanlux24 final : public TRandomEngine {

private:
   using ImplType = RanluxppEngineImpl<24, 223, 23>;
   std::unique_ptr<ImplType> fImpl;

public:
   RanluxppCompatEngineStdRanlux24(uint64_t seed = 19780503);
   virtual ~RanluxppCompatEngineStdRanlux24();

   /// Generate a floating point random number with 24 bits of randomness
   double Rndm() override;
   /// Generate a floating point random number (non-virtual method)
   double operator()();
   /// Generate a random integer value with 24 bits
   uint64_t IntRndm();

   /// Initialize and seed the state of the generator
   void SetSeed(uint64_t seed);
   /// Skip `n` random numbers without generating them
   void Skip(uint64_t n);

   /// Get name of the generator
   static const char *Name() { return "RanluxppCompatStdRanlux24"; }
};


/// Compatibility engine for `std::ranlux48` from the C++ standard.
class RanluxppCompatEngineStdRanlux48 final : public TRandomEngine {

private:
   using ImplType = RanluxppEngineImpl<48, 2 * 389, 11>;
   std::unique_ptr<ImplType> fImpl;

public:
   RanluxppCompatEngineStdRanlux48(uint64_t seed = 19780503);
   virtual ~RanluxppCompatEngineStdRanlux48();

   /// Generate a floating point random number with 48 bits of randomness
   double Rndm() override;
   /// Generate a floating point random number (non-virtual method)
   double operator()();
   /// Generate a random integer value with 48 bits
   uint64_t IntRndm();

   /// Initialize and seed the state of the generator
   void SetSeed(uint64_t seed);
   /// Skip `n` random numbers without generating them
   void Skip(uint64_t n);

   /// Get name of the generator
   static const char *Name() { return "RanluxppCompatStdRanlux48"; }
};

} // end namespace Math
} // end namespace ROOT

#endif /* ROOT_Math_RanluxppEngine */
