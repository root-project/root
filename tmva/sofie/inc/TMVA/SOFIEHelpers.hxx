#ifndef TMVA_SOFIE_SOFIE_HELPERS
#define TMVA_SOFIE_SOFIE_HELPERS


#include <type_traits>
#include <utility>
#include <vector>
#include <memory>


namespace TMVA{
namespace Experimental{

///Helper class used by SOFIEFunctor to wrap the
///infer signature interface to RDataFrame
template <typename I, typename F, typename T>
class SofieFunctorHelper;

template <std::size_t... N,  typename Session_t, typename T>
class SofieFunctorHelper<std::index_sequence<N...>, Session_t, T> {
   /// this is the magic to defined the operator () with N fixed parameter arguments
   template <std::size_t Idx>
   using AlwaysT = T;

   std::vector<std::vector<T>> fInput;
   std::vector<std::shared_ptr<Session_t>> fSessions;

public:

   SofieFunctorHelper(unsigned int nslots = 0) :
      fInput(1)
   {
      // create Sessions according to given number of slots.
      // if number of slots is zero create a single session
      if (nslots < 1) nslots = 1;
      fInput.resize(nslots);
      for (unsigned int i = 0; i < nslots; i++) {
         fSessions.emplace_back(std::make_shared<Session_t>());
      }
   }

   double operator()(unsigned slot, AlwaysT<N>... args) {
      fInput[slot] = {args...};
      auto y =  fSessions[slot]->infer(fInput[slot].data());
      return y[0];
   }
};

/// SofieFunctor : used to wrap the infer function of the
/// generated model by SOFIE in a RDF compatible signature.
/// The number of slots is an optional parameter used to
/// create multiple SOFIE Sessions, which can be run in a parallel
/// model evaluation. One shouild use as number of slots the number of slots used by
/// RDataFrame. By default, in case of `nslots=0`, only a single Session will be created
/// and the Functor cannot be run in parallel.
/// Examples of using the SofieFunctor are the C++ tutorial TMVA_SOFIE_RDataFrame.C
/// and the Python tutorial TMVA_SOFIE_RDataFrame.py which makes use of the ROOT JIT
/// to compile on the fly the generated SOFIE model.
template <std::size_t N, typename Session_t>
auto SofieFunctor(unsigned int nslots = 0) -> SofieFunctorHelper<std::make_index_sequence<N>, Session_t, float>
{
   return SofieFunctorHelper<std::make_index_sequence<N>, Session_t, float>(nslots);
}

}//Experimental
}//TMVA

#endif //TMVA_SOFIE_SOFIE_HELPERS
