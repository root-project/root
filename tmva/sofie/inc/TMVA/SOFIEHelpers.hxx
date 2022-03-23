#ifndef TMVA_SOFIE_SOFIE_HELPERS
#define TMVA_SOFIE_SOFIE_HELPERS


#include <type_traits>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <regex>

namespace TMVA{
namespace Experimental{

// Functor to wrap a SOFIE session to RDF functor signature

template <typename I, typename F, typename T>
class SofieFunctorHelper;

template <std::size_t... N,  typename S, typename T>
class SofieFunctorHelper<std::index_sequence<N...>, S, T> {
   /// this is the magic to defined the operator () with N fixed parameter arguments
   template <std::size_t Idx>
   using AlwaysT = T;

   std::vector<std::vector<T>> fInput;
   std::vector<std::shared_ptr<S>> fSessions;

public:

   SofieFunctorHelper(int nslots = 0) :
      fInput(1)
   {
      // create Sessions according to given nunber of slots.
      // if number of slots is zero create a single session
      if (nslots < 1) nslots = 1;
      fInput.resize(nslots);
      for (int i = 0; i < nslots; i++) {
         fSessions.emplace_back(std::make_shared<S>());
      }
   }

   double operator()(unsigned slot, AlwaysT<N>... args) {
      fInput[slot] = {args...};
      auto y =  fSessions[slot]->infer(fInput[slot].data());
      return y[0];
   }
};

template <std::size_t N, typename F>
auto SofieFunctor(int nslot) -> SofieFunctorHelper<std::make_index_sequence<N>, F, float>
{
   return SofieFunctorHelper<std::make_index_sequence<N>, F, float>(nslot);
}

}//Experimental
}//TMVA

#endif //TMVA_SOFIE_SOFIE_HELPERS
