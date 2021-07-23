
// edm::Ref<vector<reco::Muon>,reco::Muon,edm::refhelper::FindUsingAdvance<vector<reco::Muon>,reco::Muon> >

#include <vector>

namespace reco {
   class Muon {};
   class PFCandidate {};
}

#include <functional>
#include <algorithm>

#if defined(_MSC_VER) && __cplusplus > 201402L
namespace std {
   // std::unary_function and std::binary_function were both removed
   // in C++17.

   template <typename Arg1, typename Result>
   struct unary_function {
      typedef Arg1 argument_type;
      typedef Result result_type;
   };

   template <typename Arg1, typename Arg2, typename Result>
   struct binary_function {
      typedef Arg1 first_argument_type;
      typedef Arg2 second_argument_type;
      typedef Result result_type;
   };
}
#endif

namespace edm {
   template<typename C, typename T, typename F> class RefVector;
   template<typename T> class RefToBaseVector;

   namespace refhelper {
      template<typename C, typename T>
      struct FindUsingAdvance {
         typedef C const&              first_argument_type;
         typedef unsigned int          second_argument_type;
         typedef T const*              result_type;

         result_type operator()(first_argument_type iContainer, second_argument_type iIndex) {
            typename C::const_iterator it = iContainer.begin();
            std::advance(it, static_cast<typename C::size_type>(iIndex));
            return it.operator->();
         }
      };

      template<typename REFV>
      struct FindRefVectorUsingAdvance : public std::binary_function<REFV const&,
      typename REFV::key_type,
      typename REFV::member_type const*> {
         typedef FindRefVectorUsingAdvance<REFV> self;
         typename self::result_type operator()(typename self::first_argument_type iContainer,
                                               typename self::second_argument_type iIndex) {
            typename REFV::const_iterator it = iContainer.begin();
            std::advance(it, iIndex);
            return it.operator->()->get();;
         }
      };

      //Used in edm::Ref to set the default 'find' method to use based on the Container and 'contained' type
      template<typename C, typename T>
      struct FindTrait {
         typedef FindUsingAdvance<C, T> value;
      };

      template<typename C, typename T, typename F>
      struct FindTrait<RefVector<C, T, F>, T> {
         typedef FindRefVectorUsingAdvance<RefVector<C, T, F> > value;
      };

      template<typename T>
      struct FindTrait<RefToBaseVector<T>, T> {
         typedef FindRefVectorUsingAdvance<RefToBaseVector<T> > value;
      };

      template<typename C>
      struct ValueTrait {
         typedef typename C::value_type value;
      };

      template<typename C, typename T, typename F>
      struct ValueTrait<RefVector<C, T, F> > {
         typedef T value;
      };

      template<typename T>
      struct ValueTrait<RefToBaseVector<T> > {
         typedef T value;
      };

   }
}

namespace edm {
//   namespace refhelper {
//      template <typename T, typename Q = typename T::value_type>
//      class FindUsingAdvance {};
//   }

   //   template <typename T, typename Q = typename T::value_type, typename H = edm::refhelper::FindUsingAdvance<T,Q> >
   // class Ref {};

   template <typename C,
   typename T = typename refhelper::ValueTrait<C>::value,
   typename F = typename refhelper::FindTrait<C, T>::value>
   class Ref {};

#define REF_FOR_VECTOR_ARGS std::vector<E>,typename refhelper::ValueTrait<std::vector<E> >::value,typename refhelper::FindTrait<std::vector<E>, typename refhelper::ValueTrait<std::vector<E> >::value>::value

   template <typename E>
   class Ref<REF_FOR_VECTOR_ARGS> {};


   //edm::ValueMap<std::vector<edm::Ref<std::vector<reco::PFCandidate>,reco::PFCandidate,edm::refhelper::FindUsingAdvance<std::vector<reco::PFCandidate>,reco::PFCandidate> > > >::value_type

   template <typename T>
   class ValueMap {
   public:
      typedef T value_type;
   };
}

// vector<edm::Ref<vector<reco::PFCandidate>,reco::PFCandidate,edm::refhelper::FindUsingAdvance<std::vector<reco::PFCandidate>,reco::PFCandidate> > >

// vector<edm::Ref<vector<reco::PFCandidate>,reco::PFCandidate,edm::refhelper::FindUsingAdvance<vector<reco::PFCandidate>,reco::PFCandidate> > >

// vector<edm::Ref<vector<reco::PFCandidate>,reco::PFCandidate,edm::refhelper::FindUsingAdvance<vector<reco::PFCandidate>,reco::PFCandidate> > >

