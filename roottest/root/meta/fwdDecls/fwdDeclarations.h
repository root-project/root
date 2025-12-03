#ifndef __fwdDeclarations_h__
#define __fwdDeclarations_h__
#include "Math/GenVector/PositionVector3D.h"
#include "Math/Point3Dfwd.h"

#include <unordered_map>

// Gaudi::XYZ point: this typedef should be fwd declared surrounded by its namespace
namespace Gaudi {
   typedef ROOT::Math::XYZPoint XYZPoint;
}

// Gaudi MyTrack: startic const reference
typedef unsigned int CLID;
namespace Gaudi
{
  namespace Examples
  {
    static const CLID& CLID_MyTrack = 355;
    class MyTrack{};
  }
}

// This is a complex CMS template
namespace reco {
   class Candidate {};
}

namespace edm{
   namespace refhelper {

      template<typename C, typename T>
      struct FindUsingAdvance {
         typedef C const&              first_argument_type;
         typedef unsigned int      second_argument_type;
         typedef T const*              result_type;

         result_type operator()(first_argument_type iContainer, second_argument_type iIndex) {
            typename C::const_iterator it = iContainer.begin();
            std::advance(it, static_cast<typename C::size_type>(iIndex));
            return it.operator->();
         }
      };

      template<typename C, typename T>
      struct FindTrait {
         typedef FindUsingAdvance<C, T> value;
      };

      template<typename C>
      struct ValueTrait {
         typedef typename C::value_type value;
      };
   }

   template <class T> class ClonePolicy{};

   template <class T, class P=ClonePolicy<T>> class OwnVector{};

   template <typename C,
             typename T = typename refhelper::ValueTrait<C>::value,
             typename F = typename refhelper::FindTrait<C, T>::value>
   class Ref {};
}

namespace reco {
   typedef edm::Ref<
      edm::OwnVector<
         reco::Candidate,edm::ClonePolicy<reco::Candidate>
      >,reco::Candidate,edm::refhelper::FindUsingAdvance<
                                                   edm::OwnVector<
                                                      reco::Candidate,
                                                      edm::ClonePolicy<reco::Candidate>
                                                   >,
         reco::Candidate>
      > CandidateRef;
}


// This is something that on linux produced invalid payloads. pair was for example
// fwd declared within the __gnu_cxx namespace rather than std. And it was fwd declared,
// strangely enough
// (This used to test with std::hash_map; let's assume unordered_map is similar
// enough to still test the issue.)
typedef std::unordered_map<int,int> MyMap;

// This is to stress the scoping of tmplt arguments
namespace ns {
   class A{};
   namespace ns2 {
      class B{};
      class D{};
   }
   typedef ns2::D nestedD;
}

template <class X, class T = ns::A, class V = ns::ns2::B, class W = ns::nestedD>
class C{};

C<int> instanceOfC;

// This is to stress the spurrious template instantiations
template<class T>
class unfortunate{
public:
   typedef double value;
};
template <class T, class V = typename unfortunate<T>::value>
class E{};
typedef E<int> e_int;

#endif // end of header
