#ifndef Wrapper_h
#define Wrapper_h

#include <vector>

namespace edm {
   template <typename T> class Wrapper {
   public:
      typedef T value_type;
      typedef T wrapped_type; // used with the dictionary to identify Wrappers
   };

   namespace refhelper {
      template <typename C> struct ValueTrait
      {
         typedef typename C::value_type value;
      };

      template <typename C, typename T> struct FindTrait
      {
         typedef typename C::value_type value;
      };
   }
   template <typename C,
   typename T = typename refhelper::ValueTrait<C>::value,
   typename F = typename refhelper::FindTrait<C, T>::value>
   class Ref {
   private:
#if 0
      typedef refhelper::FindRefVectorUsingAdvance<RefVector<C, T, F> > VF;
      typedef refhelper::FindRefVectorUsingAdvance<RefToBaseVector<T> > VBF;
      friend class RefVectorIterator<C, T, F>;
      friend class RefVector<C, T, F>;
      friend class RefVector<RefVector<C, T, F>, T, VF>;
      friend class RefVector<RefVector<C, T, F>, T, VBF>;
#endif

   public:
      /// for export
      typedef C product_type;
      typedef T value_type;
      typedef T const element_type; //used for generic programming
      typedef F finder_type;
#if 0
      typedef typename boost::binary_traits<F>::second_argument_type argument_type;
      typedef typename boost::remove_cv<typename boost::remove_reference<argument_type>::type>::type key_type;
      /// C is the type of the collection
#endif
   };

}

#endif
