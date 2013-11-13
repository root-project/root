#ifndef DataFormats_Common_RefTraits_h
#define DataFormats_Common_RefTraits_h

#include <functional>
#include <algorithm>

namespace edm {
  template<typename C, typename T, typename F> class RefVector;
  template<typename T> class RefToBaseVector;
 
  namespace refhelper {
    template<typename C, typename T>
    struct FindUsingAdvance {
      typedef C const&              first_argument_type;
      typedef unsigned int 	    second_argument_type;
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
#endif
