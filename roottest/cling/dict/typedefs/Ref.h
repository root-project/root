#ifndef DataFormats_Common_Ref_h
#define DataFormats_Common_Ref_h

#include "RefCore.h"
#include "traits.h"

#include "boost/functional.hpp"

#include "RefTraits.h"

namespace edm {
  template<typename C, typename T, typename F>
  class RefVector;

  template<typename T>
  class RefToBaseVector;

  template <typename C, 
	    typename T = typename refhelper::ValueTrait<C>::value, 
	    typename F = typename refhelper::FindTrait<C, T>::value>
  class Ref {
  public:
    /// for export
    typedef C product_type;
    typedef T value_type; 
    typedef T const element_type; //used for generic programming
    typedef F finder_type;
    typedef typename boost::binary_traits<F>::second_argument_type key_type;
    //typedef int key_type;

    /// Default constructor needed for reading from persistent store. Not for direct use.
    Ref() : product_(), index_() {}

    /// Destructor
    ~Ref() {}

  private:
    mutable RefCore product_;
    key_type index_;

  };

}
#endif
