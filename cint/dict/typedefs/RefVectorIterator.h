#ifndef DataFormats_Common_RefVectorIterator_h
#define DataFormats_Common_RefVectorIterator_h

#include <memory>
#include "RefCore.h"
#include "Ref.h"

namespace edm {

  template <typename C, typename T, typename F = typename Ref<C>::finder_type>
  class RefVectorIterator {
  public:
    RefVectorIterator() : product_() {}
  private:
    RefCore product_;
  };
}
#endif
