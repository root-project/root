#ifndef DataFormats_Common_RefCore_h
#define DataFormats_Common_RefCore_h

/*----------------------------------------------------------------------
  
RefCore: The component of edm::Ref containing the product ID and product getter.

----------------------------------------------------------------------*/
#include <algorithm>
#include <typeinfo>

namespace edm {
  class RefCoreWithIndex;
  
  class RefCore {
    //RefCoreWithIndex is a specialization of RefCore done for performance
    // Since we need to freely convert one to the other the friendship is used
    friend class RefCoreWithIndex;
  public:
    RefCore() :  cachePtr_(0),processIndex_(0),productIndex_(0){}

    /**If productPtr is not 0 then productGetter will be 0 since only one is available at a time */
    void const* productPtr() const;

    void setProductPtr(void const* prodPtr) const { 
      cachePtr_=prodPtr;
      setCacheIsProductPtr();
    }

    // Checks for null
    bool isNull() const {return !isNonnull(); }

    // Checks for non-null
    bool isNonnull() const;

    // Checks for null
    bool operator!() const {return isNull();}

    // Checks if collection is in memory or available
    // in the Event. No type checking is done.

    bool isAvailable() const;

    void productNotFoundException(std::type_info const& type) const;

    void wrongTypeException(std::type_info const& expectedType, std::type_info const& actualType) const;

    void nullPointerForTransientException(std::type_info const& type) const;

    void swap(RefCore &);
    
    bool isTransient() const;

    int isTransientInt() const {return isTransient() ? 1 : 0;}

    void pushBackItem(RefCore const& productToBeInserted, bool checkPointer);

 private:
    RefCore(void const* iCache, unsigned short iProcessIndex, unsigned short iProductIndex):
    cachePtr_(iCache), processIndex_(iProcessIndex), productIndex_(iProductIndex) {}
    void setTransient();
    void setCacheIsProductPtr() const;
    void unsetCacheIsProductPtr() const;
    bool cacheIsProductPtr() const;

    
    mutable void const* cachePtr_;               // transient
    mutable unsigned short processIndex_;
    unsigned short productIndex_;

  };

  inline
  bool
  operator<(RefCore const& lhs, RefCore const& rhs) {
    return lhs.isTransient() ? (rhs.isTransient() ? lhs.productPtr() < rhs.productPtr() : false) : (rhs.isTransient() ? true : false);
  }

  inline 
  void
  RefCore::swap(RefCore & other) {
    std::swap(processIndex_, other.processIndex_);
    std::swap(productIndex_, other.productIndex_);
    std::swap(cachePtr_, other.cachePtr_);
  }

  inline void swap(edm::RefCore & lhs, edm::RefCore & rhs) {
    lhs.swap(rhs);
  }
}

#endif
