#ifndef DetSetNew_h
#define DetSetNew_h

#include <vector>

namespace edmNew {

   template<typename T> class DetSetVector;

   template<typename T>
   class DetSet {
   public:
      typedef DetSetVector<T> Container;
      typedef unsigned int size_type; // for persistency
      typedef unsigned int id_type;
      typedef T data_type;

      typedef std::vector<data_type> DataContainer;
      typedef data_type * iterator;
      typedef data_type const * const_iterator;

      typedef data_type value_type;
      typedef id_type key_type;

   };
}

#endif