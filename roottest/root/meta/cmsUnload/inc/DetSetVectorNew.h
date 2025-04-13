#ifndef DetSetVectorNew_h
#define DetSetVectorNew_h

#include "DetSetNew.h"
#include <vector>

namespace edm { namespace refhelper { template<typename T> struct FindForNewDetSetVector; } }


namespace edmNew {

   namespace dslv {
      template< typename T> class LazyGetter;
   }
   namespace dstvdetails {
      struct DetSetVectorTrans {
         DetSetVectorTrans(): filling(false){}
         bool filling;
         //boost::any getter;


         void swap(DetSetVectorTrans& rh) {
            std::swap(filling,rh.filling);
            //std::swap(getter,rh.getter);
         }

         typedef unsigned int size_type; // for persistency
         typedef unsigned int id_type;

         struct Item {
            Item(id_type i=0, int io=-1, size_type is=0) : id(i), offset(io), size(is){}
            bool isValid() const { return offset>=0;}
            id_type id;
            int offset;
            size_type size;
            bool operator<(Item const &rh) const { return id<rh.id;}
            operator id_type() const { return id;}
         };
         
      };

   }

   template<typename T>
   class DetSetVector  : private dstvdetails::DetSetVectorTrans {
   public:
      //typedef dstvdetails::DetSetVectorTrans Trans;
      //typedef Trans::Item Item;
      typedef unsigned int size_type; // for persistency
      typedef unsigned int id_type;
      typedef T data_type;
      typedef edmNew::DetSetVector<T> self;
      typedef edmNew::DetSet<T> DetSet;
      typedef dslv::LazyGetter<T> Getter;
      // FIXME not sure make sense....
      typedef DetSet value_type;
      typedef id_type key_type;


      typedef std::vector<Item> IdContainer;
      typedef std::vector<data_type> DataContainer;
      typedef typename IdContainer::iterator IdIter;
      typedef typename std::vector<data_type>::iterator DataIter;
      typedef std::pair<IdIter,DataIter> IterPair;
      typedef typename IdContainer::const_iterator const_IdIter;
      typedef typename std::vector<data_type>::const_iterator const_DataIter;
      typedef std::pair<const_IdIter,const_DataIter> const_IterPair;

      typedef typename edm::refhelper::FindForNewDetSetVector<data_type>  RefFinder;
      
   };

   namespace dslv {
      template< typename T>
      class LazyGetter {
      public:
         virtual ~LazyGetter() {}
         virtual void fill(typename DetSetVector<T>::FastFiller&) = 0;
      };
   }
   
}

#endif
