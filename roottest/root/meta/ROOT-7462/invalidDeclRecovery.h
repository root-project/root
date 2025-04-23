#ifndef invalidDeclRecovery_h
#define invalidDeclRecovery_h

namespace edm {
   template <class T>
   class StrictWeakOrdering
   {
      typedef typename T::key_type key_type;
   };

   template <class Key, class key_compare>
   class SortedCollection
   {
   public:
      typedef typename key_compare::key_type key_type;
   };

   template <class T>
   class Wrapper
   {
   public:
      T obj;
   };
}

// In the real life example, this comes from the rootmap file.
class ESKCHIPBlock;

#endif // invalidDeclRecovery_h
