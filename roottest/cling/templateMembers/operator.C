#ifdef __MAKECINT__

#endif

namespace edm {

   template <class T> class RCPtr;
   
   template <class T>
   bool
   operator< (const RCPtr<T>& p1, const RCPtr<T>& p2);

   template <class T>
   class RCPtr {
   public:
      
      friend bool operator< <> (const RCPtr<T>& p1, const RCPtr<T>& p2);
   };
   
   template <class T> 
   class sub {

   };

   class THandle
   {
   public:
      typedef RCPtr<double > rcptr_type;     
   };
   
   
} // namespace edm

#ifdef __MAKECINT__
#pragma link C++ namespace edm; 
#endif


