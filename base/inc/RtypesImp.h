
#ifndef G__DICTIONARY
#error RtypesImp.h should only be included by ROOT dictionaries.
#endif

namespace ROOT {
   inline const InitBehavior *DefineBehavior(void * /*parent_type*/,
                                             void * /*actual_type*/)
   {
      return new DefaultInitBehavior();
   }

   template <class T> const InitBehavior *ClassInfo<T>::fgAction       = 0;
   template <class T> TClass             *ClassInfo<T>::fgClass        = 0;
   template <class T> const char         *ClassInfo<T>::fgClassName    = 0;
   template <class T> Int_t               ClassInfo<T>::fgVersion      = 0;
   template <class T> const char         *ClassInfo<T>::fgImplFileName = 0;
   template <class T> Int_t               ClassInfo<T>::fgImplFileLine = 0;
   template <class T> const char         *ClassInfo<T>::fgDeclFileName = 0;
   template <class T> Int_t               ClassInfo<T>::fgDeclFileLine = 0;
   template <class T> typename ClassInfo<T>::ShowMembersFunc_t  ClassInfo<T>::fgShowMembers  = 0;

}

