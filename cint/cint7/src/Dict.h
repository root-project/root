/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file Class.h
 ************************************************************************
 * Description:
 *  Extended Run Time Type Identification API
 ************************************************************************
 * Copyright(c) 1995~1998  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


#ifndef G__DICT_H
#define G__DICT_H 

#include "Api.h"
#include <vector>

#ifdef _WIN32
#pragma warning( push )
#pragma warning( disable : 4251 )
#endif

namespace Cint {
   namespace Internal {

   /*********************************************************************
   * class G__Dict
   *
   * Hold information about the dictionary (possibly the local thread
   * dictionary.
   *********************************************************************/
   class 
   #ifndef __CINT__
   G__EXPORT
   #endif
   G__Dict {
   private:
      std::vector<Reflex::Type> mTypenums; // link between typenums/tagnums and Reflex types
      std::vector<Reflex::Scope> mScopes; 

      ::Reflex::Type GetTypeImpl(int tagnum);
      
      G__Dict();
      ~G__Dict() {};
   public:

      static G__Dict &GetDict();

      
      ::Reflex::Type GetType(int tagnum) {
         // Return the Reflex type (class or namespace) corresponding 
         // to the CINT tagnum.

         static ::Reflex::Type null_type;
         if (-1 == tagnum || tagnum>=((int)mScopes.size())) return null_type;
         ::Reflex::Type t = mScopes[tagnum];
         if (t) return t;
         return GetTypeImpl(tagnum);
      }
      static inline Reflex::Type GetTypeFromId(size_t handle) {
         return Reflex::Type( reinterpret_cast< const Reflex::TypeName* >(handle));
      }
      ::Reflex::Type GetTypedef(int typenum);
      ::Reflex::Scope GetScope(int tagnum);
      static inline ::Reflex::Scope GetScope(const struct G__ifunc_table* funcnum) {
         return Reflex::Scope(reinterpret_cast<const Reflex::ScopeName*>(funcnum));
      }
      static inline ::Reflex::Scope GetScope(const struct G__var_array* varnum) {
         return Reflex::Scope(reinterpret_cast<const Reflex::ScopeName*>(varnum));
      }
      ::Reflex::Member GetFunction( const struct G__ifunc_table* funcnum, int ifn );
      static inline ::Reflex::Member GetFunction(size_t handle) {
         return Reflex::Member(reinterpret_cast< const Reflex::MemberBase* >(handle));
      }
      ::Reflex::Member GetDataMember( const struct G__var_array* varnum, int ig15 );
      static inline ::Reflex::Member GetDataMember(size_t handle) {
         return Reflex::Member(reinterpret_cast< const Reflex::MemberBase* >(handle));
      }
      int Register(const ::Reflex::Type&);
      bool RegisterScope(int tagnum, const ::Reflex::Scope&);

      size_t GetNumTypes() { return mTypenums.size(); }

   };

   }
}

#ifdef _WIN32
#pragma warning( pop )
#endif

#endif
