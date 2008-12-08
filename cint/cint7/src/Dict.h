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
      Reflex::Scope mScopes[G__MAXSTRUCT]; 

      G__Dict() {};
      ~G__Dict() {};
   public:

      static G__Dict &GetDict();

      ::Reflex::Type GetType(int tagnum);
      ::Reflex::Type GetTypeFromId(size_t handle);
      ::Reflex::Type GetTypedef(int typenum);
      ::Reflex::Scope GetScope(int tagnum);
      ::Reflex::Scope GetScope(const struct G__ifunc_table* funcnum);
      ::Reflex::Scope GetScope(const struct G__var_array* varnum);
      ::Reflex::Member GetFunction( const struct G__ifunc_table* funcnum, int ifn );
      ::Reflex::Member GetFunction(size_t handle);
      ::Reflex::Member GetDataMember( const struct G__var_array* varnum, int ig15 );
      ::Reflex::Member GetDataMember(size_t handle);
      int Register(const ::Reflex::Type&);
      bool RegisterScope(int tagnum, const ::Reflex::Scope&);

      size_t GetNumTypes() { return mTypenums.size(); }

   };

   }
}

#endif
