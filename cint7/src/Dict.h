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
      std::vector<ROOT::Reflex::Type> mTypenums; // link between typenums/tagnums and Reflex types

   public:
       G__Dict() {};
      ~G__Dict() {};

      static G__Dict &GetDict();

      ::ROOT::Reflex::Type GetType(int tagnum);
      ::ROOT::Reflex::Scope GetScope(int tagnum);
      ::ROOT::Reflex::Type GetTypedef(int typenum);
      int Register(const ::ROOT::Reflex::Type&);

      int GetNumTypes() { return mTypenums.size(); }

   };

   }
}

#endif
