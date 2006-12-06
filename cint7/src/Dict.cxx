/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file Class.cxx
 ************************************************************************
 * Description:
 *  Dictionary information interface
 ************************************************************************
 * Author                  Philippe Canal 
 * Copyright(c) 1995~2004  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "Api.h"
#include "common.h"
#include "Dict.h"

/*********************************************************************
* class G__ClassInfo
*********************************************************************/
namespace Cint 
{
   namespace Internal 
   {
      
      ///////////////////////////////////////////////////////////////////////////
      G__Dict &G__Dict::GetDict() 
      {
         // Return the current instance
         static G__Dict dict;
         return dict;
      }

      ///////////////////////////////////////////////////////////////////////////
      ::Reflex::Type G__Dict::GetType(int tagnum)
      {
         // Return the Reflex type (class or namespace) corresponding 
         // to the CINT tagnum.
         static ::ROOT::Reflex::Type null_type;
         if (-1 == tagnum) return null_type;
         std::string fulltagname(G__fulltagname(tagnum,0));
         ::ROOT::Reflex::Type t = ::ROOT::Reflex::Type::ByName(fulltagname); // We stored the dollar (at least for now)
         if (!t) {
#ifdef __GNUC__
#else
#pragma message (FIXME("This is very bad we need to fix the std lookup more globabaly"))
#endif
            fulltagname.insert(0, "std::");
	    t = ::ROOT::Reflex::Type::ByName(fulltagname); 
         }
         return t;
      }

      ///////////////////////////////////////////////////////////////////////////
      ::ROOT::Reflex::Scope G__Dict::GetScope(int tagnum)
      {
         // Return the Reflex type (class or namespace) corresponding 
         // to the CINT tagnum.
#ifdef __GNUC__
#else
#pragma message(FIXME("Check that ambiguity is gone between global and invalid scope!"))
#endif
         if (-1 == tagnum) 
            return ::ROOT::Reflex::Scope::GlobalScope();
         std::string fulltagname(G__fulltagname(tagnum,0));
         ::ROOT::Reflex::Scope s = ::ROOT::Reflex::Scope::ByName(fulltagname);
         if (!s) {
#ifdef __GNUC__
#else
#pragma message (FIXME("This is very bad we need to fix the std lookup more globabaly"))
#endif
            fulltagname.insert(0, "std::");
	    s = ::ROOT::Reflex::Type::ByName(fulltagname); 
         }
         return s;
      }

      ////////////////////////////////////////////////////////////////////////
      ::ROOT::Reflex::Type G__Dict::GetTypedef(int typenum)
      {
         // Return the Reflex typedef  corresponding 
         // to the CINT typenum.
         if (
             (typenum < 0) ||
             (typenum >= ((int) mTypenums.size())) ||
             !mTypenums[typenum].IsTypedef()
         ) {
            static ::ROOT::Reflex::Type null_type;
            return null_type;
         }
         return mTypenums[typenum];
      }

      ///////////////////////////////////////////////////////////////////////////
      int G__Dict::Register(const ::ROOT::Reflex::Type& in) 
      {
         // Register a Reflex type and returns the assigned 
         // CINT number (used for emulating typenum and tagnum).
         mTypenums.push_back(in);
         //fprintf(stderr, "%06d, Registered type '%s'\n", mTypenums.size() - 1, in.Name().c_str());
         return mTypenums.size()-1;
      }


   } // End of Internal
} // End of Cint
