// @(#)root/base:$Name:  $:$Id: TClassEdit.h,v 1.4 2004/01/29 23:08:16 brun Exp $
// Author: Victor Perev   10/04/2003

#ifndef ROOT_TClassEdit
#define ROOT_TClassEdit

#include "RConfig.h"
#include <string>
#include <vector>

#ifdef R__HPUX
namespace std {
   using ::string;
   using ::vector;
}
#endif

// TClassEdit is used to manipulate class and type names.
//
// This class does not dependent on any other ROOT facility
// so that it can be used by rootcint.

class TClassEdit{
public:
   enum { kDropTrailStar    = 1<<0,
          kDropDefaultAlloc = 1<<1,
          kDropAlloc        = 1<<2,
          kInnerClass       = 1<<3,
          kInnedMostClass   = 1<<4,
          kDropStlDefault   = 1<<5, /* implies kDropDefaultAlloc */
          kDropComparator   = 1<<6  /* if the class has a comparator, drops BOTH the comparator and the Allocator */
   };

   enum { 
      kNone     = 0,
      kVector   = 1,
      kList     = 2,
      kDeque    = 3,
      kMap      = 4,
      kMultiMap = 5,
      kSet      = 6,
      kMultiSet = 7,
      kEnd      = 8
   };

          
   static std::string CleanType (const char *typeDesc,int mode = 0,const char **tail=0);
   static bool        IsDefAlloc(const char *alloc, const char *classname);
   static bool        IsDefAlloc(const char *alloc, const char *keyclassname, const char *valueclassname);
   static bool        IsDefComp (const char *comp , const char *classname);
   static int         IsSTLCont (const char *type,int testAlloc=0);
   static bool        IsStdClass(const char *type);
   static bool        IsVectorBool(const char *name);
   static int         GetSplit  (const char *type, std::vector<std::string> &output);
   static int         STLKind   (const char *type);    //Kind of stl container
   static int         STLArgs   (int kind);            //Min number of arguments without allocator
   static std::string ShortType (const char *typeDesc, int mode);
};

#endif
