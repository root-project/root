// @(#)root/base:$Name:  $:$Id: TClassEdit.h,v 1.6 2004/01/31 08:59:09 brun Exp $
// Author: Victor Perev   10/04/2003
//         Philippe Canal 05/2004

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

namespace TClassEdit {

   typedef enum { 
      kDropTrailStar    = 1<<0,
      kDropDefaultAlloc = 1<<1,
      kDropAlloc        = 1<<2,
      kInnerClass       = 1<<3,
      kInnedMostClass   = 1<<4,
      kDropStlDefault   = 1<<5, /* implies kDropDefaultAlloc */
      kDropComparator   = 1<<6  /* if the class has a comparator, drops BOTH the comparator and the Allocator */
   } EModType;

   typedef enum { 
      kNone     = 0,
      kVector   = 1,
      kList     = 2,
      kDeque    = 3,
      kMap      = 4,
      kMultiMap = 5,
      kSet      = 6,
      kMultiSet = 7,
      kEnd      = 8
   } ESTLType;

          
   std::string CleanType (const char *typeDesc,int mode = 0,const char **tail=0);
   bool        IsDefAlloc(const char *alloc, const char *classname);
   bool        IsDefAlloc(const char *alloc, const char *keyclassname, const char *valueclassname);
   bool        IsDefComp (const char *comp , const char *classname);
   int         IsSTLCont (const char *type,int testAlloc=0);
   bool        IsStdClass(const char *type);
   bool        IsVectorBool(const char *name);
   std::string GetLong64_Name(const std::string& original);
   int         GetSplit  (const char *type, std::vector<std::string> &output);
   int         STLKind   (const char *type);    //Kind of stl container
   int         STLArgs   (int kind);            //Min number of arguments without allocator
   std::string ResolveTypedef(const char *tname, bool resolveAll = false);
   std::string ShortType (const char *typeDesc, int mode);
};

#endif
