// @(#)root/base:$Id$
// Author: Victor Perev   10/04/2003
//         Philippe Canal 05/2004

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TClassEdit
#define ROOT_TClassEdit

#include "RConfig.h"
#include <string>
#include <vector>

#ifdef R__OLDHPACC
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

   enum EModType {
      kNone             = 0,
      kDropTrailStar    = 1<<0,
      kDropDefaultAlloc = 1<<1,
      kDropAlloc        = 1<<2,
      kInnerClass       = 1<<3,
      kInnedMostClass   = 1<<4,
      kDropStlDefault   = 1<<5, /* implies kDropDefaultAlloc */
      kDropComparator   = 1<<6, /* if the class has a comparator, drops BOTH the comparator and the Allocator */
      kDropAllDefault   = 1<<7, /* Drop default template parameter even in non STL classes */
      kLong64           = 1<<8, /* replace all 'long long' with Long64_t. */
      kDropStd          = 1<<9  /* Drop any std:: */
   };

   enum ESTLType {
      kNotSTL   = 0,
      kVector   = 1,
      kList     = 2,
      kDeque    = 3,
      kMap      = 4,
      kMultiMap = 5,
      kSet      = 6,
      kMultiSet = 7,
      kBitSet   = 8,
      kEnd      = 9
   };

   struct TSplitType {
      
      const char *fName; // Original spelling of the name.
      std::vector<std::string> fElements;
      int fNestedLocation; // Stores the location of the tail (nested names) in nestedLoc (0 indicates no tail).

      TSplitType(const char *type2split, EModType mode = TClassEdit::kNone);

      int  IsSTLCont(int testAlloc=0) const;
      void ShortType(std::string &answer, int mode);

   private:
      TSplitType(const TSplitType&); // intentionally not implemented
      TSplitType &operator=(const TSplitType &); // intentionally not implemented
   };

   std::string CleanType (const char *typeDesc,int mode = 0,const char **tail=0);
   bool        IsDefAlloc(const char *alloc, const char *classname);
   bool        IsDefAlloc(const char *alloc, const char *keyclassname, const char *valueclassname);
   bool        IsDefComp (const char *comp , const char *classname);
   bool        IsSTLBitset(const char *type);
   int         IsSTLCont (const char *type,int testAlloc=0);
   bool        IsStdClass(const char *type);
   bool        IsVectorBool(const char *name);
   std::string GetLong64_Name(const std::string& original);
   int         GetSplit  (const char *type, std::vector<std::string> &output, int &nestedLoc, EModType mode = TClassEdit::kNone);
   int         STLKind   (const char *type);    //Kind of stl container
   int         STLArgs   (int kind);            //Min number of arguments without allocator
   std::string ResolveTypedef(const char *tname, bool resolveAll = false);
   std::string ShortType (const char *typeDesc, int mode);
   std::string InsertStd(const char *tname);
}

#endif
