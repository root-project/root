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


#ifndef G__DATAMEMBERHANDLE_H
#define G__DATAMEMBERHANDLE_H 

#include "common.h"

namespace Cint {
   class G__DataMemberHandle {
      int fTagnum;
      int fMemvarNum; 
      int fIndex;
   public:
      G__DataMemberHandle() : fTagnum(-1),fMemvarNum(-1),fIndex(-1) {}
      
      G__var_array *GetVarArray() const {
         G__var_array *var = 0;
         if (fTagnum >= 0) {
            var = G__struct.memvar[fTagnum];
         } else if (fMemvarNum >= 0) {
            var = &G__global;
         }
         for (int i = 0; var!=0 && i < fMemvarNum; ++i ) {            
            var = var->next;
         }
         return(var);
      }
      
      void Set(G__var_array* invar, int index) {
         if (!invar) return;
         
         fIndex = index;
         fTagnum = invar->tagnum;
         G__var_array *var = 0;
         if (fTagnum >= 0) {
            var = G__struct.memvar[fTagnum];
         } else {
            var = &G__global;
         }
         for(int i = 0 ; var != 0; var = var->next, ++i) {
            if (var == invar) {
               fMemvarNum = i;
            }
         }
      }
      
      int GetIndex() const { return fIndex; };
   };
}

#endif

