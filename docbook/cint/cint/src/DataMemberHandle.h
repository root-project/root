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
      G__DataMemberHandle() : fTagnum(-1),fMemvarNum(-1),fIndex(-1) 
      {
         // Default constructor.
      }

      G__DataMemberHandle(G__var_array* invar, int index) : fTagnum(-1),fMemvarNum(-1),fIndex(-1) 
      {
         // Set this member to point to the given variable.
         Set(invar,index);
      }
      
      int DeleteVariable();
      
      int GetIndex() const 
      { 
         // Return the index within the var_array for this member.
         return fIndex; 
      };

      G__var_array *GetVarArray() const {
         // Return the var array for this data member.
         
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
      
      void Set(G__var_array* invar, int index, int index_of_invar = -1) {
         // Set this member to point to the given variable.
         // This function is used in one of the constructors and must not be virtual
         // If index_of_invar is given, rather than recalculating we trust this index.
         
         if (!invar) return;
         
         fIndex = index;
         fTagnum = invar->tagnum;
         if (index_of_invar >= 0) {
            fMemvarNum = index_of_invar;
         } else {
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
      }
      
   };
}

#endif

