// @(#)root/treeplayer:$Name:  $:$Id: TBranchProxyDescriptor.cxx,v 1.2 2004/06/28 05:29:07 brun Exp $
// Author: Philippe Canal 06/06/2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TBranchProxyDescriptor.h"

#include "TClass.h"
#include "TClonesArray.h"
#include "TError.h"
#include "TROOT.h"

#include "TTreeFormula.h"
#include "TFormLeafInfo.h"

namespace ROOT {

   TBranchProxyDescriptor::TBranchProxyDescriptor(const char *dataname, 
                                                  const char *type, 
                                                  const char *branchname, 
                                                  Bool_t split) :
      TNamed(dataname,type),fBranchName(branchname),fIsSplit(split) 
   {
      fDataName = GetName();
      fDataName.ReplaceAll("<","_");
      fDataName.ReplaceAll(">","_");     
   }
   
   const char *TBranchProxyDescriptor::GetDataName() 
   { 
      return fDataName; 
   }

   const char *TBranchProxyDescriptor::GetTypeName() 
   { 
      return GetTitle(); 
   }
   
   const char *TBranchProxyDescriptor::GetBranchName() 
   { 
      return fBranchName.Data(); 
   }

   Bool_t TBranchProxyDescriptor::IsEquivalent(const TBranchProxyDescriptor *other,
                                               Bool_t inClass) 
   {
      // Return true if this description is the 'same' as the other decription.

      if ( !other ) return false;

      if ( inClass ) {
         // If this description belong to a class, the branchname will be 
         // stripped.
      } else {
         if ( fBranchName != other->fBranchName ) return false;
      }
      if ( fIsSplit != other->fIsSplit ) return false;
      if ( strcmp(GetName(),other->GetName()) ) return false;
      if ( strcmp(GetTitle(),other->GetTitle()) ) return false;
      return true;
   }

   Bool_t TBranchProxyDescriptor::IsSplit() const 
   { 
      return fIsSplit; 
   }

   void TBranchProxyDescriptor::OutputDecl(FILE *hf, int offset, UInt_t maxVarname)
   {
      fprintf(hf,"%-*s%-*s %s;\n",  offset," ",  maxVarname, GetTypeName(), GetDataName()); // might want to add a comment
   }

   void TBranchProxyDescriptor::OutputInit(FILE *hf, int offset, 
                                           UInt_t maxVarname,
                                           const char *prefix) 
   {
      if (fIsSplit) {
         const char *subbranchname = GetBranchName();
         const char *above = "";
         if (strncmp(prefix,subbranchname,strlen(prefix))==0
             && strcmp(prefix,subbranchname)!=0)  {
            subbranchname += strlen(prefix)+1; // +1 for the dot "."
            above = "ffPrefix, ";
         }

         fprintf(hf,"\n%-*s      %-*s(director, %s\"%s\")",
                 offset," ", maxVarname, GetDataName(), above, subbranchname);
      } else {

         fprintf(hf,"\n%-*s      %-*s(director, obj.proxy(), \"%s\")",
                 offset," ", maxVarname, GetDataName(), GetBranchName() );

         //fprintf(hf,"\n%-*s      %-*s(director, ffPrefix, \"\", \"%s\")",
         //        offset," ", maxVarname, GetName(), GetBranchName() );

      }
   }
}
