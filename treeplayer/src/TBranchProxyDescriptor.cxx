// @(#)root/treeplayer:$Name:  $:$Id: TFormLeafInfo.h,v 1.1 2004/06/17 17:37:10 brun Exp $
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
                                                  bool split) :
      TNamed(dataname,type),fBranchName(branchname),fIsSplit(split) 
   {
   }
   
   const char *TBranchProxyDescriptor::GetDataName() 
   { 
      return GetName(); 
   }

   const char *TBranchProxyDescriptor::GetTypeName() 
   { 
      return GetTitle(); 
   }
   
   const char *TBranchProxyDescriptor::GetBranchName() 
   { 
      return fBranchName.Data(); 
   }

   bool TBranchProxyDescriptor::IsEqual(const TBranchProxyDescriptor *other) 
   {
      if ( !other ) return false;
      if ( fBranchName != other->fBranchName ) return false;
      if ( fIsSplit != other->fIsSplit ) return false;
      if ( strcmp(GetName(),other->GetName()) ) return false;
      if ( strcmp(GetTitle(),other->GetTitle()) ) return false;
      return true;
   }

   bool TBranchProxyDescriptor::IsSplit() const 
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
                 offset," ", maxVarname, GetName(), above, subbranchname);
      } else {

         fprintf(hf,"\n%-*s      %-*s(director, obj.proxy(), \"%s\")",
                 offset," ", maxVarname, GetName(), GetBranchName() );

         //fprintf(hf,"\n%-*s      %-*s(director, ffPrefix, \"\", \"%s\")",
         //        offset," ", maxVarname, GetName(), GetBranchName() );

      }
   }
}
