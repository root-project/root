// @(#)root/treeplayer:$Id$
// Author: Philippe Canal 06/06/2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TBranchProxyDescriptor
/Hold the processed information about a TBranch while
/TTreeProxyGenerator is parsing the TTree information.
/Also contains the routine use to generate the appropriate code
/fragment in the result of MakeProxy.
*/

#include "TBranchProxyDescriptor.h"

#include "TClonesArray.h"
#include "TError.h"
#include "TROOT.h"

#include "TTreeFormula.h"
#include "TFormLeafInfo.h"
#include <ctype.h>

ClassImp(ROOT::Internal::TBranchProxyDescriptor);

namespace ROOT {
namespace Internal {

   ////////////////////////////////////////////////////////////////////////////////
   /// Constructor.

   TBranchProxyDescriptor::TBranchProxyDescriptor(const char *dataname,
                                                  const char *type,
                                                  const char *branchname,
                                                  Bool_t split,
                                                  Bool_t skipped,
                                                  Bool_t isleaflist) :
      TNamed(dataname,type),fBranchName(branchname),fIsSplit(split),fBranchIsSkipped(skipped),fIsLeafList(isleaflist)
   {
      fDataName = GetName();
      if (fDataName.Length() && fDataName[fDataName.Length()-1]=='.') fDataName.Remove(fDataName.Length()-1);

      fDataName.ReplaceAll(".","_");
      fDataName.ReplaceAll(":","_");
      fDataName.ReplaceAll("<","_");
      fDataName.ReplaceAll(">","_");
      if (!isalpha(fDataName[0])) fDataName.Insert(0,"_");
      fDataName.ReplaceAll(" ","");
      fDataName.ReplaceAll("*","st");
      fDataName.ReplaceAll("&","rf");

   }

   ////////////////////////////////////////////////////////////////////////////////
   /// Get the name of the data member.

   const char *TBranchProxyDescriptor::GetDataName()
   {
      return fDataName;
   }

   ////////////////////////////////////////////////////////////////////////////////
   /// Get the name of the type of the data member

   const char *TBranchProxyDescriptor::GetTypeName()
   {
      return GetTitle();
   }

   ////////////////////////////////////////////////////////////////////////////////
   /// Get the branch name.

   const char *TBranchProxyDescriptor::GetBranchName()
   {
      return fBranchName.Data();
   }

   ////////////////////////////////////////////////////////////////////////////////
   /// Return true if this description is the 'same' as the other description.

   Bool_t TBranchProxyDescriptor::IsEquivalent(const TBranchProxyDescriptor *other,
                                               Bool_t inClass)
   {
      if ( !other ) return false;
      if ( other == this ) return true;

      if ( inClass ) {
         // If this description belong to a class, the branchname will be
         // stripped.
      } else {
         if ( fBranchName != other->fBranchName ) return false;
      }
      if ( fIsSplit != other->fIsSplit ) return false;
      if ( fBranchIsSkipped != other->fBranchIsSkipped) return false;
      if ( strcmp(GetName(),other->GetName()) ) return false;
      if ( strcmp(GetTitle(),other->GetTitle()) ) return false;
      return true;
   }

   ////////////////////////////////////////////////////////////////////////////////
   /// Return true if the branch is split

   Bool_t TBranchProxyDescriptor::IsSplit() const
   {
      return fIsSplit;
   }

   ////////////////////////////////////////////////////////////////////////////////
  /// Output the declaration corresponding to this proxy

   void TBranchProxyDescriptor::OutputDecl(FILE *hf, int offset, UInt_t maxVarname)
   {
      fprintf(hf,"%-*s%-*s %s;\n",  offset," ",  maxVarname, GetTypeName(), GetDataName()); // might want to add a comment
   }

   ////////////////////////////////////////////////////////////////////////////////
   /// Output the initialization corresponding to this proxy

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

         if (fBranchIsSkipped) {
            fprintf(hf,"\n%-*s      %-*s(director, obj.GetProxy(), \"%s\", %s\"%s\")",
                    offset," ", maxVarname, GetDataName(), GetDataName(), above, subbranchname);
         } else {
            if (fIsLeafList) {
               if (above[0]=='\0') {
                  fprintf(hf,"\n%-*s      %-*s(director, \"%s\", \"\", \"%s\")",
                          offset," ", maxVarname, GetDataName(), subbranchname, GetDataName());
               } else {
                  fprintf(hf,"\n%-*s      %-*s(director, %s\"%s\", \"%s\")",
                          offset," ", maxVarname, GetDataName(), above, subbranchname, GetDataName());
               }
            } else {
               fprintf(hf,"\n%-*s      %-*s(director, %s\"%s\")",
                       offset," ", maxVarname, GetDataName(), above, subbranchname);
            }
         }
      } else {

         fprintf(hf,"\n%-*s      %-*s(director, obj.GetProxy(), \"%s\")",
                 offset," ", maxVarname, GetDataName(), GetBranchName() );

         //fprintf(hf,"\n%-*s      %-*s(director, ffPrefix, \"\", \"%s\")",
         //        offset," ", maxVarname, GetName(), GetBranchName() );

      }
   }

} // namespace Internal
} // namespace ROOT
