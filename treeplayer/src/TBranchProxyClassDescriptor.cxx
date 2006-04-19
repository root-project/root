// @(#)root/treeplayer:$Name:  $:$Id: TBranchProxyClassDescriptor.cxx,v 1.8 2005/11/11 23:21:43 pcanal Exp $
// Author: Philippe Canal 06/06/2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TBranchProxyDescriptor.h"
#include "TBranchProxyClassDescriptor.h"

#include "TROOT.h"
#include "TClass.h"
#include "TError.h"
#include "TStreamerInfo.h"

namespace ROOT {

   void TBranchProxyClassDescriptor::NameToSymbol() {

      // Make the typename a proper class name without having the really deal with
      // namespace and templates.

      fRawSymbol = GetName();
      fRawSymbol.ReplaceAll(":","_");
      fRawSymbol.ReplaceAll("<","_");
      fRawSymbol.ReplaceAll(">","_");
      fRawSymbol.ReplaceAll(" ","");
      fRawSymbol.ReplaceAll("*","st");
      fRawSymbol.ReplaceAll("&","rf");
      if (fIsClones!=kOut)
         fRawSymbol.Prepend("TClaPx_");
      else
         fRawSymbol.Prepend("TPx_");
      if (fRawSymbol.Length() && fRawSymbol[fRawSymbol.Length()-1]=='.')
         fRawSymbol.Remove(fRawSymbol.Length()-1);

      SetName(fRawSymbol);
   }

   TBranchProxyClassDescriptor::TBranchProxyClassDescriptor(const char *type,
                                                            TStreamerInfo *info,
                                                            const char *branchname,
                                                            UInt_t isclones,
                                                            UInt_t splitlevel) :
      TNamed(type,type),
      fIsClones(isclones),
      fIsLeafList(false),
      fSplitLevel(splitlevel),
      fBranchName(branchname),
      fSubBranchPrefix(branchname),
      fInfo(info),
      fMaxDatamemberType(3)
   {
      // Constructor.

      R__ASSERT( strcmp(fInfo->GetName(), type)==0 );
      NameToSymbol();
      if (fSubBranchPrefix.Length() && fSubBranchPrefix[fSubBranchPrefix.Length()-1]=='.') fSubBranchPrefix.Remove(fSubBranchPrefix.Length()-1);
   }

   TBranchProxyClassDescriptor::TBranchProxyClassDescriptor(const char *branchname) :
      TNamed(branchname,branchname),
      fIsClones(false),
      fIsLeafList(true),
      fSplitLevel(0),
      fBranchName(branchname),
      fSubBranchPrefix(branchname),
      fInfo(0),
      fMaxDatamemberType(3)
   {
      // Constructor for a branch constructed from a leaf list.

      NameToSymbol();
      if (fSubBranchPrefix.Length() && fSubBranchPrefix[fSubBranchPrefix.Length()-1]=='.') fSubBranchPrefix.Remove(fSubBranchPrefix.Length()-1);
   }

   TBranchProxyClassDescriptor::TBranchProxyClassDescriptor(const char *type, TStreamerInfo *info,
                                                            const char *branchname,
                                                            const char *branchPrefix, UInt_t isclones,
                                                            UInt_t splitlevel) :
      TNamed(type,type),
      fIsClones(isclones),
      fIsLeafList(true),
      fSplitLevel(splitlevel),
      fBranchName(branchname),
      fSubBranchPrefix(branchPrefix),
      fInfo(info),
      fMaxDatamemberType(3)
   {
      // Constructor.

      R__ASSERT( strcmp(fInfo->GetName(), type)==0 );
      NameToSymbol();
      if (fSubBranchPrefix.Length() && fSubBranchPrefix[fSubBranchPrefix.Length()-1]=='.') fSubBranchPrefix.Remove(fSubBranchPrefix.Length()-1);
   }

   const char* TBranchProxyClassDescriptor::GetBranchName() const
   {
      // Get the branch name
      return fBranchName.Data();
   }

   const char* TBranchProxyClassDescriptor::GetSubBranchPrefix() const
   {
      // Get the prefix from the branch name
      return fSubBranchPrefix.Data();
   }

   const char* TBranchProxyClassDescriptor::GetRawSymbol() const
   {
      // Get the real symbol name

      return fRawSymbol;
   }

   UInt_t TBranchProxyClassDescriptor::GetSplitLevel() const {
      // Return the split level of the branch.
      return fSplitLevel;
   }

   Bool_t TBranchProxyClassDescriptor::IsEquivalent(const TBranchProxyClassDescriptor* other)
   {
      // Return true if this description is the 'same' as the other decription.

      if ( !other ) return kFALSE;
      // Purposely do not test on the name!
      if ( strcmp(GetTitle(),other->GetTitle()) ) return kFALSE;
      // if ( fBranchName != other->fBranchName ) return kFALSE;
      // if ( fSubBranchPrefix != other->fSubBranchPrefix ) return kFALSE;

      if (fIsClones != other->fIsClones) return kFALSE;

      TBranchProxyDescriptor *desc;
      TBranchProxyDescriptor *othdesc;

      if ( fListOfBaseProxies.GetSize() != other->fListOfBaseProxies.GetSize() ) return kFALSE;
      TIter next(&fListOfBaseProxies);
      TIter othnext(&other->fListOfBaseProxies);
      while ( (desc=(TBranchProxyDescriptor*)next()) ) {
         othdesc=(TBranchProxyDescriptor*)othnext();
         if (!desc->IsEquivalent(othdesc,kTRUE) ) return kFALSE;
      }

      if ( fListOfSubProxies.GetSize() != other->fListOfSubProxies.GetSize() ) return kFALSE;
      next = &fListOfSubProxies;
      othnext = &(other->fListOfSubProxies);

      while ( (desc=(TBranchProxyDescriptor*)next()) ) {
         othdesc=(TBranchProxyDescriptor*)othnext();
         if (!desc->IsEquivalent(othdesc,kTRUE)) return kFALSE;
         if (desc->IsSplit()) {
            TString leftname (  desc->GetBranchName() );
            TString rightname(  othdesc->GetBranchName() );

            if (leftname.Index(GetBranchName())==0) leftname.Remove( 0,strlen(GetBranchName()));
            if (leftname.Length() && leftname[0]=='.') leftname.Remove(0,1);
            if (rightname.Index(other->GetBranchName())==0) rightname.Remove(0,strlen(other->GetBranchName()));
            if (rightname.Length() && rightname[0]=='.') rightname.Remove(0,1);
            if (leftname != rightname ) return kFALSE;
         }
      }
      return true;
   }

   void TBranchProxyClassDescriptor::AddDescriptor(TBranchProxyDescriptor *desc, Bool_t isBase)
   {
      // Add a descriptor to this proxy.

      if (desc) {
         if (isBase) {
            fListOfBaseProxies.Add(desc);
         } else {
            fListOfSubProxies.Add(desc);
            UInt_t len = strlen(desc->GetTypeName());
            if ((len+2)>fMaxDatamemberType) fMaxDatamemberType = len+2;
         }
      }
   }

   Bool_t TBranchProxyClassDescriptor::IsLoaded() const
   {
      // Return true if the class needed by the branch is loaded
      TClass *cl = gROOT->GetClass(GetTitle());
      return (cl && cl->IsLoaded());
   }

   Bool_t TBranchProxyClassDescriptor::IsClones() const
   {
      // Return true if this proxy is for a TClonesArray.
      return fIsClones!=kOut;
   }
   UInt_t TBranchProxyClassDescriptor::GetIsClones() const
   {
      // Return whether the branch is inside, nested in or outside of a TClonesArray
      return fIsClones;
   }

   void TBranchProxyClassDescriptor::OutputDecl(FILE *hf, int offset, UInt_t /* maxVarname */)
   {
      // Output the declaration and implementation of this emulation class

      TBranchProxyDescriptor *desc;


      // Start the class declaration with the eventual list of base classes
      fprintf(hf,"%-*sstruct %s\n", offset," ", GetName() );

      if (fListOfBaseProxies.GetSize()) {
         fprintf(hf,"%-*s   : ", offset," ");

         TIter next(&fListOfBaseProxies);

         desc = (TBranchProxyDescriptor*)next();
         fprintf(hf,"public %s", desc->GetTypeName());

         while ( (desc = (TBranchProxyDescriptor*)next()) ) {
            fprintf(hf,",\n%-*spublic %s", offset+5," ", desc->GetTypeName());
         }

         fprintf(hf,"\n");
      }
      fprintf(hf,"%-*s{\n", offset," ");


      // Write the constructor
      fprintf(hf,"%-*s   %s(TBranchProxyDirector* director,const char *top,const char *mid=0) :",
              offset," ", GetName());

      Bool_t wroteFirst = kFALSE;

      if (fListOfBaseProxies.GetSize()) {

         TIter next(&fListOfBaseProxies);

         desc = (TBranchProxyDescriptor*)next();
         fprintf(hf,"\n%-*s%-*s(director, top, mid)",  offset+6, " ", fMaxDatamemberType,desc->GetTypeName());
         wroteFirst = true;

         while ( (desc = (TBranchProxyDescriptor*)next()) ) {
            fprintf(hf,",\n%-*s%-*s(director, top, mid)",  offset+6, " ", fMaxDatamemberType,desc->GetTypeName());
         }

      }
      fprintf(hf,"%s\n%-*s      %-*s(top,mid)",wroteFirst?",":"",offset," ",fMaxDatamemberType,"ffPrefix");
      wroteFirst = true;


      TString objInit = "top, mid";
      if ( GetIsClones() == kInsideClones ) {
         if (fListOfSubProxies.GetSize()) {
            desc = (TBranchProxyDescriptor*)fListOfSubProxies.At(0);
            if (desc && desc->IsSplit()) {

               // In the case of a split sub object is TClonesArray, the
               // object itself does not have its own branch, so we need to
               // use its first (semantic) sub-branch as a proxy

               TString main = GetBranchName();
               TString sub = desc->GetBranchName();
               sub.Remove(0,main.Length()+1);

               objInit  = "ffPrefix, ";
               objInit += "\"";
               objInit += sub;
               objInit += "\"";

               objInit = "top, \"\", mid";
            }
         }
      }

      fprintf(hf,"%s\n%-*s      %-*s(director, %s)",
              wroteFirst?",":"",offset," ",fMaxDatamemberType,"obj",objInit.Data());
      wroteFirst = true;

      TIter next(&fListOfSubProxies);
      while ( (desc = (TBranchProxyDescriptor*)next()) ) {
         if (wroteFirst) fprintf(hf,",");
         desc->OutputInit(hf,offset,fMaxDatamemberType,GetSubBranchPrefix());
         wroteFirst = true;
      }
      fprintf(hf,"\n%-*s   {};\n",offset," ");


      // Write the 2nd constructor
      fprintf(hf,"%-*s   %s(TBranchProxyDirector* director, TBranchProxy *parent, const char *membername) :",
              offset," ", GetName());

      wroteFirst = kFALSE;

      if (fListOfBaseProxies.GetSize()) {

         TIter next(&fListOfBaseProxies);

         desc = (TBranchProxyDescriptor*)next();
         fprintf(hf,"\n%-*s%-*s(director, parent, membername)",  offset+6, " ", fMaxDatamemberType,desc->GetTypeName());
         wroteFirst = true;

         while ( (desc = (TBranchProxyDescriptor*)next()) ) {
            fprintf(hf,",\n%-*s%-*s(director, parent, membername)",  offset+6, " ", fMaxDatamemberType,desc->GetTypeName());
         }

      }
      fprintf(hf,"%s\n%-*s      %-*s(\"\")",wroteFirst?",":"",offset," ",fMaxDatamemberType,"ffPrefix");
      wroteFirst = true;

      if ( true ||  IsLoaded() || IsClones() ) {
         fprintf(hf,"%s\n%-*s      %-*s(director, parent, membername)",
                 wroteFirst?",":"",offset," ",fMaxDatamemberType,"obj");
         wroteFirst = true;
      }

      next = &fListOfSubProxies;
      while ( (desc = (TBranchProxyDescriptor*)next()) ) {
         if (wroteFirst) fprintf(hf,",");
         desc->OutputInit(hf,offset,fMaxDatamemberType,GetSubBranchPrefix());
         wroteFirst = true;
      }
      fprintf(hf,"\n%-*s   {};\n",offset," ");


      // Declare the data members.
      fprintf(hf,"%-*s%-*s %s;\n",  offset+3," ",  fMaxDatamemberType, "TBranchProxyHelper", "ffPrefix");

      // If the real class is available, make it available via the arrow operator:
      if (IsLoaded()) {

         const char *type = GetTitle(); /* IsClones() ? "TClonesArray" : GetTitle(); */
         fprintf(hf,"%-*sInjecTBranchProxyInterface();\n", offset+3," ");
         //Can the real type contain a leading 'const'? If so the following is incorrect.
         if ( IsClones() ) {
            fprintf(hf,"%-*sconst %s* operator[](int i) { return obj.At(i); }\n", offset+3," ",type);
            fprintf(hf,"%-*sTClaObjProxy<%s > obj;\n", offset+3, " ", type);
         } else {
            fprintf(hf,"%-*sconst %s* operator->() { return obj.GetPtr(); }\n", offset+3," ",type);
            fprintf(hf,"%-*sTObjProxy<%s > obj;\n", offset+3, " ", type);
         }

      } else if ( IsClones()) {

         fprintf(hf,"%-*sInjecTBranchProxyInterface();\n", offset+3," ");
         fprintf(hf,"%-*sconst TClonesArray* operator->() { return obj.GetPtr(); }\n", offset+3," ");
         fprintf(hf,"%-*sTClaProxy obj;\n", offset+3," ");

      } else {

         fprintf(hf,"%-*sInjecTBranchProxyInterface();\n", offset+3," ");
         fprintf(hf,"%-*sTBranchProxy obj;\n", offset+3," ");

      }

      fprintf(hf,"\n");

      next.Reset();
      while( (desc = ( TBranchProxyDescriptor *)next()) ) {
         desc->OutputDecl(hf,offset+3,fMaxDatamemberType);
      }
      fprintf(hf,"%-*s};\n",offset," ");

      //TBranchProxyDescriptor::OutputDecl(hf,offset,maxVarname);
   }

}
