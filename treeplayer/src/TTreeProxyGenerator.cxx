// @(#)root/treeplayer:$Name:  $:$Id: TTreeProxyGenerator.cxx,v 1.2 2004/06/28 05:29:07 brun Exp $
// Author: Philippe Canal 06/06/2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*
  TODO:
  Have separate names for the wrapper classes in the cases of: [done]
  clones/non clones
  split/non split
  split levels

  Have a solution for passing top+"."+middle to the parents classes [probably done .. need testing]

  Have a solution for the return by references of abstract classes [not done]

  Have object inside ClonesArray properly treated! [done]
  Why is there 2 TRef proxy classes? [done]

  check why some inheritance are TObjProxy and not TPx_

  Be smart enough to avoid issue about having 2 classes one unrolled and one non unrolled!

  When using in interpreted mode understand why the reloading reloads the calling script and then crashes :(

  CINT does not properly call the custom operators when doing return fNtrack.

  CINT does not handle fMatrix[2][1] well.

  The user's function in script.h are not exposed by ACLiC.

  Revice the method to avoid the useless refreshing of the generated file
  - for most efficiency it would require a different name for each tree
*/

#include "TTreeProxyGenerator.h"

#include "TBranchProxyDescriptor.h"
#include "TBranchProxyClassDescriptor.h"

#include "TList.h"
#include <stdio.h>

class TTree;
class TBranch;
class TStreamerElement;

#include "TClass.h"
#include "TClonesArray.h"
#include "TError.h"
#include "TROOT.h"

#include "TTreeFormula.h"
#include "TFormLeafInfo.h"


#include "TBranchElement.h"
#include "TChain.h"
#include "TFile.h"
#include "TLeaf.h"
#include "TTree.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TSystem.h"
#include "TLeafObject.h"

namespace {

   Bool_t AreDifferent(const TString& from, const TString& to)
   {
      FILE *left = fopen(from.Data(),"r");
      FILE *right = fopen(to.Data(),"r");

      char leftbuffer[256];
      char rightbuffer[256];

      char *lvalue,*rvalue;

      Bool_t areEqual = kTRUE;

      do {
         lvalue = fgets(leftbuffer, sizeof(leftbuffer), left);
         rvalue = fgets(rightbuffer, sizeof(rightbuffer), right);

         if (lvalue&&rvalue) {
            if (strstr(lvalue,"by ROOT version")) {
               // skip the comment line with the time and date
            } else {
               areEqual = areEqual && (0 == strncmp(lvalue,rvalue,sizeof(leftbuffer)));
            }
         }
         if (lvalue&&!rvalue) areEqual = kFALSE;
         if (rvalue&&!lvalue) areEqual = kFALSE;

      } while(areEqual && lvalue && rvalue);

      fclose(left);
      fclose(right);

      return !areEqual;
   }
}

namespace ROOT {

   TString GetArrayType(TStreamerElement *element, const char *subtype,
                        TTreeProxyGenerator::EContainer container)
   {
      TString result;
      int ndim = 0;
      if (element && element->InheritsFrom(TStreamerBasicPointer::Class())) {
         TStreamerBasicPointer * elem = (TStreamerBasicPointer*)element;
         const char *countname = elem->GetCountName();
         if (countname && strlen(countname)>0) ndim = 1;
      }
      ndim += element->GetArrayDim();

      TString middle;
      if (container == TTreeProxyGenerator::kClones) {
         middle = "Cla";
      }

      if (ndim==0) {
         result = "T";
         result += middle;
         result += subtype;
         result += "Proxy";
      } else if (ndim==1) {
         result = "T";
         result += middle;
         result += "Array";
         result += subtype;
         result += "Proxy";
      } else if (ndim==2) {
         result = "T";
         result += middle;
         result += "Array2Proxy<";
         result += element->GetTypeName();
         result += ",";
         result += element->GetMaxIndex(1);
         result += " >";
      }  else if (ndim==3) {
         result = "T";
         result += middle;
         result += "Array3Proxy<";
         result += element->GetTypeName();
         result += ",";
         result += element->GetMaxIndex(1);
         result += ",";
         result += element->GetMaxIndex(2);
         result += " >";
      } else {
         Error("GetArrayTyep","Array of more than 3 dimentsions not implemented yet.");
      }
      return result;

      /*
        if (!strcmp("unsigned int", name))
        sprintf(line, "%u", *(unsigned int *)buf);
        else if (!strcmp("int", name))
        sprintf(line, "%d", *(int *)buf);
        else if (!strcmp("unsigned long", name))
        sprintf(line, "%lu", *(unsigned long *)buf);
        else if (!strcmp("long", name))
        sprintf(line, "%ld", *(long *)buf);
        else if (!strcmp("unsigned short", name))
        sprintf(line, "%hu", *(unsigned short *)buf);
        else if (!strcmp("short", name))
        sprintf(line, "%hd", *(short *)buf);
        else if (!strcmp("unsigned char", name))
        sprintf(line, "%u", *(unsigned char *)buf);
        else if (!strcmp("bool", name))
        sprintf(line, "%u", *(unsigned char *)buf);
        else if (!strcmp("char", name))
        sprintf(line, "%d", *(char *)buf);
        else if (!strcmp("float", name))
        sprintf(line, "%g", *(float *)buf);
        else if (!strcmp("double", name))
        sprintf(line, "%g", *(double *)buf);
      */
   }

   TTreeProxyGenerator::TTreeProxyGenerator(TTree* tree,
                                            const char *script,
                                            const char *fileprefix, UInt_t maxUnrolling) :
      fMaxDatamemberType(2),
      fScript(script),
      fCutScript(),
      fPrefix(fileprefix),
      fHeaderFilename(),
      fMaxUnrolling(maxUnrolling),
      fTree(tree)
   {

      AnalyzeTree();

      WriteProxy();
   }

   TTreeProxyGenerator::TTreeProxyGenerator(TTree* tree,
                                            const char *script, const char *cutscript,
                                            const char *fileprefix, UInt_t maxUnrolling) :
      fMaxDatamemberType(2),
      fScript(script),
      fCutScript(cutscript),
      fPrefix(fileprefix),
      fHeaderFilename(),
      fMaxUnrolling(maxUnrolling),
      fTree(tree)
   {

      AnalyzeTree();

      WriteProxy();
   }

   Bool_t TTreeProxyGenerator::NeedToEmulate(TClass *cl, UInt_t /* level */)
   {
      // Return true if we should create a nested class representing this class

      return cl->TestBit(TClass::kIsEmulation);
   }

   TBranchProxyClassDescriptor*
   TTreeProxyGenerator::AddClass( TBranchProxyClassDescriptor* desc )
   {
      if (desc==0) return 0;

      TBranchProxyClassDescriptor *existing =
         (TBranchProxyClassDescriptor*)fListOfClasses(desc->GetName());

      int count = 0;
      while (existing) {
         if (! existing->IsEquivalent( desc )  ) {
            TString newname = desc->GetRawSymbol();
            count++;
            newname += "_";
            newname += count;

            desc->SetName(newname);
            existing = (TBranchProxyClassDescriptor*)fListOfClasses(desc->GetName());
         } else {
            // we already have the exact same class
            delete desc;
            return existing;
         }
      }
      fListOfClasses.Add(desc);
      return desc;
   }

   void TTreeProxyGenerator::AddForward( const char *classname )
   {
      TObject *obj = fListOfForwards.FindObject(classname);
      if (obj) return;

      if (strstr(classname,"<")!=0) {
         // this is a template instantiation.
         // let's ignore it for now

         Error("AddForward","Forward declaration of templated class not implemented yet.");
      } else {
         fListOfForwards.Add(new TNamed(classname,Form("class %s;\n",classname)));
      }
      return;
   }

   void TTreeProxyGenerator::AddForward(TClass *cl)
   {
      if (cl) AddForward(cl->GetName());
   }

   void TTreeProxyGenerator::AddHeader(TClass *cl)
   {
      if (cl==0) return;

      TObject *obj = fListOfHeaders.FindObject(cl->GetName());
      if (obj) return;

      if (cl->GetDeclFileName() && strlen(cl->GetDeclFileName()) ) {
         // Actually we probably should look for the file ..
         TString header = gSystem->BaseName(cl->GetDeclFileName());
         fListOfHeaders.Add(new TNamed(cl->GetName(),Form("#include \"%s\"\n",
                                                          header.Data())));
      }
   }

   void TTreeProxyGenerator::AddHeader(const char *classname)
   {
      AddHeader(gROOT->GetClass(classname));
   }

   void TTreeProxyGenerator::AddDescriptor(TBranchProxyDescriptor *desc)
   {
      if (desc) {
         fListOfTopProxies.Add(desc);
         UInt_t len = strlen(desc->GetTypeName());
         if ((len+2)>fMaxDatamemberType) fMaxDatamemberType = len+2;
      }
   }

   UInt_t TTreeProxyGenerator::AnalyzeBranch(TBranch *branch, UInt_t level,
                                             TBranchProxyClassDescriptor *topdesc)
   {
      // Analyze the branch and populate the TTreeProxyGenerator or the topdesc with
      // its findings.  Sometimes several branch of the mom are also analyzed,
      // the number of such branches is returned (this happens in the case of
      // embedded objects inside an object inside a clones array split more than
      // one level.

      TString proxyTypeName;
      TString prefix;
      bool isBase = false;
      TString dataMemberName;
      TString cname;
      TString middle;
      UInt_t  extraLookedAt = 0;
      Bool_t  isclones = false;
      EContainer container = kNone;

      if (topdesc && topdesc->IsClones()) {
         container = kClones;
         middle = "Cla";
         isclones = true;
      }

      if (branch->IsA()==TBranchElement::Class()) {

         TBranchElement *be = (TBranchElement*)branch;

         Int_t bid = be->GetID();

         TStreamerElement *element = 0;
         TStreamerInfo *info = be->GetInfo();

         if (bid==-2) {
            Error("AnalyzeBranch","Support for branch ID: %d not yet implement.",
                  bid);
         } else if (bid==-1) {
            Error("AnalyzeBranch","Support for branch ID: %d not yet implement.",
                  bid);
         } else if (bid>=0) {

            element = (TStreamerElement *)info->GetElements()->At(bid);

         } else {
            Error("AnalyzeBranch","Support for branch ID: %d not yet implement.",
                  bid);
         }

         if (element) {
            bool ispointer = false;
            switch(element->GetType()) {

               case TStreamerInfo::kChar:    { proxyTypeName = "T" + middle + "CharProxy"; break; }
               case TStreamerInfo::kShort:   { proxyTypeName = "T" + middle + "ShortProxy"; break; }
               case TStreamerInfo::kInt:     { proxyTypeName = "T" + middle + "IntProxy"; break; }
               case TStreamerInfo::kLong:    { proxyTypeName = "T" + middle + "LongProxy"; break; }
               case TStreamerInfo::kLong64:  { proxyTypeName = "T" + middle + "Long64Proxy"; break; }
               case TStreamerInfo::kFloat:   { proxyTypeName = "T" + middle + "FloatProxy"; break; }
               case TStreamerInfo::kDouble:  { proxyTypeName = "T" + middle + "DoubleProxy"; break; }
               case TStreamerInfo::kDouble32:{ proxyTypeName = "T" + middle + "Double32Proxy"; break; }
               case TStreamerInfo::kUChar:   { proxyTypeName = "T" + middle + "UCharProxy"; break; }
               case TStreamerInfo::kUShort:  { proxyTypeName = "T" + middle + "UShortProxy"; break; }
               case TStreamerInfo::kUInt:    { proxyTypeName = "T" + middle + "UIntProxy"; break; }
               case TStreamerInfo::kULong:   { proxyTypeName = "T" + middle + "ULongProxy"; break; }
               case TStreamerInfo::kULong64: { proxyTypeName = "T" + middle + "ULong64Proxy"; break; }
               case TStreamerInfo::kBits:    { proxyTypeName = "T" + middle + "UIntProxy"; break; }

               case TStreamerInfo::kCharStar: { proxyTypeName = GetArrayType(element,"Char",container); break; }

                  // array of basic types  array[8]
               case TStreamerInfo::kOffsetL + TStreamerInfo::kChar:    { proxyTypeName = GetArrayType(element,"Char",container ); break; }
               case TStreamerInfo::kOffsetL + TStreamerInfo::kShort:   { proxyTypeName = GetArrayType(element,"Short",container ); break; }
               case TStreamerInfo::kOffsetL + TStreamerInfo::kInt:     { proxyTypeName = GetArrayType(element,"Int",container ); break; }
               case TStreamerInfo::kOffsetL + TStreamerInfo::kLong:    { proxyTypeName = GetArrayType(element,"Long",container ); break; }
               case TStreamerInfo::kOffsetL + TStreamerInfo::kLong64:  { proxyTypeName = GetArrayType(element,"Long64",container ); break; }
               case TStreamerInfo::kOffsetL + TStreamerInfo::kFloat:   { proxyTypeName = GetArrayType(element,"Float",container ); break; }
               case TStreamerInfo::kOffsetL + TStreamerInfo::kDouble:  { proxyTypeName = GetArrayType(element,"Double",container ); break; }
               case TStreamerInfo::kOffsetL + TStreamerInfo::kDouble32:{ proxyTypeName = GetArrayType(element,"Double32",container ); break; }
               case TStreamerInfo::kOffsetL + TStreamerInfo::kUChar:   { proxyTypeName = GetArrayType(element,"UChar",container ); break; }
               case TStreamerInfo::kOffsetL + TStreamerInfo::kUShort:  { proxyTypeName = GetArrayType(element,"UShort",container ); break; }
               case TStreamerInfo::kOffsetL + TStreamerInfo::kUInt:    { proxyTypeName = GetArrayType(element,"UInt",container ); break; }
               case TStreamerInfo::kOffsetL + TStreamerInfo::kULong:   { proxyTypeName = GetArrayType(element,"ULong",container ); break; }
               case TStreamerInfo::kOffsetL + TStreamerInfo::kULong64: { proxyTypeName = GetArrayType(element,"ULong64",container ); break; }
               case TStreamerInfo::kOffsetL + TStreamerInfo::kBits:    { proxyTypeName = GetArrayType(element,"UInt",container ); break; }

                  // pointer to an array of basic types  array[n]
               case TStreamerInfo::kOffsetP + TStreamerInfo::kChar:    { proxyTypeName = GetArrayType(element,"Char",container ); break; }
               case TStreamerInfo::kOffsetP + TStreamerInfo::kShort:   { proxyTypeName = GetArrayType(element,"Short",container ); break; }
               case TStreamerInfo::kOffsetP + TStreamerInfo::kInt:     { proxyTypeName = GetArrayType(element,"Int",container ); break; }
               case TStreamerInfo::kOffsetP + TStreamerInfo::kLong:    { proxyTypeName = GetArrayType(element,"Long",container ); break; }
               case TStreamerInfo::kOffsetP + TStreamerInfo::kLong64:  { proxyTypeName = GetArrayType(element,"Long64",container ); break; }
               case TStreamerInfo::kOffsetP + TStreamerInfo::kFloat:   { proxyTypeName = GetArrayType(element,"Float",container ); break; }
               case TStreamerInfo::kOffsetP + TStreamerInfo::kDouble:  { proxyTypeName = GetArrayType(element,"Double",container ); break; }
               case TStreamerInfo::kOffsetP + TStreamerInfo::kDouble32:{ proxyTypeName = GetArrayType(element,"Double32",container ); break; }
               case TStreamerInfo::kOffsetP + TStreamerInfo::kUChar:   { proxyTypeName = GetArrayType(element,"UChar",container ); break; }
               case TStreamerInfo::kOffsetP + TStreamerInfo::kUShort:  { proxyTypeName = GetArrayType(element,"UShort",container ); break; }
               case TStreamerInfo::kOffsetP + TStreamerInfo::kUInt:    { proxyTypeName = GetArrayType(element,"UInt",container ); break; }
               case TStreamerInfo::kOffsetP + TStreamerInfo::kULong:   { proxyTypeName = GetArrayType(element,"ULong",container ); break; }
               case TStreamerInfo::kOffsetP + TStreamerInfo::kULong64: { proxyTypeName = GetArrayType(element,"ULong64",container ); break; }
               case TStreamerInfo::kOffsetP + TStreamerInfo::kBits:    { proxyTypeName = GetArrayType(element,"UInt",container ); break; }

                  // array counter //[n]
               case TStreamerInfo::kCounter: { proxyTypeName = "T" + middle + "IntProxy"; break; }


               case TStreamerInfo::kOffsetL + TStreamerInfo::kObjectp:
               case TStreamerInfo::kOffsetL + TStreamerInfo::kObjectP:
               case TStreamerInfo::kObjectp:
               case TStreamerInfo::kObjectP:
               case TStreamerInfo::kAnyp:
               case TStreamerInfo::kAnyP:
                  // set as pointers and fall through to the next switches
                  ispointer = true;
               case TStreamerInfo::kOffsetL + TStreamerInfo::kObject:
               case TStreamerInfo::kObject:
               case TStreamerInfo::kTString:
               case TStreamerInfo::kTNamed:
               case TStreamerInfo::kTObject:
               case TStreamerInfo::kAny: {
                  TClass *cl = element->GetClassPointer();
                  if (cl) {
                     proxyTypeName = Form("T%sObjProxy<%s >", middle.Data(), cl->GetName());
                     cname = cl->GetName();
                     if (cl==TClonesArray::Class()) {
                        isclones = true;
                        cname = be->GetClonesName();
                        if (cname.Length()==0) {
                           // We may have any unsplit clones array
                           Int_t i = branch->GetTree()->GetReadEntry();
                           if (i<0) i = 0;
                           branch->GetEntry(i);
                           char *obj = be->GetObject();

                           const char *ename = 0;
                           ename = element->GetName();

                           TBranchElement *parent = (TBranchElement*)be->GetMother()->GetSubBranch(be);
                           const char *pclname = parent->GetClassName();

                           TClass *clparent = gROOT->GetClass(pclname);
                           // TClass *clm = gROOT->GetClass(GetClassName());
                           Int_t lOffset = 0; // offset in the local streamerInfo.
                           if (clparent) lOffset = clparent->GetStreamerInfo()->GetOffset(ename);
                           else Error("AnalyzeBranch", "Missing parent for %s.", branch->GetName());

                           TClonesArray *arr;
                           if (ispointer) {
                              arr = (TClonesArray*)*(void**)(obj+lOffset);
                           } else {
                              arr = (TClonesArray*)(obj+lOffset);
                           }
                           cname = arr->GetClass()->GetName();

                        }
                        if (cname.Length()==0) {
                           Error("AnalyzeBranch",
                                 "Introspection of TClonesArray in older file not implemented yet.");
                        }
                     }
                  }
                  else Error("AnalyzeBranch",
                             "Missing class for %s.",
                             branch->GetName());
                  if (element->IsA()==TStreamerBase::Class()) {
                     isBase = true;
                     prefix  = "base";
                  }
                  AddForward(cl);
                  AddHeader(cl);
                  break;
               }

               default:
                  Error("AnalyzeBranch",
                        "Unsupported type for %s (%d).",
                        branch->GetName(),element->GetType());

            }

         }

      } else {

         Error("AnalyzeBranch",
               "Non TBranchElement not implemented yet in AnalyzeBranch (this should not happen)");
         return extraLookedAt;

      }

      if ( branch->GetListOfBranches()->GetEntries() > 0 ) {
         // The branch has sub-branch corresponding the split data member of a class


         // See AnalyzeTree for similar code!
         TBranchProxyClassDescriptor *cldesc = 0;

         TClass *cl = gROOT->GetClass(cname);
         if (cl) {
            cldesc = new TBranchProxyClassDescriptor(cl->GetName(), branch->GetName(),
                                                     isclones, branch->GetSplitLevel());
         }

         if (cldesc) {
            TBranch *subbranch;
            TIter subnext( branch->GetListOfBranches() );
            while ( (subbranch = (TBranch*)subnext()) ) {
               Int_t skipped = AnalyzeBranch(subbranch,level+1,cldesc);
               Int_t s = 0;
               while( s<skipped && subnext() ) { s++; };
            }

            TBranchProxyClassDescriptor *added = AddClass(cldesc);
            if (added) proxyTypeName = added->GetName();
            // this codes and the previous 2 lines move from inside the if (cl)
            // aboce and this line was used to avoid unecessary work:
            // if (added!=cldesc) cldesc = 0;
         }


      } else if ( cname.Length() ) {
         // The branch contains a non-split object that we are unfolding!

         // See AnalyzeTree for similar code!
         TBranchProxyClassDescriptor *cldesc = 0;

         TClass *cl = gROOT->GetClass(cname);
         if (cl) {
            cldesc = new TBranchProxyClassDescriptor(cl->GetName(), branch->GetName(),
                                                     isclones, 0 /* unsplit object */);
         }
         if (cldesc) {
            TStreamerInfo *info = cl->GetStreamerInfo();
            TStreamerElement *elem = 0;

            TIter next(info->GetElements());
            while( (elem = (TStreamerElement*)next()) ) {
               AnalyzeElement(branch,elem,level+1,cldesc,"");
            }

            TBranchProxyClassDescriptor *added = AddClass(cldesc);
            if (added) proxyTypeName = added->GetName();
            // this codes and the previous 2 lines move from inside the if (cl)
            // aboce and this line was used to avoid unecessary work:
            // if (added!=cldesc) cldesc = 0;
         }

      }

      TLeaf *leaf = (TLeaf*)branch->GetListOfLeaves()->At(0);

      if (leaf && strlen(leaf->GetTypeName()) == 0) {
         // no type is know for the first leaf (which case is that?)
         return extraLookedAt;
      }

      if (leaf && proxyTypeName.Length()==0) {
         proxyTypeName = leaf->GetTypeName() ;
      }

      if (leaf && !isclones) dataMemberName = leaf->GetName();
      else dataMemberName = branch->GetName();

      Int_t pos;
      pos = dataMemberName.Index(".");
      if (pos != -1) {
         dataMemberName.Remove(0,pos+1);
      }
      pos = dataMemberName.Index("[");
      if (pos != -1) {
         dataMemberName.Remove(pos);
      }
      pos = dataMemberName.Index(".");

      TString branchName = branch->GetName();

      if (pos != -1 && container==kClones && branch->IsA()==TBranchElement::Class()) {
         // We still have a "." in the name, we assume that we are in the case
         // where we reach an embedded object in the object contained in the
         // TClonesArray

         // Discover the type of this object.
         TString name = dataMemberName(0,pos);

         TBranchElement *mom = (TBranchElement*)branch->GetMother()->GetSubBranch(branch);
         TString cname = mom->GetClonesName();
         TString prefix = mom->GetName();
         prefix += ".";
         prefix += name;
         // prefix += ".";


         if ( topdesc && strcmp(topdesc->GetBranchName(),prefix.Data())==0 ) {

            // Assume we coming recursively from the previous case!
            dataMemberName.Remove(0,pos+1);

         } else {

            TStreamerElement* branchStreamerElem = 0;

            TStreamerInfo *momInfo = mom->GetInfo();
            if (cname != momInfo->GetName()) {
               // We do not have the correct TStreamerInfo, this is
               // because there is no proper 'branch' holding this sub-object
               // they are all stored in the branch for the owner of the object

               // We need to discover if 'branch' represents a direct datamember
               // of the class in 'mom' or if it is an indirect one (with a
               // missing branch in the hierachy)

               TClass *momBranchClass = TClass::GetClass(cname);

               // We are in the case where there is a missing branch in the hiearchy

               momInfo = momBranchClass->GetStreamerInfo();

               // remove the main branch name (if present)
               TString parentDataName = branch->GetName();
               if (parentDataName.Index(mom->GetName())==0) {
                  parentDataName.Remove(0,strlen(mom->GetName()));
                  if (parentDataName[0]=='.') {
                     parentDataName.Remove(0,1);
                  }
               }

               // remove the current data member name
               Ssiz_t pos = parentDataName.Last('.');
               if (pos>0) {
                  // We had a branch name of the style:
                  //     [X.]Y.Z
                  // and we are looking up 'Y'
                  parentDataName.Remove(pos);
                  branchStreamerElem = (TStreamerElement*)momInfo->GetElements()->FindObject( parentDataName.Data() );

               } else {
                  // We had a branch name of the style:
                  //     [X.]Z
                  // and we are looking up 'Z'
                  // Because we are missing 'Y' (or more exactly the name of the
                  // thing that contains Z, ... we don't know what do yet.

                  Error("AnalyzeBranch",
                        "The case of the branch '%s' is not implemented yet.\n"
                        "Please send your data file to the root developers.",
                        branch->GetName());
               }

            } else {

               branchStreamerElem = (TStreamerElement*)
                  momInfo->GetElements()->FindObject(name.Data());

            }


            if (branchStreamerElem==0) {
               Error("AnalyzeBranch","We did not find %s when looking into %s.",
                     name.Data(),mom->GetName());
               //             mom->GetInfo()->Print();
               return extraLookedAt;
            } else {
               //             fprintf(stderr,"SUCC: We did find %s when looking into %s %p.\n",
               //                     name.Data(),mom->GetName(),branchStreamerElem);
               //             mom->GetInfo()->Print();
            }

            TClass *cl = branchStreamerElem->GetClassPointer();

            cname = cl->GetName();
            proxyTypeName = Form("TClaObjProxy<%s >",cname.Data());

            TBranchProxyClassDescriptor *cldesc;

            cldesc = new TBranchProxyClassDescriptor( cl->GetName(), prefix.Data(), prefix.Data(),
                                                      TBranchProxyClassDescriptor::kInsideClones,
                                                      branch->GetSplitLevel()-1);

            TIter next(mom->GetListOfBranches());
            TBranch *subbranch;
            while ( (subbranch = (TBranch*)next()) && subbranch!=branch ) {};

            Assert( subbranch == branch );

            do {
               TString subname = subbranch->GetName();
               if ( subname.BeginsWith( prefix ) ) {
                  Int_t skipped = 0;
                  if (cldesc) {

                     skipped = AnalyzeBranch( subbranch, level+1, cldesc);
                     Int_t s = 0;
                     while( s<skipped && next() ) { s++; };

                  }
                  extraLookedAt += 1 + skipped;
               } else {
                  break;
               }
            } while ( (subbranch = (TBranch*)next()) );

            dataMemberName.Remove(pos);
            //fprintf(stderr,"will use %s\n", dataMemberName.Data());

            // this codes and the previous 2 lines move from inside the if (cl)
            // aboce and this line was used to avoid unecessary work:
            TBranchProxyClassDescriptor *added = AddClass(cldesc);
            if (added) proxyTypeName = added->GetName();
            // if (added!=cldesc) cldesc = 0;

            pos = branchName.Last('.');
            if (pos != -1) {
               branchName.Remove(pos);
            }

         }
      }

      TBranchProxyDescriptor *desc;
      if (topdesc) {
         topdesc->AddDescriptor( desc = new TBranchProxyDescriptor( dataMemberName.Data(),
                                                                    proxyTypeName, branchName.Data() ),
                                 isBase );
      } else {
         dataMemberName.Prepend(prefix);
         AddDescriptor( desc = new TBranchProxyDescriptor( dataMemberName.Data(),
                                                           proxyTypeName,
                                                           branchName.Data() ) );
      }
      //fprintf(stderr,"%-*s      %-*s(director,\"%s\")\n",
      //        0," ",10,desc->GetName(), desc->GetBranchName());
      return extraLookedAt;
   }

   UInt_t TTreeProxyGenerator::AnalyzeOldLeaf(TLeaf *leaf, UInt_t /* level */,
                                              TBranchProxyClassDescriptor *topdesc)
   {
      // Analyze the leaf and populate the `TTreeProxyGenerator or
      // the topdesc with its findings.

      if (leaf->IsA()==TLeafObject::Class()) {
         Error("AnalyzeOldLeaf","TLeafObject not supported yet");
         return 0;
      }

      TString leafTypeName = leaf->GetTypeName();
      Int_t pos = leafTypeName.Last('_');
      if (pos!=-1) leafTypeName.Remove(pos);

      Int_t len = leaf->GetLen();
      TLeaf *leafcount = leaf->GetLeafCount();

      UInt_t dim = 0;
      Int_t maxDim[3];
      maxDim[0] = maxDim[1] = maxDim[2] = 1;

      TString dimensions;
      TString temp = leaf->GetName();
      pos = temp.Index("[");
      if (pos!=-1) {
         if (pos) temp.Remove(0,pos);
         dimensions.Append(temp);
      }
      temp = leaf->GetTitle();
      pos = temp.Index("[");
      if (pos!=-1) {
         if (pos) temp.Remove(0,pos);
         dimensions.Append(temp);
      }

      Int_t dimlen = dimensions.Length();

      if (dimlen) {
         const char *current = dimensions.Data();

         Int_t index;
         Int_t scanindex ;
         while (current) {
            current++;
            if (current[0] == ']') {
               maxDim[dim] = -1; // Loop over all elements;
            } else {
               scanindex = sscanf(current,"%d",&index);
               if (scanindex) {
                  maxDim[dim] = index;
               } else {
                  maxDim[dim] = -2; // Index is calculated via a variable.
               }
            }
            dim ++;
            if (dim >= 3) {
               // NOTE: test that dim this is NOT too big!!
               break;
            }
            current = (char*)strstr( current, "[" );
         }

      }
      //char *twodim = (char*)strstr(leaf->GetTitle(),"][");

      if (leafcount) {
         len = leafcount->GetMaximum();
      }


      TString type;
      switch (dim) {
         case 0: {
            type = "T";
            type += leafTypeName;
            type += "Proxy";
            break;
         }
         case 1: {
            type = "TArray";
            type += leafTypeName;
            type += "Proxy";
            break;
         }
         case 2: {
            type = "TArray2Proxy<";
            type += leaf->GetTypeName();
            type += ",";
            type += maxDim[1];
            type += " >";
            break;
         }
         case 3: {
            type = "TArray3Proxy<";
            type += leaf->GetTypeName();
            type += ",";
            type += maxDim[1];
            type += ",";
            type += maxDim[2];
            type += " >";
            break;
         }
         default:  {
            Error("AnalyzeOldLeaf","Array of more than 3 dimentsions not implemented yet.");
            return 0;
         }
      }

      TString branchName = leaf->GetBranch()->GetName();
      TBranchProxyDescriptor *desc;
      if (topdesc) {
         topdesc->AddDescriptor( desc = new TBranchProxyDescriptor( branchName.Data(),
                                                                    type,
                                                                    branchName.Data() ),
                                 0 );
      } else {
         // leafname.Prepend(prefix);
         AddDescriptor( desc = new TBranchProxyDescriptor( branchName.Data(),
                                                           type,
                                                           branchName.Data() ) );
      }

      return 0;

   }

   UInt_t TTreeProxyGenerator::AnalyzeOldBranch(TBranch *branch, UInt_t level,
                                                TBranchProxyClassDescriptor *topdesc)
   {
      // Analyze the branch and populate the TTreeProxyGenerator or the topdesc with
      // its findings.  Sometimes several branch of the mom are also analyzed,
      // the number of such branches is returned (this happens in the case of
      // embedded objects inside an object inside a clones array split more than
      // one level.

      UInt_t extraLookedAt = 0;
      TString prefix;

      TString branchName = branch->GetName();

      TObjArray *leaves = branch->GetListOfLeaves();
      Int_t nleaves = leaves ? leaves->GetEntriesFast() : 0;

      if (nleaves>1) {

         // Create a holder
         TString type = "unknown";
         TBranchProxyClassDescriptor *cldesc = new TBranchProxyClassDescriptor(branch->GetName());
         TBranchProxyClassDescriptor *added = AddClass(cldesc);
         if (added) type = added->GetName();

         for(int l=0;l<nleaves;l++) {
            TLeaf *leaf = (TLeaf*)leaves->UncheckedAt(l);
            extraLookedAt += AnalyzeOldLeaf(leaf,level+1,cldesc);
         }

         TBranchProxyDescriptor *desc;
         if (topdesc) {
            topdesc->AddDescriptor( desc = new TBranchProxyDescriptor( branchName.Data(),
                                                                       type,
                                                                       branchName.Data() ),
                                    0 );
         } else {
            // leafname.Prepend(prefix);
            AddDescriptor( desc = new TBranchProxyDescriptor( branchName.Data(),
                                                              type,
                                                              branchName.Data() ) );
         }

      } else {

         TLeaf *leaf = (TLeaf*)leaves->UncheckedAt(0);
         extraLookedAt += AnalyzeOldLeaf(leaf,level,topdesc);

      }


      return extraLookedAt;

   }

   void TTreeProxyGenerator::AnalyzeTree() {

      TIter next( fTree->GetListOfBranches() );
      TBranch *branch;
      while ( (branch = (TBranch*)next()) ) {
         const char *branchname = branch->GetName();
         const char *classname = branch->GetClassName();
         if (classname && strlen(classname)) {
            AddForward( classname );
            AddHeader( classname );
         }

         TBranchProxyClassDescriptor *desc = 0;
         TClass *cl = gROOT->GetClass(classname);
         TString type = "unknown";
         if (cl) {
            Bool_t isclones = false;
            if (cl==TClonesArray::Class()) {
               isclones = true;
               if (branch->IsA()==TBranchElement::Class()) {
                  const char *cname = ((TBranchElement*)branch)->GetClonesName();
                  TClass *ncl = gROOT->GetClass(cname);
                  if (ncl) {
                     cl = ncl;
                  } else {
                     Error("AnalyzeTree",
                           "Introspection of TClonesArray in older file not implemented yet.");
                  }
               } else {
                  Error("AnalyzeTree",
                        "Introspection of TClonesArray in older file not implemented yet.");
               }

            }
            if (NeedToEmulate(cl,0) || branchname[strlen(branchname)-1] == '.' ) {
               desc = new TBranchProxyClassDescriptor(cl->GetName(), branchname,
                                                      isclones, branch->GetSplitLevel());
            } else {
               type = Form("TObjProxy<%s >",cl->GetName());
            }
         }

         if ( branch->GetListOfBranches()->GetEntries() == 0 ) {

            if (cl) {
               // We have a non-splitted object!

               TStreamerInfo *info = cl->GetStreamerInfo();
               TStreamerElement *elem = 0;

               TIter next(info->GetElements());
               while( (elem = (TStreamerElement*)next()) ) {
                  AnalyzeElement(branch,elem,1,desc,"");
               }

               desc = AddClass(desc);
               type = desc->GetName();
               AddDescriptor( new TBranchProxyDescriptor( branchname, type, branchname ) );
            } else {

               // We have a top level raw type.

               AnalyzeOldBranch(branch, 0, 0);

            }

         } else {

            // We have a splitted object

            TBranch *subbranch;
            TIter subnext( branch->GetListOfBranches() );
            UInt_t skipped = 0;
            if (desc) {
               while ( (subbranch = (TBranch*)subnext()) ) {
                  skipped = AnalyzeBranch(subbranch,1,desc);
                  if (skipped != 0) Error("AnalyzeTree",
                                          "Unexpectly read more than one branch in AnalyzeTree.");
               }
            }
            desc = AddClass(desc);
            type = desc->GetName();
            AddDescriptor( new TBranchProxyDescriptor( branchname, type, branchname ) );

            if ( branchname[strlen(branchname)-1] != '.' ) {
               // If there is no dot also included the data member directly
               subnext.Reset();
               while ( (subbranch = (TBranch*)subnext()) ) {
                  skipped = AnalyzeBranch(subbranch,1,0);
                  if (skipped != 0) Error("AnalyzeTree",
                                          "Unexpectly read more than one branch in AnalyzeTree.");
               }
            }

         } // if split or non split
      }

   }

   void TTreeProxyGenerator::AnalyzeElement(TBranch *branch, TStreamerElement *element,
                                            UInt_t level, TBranchProxyClassDescriptor *topdesc,
                                            const char *path)
   {
      // Analyze the element and populate the TTreeProxyGenerator or the topdesc with
      // its findings.

      TString dataMemberName;
      TString pxDataMemberName;
      TString type;

      // TString prefix;
      bool isBase = false;
      TString cname;
      TString middle;
      Bool_t  isclones = false;
      EContainer container = kNone;
      if (topdesc && topdesc->IsClones()) {
         container = kClones;
         middle = "Cla";
         isclones = true;
      }

      if (!element) return;
      bool ispointer = false;
      switch(element->GetType()) {

         case TStreamerInfo::kChar:    { type = "T" + middle + "CharProxy"; break; }
         case TStreamerInfo::kShort:   { type = "T" + middle + "ShortProxy"; break; }
         case TStreamerInfo::kInt:     { type = "T" + middle + "IntProxy"; break; }
         case TStreamerInfo::kLong:    { type = "T" + middle + "LongProxy"; break; }
         case TStreamerInfo::kLong64:  { type = "T" + middle + "Long64Proxy"; break; }
         case TStreamerInfo::kFloat:   { type = "T" + middle + "FloatProxy"; break; }
         case TStreamerInfo::kDouble:  { type = "T" + middle + "DoubleProxy"; break; }
         case TStreamerInfo::kDouble32:{ type = "T" + middle + "Double32Proxy"; break; }
         case TStreamerInfo::kUChar:   { type = "T" + middle + "UCharProxy"; break; }
         case TStreamerInfo::kUShort:  { type = "T" + middle + "UShortProxy"; break; }
         case TStreamerInfo::kUInt:    { type = "T" + middle + "UIntProxy"; break; }
         case TStreamerInfo::kULong:   { type = "T" + middle + "ULongProxy"; break; }
         case TStreamerInfo::kULong64: { type = "T" + middle + "ULong64Proxy"; break; }
         case TStreamerInfo::kBits:    { type = "T" + middle + "UIntProxy"; break; }

         case TStreamerInfo::kCharStar: { type = GetArrayType(element,"Char",container); break; }

            // array of basic types  array[8]
         case TStreamerInfo::kOffsetL + TStreamerInfo::kChar:    { type = GetArrayType(element,"Char",container ); break; }
         case TStreamerInfo::kOffsetL + TStreamerInfo::kShort:   { type = GetArrayType(element,"Short",container ); break; }
         case TStreamerInfo::kOffsetL + TStreamerInfo::kInt:     { type = GetArrayType(element,"Int",container ); break; }
         case TStreamerInfo::kOffsetL + TStreamerInfo::kLong:    { type = GetArrayType(element,"Long",container ); break; }
         case TStreamerInfo::kOffsetL + TStreamerInfo::kLong64:  { type = GetArrayType(element,"Long64",container ); break; }
         case TStreamerInfo::kOffsetL + TStreamerInfo::kFloat:   { type = GetArrayType(element,"Float",container ); break; }
         case TStreamerInfo::kOffsetL + TStreamerInfo::kDouble:  { type = GetArrayType(element,"Double",container ); break; }
         case TStreamerInfo::kOffsetL + TStreamerInfo::kDouble32:{ type = GetArrayType(element,"Double32",container ); break; }
         case TStreamerInfo::kOffsetL + TStreamerInfo::kUChar:   { type = GetArrayType(element,"UChar",container ); break; }
         case TStreamerInfo::kOffsetL + TStreamerInfo::kUShort:  { type = GetArrayType(element,"UShort",container ); break; }
         case TStreamerInfo::kOffsetL + TStreamerInfo::kUInt:    { type = GetArrayType(element,"UInt",container ); break; }
         case TStreamerInfo::kOffsetL + TStreamerInfo::kULong:   { type = GetArrayType(element,"ULong",container ); break; }
         case TStreamerInfo::kOffsetL + TStreamerInfo::kULong64: { type = GetArrayType(element,"ULong64",container ); break; }
         case TStreamerInfo::kOffsetL + TStreamerInfo::kBits:    { type = GetArrayType(element,"UInt",container ); break; }

            // pointer to an array of basic types  array[n]
         case TStreamerInfo::kOffsetP + TStreamerInfo::kChar:    { type = GetArrayType(element,"Char",container ); break; }
         case TStreamerInfo::kOffsetP + TStreamerInfo::kShort:   { type = GetArrayType(element,"Short",container ); break; }
         case TStreamerInfo::kOffsetP + TStreamerInfo::kInt:     { type = GetArrayType(element,"Int",container ); break; }
         case TStreamerInfo::kOffsetP + TStreamerInfo::kLong:    { type = GetArrayType(element,"Long",container ); break; }
         case TStreamerInfo::kOffsetP + TStreamerInfo::kLong64:  { type = GetArrayType(element,"Long64",container ); break; }
         case TStreamerInfo::kOffsetP + TStreamerInfo::kFloat:   { type = GetArrayType(element,"Float",container ); break; }
         case TStreamerInfo::kOffsetP + TStreamerInfo::kDouble:  { type = GetArrayType(element,"Double",container ); break; }
         case TStreamerInfo::kOffsetP + TStreamerInfo::kDouble32:{ type = GetArrayType(element,"Double32",container ); break; }
         case TStreamerInfo::kOffsetP + TStreamerInfo::kUChar:   { type = GetArrayType(element,"UChar",container ); break; }
         case TStreamerInfo::kOffsetP + TStreamerInfo::kUShort:  { type = GetArrayType(element,"UShort",container ); break; }
         case TStreamerInfo::kOffsetP + TStreamerInfo::kUInt:    { type = GetArrayType(element,"UInt",container ); break; }
         case TStreamerInfo::kOffsetP + TStreamerInfo::kULong:   { type = GetArrayType(element,"ULong",container ); break; }
         case TStreamerInfo::kOffsetP + TStreamerInfo::kULong64: { type = GetArrayType(element,"ULong64",container ); break; }
         case TStreamerInfo::kOffsetP + TStreamerInfo::kBits:    { type = GetArrayType(element,"UInt",container ); break; }

            // array counter //[n]
         case TStreamerInfo::kCounter: { type = "T" + middle + "IntProxy"; break; }


         case TStreamerInfo::kOffsetL + TStreamerInfo::kObjectp:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kObjectP:
         case TStreamerInfo::kObjectp:
         case TStreamerInfo::kObjectP:
         case TStreamerInfo::kAnyp:
         case TStreamerInfo::kAnyP:
            // set as pointers and fall through to the next switches
            ispointer = true;
         case TStreamerInfo::kOffsetL + TStreamerInfo::kObject:
         case TStreamerInfo::kObject:
         case TStreamerInfo::kTString:
         case TStreamerInfo::kTNamed:
         case TStreamerInfo::kTObject:
         case TStreamerInfo::kAny:
         case TStreamerInfo::kBase: {
            TClass *cl = element->GetClassPointer();
            if (cl) {
               type = Form("T%sObjProxy<%s >",
                           middle.Data(),cl->GetName());
               cname = cl->GetName();
               if (cl==TClonesArray::Class()) {
                  isclones = true;

                  Int_t i = branch->GetTree()->GetReadEntry();
                  if (i<0) i = 0;
                  branch->GetEntry(i);

                  //char *obj = branch->GetObject();

                  // now need to follow it through to this pointer!

                  TClonesArray *arr;

                  TString fullpath = branch->GetName();
                  fullpath += ".";
                  if (path && strlen(path)>0) fullpath.Append(path).Append(".");
                  fullpath += element->GetName();

                  TTreeFormula *formula = new TTreeFormula("clones",fullpath,branch->GetTree());

                  TFormLeafInfo *leafinfo = formula->GetLeafInfo(0);
                  TLeaf *leaf = formula->GetLeaf(0);
                  Assert(leaf && leafinfo);

                  arr = (TClonesArray*)leafinfo->GetLocalValuePointer(leaf,0);

                  /*
                    if (ispointer) {
                    arr = (TClonesArray*)*(void**)(obj+lOffset);
                    } else {
                    arr = (TClonesArray*)(obj+lOffset);
                    }
                  */
                  cname = arr->GetClass()->GetName();

                  if (cname.Length()==0) {
                     Error("AnalyzeTree",
                           "Introspection of TClonesArray in older file not implemented yet.");
                  }
               }
            }
            else Error("AnalyzeTree","missing class for %s.",branch->GetName());
            if (element->IsA()==TStreamerBase::Class()) {
               // prefix  = "base";
               isBase = true;
            }
            AddForward(cl);
            AddHeader(cl);
            break;
         }

         default:
            Error("AnalyzeTree",
                  "Unsupported type for %s %s %d",
                  branch->GetName(), element->GetName(), element->GetType());

      }

      dataMemberName = element->GetName();

      if (level<=fMaxUnrolling) {

         // See AnalyzeTree for similar code!
         TBranchProxyClassDescriptor *cldesc;

         TClass *cl = gROOT->GetClass(cname);
         if (cl) {
            cldesc = new TBranchProxyClassDescriptor(cl->GetName(), branch->GetName(),
                                                     isclones, 0 /* non-split object */);

            TStreamerInfo *info = cl->GetStreamerInfo();
            TStreamerElement *elem = 0;

            TString subpath = path;
            if (subpath.Length()>0) subpath += ".";
            subpath += dataMemberName;

            TIter next(info->GetElements());
            while( (elem = (TStreamerElement*)next()) ) {
               AnalyzeElement(branch, elem, level+1, cldesc, subpath.Data());
            }

            TBranchProxyClassDescriptor *added = AddClass(cldesc);
            if (added) type = added->GetName();
         }

      }

      pxDataMemberName = /* prefix + */ dataMemberName;
      TBranchProxyDescriptor *desc;
      if (topdesc) {
         topdesc->AddDescriptor( desc = new TBranchProxyDescriptor( pxDataMemberName.Data(), type,
                                                                    dataMemberName.Data(), false),
                                 isBase );
      } else {
         Error("AnalyzeTree","topdesc should not be null in TTreeProxyGenerator::AnalyzeElement.");
      }
   }

   void TTreeProxyGenerator::WriteProxy() 
   {
      // Check whether the file exist and do something useful if it does
      if (fScript.Length()==0) {
         Error("WriteProxy","No user script has been specified.");
         return;
      }

      TString fileLocation = gSystem->DirName(fScript);

      TString incPath = gSystem->GetIncludePath(); // of the form -Idir1  -Idir2 -Idir3
      incPath.Append(":").Prepend(" ");
      incPath.ReplaceAll(" -I",":");       // of form :dir1 :dir2:dir3
      while ( incPath.Index(" :") != -1 ) {
         incPath.ReplaceAll(" :",":");
      }
      incPath.Prepend(fileLocation+":.:");

      const char *filename = gSystem->Which(incPath,fScript);
      if (filename==0) {
         Error("WriteProxy","Can not find the user's script: %s",fScript.Data());
         return;
      }
      const char *cutfilename = 0;
      if (fCutScript.Length()) {
         fileLocation = gSystem->DirName(fCutScript);
         incPath.Prepend(fileLocation+":.:");
         cutfilename = gSystem->Which(incPath,fCutScript);
         if (cutfilename==0) {
            Error("WriteProxy","Can not find the user's cut script: %s",fCutScript.Data());
            return;
         }
      }

      fHeaderFilename = fPrefix;
      fHeaderFilename.Append(".h");

      // Check to see if the target file exist.
      // If they do we will generate the proxy in temporary file and modify the original
      // if and only if it is different.

      Bool_t updating = kFALSE;
      if (gSystem->GetPathInfo( fHeaderFilename, 0, (Long_t*)0, 0, 0 ) == 0) {
         // file already exist
         updating = kTRUE;
      }

      TString classname = fPrefix;

      TString treefile;
      bool ischain = fTree->InheritsFrom(TChain::Class());
      if (fTree->GetDirectory() && fTree->GetDirectory()->GetFile())
         treefile = fTree->GetDirectory()->GetFile()->GetName();
      else
         treefile = "Memory Directory";

      TString scriptfunc = fScript;
      Ssiz_t dot_pos = scriptfunc.Last('.');
      scriptfunc.Replace( dot_pos, fScript.Length()-dot_pos, "");
      TString scriptHeader = scriptfunc;
      const char * extensions[] = { ".h", ".hh", ".hpp", ".hxx",  ".hPP", ".hXX" };

      int i;
      for (i = 0; i < 6; i++ ) {
         TString possible = scriptHeader;
         possible.Append(extensions[i]);
         const char *name = gSystem->Which(incPath,possible);
         if (name) {
            scriptHeader = possible;
            fListOfHeaders.Add(new TNamed("script",Form("#include \"%s\"\n",
                                                        scriptHeader.Data())));
            break;
         }
      }
      scriptfunc = gSystem->BaseName(scriptfunc);


      TString cutscriptfunc = fCutScript;
      if (cutfilename) {
         Ssiz_t dot_pos = cutscriptfunc.Last('.');
         cutscriptfunc.Replace( dot_pos, fCutScript.Length()-dot_pos, "");
         TString cutscriptHeader = cutscriptfunc;
         const char * extensions[] = { ".h", ".hh", ".hpp", ".hxx",  ".hPP", ".hXX" };

         for (i = 0; i < 6; i++ ) {
            TString possible = cutscriptHeader;
            possible.Append(extensions[i]);
            const char *name = gSystem->Which(incPath,possible);
            if (name) {
               cutscriptHeader = possible;
               fListOfHeaders.Add(new TNamed("cutscript",Form("#include \"%s\"\n",
                                                              cutscriptHeader.Data())));
               break;
            }
         }
         cutscriptfunc = gSystem->BaseName(cutscriptfunc);
      }

      FILE *hf;
      TString tmpfilename;
      if (updating) {
         tmpfilename = gSystem->BaseName( tmpnam(0) );
         tmpfilename.Append("_proxy.h");
         hf = fopen(tmpfilename.Data(),"w");
      } else {
         hf = fopen(fHeaderFilename.Data(),"w");
      }

      TDatime td;
      fprintf(hf,   "/////////////////////////////////////////////////////////////////////////\n");
      fprintf(hf,   "//   This class has been automatically generated \n");
      fprintf(hf,   "//   (at %s by ROOT version %s)\n",td.AsString(),gROOT->GetVersion());
      if (!ischain) {
         fprintf(hf,"//   from TTree %s/%s\n",fTree->GetName(),fTree->GetTitle());
         fprintf(hf,"//   found on file: %s\n",treefile.Data());
      } else {
         fprintf(hf,"//   from TChain %s/%s\n",fTree->GetName(),fTree->GetTitle());
      }
      fprintf(hf,   "/////////////////////////////////////////////////////////////////////////\n");
      fprintf(hf,"\n");
      fprintf(hf,"\n");

      fprintf(hf,"#ifndef %s_h\n",classname.Data());
      fprintf(hf,"#define %s_h\n",classname.Data());
      fprintf(hf,"\n");


      fprintf(hf,"// System Headers needed by the proxy\n");
      fprintf(hf,"#include <TROOT.h>\n");
      fprintf(hf,"#include <TChain.h>\n");
      fprintf(hf,"#include <TFile.h>\n");
      fprintf(hf,"#include <TSelectorDraw.h>\n");
      fprintf(hf,"#include <TPad.h>\n");
      fprintf(hf,"#include <TH1.h>\n");
      fprintf(hf,"#include <TBranchProxy.h>\n");
      fprintf(hf,"#include <TBranchProxyDirector.h>\n");
      fprintf(hf,"#include <TBranchProxyTemplate.h>\n");
      fprintf(hf,"#if defined(__CINT__) && !defined(__MAKECINT__)\n");
      fprintf(hf,"   #define ROOT_Rtypes\n");
      fprintf(hf,"#endif\n");
      fprintf(hf,"using namespace ROOT;\n");       // questionable
      fprintf(hf,"\n\n");

      fprintf(hf,"// forward declarations needed by this particular proxy\n");
      TIter next( &fListOfForwards );
      TObject *current;
      while ( (current=next()) ) {
         fprintf(hf,current->GetTitle());
      }

      fprintf(hf,"\n\n");
      fprintf(hf,"// Header needed by this particular proxy\n");
      next = &fListOfHeaders;
      TObject *header;
      while ( (header = next()) ) {
         fprintf(hf,header->GetTitle());
      }
      fprintf(hf,"\n\n");

      fprintf(hf, "class %s : public TSelector {\n", classname.Data());
      fprintf(hf, "   public :\n");
      fprintf(hf, "   TTree          *fChain;    //!pointer to the analyzed TTree or TChain\n");
      fprintf(hf, "   TSelectorDraw  *fHelper;   //!helper class to create the default histogram\n");
      fprintf(hf, "   TList          *fInput;    //!input list of the helper\n");
      fprintf(hf, "   TH1            *htemp;     //!pointer to the histogram\n");
      fprintf(hf, "   TBranchProxyDirector  fDirector; //!Manages the proxys\n\n");

      fprintf(hf, "   // Wrapper class for each unwounded class\n");
      next = &fListOfClasses;
      TBranchProxyClassDescriptor *clp;
      while ( (clp = (TBranchProxyClassDescriptor*)next()) ) {
         clp->OutputDecl(hf, 3, fMaxDatamemberType);
      }
      fprintf(hf,"\n\n");

      fprintf(hf, "   // Proxy for each of the branches and leaves of the tree\n");
      next = &fListOfTopProxies;
      TBranchProxyDescriptor *data;
      while ( (data = (TBranchProxyDescriptor*)next()) ) {
         data->OutputDecl(hf, 3, fMaxDatamemberType);
      }
      fprintf(hf,"\n\n");

      // Constructor
      fprintf(hf,      "   %s(TTree *tree=0) : \n",classname.Data());
      fprintf(hf,      "      fChain(0)");
      fprintf(hf,   ",\n      fHelper(0)");
      fprintf(hf,   ",\n      fInput(0)");
      fprintf(hf,   ",\n      htemp(0)");
      fprintf(hf,   ",\n      fDirector(tree,-1)");
      next.Reset();
      while ( (data = (TBranchProxyDescriptor*)next()) ) {
         fprintf(hf,",\n      %-*s(&fDirector,\"%s\")",
                 fMaxDatamemberType, data->GetName(), data->GetBranchName());
      }

      fprintf(hf,    "\n      { }\n");

      // Other functions.
      fprintf(hf,"   ~%s();\n",classname.Data());
      fprintf(hf,"   Int_t   Version() const {return 1;}\n");
      fprintf(hf,"   void    Begin(::TTree *tree);\n");
      fprintf(hf,"   void    SlaveBegin(::TTree *tree);\n");
      fprintf(hf,"   void    Init(::TTree *tree);\n");
      fprintf(hf,"   Bool_t  Notify();\n");
      fprintf(hf,"   Bool_t  Process(Int_t entry);\n");
      fprintf(hf,"   void    SetOption(const char *option) { fOption = option; }\n");
      fprintf(hf,"   void    SetObject(TObject *obj) { fObject = obj; }\n");
      fprintf(hf,"   void    SetInputList(TList *input) {fInput = input;}\n");
      fprintf(hf,"   TList  *GetOutputList() const { return fOutput; }\n");
      fprintf(hf,"   void    SlaveTerminate();\n");
      fprintf(hf,"   void    Terminate();\n");
      fprintf(hf,"\n");
      fprintf(hf,"   ClassDef(%s,0);\n",classname.Data());
      fprintf(hf,"\n\n");

      fprintf(hf,"//inject the user's code\n");
      fprintf(hf,"#include \"%s\"\n",fScript.Data());

      if (cutfilename) {
         fprintf(hf,"#include \"%s\"\n",fCutScript.Data());
      }

      // Close the class.
      fprintf(hf,"};\n");
      fprintf(hf,"\n");
      fprintf(hf,"#endif\n");
      fprintf(hf,"\n\n");

      fprintf(hf,"#ifdef __MAKECINT__\n");
      next = &fListOfClasses;
      while ( (clp = (TBranchProxyClassDescriptor*)next()) ) {
         fprintf(hf,"#pragma link C++ class %s::%s-;\n",classname.Data(),clp->GetName());
      }
      fprintf(hf,"#pragma link C++ class %s;\n",classname.Data());
      fprintf(hf,"#endif\n");
      fprintf(hf,"\n\n");

      // Write the implementations.
      fprintf(hf,"%s::~%s() {\n",classname.Data(),classname.Data());
      fprintf(hf,"   // destructor. Clean up helpers.\n");
      fprintf(hf,"\n");
      fprintf(hf,"   delete fHelper;\n");
      fprintf(hf,"   delete fInput;\n");
      fprintf(hf,"}\n");
      fprintf(hf,"\n");
      fprintf(hf,"void %s::Init(TTree *tree)\n",classname.Data());
      fprintf(hf,"{\n");
      fprintf(hf,"//   Set branch addresses\n");
      fprintf(hf,"   if (tree == 0) return;\n");
      fprintf(hf,"   fChain = tree;\n");
      fprintf(hf,"   fDirector.SetTree(fChain);\n");
      fprintf(hf,"   fHelper = new TSelectorDraw();\n");
      fprintf(hf,"   fInput  = new TList();\n");
      fprintf(hf,"   fInput->Add(new TNamed(\"varexp\",\"0.0\")); // Fake a double size histogram\n");
      fprintf(hf,"   fInput->Add(new TNamed(\"selection\",\"\"));\n");
      fprintf(hf,"   fHelper->SetInputList(fInput);\n");
      fprintf(hf,"}\n");
      fprintf(hf,"\n");
      fprintf(hf,"Bool_t %s::Notify()\n",classname.Data());
      fprintf(hf,"{\n");
      fprintf(hf,"   // Called when loading a new file.\n");
      fprintf(hf,"   // Get branch pointers.\n");
      fprintf(hf,"   fDirector.SetTree(fChain);\n");
      fprintf(hf,"   \n");
      fprintf(hf,"   return kTRUE;\n");
      fprintf(hf,"}\n");
      fprintf(hf,"   \n");

      // generate code for class member function Begin
      fprintf(hf,"\n");
      fprintf(hf,"void %s::Begin(TTree *tree)\n",classname.Data());
      fprintf(hf,"{\n");
      fprintf(hf,"   // The Begin() function is called at the start of the query.\n");
      fprintf(hf,"   // When running with PROOF Begin() is only called on the client.\n");
      fprintf(hf,"   // The tree argument is deprecated (on PROOF 0 is passed).\n");
      fprintf(hf,"\n");
      fprintf(hf,"   TString option = GetOption();\n");
      fprintf(hf,"\n");
      fprintf(hf,"}\n");

      // generate code for class member function SlaveBegin
      fprintf(hf,"\n");
      fprintf(hf,"void %s::SlaveBegin(TTree *tree)\n",classname.Data());
      fprintf(hf,"{\n");
      fprintf(hf,"   // The SlaveBegin() function is called after the Begin() function.\n");
      fprintf(hf,"   // When running with PROOF SlaveBegin() is called on each slave server.\n");
      fprintf(hf,"   // The tree argument is deprecated (on PROOF 0 is passed).\n");
      fprintf(hf,"\n");
      fprintf(hf,"   Init(tree);\n");
      fprintf(hf,"\n");
      fprintf(hf,"   TString option = GetOption();\n");
      fprintf(hf,"   fHelper->SetOption(option);\n");
      fprintf(hf,"   fHelper->Begin(tree);\n");
      fprintf(hf,"   htemp = (TH1*)fHelper->GetObject();\n");
      if (cutfilename) {
         fprintf(hf,"   htemp->SetTitle(\"%s {%s}\");\n",fScript.Data(),fCutScript.Data());
      } else {
         fprintf(hf,"   htemp->SetTitle(\"%s\");\n",fScript.Data());
      }
      fprintf(hf,"   fObject = htemp;\n");
      fprintf(hf,"\n");
      fprintf(hf,"}\n");
      fprintf(hf,"\n");

      // generate code for class member function Process
      fprintf(hf,"Bool_t %s::Process(Int_t entry)\n",classname.Data());
      fprintf(hf,"{\n");

      fprintf(hf,"   // The Process() function is called for each entry in the tree (or possibly\n"
              "   // keyed object in the case of PROOF) to be processed. The entry argument\n"
              "   // specifies which entry in the currently loaded tree is to be processed.\n"
              "   // It can be passed to either TTree::GetEntry() or TBranch::GetEntry()\n"
              "   // to read either all or the required parts of the data. When processing\n"
              "   // keyed objects with PROOF, the object is already loaded and is available\n"
              "   // via the fObject pointer.\n"
              "   //\n"
              "   // This function should contain the \"body\" of the analysis. It can contain\n"
              "   // simple or elaborate selection criteria, run algorithms on the data\n"
              "   // of the event and typically fill histograms.\n\n");
      fprintf(hf,"   // WARNING when a selector is used with a TChain, you must use\n");
      fprintf(hf,"   //  the pointer to the current TTree to call GetEntry(entry).\n");
      fprintf(hf,"   //  The entry is always the local entry number in the current tree.\n");
      fprintf(hf,"   //  Assuming that fChain is the pointer to the TChain being processed,\n");
      fprintf(hf,"   //  use fChain->GetTree()->GetEntry(entry).\n");
      fprintf(hf,"\n");
      fprintf(hf,"\n");
      fprintf(hf,"   fDirector.SetReadEntry(entry);\n");
      if (cutfilename) {
         fprintf(hf,"   if (%s()) htemp->Fill(%s());\n",cutscriptfunc.Data(),scriptfunc.Data());
      } else {
         fprintf(hf,"   htemp->Fill(%s());\n",scriptfunc.Data());
      }
      fprintf(hf,"   return kTRUE;\n");
      fprintf(hf,"\n");
      fprintf(hf,"}\n");
      fprintf(hf,"\n");

      // generate code for class member function SlaveTerminate
      fprintf(hf,"\n");
      fprintf(hf,"void %s::SlaveTerminate()\n",classname.Data());
      fprintf(hf,"{\n");
      fprintf(hf,"   // The SlaveTerminate() function is called after all entries or objects\n"
              "   // have been processed. When running with PROOF SlaveTerminate() is called\n"
              "   // on each slave server.");
      fprintf(hf,"\n");
      fprintf(hf,"\n");
      fprintf(hf,"}\n");

      // generate code for class member function Terminate
      fprintf(hf,"void %s::Terminate()\n",classname.Data());
      fprintf(hf,"{\n");
      fprintf(hf,"   // Function called at the end of the event loop.\n");
      fprintf(hf,"   Int_t drawflag = (htemp && htemp->GetEntries()>0);\n");
      fprintf(hf,"   \n");
      fprintf(hf,"   if (!drawflag && !fOption.Contains(\"goff\") && !fOption.Contains(\"same\")) {\n");
      fprintf(hf,"      gPad->Clear();\n");
      fprintf(hf,"      return;\n");
      fprintf(hf,"  }\n");
      fprintf(hf,"   if (fOption.Contains(\"goff\")) drawflag = false;\n");
      fprintf(hf,"   if (drawflag) htemp->Draw(fOption);\n");
      fprintf(hf,"\n");
      fprintf(hf,"}\n");

      fclose(hf);

      if (updating) {
         // over-write existing file only if needed.
         if (AreDifferent(fHeaderFilename,tmpfilename)) gSystem->Rename(tmpfilename,fHeaderFilename);
         else gSystem->Unlink(tmpfilename);
      }
   }

}
