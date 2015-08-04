// @(#)root/treeplayer:$Id$
// Author: Akos Hajdu 22/06/2015

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TTreeReaderGenerator.h"
#include <stdio.h>

#include "TBranchElement.h"
#include "TChain.h"
#include "TClass.h"
#include "TClassEdit.h"
#include "TClonesArray.h"
#include "TDirectory.h"
#include "TError.h"
#include "TFile.h"
#include "TLeaf.h"
#include "TLeafC.h"
#include "TLeafObject.h"
#include "TROOT.h"
#include "TStreamerElement.h"
#include "TStreamerInfo.h"
#include "TTree.h"
#include "TVirtualCollectionProxy.h"
#include "TVirtualStreamerInfo.h"

namespace ROOT {

   ////////////////////////////////////////////////////////////////////////////////
   /// Constructor. Analyzes the tree and writes selector.

   TTreeReaderGenerator::TTreeReaderGenerator(TTree* tree, const char *classname, Option_t *option) :
      fTree(tree), fClassname(classname), fOptions(option),
      fIncludeAllLeaves(kFALSE), fIncludeAllTopmost(kFALSE)
   {
      ParseOptions();
      AnalyzeTree(fTree);
      WriteSelector();
   }

static Bool_t FindStringInVector(TString const& string, std::vector<TString> vec)
{
   for (TString const& s : vec) {
      if (string.EqualTo(s)) return kTRUE;
   }
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Check if element is a base class and if yes, return the base class.

static TVirtualStreamerInfo *GetBaseClass(TStreamerElement *element)
{
   TStreamerBase *base = dynamic_cast<TStreamerBase*>(element);
   if (base) {
      TVirtualStreamerInfo *info = base->GetBaseStreamerInfo();
      if (info) return info;
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Get name of class inside a container.

static TString GetContainedClassName(TBranchElement *branch, TStreamerElement *element, Bool_t ispointer)
{
   TString cname = branch->GetClonesName();
   if (cname.Length()==0) {
      // We may have any unsplit clones array
      Long64_t i = branch->GetTree()->GetReadEntry();
      if (i<0) i = 0;
      branch->GetEntry(i);
      char *obj = branch->GetObject();

      TBranchElement *parent = (TBranchElement*)branch->GetMother()->GetSubBranch(branch);
      const char *pclname = parent->GetClassName();

      TClass *clparent = TClass::GetClass(pclname);
      // TClass *clm = TClass::GetClass(GetClassName());
      Int_t lOffset = 0; // offset in the local streamerInfo.
      if (clparent) {
         const char *ename = 0;
         if (element) {
            ename = element->GetName();
            lOffset = clparent->GetStreamerInfo()->GetOffset(ename);
         } else {
            lOffset = 0;
         }
      }
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
   return cname;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the correct TStreamerInfo of class 'cl' in the list of branches
/// (current) [Assuming these branches correspond to a flattened version of
/// the class.]

static TVirtualStreamerInfo *GetStreamerInfo(TBranch *branch, TIter current, TClass *cl)
{
   TVirtualStreamerInfo *objInfo = 0;
   TBranchElement *b = 0;
   TString cname = cl->GetName();

   while( ( b = (TBranchElement*)current() ) ) {
      if ( cname == b->GetInfo()->GetName() ) {
         objInfo = b->GetInfo();
         break;
      }
   }
   if (objInfo == 0 && branch->GetTree()->GetDirectory()->GetFile()) {
      const TList *infolist = branch->GetTree()->GetDirectory()->GetFile()->GetStreamerInfoCache();
      if (infolist) {
         TVirtualStreamerInfo *i = (TVirtualStreamerInfo *)infolist->FindObject(cname);
         if (i) {
            // NOTE: Is this correct for Foreigh classes?
            objInfo = (TVirtualStreamerInfo *)cl->GetStreamerInfo(i->GetClassVersion());
         }
      }
   }
   if (objInfo == 0) {
      // We still haven't found it ... this is likely to be an STL collection .. anyway, use the current StreamerInfo.
      objInfo = cl->GetStreamerInfo();
   }
   return objInfo;
}

   ////////////////////////////////////////////////////////////////////////////////
   /// Add a header inclusion request. If the header is already included it will
   /// not be included again.

   void TTreeReaderGenerator::AddHeader(TClass *cl)
   {
      if (cl==0) return;

      // Check if already included
      TObject *obj = fListOfHeaders.FindObject(cl->GetName());
      if (obj) return;

      TString directive;

      // Extract inner class from collection
      if (cl->GetCollectionProxy() && cl->GetCollectionProxy()->GetValueClass()) {
         AddHeader( cl->GetCollectionProxy()->GetValueClass() );
      }

      // Construct directive
      Int_t stlType;
      if (0 == strcmp(cl->GetName(), "string")) { // Check if it is a string
         directive = "#include <string>\n";
      } else if (cl->GetCollectionProxy() && (stlType = cl->GetCollectionType())) { // Check if it is an STL container
         const char *what = "";
         switch(stlType)  {
            case  ROOT::kSTLvector:            what = "vector"; break;
            case  ROOT::kSTLlist:              what = "list"; break;
            case  ROOT::kSTLforwardlist:       what = "forward_list"; break;
            case -ROOT::kSTLdeque:             // same as positive
            case  ROOT::kSTLdeque:             what = "deque"; break;
            case -ROOT::kSTLmap:               // same as positive
            case  ROOT::kSTLmap:               what = "map"; break;
            case -ROOT::kSTLmultimap:          // same as positive
            case  ROOT::kSTLmultimap:          what = "map"; break;
            case -ROOT::kSTLset:               // same as positive
            case  ROOT::kSTLset:               what = "set"; break;
            case -ROOT::kSTLmultiset:          // same as positive
            case  ROOT::kSTLmultiset:          what = "set"; break;
            case -ROOT::kSTLunorderedset:      // same as positive
            case  ROOT::kSTLunorderedset:      what = "unordered_set"; break;
            case -ROOT::kSTLunorderedmultiset: // same as positive
            case  ROOT::kSTLunorderedmultiset: what = "unordered_multiset"; break;
            case -ROOT::kSTLunorderedmap:      // same as positive
            case  ROOT::kSTLunorderedmap:      what = "unordered_map"; break;
            case -ROOT::kSTLunorderedmultimap: // same as positive
            case  ROOT::kSTLunorderedmultimap: what = "unordered_multimap"; break;
         }
         if (what[0]) {
            directive = "#include <";
            directive.Append(what);
            directive.Append(">\n");
         }
      } else if (cl->GetDeclFileName() && strlen(cl->GetDeclFileName()) ) { // Custom file
         const char *filename = cl->GetDeclFileName();

         if (!filename) return;

#ifdef R__WIN32
         TString inclPath("include;prec_stl"); // GetHtml()->GetIncludePath());
#else
         TString inclPath("include:prec_stl"); // GetHtml()->GetIncludePath());
#endif
         Ssiz_t posDelim = 0;
         TString inclDir;
         TString sIncl(filename);
#ifdef R__WIN32
         const char* pdelim = ";";
         static const char ddelim = '\\';
#else
         const char* pdelim = ":";
         static const char ddelim = '/';
#endif
         while (inclPath.Tokenize(inclDir, posDelim, pdelim))
         {
            if (sIncl.BeginsWith(inclDir)) {
               filename += inclDir.Length();
               if (filename[0] == ddelim || filename[0] == '/') {
                  ++filename;
               }
               break;
            }
         }
         directive = Form("#include \"%s\"\n",filename);
      } else if (!strncmp(cl->GetName(), "pair<", 5)
                 || !strncmp(cl->GetName(), "std::pair<", 10)) { // TODO: what is this?
         TClassEdit::TSplitType split(cl->GetName());
         if (split.fElements.size() == 3) {
            for (int arg = 1; arg < 3; ++arg) {
               TClass* clArg = TClass::GetClass(split.fElements[arg].c_str());
               if (clArg) AddHeader(clArg);
            }
         }
      }
      // Add directive (if it is not added already)
      if (directive.Length()) {
         TIter i( &fListOfHeaders );
         for(TNamed *n = (TNamed *)i(); n; n = (TNamed*)i()) {
            if (directive == n->GetTitle()) {
               return;
            }
         }
         fListOfHeaders.Add(new TNamed(cl->GetName(), directive.Data()));
      }
   }

   ////////////////////////////////////////////////////////////////////////////////
   /// Add a reader to the generated code.

   void TTreeReaderGenerator::AddReader(TTreeReaderDescriptor::ReaderType type, TString dataType, TString name,
                                        TString branchName, TBranchDescriptor *parent, Bool_t isLeaf)
   {
      if(BranchNeedsReader(branchName, parent, isLeaf)) {
         fListOfReaders.Add( new TTreeReaderDescriptor(type, dataType, name, branchName) );
         printf("Added reader: TTreeReader%s<%s> %s (branch: %s)\n", type == TTreeReaderDescriptor::ReaderType::kValue ? "Value" : "Array",
                                                                     dataType.Data(),
                                                                     name.Data(),
                                                                     branchName.Data());
      }
   }

   ////////////////////////////////////////////////////////////////////////////////
   /// Analyse sub-branches of 'branch' recursively and extract readers.

   UInt_t TTreeReaderGenerator::AnalyzeBranches(TBranchDescriptor *desc, TBranchElement *branch, TVirtualStreamerInfo *info)
   {
      if (info==0) info = branch->GetInfo();

      TIter branches(branch->GetListOfBranches());

      return AnalyzeBranches(desc, branches, info);
   }

   ////////////////////////////////////////////////////////////////////////////////
   /// Analyse sub-branches 'branches' recursively and extract readers.

   UInt_t TTreeReaderGenerator::AnalyzeBranches(TBranchDescriptor *desc, TIter &branches, TVirtualStreamerInfo *info)
   {
      UInt_t lookedAt = 0;     // Number of sub-branches analyzed
      ELocation outer_isclones = kOut; // Is the parent branch a container
      TString containerName;   // Container name
      TString subBranchPrefix; // Prefix of sub-branch (if the elements and sub-branches do not match).
      Bool_t skipped = false;  // Should the branch be skipped

      // Check for containers (TClonesArray or STL)
      {
         TIter peek = branches;
         TBranchElement *branch = (TBranchElement*)peek();
         if (desc && desc->IsClones()) {
            outer_isclones = kClones;
            containerName = "TClonesArray";
         } else if (desc && desc->IsSTL()) {
            outer_isclones = kSTL;
            containerName = desc->fContainerName;
         } else if (!desc && branch && branch->GetBranchCount() == branch->GetMother()) {
            if ( ((TBranchElement*)(branch->GetMother()))->GetType()==3)  {
               outer_isclones = kClones;
               containerName = "TClonesArray";
            } else {
               outer_isclones = kSTL;
               containerName = branch->GetMother()->GetClassName();
            }
         }
         // FIXME: this is wrong because 'branch' is the first sub-branch and even though
         // it can be a collection, it does not mean that the other sub-branches are
         // collections as well.
         /* else if (branch->GetType() == 3) {
            outer_isclones = kClones;
            containerName = "TClonesArray";
         } else if (branch->GetType() == 4) {
            outer_isclones = kSTL;
            containerName = branch->GetMother()->GetSubBranch(branch)->GetClassName();
         }*/
         if (desc) {
            subBranchPrefix = desc->fBranchName;
         } else {
            TBranchElement *mom = (TBranchElement*)branch->GetMother();
            subBranchPrefix = mom->GetName();
            if (subBranchPrefix[subBranchPrefix.Length()-1]=='.') {
               subBranchPrefix.Remove(subBranchPrefix.Length()-1);
            } else if (mom->GetType()!=3 && mom->GetType() != 4) {
               subBranchPrefix = "";
            }
         }
      }

      // Loop through elements (i.e., sub-branches). The problem is that the elements
      // and sub-branches do not always match. For example suppose that class A contains
      // members x and y, and there is a class B inheriting from A and containing an extra
      // member z. It is possible that B has two elements: A and z but three sub-branches
      // x, y and z. Therefore, the branch iterator is treated differently.
      TIter elements( info->GetElements() );
      for( TStreamerElement *element = (TStreamerElement*)elements();
           element;
           element = (TStreamerElement*)elements() )
      {
         Bool_t isBase = false;     // Does the element correspond to a base class
         Bool_t usedBranch = kTRUE; // Does the branch correspond to the element (i.e., they match)
         Bool_t isLeaf = true;    // Is the branch a leaf (i.e. no sub-branches)
         TIter peek = branches;     // Iterator for sub-branches
         // Always start with the first available sub-branch and if it does not match the element,
         // try the next ones
         TBranchElement *branch = (TBranchElement*)peek();
         // There is a problem if there are more elements than branches
         if (branch==0) {
            if (desc) {
               Error("AnalyzeBranches","Ran out of branches when looking in branch %s, class %s",
                     desc->fBranchName, info->GetName());
            } else {
               Error("AnalyzeBranches","Ran out of branches when looking in class %s, element %s",
                     info->GetName(), element->GetName());
            }
            return lookedAt;
         }

         if (info->GetClass()->GetCollectionProxy() && strcmp(element->GetName(),"This")==0) {
            continue; // Skip the artifical streamer element.
         }

         if (element->GetType() == -1) {
            continue; // This is an ignored TObject base class.
         }

         // Get branch name
         TString branchname = branch->GetName();
         TString branchEndName;
         {
            TLeaf *leaf = (TLeaf*)branch->GetListOfLeaves()->At(0);
            if (leaf && outer_isclones == kOut
                && !(branch->GetType() == 3 || branch->GetType() == 4)) branchEndName = leaf->GetName();
            else branchEndName = branch->GetName();
            Int_t pos;
            pos = branchEndName.Index(".");
            if (pos!=-1) {
               if (subBranchPrefix.Length() && branchEndName.BeginsWith(subBranchPrefix)) {
                  branchEndName.Remove(0, subBranchPrefix.Length() + 1);
               }
            }
         }

         TString dataType; // Data type of reader
         TTreeReaderDescriptor::ReaderType readerType = TTreeReaderDescriptor::ReaderType::kValue;
         Bool_t ispointer = false;
         ELocation isclones = outer_isclones; // Is the actual sub-branch a collection (inherit from parent branch)
         // Get data type
         switch(element->GetType()) {
            // Built-in types
            case TVirtualStreamerInfo::kBool:    { dataType = "Bool_t";         readerType = TTreeReaderDescriptor::ReaderType::kValue; break; }
            case TVirtualStreamerInfo::kChar:    { dataType = "Char_t";         readerType = TTreeReaderDescriptor::ReaderType::kValue; break; }
            case TVirtualStreamerInfo::kShort:   { dataType = "Short_t";        readerType = TTreeReaderDescriptor::ReaderType::kValue; break; }
            case TVirtualStreamerInfo::kInt:     { dataType = "Int_t";          readerType = TTreeReaderDescriptor::ReaderType::kValue; break; }
            case TVirtualStreamerInfo::kLong:    { dataType = "Long_t";         readerType = TTreeReaderDescriptor::ReaderType::kValue; break; }
            case TVirtualStreamerInfo::kLong64:  { dataType = "Long64_t";       readerType = TTreeReaderDescriptor::ReaderType::kValue; break; }
            case TVirtualStreamerInfo::kFloat:   { dataType = "Float_t";        readerType = TTreeReaderDescriptor::ReaderType::kValue; break; }
            case TVirtualStreamerInfo::kFloat16: { dataType = "Float16_t";      readerType = TTreeReaderDescriptor::ReaderType::kValue; break; }
            case TVirtualStreamerInfo::kDouble:  { dataType = "Double_t";       readerType = TTreeReaderDescriptor::ReaderType::kValue; break; }
            case TVirtualStreamerInfo::kDouble32:{ dataType = "Double32_t";     readerType = TTreeReaderDescriptor::ReaderType::kValue; break; }
            case TVirtualStreamerInfo::kUChar:   { dataType = "UChar_t";        readerType = TTreeReaderDescriptor::ReaderType::kValue; break; }
            case TVirtualStreamerInfo::kUShort:  { dataType = "unsigned short"; readerType = TTreeReaderDescriptor::ReaderType::kValue; break; }
            case TVirtualStreamerInfo::kUInt:    { dataType = "unsigned int";   readerType = TTreeReaderDescriptor::ReaderType::kValue; break; }
            case TVirtualStreamerInfo::kULong:   { dataType = "ULong_t";        readerType = TTreeReaderDescriptor::ReaderType::kValue; break; }
            case TVirtualStreamerInfo::kULong64: { dataType = "ULong64_t";      readerType = TTreeReaderDescriptor::ReaderType::kValue; break; }
            case TVirtualStreamerInfo::kBits:    { dataType = "unsigned int";   readerType = TTreeReaderDescriptor::ReaderType::kValue; break; }
            // Character arrays
            case TVirtualStreamerInfo::kCharStar: { dataType = "Char_t"; readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            // Array of built-in types [8]
            case TVirtualStreamerInfo::kOffsetL + TVirtualStreamerInfo::kBool:    { dataType = "Bool_t";         readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            case TVirtualStreamerInfo::kOffsetL + TVirtualStreamerInfo::kChar:    { dataType = "Char_t";         readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            case TVirtualStreamerInfo::kOffsetL + TVirtualStreamerInfo::kShort:   { dataType = "Short_t";        readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            case TVirtualStreamerInfo::kOffsetL + TVirtualStreamerInfo::kInt:     { dataType = "Int_t";          readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            case TVirtualStreamerInfo::kOffsetL + TVirtualStreamerInfo::kLong:    { dataType = "Long_t";         readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            case TVirtualStreamerInfo::kOffsetL + TVirtualStreamerInfo::kLong64:  { dataType = "Long64_t";       readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            case TVirtualStreamerInfo::kOffsetL + TVirtualStreamerInfo::kFloat:   { dataType = "Float_t";        readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            case TVirtualStreamerInfo::kOffsetL + TVirtualStreamerInfo::kFloat16: { dataType = "Float16_t";      readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            case TVirtualStreamerInfo::kOffsetL + TVirtualStreamerInfo::kDouble:  { dataType = "Double_t";       readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            case TVirtualStreamerInfo::kOffsetL + TVirtualStreamerInfo::kDouble32:{ dataType = "Double32_t";     readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            case TVirtualStreamerInfo::kOffsetL + TVirtualStreamerInfo::kUChar:   { dataType = "UChar_t";        readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            case TVirtualStreamerInfo::kOffsetL + TVirtualStreamerInfo::kUShort:  { dataType = "unsigned short"; readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            case TVirtualStreamerInfo::kOffsetL + TVirtualStreamerInfo::kUInt:    { dataType = "unsigned int";   readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            case TVirtualStreamerInfo::kOffsetL + TVirtualStreamerInfo::kULong:   { dataType = "ULong_t";        readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            case TVirtualStreamerInfo::kOffsetL + TVirtualStreamerInfo::kULong64: { dataType = "ULong64_t";      readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            case TVirtualStreamerInfo::kOffsetL + TVirtualStreamerInfo::kBits:    { dataType = "unsigned int";   readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            // Array of built-in types [n]
            case TVirtualStreamerInfo::kOffsetP + TVirtualStreamerInfo::kBool:    { dataType = "Bool_t";         readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            case TVirtualStreamerInfo::kOffsetP + TVirtualStreamerInfo::kChar:    { dataType = "Char_t";         readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            case TVirtualStreamerInfo::kOffsetP + TVirtualStreamerInfo::kShort:   { dataType = "Short_t";        readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            case TVirtualStreamerInfo::kOffsetP + TVirtualStreamerInfo::kInt:     { dataType = "Int_t";          readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            case TVirtualStreamerInfo::kOffsetP + TVirtualStreamerInfo::kLong:    { dataType = "Long_t";         readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            case TVirtualStreamerInfo::kOffsetP + TVirtualStreamerInfo::kLong64:  { dataType = "Long64_t";       readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            case TVirtualStreamerInfo::kOffsetP + TVirtualStreamerInfo::kFloat:   { dataType = "Float_t";        readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            case TVirtualStreamerInfo::kOffsetP + TVirtualStreamerInfo::kFloat16: { dataType = "Float16_t";      readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            case TVirtualStreamerInfo::kOffsetP + TVirtualStreamerInfo::kDouble:  { dataType = "Double_t";       readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            case TVirtualStreamerInfo::kOffsetP + TVirtualStreamerInfo::kDouble32:{ dataType = "Double32_t";     readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            case TVirtualStreamerInfo::kOffsetP + TVirtualStreamerInfo::kUChar:   { dataType = "UChar_t";        readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            case TVirtualStreamerInfo::kOffsetP + TVirtualStreamerInfo::kUShort:  { dataType = "unsigned short"; readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            case TVirtualStreamerInfo::kOffsetP + TVirtualStreamerInfo::kUInt:    { dataType = "unsigned int";   readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            case TVirtualStreamerInfo::kOffsetP + TVirtualStreamerInfo::kULong:   { dataType = "ULong_t";        readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            case TVirtualStreamerInfo::kOffsetP + TVirtualStreamerInfo::kULong64: { dataType = "ULong64_t";      readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            case TVirtualStreamerInfo::kOffsetP + TVirtualStreamerInfo::kBits:    { dataType = "unsigned int";   readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            // array counter [n]
            case TVirtualStreamerInfo::kCounter: { dataType = "Int_t"; readerType = TTreeReaderDescriptor::ReaderType::kArray; break; }
            // other stuff (containers, classes, ...)
            case TVirtualStreamerInfo::kOffsetL + TVirtualStreamerInfo::kObjectp:
            case TVirtualStreamerInfo::kOffsetL + TVirtualStreamerInfo::kObjectP:
            case TVirtualStreamerInfo::kObjectp:
            case TVirtualStreamerInfo::kObjectP:
            case TVirtualStreamerInfo::kAnyp:
            case TVirtualStreamerInfo::kAnyP:
            case TVirtualStreamerInfo::kSTL + TVirtualStreamerInfo::kObjectp:
            case TVirtualStreamerInfo::kSTL + TVirtualStreamerInfo::kObjectP:
            // set as pointers and fall through to the next switches
               ispointer = true;
            case TVirtualStreamerInfo::kOffsetL + TVirtualStreamerInfo::kObject:
            case TVirtualStreamerInfo::kObject:
            case TVirtualStreamerInfo::kTString:
            case TVirtualStreamerInfo::kTNamed:
            case TVirtualStreamerInfo::kTObject:
            case TVirtualStreamerInfo::kAny:
            case TVirtualStreamerInfo::kBase:
            case TVirtualStreamerInfo::kSTL: {
               TClass *cl = element->GetClassPointer();
               R__ASSERT(cl);
               dataType = cl->GetName();
               // Check for containers
               if (cl == TClonesArray::Class()) { // TClonesArray
                  isclones = kClones;
                  containerName = "TClonesArray";
                  if (outer_isclones) { // If the parent is already a collection
                     dataType = "TClonesArray";
                  } else {
                     readerType = TTreeReaderDescriptor::ReaderType::kArray;
                     dataType = GetContainedClassName(branch, element, ispointer);
                  }
               } else if (cl->GetCollectionProxy()) { // STL collection
                  isclones = kSTL;
                  containerName = cl->GetName();
                  if (outer_isclones) { // If the parent is already a collection
                     dataType = cl->GetName();
                  } else {
                     readerType = TTreeReaderDescriptor::ReaderType::kArray;
                     TClass *valueClass = cl->GetCollectionProxy()->GetValueClass();
                     if (valueClass) { // Get class inside container
                        dataType = valueClass->GetName();
                     }
                     else { // Get built-in type inside container
                        TDataType *valueClassBuiltIn = TDataType::GetDataType(cl->GetCollectionProxy()->GetType());
                        if (valueClassBuiltIn) dataType = valueClassBuiltIn->GetName();
                        else Error("AnalyzeBranches", "Could not get type from collection %s in branch %s", cl->GetName(), branch->GetName());
                     }
                  }
               }

               // Analyze the actual element
               // Possible cases:
               // - Base class
               //   - Element and branch matches
               //     - Non-split: do nothing
               //     - Split: recurse with sub-sub-branches
               //   - Element and branch does not match: recurse with same branches
               // - Not base class
               //   - Element and branch matches
               //     - Non-split: do nothing
               //     - Split: recurse with sub-sub-branches
               //   - Element and branch does not match: recurse with same branches
               TBranch *parent = branch->GetMother()->GetSubBranch(branch);
               TVirtualStreamerInfo *objInfo = 0;
               if (branch->GetListOfBranches()->GetEntries()) {
                  objInfo = ((TBranchElement*)branch->GetListOfBranches()->At(0))->GetInfo();
               } else {
                  objInfo = branch->GetInfo();
               }
               if (element->IsBase()) { // Base class
                  isBase = true;
                  if (cl == TObject::Class() && info->GetClass()->CanIgnoreTObjectStreamer())
                  {
                     continue; // Ignore TObject
                  }

                  TBranchDescriptor *bdesc = 0;

                  if (branchEndName == element->GetName()) { // The element and the branch matches
                     if (branch->GetListOfBranches()->GetEntries() == 0) { // The branch contains a non-split base class
                        // FIXME: nothing to do in such cases, because readers cannot access
                        // non-split members and a reader for the whole branch will be added
                     } else { // The branch contains a split base class
                        Int_t pos = branchname.Last('.');
                        if (pos != -1) {
                           branchname.Remove(pos);
                        }
                        TString local_prefix = desc ? desc->fBranchName : TString(parent->GetName());
                        bdesc = new TBranchDescriptor(cl->GetName(), objInfo, local_prefix.Data(),
                                                      isclones, containerName, desc ? desc->fFullBranchName.Data() : 0, desc);
                        // Recurse: analyze sub-branches of the sub-branch
                        lookedAt += AnalyzeBranches(bdesc, branch, objInfo);
                        isLeaf = false;

                     }
                  } else { // The element and the branch does not match, we need to loop over the next branches
                     Int_t pos = branchname.Last('.');
                     if (pos != -1) {
                        branchname.Remove(pos);
                     }
                     TString local_prefix = desc ? desc->fBranchName : TString(parent->GetName());
                     objInfo = GetBaseClass(element);
                     if (objInfo == 0) {
                        continue; // There is no data in this base class
                     }
                     cl = objInfo->GetClass();
                     bdesc = new TBranchDescriptor(cl->GetName(), objInfo, local_prefix.Data(),
                                                    isclones, containerName, desc ? desc->fFullBranchName.Data() : 0, desc);
                     usedBranch = kFALSE;
                     // Recurse: analyze the sub-elements with the same branches
                     lookedAt += AnalyzeBranches(bdesc, branches, objInfo);
                  }
                  delete bdesc;
               } else { // Not base class
                  TBranchDescriptor *bdesc = 0;
                  if (branchEndName == element->GetName()) { // The element and the branch matches
                     if (branch->GetListOfBranches()->GetEntries() == 0) { // The branch contains a non-split class
                        // FIXME: nothing to do in such cases, because readers cannot access
                        // non-split members and a reader for the whole branch will be added
                     } else { // The branch contains a split class
                        if (isclones != kOut) {
                           // We have to guess the version number!
                           cl = TClass::GetClass(dataType);
                           objInfo = GetStreamerInfo(branch, branch->GetListOfBranches(), cl);
                        }
                        bdesc = new TBranchDescriptor(cl->GetName(), objInfo, branch->GetName(),
                                                      isclones, containerName, desc ? desc->fFullBranchName.Data() : 0, desc);
                        // Recurse: analyze sub-branches of the sub-branch
                        lookedAt += AnalyzeBranches(bdesc, branch, objInfo);
                        isLeaf = false;
                     }
                  } else { // The element and the branch does not match, we need to loop over the next branches
                     TString local_prefix = desc ? desc->fBranchName : TString(parent->GetName());
                     if (local_prefix.Length()) local_prefix += ".";
                     local_prefix += element->GetName();
                     objInfo = branch->GetInfo();
                     Int_t pos = branchname.Last('.');
                     if (pos != -1) {
                        branchname.Remove(pos);
                     }
                     if (isclones != kOut) {
                        // We have to guess the version number!
                        cl = TClass::GetClass(dataType);
                        objInfo = GetStreamerInfo(branch, branches, cl);
                     }
                     bdesc = new TBranchDescriptor(cl->GetName(), objInfo, local_prefix.Data(),
                                                   isclones, containerName, desc ? desc->fFullBranchName.Data() : 0, desc);
                     usedBranch = kFALSE;
                     skipped = kTRUE;
                     // Recurse: analyze the sub-elements with the same branches
                     lookedAt += AnalyzeBranches(bdesc, branches, objInfo);
                  }
                  delete bdesc;
               }

               break;
            }
            default:
               Error("AnalyzeBranch", "Unsupported type for %s (%d).", branch->GetName(), element->GetType());
         }

         if (!isBase && !skipped) { // Add reader for the whole branch
            TString dataMemberName = element->GetName();
            if (desc) {
               dataMemberName.Form("%s_%s", desc->fFullBranchName.Data(), element->GetName());
            }
            if (outer_isclones != kOut && readerType == TTreeReaderDescriptor::ReaderType::kArray) {
               Error("AnalyzeBranch", "Arrays inside collections are not supported yet (branch: %s).", branch->GetName());
            } else {
               if (outer_isclones != kOut || isclones != kOut) {
                  readerType = TTreeReaderDescriptor::ReaderType::kArray;
               }
               AddReader(readerType, dataType, dataMemberName, branch->GetName(), desc, isLeaf);
            }
         }

         // If the branch was used, jump to the next
         if (usedBranch) {
            branches.Next();
            ++lookedAt;
         }
      }

      return lookedAt;
   }

   ////////////////////////////////////////////////////////////////////////////////
   /// Analyze branch and add the variables found. The number of analyzed
   /// sub-branches is returned.

   UInt_t TTreeReaderGenerator::AnalyzeOldBranch(TBranch *branch)
   {
      UInt_t extraLookedAt = 0;
      TString prefix;

      TString branchName = branch->GetName();

      TObjArray *leaves = branch->GetListOfLeaves();
      Int_t nleaves = leaves ? leaves->GetEntriesFast() : 0;

      // Loop through leaves and analyze them
      for(int l=0;l<nleaves;l++) {
         TLeaf *leaf = (TLeaf*)leaves->UncheckedAt(l);
         extraLookedAt += AnalyzeOldLeaf(leaf, nleaves);
      }

      return extraLookedAt;
   }

   ////////////////////////////////////////////////////////////////////////////////
   /// Analyze the leaf and add the variables found.

   UInt_t TTreeReaderGenerator::AnalyzeOldLeaf(TLeaf *leaf, Int_t nleaves)
   {
      if (leaf->IsA()==TLeafObject::Class()) {
         Error("AnalyzeOldLeaf","TLeafObject not supported yet");
         return 0;
      }

      TString leafTypeName = leaf->GetTypeName();
      Int_t pos = leafTypeName.Last('_');
      //if (pos != -1) leafTypeName.Remove(pos); // FIXME: this is not required since it makes Float_t -> Float

      // Analyze dimensions
      UInt_t dim = 0;
      std::vector<Int_t> maxDim;

      TString dimensions;
      TString temp = leaf->GetName();
      pos = temp.Index("[");
      if (pos != -1) {
         if (pos) temp.Remove(0, pos);
         dimensions.Append(temp);
      }
      temp = leaf->GetTitle();
      pos = temp.Index("[");
      if (pos != -1) {
         if (pos) temp.Remove(0, pos);
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
               maxDim.push_back(-1); // maxDim[dim] = -1; // Loop over all elements;
            } else {
               scanindex = sscanf(current,"%d",&index);
               if (scanindex) {
                  maxDim.push_back(index); // maxDim[dim] = index;
               } else {
                  maxDim.push_back(-2); // maxDim[dim] = -2; // Index is calculated via a variable.
               }
            }
            dim ++;
            current = (char*)strstr( current, "[" );
         }
      }

      if (dim == 0 && leaf->IsA() == TLeafC::Class()) {
         dim = 1; // For C style strings
      }

      TTreeReaderDescriptor::ReaderType type;
      TString dataType;
      switch (dim) {
         case 0: {
            type = TTreeReaderDescriptor::ReaderType::kValue;
            dataType = leafTypeName;
            break;
         }
         case 1: {
            type = TTreeReaderDescriptor::ReaderType::kArray;
            dataType = leafTypeName;
            break;
         }
         default: {
            // TODO: transform this
            /*type = "TArrayProxy<";
            for(Int_t ind = dim - 2; ind > 0; --ind) {
               type += "TMultiArrayType<";
            }
            type += "TArrayType<";
            type += leaf->GetTypeName();
            type += ",";
            type += maxDim[dim-1];
            type += "> ";
            for(Int_t ind = dim - 2; ind > 0; --ind) {
               type += ",";
               type += maxDim[ind];
               type += "> ";
            }
            type += ">";*/
            break;
         }
      }

      // If there are multiple leaves (leaflist) the name of the branch is
      // <branch_name>.<leaf_name>
      // (Otherwise the brach name does not change)
      TString branchName = leaf->GetBranch()->GetName();
      if (nleaves > 1) {
         branchName.Form("%s.%s", leaf->GetBranch()->GetName(), leaf->GetName());
      }

      AddReader(type, dataType, leaf->GetName(), branchName, 0, kTRUE);

      return 0;
   }

   ////////////////////////////////////////////////////////////////////////////////
   /// Check whether a branch should have a corresponding reader added, depending
   /// on the options provided by the user.

   Bool_t TTreeReaderGenerator::BranchNeedsReader(TString branchName, TBranchDescriptor *parent, Bool_t isLeaf)
   {
      if (isLeaf) { // Branch is a leaf
         if (fIncludeAllLeaves) return kTRUE;
         if (FindStringInVector(branchName, fIncludeLeaves)) return kTRUE;
         if (FindStringInVector(branchName, fIncludeStruct)) return kTRUE;
         if (!parent) { // Branch is topmost (top-level leaf)
            if (fIncludeAllTopmost) return kTRUE;
         } else {       // Branch is not topmost
            while (parent) {
               if (FindStringInVector(parent->fBranchName, fIncludeLeaves)) {
                  return kTRUE;
               }
               parent = parent->fParent;
            }
         }
      } else {      // Branch is not a leaf (has sub-branches)
         if (FindStringInVector(branchName, fIncludeStruct)) return kTRUE;
         if (!parent) { // Branch is topmost
            if (fIncludeAllTopmost) return kTRUE;
         } else {       // Branch is not topmost
            
         }
      }
      return false;
   }

   ////////////////////////////////////////////////////////////////////////////////
   /// Parse the user options.

   void TTreeReaderGenerator::ParseOptions() {
      if (fOptions.EqualTo("")) {
         fIncludeAllLeaves = kTRUE;
      } else if (fOptions.EqualTo("@")) {
         fIncludeAllTopmost = kTRUE;
      } else {
         TObjArray *tokens = fOptions.Tokenize(TString(";"));
         for (Int_t i = 0; i < tokens->GetEntries(); ++i) {
            TString token = ((TObjString*)tokens->At(i))->GetString();
            if ( token.Length() == 0 || (token.Length() == 1 && token[0] == '@') ) {
               Warning("ParseOptions", "Ignored empty branch name in option string.");
            } else if (token[0] == '@') {
               fIncludeStruct.push_back(TString(token.Data()+1));
            } else {
               fIncludeLeaves.push_back(token);
            }
         }
         delete tokens;
      }

      printf("Parsed options: ");
      if(fIncludeAllLeaves) printf("Include all leaves\n");
      if(fIncludeAllTopmost) printf("Include all topmost\n");
      if(fIncludeStruct.size()>0) { printf("Struct: "); for(TString s : fIncludeStruct) printf("%s ", s.Data()); printf("\n"); }
      if(fIncludeLeaves.size()>0) { printf("Leaves: "); for(TString s : fIncludeLeaves) printf("%s ", s.Data()); printf("\n"); }
   }

   ////////////////////////////////////////////////////////////////////////////////
   /// Analyze tree and extract readers.

   void TTreeReaderGenerator::AnalyzeTree(TTree *tree)
   {
      TIter next(tree->GetListOfBranches());
      TBranch *branch;

      // Loop through branches
      while ( (branch = (TBranch*)next()) ) {
         TVirtualStreamerInfo *info = 0;
         // Get the name and the class of the branch
         const char *branchName = branch->GetName();
         const char *branchClassName = branch->GetClassName();
         TClass *cl = TClass::GetClass(branchClassName);

         // Add headers for user classes
         if (branchClassName && strlen(branchClassName)) {
            AddHeader(cl);
         }

         TString type = "unknown";   // Type of branch
         ELocation isclones = kOut;  // Type of container
         TString containerName = ""; // Name of container
         TBranchDescriptor *desc = 0;
         // Check whether the branch is a container
         if (cl) {
            // Check if it is a TClonesArray
            if (cl == TClonesArray::Class()) {
               isclones = kClones;
               containerName = "TClonesArray";
               if (branch->IsA()==TBranchElement::Class()) {
                  // Get the class inside the TClonesArray
                  const char *cname = ((TBranchElement*)branch)->GetClonesName();
                  TClass *ncl = TClass::GetClass(cname);
                  if (ncl) {
                     cl = ncl;
                     info = GetStreamerInfo(branch, branch->GetListOfBranches(), cl);
                  } else {
                     Error("AnalyzeTree",
                           "Introspection of TClonesArray in older file not implemented yet.");
                  }
               } else {
                  TClonesArray **ptr = (TClonesArray**)branch->GetAddress();
                  TClonesArray *clones = 0;
                  if (ptr==0) {
                     clones = new TClonesArray;
                     branch->SetAddress(&clones);
                     ptr = &clones;
                  }
                  branch->GetEntry(0);
                  TClass *ncl = *ptr ? (*ptr)->GetClass() : 0;
                  if (ncl) {
                     cl = ncl;
                  } else {
                     Error("AnalyzeTree",
                           "Introspection of TClonesArray for %s failed.",branch->GetName());
                  }
               }
            // Check if it is an STL container
            } else if (cl->GetCollectionProxy()) {
               isclones = kSTL;
               containerName = cl->GetName();
               // Get the type inside container
               if (cl->GetCollectionProxy()->GetValueClass()) { // Class inside container
                  cl = cl->GetCollectionProxy()->GetValueClass();
               } else { // RAW type (or missing class) inside container
                  // TODO: CheckForMissingClass?
                  AddReader(TTreeReaderDescriptor::ReaderType::kArray,
                            TDataType::GetDataType(cl->GetCollectionProxy()->GetType())->GetName(),
                            branch->GetName(), branch->GetName(), 0, kTRUE);
                  continue; // Nothing else to with this branch in these cases
               }
            }

            // Check class inside container and create a descriptor or add a reader
            if (cl) {
               if (cl->TestBit(TClass::kIsEmulation) || branchName[strlen(branchName)-1] == '.' || branch->GetSplitLevel()) {
                  TBranchElement *be = dynamic_cast<TBranchElement*>(branch);
                  TVirtualStreamerInfo *beinfo = (be && isclones == kOut)
                     ? be->GetInfo() : cl->GetStreamerInfo(); // the 2nd hand need to be fixed
                  // Create descriptor
                  desc = new TBranchDescriptor(cl->GetName(), beinfo, branchName, isclones, containerName);
                  info = beinfo;
               } else {
                  // Add a reader for non-split classes
                  AddReader(isclones == kOut ?
                              TTreeReaderDescriptor::ReaderType::kValue
                            : TTreeReaderDescriptor::ReaderType::kArray,
                            cl->GetName(), branchName, branchName, 0, kTRUE);
                  // TODO: can't we just put a continue here?
               }
            }
         }

         // Analyze sub-branches (if exist) and add readers
         if (branch->GetListOfBranches()->GetEntries() == 0) { // Branch is non-splitted
            if (cl) { // Non-split object
               if (desc) { // If there is a descriptor add reader (otherwise
                           // it was already added).
                  AddReader(isclones == kOut ?
                              TTreeReaderDescriptor::ReaderType::kValue
                            : TTreeReaderDescriptor::ReaderType::kArray,
                            desc->GetName(), desc->fBranchName, desc->fBranchName, 0, kTRUE);
               }
            } else { // Top-level RAW type
               AnalyzeOldBranch(branch); // Analyze branch and extract readers
            }
         } else { // Branch is splitted
            TIter subnext( branch->GetListOfBranches() );
            if (desc) {
               // Analyze sub-branches and extract readers
               AnalyzeBranches(desc, dynamic_cast<TBranchElement*>(branch), info);
               // Also add a reader for the whole branch
               AddReader(isclones == kOut ?
                              TTreeReaderDescriptor::ReaderType::kValue
                            : TTreeReaderDescriptor::ReaderType::kArray,
                            desc->GetName(), desc->fBranchName, desc->fBranchName, 0, kFALSE);
            }
         }
         delete desc;
      }
   }

   ////////////////////////////////////////////////////////////////////////////////
   /// Generate code for selector class.

   void TTreeReaderGenerator::WriteSelector()
   {
      // If no name is given, set to default (name of the tree)
      if (!fClassname) fClassname = fTree->GetName();

      TString treefile;
      if (fTree->GetDirectory() && fTree->GetDirectory()->GetFile()) {
         treefile = fTree->GetDirectory()->GetFile()->GetName();
      } else {
         treefile = "Memory Directory";
      }
      // In the case of a chain, the GetDirectory information usually does
      // pertain to the Chain itself but to the currently loaded tree.
      // So we can not rely on it.
      Bool_t ischain = fTree->InheritsFrom(TChain::Class());
      Bool_t isHbook = fTree->InheritsFrom("THbookTree");
      if (isHbook)
         treefile = fTree->GetTitle();

      //======================Generate classname.h=====================
      TString thead;
      thead.Form("%s.h", fClassname.Data());
      FILE *fp = fopen(thead, "w");
      if (!fp) {
         Error("WriteSelector","cannot open output file %s", thead.Data());
         return;
      }
      // Print header
      TDatime td;
      fprintf(fp,"//////////////////////////////////////////////////////////\n");
      fprintf(fp,"// This class has been automatically generated on\n");
      fprintf(fp,"// %s by ROOT version %s\n", td.AsString(), gROOT->GetVersion());
      if (!ischain) {
         fprintf(fp,"// from TTree %s/%s\n", fTree->GetName(), fTree->GetTitle());
         fprintf(fp,"// found on file: %s\n", treefile.Data());
      } else {
         fprintf(fp,"// from TChain %s/%s\n", fTree->GetName(), fTree->GetTitle());
      }
      fprintf(fp,"//////////////////////////////////////////////////////////\n");
      fprintf(fp,"\n");
      fprintf(fp,"#ifndef %s_h\n", fClassname.Data());
      fprintf(fp,"#define %s_h\n", fClassname.Data());
      fprintf(fp,"\n");
      fprintf(fp,"#include <TROOT.h>\n");
      fprintf(fp,"#include <TChain.h>\n");
      fprintf(fp,"#include <TFile.h>\n");
      if (isHbook) fprintf(fp, "#include <THbookFile.h>\n");
      fprintf(fp,"#include <TSelector.h>\n");
      fprintf(fp,"#include <TTreeReader.h>\n");
      fprintf(fp,"#include <TTreeReaderValue.h>\n"); // TODO: optimization: only if there are leaf values
      fprintf(fp,"#include <TTreeReaderArray.h>\n"); // TODO: optimization: only if there are leaf arrays

      // Add headers for user classes
      fprintf(fp,"\n\n");
      fprintf(fp,"// Headers needed by this particular selector\n");
      TIter next(&fListOfHeaders);
      TObject *header;
      while ( (header = next()) ) {
         fprintf(fp, "%s", header->GetTitle());
      }
      fprintf(fp, "\n\n");

      // Generate class declaration
      fprintf(fp,"class %s : public TSelector {\n", fClassname.Data());
      fprintf(fp,"public :\n");
      fprintf(fp,"   TTreeReader     fReader;  //!the tree reader\n");
      fprintf(fp,"   TTree          *fChain;   //!pointer to the analyzed TTree or TChain\n");
      // Generate TTreeReaderValues and Arrays
      fprintf(fp,"\n   // Variables used to access and store the data\n");
      next = &fListOfReaders;
      TTreeReaderDescriptor *descriptor;
      while ( ( descriptor = (TTreeReaderDescriptor*)next() ) ) {
         fprintf(fp, "   TTreeReader%s<%s> %s;\n", descriptor->fType == TTreeReaderDescriptor::ReaderType::kValue ? "Value" : "Array",
                                                   descriptor->fDataType.Data(),
                                                   descriptor->fName.Data() );
      }
      fprintf(fp, "\n\n");
      // Generate class member functions prototypes
      next.Reset();
      fprintf(fp,"   %s(TTree * /*tree*/ =0) : fChain(0)", fClassname.Data());
      while ( ( descriptor = (TTreeReaderDescriptor*)next() ) ) {
         fprintf(fp, ",\n            %s(fReader, \"%s\")", descriptor->fName.Data(), descriptor->fBranchName.Data());
      }
      fprintf(fp," { }\n");
      fprintf(fp,"   virtual ~%s() { }\n", fClassname.Data());
      fprintf(fp,"   virtual Int_t   Version() const { return 2; }\n");
      fprintf(fp,"   virtual void    Begin(TTree *tree);\n");
      fprintf(fp,"   virtual void    SlaveBegin(TTree *tree);\n");
      fprintf(fp,"   virtual void    Init(TTree *tree);\n");
      fprintf(fp,"   virtual Bool_t  Notify();\n");
      fprintf(fp,"   virtual Bool_t  Process(Long64_t entry);\n");
      fprintf(fp,"   virtual Int_t   GetEntry(Long64_t entry, Int_t getall = 0) { return fChain ? fChain->GetTree()->GetEntry(entry, getall) : 0; }\n");
      fprintf(fp,"   virtual void    SetOption(const char *option) { fOption = option; }\n");
      fprintf(fp,"   virtual void    SetObject(TObject *obj) { fObject = obj; }\n");
      fprintf(fp,"   virtual void    SetInputList(TList *input) { fInput = input; }\n");
      fprintf(fp,"   virtual TList  *GetOutputList() const { return fOutput; }\n");
      fprintf(fp,"   virtual void    SlaveTerminate();\n");
      fprintf(fp,"   virtual void    Terminate();\n\n");
      fprintf(fp,"   ClassDef(%s,0);\n", fClassname.Data());
      fprintf(fp,"};\n");
      fprintf(fp,"\n");
      fprintf(fp,"#endif\n\n");
      // Generate code for Init and Notify
      fprintf(fp,"#ifdef %s_cxx\n", fClassname.Data());
      fprintf(fp,"void %s::Init(TTree *tree)\n", fClassname.Data());
      fprintf(fp,"{\n");
      fprintf(fp,"   // The Init() function is called when the selector needs to initialize\n"
                 "   // a new tree or chain. Typically here the branch addresses and branch\n" // TODO: replace comment?
                 "   // pointers of the tree will be set.\n"
                 "   // It is normally not necessary to make changes to the generated\n"
                 "   // code, but the routine can be extended by the user if needed.\n"
                 "   // Init() will be called many times when running on PROOF\n"
                 "   // (once per file to be processed).\n\n");
      fprintf(fp,"   fReader.SetTree(tree);");
      fprintf(fp,"}\n\n");

      fprintf(fp,"Bool_t %s::Notify()\n", fClassname.Data());
      fprintf(fp,"{\n");
      fprintf(fp,"   // The Notify() function is called when a new file is opened. This\n"
                 "   // can be either for a new TTree in a TChain or when when a new TTree\n"
                 "   // is started when using PROOF. It is normally not necessary to make changes\n"
                 "   // to the generated code, but the routine can be extended by the\n"
                 "   // user if needed. The return value is currently not used.\n\n");
      fprintf(fp,"   return kTRUE;\n");
      fprintf(fp,"}\n\n");
      fprintf(fp,"#endif // #ifdef %s_cxx\n", fClassname.Data());
      fclose(fp);

      //======================Generate classname.C=====================
      TString tcimp;
      tcimp.Form("%s.C", fClassname.Data());
      FILE *fpc = fopen(tcimp, "w");
      if (!fpc) {
         Error("WriteSelector","cannot open output file %s", tcimp.Data());
         fclose(fp);
         return;
      }

      fprintf(fpc,"#define %s_cxx\n", fClassname.Data());
      fprintf(fpc,"// The class definition in %s.h has been generated automatically\n", fClassname.Data());
      fprintf(fpc,"// by the ROOT utility TTree::MakeSelector(). This class is derived\n");
      fprintf(fpc,"// from the ROOT class TSelector. For more information on the TSelector\n"
                  "// framework see $ROOTSYS/README/README.SELECTOR or the ROOT User Manual.\n\n");
      fprintf(fpc,"// The following methods are defined in this file:\n");
      fprintf(fpc,"//    Begin():        called every time a loop on the tree starts,\n");
      fprintf(fpc,"//                    a convenient place to create your histograms.\n");
      fprintf(fpc,"//    SlaveBegin():   called after Begin(), when on PROOF called only on the\n"
                  "//                    slave servers.\n");
      fprintf(fpc,"//    Process():      called for each event, in this function you decide what\n");
      fprintf(fpc,"//                    to read and fill your histograms.\n");
      fprintf(fpc,"//    SlaveTerminate: called at the end of the loop on the tree, when on PROOF\n"
                  "//                    called only on the slave servers.\n");
      fprintf(fpc,"//    Terminate():    called at the end of the loop on the tree,\n");
      fprintf(fpc,"//                    a convenient place to draw/fit your histograms.\n");
      fprintf(fpc,"//\n");
      fprintf(fpc,"// To use this file, try the following session on your Tree T:\n");
      fprintf(fpc,"//\n");
      fprintf(fpc,"// root> T->Process(\"%s.C\")\n", fClassname.Data());
      fprintf(fpc,"// root> T->Process(\"%s.C\",\"some options\")\n", fClassname.Data());
      fprintf(fpc,"// root> T->Process(\"%s.C+\")\n", fClassname.Data());
      fprintf(fpc,"//\n\n");
      fprintf(fpc,"#include \"%s\"\n",thead.Data());
      fprintf(fpc,"#include <TH2.h>\n");
      fprintf(fpc,"#include <TStyle.h>\n");
      fprintf(fpc,"\n");
      // generate code for class member function Begin
      fprintf(fpc,"\n");
      fprintf(fpc,"void %s::Begin(TTree * /*tree*/)\n", fClassname.Data());
      fprintf(fpc,"{\n");
      fprintf(fpc,"   // The Begin() function is called at the start of the query.\n");
      fprintf(fpc,"   // When running with PROOF Begin() is only called on the client.\n");
      fprintf(fpc,"   // The tree argument is deprecated (on PROOF 0 is passed).\n");
      fprintf(fpc,"\n");
      fprintf(fpc,"   TString option = GetOption();\n");
      fprintf(fpc,"\n");
      fprintf(fpc,"}\n");
      // generate code for class member function SlaveBegin
      fprintf(fpc,"\n");
      fprintf(fpc,"void %s::SlaveBegin(TTree * /*tree*/)\n", fClassname.Data());
      fprintf(fpc,"{\n");
      fprintf(fpc,"   // The SlaveBegin() function is called after the Begin() function.\n");
      fprintf(fpc,"   // When running with PROOF SlaveBegin() is called on each slave server.\n");
      fprintf(fpc,"   // The tree argument is deprecated (on PROOF 0 is passed).\n");
      fprintf(fpc,"\n");
      fprintf(fpc,"   TString option = GetOption();\n");
      fprintf(fpc,"\n");
      fprintf(fpc,"}\n");
      // generate code for class member function Process
      fprintf(fpc,"\n");
      fprintf(fpc,"Bool_t %s::Process(Long64_t entry)\n", fClassname.Data());
      fprintf(fpc,"{\n");
      fprintf(fpc,"   // The Process() function is called for each entry in the tree (or possibly\n"
                  "   // keyed object in the case of PROOF) to be processed. The entry argument\n"
                  "   // specifies which entry in the currently loaded tree is to be processed.\n"
                  "   // It can be passed to either %s::GetEntry() or TBranch::GetEntry()\n"
                  "   // to read either all or the required parts of the data. When processing\n"
                  "   // keyed objects with PROOF, the object is already loaded and is available\n"
                  "   // via the fObject pointer.\n"
                  "   //\n"
                  "   // This function should contain the \"body\" of the analysis. It can contain\n"
                  "   // simple or elaborate selection criteria, run algorithms on the data\n"
                  "   // of the event and typically fill histograms.\n"
                  "   //\n"
                  "   // The processing can be stopped by calling Abort().\n"
                  "   //\n"
                  "   // Use fStatus to set the return value of TTree::Process().\n"
                  "   //\n"
                  "   // The return value is currently not used.\n\n", fClassname.Data());
      fprintf(fpc,"\n");
      fprintf(fpc,"   fReader.SetEntry(entry);\n\n\n");
      fprintf(fpc,"   return kTRUE;\n");
      fprintf(fpc,"}\n");
      // generate code for class member function SlaveTerminate
      fprintf(fpc,"\n");
      fprintf(fpc,"void %s::SlaveTerminate()\n", fClassname.Data());
      fprintf(fpc,"{\n");
      fprintf(fpc,"   // The SlaveTerminate() function is called after all entries or objects\n"
                  "   // have been processed. When running with PROOF SlaveTerminate() is called\n"
                  "   // on each slave server.");
      fprintf(fpc,"\n");
      fprintf(fpc,"\n");
      fprintf(fpc,"}\n");
      // generate code for class member function Terminate
      fprintf(fpc,"\n");
      fprintf(fpc,"void %s::Terminate()\n", fClassname.Data());
      fprintf(fpc,"{\n");
      fprintf(fpc,"   // The Terminate() function is the last function to be called during\n"
                  "   // a query. It always runs on the client, it can be used to present\n"
                  "   // the results graphically or save the results to file.");
      fprintf(fpc,"\n");
      fprintf(fpc,"\n");
      fprintf(fpc,"}\n");
      fclose(fpc);
   }
}
