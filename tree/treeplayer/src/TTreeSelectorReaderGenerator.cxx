// @(#)root/treeplayer:$Id$
// Author: Akos Hajdu 22/06/2015

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TTreeSelectorReaderGenerator.h"
#include <stdio.h>

#include "TClass.h"
#include "TClassEdit.h"
#include "TError.h"
#include "TLeaf.h"
#include "TLeafC.h"
#include "TLeafObject.h"
#include "TTree.h"
#include "TVirtualCollectionProxy.h"

namespace ROOT {

   TTreeSelectorReaderGenerator::TTreeSelectorReaderGenerator(TTree* tree,
                                            const char *classname, UInt_t maxUnrolling) : 
      fTree(tree),
      fClassname(classname),
      fMaxUnrolling(maxUnrolling)
   {
      // Constructor.
      AnalyzeTree(fTree);
      
      WriteSelector();
   }
   
   void TTreeSelectorReaderGenerator::AddHeader(TClass *cl)
   {
      // Add a header inclusion request.
      if (cl==0) return;
      
      printf("\tAddHeader(%s)\n", cl->GetName());
      
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
         printf("\t\tAdded directive: %s", directive.Data());
      }
   }
   
   
   void TTreeSelectorReaderGenerator::AddReader(TTreeReaderDescriptor::ReaderType type, TString dataType, TString name, TString branchName)
   {
      fListOfReaders.Add( new TTreeReaderDescriptor(type, dataType, name, branchName) );
      printf("Added reader: TTreeReader%s<%s> %s (branch: %s)\n", type == TTreeReaderDescriptor::ReaderType::kValue ? "Value" : "Array",
                                                                  dataType.Data(),
                                                                  name.Data(),
                                                                  branchName.Data());
   }
   
   UInt_t TTreeSelectorReaderGenerator::AnalyzeOldBranch(TBranch *branch, UInt_t level)
   {
      // Analyze branch and add the variables found.
      // The number of analyzed sub-branches is returned.
      
      UInt_t extraLookedAt = 0;
      TString prefix;

      TString branchName = branch->GetName();

      TObjArray *leaves = branch->GetListOfLeaves();
      Int_t nleaves = leaves ? leaves->GetEntriesFast() : 0;
      
      if (nleaves>1) {
         // TODO: implement this
      } else {
         TLeaf *leaf = (TLeaf*)leaves->UncheckedAt(0);
         extraLookedAt += AnalyzeOldLeaf(leaf);
      }
      
      return extraLookedAt;
   }
   
   UInt_t TTreeSelectorReaderGenerator::AnalyzeOldLeaf(TLeaf *leaf)
   {
      // Analyze the leaf and add the variables found.
      
      if (leaf->IsA()==TLeafObject::Class()) {
         Error("AnalyzeOldLeaf","TLeafObject not supported yet");
         return 0;
      }
      
      TString leafTypeName = leaf->GetTypeName();
      Int_t pos = leafTypeName.Last('_');
      if (pos != -1) leafTypeName.Remove(pos);
      
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
      
      
      AddReader(type, dataType, leaf->GetName(), leaf->GetBranch()->GetName());
      
      return 0;
   }
   
   void TTreeSelectorReaderGenerator::AnalyzeTree(TTree *tree)
   {
      // Analyze tree.
      TIter next(tree->GetListOfBranches());
      TBranch *branch;
      
      // Loop through branches
      while ( (branch = (TBranch*)next()) ) {
         // Get the name and the class of the branch
         const char *branchName = branch->GetName();
         const char *branchClassName = branch->GetClassName();
         TClass *cl = TClass::GetClass(branchClassName);
         printf("Branch name: %s, class name: %s\n", branch->GetName(), branch->GetClassName());
         
         // Add headers for user classes
         if (branchClassName && strlen(branchClassName)) {
            AddHeader(cl);
         }
         
         TString type = "unknown";
         // Classes
         if (cl) {
            // TODO: implement this
         }
         
         if (branch->GetListOfBranches()->GetEntries() == 0) { // Branch is non-splitted
            
            if (cl) { // Non-split object
               // TODO: implement this
            } else { // Top-level RAW type
               AnalyzeOldBranch(branch, 0);
            }
         
         } else { // Branch is splitted
            // TODO: implement this
         }
      }
      
      
   }
   
   void TTreeSelectorReaderGenerator::WriteSelector()
   {
      // Generate code for selector class.
   }
}
