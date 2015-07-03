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
#include "TStreamerInfo.h"
#include "TTree.h"
#include "TVirtualCollectionProxy.h"
#include "TVirtualStreamerInfo.h"

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

static TVirtualStreamerInfo *GetStreamerInfo(TBranch *branch, TIter current, TClass *cl)
{
   // Return the correct TStreamerInfo of class 'cname' in the list of
   // branch (current) [Assuming these branches correspond to a flattened
   // version of the class.]

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
         // (this happens in the case of embedded objects inside an object inside
         // a clones array split more than one level)
         printf("TODO: AnalyzeOldBranch with nleaves>1\n");
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
         TVirtualStreamerInfo *info = 0;
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
         ELocation isclones = kOut;
         TString containerName = "";
         // Check for container classes
         if (cl) {
            // Check if it is a TClonesArray
            if (cl == TClonesArray::Class()) {
               isclones = kClones;
               containerName = "TClonesArray";
               if (branch->IsA()==TBranchElement::Class()) {
                  // Get the class inside the TClonesArray
                  const char *cname = ((TBranchElement*)branch)->GetClonesName();
                  TClass *ncl = TClass::GetClass(cname);
                  printf("\tClass inside TClonesArray: %s\n", cname);
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
            // Check if it is an STL collection
            } else if (cl->GetCollectionProxy()) {
               isclones = kSTL; // It is an STL container
               containerName = cl->GetName();
               // Check the type inside container
               if (cl->GetCollectionProxy()->GetValueClass()) { // Class inside container
                  cl = cl->GetCollectionProxy()->GetValueClass();
               } else { // RAW type (or missing class) inside container
                  // TODO: CheckForMissingClass?
                  // TODO: AddHeader(cl); (if using TRV)
                  AddReader(TTreeReaderDescriptor::ReaderType::kArray,
                            TDataType::GetDataType(cl->GetCollectionProxy()->GetType())->GetName(),
                            branch->GetName(),
                            branch->GetName());
                  continue; // Nothing else to with this branch in these cases
               }
            }
            
            if (cl) {
               if (cl->TestBit(TClass::kIsEmulation) || branchName[strlen(branchName)-1] == '.' || branch->GetSplitLevel()) {
                  // TODO: implement this
                  printf("Classes, emulation/split case\n");
               } else {
                  // Generate a value or an array for non-split classes
                  AddReader(isclones == kOut ?
                              TTreeReaderDescriptor::ReaderType::kValue
                            : TTreeReaderDescriptor::ReaderType::kArray,
                            cl->GetName(), branchName, branchName);
                  // TODO: can't we just put a continue here?
               }
            }
         }
         
         if (branch->GetListOfBranches()->GetEntries() == 0) { // Branch is non-splitted
            
            if (cl) { // Non-split object
               // TODO: implement this
               printf("TODO: non-split object\n");
            } else { // Top-level RAW type
               AnalyzeOldBranch(branch, 0);
            }
         
         } else { // Branch is splitted
            // TODO: implement this
            printf("TODO: splitted branch\n");
         }
      }
      
      
   }
   
   void TTreeSelectorReaderGenerator::WriteSelector()
   {
      // Generate code for selector class.

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
