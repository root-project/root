// @(#)root/tree:$Id$
// Author: Rene Brun   05/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// A TSelector object is used by the TTree::Draw, TTree::Scan,            //
// TTree::Process to navigate in a TTree and make selections.             //
// It contains the following main methods:                                //
//                                                                        //
// void TSelector::Init(TTree *t). Called every time a new TTree is       //
//    attached.                                                           //
//                                                                        //
// void TSelector::SlaveBegin(). Create e.g. histograms in this method.   //
//    This method is called (with or without PROOF) before looping on the //
//    entries in the Tree. When using PROOF, this method is called on     //
//    each worker node.                                                   //
// void TSelector::Begin(). Mostly for backward compatibility; use        //
//    SlaveBegin() instead. Both methods are called before looping on the //
//    entries in the Tree. When using PROOF, Begin() is called on the     //
//    client only.                                                        //
//                                                                        //
// Bool_t TSelector::Notify(). This method is called at the first entry   //
//    of a new file in a chain.                                           //
//                                                                        //
// Bool_t TSelector::Process(Long64_t entry). This method is called       //
//    to process an entry. It is the user's responsability to read        //
//    the corresponding entry in memory (may be just a partial read).     //
//    Once the entry is in memory one can apply a selection and if the    //
//    entry is selected histograms can be filled. Processing stops        //
//    when this function returns kFALSE. This function combines the       //
//    next two functions in one, avoiding to have to maintain state       //
//    in the class to communicate between these two functions.            //
//    See WARNING below about entry.                                      //
//    This method is used by PROOF.                                       //
// Bool_t TSelector::ProcessCut(Long64_t entry). This method is called    //
//    before processing entry. It is the user's responsability to read    //
//    the corresponding entry in memory (may be just a partial read).     //
//    The function returns kTRUE if the entry must be processed,          //
//    kFALSE otherwise. This method is obsolete, use Process().           //
//    See WARNING below about entry.                                      //
// void TSelector::ProcessFill(Long64_t entry). This method is called     //
//    for all selected entries. User fills histograms in this function.   //
//    This method is obsolete, use Process().                             //
//    See WARNING below about entry.                                      //
// void TSelector::SlaveTerminate(). This method is called at the end of  //
//    the loop on all PROOF worker nodes. In local mode this method is    //
//    called on the client too.                                           //
// void TSelector::Terminate(). This method is called at the end of       //
//    the loop on all entries. When using PROOF Terminate() is call on    //
//    the client only. Typically one performs the fits on the produced    //
//    histograms or write the histograms to file in this method.          //
//                                                                        //
// WARNING when a selector is used with a TChain:                         //
//    in the Process, ProcessCut, ProcessFill function, you must use      //
//    the pointer to the current Tree to call GetEntry(entry).            //
//    entry is always the local entry number in the current tree.         //
//    Assuming that fChain is the pointer to the TChain being processed,  //
//    use fChain->GetTree()->GetEntry(entry);                             //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#include "TROOT.h"
#include "TSystem.h"
#include "TTree.h"
#include "TError.h"
#include "TSelectorCint.h"
#include "TClass.h"
#include "TInterpreter.h"


ClassImp(TSelector)

//______________________________________________________________________________
TSelector::TSelector() : TObject()
{
   // Default selector ctor.

   fStatus = 0;
   fAbort  = kContinue;
   fObject = 0;
   fInput  = 0;
   fOutput = new TSelectorList;
   fOutput->SetOwner();
}

//______________________________________________________________________________
TSelector::~TSelector()
{
   // Selector destructor.

   delete fOutput;
}

//______________________________________________________________________________
void TSelector::Abort(const char *why, EAbort what)
{
   // Abort processing. If what = kAbortProcess, the Process() loop will be
   // aborted. If what = kAbortFile, the current file in a chain will be
   // aborted and the processing will continue with the next file, if there
   // is no next file then Process() will be aborted. Abort() can also  be
   // called from Begin(), SlaveBegin(), Init() and Notify(). After abort
   // the SlaveTerminate() and Terminate() are always called. The abort flag
   // can be checked in these methods using GetAbort().

   fAbort = what;
   TString mess = "Abort";
   if (fAbort == kAbortProcess)
      mess = "AbortProcess";
   else if (fAbort == kAbortFile)
      mess = "AbortFile";

   Info(mess, "%s", why);
}

//______________________________________________________________________________
TSelector *TSelector::GetSelector(const char *filename)
{
   // The code in filename is loaded (interpreted or compiled, see below),
   // filename must contain a valid class implementation derived from TSelector.
   //
   // If filename is of the form file.C, the file will be interpreted.
   // If filename is of the form file.C++, the file file.C will be compiled
   // and dynamically loaded. The corresponding binary file and shared
   // library will be deleted at the end of the function.
   // If filename is of the form file.C+, the file file.C will be compiled
   // and dynamically loaded. At next call, if file.C is older than file.o
   // and file.so, the file.C is not compiled, only file.so is loaded.
   //
   // The static function returns a pointer to a TSelector object

   // If the filename does not contain "." assume class is compiled in
   TString localname;
   Bool_t fromFile = kFALSE;
   if (strchr(filename, '.') != 0) {
      //Interpret/compile filename via CINT
      localname  = ".L ";
      localname += filename;
      gROOT->ProcessLine(localname);
      fromFile = kTRUE;
   }

   //loop on all classes known to CINT to find the class on filename
   //that derives from TSelector
   const char *basename = gSystem->BaseName(filename);
   if (!basename) {
      ::Error("TSelector::GetSelector","unable to determine the classname for file %s", filename);
      return 0;
   }
   TString aclicmode,args,io;
   localname = gSystem->SplitAclicMode(basename,aclicmode,args,io);
   if (localname.Last('.') != kNPOS)
      localname.Remove(localname.Last('.'));

   // if a file was not specified, try to load the class via the interpreter;
   // this returns 0 (== failure) in the case the class is already in memory
   // but does not have a dictionary, so we just raise a flag for better
   // diagnostic in the case the class is not found in the CINT ClassInfo table.
   Bool_t autoloaderr = kFALSE;
   if (!fromFile && gCling->AutoLoad(localname) != 1)
      autoloaderr = kTRUE;

   TClass *selCl = TClass::GetClass(localname);
   if (selCl) {
      // We have all we need.
      auto offset = selCl->GetBaseClassOffset(TSelector::Class());
      if (offset == -1) {
         // TSelector is not a based class.
         if (fromFile)
            ::Error("TSelector::GetSelector",
                    "The class %s in file %s does not derive from TSelector.", localname.Data(), filename);
         else if (autoloaderr)
            ::Error("TSelector::GetSelector", "class %s could not be loaded", filename);
         else
            ::Error("TSelector::GetSelector",
                    "class %s does not exist or does not derive from TSelector", filename);
         return 0;
      }
      char *result = (char*)selCl->New();
      // By adding offset, we support the case where TSelector is not the
      // "left-most" base class (i.e. offset != 0)
      return (TSelector*)(result+offset);

   } else {
      ClassInfo_t *cl = gCling->ClassInfo_Factory(localname);
      Bool_t ok = kFALSE;
      Bool_t nameFound = kFALSE;
      if (cl && gCling->ClassInfo_IsValid(cl)) {
         if (localname == gCling->ClassInfo_FullName(cl)) {
            nameFound = kTRUE;
            if (gCling->ClassInfo_IsBase(cl,"TSelector")) ok = kTRUE;
         }
      }
      if (!ok) {
         if (fromFile) {
            if (nameFound) {
               ::Error("TSelector::GetSelector",
                       "The class %s in file %s does not derive from TSelector.", localname.Data(), filename);
            } else {
               ::Error("TSelector::GetSelector",
                       "The file %s does not define a class named %s.", filename, localname.Data());
            }
         } else {
            if (autoloaderr)
               ::Error("TSelector::GetSelector", "class %s could not be loaded", filename);
            else
               ::Error("TSelector::GetSelector",
                       "class %s does not exist or does not derive from TSelector", filename);
         }
         gCling->ClassInfo_Delete(cl);
         return 0;
      }

      // we can now create an instance of the class
      TSelector *selector = (TSelector*)gCling->ClassInfo_New(cl);
      gCling->ClassInfo_Delete(cl);
      return selector;
   }
}

//______________________________________________________________________________
Bool_t TSelector::IsStandardDraw(const char *selec)
{
   // Find out if this is a standard selection used for Draw actions
   // (either TSelectorDraw, TProofDraw or deriving from them).

   // Make sure we have a name
   if (!selec) {
      ::Info("TSelector::IsStandardDraw",
             "selector name undefined - do nothing");
      return kFALSE;
   }

   Bool_t stdselec = kFALSE;
   if (!strchr(selec, '.')) {
      if (strstr(selec, "TSelectorDraw")) {
         stdselec = kTRUE;
      } else {
         TClass *cl = TClass::GetClass(selec);
         if (cl && (cl->InheritsFrom("TProofDraw") ||
                    cl->InheritsFrom("TSelectorDraw")))
            stdselec = kTRUE;
      }
   }

   // We are done
   return stdselec;
}

Bool_t TSelector::ProcessCut(Long64_t /*entry*/)
{
   //    This method is called before processing entry. It is the user's responsability to read
   //    the corresponding entry in memory (may be just a partial read).
   //    The function returns kTRUE if the entry must be processed,
   //    kFALSE otherwise. This method is obsolete, use Process().
   //
   // WARNING when a selector is used with a TChain:
   //    in the Process, ProcessCut, ProcessFill function, you must use
   //    the pointer to the current Tree to call GetEntry(entry).
   //    entry is always the local entry number in the current tree.
   //    Assuming that fChain is the pointer to the TChain being processed,
   //    use fChain->GetTree()->GetEntry(entry);

   return kTRUE;
}

void TSelector::ProcessFill(Long64_t /*entry*/)
{
   // This method is called for all selected entries. User fills histograms
   // in this function.  This method is obsolete, use Process().
   //
   // WARNING when a selector is used with a TChain:
   //    in the Process, ProcessCut, ProcessFill function, you must use
   //    the pointer to the current Tree to call GetEntry(entry).
   //    entry is always the local entry number in the current tree.
   //    Assuming that fChain is the pointer to the TChain being processed,
   //    use fChain->GetTree()->GetEntry(entry);
}

Bool_t TSelector::Process(Long64_t /*entry*/) {
   // The Process() function is called for each entry in the tree (or possibly
   // keyed object in the case of PROOF) to be processed. The entry argument
   // specifies which entry in the currently loaded tree is to be processed.
   // It can be passed to either t01::GetEntry() or TBranch::GetEntry()
   // to read either all or the required parts of the data. When processing
   // keyed objects with PROOF, the object is already loaded and is available
   // via the fObject pointer.
   //
   // This function should contain the "body" of the analysis. It can contain
   // simple or elaborate selection criteria, run algorithms on the data
   // of the event and typically fill histograms.
   //
   // The processing can be stopped by calling Abort().
   //
   // Use fStatus to set the return value of TTree::Process().
   //
   // The return value is currently not used.

   return kFALSE;
}

