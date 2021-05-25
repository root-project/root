/// \file
/// \ingroup tutorial_gui
/// This utility macro executes the macro "macro" given as first argument and save a capture in a png file.
/// This macro is used by stressGUI to execute and compare the output of the GUI tutorials.
///
/// \macro_code
///
/// \author Bertrand Bellenot

#include "TSystem.h"
#include "TString.h"
#include "TGClient.h"
#include "TGWindow.h"
#include "TClass.h"
#include "THashList.h"
#include "TROOT.h"
#include "TInterpreter.h"
#include "TEnv.h"
#include "TVirtualX.h"
#include "TImage.h"

//______________________________________________________________________________
Int_t exec_macro(const char *macro, Bool_t comp = kFALSE, Bool_t save = kTRUE)
{

   enum EErrorCodes {
      kSuccess,
      kScriptDirNotFound,
      kCannotRunScript,
      kNumErrorCodes
   };

   if (gROOT->IsBatch() || !(gClient))
      return kCannotRunScript;
   TString pwd(gSystem->pwd());
   if (!gSystem->cd(gSystem->GetDirName(macro)))
      return kScriptDirNotFound;
   Int_t err = 0;
   TString cmd(".x ");
   cmd += gSystem->BaseName(macro);
   if (comp) cmd += "+";
   gVirtualX->Sync(1);
   gROOT->ProcessLine(cmd, &err);
   if (err != TInterpreter::kNoError)
      return kCannotRunScript;
   gSystem->cd(pwd);

   UInt_t nMainFrames = 0;
   TClass* clGMainFrame = TClass::GetClass("TGMainFrame");
   TGWindow* win = 0;
   TIter iWin(gClient->GetListOfWindows());
   while ((win = (TGWindow*)iWin())) {
      const TObject* winGetParent = win->GetParent();
      Bool_t winIsMapped = kFALSE;
      if (winGetParent == gClient->GetDefaultRoot())
         winIsMapped = kTRUE;//win->IsMapped();
      if (winIsMapped && win->InheritsFrom(clGMainFrame)) {
         win->MapRaised();
         if (save) {
            TString outfile = gSystem->BaseName(macro);
            outfile.ReplaceAll(".C", TString::Format("_%d.png",
                               ++nMainFrames));
            TImage *img = TImage::Create();
            win->RaiseWindow();
            img->FromWindow(win->GetId());
            img->WriteImage(outfile.Data());
            delete img;
         }
      }
   }
   if (!gEnv->GetValue("X11.Sync", 0))
      gVirtualX->Sync(0);
   return kSuccess;
}
