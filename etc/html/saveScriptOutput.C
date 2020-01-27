#include "TString.h"
#include "TSystem.h"
#include "TGWindow.h"
#include "TGClient.h"
#include "TClass.h"
#include "THashList.h"
#include "TROOT.h"
#include "TInterpreter.h"
#include "TRootCanvas.h"
#include "TCanvas.h"
#include "TVirtualViewer3D.h"
#include "TEnv.h"
#include "TVirtualX.h"

int saveScriptOutput(const char* script, const char* outdir, Bool_t compiled)
{
   // Run script and save all windows to
   // outdir/script_0.png, outdir/script_1.png, ...

   enum EErrorCodes {
      kSuccess,
      kScriptDirNotFound,
      kCannotRunScript,
      kOutDirNotFound,
      kNumErrorCodes
   };

   TString pwd(gSystem->pwd());
   if (!gSystem->cd(gSystem->GetDirName(script)))
      return kScriptDirNotFound;
   Int_t err = 0;
   TString cmd(".x ");
   cmd += gSystem->BaseName(script);
   if (compiled)
      cmd += "+";
   if (!gROOT->IsBatch())
      gVirtualX->Sync(1);

   // save current interpreter context to avoid gROOT->Reset()
   // in the script to cause havoc by wiping everything away
   gInterpreter->SaveContext();
   gInterpreter->SaveGlobalsContext();

   gROOT->ProcessLine(cmd, &err);
   if (err != TInterpreter::kNoError)
      return kCannotRunScript;

   gSystem->cd(pwd);
   if (!gSystem->cd(outdir))
      return kOutDirNotFound;

   UInt_t nCanvases = 0;
   if (gClient) {
      TClass* clRootCanvas = TClass::GetClass("TRootCanvas");
      TClass* clGMainFrame = TClass::GetClass("TGMainFrame");
      TGWindow* win = 0;
      TIter iWin(gClient->GetListOfWindows());
      while((win = (TGWindow*)iWin())) {
         const TObject* winGetParent = win->GetParent();
         Bool_t winIsMapped = kFALSE;
         if (winGetParent == gClient->GetDefaultRoot())
            winIsMapped = win->IsMapped();
         if (winIsMapped && win->InheritsFrom(clGMainFrame)) {
            win->MapRaised();
            Bool_t isRootCanvas = win->InheritsFrom(clRootCanvas);
            Bool_t hasEditor = false;
            if (isRootCanvas) {
               hasEditor = ((TRootCanvas*)win)->HasEditor();
            }
            if (isRootCanvas && !hasEditor) {
               TVirtualPad* pad = ((TRootCanvas*)win)->Canvas();
               if (!pad->HasViewer3D() || pad->GetViewer3D()->InheritsFrom("TViewer3DPad")) {
                  pad->SaveAs(TString::Format("%s_%d.png", gSystem->BaseName(script), nCanvases++));
               }
            } else
               win->SaveAs(TString::Format("%s_%d.png", gSystem->BaseName(script), nCanvases++));
         }
      }
   } else {
      // no gClient
      TVirtualPad* pad = 0;
      TIter iCanvas(gROOT->GetListOfCanvases());
      while ((pad = (TVirtualPad*) iCanvas())) {
         pad->SaveAs(TString::Format("%s_%d.png", gSystem->BaseName(script), nCanvases++));
      }
   }
   if (!gROOT->IsBatch() && !gEnv->GetValue("X11.Sync", 0))
      gVirtualX->Sync(0);
   return kSuccess;
}
