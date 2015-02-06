#include "TH1.h"
#include "TH2.h"
#include "TRandom3.h"
#include "TSystem.h"
#include "THttpServer.h"

Bool_t RunControl = kTRUE;

void httpcontrol()
{
//  This program demonstrates simple control via http server
//  Two histogram are filled within endless loop.
//  Via published commands one can enable/disable histograms filling
//  Or one could clear registered histograms
//  After macro started, open in browser with url
//      http://localhost:8080?monitoring=1000

   // create histograms
   TH1D *hpx = new TH1D("hpx","This is the px distribution",100,-4,4);
   hpx->SetFillColor(48);
   hpx->SetDirectory(0);
   TH2D *hpxpy = new TH2D("hpxpy","py vs px",40,-4,4,40,-4,4);
   hpxpy->SetDirectory(0);

   // start http server
   THttpServer* serv = new THttpServer("http:8080");

   // register histograms
   serv->Register("/", hpx);
   serv->Register("/", hpxpy);

   // register simple commands
   serv->RegisterCommand("/Start", "RunControl=kTRUE;", "button;/rootsys/icons/ed_execute.png");
   serv->RegisterCommand("/Stop",  "RunControl=kFALSE;", "button;/rootsys/icons/ed_interrupt.png");

   // register commands, invoking object methods
   serv->RegisterCommand("/ResetHPX","/hpx/->Reset()", "button;/rootsys/icons/ed_delete.png");
   serv->RegisterCommand("/ResetHPXPY","/hpxpy/->Reset()", "button;/rootsys/icons/ed_delete.png");

   // one could hide commands and let them appear only as buttons
   serv->Hide("/Start");
   serv->Hide("/Stop");

   // Fill histograms randomly
   TRandom3 random;
   Float_t px, py;
   const Long_t kUPDATE = 1000;
   Long_t cnt = 0;
   while (kTRUE) {
      if (RunControl) {
         random.Rannor(px,py);
         hpx->Fill(px);
         hpxpy->Fill(px,py);
         cnt++;
      } else {
         gSystem->Sleep(10); // sleep minimal time
      }

      if ((cnt % kUPDATE==0) || !RunControl) {
         // IMPORTANT: one should regularly call ProcessEvents
         // to let http server process requests
         if (gSystem->ProcessEvents()) break;
      }
   }
}
