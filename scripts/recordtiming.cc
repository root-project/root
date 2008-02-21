// void Branch(TTree *timing, const char *testdir) {
//    TBranch * b = timing->GetBranch(testdir);
//    if (b==0) {
//       b = timing->Branch(testdir,"duration/D",0);
//    }
//    return b;
// }

#include "TFile.h"
#include "TTree.h"
#include <stdio.h>
#include "TDatime.h"
#include "TEnv.h"
#include "TSystem.h"
#include "TBranchElement.h"
#include "TStreamerInfo.h"

class TTestResult {
public:
   TTestResult() : fRunId(0),fTestId(0), fDuration(0) {}
   TTestResult(UInt_t runid, UInt_t testid, const char *test, double duration, const char *model, Double_t mhz) : 
        fRunId(runid), 
        fTestId(testid), 
        fTestName(test),
        fDuration(duration),
        fHostModel(model),
        fHostMhz(mhz) {
      fHostName = gSystem->HostName();
   }
   virtual ~TTestResult() {}
   TDatime    fDate;     //||
   UInt_t     fRunId;
   UInt_t     fTestId;
   TString    fTestName;
   Double32_t fDuration;
   TString    fHostName;
   TString    fHostModel;
   Double32_t fHostMhz;

   ClassDef(TTestResult,4);
};

void recordtiming(const char *roottesthome, UInt_t runid, UInt_t testid, const char *testdir, double duration) 
{
   TString stestdir( testdir );
   stestdir.Remove(0,strlen(roottesthome));

   TString logfile( roottesthome );
   logfile.Append( '/' );
   logfile.Append( "roottesttiming.root" );
   
   // fprintf(stderr,"%s %g %s\n",stestdir.Data(), duration,logfile.Data() );
   TEnv env(Form("%s/roottest.arch",roottesthome));
   TString modelname = env.GetValue("modelname","");
   double mhz = env.GetValue("cpuMHz",0.00);
   
   TTestResult *res = new TTestResult(runid,testid,stestdir,duration,modelname,mhz);

   TFile *f = TFile::Open(logfile,"UPDATE");
   TTree *t; f->GetObject("timing",t);
   if (t!=0) {
      TBranchElement *b = (TBranchElement*)t->GetBranch("test");
      TStreamerInfo *info = b->GetInfo();
      if (info->GetClassVersion() < 3) {
        t->Write(Form("timing_v%d",info->GetClassVersion()));
        delete t;
        t = 0;
      }
   }
   if (t==0) {
      t = new TTree("timing","roottest timing");
      t->Branch("test",&res);
   }

   t->SetBranchAddress("test",&res);
   t->Fill();
   f->Write("",TObject::kOverwrite);
}

void recordtiming(const char *roottesthome,UInt_t runid, UInt_t testid, const char *testdir, const char *timingfile) 
{
   FILE *f = fopen(timingfile,"r");
   double duration;
   fscanf(f,"%lg",&duration);
   fclose(f);
   recordtiming(roottesthome,runid,testid,testdir,duration);
}
