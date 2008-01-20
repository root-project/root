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

class TTestResult {
public:
   TTestResult() : fRunId(0),fTestId(0), fDuration(0) {}
   TTestResult(UInt_t runid, UInt_t testid, const char *test, double duration) : fRunId(runid), fTestId(testid), fTestName(test),fDuration(duration) {
      // set time stamp
   }
   TDatime    fDate;     //||
   UInt_t     fRunId;
   UInt_t     fTestId;
   TString    fTestName;
   Double32_t fDuration;
};

void recordtiming(const char *roottesthome, UInt_t runid, UInt_t testid, const char *testdir, double duration) 
{
   TString stestdir( testdir );
   stestdir.Remove(0,strlen(roottesthome));
   TString logfile( roottesthome );
   logfile.Append( '/' );
   logfile.Append( "roottesttiming.root" );
   
   // fprintf(stderr,"%s %g %s\n",stestdir.Data(), duration,logfile.Data() );

   TFile *f = TFile::Open(logfile,"UPDATE");
   TTree *t; f->GetObject("timing",t);
   TTestResult *res = new TTestResult(runid,testid,stestdir,duration);

   if (t==0) {
      t = new TTree("timing","roottest timing");
      t->Branch("test",&res);
   }

   t->SetBranchAddress("test",&res);
   t->Fill();
   f->Write();
}

void recordtiming(const char *roottesthome,UInt_t runid, UInt_t testid, const char *testdir, const char *timingfile) 
{
   FILE *f = fopen(timingfile,"r");
   double duration;
   fscanf(f,"%lg",&duration);
   fclose(f);
   recordtiming(roottesthome,runid,testid,testdir,duration);
}
