#include "TH1.h"
#include "TFile.h"
#include "TBenchmark.h"
#include "TROOT.h"
#include "TClass.h"

TFile *Create(int ntop, int nsub, int nhist)
{
   TFile *f = new TFile("many.root","RECREATE");
   for(int i=0; i<ntop; ++i) {
      TString name( Form("top-%d",i) );
      f->mkdir( name );
      f->cd( name );

      if (nsub) {
         for(int j=0; j<nsub; ++j) {
            name = Form("sub-%d-%d",i,j);

            TDirectory *sav = gDirectory;

            gDirectory->mkdir( name );
            gDirectory->cd( name );

            for(int k=0;k<nhist;++k) {
               new TH1F(Form("hist-%d-%d-%d",i,j,k),"sample",1,0,1);
            }

            sav->cd();
         }
      } else {
         for(int k=0;k<nhist;++k) {
            new TH1F(Form("hist-%d-na-%d",i,k),"sample",1,0,1);
         }
      }
   }
   return f;
}

void cleanup(TDirectory *dir)
{
   // return;

   gROOT->GetListOfFiles()->Remove(dir);
   return;

   TIter nextkey( dir->GetList() );

   TObject *obj;
   while ( (obj = nextkey())) {
      if ( obj->IsA()->InheritsFrom( "TDirectory" ) ) {
         cleanup( (TDirectory*)obj );
      }
   }
   //gROOT->SetMustClean(kFALSE);
   dir->GetList()->Delete("fast");
   //delete dir;
   // dir->GetListOfKeys()->Delete("fast");
}


void test_directories(int ntop, int nsub, int nhist)
{
   fprintf(stdout,"Test top=%d sub=%d hist=%d\n",ntop,nsub,nhist);
   gBenchmark = new TBenchmark();
   gBenchmark->Start("directories");
   TFile *f = Create(ntop,nsub,nhist);
   fprintf(stdout,"Test Create: "); gBenchmark->Show("directories");gBenchmark->Start("directories");
   f->Write();
   fprintf(stdout,"Test Write : "); gBenchmark->Show("directories");gBenchmark->Start("directories");
   cleanup(f);
   fprintf(stdout,"Test Clean : "); gBenchmark->Show("directories");gBenchmark->Start("directories");
   f->Close();
   fprintf(stdout,"Test Close : "); gBenchmark->Show("directories");gBenchmark->Start("directories");
   delete f;
}

void directories(int ntop = 100, int nsub = 2, int nhist = 24)
{
//    for(int h=100; h<500; h += 50) {
//       test_directories(2,2,h);
//    }

   int step = ntop / 5;
   for(int t=1; t<ntop; t += step) {
      test_directories(t,nsub,nhist);
   }
}


