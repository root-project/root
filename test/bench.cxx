// this test program compares the I/O performance obtained with 
// STL vector of objects or pointers to objects versus the native
// Root collection class TClonesArray.
// Trees in compression and non compression mode are created for each
// of the following cases:
//  -vector<THit>
//  -vector<THit*>
//  -TClonesArray(TObjHit) in no split mode
//  -TClonesArray(TObjHit) in split mode
// where:
//  THit is a class not derived from TObject
//  TObjHit derives from TObject and THit
//
// The test prints a summary table comparing performances for all above cases
// (CPU, file size, compression factors).
// Reference numbers on a Pentium III 650 Mhz machine are given as reference.
//      Author:  Rene Brun
   
#include "TROOT.h"
#include "TClonesArray.h"
#include "TStopwatch.h"
#include "TFile.h"
#include "TTree.h"
#include "TSystem.h"

#include "TBench.h"

int main(int argc, char** argv) {
  TROOT root("bench","Benchmarking STL vector against TClonesArray");

  int nhits       = 1000;
  int nevents     = 400;
  Float_t cx;
  
  //testing STL vector of THit
  TStopwatch timer;
  timer.Start();
  TSTLhit *STLhit = new TSTLhit(nhits);
  STLhit->MakeTree(0,nevents,0,0,cx);
  timer.Stop();
  Double_t rt1 = timer.RealTime();
  Double_t cp1 = timer.CpuTime();
  printf("1  STLhit :  RT=%6.2f s  Cpu=%6.2f s\n",rt1,cp1);
  timer.Start(kTRUE);
  Int_t nbytes1 = STLhit->MakeTree(1,nevents,0,0,cx);
  timer.Stop();
  Double_t rt2w = timer.RealTime();
  Double_t cp2w = timer.CpuTime();
  printf("2  STLhitw:  RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt2w-rt1,cp2w-cp1,nbytes1,cx);
  timer.Start(kTRUE);
  STLhit->ReadTree();
  timer.Stop();
  Double_t rt2r = timer.RealTime();
  Double_t cp2r = timer.CpuTime();
  printf("3  STLhitr:  RT=%6.2f s  Cpu=%6.2f s\n",rt2r,cp2r);
  timer.Start(kTRUE);
  Float_t cx3;
  Int_t nbytes3 = STLhit->MakeTree(1,nevents,1,0,cx3);
  timer.Stop();
  Double_t rt3w = timer.RealTime();
  Double_t cp3w = timer.CpuTime();
  printf("4  STLhitw:  RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt3w-rt1,cp3w-cp1,nbytes3,cx3);
  timer.Start(kTRUE);
  STLhit->ReadTree();
  timer.Stop();
  Double_t rt3r = timer.RealTime();
  Double_t cp3r = timer.CpuTime();
  printf("5  STLhitr:  RT=%6.2f s  Cpu=%6.2f s\n",rt3r,cp3r);
  
  //testing STL vector of pointers to THit
  timer.Start();
  TSTLhitStar *STLhitStar = new TSTLhitStar(nhits);
  STLhitStar->MakeTree(0,nevents,0,0,cx);
  timer.Stop();
  Double_t rt4 = timer.RealTime();
  Double_t cp4 = timer.CpuTime();
  printf("6  STLhit* : RT=%6.2f s  Cpu=%6.2f s\n",rt4,cp4);
  timer.Start(kTRUE);
  Int_t nbytes5 = STLhitStar->MakeTree(1,nevents,0,1,cx);
  timer.Stop();
  Double_t rt5w = timer.RealTime();
  Double_t cp5w = timer.CpuTime();
  printf("7  STLhit*w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt5w-rt4,cp5w-cp4,nbytes5,cx);
  timer.Start(kTRUE);
  STLhitStar->ReadTree();
  timer.Stop();
  Double_t rt5r = timer.RealTime();
  Double_t cp5r = timer.CpuTime();
  printf("8  STLhit*r: RT=%6.2f s  Cpu=%6.2f s\n",rt5r,cp5r);
  timer.Start(kTRUE);
  Float_t cx6;
  Int_t nbytes6 = STLhitStar->MakeTree(1,nevents,1,1,cx6);
  timer.Stop();
  Double_t rt6w = timer.RealTime();
  Double_t cp6w = timer.CpuTime();
  printf("9  STLhit*w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt6w-rt4,cp6w-cp4,nbytes6,cx6);
  timer.Start(kTRUE);
  STLhitStar->ReadTree();
  timer.Stop();
  Double_t rt6r = timer.RealTime();
  Double_t cp6r = timer.CpuTime();
  printf("10 STLhit*r: RT=%6.2f s  Cpu=%6.2f s\n",rt6r,cp6r);
  
  //testing TClonesArray of TObjHit deriving from THit
  timer.Start();
  TCloneshit *Cloneshit = new TCloneshit(nhits);
  Cloneshit->MakeTree(0,nevents,0,0,cx);
  timer.Stop();
  Double_t rt7 = timer.RealTime();
  Double_t cp7 = timer.CpuTime();
  printf("11 Clones1 : RT=%6.2f s  Cpu=%6.2f s\n",rt7,cp7);
  timer.Start(kTRUE);
  Int_t nbytes8 = Cloneshit->MakeTree(1,nevents,0,1,cx);
  timer.Stop();
  Double_t rt8w = timer.RealTime();
  Double_t cp8w = timer.CpuTime();
  printf("12 Clones1w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt8w-rt7,cp8w-cp7,nbytes8,cx);
  timer.Start(kTRUE);
  Cloneshit->ReadTree();
  timer.Stop();
  Double_t rt8r = timer.RealTime();
  Double_t cp8r = timer.CpuTime();
  printf("13 Clones1r: RT=%6.2f s  Cpu=%6.2f s\n",rt8r,cp8r);
  timer.Start(kTRUE);
  Float_t cx9;
  Int_t nbytes9 = Cloneshit->MakeTree(1,nevents,1,1,cx9);
  timer.Stop();
  Double_t rt9w = timer.RealTime();
  Double_t cp9w = timer.CpuTime();
  printf("14 Clones1w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt9w-rt7,cp9w-cp7,nbytes9,cx9);
  timer.Start(kTRUE);
  Cloneshit->ReadTree();
  timer.Stop();
  Double_t rt9r = timer.RealTime();
  Double_t cp9r = timer.CpuTime();
  printf("15 Clones1r: RT=%6.2f s  Cpu=%6.2f s\n",rt9r,cp9r);
  timer.Start(kTRUE);
  Int_t nbytes10 = Cloneshit->MakeTree(1,nevents,0,2,cx);
  timer.Stop();
  Double_t rt10w = timer.RealTime();
  Double_t cp10w = timer.CpuTime();
  printf("16 Clones2w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt10w-rt7,cp10w-cp7,nbytes10,cx);
  timer.Start(kTRUE);
  Cloneshit->ReadTree();
  timer.Stop();
  Double_t rt10r = timer.RealTime();
  Double_t cp10r = timer.CpuTime();
  printf("17 Clones2r: RT=%6.2f s  Cpu=%6.2f s\n",rt10r,cp10r);
  timer.Start(kTRUE);
  Float_t cx11;
  Int_t nbytes11 = Cloneshit->MakeTree(1,nevents,1,2,cx11);
  timer.Stop();
  Double_t rt11w = timer.RealTime();
  Double_t cp11w = timer.CpuTime();
  printf("18 Clones2w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt11w-rt7,cp11w-cp7,nbytes11,cx11);
  timer.Start(kTRUE);
  Cloneshit->ReadTree();
  timer.Stop();
  Double_t rt11r = timer.RealTime();
  Double_t cp11r = timer.CpuTime();
  printf("19 Clones2r: RT=%6.2f s  Cpu=%6.2f s\n",rt11r,cp11r);
  
  //delete temp file used for the benchmark
  gSystem->Exec("rm -f /tmp/bench.root");
  
  //print all results
  printf("\n");
  printf("******************************************************************************\n");
  printf("*  Comparing STL vector with TClonesArray: Root%-8s  %8d/%4d       *\n",gROOT->GetVersion(),gROOT->GetVersionDate(),gROOT->GetVersionTime());
  Bool_t UNIX = strcmp(gSystem->GetName(), "Unix") == 0;
  if (UNIX) {
     FILE *fp = gSystem->OpenPipe("uname -a", "r");
     char line[60];
     fgets(line,60,fp); line[59] = 0;
     printf("*  %s\n",line);
     gSystem->ClosePipe(fp);
  } else {
     const char *os = gSystem->Getenv("OS");
     if (!os) printf("*  Windows 95\n");
     else     printf("*  %s %s \n",os,gSystem->Getenv("PROCESSOR_IDENTIFIER"));
  }
  printf("*     Reference machine pcnotebrun.cern.ch  RedHat Linux 6.1                 *\n");
  printf("*         (Pentium III 650 Mhz 256 Mbytes RAM, IDE disk)                     *\n");
  printf("*           (send your results to rootdev@root.cern.ch)                      *\n");
  printf("******************************************************************************\n");
  printf("* Time to fill the structures (seconds)   Reference      cx      Reference   *\n");
  printf("******************************************************************************\n");
  printf("* vector<THit>                  %6.2f        1.91     %5.2f        4.57     *\n",cp1,cx3);
  printf("* vector<THit*>                 %6.2f        1.86     %5.2f        4.57     *\n",cp4,cx6);
  printf("* TClonesArray(TObjHit)         %6.2f        1.62     %5.2f        6.52     *\n",cp7,cx9);
  printf("* TClonesArray(TObjHit) split   %6.2f        1.62     %5.2f        6.70     *\n",cp7,cx11);
  printf("******************************************************************************\n");
  printf("* Size of file in bytes         comp 0    Reference    comp 1    Reference   *\n");
  printf("******************************************************************************\n");
  printf("* vector<THit>                  %8d   42053186   %8d    9211657    *\n",nbytes1,nbytes3);
  printf("* vector<THit*>                 %8d   42082028   %8d    9215931    *\n",nbytes5,nbytes6);
  printf("* TClonesArray(TObjHit)         %8d   39691759   %8d    6090891    *\n",nbytes8,nbytes9);
  printf("* TClonesArray(TObjHit) split   %8d   39782633   %8d    5937761    *\n",nbytes10,nbytes11);
  printf("******************************************************************************\n");
  printf("* Time to write in seconds      comp 0    Reference    comp 1    Reference   *\n");
  printf("******************************************************************************\n");
  printf("* vector<THit>                  %6.2f        1.74    %6.2f        9.58     *\n",cp2w-cp1, cp3w-cp1);
  printf("* vector<THit*>                 %6.2f        1.80    %6.2f        9.62     *\n",cp5w-cp1, cp6w-cp1);
  printf("* TClonesArray(TObjHit)         %6.2f        1.60    %6.2f        7.32     *\n",cp8w-cp1, cp9w-cp1);
  printf("* TClonesArray(TObjHit) split   %6.2f        1.51    %6.2f        6.18     *\n",cp10w-cp1,cp11w-cp1);
  printf("******************************************************************************\n");
  printf("* Time to read in seconds       comp 0    Reference    comp 1    Reference   *\n");
  printf("******************************************************************************\n");
  printf("* vector<THit>                  %6.2f        2.29    %6.2f        3.67     *\n",cp2r,cp3r);
  printf("* vector<THit*>                 %6.2f        2.10    %6.2f        3.27     *\n",cp5r,cp6r);
  printf("* TClonesArray(TObjHit)         %6.2f        1.53    %6.2f        2.14     *\n",cp8r,cp9r);
  printf("* TClonesArray(TObjHit) split   %6.2f        1.35    %6.2f        1.94     *\n",cp10r,cp11r);
  printf("******************************************************************************\n");
}
