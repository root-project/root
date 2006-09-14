// @(#)root/test:$Name:  $:$Id: stressHepix.cxx,v 1.4 2006/09/13 06:53:14 brun Exp $
// Author: Rene Brun   12/09/2006

///////////////////////////////////////////////////////////////////////////////
//
//    R O O T   S T R E S S H E P I X  G L O B A L  B E N C H M A R K 
//    ===============================================================
//
// HEPiX-HEPNT is an organization comprised of UNIX and Windows support staff 
// in the High Energy Physics community. 
// One of the HEPIX activities is to gather knowledge about new hardware
// and software and to recommend common solutions (eg Scientific Linux)
//   see: http://wwwhepix.web.cern.ch/wwwhepix/
//
// This benchmark suite has been implemented following several requests
// from HEPIX members interested by a collection of benchmarks representative
// of typical applications.
//
// stressHepix is a single benchmark inclusing several standard ROOT benchmarks
// with a mixture of CPU intensive tests and I/O tests.
// The output of stressHepix is one single number (the ROOTMARK).
// A Pentium IV 2.8GHz running Linux SLC3 and gcc3.2.3 runs this benchmark
// with a reference at 800 ROOTMARKs.
// To build the executable for this benchmark, do
//   cd $ROOTSYS/test
//   make
//
// The default configuration of ROOT is enough.
// The ouput of this benchmark looks like:
//
/// stressHepix
///
///
///Starting stressHepix benchmark (details will be in stressHepix.log)
///Takes 558 seconds on a  500 rootmarks machine (IBM Thinkpad centrino 1.4GHz VC++7.1)
///Takes 368 seconds on a  800 rootmarks reference machine (P IV 2.8 GHz, SLC3 gcc3.2.3)
///Takes 294 seconds on a  924 rootmarks machine (MacBookPro 1.8GHz gcc4.0.1)
///Takes 253 seconds on a 1056 rootmarks machine (MacBookPro 1.8GHz icc9.1)
///Takes 218 seconds on a 1550 rootmarks machine (AMD64/280, FC5 gcc4.1)
///Takes 149 seconds on a 1828 rootmarks machine (MacBookPro 3.0GHz icc9.1)
///
///Running : stressFit Minuit  2000, (takes 11 seconds on the ref machine)
///Running : stressLinear, (takes 26 seconds on the ref machine)
///Running : stressGeometry, (takes 77 seconds on the ref machine)
///Running : stressSpectrum 1000, (takes 116 seconds on the ref machine)
///Running : stress -b 3000, (takes 138 seconds on the ref machine)
///
///
///****************************************************************************
///*                                                                          *
///*               S T R E S S   H E P I X  S U M M A R Y                     *
///*                                                                          *
///*       ROOTMARKS = 789.3   *  Root5.13/03   20060830/1441
///*                                                                          *
///*  Real Time =  401.1 seconds, CpuTime =  281.8 seconds
///*  Linux pcbrun 2.4.21-47.EL.cernsmp #1 SMP Mon Jul 24 15:33:5
///****************************************************************************
//
// If you run this benchmark on a new platform, please report the results
// at rootdev@root.cern.ch. Send the output shown above and also the
// log file stressHepix.log that contains more details about the individual
// tests. Your results will be shown at http://root.cern.ch/root/Benchmark.html
//
///////////////////////////////////////////////////////////////////////////////
    
#include <TROOT.h>
#include "TApplication.h"
#include <TSystem.h>
#include <TStopwatch.h>

void runTest(const char *atest, int estimate) {
   printf("Running : %s, (takes %d seconds on the ref machine)\n",atest,estimate);
   gSystem->Exec(Form("%s >>stressHepix.log",atest));
}

int main(int argc, char **argv)
{
   TApplication theApp("App", &argc, argv);
   printf("\n\nStarting stressHepix benchmark (details will be in stressHepix.log)\n");
   printf("Takes %d seconds on a  500 rootmarks machine (IBM Thinkpad centrino 1.4GHz VC++7.1)\n",56+41+70+209+182);
   printf("Takes %d seconds on a  800 rootmarks reference machine (P IV 2.8 GHz, SLC3 gcc3.2.3)\n",11+26+77+116+138);
   printf("Takes %d seconds on a  924 rootmarks machine (MacBookPro 1.8GHz gcc4.0.1)\n",10+31+47+80+126);
   printf("Takes %d seconds on a 1056 rootmarks machine (MacBookPro 1.8GHz icc9.1)\n",8+26+38+75+106);
   printf("Takes %d seconds on a 1550 rootmarks machine (AMD64/280, FC5 gcc4.1)\n",6+18+50+43+101);
   printf("Takes %d seconds on a 1828 rootmarks machine (MacBookPro 3.0GHz icc9.1)\n\n",149);
   if (gSystem->AccessPathName("atlas.root")) {
      printf("\nPreparing geometry files from http://root.cern.ch\n\n");
      gSystem->Exec("stressGeometry >stressHepix.log");
   }
   TStopwatch timer;
   timer.Start();
   gSystem->Exec("echo stressHepix > stressHepix.log");
   runTest("stressFit Minuit  2000",11);
   runTest("stressLinear",26);
   runTest("stressGeometry",77);
   runTest("stressSpectrum 1000",116);
   runTest("stress -b 3000",138);
   timer.Stop();
   Double_t rt = timer.RealTime();
   //scan log file to accumulate the individual Cpu Times
   FILE *fp = fopen("stressHepix.log","r");
   char line[180];
   float cput;
   Double_t ct = 0;
   {
      while (fgets(line,180,fp)) {
      char *cpu = strstr(line,"Cpu Time =");
      if (cpu) {sscanf(cpu+10,"%g",&cput); ct += cput;}
   }
   }
   fclose(fp);
   Double_t reftime = 278.04; //pcbrun compiled and 368 seconds real time
   const Double_t rootmarks = 800*reftime/ct;
   
   //Print table with results
   Bool_t UNIX = strcmp(gSystem->GetName(), "Unix") == 0;
   printf("\n\n");
   printf("****************************************************************************\n");
   printf("*                                                                          *\n");
   printf("*               S T R E S S   H E P I X  S U M M A R Y                     *\n");
   printf("*                                                                          *\n");
   printf("*       ROOTMARKS =%6.1f   *  Root%-8s  %d/%d\n",rootmarks,gROOT->GetVersion(),
         gROOT->GetVersionDate(),gROOT->GetVersionTime());
   printf("*                                                                          *\n");
   printf("*  Real Time = %6.1f seconds, CpuTime = %6.1f seconds\n",rt,ct);
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
   printf("****************************************************************************\n");
}    
   
