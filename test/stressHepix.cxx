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
   printf("\n\nStarting stressHepix benchmark\n");
   printf("Takes %d seconds on a  500 rootmarks machine (IBM Thinkpad centrino 1.4GHz VC++7.1)\n\n",56+41+70+209+182);
   printf("Takes %d seconds on a  800 rootmarks reference machine (P IV 2.8 GHz, SLC3 gcc3.2.3)\n",11+26+77+116+138);
   printf("Takes %d seconds on a  924 rootmarks machine (MacBookPro 1.8GHz gcc4.0.1)\n",10+31+47+80+126);
   printf("Takes %d seconds on a 1056 rootmarks machine (MacBookPro 1.8GHz icc9.1)\n",8+26+38+75+106);
   printf("Takes %d seconds on a 1550 rootmarks machine (AMD64/280, FC5 gcc4.1)\n\n",6+18+50+43+101);
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
   
