{
//  Prints a summary of all ROOT benchmarks (must be run before)
//  The ROOTMARK number printed is by reference to a Pentium III 650 Mhz
//  (DELL Inspiron 7500 with 256 MBytes memory and 18 GBytes IDE disk)
//  taken by definition as 200 ROOTMARKS in batch mode in executing
//     root -b -q benchmarks.C
//

   Float_t rtall   = 0;
   Float_t cpall   = 0;
   Float_t norm    = 200;  //obtained rootmarks on Dell Inspiron 600 Mhz
   Float_t rtmark,cpmark;
   Bool_t batch = gROOT->IsBatch();

   printf("---------------ROOT %s benchmarks summary--------------------\n",gROOT->GetVersion());
   gBenchmark->Summary(rtall,cpall);
   printf("\n---------------ROOT %s benchmarks summary (in ROOTMARKS)-----\n",gROOT->GetVersion());
   printf("   For comparison, a Pentium III 650Mhz is benchmarked at 200 ROOTMARKS\n");
   Float_t hsimple_rt = gBenchmark->GetRealTime("hsimple");
   Float_t hsimple_ct = gBenchmark->GetCpuTime("hsimple");
   if (hsimple_rt > 0) {
      if (batch) {
         rtmark = norm*(0.80/hsimple_rt);
         cpmark = norm*(0.81/hsimple_ct);
      } else {
         rtmark = norm*(2.01/hsimple_rt);
         cpmark = norm*(1.25/hsimple_ct);
      }
      printf("hsimple     = %7.2f RealMARKS,  = %7.2f CpuMARKS\n",rtmark,cpmark);
   }

   Float_t hsum_rt = gBenchmark->GetRealTime("hsum");
   Float_t hsum_ct = gBenchmark->GetCpuTime("hsum");
   if (hsum_rt > 0) {
      if (batch) {
         rtmark = norm*(0.44/hsum_rt);
         cpmark = norm*(0.40/hsum_ct);
      } else {
         rtmark = norm*(1.25/hsum_rt);
         cpmark = norm*(0.75/hsum_ct);
      }
      printf("hsum        = %7.2f RealMARKS,  = %7.2f CpuMARKS\n",rtmark,cpmark);
   }

   Float_t fillrandom_rt = gBenchmark->GetRealTime("fillrandom");
   Float_t fillrandom_ct = gBenchmark->GetCpuTime("fillrandom");
   if (fillrandom_rt > 0) {
      if (batch) {
         rtmark = norm*(0.11/fillrandom_rt);
         cpmark = norm*(0.04/fillrandom_ct);
      } else {
         rtmark = norm*(0.21/fillrandom_rt);
         cpmark = norm*(0.06/fillrandom_ct);
      }
      printf("fillrandom  = %7.2f RealMARKS,  = %7.2f CpuMARKS\n",rtmark,cpmark);
   }

   Float_t fit1_rt = gBenchmark->GetRealTime("fit1");
   Float_t fit1_ct = gBenchmark->GetCpuTime("fit1");
   if (fit1_rt > 0) {
      if (batch) {
         rtmark = norm*(0.13/fit1_rt);
         cpmark = norm*(0.07/fit1_ct);
      } else {
         rtmark = norm*(0.20/fit1_rt);
         cpmark = norm*(0.13/fit1_ct);
      }
      printf("fit1        = %7.2f RealMARKS,  = %7.2f CpuMARKS\n",rtmark,cpmark);
   }

   Float_t tornado_rt = gBenchmark->GetRealTime("tornado");
   Float_t tornado_ct = gBenchmark->GetCpuTime("tornado");
   if (tornado_rt > 0) {
      if (batch) {
         rtmark = norm*(0.14/tornado_rt);
         cpmark = norm*(0.11/tornado_ct);
      } else {
         rtmark = norm*(0.20/tornado_rt);
         cpmark = norm*(0.12/tornado_ct);
      }
      printf("tornado     = %7.2f RealMARKS,  = %7.2f CpuMARKS\n",rtmark,cpmark);
   }

   Float_t na49_rt = gBenchmark->GetRealTime("na49");
   Float_t na49_ct = gBenchmark->GetCpuTime("na49");
   if (na49_rt > 0) {
      rtmark = norm*(3.30/na49_rt);
      cpmark = norm*(3.28/na49_ct);
      printf("na49        = %7.2f RealMARKS,  = %7.2f CpuMARKS\n",rtmark,cpmark);
   }

   Float_t geometry_rt = gBenchmark->GetRealTime("geometry");
   Float_t geometry_ct = gBenchmark->GetCpuTime("geometry");
   if (geometry_rt > 0) {
      rtmark = norm*(0.63/geometry_rt);
      cpmark = norm*(0.61/geometry_ct);
      printf("geometry    = %7.2f RealMARKS,  = %7.2f CpuMARKS\n",rtmark,cpmark);
   }

   Float_t na49view_rt = gBenchmark->GetRealTime("na49view");
   Float_t na49view_ct = gBenchmark->GetCpuTime("na49view");
   if (na49view_rt > 0) {
      if (batch) {
         rtmark = norm*(0.10/na49view_rt);
         cpmark = norm*(0.10/na49view_ct);
      } else {
         rtmark = norm*(0.73/na49view_rt);
         cpmark = norm*(0.18/na49view_ct);
      }
      printf("na49view    = %7.2f RealMARKS,  = %7.2f CpuMARKS\n",rtmark,cpmark);
   }

   Float_t ntuple1_rt = gBenchmark->GetRealTime("ntuple1");
   Float_t ntuple1_ct = gBenchmark->GetCpuTime("ntuple1");
   if (ntuple1_rt > 0) {
      if (batch) {
         rtmark = norm*(0.85/ntuple1_rt);
         cpmark = norm*(0.78/ntuple1_ct);
      } else {
         rtmark = norm*(1.18/ntuple1_rt);
         cpmark = norm*(0.88/ntuple1_ct);
      }
      printf("ntuple1     = %7.2f RealMARKS,  = %7.2f CpuMARKS\n",rtmark,cpmark);
   }

   if (rtall) {
      Float_t rtdell, cpdell;
      if (batch) {
         rtdell    = 6.46;
         cpdell    = 6.14;
      } else {
         rtdell    = 9.83;
         cpdell    = 7.43;
      }
      Float_t rootmarks = norm*(rtdell+cpdell)/(rtall+cpall);
      printf("\n");
      printf("****************************************************\n");
      printf("* Your machine is estimated at %7.2f ROOTMARKS   *\n",rootmarks);
      printf("****************************************************\n");
   } else {
      printf(" You must run the ROOT benchmarks before executing this command\n");
   }
}
