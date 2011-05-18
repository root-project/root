{
//  Prints a summary of all ROOT benchmarks (must be run before)
//  The ROOTMARK number printed is by reference to a Pentium IV 2.4 Ghz
//  (with 512 MBytes memory and 120 GBytes IDE disk)
//  taken by definition as 600 ROOTMARKS in batch mode in executing
//     root -b -q benchmarks.C
//

   Float_t rtall   = 0;
   Float_t cpall   = 0;
   Float_t norm    = 600;
   Float_t rtmark,cpmark;
   Bool_t batch = gROOT->IsBatch();

   printf("---------------ROOT %s benchmarks summary--------------------\n",gROOT->GetVersion());
   gBenchmark->Summary(rtall,cpall);
   printf("\n---------------ROOT %s benchmarks summary (in ROOTMARKS)-----\n",gROOT->GetVersion());
   printf("   For comparison, a Pentium IV 2.4Ghz is benchmarked at 600 ROOTMARKS\n");
   Float_t hsimple_rt = gBenchmark->GetRealTime("hsimple");
   Float_t hsimple_ct = gBenchmark->GetCpuTime("hsimple");
   if (hsimple_rt > 0) {
      if (batch) {
         rtmark = norm*(0.29/hsimple_rt);
         cpmark = norm*(0.28/hsimple_ct);
      } else {
         rtmark = norm*(0.99/hsimple_rt);
         cpmark = norm*(0.43/hsimple_ct);
      }
      printf("hsimple     = %7.2f RealMARKS,  = %7.2f CpuMARKS\n",rtmark,cpmark);
   }

   Float_t hsum_rt = gBenchmark->GetRealTime("hsum");
   Float_t hsum_ct = gBenchmark->GetCpuTime("hsum");
   if (hsum_rt > 0) {
      if (batch) {
         rtmark = norm*(0.16/hsum_rt);
         cpmark = norm*(0.15/hsum_ct);
      } else {
         rtmark = norm*(0.99/hsum_rt);
         cpmark = norm*(0.24/hsum_ct);
      }
      printf("hsum        = %7.2f RealMARKS,  = %7.2f CpuMARKS\n",rtmark,cpmark);
   }

   Float_t fillrandom_rt = gBenchmark->GetRealTime("fillrandom");
   Float_t fillrandom_ct = gBenchmark->GetCpuTime("fillrandom");
   if (fillrandom_rt > 0) {
      if (batch) {
         rtmark = norm*(0.02/fillrandom_rt);
         cpmark = norm*(0.01/fillrandom_ct);
      } else {
         rtmark = norm*(0.48/fillrandom_rt);
         cpmark = norm*(0.04/fillrandom_ct);
      }
      printf("fillrandom  = %7.2f RealMARKS,  = %7.2f CpuMARKS\n",rtmark,cpmark);
   }

   Float_t fit1_rt = gBenchmark->GetRealTime("fit1");
   Float_t fit1_ct = gBenchmark->GetCpuTime("fit1");
   if (fit1_rt > 0) {
      if (batch) {
         rtmark = norm*(0.04/fit1_rt);
         cpmark = norm*(0.03/fit1_ct);
      } else {
         rtmark = norm*(0.13/fit1_rt);
         cpmark = norm*(0.03/fit1_ct);
      }
      printf("fit1        = %7.2f RealMARKS,  = %7.2f CpuMARKS\n",rtmark,cpmark);
   }

   Float_t tornado_rt = gBenchmark->GetRealTime("tornado");
   Float_t tornado_ct = gBenchmark->GetCpuTime("tornado");
   if (tornado_rt > 0) {
      if (batch) {
         rtmark = norm*(0.05/tornado_rt);
         cpmark = norm*(0.04/tornado_ct);
      } else {
         rtmark = norm*(0.11/tornado_rt);
         cpmark = norm*(0.03/tornado_ct);
      }
      printf("tornado     = %7.2f RealMARKS,  = %7.2f CpuMARKS\n",rtmark,cpmark);
   }

   Float_t na49_rt = gBenchmark->GetRealTime("na49");
   Float_t na49_ct = gBenchmark->GetCpuTime("na49");
   if (na49_rt > 0) {
      rtmark = norm*(1.39/na49_rt);
      cpmark = norm*(1.39/na49_ct);
      printf("na49        = %7.2f RealMARKS,  = %7.2f CpuMARKS\n",rtmark,cpmark);
   }

   Float_t geometry_rt = gBenchmark->GetRealTime("geometry");
   Float_t geometry_ct = gBenchmark->GetCpuTime("geometry");
   if (geometry_rt > 0) {
      rtmark = norm*(0.19/geometry_rt);
      cpmark = norm*(0.18/geometry_ct);
      printf("geometry    = %7.2f RealMARKS,  = %7.2f CpuMARKS\n",rtmark,cpmark);
   }

   Float_t na49view_rt = gBenchmark->GetRealTime("na49view");
   Float_t na49view_ct = gBenchmark->GetCpuTime("na49view");
   if (na49view_rt > 0) {
      if (batch) {
         rtmark = norm*(0.03/na49view_rt);
         cpmark = norm*(0.03/na49view_ct);
      } else {
         rtmark = norm*(0.33/na49view_rt);
         cpmark = norm*(0.05/na49view_ct);
      }
      printf("na49view    = %7.2f RealMARKS,  = %7.2f CpuMARKS\n",rtmark,cpmark);
   }

   Float_t ntuple1_rt = gBenchmark->GetRealTime("ntuple1");
   Float_t ntuple1_ct = gBenchmark->GetCpuTime("ntuple1");
   if (ntuple1_rt > 0) {
      if (batch) {
         rtmark = norm*(0.29/ntuple1_rt);
         cpmark = norm*(0.27/ntuple1_ct);
      } else {
         rtmark = norm*(1.79/ntuple1_rt);
         cpmark = norm*(0.28/ntuple1_ct);
      }
      printf("ntuple1     = %7.2f RealMARKS,  = %7.2f CpuMARKS\n",rtmark,cpmark);
   }

   if (rtall) {
      Float_t rtbrun, cpbrun;
      if (batch) {
         rtbrun    = 3.45;
         cpbrun    = 3.24;
      } else {
         rtbrun    = 5.79;
         cpbrun    = 4.08;
      }
      Float_t rootmarks = norm*(rtbrun+cpbrun)/(rtall+cpall);
      printf("\n");
      printf("****************************************************\n");
      printf("* Your machine is estimated at %7.2f ROOTMARKS   *\n",rootmarks);
      printf("****************************************************\n");
   } else {
      printf(" You must run the ROOT benchmarks before executing this command\n");
   }
}
