{
//  Prints a summary of all ROOT benchmarks (must be run before)
//  The ROOTMARK number printed is by reference to an HP735/99
//  taken by definition as 27 ROOTMARKS in interactive mode
//  and 46 ROOTMARKS in batch mode.

   Int_t nbench  = 0;
   Float_t rtall = 0;
   Float_t cpall = 0;
   Float_t hp735 = 27;
   Float_t norm  = hp735;
   Float_t rtmark,cpmark;

   printf("---------------ROOT %s benchmarks summary--------------------\n",gROOT->GetVersion());
   gBenchmark->Summary();
   printf("\n---------------ROOT %s benchmarks summary (in ROOTMARKS)-----\n",gROOT->GetVersion());
   printf("   For comparison, an HP735/99 is benchmarked at 27 ROOTMARKS\n");
   Float_t hsimple_rt = gBenchmark->GetRealTime("hsimple");
   Float_t hsimple_ct = gBenchmark->GetCpuTime("hsimple");
   if (hsimple_rt > 0) {
      rtmark = norm*(10.62/hsimple_rt);
      cpmark = norm*(8.19/hsimple_ct);
      nbench++;
      rtall += rtmark;
      cpall += cpmark;
      printf("hsimple     = %7.2f RealMARKS,  = %7.2f CpuMARKS\n",rtmark,cpmark);
   }

   Float_t hsum_rt = gBenchmark->GetRealTime("hsum");
   Float_t hsum_ct = gBenchmark->GetCpuTime("hsum");
   if (hsum_rt > 0) {
      rtmark = norm*(6.09/hsum_rt);
      cpmark = norm*(4.21/hsum_ct);
      nbench++;
      rtall += rtmark;
      cpall += cpmark;
      printf("hsum        = %7.2f RealMARKS,  = %7.2f CpuMARKS\n",rtmark,cpmark);
   }

   Float_t fillrandom_rt = gBenchmark->GetRealTime("fillrandom");
   Float_t fillrandom_ct = gBenchmark->GetCpuTime("fillrandom");
   if (fillrandom_rt > 0) {
      rtmark = norm*(0.92/fillrandom_rt);
      cpmark = norm*(0.29/fillrandom_ct);
      nbench++;
      rtall += rtmark;
      cpall += cpmark;
      printf("fillrandom  = %7.2f RealMARKS,  = %7.2f CpuMARKS\n",rtmark,cpmark);
   }

   Float_t fit1_rt = gBenchmark->GetRealTime("fit1");
   Float_t fit1_ct = gBenchmark->GetCpuTime("fit1");
   if (fit1_rt > 0) {
      rtmark = norm*(1.42/fit1_rt);
      cpmark = norm*(0.76/fit1_ct);
      nbench++;
      rtall += rtmark;
      cpall += cpmark;
      printf("fit1        = %7.2f RealMARKS,  = %7.2f CpuMARKS\n",rtmark,cpmark);
   }

   Float_t tornado_rt = gBenchmark->GetRealTime("tornado");
   Float_t tornado_ct = gBenchmark->GetCpuTime("tornado");
   if (tornado_rt > 0) {
      rtmark = norm*(1.04/tornado_rt);
      cpmark = norm*(0.88/tornado_ct);
      nbench++;
      rtall += rtmark;
      cpall += cpmark;
      printf("tornado     = %7.2f RealMARKS,  = %7.2f CpuMARKS\n",rtmark,cpmark);
   }

   Float_t na49_rt = gBenchmark->GetRealTime("na49");
   Float_t na49_ct = gBenchmark->GetCpuTime("na49");
   if (na49_rt > 0) {
      rtmark = norm*(31.08/na49_rt);
      cpmark = norm*(30.64/na49_ct);
      nbench++;
      rtall += rtmark;
      cpall += cpmark;
      printf("na49        = %7.2f RealMARKS,  = %7.2f CpuMARKS\n",rtmark,cpmark);
   }

   Float_t na49view_rt = gBenchmark->GetRealTime("na49view");
   Float_t na49view_ct = gBenchmark->GetCpuTime("na49view");
   if (na49view_rt > 0) {
      rtmark = norm*(2.82/na49view_rt);
      cpmark = norm*(1.48/na49view_ct);
      nbench++;
      rtall += rtmark;
      cpall += cpmark;
      printf("na49view    = %7.2f RealMARKS,  = %7.2f CpuMARKS\n",rtmark,cpmark);
   }

   Float_t ntuple1_rt = gBenchmark->GetRealTime("ntuple1");
   Float_t ntuple1_ct = gBenchmark->GetCpuTime("ntuple1");
   if (ntuple1_rt > 0) {
      rtmark = norm*(8.27/ntuple1_rt);
      cpmark = norm*(7.24/ntuple1_ct);
      nbench++;
      rtall += rtmark;
      cpall += cpmark;
      printf("ntuple1     = %7.2f RealMARKS,  = %7.2f CpuMARKS\n",rtmark,cpmark);
   }

   Float_t tree_rt = gBenchmark->GetRealTime("tree");
   Float_t tree_ct = gBenchmark->GetCpuTime("tree");
   if (tree_rt > 0) {
      rtmark = norm*(1.35/tree_rt);
      cpmark = norm*(0.90/tree_ct);
      nbench++;
      rtall += rtmark;
      cpall += cpmark;
      printf("tree        = %7.2f RealMARKS,  = %7.2f CpuMARKS\n",rtmark,cpmark);
   }

   Float_t geometry_rt = gBenchmark->GetRealTime("geometry");
   Float_t geometry_ct = gBenchmark->GetCpuTime("geometry");
   if (geometry_rt > 0) {
      rtmark = norm*(7.15/geometry_rt);
      cpmark = norm*(6.14/geometry_ct);
      nbench++;
      rtall += rtmark;
      cpall += cpmark;
      printf("geometry    = %7.2f RealMARKS,  = %7.2f CpuMARKS\n",rtmark,cpmark);
   }

   if (nbench) {
      Float_t rtmean  = rtall/nbench;
      Float_t cpmean  = cpall/nbench;
      Float_t rtmarks = (rtall+cpall)/(2*nbench);
      if (gROOT->IsBatch()) rtmarks *= 27./46.;
      printf("MEAN        = %7.2f RealMARKS,  = %7.2f CpuMARKS\n",rtmean,cpmean);
      printf("\n");
      printf("****************************************************\n");
      printf("* Your machine is estimated at %7.2f ROOTMARKS   *\n",rtmarks);
      printf("****************************************************\n");
   } else {
      printf(" You must run the ROOT benchmarks before executing this command\n");
   }
}
