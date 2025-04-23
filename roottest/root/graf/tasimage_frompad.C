
int tasimage_frompad()
{
   auto GetVMem = []() {
      ProcInfo_t info;
      gSystem->GetProcInfo(&info);
      return float(info.fMemVirtual);
   };

   gROOT->SetBatch();

   auto img = TImage::Create();
   const auto initVmem = GetVMem();
   for (auto i : ROOT::TSeqI(5000)) {
      TCanvas pad;
      img->FromPad(&pad);
   }
   gSystem->Sleep(1); // give time to the counters to be updated
   const auto endVmem = GetVMem();
   const auto ratio   = endVmem / initVmem;

   if (ratio > 1.5) {
      std::cerr << "Final memory is " << ratio << " times than the initial one. This can be a sign that interpretation of code in "
                   "TASImage::FromPad is taking place."
                << std::endl;
      return 1;
   }

   return 0;
}