
int execFindBackPCMNames(){

   std::vector<std::string>* pcmNames = nullptr;
   TFile f("classesFindBackPCMNames_rflx_rdict.pcm");
   f.ls();
   f.GetObject("__AncestorPCMNames",pcmNames);
   if (!pcmNames) return 1;
   std::cout << "Ancestor pcms:\n";
   for (auto const & name: *pcmNames){
      std::cout << name << std::endl;
   }

   return 0;

}
