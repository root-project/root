void execCheckComplicatedAllocator()
{

   auto vName ="vector<int,__gnu_cxx::__mt_alloc<int,__gnu_cxx::__common_pool_policy<__gnu_cxx::__pool,true> > >";
   gSystem->Load("libcomplicatedAllocator_dictrflx");
   auto d = TClass::GetDict(vName);
   if (!d){
      std::cerr << "ERROR: Dictionary for " << vName << " not found.\n";
   }
   else {
      std::cout << "Dictionary for " << vName << " found.\n";
   }
   return;
}
