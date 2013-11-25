//______________________________________________________________________________
int execLibLoad()
{
     
   gSystem->Load("libclasses_dictrflx.so");

   if (NULL != TClass::GetDict("A")){
      std::cout << "Dictionary found for A\n";
   }

   if (NULL != TClass::GetDict("B")){
      std::cout << "Dictionary found for B\n";
   }
   
#ifdef __GXX_EXPERIMENTAL_CXX0X__

   if (NULL != TClass::GetDict("C")){
      std::cout << "Dictionary found for C\n";
   }     

#endif

   return 0;
}
