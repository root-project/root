//______________________________________________________________________________
int execLibLoad()
{
     
   gSystem->Load("libclasses_dictrflx");

   if (NULL != TClass::GetDict("MyClass")){
      std::cout << "Dictionary found for MyClass\n";
   }
   if (NULL == TClass::GetDict("std::list<MyClass>")){
      std::cout << "Dictionary not found for std::list<MyClass>\n";
   }  

   return 0;
}
