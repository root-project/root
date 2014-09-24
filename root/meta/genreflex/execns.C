void execns(){
   if(TClass::GetClass("ns")) {
      std::cout << "Namespace ns found\n";
   } else {
      std::cout << "Namespace ns not found\n";
   }
}
