void execns(){
   if(TClass::GetClass("the_ns")) {
      std::cout << "Namespace the_ns found\n";
   } else {
      std::cout << "Namespace the_ns not found\n";
   }
}
