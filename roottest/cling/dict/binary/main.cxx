#include <TClass.h>
#include <iostream>

int main() {
   if (!TClass::GetClass("TheClass")) {
      std::cerr << "TheClass not found!\n";
      exit(1);
   }
   return 0;
}
