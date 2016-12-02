#include "TFile.h"


void writeFile() {
   TFile f("VectorWithoutDictionary.root", "recreate");
   ECont  obj;
   obj.elems.push_back( Elem(5) );
   f.WriteObject(&obj, "myobj");

   obj.elems.push_back( Elem(6) );
   obj.elems.push_back( Elem(7) );
   f.WriteObject(&obj, "myobj2");

   f.Close();
}

int readFile() {
   TFile f("VectorWithoutDictionary.root", "read");

   if (f.IsZombie()) {
      std::cerr << "Failed to open file: " << f.GetName() << '\n';
      return 1;
   }

   ECont  *obj;

   f.GetObject("myobj",obj);
   if (!obj) {
      std::cerr << "Missing obj in file: " << f.GetName() << '\n';
      return 2;
   }
   std::cout << obj->elems.size() << '\n';
   if ( obj->elems.size() ) std::cout << obj->elems[0].i << '\n';
   if( obj->elems.size() == 1 && obj->elems[0].i == 5 ) {
      std::cout << "test OK" << std::endl;
   } else {
      std::cout << "test FAILED" << std::endl;
      return 3;
   }

   f.GetObject("myobj2",obj);
   if (!obj) {
      std::cerr << "Missing obj2 in file: " << f.GetName() << '\n';
      return 4;
   }

   std::cout << "Second object\n";
   std::cout << obj->elems.size() << '\n';
   for(auto e : obj->elems) {
      std::cout << e.i << '\n';
   }
   if( obj->elems.size()==3 && obj->elems[1].i == 6 ) {
      std::cout << "test OK" << std::endl;
   } else {
      std::cout << "test FAILED" << std::endl;
      return 5;
   }
   return 0;
}


int execVectorDMWriteWithoutDictionary()
{
   writeFile();
   return readFile();
}


