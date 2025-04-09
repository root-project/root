/*
This test has been adapted from code provided by Attila Krasznahorkay.
*/

// System include(s):
#include <typeinfo>
#include <iostream>
#include <vector>

// ROOT include(s):
#include <TClass.h>
#include <TList.h>
#include <TDataMember.h>

// Local include(s):
#include "ClassA_ex2.h"
#include "ClassB_ex2.h"
#include "ClassC_ex2.h"
#include "ClassD_ex2.h"
#include "testSelectNoInstance.h"

void printMembers( TClass* cl ) {

   std::cout << "Members of class " << cl->GetName() << std::endl;

   if (cl->GetCollectionProxy())
      // Members does not matter (and some implementation have them in the base
      // and some newer implementation have the data member in the class itself.
      // Either way, they are internal (since we use the collection proxy) and
      // do no matter.
      return;

   TList* members = cl->GetListOfDataMembers();
   for( Int_t i = 0; i < members->GetSize(); ++i ) {
      TDataMember* member = dynamic_cast< TDataMember* >( members->At( i ) );
      if( ! member ) {
         std::cerr << "Returned object not a TDataMember!" << std::endl;
         continue;
      }
      std::cout << "  - " << member->GetTrueTypeName() << " " << member->GetName()
                << ";" << " // persistent = "
                << ( member->IsPersistent() ? "Yes" : "No" ) << std::endl;
   }

   return;
}

template< class T >
void printDictionary() {

   TClass* cl = TClass::GetClass( typeid( T ) );
   if( ! cl ) {
      std::cerr << "Couldn't find dictionary for class " << typeid( T ).name()
                << std::endl;
      return;
   }

   printMembers( cl );

   return;
}


int execAtlasTest2() {

   if (0!=gSystem->Load("libAtlasTest2_dictrflx"))
      std::cerr << "Error loading dictionary library.\n";   
   
   // Print the properties of all interesting types:
   printDictionary< Atlas::ClassA< Atlas::ClassB > >();
   printDictionary< Atlas::ClassA< Atlas::ClassC > >();
   printDictionary< std::vector< Atlas::ClassA< Atlas::ClassB > > >();
   printDictionary< std::vector< Atlas::ClassA< Atlas::ClassC > > >();
   printDictionary< Atlas::ClassD< Atlas::ClassB > >();
   printDictionary< Atlas::ClassD< Atlas::ClassC > >();
   printDictionary< Atlas::ClassA< Atlas::ClassD< Atlas::ClassB > > >();
   printDictionary< Atlas::ClassA< Atlas::ClassD< Atlas::ClassC > > >();
   printDictionary< std::vector< Atlas::ClassA< Atlas::ClassD< Atlas::ClassB > > > >();
   printDictionary< std::vector< Atlas::ClassA< Atlas::ClassD< Atlas::ClassC > > > >();
   printDictionary< MyClass >();
   printDictionary< MyClass2 >();
   printDictionary< MyDataVector<float, double> >();

   return 0;
}
