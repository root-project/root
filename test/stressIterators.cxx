// @(#)root/test:$Id$
// Author: Anar Manafov   01/04/2008

//----------------------------------------------------------------
// This is a tests ROOT Iterators and STL algorithms.
// The test project covers the following cases:
// 1  - TList with std::for_each (Full iteration: from the Begin up to the End)
// 2  - TList with std::find_if
// 3  - TList with std::count_if
// 4  - TObjArray with std::for_each (Full iteration: from the Begin up to the End)
// 5  - TObjArray with std::find_if
// 6  - TObjArray with std::count_if
// 7  - TMap with std::for_each (Full iteration: from the Begin up to the End)
// 8  - TMap with std::for_each (Partial iteration: from the Begin up to the 3rd element)
// 9  - TMap with std::find_if
// 10 - TMap with std::count_if



// STD
#include <iostream>
#include <algorithm>
#include <sstream>
#include <stdexcept>
// ROOT
#include "TList.h"
#include "TObjString.h"
#include "TObjArray.h"
#include "TMap.h"
// Local
#include "stressIterators.h"

using namespace std;

//______________________________________________________________________________
void stressIterators() throw(exception)
{
   const Int_t size = 15;

   ostringstream ss;

   // TList
   TList list;
   for (int i = 0; i < size; ++i) {
      ss << "test string #" << i;
      TObjString *s(new TObjString(ss.str().c_str()));
      list.Add(s);
      ss.str("");
   }

   cout << "#1 ====================================" << endl;
   cout << "-----> " << "TestContainer_for_each<TList>(list, list.GetSize())" << endl;
   TestContainer_for_each<TList>(list, list.GetSize());

   cout << "\n#2 ====================================" << endl;
   cout << "-----> " << "TestContainer_find_if<TList>(list, \"test string #3\")" << endl;
   TestContainer_find_if<TList>(list, "test string #3");

   cout << "\n#3 ====================================" << endl;
   cout << "-----> " << "TestContainer_count_if<TList>(list, \"test string #3\", 1)" << endl;
   // we suppose to find exactly one match
   TestContainer_count_if<TList>(list, "test string #3", 1);


   // TObjArray
   TObjArray obj_array(size);
   for (int i = 0; i < size; ++i) {
      ss << "test string #" << i;
      TObjString *s(new TObjString(ss.str().c_str()));
      obj_array.Add(s);
      ss.str("");
   }

   cout << "\n#4 ====================================" << endl;
   cout << "-----> " << "TestContainer_for_each<TObjArray>(obj_array, obj_array.GetSize())" << endl;
   TestContainer_for_each<TObjArray>(obj_array, obj_array.GetSize());

   cout << "\n#5 ====================================" << endl;
   cout << "-----> " << "TestContainer_find_if<TObjArray>(obj_array, \"test string #3\")" << endl;
   TestContainer_find_if<TObjArray>(obj_array, "test string #3");

   cout << "\n#6 ====================================" << endl;
   cout << "-----> " << "TestContainer_count_if<TObjArray>(obj_array, \"test string #3\", 1)" << endl;
   // we suppose to find exactly one match
   TestContainer_count_if<TObjArray>(obj_array, "test string #3", 1);

   // TMap
   const char * const cszValue("value");
   TMap map_container(size);
   for (int i = 0; i < size; ++i) {
      ss << "test string #" << i;
      TObjString *s(new TObjString(ss.str().c_str()));
      map_container.Add(s, new TObjString(cszValue));
      ss.str("");
   }

   cout << "\n#7 ====================================" << endl;
   cout << "-----> " << "TestContainer_for_each<TMap>(map_container, map_container.GetSize())" << endl;
   TestContainer_for_each<TMap>(map_container, map_container.GetSize());
   cout << "\n#8 ====================================" << endl;
   cout << "-----> " << "TestContainer_for_each2<TMap>(map_container)" << endl;
   TestContainer_for_each2<TMap>(map_container);
   cout << "\n#9 ====================================" << endl;
   cout << "-----> " << "TestContainer_find_if<TMap>(map_container, cszValue)" << endl;
   TestContainer_find_if<TMap>(map_container, cszValue);
   cout << "\n#10 ====================================" << endl;
   cout << "-----> " << "TestContainer_count_if<TMap>(map_container, cszValue, map_container.GetSize());" << endl;
   TestContainer_count_if<TMap>(map_container, cszValue, map_container.GetSize());
}

//______________________________________________________________________________
// return 0 on success and 1 otherwise
int main()
{
   try {

      stressIterators();

   } catch (const exception &e) {
      cerr << "Test has failed!" << endl;
      cerr << "Detailes: " << e.what() << endl;
      return 1;
   } catch (...) {
      cerr << "Test has failed!" << endl;
      cerr << "Unexpected error occurred." << endl;
      return 1;
   }

   cout << "\nTest successfully finished!" << endl;
   return 0;
}
