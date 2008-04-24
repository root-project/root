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
// 11 - TBtree with std::for_each (Full iteration: from the Begin up to the End)
// 12 - TBtree with std::find_if
// 13 - TBtree with std::count_if
// 14 - TOrdCollection with std::for_each (Full iteration: from the Begin up to the End)
// 15 - TOrdCollection with std::find_if
// 16 - TOrdCollection with std::count_if
// 17 - TRefArray with std::for_each (Full iteration: from the Begin up to the End)
// 18 - TRefArray with std::find_if
// 19 - TRefArray with std::count_if


// STD
#include <iostream>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <functional>
// ROOT
#include "TList.h"
#include "TObjString.h"
#include "TObjArray.h"
#include "TMap.h"
#include "TBtree.h"
#include "TOrdCollection.h"
#include "TRefArray.h"
// Local
#include "stressIterators.h"

const char * const cszValue("value");

using namespace std;

template<class __T>
void fill_container(__T* _container, Int_t _count)
{
   _container->SetOwner();

   ostringstream ss;
   for (int i = 0; i < _count; ++i) {
      ss << "test string #" << i;
      TObjString *s(new TObjString(ss.str().c_str()));
      _container->Add(s);
      ss.str("");
   }
}

template<>
void fill_container<TMap>(TMap* _container, Int_t _count)
{
   _container->SetOwner();

   ostringstream ss;
   for (int i = 0; i < _count; ++i) {
      ss << "test string #" << i;
      TObjString *s(new TObjString(ss.str().c_str()));
      _container->Add(s, new TObjString(cszValue));
      ss.str("");
   }
}

//______________________________________________________________________________
void stressIterators() throw(exception)
{
   const Int_t size = 15;

   ostringstream ss;

   {
      // TList
      TList list;
      fill_container(&list, size);

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
   }

   {
      // TObjArray
      TObjArray obj_array(size);
      fill_container(&obj_array, size);

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
   }

   {
      // TMap
      TMap map_container(size);
      fill_container(&map_container, size);

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
      cout << "-----> " << "TestContainer_count_if<TMap>(map_container, cszValue, map_container.GetSize())" << endl;
      TestContainer_count_if<TMap>(map_container, cszValue, map_container.GetSize());
   }

   {
      // TBtree
      TBtree btree_container;
      fill_container(&btree_container, size);

      cout << "\n#11 ====================================" << endl;
      cout << "-----> " << "TestContainer_for_each<TBtree>(btree_container, btree_container.GetSize())" << endl;
      TestContainer_for_each<TBtree>(btree_container, btree_container.GetSize());
      cout << "\n#12 ====================================" << endl;
      cout << "-----> " << "TestContainer_find_if<TBtree>(btree_container, \"test string #3\")" << endl;
      TestContainer_find_if<TBtree>(btree_container, "test string #3");
      cout << "\n#13 ====================================" << endl;
      cout << "-----> " << "TestContainer_count_if<TBtree>(btree_container, \"test string #3\", 1)" << endl;
      TestContainer_count_if<TBtree>(btree_container, "test string #3", 1);
   }

   {
      // TOrdCollection
      TOrdCollection container;
      fill_container(&container, size);

      cout << "\n#14 ====================================" << endl;
      cout << "-----> " << "TestContainer_for_each<TOrdCollection>(container, container.GetSize())" << endl;
      TestContainer_for_each<TOrdCollection>(container, container.GetSize());
      cout << "\n#15 ====================================" << endl;
      cout << "-----> " << "TestContainer_find_if<TOrdCollection>(container, \"test string #3\");" << endl;
      TestContainer_find_if<TOrdCollection>(container, "test string #3");
      cout << "\n#16 ====================================" << endl;
      cout << "-----> " << "TestContainer_count_if<TOrdCollection>(container, \"test string #3\", 1)" << endl;
      TestContainer_count_if<TOrdCollection>(container, "test string #3", 1);
   }

   {
      // TRefArray
     TRefArray container;
     fill_container(&container, size);

     cout << "\n#17 ====================================" << endl;
     cout << "-----> " << "TestContainer_for_each<TRefArray>(container, container.GetSize())" << endl;
     TestContainer_for_each<TRefArray>(container, container.GetLast()+1); // TODO: why container.GetSize() returns 16 instead of 15
     cout << "\n#18 ====================================" << endl;
     cout << "-----> " << "TestContainer_find_if<TOrdCollection>(container, \"test string #3\");" << endl;
     TestContainer_find_if<TRefArray>(container, "test string #3");
     cout << "\n#19 ====================================" << endl;
     cout << "-----> " << "TestContainer_count_if<TOrdCollection>(container, \"test string #3\", 1)" << endl;
     TestContainer_count_if<TRefArray>(container, "test string #3", 1);
   }

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
