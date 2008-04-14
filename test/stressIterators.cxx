// @(#)root/test:$Id$
// Author: Anar Manafov   01/04/2008

//----------------------------------------------------------------
// This is a tests ROOT Iterators and STL algorithms.
// The test project covers the following cases:
// 1 - TList with std::for_each
// 2 - TList with std::find_if
// 3 - TList with std::count_if
// 4 - TObjArray with std::for_each
// 5 - TObjArray with std::find_if
// 6 - TObjArray with std::count_if



// STD
#include <iostream>
#include <algorithm>
#include <sstream>
#include <stdexcept>
// ROOT
#include "TList.h"
#include "TObjString.h"
#include "TObjArray.h"

using namespace std;

static Int_t gCount = 0;

//______________________________________________________________________________
struct SEnumFunctor {
   bool operator()(TObject *aObj) throw(exception) {     
      if (!aObj)
         throw invalid_argument("SEnumFunctor: aObj is a NULL pointer");

      TObjString *str(dynamic_cast<TObjString*>(aObj));
      if (!str)
         throw runtime_error("SEnumFunctor: Container's element is not a TObjString object.");

      ++gCount;
      cout << str->String().Data() << endl;
      return true;
   }
};

//______________________________________________________________________________
struct SFind {
   SFind(const TString &aStr): fToFind(aStr) {
   }
   bool operator()(TObject *aObj) {
      TObjString *str(dynamic_cast<TObjString*>(aObj));
      return !str->String().CompareTo(fToFind);
   }
private:
   const TString fToFind;
};

//______________________________________________________________________________
// Checking TList with for_each algorithm
template<class T>
void TestContainer_for_each(const T &container, Int_t aSize) throw(exception)
{
   gCount = 0; // TODO: using gCount is a very bad method. Needs to be revised.

   TIter iter(&container);
   for_each(iter.Begin(), TIter::End(), SEnumFunctor());
   if (aSize != gCount)
      throw runtime_error("Test case <TestList_for_each> has failed.");
   cout << "->> Ok." << endl;
}

//______________________________________________________________________________
// Checking a ROOT container with find_if algorithm
template<class T>
void TestContainer_find_if(const T &container, const TString &aToFind) throw(exception)
{
   typedef TIterCategory<T> iterator_t;

   SFind func(aToFind);
   iterator_t iter(&container);
   iterator_t found(
      find_if(iter.Begin(), iterator_t::End(), func)
   );
   if (!(*found))
      throw runtime_error("Test case <TestContainer_find_if> has failed.");

   TObjString *str(dynamic_cast<TObjString*>(*found));
   if (!str)
      throw runtime_error("Test case <TestContainer_find_if> has failed.");

   std::cout << "I found: " << str->String().Data() << std::endl;
   cout << "->> Ok." << endl;
}

//______________________________________________________________________________
// Checking a ROOT container with count_if algorithm
template<class T>
void TestContainer_count_if(const T &container, const TString &aToFind) throw(exception)
{
   typedef TIterCategory<T> iterator_t;

   SFind func(aToFind);
   iterator_t iter(&container);
   typename iterator_t::difference_type cnt(
      count_if(iter.Begin(), iterator_t::End(), func)
   );

   if (1 != cnt)
      throw runtime_error("Test case <TestContainer_count_if> has failed.");

   cout << "->> Ok." << endl;
}

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

   cout << "====================================" << endl;
   cout << "-----> " << "TestContainer_for_each<TList>(list, list.GetSize())" << endl;
   TestContainer_for_each<TList>(list, list.GetSize());

   cout << "====================================" << endl;
   cout << "-----> " << "TestContainer_find_if<TList>(list, \"test string #3\")" << endl;
   TestContainer_find_if<TList>(list, "test string #3");

   cout << "====================================" << endl;
   cout << "-----> " << "TestContainer_count_if<TList>(list, \"test string #3\")" << endl;
   TestContainer_count_if<TList>(list, "test string #3");


   // TObjArray
   TObjArray obj_array(size);
   for (int i = 0; i < size; ++i) {
      ss << "test string #" << i;
      TObjString *s(new TObjString(ss.str().c_str()));
      obj_array.Add(s);
      ss.str("");
   }

   cout << "====================================" << endl;
   cout << "-----> " << "TestContainer_for_each<TObjArray>(obj_array, obj_array.GetSize())" << endl;
   TestContainer_for_each<TObjArray>(obj_array, obj_array.GetEntriesFast());
   
   cout << "====================================" << endl;
   cout << "-----> " << "TestContainer_find_if<TObjArray>(obj_array, \"test string #3\")" << endl;
   TestContainer_find_if<TObjArray>(obj_array, "test string #3");

   cout << "====================================" << endl;
   cout << "-----> " << "TestContainer_count_if<TObjArray>(obj_array, \"test string #3\")" << endl;
   TestContainer_count_if<TObjArray>(obj_array, "test string #3");
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
