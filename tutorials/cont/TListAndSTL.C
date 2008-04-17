// @(#)root/test:$Id$
// Author: Anar Manafov   02/04/2008

// This is an exmpale of using TList with STL algoritms in CINT.

// STD
#include <algorithm>
#include <iostream>
#include <sstream>

// ROOT
#include "TList.h"
#include "TCollection.h"
#include "TObjString.h"

// A functor for the for_each algorithm
struct SEnumFunctor {
   bool operator()(TObject *aObj) {
      if (!aObj)
         return false;

      TObjString *str(dynamic_cast<TObjString*>(aObj));
      if (!str)
         return false;

      cout << "Value: " << str->String().Data() << endl;
      return true;
   }
};

// A functor for the find_if algorithm
struct SFind {
   // using this ugly constructor, since there is problems with std::bindX in CINT
#ifdef __CINT__
   SFind(const SFind &oth) : fToFind(oth.fToFind) {}
   SFind(const TString aStr): fToFind(aStr) {
#else
   SFind(const TString &aStr): fToFind(aStr) {
#endif
   }
   bool operator()(TObject *aObj) {
      TObjString *str(dynamic_cast<TObjString*>(aObj));
      return !str->String().CompareTo(fToFind);
   }
private:
   const TString fToFind;
};


// The "main" function
void TListAndSTL()
{
   const Int_t size(10);

   // Initializing TList container
   TList list;
   ostringstream ss;
   for (int i = 0; i < size; ++i) {
      ss << "test string #" << i;
      TObjString *s(new TObjString(ss.str().c_str()));
      list.Add(s);
      ss.str("");
   }


   //->>>>>>> Example #1 <<<<<<<-
   // running the std::for_each algorithm on the list
   TIter iter(&list);
   for_each(iter.Begin(), TIter::End(), SEnumFunctor());

   //->>>>>>> Example #2 <<<<<<<-
   // we can try to find something in the container
   // using the std::find_if algorithm on the list
   string strToFind("test string #4");
   SFind func(strToFind.c_str());


#ifdef __CINT__
   TIter found(
      find_if(iter.Begin(), TIter::End(), func)
   );
#else // in compilation mode you need to use TIterCategory as an iterator for such a algorithm like find_if
   TIterCategory<TList> iter_cat(&list);
      TIterCategory<TList> found(
         find_if(iter_cat.Begin(), TIterCategory<TList>::End(), func)
      );
#endif

   // Checking the result
   if (!(*found)) {
      cerr << "Can't find the string: \"" << strToFind << "\" in the container" << endl;
      return;
   }

   TObjString *str(dynamic_cast<TObjString*>(*found));
   if (!str) {
      cerr << "Can't find the string: \"" << strToFind << "\" in the container" << endl;
      return;
   }

   cout << "The string has been found: " << str->String().Data() << endl;
}
