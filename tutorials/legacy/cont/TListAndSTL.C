/// \file
/// \ingroup tutorial_cont_legacy
/// \notebook -nodraw
/// This is an example of using TList with STL algoritms in CLING.
///
/// #### Output produced by `.x TListAndSTL.C`
/// \macro_output
///
/// #### `TListAndSTL.C` code
/// \macro_code
///
/// \author Anar Manafov

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

   SFind(const TString &aStr): fToFind(aStr) {

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
   TList stringList;
   ostringstream ss;
   for (int i = 0; i < size; ++i) {
      ss << "test string #" << i;
      TObjString *s(new TObjString(ss.str().c_str()));
      stringList.Add(s);
      ss.str("");
   }


   // ### Example #1
   // Running the std::for_each algorithm on the list
   for_each(stringList.begin(), stringList.end(), SEnumFunctor());

   // ### Example #2
   // We can try to find something in the container
   // using the std::find_if algorithm on the list
   string strToFind("test string #4");
   SFind func(strToFind.c_str());

   TIterCategory<TList> iter_cat(&stringList);
   TIterCategory<TList> found
      = find_if(iter_cat.Begin(), TIterCategory<TList>::End(), func);

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
