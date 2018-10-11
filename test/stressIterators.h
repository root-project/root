// @(#)root/test:$Id$
// Author: Anar Manafov   18/04/2008
#ifndef ROOT_stressIterators
#define ROOT_stressIterators

#ifdef WIN32
#pragma warning(disable: 4290)
#endif

static Int_t gCount = 0;

// Here we have a collection of functors and functions used by the test suit
//______________________________________________________________________________
template<class T>
struct SEnumFunctor {
   bool operator()(TObject *aObj) const {
      if (!aObj)
         throw std::invalid_argument("SEnumFunctor: aObj is a NULL pointer");

      if ((aObj->IsA() == TObjString::Class())) {
         TObjString *str(dynamic_cast<TObjString*>(aObj));
         if (!str)
            throw std::runtime_error("SEnumFunctor: Container's element is not a TObjString object.");

         ++gCount;
         std::cout << str->String().Data() << std::endl;
      }

      return true;
   }
};

//______________________________________________________________________________
template<>
struct SEnumFunctor<TMap> {
   bool operator()(TObject *aObj) const {
      if (!aObj)
         throw std::invalid_argument("SEnumFunctor: aObj is a NULL pointer");

      if ((aObj->IsA() == TPair::Class())) {
         TPair *pair(dynamic_cast<TPair*>(aObj));
         if (!pair)
            throw std::runtime_error("SEnumFunctor: Container's element is not a TPair object.");

         TObjString *key(dynamic_cast<TObjString*>(pair->Key()));
         TObjString *value(dynamic_cast<TObjString*>(pair->Value()));
         if (!key || !value)
            throw std::runtime_error("SEnumFunctor: Can't retriev key/value of a pair");

         ++gCount;
         std::cout << key->String().Data() << " : " << value->String().Data() << std::endl;
      }

      return true;
   }
};

//______________________________________________________________________________
template<class T>
struct SFind : std::function<TObject*(TString, bool)> {
   bool operator()(TObject *_Obj, const TString &_ToFind) const {
      TObjString *str(dynamic_cast<TObjString*>(_Obj));
      if (!str)
         throw std::runtime_error("SFind: Container's element is not a TObString object.");
      return !str->String().CompareTo(_ToFind);
   }
};

//______________________________________________________________________________
template<>
struct SFind<TMap> : std::function<TObject*(TString, bool)> {
   bool operator()(TObject *_Obj, const TString &_ToFind) const {
      TPair *pair(dynamic_cast<TPair*>(_Obj));
      if (!pair)
         throw std::runtime_error("SFind: Container's element is not a TPair object.");
      // Checking the VALUE of the pair
      TObjString *str(dynamic_cast<TObjString*>(pair->Value()));
      return !str->String().CompareTo(_ToFind);
   }
};

//______________________________________________________________________________
// Checking a given container with for_each algorithm
// Full iteration: from Begin to End
template<class T>
void TestContainer_for_each(const T &container, Int_t aSize)
{
   gCount = 0; // TODO: a use of gCount is a very bad method. Needs to be revised.

   TIterCategory<T> iter(&container);
   std::for_each(iter.Begin(), TIterCategory<T>::End(), SEnumFunctor<T>());
   if (aSize != gCount)
      throw std::runtime_error("Test case <TestList_for_each> has failed.");
   std::cout << "->> Ok." << std::endl;
}

//______________________________________________________________________________
// Checking a given container with for_each algorithm
// Partial iteration: from Begin to 3rd element
template<class T>
void TestContainer_for_each2(const T &container)
{
   gCount = 0; // TODO: a use of gCount is a very bad method. Needs to be revised.

   TIterCategory<T> iter(&container);
   TIterCategory<T> iter_end(&container);
   // Artificially shifting the iterator to the 4th potision - a new End iterator
   iter_end();
   iter_end();
   iter_end();
   iter_end();
   std::for_each(iter.Begin(), iter_end, SEnumFunctor<T>());
   if (3 != gCount)
      throw std::runtime_error("Test case <TestList_for_each2> has failed.");
   std::cout << "->> Ok." << std::endl;
}

//______________________________________________________________________________
// Checking a ROOT container with find_if algorithm
template<class T>
void TestContainer_find_if(const T &container, const TString &aToFind)
{
   typedef TIterCategory<T> iterator_t;

   iterator_t iter(&container);
   iterator_t found(
      std::find_if(iter.Begin(), iterator_t::End(), std::bind(SFind<T>(), std::placeholders::_1, aToFind))
   );
   if (!(*found))
      throw std::runtime_error("Test case <TestContainer_find_if> has failed. Can't find object.");

   // Checking whether element is a Pair or not
   if (((*found)->IsA() == TPair::Class())) {
      TPair *pair(dynamic_cast<TPair*>(*found));
      if (!pair)
         throw std::runtime_error("TestContainer_find_if: Container's element is not a TPair object.");
      TObjString *key(dynamic_cast<TObjString*>(pair->Key()));
      TObjString *val(dynamic_cast<TObjString*>(pair->Value()));
      std::cout << "I found: [" << key->String().Data() << " : " << val->String().Data() << "]" << std::endl;
      std::cout << "->> Ok." << std::endl;
      return;
   }

   TObjString *str(dynamic_cast<TObjString*>(*found));
   if (!str)
      throw std::runtime_error("Test case <TestContainer_find_if> has failed. String object is NULL");

   std::cout << "I found: " << str->String().Data() << std::endl;
   std::cout << "->> Ok." << std::endl;
}

//______________________________________________________________________________
// Checking a ROOT container with count_if algorithm
template<class T>
void TestContainer_count_if(const T &container, const TString &aToFind, Int_t Count)
{
   typedef TIterCategory<T> iterator_t;

   iterator_t iter(&container);
   typename iterator_t::difference_type cnt(
      std::count_if(iter.Begin(), iterator_t::End(), std::bind(SFind<T>(), std::placeholders::_1, aToFind))
   );

   if (Count != cnt)
      throw std::runtime_error("Test case <TestContainer_count_if> has failed.");

   std::cout << "->> Ok." << std::endl;
}

#endif
