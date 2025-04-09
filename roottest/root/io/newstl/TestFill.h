#ifndef TEST_FILL_H
#define TEST_FILL_H

#include "ROOT/RVec.hxx"

template <class T> void fill(T& filled, UInt_t seed);
template <class F, class S> void fill(std::pair<F, S> &filled, UInt_t seed);
template <class F, class S> void fill(std::pair<F, S*> &filled, UInt_t seed);
template <class F, class S> void fill(std::pair<F*, S> &filled, UInt_t seed);
template <class F, class S> void fill(std::pair<F*, S*> &filled, UInt_t seed);

template <> void fill(Helper& filled, UInt_t seed) {
   filled.val = seed;
   filled.dval = seed/33.3;
}

template <> void fill(THelper& filled, UInt_t seed) {
   filled.val = seed;
   filled.dval = seed/66.6;
}

template <> void fill(THelperDerived& filled, UInt_t seed) {
   filled.val = seed;
   filled.dval = seed/66.6;
   filled.f = seed*22.2;
}

template <> void fill(HelperDerived& filled, UInt_t seed) {
   filled.val = seed;
   filled.dval = seed/66.6;
   filled.f = seed*22.2;
}

template <class T> void fill(GHelper<T>& filled, UInt_t seed) {
   fill(filled.val,seed);
}

template <> void fill(double& filled, UInt_t seed) {
   filled = seed;
}

template <> void fill(float& filled, UInt_t seed) {
   filled = seed;
}

template <> void fill(int& filled, UInt_t seed) {
   filled = seed;
}

template <> void fill(short& filled, UInt_t seed) {
   filled = seed;
}

template <> void fill(char& filled, UInt_t seed) {
   filled = seed;
}

template <> void fill(bool& filled, UInt_t seed) {
   filled = seed;
}

template <> void fill(TString& filled, UInt_t seed) {
   UInt_t size = seed%20;
   filled = "";

   for(UInt_t i=0; i<size; i++) {
      char c = (i&seed)+'a';
      filled += c;
   }
}

template <> void fill(std::string& filled, UInt_t seed) {
   TString temp; 
   fill(temp,seed);
   const_cast<std::string&>(filled) = temp.Data();
}

template <> void fill(const std::string& filled, UInt_t seed) {
   TString temp; 
   fill(temp,seed);
   const_cast<std::string&>(filled) = temp.Data();
}

template <> void fill(TNamed& filled, UInt_t seed) {
   TString temp; 
   fill(temp,seed);
   filled.SetName(temp.Data());
   fill(temp,seed+1);
   filled.SetTitle(temp.Data());
}

template <> void fill(const TNamed& cfilled, UInt_t seed) {
   TNamed &filled = const_cast<TNamed&>(cfilled);
   fill(filled,seed);
}
const char* GetEHelperStringValue(const EHelper &eval);
template <> void fill(EHelper& filled, UInt_t seed) {
   switch ( seed%3 ) {
      case 0: filled = kZero; break;
      case 1: filled = kOne;  break;
      case 2: filled = kTwo;  break;
      default: filled = kHelperEnd; break;
   }
}

template <class T> void fill(T& filled, UInt_t seed) {
   UInt_t size = seed%10;

   filled.clear();
   for(UInt_t i=0; i<size; i++) {
      typename T::value_type val;
      fill(val,seed*10+i);
      filled.push_back(val);
  }  
}

template <class T> void fill(std::set<T>& filled, UInt_t seed) {
   UInt_t size = seed%10;

   filled.clear();
   for(UInt_t i=0; i<size; i++) {
      T val;
      fill(val,seed*10+i);
      filled.insert(val);
  }  
}

template <class T> void fill(std::multiset<T>& filled, UInt_t seed) {
   UInt_t size = seed%10;

   filled.clear();
   for(UInt_t i=0; i<size; i++) {
      T val;
      fill(val,seed*10+i);
      filled.insert(val);
  }  
}

template <class Key, class T> void fill(std::map<Key,T>& filled, UInt_t seed) {
   UInt_t size = seed%5;

   filled.clear();
   for(UInt_t i=0; i<size; i++) {
      std::pair<Key,T> p;
      fill(p,seed*5+i);
      filled.insert(p);

//       Key key;
//       fill(key,seed*5+i);
//       T val;
//       fill(val,seed*10+i);
//       filled.insert(make_pair(key,val));
  }  
}

template <class Key, class T> void fill(std::multimap<Key,T>& filled, UInt_t seed) {
   UInt_t size = seed%5;

   filled.clear();
   for(UInt_t i=0; i<size; i++) {
      std::pair<Key,T> p;
      fill(p,seed*5+i);
      filled.insert(p);

//       Key key;
//       fill(key,seed*5+i);
//       T val;
//       fill(val,seed*10+i);
//       filled.insert(make_pair(key,val));
  }  
}

// template <class T> T* createDerived(const T* /* placeHolder used only for overload resolution */);

THelper* createDerived( const THelper* ) {
   return new THelperDerived;
}

HelperClassDef* createDerived( const HelperClassDef* ) {
   return new HelperDerived;
}

void fillDerived(THelper& input, UInt_t seed) {
   THelperDerived *val = dynamic_cast<THelperDerived*>(&input);
   fill(*val,seed);
}

void fillDerived(HelperClassDef& input, UInt_t seed) {
   HelperDerived *val = dynamic_cast<HelperDerived*>(&input);
   fill(*val,seed);
}

template <class T> void fillDerived(ROOT::RVec<T*>& filled, UInt_t seed) {
   UInt_t size = seed%10;

   filled.clear();
   for(UInt_t i=0; i<size; i++) {
      T* val = createDerived( (T*)0 );
      fillDerived(*val,seed*10+i);
      filled.push_back(val);
  }  
}

template <class T> void fillDerived(std::vector<T*>& filled, UInt_t seed) {
   UInt_t size = seed%10;

   filled.clear();
   for(UInt_t i=0; i<size; i++) {
      T* val = createDerived( (T*)0 );
      fillDerived(*val,seed*10+i);
      filled.push_back(val);
  }  
}

template <class T> void fillDerived(std::deque<T*>& filled, UInt_t seed) {
   UInt_t size = seed%10;

   filled.clear();
   for(UInt_t i=0; i<size; i++) {
      T* val =  createDerived( (T*)0 );
      fillDerived(*val,seed*10+i);
      filled.push_back(val);
  }  
}

template <class T> void fillDerived(std::list<T*>& filled, UInt_t seed) {
   UInt_t size = seed%10;

   filled.clear();
   for(UInt_t i=0; i<size; i++) {
      T* val =  createDerived( (T*)0 );
      fillDerived(*val,seed*10+i);
      filled.push_back(val);
  }  
}

template <class T> void fill(ROOT::RVec<T*>& filled, UInt_t seed) {
   UInt_t size = seed%10;

   filled.clear();
   for(UInt_t i=0; i<size; i++) {
      T* val = new T;
      fill(*val,seed*10+i);
      filled.push_back(val);
  }  
}

template <class T> void fill(std::vector<T*>& filled, UInt_t seed) {
   UInt_t size = seed%10;

   filled.clear();
   for(UInt_t i=0; i<size; i++) {
      T* val = new T;
      fill(*val,seed*10+i);
      filled.push_back(val);
  }  
}

template <class T> void fill(std::deque<T*>& filled, UInt_t seed) {
   UInt_t size = seed%10;

   filled.clear();
   for(UInt_t i=0; i<size; i++) {
      T* val = new T;
      fill(*val,seed*10+i);
      filled.push_back(val);
  }  
}

template <class T> void fill(std::list<T*>& filled, UInt_t seed) {
   UInt_t size = seed%10;

   filled.clear();
   for(UInt_t i=0; i<size; i++) {
      T* val = new T;
      fill(*val,seed*10+i);
      filled.push_back(val);
  }  
}

template <class T> void fill(std::set<T*, PtrCmp<T> >& filled, UInt_t seed) {
   UInt_t size = seed%10;

   filled.clear();
   for(UInt_t i=0; i<size; i++) {
      T* val = new T;
      fill(*val,seed*10+i);
      filled.insert(val);
  }  
}

template <class T> void fill(std::multiset<T*, PtrCmp<T> >& filled, UInt_t seed) {
   UInt_t size = seed%10;

   filled.clear();
   for(UInt_t i=0; i<size; i++) {
      T* val = new T;
      fill(*val,seed*10+i);
      filled.insert(val);
  }  
}

template <class F, class S> void fill(std::pair<F, S> &filled, UInt_t seed) {
   UInt_t fseed = (seed * 2 + 3 ) % 7;
   UInt_t sseed = (seed * 3 + 2 ) % 7;

   fill(filled.first,fseed);
   fill(filled.second,sseed);
}

template <class F, class S> void fill(std::pair<F, S*> &filled, UInt_t seed) {
   UInt_t fseed = (seed * 2 + 3 ) % 7;
   UInt_t sseed = (seed * 3 + 2 ) % 7;

   fill(filled.first,fseed);
   filled.second = new S;
   fill(*filled.second,sseed);
}

template <class F, class S> void fill(std::pair<F*, S> &filled, UInt_t seed) {
   UInt_t fseed = (seed * 2 + 3 ) % 7;
   UInt_t sseed = (seed * 3 + 2 ) % 7;

   filled.first = new F;
   fill(*filled.first,fseed);
   fill(filled.second,sseed);
}

template <class F, class S> void fill(std::pair<F*, S*> &filled, UInt_t seed) {
   UInt_t fseed = (seed * 2 + 3 ) % 7;
   UInt_t sseed = (seed * 3 + 2 ) % 7;

   filled.first = new F;
   fill(*filled.first,fseed);
   filled.second = new S;
   fill(*filled.second,sseed);
}

template <class Key, class T> void fill(std::map<Key*, T, PtrCmp<Key> >& filled, UInt_t seed) {
   UInt_t size = seed%5;

   filled.clear();
   for(UInt_t i=0; i<size; i++) {
      std::pair<Key*,T> p;
      fill(p,seed*5+i);
      filled.insert(p);

/*       Key *key = new Key; */
/*       fill(*key,seed*5+i); */
/*       T val; */
/*       fill(val,seed*10+i); */
/*       filled.insert(make_pair(key,val)); */
  }  
}

template <class Key, class T> void fill(std::multimap<Key*, T, PtrCmp<Key> >& filled, UInt_t seed) {
   UInt_t size = seed%5;

   filled.clear();
   for(UInt_t i=0; i<size; i++) {
      std::pair<Key*,T> p;
      fill(p,seed*5+i);
      filled.insert(p);

/*       TKey *key = new TKey; */
/*       fill(*key,seed*5+i); */
/*       T val; */
/*       fill(val,seed*10+i); */
/*       filled.insert(make_pair(key,val)); */
  }  
}
#endif

