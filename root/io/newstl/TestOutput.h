#ifndef TEST__OUTPUT
#define TEST__OUTPUT

#include <iostream>
#include <sstream>

#include <vector>
#include <deque>
#include <list>
#include <set>
#include <map>
#include <string>

#include "TClass.h"
#include "ROOT/RVec.hxx"

namespace TestDebug {
   enum { kValues    = 1<<0, 
          kAddresses = 1<<1 };
}

UInt_t DebugTest(Int_t newLevel = -1) {
   static UInt_t debugLevel = 0;

   if (newLevel>0) debugLevel |= newLevel;
   else if (newLevel==0) debugLevel = 0;

   return debugLevel;
}

void Debug(const std::string &msg) {
   if (DebugTest()) std::cerr << "Debug: " << msg << "\n";
}

void Unsupported(const std::string &what) {
   std::cerr << "ROOT " << ROOT_RELEASE << " does not support "
             << what << std::endl;
}

void Unsupported(Int_t version, const std::string &what) {
   std::cerr << "ROOT " << version << " did not support "
             << what << std::endl;
}

void TestError(const std::string &test, const char *msg) {
   std::cerr << "Error for '" << test << "' : " << msg << "\n";
}

void TestError(const std::string &test, const std::string &msg) {
   TestError(test,msg.c_str());
}

template <class T> void TestError(const std::string &test, const T &orig, const T &copy);
template <class T> void TestError(const std::string &test, T* orig, T* copy);

template <class T> void TestError(const std::string &test, 
                                  const ROOT::RVec<T> &/*orig*/, 
                                  const ROOT::RVec<T> &/*copy*/) {
   TestError(test,"Containers are not equivalent! See previous errors");
}

template <class T> void TestError(const std::string &test, 
                                  const std::vector<T> &/*orig*/, 
                                  const std::vector<T> &/*copy*/) {
   TestError(test,"Containers are not equivalent! See previous errors");
}

template <class T> void TestError(const std::string &test, 
                                  std::vector<T> * /* orig */, 
                                  std::vector<T> * /* copy */) {
   TestError(test,"Containers are not equivalent! See previous errors");
}

template <class T> void TestError(const std::string &test, 
                                  const std::deque<T> &/*orig*/, 
                                  const std::deque<T> &/*copy*/) {
   TestError(test,"Containers are not equivalent! See previous errors");
}

template <class T> void TestError(const std::string &test, 
                                  std::deque<T> */*orig*/, 
                                  std::deque<T> */*copy*/) {
   TestError(test,"Containers are not equivalent! See previous errors");
}


template <class T> void TestError(const std::string &test, 
                                  const std::list<T> &/*orig*/, 
                                  const std::list<T> &/*copy*/) {
   TestError(test,"Containers are not equivalent! See previous errors");
}

template <class T> void TestError(const std::string &test, 
                                  std::list<T> */*orig*/, 
                                  std::list<T> */*copy*/) {
   TestError(test,"Containers are not equivalent! See previous errors");
}

template <class T> void TestError(const std::string &test, 
                                  const std::set<T> &/*orig*/, 
                                  const std::set<T> &/*copy*/) {
   TestError(test,"Containers are not equivalent! See previous errors");
}

template <class T> void TestError(const std::string &test, 
                                  std::set<T> */*orig*/, 
                                  std::set<T> */*copy*/) {
   TestError(test,"Containers are not equivalent! See previous errors");
}

template <class T> void TestError(const std::string &test, 
                                  const std::multiset<T> &/*orig*/, 
                                  const std::multiset<T> &/*copy*/) {
   TestError(test,"Containers are not equivalent! See previous errors");
}

template <class T> void TestError(const std::string &test, 
                                  std::multiset<T> */*orig*/, 
                                  std::multiset<T> */*copy*/) {
   TestError(test,"Containers are not equivalent! See previous errors");
}

template <class Key, class T> void TestError(const std::string &test, 
                                             const std::map<Key, T> &/*orig*/, 
                                             const std::map<Key, T> &/*copy*/) {
   TestError(test,"Containers are not equivalent! See previous errors");
}

template <class Key, class T> void TestError(const std::string &test, 
                                             std::map<Key, T> */*orig*/, 
                                             std::map<Key, T> */*copy*/) {
   TestError(test,"Containers are not equivalent! See previous errors");
}

template <class Key, class T> void TestError(const std::string &test, 
                                             const std::multimap<Key, T> &/*orig*/, 
                                             const std::multimap<Key, T> &/*copy*/) {
   TestError(test,"Containers are not equivalent! See previous errors");
}

template <class Key, class T> void TestError(const std::string &test, 
                                             std::multimap<Key, T> */*orig*/, 
                                             std::multimap<Key, T> */*copy*/) {
   TestError(test,"Containers are not equivalent! See previous errors");
}

void TestError(const std::string &test, const Helper &orig, const Helper &copy) {
   TestError(test, orig.CompMsg(copy));
}

void TestError(const std::string &test, const HelperClassDef &orig, const HelperClassDef &copy) {
   TestError(test, orig.CompMsg(copy));
}

void TestError(const std::string &test, 
               const nonvirtHelper &orig, 
               const nonvirtHelper &copy) {
   TestError(test, Form("nonvirtHelper object wrote %d and read %d\n",
                        orig.val,copy.val));
}

template <class T> void TestError(const std::string &test, const GHelper<T> &orig, const GHelper<T> &copy) {
   TClass *cl = TClass::GetClass(typeid(T));
   const char* classname = cl?cl->GetName():typeid(T).name();   
   std::stringstream s;
   s << test << " on GHelper of " << classname; //  << std::ends;
   TestError(s.str(), orig.val, copy.val);
}

const char* GetEHelperStringValue(const EHelper &eval) {
   switch(eval) {
      case kZero: return "kZero";
      case kOne:  return "kOne";
      case kTwo:  return "kTwo";
      case kHelperEnd:  return "kEnd";
      default:    return "unknown val";
   }
}

void TestError(const std::string &test, const EHelper &orig, const EHelper &copy) {
   std::stringstream s;
   s << "We wrote: " << GetEHelperStringValue(orig) 
     << " but read " << GetEHelperStringValue(copy); //  << std::ends;
   TestError(test, s.str());
}

void TestError(const std::string &test, const THelper &orig, const THelper &copy) {
   TestError(test, orig.CompMsg(copy));
}

void TestError(const std::string &test, const TNamed &orig, const TNamed &copy) {
   std::stringstream s;
   s << "We wrote: name=" << orig.GetName() << " title=" << orig.GetTitle() << " but read " 
     << "name=" << copy.GetName() << " title=" << copy.GetTitle(); //  << std::ends;
   TestError(test, s.str());
}

template <class T> void TestError(const std::string &test, const T &orig, const T &copy) {
   std::stringstream s;
   s << "We wrote: " << orig << " but read " << copy; //  << std::ends;
   TestError(test, s.str());
}

/*
void TestError(const std::string &test, Helper* orig, Helper* copy) {
   if (orig==0 || copy==0) {
      TestError(test,Form("For Helper, non-initialized pointer %p %p",orig,copy));
   } else {
      TestError(test, *orig, *copy); 
   }
}
*/

template <class T> void TestError(const std::string &test, T* orig, T* copy) {
   TClass *cl = TClass::GetClass(typeid(T));
   const char* classname = cl?cl->GetName():typeid(T).name();   
   
   if (orig==0 || copy==0) {
      TestError(test,Form("For %s, non-initialized pointer %p %p",classname,(void*)orig,(void*)copy));
   } else {
      TestError(test, *orig, *copy); 
   }
}

template <class F, class S> void TestError(const std::string &test,
                                           const std::pair<F,S> &orig, const std::pair<F,S> &copy) {
   TestError(test,"pair not equal!");
   TestError(test, orig.first, copy.first);
   TestError(test, orig.second, copy.second);
}

#endif // TEST__OUTPUT
