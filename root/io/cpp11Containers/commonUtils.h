#include <iostream>
#include "TH1F.h"
#include <unordered_set>
#include <forward_list>
#include <list>
#include <deque>
#include <algorithm>
#include "TFile.h"
#include "TRandom.h"

#ifndef ROOTTEST_COMMON_UTILS
#define ROOTTEST_COMMON_UTILS

template <class T>
bool IsSame(const T& a, const T& b){
   cout << "ERROR\n";
   return a==b;
}

template <>
bool IsSame<>(const double& a, const double& b){
   if (a==b) return true;
   cout << "Error numbers differ: " << a << " " << b << std::endl;
   return false;
}

template<class Cont>
bool IsSameCont(const Cont& a, const Cont& b){
   auto size =std::distance(a.begin(),a.end());
   if (size != std::distance(b.begin(),b.end())) return false;
   for (auto aIt = a.cbegin(), bIt = b.begin();aIt!=a.end();aIt++,bIt++ ){
      if (! IsSame(*aIt, *bIt)) return false;
   }

   return true;
}

template <class T, class ALLOCATOR>
bool IsSame(const std::forward_list<T,ALLOCATOR>& a, const std::forward_list<T,ALLOCATOR>& b){
   return IsSameCont(a,b);
}

template <class T, class ALLOCATOR>
bool IsSame(const std::list<T,ALLOCATOR>& a, const std::list<T,ALLOCATOR>& b){
   return IsSameCont(a,b);
}
template <class T, class ALLOCATOR>
bool IsSame(const std::vector<T,ALLOCATOR>& a, const std::vector<T,ALLOCATOR>& b){
   return IsSameCont(a,b);
}
template <class T, class ALLOCATOR>
bool IsSame(const std::deque<T,ALLOCATOR>& a, const std::deque<T,ALLOCATOR>& b){
   return IsSameCont(a,b);
}

template<class T>
bool IsSame(const std::unordered_set<T>& a, const std::unordered_set<T>& b){
   std::vector<T> v1;
   for (auto& el : a) v1.emplace_back(el);
   std::vector<T> v2;
   for (auto& el : b) v2.emplace_back(el);

   auto sortingFunction = [](const T& a, const T& b){return std::string(a.GetName())<std::string(b.GetName());};
   std::sort(v1.begin(),v1.end(),sortingFunction); // FUNDAMENTAL!
   std::sort(v2.begin(),v2.end(),sortingFunction); // FUNDAMENTAL!
   return IsSame(v1,v2);
}

template<class T>
bool IsSame(const std::unordered_set<std::vector<T>>& a, const std::unordered_set<std::vector<T>>& b){
   std::vector<std::vector<T>> v1;
   for (auto& el : a) v1.emplace_back(el);
   std::vector<std::vector<T>> v2;
   for (auto& el : b) v2.emplace_back(el);

   auto sortingFunction = [](const std::vector<T>& a, const std::vector<T>& b){
      std::string namesA,namesB;
      for (auto&& h:a) namesA+=h.GetName();
      for (auto&& h:b) namesB+=h.GetName();
      return namesA<namesB;
   };
   std::sort(v1.begin(),v1.end(),sortingFunction); // FUNDAMENTAL!
   std::sort(v2.begin(),v2.end(),sortingFunction); // FUNDAMENTAL!
   return IsSame(v1,v2);
}

template<>
bool IsSame<double>(const std::unordered_set<double>& a, const std::unordered_set<double>& b){
   std::vector<double> v1;
   for (auto& el : a) v1.emplace_back(el);
   std::vector<double> v2;
   for (auto& el : b) v2.emplace_back(el);

   std::sort(v1.begin(),v1.end()); // FUNDAMENTAL!
   std::sort(v2.begin(),v2.end()); // FUNDAMENTAL!
   return IsSame(v1,v2);
}


template <>
bool IsSame<>(const TH1F& a, const TH1F& b){
   if( 0 != strcmp(a.GetName(),b.GetName())) return false;
   if( 0 != strcmp(a.GetTitle(),b.GetTitle())) return false;
   if( a.GetNbinsX() != b.GetNbinsX()) return false;
   for (size_t i=0;i<a.GetNbinsX();++i){
      if (a.GetBinContent(i)!=b.GetBinContent(i)) return false;
      if (a.GetBinError(i)!=b.GetBinError(i)) return false;
   }
   return true;
}

void createFile(const char* filename){
   auto file = TFile::Open(filename,"RECREATE");
   delete file;
}

template<class T>
void writeToFile(const T& obj, const char* objName, const char* filename){
   auto file = TFile::Open(filename,"UPDATE");
   file->WriteObject(&obj,objName);
   delete file;
}

template<class T>
void readAndCheckFromFile(const T& obj, const char* objName, const char* filename){
   auto file = TFile::Open(filename,"READ");
   auto objFromFile =(T*)file->Get(objName);
   if (!objFromFile){
      std::cerr << "Error in reading object " << objName << " from file " << filename << "\n";
      delete file;
      return;
   }
   if (!IsSame(obj,*objFromFile)) {
      std::cerr << "Error: object " << objName << " read from file " << filename << " from file and in memory are not identical!\n";
   }

   delete file;
}

template<class T>
void writeReadCheck(const T& obj, const char* objName, const char* filename){
   writeToFile(obj,objName,filename);
   readAndCheckFromFile(obj,objName,filename);
}

template<class Cont>
void fillHistoCont(Cont& cont, unsigned int n=5000){
   for (auto& h:cont) h.FillRandom("gaus",n);
}
template<class NestedCont>
void fillHistoNestedCont(NestedCont& nestedCont, unsigned int n=5000){
   for (auto& hCont:nestedCont) {
      fillHistoCont(hCont,n);
   }
}

template<class Cont>
void randomizeCont(Cont& cont){
   for (auto& el : cont){
      el*=gRandom->Uniform(1,2);
   }
}

//------------------------------------------------------------------------------
// For the unordered set
namespace std {
   template <>
   class hash<TH1F> {
   public:
      size_t operator()(const TH1F &h) const {
         std::hash<std::string> shash;
         return shash(h.GetName());
      }
   }; // hash
   template <>
   class hash<std::vector<TH1F>> {
   public:
      size_t operator()(const std::vector<TH1F> &hVect) const {
         std::string names;
         for (auto&& h:hVect) names+=h.GetName();
         std::hash<std::string> shash;
         return shash(names);
      }
   }; // hash

   template<>
   struct equal_to<TH1F>{
      bool operator()( const TH1F& a, const TH1F& b ) const {
         return IsSame(a,b);
      }
   };
   template<>
   struct equal_to<std::vector<TH1F>>{
      bool operator()( const std::vector<TH1F>& a, const std::vector<TH1F>& b ) const {
         return IsSame(a,b);
      }
   };


} // std

template<class T, class HASH, class EQ, class ALLOC>
void fillHistoCont(std::unordered_set<T,HASH,EQ,ALLOC>& cont, unsigned int n=5000){
   std::vector<T> v;
   for (auto& el : cont) v.emplace_back(el);
   cont.clear();
   std::sort(v.begin(),v.end(),[](const T& a, const T& b){return std::string(a.GetName())<std::string(b.GetName());}); // FUNDAMENTAL!
   for (auto& h:v) {
      h.FillRandom("gaus",n);
      cont.insert(h);
   }
}

template<class T, class HASH, class EQ, class ALLOC>
void fillHistoNestedCont(std::unordered_set<T,HASH,EQ,ALLOC>& nestedCont, unsigned int n=5000){
   std::vector<T> v;
   for (auto& hCont:nestedCont) v.emplace_back(hCont);
   std::sort(v.begin(),v.end(),[](const T& a, const T& b){
      std::string namesA, namesB;
      for (auto&& h : a) namesA+=h.GetName();
      for (auto&& h : b) namesB+=h.GetName();
      return namesA < namesB;
   });
   nestedCont.clear();
   for (auto& hCont:v) {
      fillHistoCont(hCont,n);
      nestedCont.insert(hCont);
   }
}


template<class T, class HASH, class EQ, class ALLOC>
void randomizeCont(std::unordered_set<T,HASH,EQ,ALLOC>& cont){
   std::vector<T> v;
   for (auto& el : cont) v.emplace_back(el);
   std::sort(v.begin(),v.end()); // FUNDAMENTAL!
   cont.clear();
   for (auto& el : v){
      cont.insert(el*gRandom->Uniform(1,2));
   }
}


#endif
