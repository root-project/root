#include <iostream>
#include "TH1F.h"
#include <unordered_set>
#include <forward_list>
#include <list>
#include <deque>
#include <map>
#include <unordered_map>
#include <algorithm>
#include "TFile.h"
#include "TRandom.h"
#include <complex>


#ifndef ROOTTEST_COMMON_UTILS
#define ROOTTEST_COMMON_UTILS

bool gVerboseComparison = false;

template <class T>
bool IsSame(const T& a, const T& b){
   std::cout << "ERROR\n";
   return a==b;
}

template <>
bool IsSame<>(const int& a, const int& b){
   if (a==b) return true;
   std::cout << "Error numbers differ: " << a << " " << b << std::endl;
   return false;
}

template <>
bool IsSame<>(const Long64_t& a, const Long64_t& b){
   if (a==b) return true;
   std::cout << "Error numbers differ: " << a << " " << b << std::endl;
   return false;
}

template <>
bool IsSame<>(const double& a, const double& b){
   if (a==b) return true;
   std::cout << "Error numbers differ: " << a << " " << b << std::endl;
   return false;
}

template <>
bool IsSame<>(const float& a, const float& b){
   if (a==b) return true;
   std::cout << "Error numbers differ: " << a << " " << b << std::endl;
   return false;
}

template <class T>
bool IsSame(const std::complex<T>& a, const std::complex<T>& b){
   if (a==b) return true;
   std::cout << "Error complex numbers differ: " << a << " " << b << std::endl;
   return false;
}


template <class T, class U>
bool IsSame(const std::pair<T,U>& a, const std::pair<T,U>& b){
   return IsSame(a.first,b.first) && IsSame(a.second,b.second);
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

   auto sortingFunction = [](const T& o1, const T& o2){return std::string(o1.GetName())<std::string(o2.GetName());};
   std::sort(v1.begin(),v1.end(),sortingFunction); // FUNDAMENTAL!
   std::sort(v2.begin(),v2.end(),sortingFunction); // FUNDAMENTAL!
   return IsSame(v1,v2);
}

template<class T>
bool IsSame(const std::unordered_multiset<T>& a, const std::unordered_multiset<T>& b){
   std::vector<T> v1;
   for (auto& el : a) v1.emplace_back(el);
   std::vector<T> v2;
   for (auto& el : b) v2.emplace_back(el);

   auto sortingFunction = [](const T& o1, const T& o2){return std::string(o1.GetName())<std::string(o2.GetName());};
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

   auto sortingFunction = [](const std::vector<T>& vt1, const std::vector<T>& vt2){
      std::string namesA,namesB;
      for (auto&& h:vt1) namesA+=h.GetName();
      for (auto&& h:vt2) namesB+=h.GetName();
      return namesA<namesB;
   };

   std::sort(v1.begin(),v1.end(),sortingFunction); // FUNDAMENTAL!
   std::sort(v2.begin(),v2.end(),sortingFunction); // FUNDAMENTAL!
   return IsSame(v1,v2);
}

template<class T>
bool IsSame(const std::unordered_multiset<std::vector<T>>& a, const std::unordered_multiset<std::vector<T>>& b){
   std::vector<std::vector<T>> v1;
   for (auto& el : a) v1.emplace_back(el);
   std::vector<std::vector<T>> v2;
   for (auto& el : b) v2.emplace_back(el);

   auto sortingFunction = [](const std::vector<T>& vt1, const std::vector<T>& vt2){
      std::string namesA,namesB;
      for (auto&& h:vt1) namesA+=h.GetName();
      for (auto&& h:vt2) namesB+=h.GetName();
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

template<>
bool IsSame<double>(const std::unordered_multiset<double>& a, const std::unordered_multiset<double>& b){
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
   if( 0 != strcmp(a.GetName(),b.GetName())) {
      if (gVerboseComparison) std::cout << "The names of the histograms differ: " << a.GetName() << " " << b.GetName() << std::endl;
      return false;
   }
   if( 0 != strcmp(a.GetTitle(),b.GetTitle())) {
      if (gVerboseComparison) std::cout << "The title of the histograms differ: " << a.GetTitle() << " " << b.GetTitle() << std::endl;
      return false;
   }
   auto nbinsa = a.GetNbinsX();
   auto nbinsb = b.GetNbinsX();
   if( nbinsa != nbinsb) {
      if (gVerboseComparison) std::cout << "The # of bins of the histograms differ: " << nbinsa << " " << nbinsb << std::endl;
      return false;
   }
   for (int i=0;i<a.GetNbinsX();++i) {
      auto binca = a.GetBinContent(i);
      auto bincallu = *(ULong64_t*)(&binca);
      auto bincb = b.GetBinContent(i);
      auto bincbllu = *(ULong64_t*)(&bincb);
      if (bincallu != bincbllu) {
         if (gVerboseComparison) std::cout << "The content of bin " << i << "  of the histograms differ: " << binca << " " << bincb << std::endl;
         return false;
      }
      auto binea = a.GetBinError(i);
      auto bineallu = *(ULong64_t*)(&binea);
      auto bineb = b.GetBinError(i);
      auto binebllu = *(ULong64_t*)(&bineb);
      if (bineallu != binebllu) {
         if (gVerboseComparison) std::cout << "The error of bin " << i << "  of the histograms differ: " << binea << " " << bineb << std::endl;
         return false;
      }
   }

   return true;
}



template<class T, class U>
bool IsSame(const std::map<T,U>& a, const std::map<T,U>& b){
   std::vector<std::pair<T,U>> v1;
   for (auto& el : a) v1.emplace_back(el);
   std::vector<std::pair<T,U>> v2;
   for (auto& el : b) v2.emplace_back(el);

   return IsSame(v1,v2);
}

template<class T, class U>
bool IsSame(const std::multimap<T,U>& a, const std::multimap<T,U>& b){
   std::vector<std::pair<T,U>> v1;
   for (auto& el : a) v1.emplace_back(el);
   std::vector<std::pair<T,U>> v2;
   for (auto& el : b) v2.emplace_back(el);
   return IsSame(v1,v2);
}

// For gcc48: lambda not correctly compiled
// template<class T, class U> bool sortingFunction(std::pair<T,U>& p1,std::pair<T,U>& p2){return p1.first<p2.first;};

template<class T, class U>
bool IsSame(const std::unordered_map<T,U>& a, const std::unordered_map<T,U>& b){
   std::vector<std::pair<T,U>> v1;
   for (auto& el : a) v1.emplace_back(el);
   std::vector<std::pair<T,U>> v2;
   for (auto& el : b) v2.emplace_back(el);

   auto sortingFunction = [](const std::pair<T,U>& p1,const std::pair<T,U>& p2){return p1.first<p2.first;};

   std::sort(v1.begin(),v1.end(),sortingFunction); // FUNDAMENTAL!
   std::sort(v2.begin(),v2.end(),sortingFunction); // FUNDAMENTAL!

   return IsSame(v1,v2);
}

template<class T, class U>
bool IsSame(const std::unordered_multimap<T,U>& a, const std::unordered_multimap<T,U>& b){
   std::vector<std::pair<T,U>> v1;
   for (auto& el : a) v1.emplace_back(el);
   std::vector<std::pair<T,U>> v2;
   for (auto& el : b) v2.emplace_back(el);

   auto sortingFunction = [](const std::pair<T,U>& p1,const std::pair<T,U>& p2){return p1.first<p2.first;};

   std::sort(v1.begin(),v1.end(),sortingFunction); // FUNDAMENTAL!
   std::sort(v2.begin(),v2.end(),sortingFunction); // FUNDAMENTAL!

   return IsSame(v1,v2);
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
   gVerboseComparison = true;
   writeToFile(obj,objName,filename);
   readAndCheckFromFile(obj,objName,filename);
   gVerboseComparison = false;
}

template<class Cont>
void fillHistoCont(Cont& cont, unsigned int n=5000){
   for (auto& h:cont) h.FillRandom("gaus",n);
}

template<class Cont>
void fillHistoAssoCont(Cont& cont, unsigned int n=5000){
   using contPair_t = std::pair<typename Cont::key_type,typename Cont::mapped_type>;
   std::vector<contPair_t> v;
   for (auto& el : cont) v.emplace_back(el);
   auto sortingFunction = [](const contPair_t& p1,const contPair_t& p2){return p1.first<p2.first;};
   std::sort(v.begin(),v.end(),sortingFunction); // FUNDAMENTAL!
   cont.clear();
   for (auto& h:v) {
      h.second.FillRandom("gaus",n);
      cont.insert(h);
   }

}

template<class NestedCont>
void fillHistoNestedCont(NestedCont& nestedCont, unsigned int n=5000){
   for (auto& hCont:nestedCont) {
      fillHistoCont(hCont,n);
   }
}

template<class NestedCont>
void fillHistoNestedAssoCont(std::vector<NestedCont>& nestedCont, unsigned int n=5000){
   for (auto& hCont:nestedCont) {
      fillHistoAssoCont(hCont,n);
   }
}

template<class NestedCont>
void fillHistoNestedAssoCont(NestedCont& nestedCont, unsigned int n=5000){
   using contPair_t = std::pair<typename NestedCont::key_type,typename NestedCont::mapped_type>;
   std::vector<contPair_t> v;
   for (auto& el : nestedCont) v.emplace_back(el);
   auto sortingFunction = [](const contPair_t& p1,const contPair_t& p2){return p1.first<p2.first;};
   std::sort(v.begin(),v.end(),sortingFunction); // FUNDAMENTAL!
   nestedCont.clear();
   for (auto& hCont:v){
      fillHistoCont(hCont.second,n);
      nestedCont.insert(hCont);
   }
}


template<class Cont>
void randomizeCont(Cont& cont){
   for (auto& el : cont){
      el*=gRandom->Uniform(1,2);
   }
}

template<class Cont>
void randomizeAssoCont(Cont& cont){
   using contPair_t = std::pair<typename Cont::key_type,typename Cont::mapped_type>;
   std::vector<contPair_t> v;
   for (auto& el : cont) v.emplace_back(el);
   auto sortingFunction = [](const contPair_t& p1,const contPair_t& p2){return p1.first<p2.first;};
   std::sort(v.begin(),v.end(),sortingFunction); // FUNDAMENTAL!
   cont.clear();
   for (auto& el : v){
      cont.insert(std::make_pair(el.first, el.second*gRandom->Uniform(1,2)));
   }
}

//------------------------------------------------------------------------------
// For the unordered set
namespace std {
   template <>
   struct hash<TH1F> {
   public:
      size_t operator()(const TH1F &h) const {
         std::hash<std::string> shash;
         return shash(h.GetName());
      }
   }; // hash
   template <>
   struct hash<std::vector<TH1F>> {
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

template<class T, class HASH, class EQ, class ALLOC>
void fillHistoCont(std::unordered_multiset<T,HASH,EQ,ALLOC>& cont, unsigned int n=5000){
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
void fillHistoNestedCont(std::unordered_multiset<T,HASH,EQ,ALLOC>& nestedCont, unsigned int n=5000){
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
void randomizeCont(std::unordered_multiset<T,HASH,EQ,ALLOC>& cont){
   std::vector<T> v;
   for (auto& el : cont) v.emplace_back(el);
   std::sort(v.begin(),v.end()); // FUNDAMENTAL!
   cont.clear();
   for (auto& el : v){
      cont.insert(el*gRandom->Uniform(1,2));
   }
}


#endif
