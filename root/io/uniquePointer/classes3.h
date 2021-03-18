#include <memory>
#include <vector>
#include <list>
#include <set>
#include <map>

class Class01{
   std::vector<std::unique_ptr<int>> a0;
   std::vector<std::unique_ptr<double>> a1;
};

class  TrackingRegion{};

namespace edm {
   template <class T>
   class Wrapper{};
}

namespace DummyNS {
   std::vector<std::unique_ptr<int>> a0;
   std::vector<std::unique_ptr<double>> a1;
   std::vector<std::list<double>*> a2;
   std::set<std::vector<std::unique_ptr<int>>> a3;
   std::map<char, std::vector<std::unique_ptr<std::list<double>>>> a4;
   std::set<std::vector<std::unique_ptr<Class01>>> a5;
   std::map<char, std::vector<std::unique_ptr<std::list<Class01>>>> a6;
   std::set<std::vector<std::unique_ptr<double>>> a7;
   std::set<std::vector<std::unique_ptr<Class01>>> a8;
   std::map<char,std::unique_ptr<std::list<Class01>>> a9;
   std::vector<std::unique_ptr<TrackingRegion>> a10;
}

class Class02{
   std::map<char,std::unique_ptr<std::list<Class01>>> a9;
};
