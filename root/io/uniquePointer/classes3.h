#include <memory>
#include <vector>
#include <list>
#include <set>
#include <map>

class Class01{
   vector<unique_ptr<int>> a0;
   vector<unique_ptr<double>> a1;
};

class  TrackingRegion{};

namespace edm {
   template <class T>
   class Wrapper{};
}

namespace DummyNS {
   vector<unique_ptr<int>> a0;
   vector<unique_ptr<double>> a1;
   vector<list<double>*> a2;
   set<vector<unique_ptr<int>>> a3;
   map<char, vector<unique_ptr<list<double>>>> a4;
   set<vector<unique_ptr<Class01>>> a5;
   map<char, vector<unique_ptr<list<Class01>>>> a6;
   set<vector<unique_ptr<double>>> a7;
   set<vector<unique_ptr<Class01>>> a8;
   map<char,unique_ptr<list<Class01>>> a9;
   vector<unique_ptr<TrackingRegion>> a10;
}

class Class02{
   map<char,unique_ptr<list<Class01>>> a9;
};
