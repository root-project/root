#include <map>

class Value {
public:
  int fA;
  int fB;
};

class Transient {
public:
  std::map<int,Value> fMap;
}; 
