#ifndef RowWiseNoProxyWarning_h
#define RowWiseNoProxyWarning_h

#include <vector>
#include <list>
#include <map>
#include <set>

class CustomClass{
public:
   CustomClass():fI(0){};
   CustomClass(int i):fI(i){};
   bool operator< (const CustomClass & other)const {
      return fI < other.fI;
   }
private:
   int fI;
};
class CustomClass2{
public:
   CustomClass2():fI(0){};
   CustomClass2(int i):fI(i){};
   bool operator< (const CustomClass2 & other)const {
      return fI < other.fI;
   }
private:
   int fI;
};

#endif
