#ifndef __theClasses_h__
#define __theClasses_h__

#include <vector>

class classWithDictionary1{
public:
   classWithDictionary1(int i):fI(i){};
   classWithDictionary1():fI(0){};
   int GetI() const {return fI;}
   void SetI(int i){fI=i;}
   bool operator==(const classWithDictionary1 &other) const {
      return fI == other.fI;
   }
   bool operator!=(const classWithDictionary1 &other) const {
      return fI != other.fI;
   }
private:
int fI;
};

class classWithoutDictionary1{
public:
   classWithoutDictionary1(int i):fI(i){};
   classWithoutDictionary1():fI(0){};
   int GetI() const {return fI;}
   void SetI(int i){fI=i;}
   bool operator==(const classWithoutDictionary1 &other) const {
      return fI == other.fI;
   }
   bool operator!=(const classWithoutDictionary1 &other) const {
      return fI != other.fI;
   }
private:
int fI;
};

class classWithDictionary2{
public:
   classWithDictionary2(int i):fI(i){};
   classWithDictionary2():fI(0){};
   int GetI() const  {return fI;}
   void SetI(int i){fI=i;}
   bool operator==(const classWithDictionary2 &other) const {
      return fI == other.fI;
   }
   bool operator!=(const classWithDictionary2 &other) const {
      return fI != other.fI;
   }
private:
int fI;
};

class classWithoutDictionary2{
public:
   classWithoutDictionary2(int i):fI(i){};
   classWithoutDictionary2():fI(0){};
   int GetI() const {return fI;}
   void SetI(int i){fI=i;}
   bool operator==(const classWithoutDictionary2 &other) const {
      return fI == other.fI;
   }
   bool operator!=(const classWithoutDictionary2 &other) const {
      return fI != other.fI;
   }
private:
int fI;
};

#endif
