#include <map>

class Transient1
{
public:
   Transient1(int) {}
};

class Transient2 : Transient1
{
public:
   Transient2(int) : Transient1(1),fData(0) {} // ,fMember(std::make_pair(Transient1(1),1)) {}
   Transient2() :  Transient1(1),fData(0) {} // ,fMember(std::make_pair(Transient1(1),1)) {}
 
   Transient1 fData;
   std::map<Transient1,long> fMember;
};


class MyClass {
public:
   Transient2 fMember; //!

   MyClass() : fMember(0) {}

};

