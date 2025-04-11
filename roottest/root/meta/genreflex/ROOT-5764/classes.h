#include <list>

class MyClass{
public:
   MyClass(int i):m_i(i){};
private:
   int m_i;
   MyClass(){};
};

typedef std::list<MyClass> MyClassList;
MyClassList lmc;
