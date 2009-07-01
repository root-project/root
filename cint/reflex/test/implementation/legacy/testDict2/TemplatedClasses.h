#include <string>
#include <map>


namespace TT {
template <class T> class A {};

template <class T> class Outer {
public:
   int
   wantedFunction() { return 99; }

   int
   unwantedFunction() { return 66; }

};

class TemplatedMemberTypes {
   TemplatedMemberTypes(): m0("bla"),
      m1(&m0),
      m2(m1) {}

private:
   std::string m0;
   std::string* m1;
   std::string*& m2;
   std::string m3[5];
   std::map<std::string, double>* m4;
   std::map<std::string, double> m5[5];
   //std::map<std::string,double> * (&m6)[5]; FIXME: Error in dictionary shadow class generation

};

} // namespace TT


namespace {
struct _Instances {
   TT::Outer<TT::A<unsigned long> > m1;
};
}
