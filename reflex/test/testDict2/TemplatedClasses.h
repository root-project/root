
namespace TT {

  template < class T > class A { };

  template < class T > class Outer {
  public:
    int wantedFunction() { return 99; }
    int unwantedFunction() { return 66; }
  };

} // namespace TT

namespace {
  struct _Instances {
    TT::Outer<TT::A<unsigned long> > m1; 
  };
}
