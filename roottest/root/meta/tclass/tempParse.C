#include <string>

template <class T> class Verifier {
public:
  Verifier() {}
  ~Verifier() {}
};

template <class T, class V=Verifier<T> > class SimpleProperty {
public: 
  SimpleProperty() {}
  SimpleProperty(const SimpleProperty& p) { m_value = p.m_value; }
  SimpleProperty( const T& v ) { m_value = v; } 
  ~SimpleProperty() {}
private:
  T m_value;   
};

#ifdef __MAKECINT__
#pragma link C++ class SimpleProperty<std::string>;
#endif





