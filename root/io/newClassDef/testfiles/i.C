#include <iostream.h>

class A {};
class B : public A {};
class C {};

class defaultBehavior {
public:
  virtual void print() const { cerr << "default" << endl; }
};
class A_Behavior : public defaultBehavior {
public:
  virtual void print() const { cerr << "A" << endl; }
};

defaultBehavior grabBehavior( void* ) { return defaultBehavior(); }

template <class T> class init {
public:
  init(T*) {};

  static const defaultBehavior & action; 
  static void run()  {
    action.print();
  }
};

template <class T> const defaultBehavior & init<T>::action( grabBehavior( (T*)0x0 ) );

template <class T, class B> class test {};

int main() {

  init<A>::run();
  init<B>::run();
  init<C>::run();

  //  init i1( ((A*)0x0) );
  //  test<A,  grabBehavior( (A*)0x0 ) > a;
  
  return 0;
}

A_Behavior grabBehavior( const A*) { return A_Behavior(); }

