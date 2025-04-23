#include <iostream.h>

class A {};
class B : public A {};
class D : public A {};
class E : public B {};
class C {};

class defaultBehavior {
public:
  virtual void print() const { cerr << "default" << endl; }
};

#include <typeinfo>

template <class T> class A_Behavior : public defaultBehavior {
public:
  virtual void print() const { cerr << "A" << " and " << typeid( T() ).name() <<endl; }
};

template <> class A_Behavior<B> : public defaultBehavior {
public:
  virtual void print() const { cerr << "B" <<  " and " << typeid( B() ).name() << endl; }
};


// template <class RootClass> defaultBehavior grabBehavior( void*, RootClass* ) { return defaultBehavior(); }
inline defaultBehavior grabBehavior( void* parent_type, void* actual_type ) { return defaultBehavior(); }

template <class T> class init {
public:
  init(T*) {};

  static const defaultBehavior & action; 
  static void run()  {
    action.print();
  }
};

template <class T> const defaultBehavior & init<T>::action = grabBehavior( (T*)0x0, (T*)0x0 );
