#include "Reflex/PluginService.h"
#include "ICommon.h"
#include <iostream>
#include <string>

using namespace std;
using namespace ROOT::Reflex;

template <class T, class V>
struct MyTClass: virtual public ICommon {
   MyTClass(): m_f(99) { if (PluginService::Debug()) { cout << "MyTClass constructed" << endl; } }

   MyTClass(T f): m_f(f) { if (PluginService::Debug()) { cout << "MyTClass constructed with T = " << f << endl; } }

   MyTClass(T f, V v): m_f(f)
   { if (PluginService::Debug()) { cout << "MyTClass constructed with T and V = [" << f << " " << v << "]" << endl; } }

   ~MyTClass() { if (PluginService::Debug()) { cout << "MyTClass destructed" << endl; } }

   string
   do_something(void) { return "MyTClass doing something"; }

   int
   do_nothing(int i) { return i; }

   double
   get_f() { return m_f; }

   T m_f;
};

struct MyClassConst: public ICommon {
   MyClassConst(const ICommon* c): m_f(0) { if (PluginService::Debug()) { cout << "MyClassConst constructed with c = " << c << endl; } }

   ~MyClassConst() { if (PluginService::Debug()) { cout << "MyClassConst destructed" << endl; } }

   string
   do_something(void) { return "MyClassConst doing something"; }

   int
   do_nothing(int i) { return i; }

   double
   get_f() { return m_f; }

   double m_f;
};

typedef MyTClass<double, int> t1;
typedef MyTClass<int, string> t2;
PLUGINSVC_FACTORY(t1, ICommon * (void));
PLUGINSVC_FACTORY(t1, ICommon * (double));
PLUGINSVC_FACTORY(t2, ICommon * (int, string));

PLUGINSVC_FACTORY(MyClassConst, ICommon * (const ICommon*));
