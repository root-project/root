#include "Reflex/PluginService.h"
#include "ICommon.h"
#include <iostream>
#include <string>

using namespace std;
using namespace ROOT::Reflex;

struct MyClass: virtual public ICommon {
   MyClass(): m_f(99.99) { if (PluginService::Debug()) { cout << "MyClass constructed" << endl; } }

   MyClass(float f): m_f(f) { if (PluginService::Debug()) { cout << "MyClass constructed with float = " << f << endl; } }

   MyClass(double f, ICommon* c): m_f(f) { if (PluginService::Debug()) { cout << "MyClass constructed with ICommon = " << c << endl; } }

   MyClass(const string& s, double* d): m_f(*d) { if (PluginService::Debug()) { cout << "MyClass constructed with const string& = " << s << " d = " << *d << endl; } }

   ~MyClass() { if (PluginService::Debug()) { cout << "MyClass destructed" << endl; } }

   string
   do_something(void) { return "MyClass doing something"; }

   int
   do_nothing(int i) { return i; }

   double
   get_f() { return m_f; }

   double m_f;
};

PLUGINSVC_FACTORY(MyClass, ICommon * (void));
PLUGINSVC_FACTORY(MyClass, ICommon * (string, double*));
PLUGINSVC_FACTORY(MyClass, ICommon * (float));
PLUGINSVC_FACTORY(MyClass, ICommon * (double, ICommon*));

PLUGINSVC_FACTORY_WITH_ID(MyClass, ID(2, 5), ICommon * (void));
