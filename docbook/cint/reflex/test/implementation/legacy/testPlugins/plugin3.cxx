#include "Reflex/PluginService.h"
#include "ICommon.h"
#include <iostream>
#include <string>

using namespace std;
using namespace ROOT::Reflex;

struct MyClassBis: virtual public ICommon {
   MyClassBis(): m_f(99.99) { if (PluginService::Debug()) { cout << "MyClassBis constructed" << endl; } }

   ~MyClassBis() { if (PluginService::Debug()) { cout << "MyClassBis destructed" << endl; } }

   string
   do_something(void) { return "MyClassBis doing something"; }

   int
   do_nothing(int i) { return i; }

   double
   get_f() { return m_f; }

   double m_f;
};

PLUGINSVC_FACTORY_WITH_ID(MyClassBis, ID(7, 7), ICommon * (void));
PLUGINSVC_FACTORY_WITH_ID(MyClassBis, ID(7, 8), ICommon * (void));
