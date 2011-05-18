#include "Reflex/PluginService.h"
#include "ICommon.h"
#include <iostream>
#include <string>

using namespace std;
using ROOT::Reflex::PluginService;

namespace MyNS {
class Another: public Base,
   virtual public ICommon {
public:
   Another(float f): m_f(f) { if (PluginService::Debug()) { cout << "Another constructed with float = " << f << endl; } }

   Another(double f, ICommon* c): m_f(f) { if (PluginService::Debug()) { cout << "Another constructed with ICommon = " << c << endl; } }

   Another(const string& s, double* d): m_f(*d) { if (PluginService::Debug()) { cout << "Another constructed with const string& = " << s << " d = " << *d << endl; } }

   ~Another() { if (PluginService::Debug()) { cout << "Another destructed" << endl; } }

   string
   do_something(void) { return "Another do_something"; }

   int
   do_nothing(int i) { return i; }

   double
   get_f() { return m_f; }

private:
   double m_f;
};
}

using MyNS::Another;
PLUGINSVC_FACTORY(Another, ICommon * (string, double*));
PLUGINSVC_FACTORY(Another, ICommon * (float));
PLUGINSVC_FACTORY(Another, Base * (double, ICommon*));
