#include "Riostream.h"
#include "Rtypes.h"

using namespace std;

class G4_ROOT_IO;

class None {};

class Regular {
public:
   Regular() {cout << "Called Regular::Regular()\n";};
};

class Private {
private:
   Private();
};

class G4 {
private:
   G4();
public:
   G4(G4_ROOT_IO*) { cout << "Called G4::G4(G4_ROOT_IO*)\n"; }
};

// G4_v2 test ROOT-7723
class __void__;
class G4_v2 {
private:
   G4_v2();
public:
   G4_v2(__void__&) { cout << "Called G4_v2::G4_v2(__void__&)\n"; }
};

class OTHER_ROOT_IO;
class Other {
   private:
   Other();
public:
   Other(G4_ROOT_IO*) { cout << "Called Other::Other(G4_ROOT_IO*)\n"; }
   Other(OTHER_ROOT_IO*) { cout << "Called Other::Other(OTHER_ROOT_IO*)\n"; }
};

class Default {
public:
   Default() { cout << "Called Default::Default()\n"; }
   Default(TRootIOCtor*) { cout << "Called Default::Default(TRootIOCtor*)\n"; }
};

typedef TRootIOCtor MyCtor;

class Typedef {
public:
   Typedef() { cout << "Called Typedef::Typedef()\n"; }
   Typedef(MyCtor*) { cout << "Called Typedef::Typedef(MyCtor*) typedefed from TRootIoCtor\n"; }
};

class Pointers {
public:
   Pointers() { cout << "Called Pointers::Pointers()\n"; }
   Pointers(Pointers*) { cout << "Called Pointers::Pointers(Pointers*)\n"; }
};

class WithTemplate {
public:
   WithTemplate() { cout << "Called WithTemplate::WithTemplate()\n"; }
   template <class T> WithTemplate(T*) {
      cout << "Called WithTemplate::WithTemplate<T>() T=" << typeid(T*).name() << "\n"; 
   }
};
