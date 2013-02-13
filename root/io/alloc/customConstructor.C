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

#include "Rtypeinfo.h"
#include <vector>

class WithTemplate {
public:
   WithTemplate() { cout << "Called WithTemplate::WithTemplate()\n"; }
   template <class T> WithTemplate(T*) {
      cout << "Called WithTemplate::WithTemplate<T>() T=" << typeid(T*).name() << "\n"; 
   }
   template <class R> WithTemplate(std::vector<R>*) {
   }
};
