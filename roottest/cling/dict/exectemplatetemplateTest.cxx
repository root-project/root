#include <TClass.h>

template <class T> class AnInnocentTemplate{};
template <template <class T> class C> class Wooha{};
Wooha<AnInnocentTemplate> b;

#ifdef __MAKECLING__
#pragma link C++ class Wooha<AnInnocentTemplate>;
#endif

int exectemplatetemplateTest() {
  if (!TClass::GetClass("Wooha<AnInnocentTemplate>")) {
    printf("ERROR: cannot find TClass for Wooha<AnInnocentTemplate>\n");
    return 1;
  }
  return 0;
}
