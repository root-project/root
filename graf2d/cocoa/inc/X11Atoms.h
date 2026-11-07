#ifndef ROOT_X11Atoms
#define ROOT_X11Atoms

#include <string>
#include <vector>
#include <map>

#include "GuiTypes.h"

namespace ROOT {
namespace MacOSX {
namespace X11 {


void InitWithPredefinedAtoms(std::map<std::string, Atom_t> &nameToAtom, std::vector<std::string> &atomNames);

}
}
}

#endif
