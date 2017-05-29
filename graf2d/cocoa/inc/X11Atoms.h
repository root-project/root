#ifndef ROOT_X11Atoms
#define ROOT_X11Atoms

#include <string>
#include <vector>
#include <map>

#include "GuiTypes.h"

namespace ROOT {
namespace MacOSX {
namespace X11 {

typedef std::map<std::string, Atom_t> name_to_atom_map;

void InitWithPredefinedAtoms(name_to_atom_map &nameToAtom, std::vector<std::string> &atomNames);

}
}
}

#endif
