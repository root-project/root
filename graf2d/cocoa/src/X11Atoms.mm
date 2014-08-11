#include "X11Atoms.h"

namespace ROOT {
namespace MacOSX {
namespace X11 {

//Predefined atoms.
const char *predefinedAtoms[] =
{
   "XA_PRIMARY",
   "XA_SECONDARY",
   "XA_ARC",
   "XA_ATOM",
   "XA_BITMAP",
   "XA_CARDINAL",
   "XA_COLORMAP",
   "XA_CURSOR",
   "XA_CUT_BUFFER0",
   "XA_CUT_BUFFER1",
   "XA_CUT_BUFFER2",
   "XA_CUT_BUFFER3",
   "XA_CUT_BUFFER4",
   "XA_CUT_BUFFER5",
   "XA_CUT_BUFFER6",
   "XA_CUT_BUFFER7",
   "XA_DRAWABLE",
   "XA_FONT",
   "XA_INTEGER",
   "XA_PIXMAP",
   "XA_POINT",
   "XA_RECTANGLE",
   "XA_RESOURCE_MANAGER",
   "XA_RGB_COLOR_MAP",
   "XA_RGB_BEST_MAP",
   "XA_RGB_BLUE_MAP",
   "XA_RGB_DEFAULT_MAP",
   "XA_RGB_GRAY_MAP",
   "XA_RGB_GREEN_MAP",
   "XA_RGB_RED_MAP",
   "XA_STRING",
   "XA_VISUALID",
   "XA_WINDOW",
   "XA_WM_COMMAND",
   "XA_WM_HINTS",
   "XA_WM_CLIENT_MACHINE",
   "XA_WM_ICON_NAME",
   "XA_WM_ICON_SIZE",
   "XA_WM_NAME",
   "XA_WM_NORMAL_HINTS",
   "XA_WM_SIZE_HINTS",
   "XA_WM_ZOOM_HINTS",
   "XA_MIN_SPACE",
   "XA_NORM_SPACE",
   "XA_MAX_SPACE",
   "XA_END_SPACE",
   "XA_SUPERSCRIPT_X",
   "XA_SUPERSCRIPT_Y",
   "XA_SUBSCRIPT_X",
   "XA_SUBSCRIPT_Y",
   "XA_UNDERLINE_POSITION",
   "XA_UNDERLINE_THICKNESS",
   "XA_STRIKEOUT_ASCENT",
   "XA_STRIKEOUT_DESCENT",
   "XA_ITALIC_ANGLE",
   "XA_X_HEIGHT",
   "XA_QUAD_WIDTH",
   "XA_WEIGHT",
   "XA_POINT_SIZE",
   "XA_RESOLUTION",
   "XA_COPYRIGHT",
   "XA_NOTICE",
   "XA_FONT_NAME",
   "XA_FAMILY_NAME",
   "XA_FULL_NAME",
   "XA_CAP_HEIGHT",
   "XA_WM_CLASS",
   "XA_WM_TRANSIENT_FOR"
};

const unsigned nPredefined = sizeof predefinedAtoms / sizeof predefinedAtoms[0];

//______________________________________________________________________________
void InitWithPredefinedAtoms(name_to_atom_map &nameToAtom, std::vector<std::string> &atomNames)
{
   nameToAtom.clear();

   for (unsigned i = 0; i < nPredefined; ++i)
      nameToAtom[predefinedAtoms[i]] = i + 1;

   atomNames.assign(predefinedAtoms, predefinedAtoms + nPredefined);
}

}//X11
}//MacOSX
}//ROOT
