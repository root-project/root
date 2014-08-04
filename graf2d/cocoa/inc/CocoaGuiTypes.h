#ifndef ROOT_CocoaGuiTypes
#define ROOT_CocoaGuiTypes

//This file extends ROOT's GuiTypes.h with additional types I need - Point/Rectangle which can use integers (not short integers).
//To be used in copy:xxxxx methods of X11Drawables and somewhere else.
//It's a shame I have to write such "classes" :))

namespace ROOT {
namespace MacOSX {
namespace X11 {

struct Point {
   int fX;
   int fY;

   Point();
   Point(int x, int y);
};

struct Rectangle {
   int fX;
   int fY;

   unsigned fWidth;
   unsigned fHeight;

   Rectangle();
   Rectangle(int x, int y, unsigned w, unsigned h);
};

}//X11
}//MacOSX
}//ROOT

#endif
