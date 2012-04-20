#ifndef ROOT_CocoaPrivate
#define ROOT_CocoaPrivate

#include <vector>
#include <map>

#ifndef ROOT_CocoaUtils
#include "CocoaUtils.h"
#endif
#ifndef ROOT_X11Colors
#include "X11Colors.h"
#endif
#ifndef ROOT_X11Events
#include "X11Events.h"
#endif
#ifndef ROOT_X11Buffer
#include "X11Buffer.h"
#endif
#ifndef ROOT_GuiTypes
#include "GuiTypes.h"
#endif
#ifndef ROOT_GuiFonts
#include "FontCache.h"
#endif

@protocol X11Drawable;
@protocol X11Window;

@class NSObject;

class TGQuartz;
class TGCocoa;

namespace ROOT {
namespace MacOSX {

namespace Details {

class CocoaPrivate {
   friend class TGCocoa;
   friend class TGQuartz;
   friend class X11::CommandBuffer;
public:
   ~CocoaPrivate();
private:
   CocoaPrivate();
   
   int GetRootWindowID()const;
   bool IsRootWindow(int wid)const;
   
   CocoaPrivate(const CocoaPrivate &rhs) = delete;
   CocoaPrivate &operator = (const CocoaPrivate &rhs) = delete;

   unsigned               RegisterDrawable(NSObject *nsObj);
   NSObject<X11Drawable> *GetDrawable(unsigned drawableD)const;
   NSObject<X11Window>   *GetWindow(unsigned windowID)const;
   void                   DeleteDrawable(unsigned drawableID);
   
   //This function resets strong reference, if you still want NSObject for drawableID to live,
   //you have to retain the pointer (probably) and also drawableID will become id for nsObj (replacement).
   void               ReplaceDrawable(unsigned drawableID, NSObject *nsObj);

   //Color "parser": either parse string like "#ddeeaa", or
   //search rgb.txt like table for named color.
   X11::ColorParser                            fX11ColorParser;
   //Event translator, converts Cocoa events into X11 events
   //and generates X11 events.
   X11::EventTranslator                        fX11EventTranslator;
   //Command buffer - for "buffered" drawing commands.
   X11::CommandBuffer                          fX11CommandBuffer;
   //Font manager - cache CTFontRef for GUI.
   FontCache                                   fFontManager;

   //Id for the new registered drawable.
   unsigned                                    fCurrentDrawableID;
   //Cache of ids.
   std::vector<unsigned>                       fFreeDrawableIDs;
   //Cocoa objects (views, windows, "pixmaps").
   //As soon as NSObject<X11Drawable> is not a C++, even in C++0x11 I need a space :(
   std::map<unsigned, Util::NSStrongReference<NSObject<X11Drawable> >> fDrawables;
};

}
}
}

#endif
