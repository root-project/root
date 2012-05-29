// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov   29/11/2011

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


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

///////////////////////////////////////////////
//                                           //
// CocoaPrivate. Hidden implementation       //
// details for TGCocoa.                      //
//                                           //
///////////////////////////////////////////////

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
   
   CocoaPrivate(const CocoaPrivate &rhs);
   CocoaPrivate &operator = (const CocoaPrivate &rhs);

   unsigned               RegisterDrawable(NSObject *nsObj);
   NSObject<X11Drawable> *GetDrawable(unsigned drawableD)const;
   NSObject<X11Window>   *GetWindow(unsigned windowID)const;
   void                   DeleteDrawable(unsigned drawableID);
   
   ULong_t                RegisterGLContextForView(unsigned viewID);
   NSObject<X11Window>   *GetWindowForGLContext(Handle_t glContextID);
   
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
   std::map<unsigned, Util::NSStrongReference<NSObject<X11Drawable> > > fDrawables;
   typedef std::map<unsigned, Util::NSStrongReference<NSObject<X11Drawable> > >::iterator drawable_iterator;
   typedef std::map<unsigned, Util::NSStrongReference<NSObject<X11Drawable> > >::const_iterator const_drawable_iterator;
   
   std::map<ULong_t, unsigned> fGLContextMap;
   ULong_t fFreeGLContextID;
};

}//Details
}//MacOSX
}//ROOT

#endif
