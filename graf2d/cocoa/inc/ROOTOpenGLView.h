// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov   26/04/2012

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_ROOTOpenGLView
#define ROOT_ROOTOpenGLView

#include "QuartzWindow.h"

///////////////////////////////////////////
//                                       //
// OpenGL view's class.                  //
//                                       //
///////////////////////////////////////////

@interface ROOTOpenGLView : QuartzView {
@private
   // Explicit i-vars are required for 32-bit build.
   NSOpenGLContext *fOpenGLContext;
   BOOL fUpdateContext;
   //
   NSOpenGLPixelFormat *fPixelFormat;
}

- (id) initWithFrame : (NSRect) frameRect pixelFormat : (NSOpenGLPixelFormat *) format;
- (void) dealloc;

//GL-view does not own GL-context, different GL contexts can be attached to the same view
//(though ROOT never does this). View has to know about GL-context only to notify it about
//geometry changes (calls -update method) and to clear drawable in a -dealloc method.

@property (nonatomic, retain) NSOpenGLContext *fOpenGLContext;

//ROOT's GL uses pixel format (TGLFormat class) when TGLWidget is
//created, after that, pixel format never changed (though I can do
//this with ROOTOpenGLView, there is no interface in ROOT's GL code for this).
//So, pixel format is a property of ROOTOpenGLView. GL-view owns pixel format,
//it can also be reset externally (again, GL module never does this).
//Later, when creating GL-context, this pixel format is used (and
//ROOT creates GL-context per GL-widget, thus using pixel format from a widget.

- (NSOpenGLPixelFormat *) pixelFormat;
- (void) setPixelFormat : (NSOpenGLPixelFormat *) pixelFormat;

//View's geometry is updated by ROOT's GUI, but view
//can be hidden at the moment (for example, tab with GL-view is not active
//at the moment). If so, when view is visible again, context must
//be notified about changes in a drawable's geometry.
@property (nonatomic, assign) BOOL fUpdateContext;

- (BOOL) fIsOpenGLWidget;

//X11Window protocol.

@property (nonatomic, retain) QuartzPixmap *fBackBuffer;//nil.

- (void) mapWindow;
- (void) mapSubwindows;
- (void) configureNotifyTree;
- (BOOL) fIsOverlapped;
- (void) setOverlapped : (BOOL) overlap;

@end

namespace ROOT {
namespace MacOSX {
namespace OpenGL {

bool GLViewIsValidDrawable(ROOTOpenGLView *glView);

}
}
}

#endif
