// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov   29/11/2011

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#define DEBUG_ROOT_COCOA

//#define NDEBUG

#ifdef DEBUG_ROOT_COCOA
#include <algorithm>
#endif

#include <stdexcept>
#include <cassert>

#include <Cocoa/Cocoa.h>

#include "CocoaPrivate.h"
#include "QuartzWindow.h"
#include "CocoaUtils.h"

namespace ROOT {
namespace MacOSX {
namespace Details {

//______________________________________________________________________________
CocoaPrivate::CocoaPrivate()
               : fCurrentDrawableID(GetRootWindowID() + 1), //Any real window has id > rootID.
                                                            //0 is also used by some X11 functions as None.
                 fFreeGLContextID(1)
{
   //Init NSApplication, if it was not done yet.
   Util::AutoreleasePool pool;
   [NSApplication sharedApplication];
}

//______________________________________________________________________________
CocoaPrivate::~CocoaPrivate()
{
}

//______________________________________________________________________________
int CocoaPrivate::GetRootWindowID()const
{
   //First I had root ID == 0, but this is None in X11 and
   //it can be used by ROOT, for example, I had trouble with
   //gClient able to found TGWindow for None - crash!
   return 1;
}

//______________________________________________________________________________
bool CocoaPrivate::IsRootWindow(int wid)const
{
   return wid == GetRootWindowID();
}

//______________________________________________________________________________
unsigned CocoaPrivate::RegisterDrawable(NSObject *nsObj)
{
   //Return integer identifier for a new "drawable" (like in X11)
   unsigned newID = fCurrentDrawableID;

   if (fFreeDrawableIDs.size()) {
      newID = fFreeDrawableIDs.back();
      fFreeDrawableIDs.pop_back();
   } else
      fCurrentDrawableID++;

   assert(fDrawables.find(newID) == fDrawables.end() && "RegisterDrawable, id for new drawable is still in use");

   fDrawables[newID] = nsObj;

   return newID;
}

//______________________________________________________________________________
NSObject<X11Drawable> *CocoaPrivate::GetDrawable(unsigned drawableID)const
{
   const_drawable_iterator drawableIter = fDrawables.find(drawableID);
   
#ifdef DEBUG_ROOT_COCOA
   if (drawableIter == fDrawables.end()) {
      NSLog(@"Fatal error: requested non-existing drawable %u", drawableID);
      //We do not care about efficiency, ROOT's gonna die on assert :)
      std::vector<unsigned>::const_iterator deletedDrawable = std::find(fFreeDrawableIDs.begin(), fFreeDrawableIDs.end(), drawableID);
      if (deletedDrawable != fFreeDrawableIDs.end()) {
         NSLog(@"This drawable was deleted already");
      } else {
         NSLog(@"This drawable not found among allocated/deleted drawables");
      }
   }
#endif
   assert(drawableIter != fDrawables.end() && "GetDrawable, non-existing drawable requested");
   return drawableIter->second.Get();
}

//______________________________________________________________________________
NSObject<X11Window> *CocoaPrivate::GetWindow(unsigned windowID)const
{
   const_drawable_iterator winIter = fDrawables.find(windowID);
#ifdef DEBUG_ROOT_COCOA
   if (winIter == fDrawables.end()) {
      NSLog(@"Fatal error: requested non-existing drawable %u", windowID);
      //We do not care about efficiency, ROOT's gonna die on assert :)
      std::vector<unsigned>::const_iterator deletedDrawable = std::find(fFreeDrawableIDs.begin(), fFreeDrawableIDs.end(), windowID);
      if (deletedDrawable != fFreeDrawableIDs.end()) {
         NSLog(@"This window was deleted already");
      } else {
         NSLog(@"This window not found among allocated/deleted drawables");
      }
   }
#endif
   assert(winIter != fDrawables.end() && "GetWindow, non-existing window requested");
   return (NSObject<X11Window> *)winIter->second.Get();
}

//______________________________________________________________________________
void CocoaPrivate::DeleteDrawable(unsigned drawableID)
{
   drawable_iterator drawableIter = fDrawables.find(drawableID);
   assert(drawableIter != fDrawables.end() && "DeleteDrawable, non existing drawableID");

   //Probably, I'll need some additional cleanup here later. Now just delete NSObject and
   //reuse its id.
   NSObject<X11Drawable> *base = drawableIter->second.Get();
   if ([base isKindOfClass : [QuartzView class]]) {
      [(QuartzView *)base removeFromSuperview];
      ((QuartzView *)base).fParentView = nil;
   } else if ([base isKindOfClass : [QuartzWindow class]]) {
      QuartzWindow *qw = (QuartzWindow *)base;
      [qw.fContentView removeFromSuperview];
      qw.fContentView.fParentView = nil;
      qw.contentView = nil;

      //Remove transient windows?
      /*
      const Util::NSScopeGuard<NSArray> children([[qw childWindows] copy]);
      for (QuartzWindow *child in children.Get()) {
         child.fMainWindow = nil;
         [qw removeChildWindow : child];
      }
      */

      if (qw.fMainWindow) {
         [qw.fMainWindow removeChildWindow : qw];
         qw.fMainWindow = nil;
      }
   }

//   fFreeDrawableIDs.push_back(drawableID);
   fDrawables.erase(drawableIter);//StrongReference should do work here.
}

//______________________________________________________________________________
ULong_t CocoaPrivate::RegisterGLContextForView(unsigned viewID)
{
   //At the moment, let's assume we attach only 1 gl context to 1 view.
   fGLContextMap[fFreeGLContextID] = viewID;
   return fFreeGLContextID++;
}

//______________________________________________________________________________
NSObject<X11Window> *CocoaPrivate::GetWindowForGLContext(Handle_t glContextID)
{
   assert(fGLContextMap.find(glContextID) != fGLContextMap.end() && "GetWindowForGLContext, bad context id");
   return GetWindow(fGLContextMap[glContextID]);
}

//______________________________________________________________________________
void CocoaPrivate::ReplaceDrawable(unsigned drawableID, NSObject *nsObj)
{
   drawable_iterator drawableIter = fDrawables.find(drawableID);
   assert(drawableIter != fDrawables.end() && "ReplaceDrawable, can not replace non existing drawable");
   drawableIter->second.Reset(nsObj);
}

}//Details
}//MacOSX
}//ROOT
