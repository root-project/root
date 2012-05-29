// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov   29/02/2012

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//#define NDEBUG

#include <stdexcept>
#include <cstring>
#include <cassert>
#include <memory>

#include <Cocoa/Cocoa.h>

#include "CocoaPrivate.h"
#include "QuartzWindow.h"
#include "QuartzPixmap.h"
#include "X11Drawable.h"
#include "X11Buffer.h"
#include "TGWindow.h"
#include "TGClient.h"
#include "TGCocoa.h"

namespace ROOT {
namespace MacOSX {
namespace X11 {

//______________________________________________________________________________
Command::Command(Drawable_t wid, const GCValues_t &gc)
            : fID(wid),
              fGC(gc)
{
}

//______________________________________________________________________________
Command::Command(Drawable_t wid)
            : fID(wid),
              fGC()
{
}

//______________________________________________________________________________
Command::~Command()
{
}

//______________________________________________________________________________
bool Command::HasOperand(Drawable_t wid)const
{
   return wid == fID;      
}

//______________________________________________________________________________
bool Command::IsGraphicsCommand()const
{
   return false;
}

//______________________________________________________________________________
DrawLine::DrawLine(Drawable_t wid, const GCValues_t &gc, const Point_t &p1, const Point_t &p2)
            : Command(wid, gc),
              fP1(p1),
              fP2(p2)
{
}

//______________________________________________________________________________
void DrawLine::Execute()const
{
   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx != 0 && "Execute, gVirtualX is either null or not of TGCocoa type");
   vx->DrawLineAux(fID, fGC, fP1.fX, fP1.fY, fP2.fX, fP2.fY);
}

//______________________________________________________________________________
DrawSegments::DrawSegments(Drawable_t wid, const GCValues_t &gc, const Segment_t *segments, Int_t nSegments)
                 : Command(wid, gc) 
{
   assert(segments != 0 && "DrawSegments, segments parameter is null");
   assert(nSegments > 0 && "DrawSegments, nSegments <= 0");
   
   fSegments.assign(segments, segments + nSegments);
}

//______________________________________________________________________________
void DrawSegments::Execute()const
{
   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx != 0 && "Execute, gVirtualX is either null or not of TGCocoa type");
   vx->DrawSegmentsAux(fID, fGC, &fSegments[0], (Int_t)fSegments.size());
}

//______________________________________________________________________________
ClearArea::ClearArea(Window_t wid, const Rectangle_t &area)
             : Command(wid),
               fArea(area)
{
}

//______________________________________________________________________________
void ClearArea::Execute()const
{
   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx != 0 && "Execute, gVirtualX is either null or not of TGCocoa type");
   vx->ClearAreaAux(fID, fArea.fX, fArea.fY, fArea.fWidth, fArea.fHeight);   
}

//______________________________________________________________________________
CopyArea::CopyArea(Drawable_t src, Drawable_t dst, const GCValues_t &gc, const Rectangle_t &area, const Point_t &dstPoint)
               : Command(dst, gc),
                 fSrc(src),
                 fArea(area),
                 fDstPoint(dstPoint)
{
}

//______________________________________________________________________________
bool CopyArea::HasOperand(Drawable_t drawable)const
{
   return fID == drawable || fSrc == drawable || fGC.fClipMask == drawable;
}

//______________________________________________________________________________
void CopyArea::Execute()const
{
   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx != 0 && "Execute, gVirtualX is either null or not of TGCocoa type");
   vx->CopyAreaAux(fSrc, fID, fGC, fArea.fX, fArea.fY, fArea.fWidth, fArea.fHeight, fDstPoint.fX, fDstPoint.fY);
}

//______________________________________________________________________________
DrawString::DrawString(Drawable_t wid, const GCValues_t &gc, const Point_t &point, const std::string &text)
               : Command(wid, gc),
                 fPoint(point),
                 fText(text)
{
}

//______________________________________________________________________________
void DrawString::Execute()const
{
   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx != 0 && "Execute, gVirtualX is either null or not of TGCocoa type");
   vx->DrawStringAux(fID, fGC, fPoint.fX, fPoint.fY, fText.c_str(), fText.length());
}

//______________________________________________________________________________
FillRectangle::FillRectangle(Drawable_t wid, const GCValues_t &gc, const Rectangle_t &rectangle)
                  : Command(wid, gc),
                    fRectangle(rectangle)
{
}

//______________________________________________________________________________
void FillRectangle::Execute()const
{
   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx != 0 && "Execute, gVirtualX is either null or not of TGCocoa type");
   vx->FillRectangleAux(fID, fGC, fRectangle.fX, fRectangle.fY, fRectangle.fWidth, fRectangle.fHeight);
}

//______________________________________________________________________________
FillPolygon::FillPolygon(Drawable_t wid, const GCValues_t &gc, const Point_t *points, Int_t nPoints)
                : Command(wid, gc)
{
   assert(points != 0 && "FillPolygon, points parameter is null");
   assert(nPoints > 0 && "FillPolygon, nPoints <= 0");
   
   fPolygon.assign(points, points + nPoints);
}

//______________________________________________________________________________   
void FillPolygon::Execute()const
{
   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx != 0 && "Execute, gVirtualX is either null or not of TGCocoa type");
   vx->FillPolygonAux(fID, fGC, &fPolygon[0], (Int_t)fPolygon.size());
}

//______________________________________________________________________________
DrawRectangle::DrawRectangle(Drawable_t wid, const GCValues_t &gc, const Rectangle_t &rectangle)
                 : Command(wid, gc),
                   fRectangle(rectangle)
{
}

//______________________________________________________________________________
void DrawRectangle::Execute()const
{
   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx != 0 && "Execute, gVirtualX is either null or not of TGCocoa type");
   vx->DrawRectangleAux(fID, fGC, fRectangle.fX, fRectangle.fY, fRectangle.fWidth, fRectangle.fHeight);
}

//______________________________________________________________________________
UpdateWindow::UpdateWindow(QuartzView *view)
                : Command(view.fID),
                  fView(view)
{
   assert(view != nil && "UpdateWindow, view parameter is nil");//view.fID will be also 0.
}

//______________________________________________________________________________
void UpdateWindow::Execute()const
{
   assert(fView.fContext != 0 && "Execute, view.fContext is null");

   if (QuartzPixmap *pixmap = fView.fBackBuffer) {
      CGImageRef image = [pixmap createImageFromPixmap];//CGBitmapContextCreateImage(pixmap.fContext);
      const CGRect imageRect = CGRectMake(0, 0, pixmap.fWidth, pixmap.fHeight);
      CGContextDrawImage(fView.fContext, imageRect, image);
      CGImageRelease(image);
   }
}

//______________________________________________________________________________
DeletePixmap::DeletePixmap(Pixmap_t pixmap)
                : Command(pixmap, GCValues_t())
{
}

//______________________________________________________________________________
void DeletePixmap::Execute()const
{
   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx != 0 && "Execute, gVirtualX is either null or not of TGCocoa type");
   vx->DeletePixmapAux(fID);
}

//______________________________________________________________________________
CommandBuffer::CommandBuffer()
{
}

//______________________________________________________________________________
CommandBuffer::~CommandBuffer()
{
   ClearCommands();
}

//______________________________________________________________________________
void CommandBuffer::AddDrawLine(Drawable_t wid, const GCValues_t &gc, Int_t x1, Int_t y1, Int_t x2, Int_t y2)
{
   try {
      Point_t p1 = {}; 
      //I'd use .fX = x1 from standard C, but ... this is already C++0x + Obj-C :)
      //So, not to make it worse :)
      p1.fX = x1;
      p1.fY = y1;
      Point_t p2 = {};
      p2.fX = x2;
      p2.fY = y2;
      std::auto_ptr<DrawLine> cmd(new DrawLine(wid, gc, p1, p2));//if this throws, I do not care.
      fCommands.push_back(cmd.get());//this can throw.
      cmd.release();
   } catch (const std::exception &) {
      throw;
   }
}

//______________________________________________________________________________
void CommandBuffer::AddDrawSegments(Drawable_t wid, const GCValues_t &gc, const Segment_t *segments, Int_t nSegments)
{
   assert(segments != 0 && "AddDrawSegments, segments parameter is null");
   assert(nSegments > 0 && "AddDrawSegments, nSegments <= 0");

   try {
      std::auto_ptr<DrawSegments> cmd(new DrawSegments(wid, gc, segments, nSegments));
      fCommands.push_back(cmd.get());
      cmd.release();
   } catch (const std::exception &) {
      throw;
   }
}

//______________________________________________________________________________
void CommandBuffer::AddClearArea(Window_t wid, Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   try {
      Rectangle_t r = {};
      r.fX = x;
      r.fY = y;
      r.fWidth = (UShort_t)w;
      r.fHeight = (UShort_t)h;
      std::auto_ptr<ClearArea> cmd(new ClearArea(wid, r));//Can throw, nothing leaks.
      fCommands.push_back(cmd.get());//this can throw.
      cmd.release();
   } catch (const std::exception &) {
      throw;
   }
}

//______________________________________________________________________________
void CommandBuffer::AddCopyArea(Drawable_t src, Drawable_t dst, const GCValues_t &gc, 
                                Int_t srcX, Int_t srcY, UInt_t width, UInt_t height, Int_t dstX, Int_t dstY)
{
   try {
      Rectangle_t area = {};
      area.fX = srcX;
      area.fY = srcY;
      area.fWidth = (UShort_t)width;
      area.fHeight = (UShort_t)height;
      Point_t dstPoint = {};
      dstPoint.fX = dstX;
      dstPoint.fY = dstY;
      std::auto_ptr<CopyArea> cmd(new CopyArea(src, dst, gc, area, dstPoint));//Can throw, nothing leaks.
      fCommands.push_back(cmd.get());//this can throw.
      cmd.release();
   } catch (const std::exception &) {
      throw;
   }
}

//______________________________________________________________________________
void CommandBuffer::AddDrawString(Drawable_t wid, const GCValues_t &gc, Int_t x, Int_t y, const char *text, Int_t len)
{
   try {
      if (len < 0)//Negative length can come from caller.
         len = std::strlen(text);
      const std::string substr(text, len);//Can throw.
      Point_t p = {};
      p.fX = x;
      p.fY = y;
      std::auto_ptr<DrawString> cmd(new DrawString(wid, gc, p, substr));//Can throw.
      fCommands.push_back(cmd.get());//can throw.
      cmd.release();
   } catch (const std::exception &) {
      throw;
   }
}

//______________________________________________________________________________
void CommandBuffer::AddFillRectangle(Drawable_t wid, const GCValues_t &gc, Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   try {
      Rectangle_t r = {};
      r.fX = x;
      r.fY = y;
      r.fWidth = (UShort_t)w;
      r.fHeight = (UShort_t)h;
      std::auto_ptr<FillRectangle> cmd(new FillRectangle(wid, gc, r));
      fCommands.push_back(cmd.get());
      cmd.release();
   } catch (const std::exception &) {
      throw;
   }
}

//______________________________________________________________________________
void CommandBuffer::AddDrawRectangle(Drawable_t wid, const GCValues_t &gc, Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   try {
      Rectangle_t r = {};
      r.fX = x;
      r.fY = y;
      r.fWidth = (UShort_t)w;
      r.fHeight = (UShort_t)h;
      std::auto_ptr<DrawRectangle> cmd(new DrawRectangle(wid, gc, r));
      fCommands.push_back(cmd.get());
      cmd.release();
   } catch (const std::exception &) {
      throw;
   }
}

//______________________________________________________________________________
void CommandBuffer::AddFillPolygon(Drawable_t wid, const GCValues_t &gc, const Point_t *polygon, Int_t nPoints)
{
   assert(polygon != 0 && "AddFillPolygon, polygon parameter is null");
   assert(nPoints > 0 && "AddFillPolygon, nPoints <= 0");
   
   try {
      std::auto_ptr<FillPolygon> cmd(new FillPolygon(wid, gc, polygon, nPoints));
      fCommands.push_back(cmd.get());
      cmd.release();
   } catch (const std::exception &) {
      throw;
   }
}

//______________________________________________________________________________
void CommandBuffer::AddUpdateWindow(QuartzView *view)
{
   assert(view != nil && "AddUpdateWindow, view parameter is nil");
   
   try {
      std::auto_ptr<UpdateWindow> cmd(new UpdateWindow(view));
      fCommands.push_back(cmd.get());
      cmd.release();
   } catch (const std::exception &) {
      throw;
   }
}

//______________________________________________________________________________
void CommandBuffer::AddDeletePixmap(Pixmap_t pixmapID)
{
   try {
      std::auto_ptr<DeletePixmap> cmd(new DeletePixmap(pixmapID));
      fCommands.push_back(cmd.get());
      cmd.release();
   } catch (const std::exception &) {
      throw;
   }
}

namespace {

//______________________________________________________________________________
void RepaintTree(QuartzView *view)
{
   //Can be only QuartzView, ROOTOpenGLView should never have children views.
   assert(view != nil && "RepaintTree, view parameter is nil");
   
   TGCocoa *vx = (TGCocoa *)gVirtualX;
   vx->CocoaDrawON();
   
   for (NSView<X11Window> *child in [view subviews]) {
      if ([child isKindOfClass : [QuartzView class]]) {
         QuartzView *qv = (QuartzView *)child;
         if ([qv lockFocusIfCanDraw]) {
            NSGraphicsContext *nsContext = [NSGraphicsContext currentContext];
            assert(nsContext != nil && "RepaintTree, nsContext is nil");
            CGContextRef cgContext = (CGContextRef)[nsContext graphicsPort];
            assert(cgContext != 0 && "RepaintTree, cgContext is null");//remove this assert?

            CGContextRef oldCtx = qv.fContext;
            qv.fContext = cgContext;
            
            TGWindow *window = gClient->GetWindowById(qv.fID);
            assert(window != 0 && "RepaintTree, window was not found");
            
            gClient->NeedRedraw(window, kTRUE);
            qv.fContext = oldCtx;
            
            [qv unlockFocus];
            if ([[qv subviews] count])
               RepaintTree(qv);
         }
      }
   }
   
   vx->CocoaDrawOFF();
}

}

//______________________________________________________________________________
void CommandBuffer::Flush(Details::CocoaPrivate *impl)
{
   assert(impl != 0 && "Flush, impl parameter is null");

   //All magic is here.
   CGContextRef prevContext = 0;
   CGContextRef currContext = 0;
   QuartzView *prevView = nil;

   for (size_type i = 0, e = fCommands.size(); i < e; ++i) {
      const Command *cmd = fCommands[i];
      if (!cmd)//Command was deleted by RemoveOperation/RemoveGraphicsOperation.
         continue;
      
      NSObject<X11Drawable> *drawable = impl->GetDrawable(cmd->fID);
      if (drawable.fIsPixmap) {
         cmd->Execute();
         continue;
      }
      
      QuartzView *view = (QuartzView *)impl->GetWindow(cmd->fID).fContentView;
      
      if (prevView != view)
         ClipOverlaps(view);
      
      if (prevView && prevView != view && [[prevView subviews] count])
         RepaintTree(prevView);
      
      prevView = view;
      
      if ([view lockFocusIfCanDraw]) {
         NSGraphicsContext *nsContext = [NSGraphicsContext currentContext];
         assert(nsContext != nil && "Flush, currentContext is nil");
         currContext = (CGContextRef)[nsContext graphicsPort];
         assert(currContext != 0 && "Flush, graphicsPort is null");//remove this assert?
         
         view.fContext = currContext;
         if (prevContext && prevContext != currContext)
            CGContextFlush(prevContext);
         prevContext = currContext;
         
         //Context can be modified by a clip mask.
         const Quartz::CGStateGuard ctxGuard(currContext);
         if (view.fClipMaskIsValid)
            CGContextClipToMask(currContext, CGRectMake(0, 0, view.fClipMask.fWidth, view.fClipMask.fHeight), view.fClipMask.fImage);
         
         cmd->Execute();
         if (view.fBackBuffer) {
            //Very "special" window.
            CGImageRef image = [view.fBackBuffer createImageFromPixmap];//CGBitmapContextCreateImage(view.fBackBuffer.fContext);
            if (image) {
               const CGRect imageRect = CGRectMake(0, 0, view.fBackBuffer.fWidth, view.fBackBuffer.fHeight);
               CGContextDrawImage(view.fContext, imageRect, image);
               CGImageRelease(image);
            }
         }
         
         [view unlockFocus];
         
         view.fContext = 0;
      }
   }
   
   if (prevView && [[prevView subviews] count])
      RepaintTree(prevView);
   
   if (currContext)
      CGContextFlush(currContext);

   ClearCommands();
}

//______________________________________________________________________________
void CommandBuffer::ClipOverlaps(QuartzView *view)
{
   typedef std::vector<QuartzView *>::reverse_iterator reverse_iterator;

   assert(view != nil && "ClipOverlaps, view parameter is nil");

   fViewBranch.clear();
   fViewBranch.reserve(view.fLevel + 1);
   
   for (QuartzView *v = view; v; v = v.fParentView)
      fViewBranch.push_back(v);
   
   if (fViewBranch.size())
      fViewBranch.pop_back();//we do not need content view, it does not have any sibling.

   NSRect frame1 = {};
   NSRect frame2 = view.frame;
   
   //[view clearClipMask];
   view.fClipMaskIsValid = NO;
   
   for (reverse_iterator it = fViewBranch.rbegin(), eIt = fViewBranch.rend(); it != eIt; ++it) {
      QuartzView *ancestorView = *it;//Actually, it's either one of ancestors, or a view itself.
      bool doCheck = false;
      for (QuartzView *sibling in [ancestorView.fParentView subviews]) {
         if (ancestorView == sibling) {
            doCheck = true;//all views after this must be checked.
            continue;
         } else if (!doCheck || sibling.fMapState != kIsViewable) {
            continue;
         }
         
         //Real check is here.
         frame1 = sibling.frame;
         frame2.origin = [view.fParentView convertPoint : view.frame.origin toView : ancestorView.fParentView];
         
         //Check if two rects intersect.
         if (RectsOverlap(frame2, frame1)) {
            if (!view.fClipMaskIsValid) {
               if (![view initClipMask])//initClipMask will issue an error message.
                  return;//Forget about clipping at all.
               view.fClipMaskIsValid = YES;
            }
            //Update view's clip mask - mask out hidden pixels.
            [view addOverlap : FindOverlapRect(frame2, frame1)];
         }
      }
   }
}

//______________________________________________________________________________
void CommandBuffer::RemoveOperationsForDrawable(Drawable_t drawable)
{
   for (size_type i = 0; i < fCommands.size(); ++i) {
      if (fCommands[i] && fCommands[i]->HasOperand(drawable)) {
         delete fCommands[i];
         fCommands[i] = 0;
      }
   }
}

//______________________________________________________________________________
void CommandBuffer::RemoveGraphicsOperationsForWindow(Window_t wid)
{
   for (size_type i = 0; i < fCommands.size(); ++i) {
      if (fCommands[i] && fCommands[i]->HasOperand(wid) && fCommands[i]->IsGraphicsCommand()) {
         delete fCommands[i];
         fCommands[i] = 0;
      }
   }
}

//______________________________________________________________________________
void CommandBuffer::ClearCommands()
{
   for (size_type i = 0, e = fCommands.size(); i < e; ++i)
      delete fCommands[i];

   fCommands.clear();
}

}//X11
}//MacOSX
}//ROOT
