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
   assert(vx != nullptr && "Execute, gVirtualX is either null or not of TGCocoa type");
   vx->DrawLineAux(fID, fGC, fP1.fX, fP1.fY, fP2.fX, fP2.fY);
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
   assert(vx != nullptr && "Execute, gVirtualX is either null or not of TGCocoa type");
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
   assert(vx != nullptr && "Execute, gVirtualX is either null or not of TGCocoa type");
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
   assert(vx != nullptr && "Execute, gVirtualX is either null or not of TGCocoa type");
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
   assert(vx != nullptr && "Execute, gVirtualX is either null or not of TGCocoa type");
   vx->FillRectangleAux(fID, fGC, fRectangle.fX, fRectangle.fY, fRectangle.fWidth, fRectangle.fHeight);
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
   assert(vx != nullptr && "Execute, gVirtualX is either null or not of TGCocoa type");
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
   assert(fView.fContext != nullptr && "Execute, view.fContext is null");

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
   assert(vx != nullptr && "Execute, gVirtualX is either null or not of TGCocoa type");
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

//______________________________________________________________________________
void CommandBuffer::Flush(Details::CocoaPrivate *impl)
{
   assert(impl != nullptr && "Flush, impl parameter is null");

   //All magic is here.
   CGContextRef prevContext = nullptr;
   CGContextRef currContext = nullptr;

   for (auto cmd : fCommands) {
      //assert(cmd != nullptr && "Flush, command is null");
      if (!cmd)//Command was deleted by RemoveOperation/RemoveGraphicsOperation.
         continue;
      
      NSObject<X11Drawable> *drawable = impl->GetDrawable(cmd->fID);
      if (drawable.fIsPixmap) {
         cmd->Execute();
         continue;
      }
      
      QuartzView *view = (QuartzView *)impl->GetWindow(cmd->fID).fContentView;
      if ([view lockFocusIfCanDraw]) {
         NSGraphicsContext *nsContext = [NSGraphicsContext currentContext];
         assert(nsContext != nil && "Flush, currentContext is nil");
         currContext = (CGContextRef)[nsContext graphicsPort];
         assert(currContext != nullptr && "Flush, graphicsPort is null");//remove this assert?
         
         view.fContext = currContext;
         if (prevContext && prevContext != currContext)
            CGContextFlush(prevContext);
         prevContext = currContext;
         
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
         view.fContext = nullptr;
      }
   }
   
   if (currContext)
      CGContextFlush(currContext);

   ClearCommands();
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
   for (auto cmd : fCommands)
      delete cmd;

   fCommands.clear();
}

}//X11
}//MacOSX
}//ROOT
