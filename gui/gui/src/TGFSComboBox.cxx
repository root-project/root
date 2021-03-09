// @(#)root/gui:$Id$
// Author: Fons Rademakers   19/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
/**************************************************************************

    This source is based on Xclass95, a Win95-looking GUI toolkit.
    Copyright (C) 1996, 1997 David Barth, Ricky Ralston, Hector Peraza.

    Xclass95 is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

**************************************************************************/


/** \class TGFSComboBox
    \ingroup guiwidgets

This is a combo box that is used in the File Selection dialog box.
It will allow the file path selection.

*/


#include "RConfigure.h"

#include "TGFSComboBox.h"
#include "TGResourcePool.h"
#include "TGPicture.h"
#include "TSystem.h"
#include "TVirtualX.h"
#include "strlcpy.h"

#include <cstdlib>
#include <iostream>

const TGFont *TGTreeLBEntry::fgDefaultFont = nullptr;
TGGC         *TGTreeLBEntry::fgDefaultGC = nullptr;

//--- this is temp here...

ClassImp(TGTreeLBEntry);
ClassImp(TGFSComboBox);

////////////////////////////////////////////////////////////////////////////////
/// Create a tree (i.e. entry can be indented) listbox entry.
/// The strings text and path are adopted by the entry.

TGTreeLBEntry::TGTreeLBEntry(const TGWindow *p, TGString *text,
                             const TGPicture *pic, Int_t id, TGString *path,
                             GContext_t norm, FontStruct_t font, UInt_t options,
                             ULong_t back) :
   TGLBEntry(p, id, options, back)
{
   if (text && !pic)
      Error("TGTreeLBEntry", "icon not found for entry %s", text->GetString());

   fPic    = pic;
   fSelPic = 0;
   fTWidth = 0;
   fText   = text;
   fPath   = path;

   fNormGC     = norm;
   fFontStruct = font;

   fActive = kFALSE;

   int max_ascent, max_descent;
   if (fText)
      fTWidth = gVirtualX->TextWidth(fFontStruct, fText->GetString(), fText->GetLength());
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);
   fTHeight = max_ascent + max_descent;
   SetWindowName();
}

////////////////////////////////////////////////////////////////////////////////
/// Delete tree listbox entry.

TGTreeLBEntry::~TGTreeLBEntry()
{
   delete fText;
   delete fPath;
   delete fSelPic;
}

////////////////////////////////////////////////////////////////////////////////
/// Make entry active (highlight picture).

void TGTreeLBEntry::Activate(Bool_t a)
{
   if (fActive == a) return;
   fActive = a;

   if (fActive) {
      fSelPic = new TGSelectedPicture(fClient, fPic);
   } else {
      if (fSelPic) delete fSelPic;
      fSelPic = 0;
   }
   DoRedraw();
}

////////////////////////////////////////////////////////////////////////////////
/// Redraw the tree listbox entry on pixmap/window.

void TGTreeLBEntry::DrawCopy(Handle_t id, Int_t x, Int_t y)
{
   int ix, iy, lx, ly;

   ix = x;
   iy = y + ((fHeight - fPic->GetHeight()) >> 1);
   lx = x + (int)(fPic->GetWidth() + 4);
   ly = y + (int)((fHeight - (fTHeight+1)) >> 1);

   if (fActive) {
      if (fSelPic) fSelPic->Draw(id, fNormGC, ix, iy);
      gVirtualX->SetForeground(fNormGC, fgDefaultSelectedBackground);
      gVirtualX->FillRectangle(id, fNormGC, lx-2, ly, fWidth-(lx-x), fTHeight+1);
      gVirtualX->SetForeground(fNormGC, fClient->GetResourcePool()->GetSelectedFgndColor());
   } else {
      fPic->Draw(id, fNormGC, ix, iy);
      gVirtualX->SetForeground(fNormGC, fgWhitePixel);
      gVirtualX->FillRectangle(id, fNormGC, lx-2, ly, fWidth-(lx-x), fTHeight+1);
      gVirtualX->SetForeground(fNormGC, fgBlackPixel);
   }

   int max_ascent, max_descent;
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);

   fText->Draw(id, fNormGC, lx, ly + max_ascent);
}

////////////////////////////////////////////////////////////////////////////////
/// Redraw the tree listbox entry.

void TGTreeLBEntry::DoRedraw()
{
   DrawCopy(fId, 0, 0);
}

////////////////////////////////////////////////////////////////////////////////
/// Return default size of tree listbox entry.

TGDimension TGTreeLBEntry::GetDefaultSize() const
{
   TGDimension isize(fPic->GetWidth(), fPic->GetHeight());
   TGDimension lsize(fTWidth, fTHeight+1);

   return TGDimension(isize.fWidth + lsize.fWidth + 4,
                      TMath::Max(isize.fHeight, lsize.fHeight) + 2);
}

////////////////////////////////////////////////////////////////////////////////
/// Update text and picture of a listbox entry.

void TGTreeLBEntry::Update(TGLBEntry *e)
{
   TGTreeLBEntry *te = (TGTreeLBEntry *) e;

   if (fText) delete fText;
   fText = new TGString(te->GetText());
   fPic = te->GetPicture();
   fTWidth = gVirtualX->TextWidth(fFontStruct, fText->GetString(), fText->GetLength());
   gVirtualX->ClearWindow(fId);
   fClient->NeedRedraw(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Return default font structure.

FontStruct_t TGTreeLBEntry::GetDefaultFontStruct()
{
   if (!fgDefaultFont)
      fgDefaultFont = gClient->GetResourcePool()->GetDefaultFont();
   return fgDefaultFont->GetFontStruct();
}

////////////////////////////////////////////////////////////////////////////////
/// Return default graphics context.

const TGGC &TGTreeLBEntry::GetDefaultGC()
{
   if (!fgDefaultGC)
      fgDefaultGC = new TGGC(*gClient->GetResourcePool()->GetFrameGC());
   return *fgDefaultGC;
}


////////////////////////////////////////////////////////////////////////////////
/// Create a file system combobox showing system directories.

TGFSComboBox::TGFSComboBox(const TGWindow *parent, Int_t id, UInt_t options,
                           ULong_t back) :
   TGComboBox(parent, id, options | kOwnBackground, back)
{
   SetTopEntry(new TGTreeLBEntry(this, new TGString("Current dir"),
               fClient->GetPicture("folder_t.xpm"), 0),
               new TGLayoutHints(kLHintsExpandX | kLHintsExpandY, 4, 0, 0, 0));

   fListBox->GetContainer()->AddInput(kButtonPressMask | kButtonReleaseMask |
                                      kPointerMotionMask);

   //--- first check for the existence of some directories...

   const char *homeDir = gSystem->HomeDirectory();
#ifndef ROOTPREFIX
   const char *rootSys = gSystem->Getenv("ROOTSYS");
#else
   // const char *rootSys = ROOTPREFIX;
#endif

   TList *volumes = gSystem->GetVolumes("all");
   TList *curvol  = gSystem->GetVolumes("cur");
   const char *curdrive = "";
   if (volumes && curvol) {
      TNamed *named = (TNamed *)curvol->At(0);
      if (named) {
         curdrive = named->GetName();
         TString infos = named->GetTitle();

         fLbc.emplace_back(infos.Data(), Form("%s\\", curdrive), "hdisk_t.xpm", 0);
         if (infos.Contains("Removable"))
            fLbc.back().fPixmap = "fdisk_t.xpm";
         else if (infos.Contains("Local"))
            fLbc.back().fPixmap = "hdisk_t.xpm";
         else if (infos.Contains("CD"))
            fLbc.back().fPixmap = "cdrom_t.xpm";
         else if (infos.Contains("Network"))
            fLbc.back().fPixmap = "netdisk_t.xpm";
      } else {
         fLbc.emplace_back("Root", "/", "hdisk_t.xpm", 1);
      }
   } else {
      fLbc.emplace_back("Root", "/", "hdisk_t.xpm", 1);
      fLbc.emplace_back("Floppy", "/floppy", "fdisk_t.xpm", 1);
      fLbc.emplace_back("CD-ROM", "/cdrom", "cdrom_t.xpm", 1);
   }
   fLbc.emplace_back("Home", "$HOME", "home_t.xpm", 1);
#ifndef ROOTPREFIX
   fLbc.emplace_back("RootSys", "$ROOTSYS", "root_t.xpm", 1);
#else
   fLbc.emplace_back(ROOTPREFIX, ROOTPREFIX, "root_t.xpm", 1);
#endif

   if (volumes && curvol) {
      TIter next(volumes);
      TNamed *drive;
      while ((drive = (TNamed *)next())) {
         if (!strcmp(drive->GetName(), curdrive))
            continue;
         TString infos = drive->GetTitle();

         fLbc.emplace_back(drive->GetTitle(), Form("%s\\", drive->GetName()), "hdisk_t.xpm", 0);

         if (infos.Contains("Removable"))
            fLbc.back().fPixmap = "fdisk_t.xpm";
         else if (infos.Contains("Local"))
            fLbc.back().fPixmap = "hdisk_t.xpm";
         else if (infos.Contains("CD"))
            fLbc.back().fPixmap = "cdrom_t.xpm";
         else if (infos.Contains("Network"))
            fLbc.back().fPixmap = "netdisk_t.xpm";
      }
      delete volumes;
      delete curvol;
   }

   Int_t cnt = 0;

   for (auto &entry : fLbc) {
      entry.fId = ++cnt * 1000; // assign ids after vector is created
      if (entry.fPath.find("$HOME") == 0) {
         if (homeDir) {
            std::string newpath = homeDir;
            newpath.append(entry.fPath.substr(5));
            entry.fPath = newpath;
         } else {
            entry.fFlags = 0;
         }
      }
#ifndef ROOTPREFIX
      // Below should _only_ be called if the prefix isn't set at build
      // time. The code below expands the occurance of `$ROOTSYS' in
      // the table above.  However, in the case of prefix being set at
      // build time, we do not need to expand the prefix, as it is
      // already known, so the entries in the table above are actually
      // fully expanded.
      if (entry.fPath.find("$ROOTSYS") == 0) {
         // Get the size of the prefix template
         const int plen = 8;
         if (rootSys) {
            std::string newpath = rootSys;
            newpath.append(entry.fPath.substr(plen));
            entry.fPath = newpath;
         } else {
            entry.fFlags = 0;
         }
      }
#endif
      if (gSystem->AccessPathName(entry.fPath.c_str(), kFileExists) == 0)
         entry.fFlags = 1;
   }

   //--- then init the contents...

   for (auto &entry : fLbc) {
      if (entry.fFlags) {
         int indent = 4 + (entry.fIndent * 10);
         auto pic = fClient->GetPicture(entry.fPixmap.c_str());
         if (!pic) Error("TGFSComboBox", "pixmap not found: %s", entry.fPixmap.c_str());
         AddEntry(new TGTreeLBEntry(fListBox->GetContainer(),
                  new TGString(entry.fName.c_str()), pic, entry.fId,
                  new TGString(entry.fPath.c_str())),
                  new TGLayoutHints(kLHintsExpandX | kLHintsTop, indent, 0, 0, 0));
      }
   }
   SetWindowName();
}

////////////////////////////////////////////////////////////////////////////////
/// Update file system combo box.

void TGFSComboBox::Update(const char *path)
{
   char dirname[1024], mpath[1024];
   const char *tailpath = 0;
   int  indent_lvl = 0, afterID = -1, sel = -1;

   if (!path) return;

   for (int i = 0; i < (int) fLbc.size() - 1; ++i)
      RemoveEntries(fLbc[i].fId+1, fLbc[i+1].fId-1);
   RemoveEntries(fLbc.back().fId+1, (fLbc.size() + 1) * 1000 - 1);

   int len = 0;
   for (auto &entry : fLbc) {
      if (entry.fFlags) {
         int slen = entry.fPath.length();
         if (strncmp(path, entry.fPath.c_str(), slen) == 0) {
            if (slen > len) {
               sel = afterID = entry.fId;
               indent_lvl = entry.fIndent + 1;
               if ((len > 0) && ((path[slen] == '\\') || (path[slen] == '/') ||
                   (path[slen] == 0)))
                  tailpath = path + slen;
               strlcpy(mpath, entry.fPath.c_str(), 1024);
               len = slen;
            }
         }
      }
   }

   if (tailpath && *tailpath) {
      if ((*tailpath == '/') || (*tailpath == '\\')) ++tailpath;
      if (*tailpath)
         while (1) {
            const char *picname;
            const char *semi = strchr(tailpath, '/');
            if (semi == 0) semi = strchr(tailpath, '\\');
            if (semi == 0) {
               strlcpy(dirname, tailpath, 1024);
               picname = "ofolder_t.xpm";
            } else {
               strlcpy(dirname, tailpath, (semi-tailpath)+1);
               picname = "folder_t.xpm";
            }
            if ((mpath[strlen(mpath)-1] != '/') &&
                (mpath[strlen(mpath)-1] != '\\')) {
               strlcat(mpath, "/", 1024-strlen(mpath));
            }
            strlcat(mpath, dirname, 1024-strlen(mpath));
            int indent = 4 + (indent_lvl * 10);
            const TGPicture *pic = fClient->GetPicture(picname);
            if (!pic) Error("Update", "pixmap not found: %s", picname);
            InsertEntry(new TGTreeLBEntry(fListBox->GetContainer(),
                        new TGString(dirname), pic, afterID+1,
                        new TGString(mpath)),
                        new TGLayoutHints(kLHintsExpandX | kLHintsTop,
                                          indent, 0, 0, 0),
                        afterID);
            sel = ++afterID;
            ++indent_lvl;
            if (semi == 0) break;
            tailpath = ++semi;
         }
   }
   if (sel > 0) Select(sel);
}

////////////////////////////////////////////////////////////////////////////////
/// Save a file system combo box as a C++ statement(s) on output stream out.

void TGFSComboBox::SavePrimitive(std::ostream &out, Option_t *option /*= ""*/)
{
   if (fBackground != GetWhitePixel()) SaveUserColor(out, option);

   out << std::endl << "   // file system combo box" << std::endl;
   out << "   TGFSComboBox *";
   out << GetName() << " = new TGFSComboBox(" << fParent->GetName()
                                          << "," << fWidgetId;
   if (fBackground == GetWhitePixel()) {
      if (GetOptions() == (kHorizontalFrame | kSunkenFrame | kDoubleBorder)) {
         out <<");" << std::endl;
      } else {
         out << "," << GetOptionString() <<");" << std::endl;
      }
   } else {
      out << "," << GetOptionString() << ",ucolor);" << std::endl;
   }
   if (option && strstr(option, "keep_names"))
      out << "   " << GetName() << "->SetName(\"" << GetName() << "\");" << std::endl;

   out << "   " << GetName() << "->Resize(" << GetWidth()  << ","
       << GetHeight() << ");" << std::endl;
   out << "   " << GetName() << "->Select(" << GetSelected() << ");" << std::endl;

}
