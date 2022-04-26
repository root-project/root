// @(#)root/base:$Id$
// Author: Rene Brun   26/06/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSystemFile
#define ROOT_TSystemFile


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSystemFile                                                          //
//                                                                      //
// Describes an Operating System file for the browser.                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TNamed.h"

class TBrowser;

class TSystemFile : public TNamed {
private:
   TString fIconName;   // icon name

public:
   TSystemFile();
   TSystemFile(const char *filename, const char *dirname);
   virtual ~TSystemFile();
   virtual void     Rename(const char *name);      // *MENU*
   virtual void     Delete();                      // *MENU*
   virtual void     Copy(const char *to);          // *MENU*
   virtual void     Move(const char *to);          // *MENU*
   virtual void     Edit();                        // *MENU*

   virtual Bool_t   IsDirectory(const char *dir = 0) const;
   virtual void     SetIconName(const char *name) { fIconName = name; }
   const char      *GetIconName() const override { return fIconName.Data(); }

   void         Browse(TBrowser *b) override;

   // dummy methods from TObject
   void        Inspect() const override;
   void        Dump() const  override;

   void        DrawClass() const override { }
   TObject    *DrawClone(Option_t *) const override { return nullptr; }
   void        SetDrawOption(Option_t *) override { }
   void        SetName(const char *name) override { TNamed::SetName(name); }
   void        SetTitle(const char *title)  override { TNamed::SetTitle(title); }
   void        Delete(Option_t *) override { }
   void        Copy(TObject &) const override { }

   ClassDefOverride(TSystemFile,0)  //A system file
};

#endif

