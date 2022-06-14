// @(#)root/base:$Id$
// Author: Christian Bormann  13/10/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TSystemDirectory
#define ROOT_TSystemDirectory


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSystemDirectory                                                     //
//                                                                      //
// Describes an Operating System directory for the browser.             //
//                                                                      //
// Author: Christian Bormann  30/09/97                                  //
//         http://www.ikf.physik.uni-frankfurt.de/~bormann/             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TSystemFile.h"

class TOrdCollection;
class TList;


class TSystemDirectory : public TSystemFile {

protected:
   TOrdCollection *fDirsInBrowser;
   TOrdCollection *fFilesInBrowser;

   Bool_t             IsItDirectory(const char *name) const;
   TSystemDirectory  *FindDirObj(const char *name);
   TSystemFile       *FindFileObj(const char *name, const char *dir);

   TSystemDirectory(const TSystemDirectory&);
   TSystemDirectory& operator=(const TSystemDirectory&);

public:
   TSystemDirectory();
   TSystemDirectory(const char *dirname, const char *path);

   virtual ~TSystemDirectory();

   Bool_t      IsFolder() const override { return kTRUE; }
   Bool_t      IsDirectory(const char * = nullptr) const override { return kTRUE; }

   void        Browse(TBrowser *b) override;
   void        Edit() override {}
   virtual TList *GetListOfFiles() const;
   virtual void   SetDirectory(const char *name);
   void        Delete() override {}
   void        Copy(const char *) override {}
   void        Move(const char *) override {}

   // dummy methods from TObject
   void        DrawClass() const override { }
   TObject    *DrawClone(Option_t *) const override { return nullptr; }
   void        SetDrawOption(Option_t *) override { }
   void        SetName(const char *name) override { TSystemFile::SetName(name); }
   void        SetTitle(const char *title) override { TSystemFile::SetTitle(title); }
   void        Delete(Option_t *) override { }
   void        Copy(TObject & ) const override { }

   ClassDefOverride(TSystemDirectory,0)  //A system directory
};

#endif

