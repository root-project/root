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

   virtual Bool_t IsFolder() const { return kTRUE; }
   virtual Bool_t IsDirectory(const char * = 0) const { return kTRUE; }

   virtual void   Browse(TBrowser *b);
   virtual void   Edit() { }
   virtual TList *GetListOfFiles() const;
   virtual void   SetDirectory(const char *name);
   virtual void   Delete() {}
   virtual void   Copy(const char *) {}
   virtual void   Move(const char *) {}

   // dummy methods from TObject
   void        DrawClass() const { }
   TObject    *DrawClone(Option_t *) const { return 0; }
   void        SetDrawOption(Option_t *) { }
   void        SetName(const char *name) { TSystemFile::SetName(name); }
   void        SetTitle(const char *title) { TSystemFile::SetTitle(title); }
   void        Delete(Option_t *) { }
   void        Copy(TObject & ) const { }
   ClassDef(TSystemDirectory,0)  //A system directory
};

#endif

