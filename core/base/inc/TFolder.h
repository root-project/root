// @(#)root/base:$Id$
// Author: Rene Brun   02/09/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFolder
#define ROOT_TFolder


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFolder                                                              //
//                                                                      //
// Describe a folder: a list of objects and folders                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TCollection;
class TBrowser;


class TFolder : public TNamed {

protected:
   TCollection       *fFolders;        //pointer to the list of folders
   Bool_t             fIsOwner;        //true if folder own its contained objects

private:
   TFolder(const TFolder &folder);  //folders cannot be copied
   void operator=(const TFolder &);

public:

   TFolder();
   TFolder(const char *name, const char *title);
   virtual ~TFolder();
   virtual void        Add(TObject *obj);
   TFolder            *AddFolder(const char *name, const char *title, TCollection *collection=0);
   virtual void        Browse(TBrowser *b);
   virtual void        Clear(Option_t *option="");
   virtual void        Copy(TObject &) const { MayNotUse("Copy(TObject &)"); }
   virtual const char *FindFullPathName(const char *name) const;
   virtual const char *FindFullPathName(const TObject *obj) const;
   virtual TObject    *FindObject(const char *name) const;
   virtual TObject    *FindObject(const TObject *obj) const;
   virtual TObject    *FindObjectAny(const char *name) const;
   TCollection        *GetListOfFolders() const { return fFolders; }
   Bool_t              IsFolder() const { return kTRUE; }
   Bool_t              IsOwner()  const;
   virtual void        ls(Option_t *option="") const;  // *MENU*
   virtual Int_t       Occurence(const TObject *obj) const;
   virtual void        RecursiveRemove(TObject *obj);
   virtual void        Remove(TObject *obj);
   virtual void        SaveAs(const char *filename="",Option_t *option="") const; // *MENU*
   virtual void        SetOwner(Bool_t owner=kTRUE);

   ClassDef(TFolder,1)  //Describe a folder: a list of objects and folders
};

#endif
