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

#include "TNamed.h"

class TCollection;
class TBrowser;


class TFolder : public TNamed {

protected:
   TCollection       *fFolders;        //pointer to the list of folders
   Bool_t             fIsOwner;        //true if folder own its contained objects

private:
   TFolder(const TFolder &folder) = delete;  //folders cannot be copied
   void operator=(const TFolder &)= delete;

public:

   TFolder();
   TFolder(const char *name, const char *title);
   virtual ~TFolder();
   virtual void        Add(TObject *obj);
   TFolder            *AddFolder(const char *name, const char *title, TCollection *collection=nullptr);
   void                Browse(TBrowser *b) override;
   void                Clear(Option_t *option="") override;
   void                Copy(TObject &) const override { MayNotUse("Copy(TObject &)"); }
   virtual const char *FindFullPathName(const char *name) const;
   virtual const char *FindFullPathName(const TObject *obj) const;
   TObject            *FindObject(const char *name) const override;
   TObject            *FindObject(const TObject *obj) const override;
   virtual TObject    *FindObjectAny(const char *name) const;
   TCollection        *GetListOfFolders() const { return fFolders; }
   Bool_t              IsFolder() const override { return kTRUE; }
   Bool_t              IsOwner()  const;
   void                ls(Option_t *option="") const override;  // *MENU*
   virtual Int_t       Occurence(const TObject *obj) const;
   void                RecursiveRemove(TObject *obj) override;
   virtual void        Remove(TObject *obj);
   void                SaveAs(const char *filename="",Option_t *option="") const override; // *MENU*
   virtual void        SetOwner(Bool_t owner=kTRUE);

   ClassDefOverride(TFolder,1)  //Describe a folder: a list of objects and folders
};

#endif
