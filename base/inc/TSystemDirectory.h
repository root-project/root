// @(#)root/base:$Name:  $:$Id: TSystemDirectory.h,v 1.2 2000/09/05 09:21:22 brun Exp $
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

#ifndef ROOT_TSystemFile
#include "TSystemFile.h"
#endif

class TOrdCollection;
class TList;


class TSystemDirectory : public TSystemFile {

private:
   TOrdCollection *fDirsInBrowser;
   TOrdCollection *fFilesInBrowser;

   Bool_t             IsDirectory(const char *name) const;
   TSystemDirectory  *FindDirObj(const char *name);
   TSystemFile       *FindFileObj(const char *name, const char *dir);

public:
   TSystemDirectory();
   TSystemDirectory(const char *dirname, const char *path);

   virtual ~TSystemDirectory();

   virtual Bool_t IsFolder() const { return kTRUE; }

   virtual void   Browse(TBrowser *b);
   virtual void   Edit() { }
   virtual TList *GetListOfFiles() const;
   virtual void   SetDirectory(const char *name); // *MENU*

   ClassDef(TSystemDirectory,0)  //A system directory
};

#endif

