// @(#)root/io:$Id$
// Author: Rene Brun   23/07/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TKeyMapFile
#define ROOT_TKeyMapFile

#include "TNamed.h"

class TBrowser;
class TMapFile;

class TKeyMapFile : public TNamed {

private:
   TKeyMapFile(const TKeyMapFile&);            // TKeyMapFile objects are not copiable.
   TKeyMapFile& operator=(const TKeyMapFile&); // TKeyMapFile objects are not copiable.

   TMapFile      *fMapFile;       ///< Pointer to map file

public:
   TKeyMapFile();
   TKeyMapFile(const char *name, const char *classname, TMapFile *mapfile);
   virtual ~TKeyMapFile() {;}
   virtual void      Browse(TBrowser *b);

   ClassDef(TKeyMapFile,0);  //Utility class for browsing TMapFile objects.
};

#endif
