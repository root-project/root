// @(#)root/base:$Name$:$Id$
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


#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TBrowser;

class TSystemFile : public TNamed {

public:
        TSystemFile();
        TSystemFile(const char *filename, const char *dirname);
        virtual ~TSystemFile();
        virtual void     Browse(TBrowser *b);
        virtual void     Edit(); // *MENU*

        ClassDef(TSystemFile,0)  //A system file
};

#endif

