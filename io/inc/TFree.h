// @(#)root/base:$Name:  $:$Id: TFree.h,v 1.1.1.1 2000/05/16 17:00:39 rdm Exp $
// Author: Rene Brun   28/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFree
#define ROOT_TFree


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFree                                                                //
//                                                                      //
// Description of free segments on a file.                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif


class TFree : public TObject {

protected:
        Seek_t        fFirst;            //First free word of segment
        Seek_t        fLast;             //Last free word of segment

public:
        TFree();
        TFree(TList *lfree, Seek_t first, Seek_t last);
        virtual ~TFree();
               TFree     *AddFree(TList *lfree, Seek_t first, Seek_t last);
        virtual void     FillBuffer(char *&buffer);
               TFree     *GetBestFree(TList *lfree, Int_t nbytes);
              Seek_t     GetFirst() {return fFirst;}
              Seek_t     GetLast() {return fLast;}
        virtual void     ReadBuffer(char *&buffer);
                void     SetFirst(Seek_t first) {fFirst=first;}
                void     SetLast(Seek_t last) {fLast=last;}
               Int_t     Sizeof() const {return sizeof(Version_t) + 2*sizeof(Seek_t);}

        ClassDef(TFree,1)  //Description of free segments on a file
};

#endif
