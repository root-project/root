// @(#)root/base:$Name:  $:$Id: TRealData.h,v 1.1.1.1 2000/05/16 17:00:39 rdm Exp $
// Author: Rene Brun   05/03/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRealData
#define ROOT_TRealData


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRealData                                                            //
//                                                                      //
// Description of persistent data members.                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class TDataMember;


class TRealData : public TObject {

private:
   TDataMember *fDataMember;         //pointer to data member descriptor
   Int_t        fThisOffset;         //offset with the THIS object pointer
   TString      fName;               //Concatenated names of this realdata
   void        *fStreamer;           //!pointer to STL Streamer function
   
public:
   TRealData();
   TRealData(const char *name, Int_t offset, TDataMember *datamember);
   virtual     ~TRealData();
   virtual const char *GetName() const {return fName.Data();}
   TDataMember *GetDataMember() {return fDataMember;}
   void        *GetStreamer() {return fStreamer;}
   Int_t        GetThisOffset() {return fThisOffset;}
   void         SetStreamer(void *p) {fStreamer = p;}
   void         WriteRealData(void *pointer, char *&buffer);

   ClassDef(TRealData,0)  //Description of persistent data members
};

#endif

