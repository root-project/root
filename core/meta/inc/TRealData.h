// @(#)root/meta:$Id$
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

#include "TObject.h"
#include "TString.h"

class TDataMember;


class TRealData : public TObject {

private:
   TDataMember     *fDataMember;     //pointer to data member descriptor
   Long_t           fThisOffset;     //offset with the THIS object pointer
   TString          fName;           //Concatenated names of this realdata
   TMemberStreamer *fStreamer;       //Object to stream the data member.
   Bool_t           fIsObject;       //true if member is an object

   TRealData(const TRealData& rhs) = delete;  // Copying TRealData in not allowed.
   TRealData& operator=(const TRealData& rhs) = delete;  // Copying TRealData in not allowed.

public:
   enum EStatusBits {
      kTransient = BIT(14)  // The member is transient.
   };

   TRealData();
   TRealData(const char *name, Long_t offset, TDataMember *datamember);
   virtual     ~TRealData();

   void                AdoptStreamer(TMemberStreamer *p);
   const char         *GetName() const override { return fName.Data(); }
   TDataMember        *GetDataMember() const { return fDataMember; }
   TMemberStreamer    *GetStreamer() const;
   Long_t              GetThisOffset() const { return fThisOffset; }
   Bool_t              IsObject() const { return fIsObject; }
   void                SetIsObject(Bool_t isObject) { fIsObject = isObject; }
   void                WriteRealData(void *pointer, char *&buffer);

   static void         GetName(TString &output, TDataMember *dm);

   ClassDefOverride(TRealData,0)  //Description of persistent data members
};

#endif

