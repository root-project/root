// @(#)root/cont:$Name:  $:$Id: TProcessUUID.h,v 1.6 2002/03/25 20:21:05 brun Exp $
// Author: Rene Brun   06/07/2002

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProcessUUID
#define ROOT_TProcessUUID


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProcessUUID                                                         //
//                                                                      //
// TProcessID managing UUIDs                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TProcessID
#include "TProcessID.h"
#endif

class THashList;
class TBits;
class TUUID;
class TObjString;

class TProcessUUID : public TProcessID {

protected:
   THashList   *fUUIDs;        //Global list of TUUIDs
   TBits       *fActive;       //Table of active UUIDs
      
public:

   TProcessUUID();
   virtual ~TProcessUUID();
   UInt_t             AddUUID(TUUID &uuid, TObject *obj);
   UInt_t             AddUUID(const char *uuids);
   TObjString        *FindUUID(UInt_t number) const;
   TBits             *GetActive() const {return fActive;}
   THashList         *GetUUIDs()  const {return fUUIDs;}
   void               RemoveUUID(UInt_t number);
   
   ClassDef(TProcessUUID,1)  //TProcessID managing UUIDs
};

#endif
