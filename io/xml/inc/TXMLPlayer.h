// @(#)root/xml:$Id$
// Author: Sergey Linev  10.05.2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXMLPlayer
#define ROOT_TXMLPlayer

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TList
#include "TList.h"
#endif

#ifndef ROOT_TXMLSetup
#include "TXMLSetup.h"
#endif

class TStreamerInfo;
class TStreamerElement;
class TStreamerSTL;
class TDataMember;

class TXMLPlayer : public TObject {
   public:
      TXMLPlayer();
      virtual ~TXMLPlayer();
      
      Bool_t ProduceCode(TList* cllist, const char* filename);
      
   protected:
   
      TString GetStreamerName(TClass* cl);
      
      const char* ElementGetter(TClass* cl, const char* membername, int specials = 0);
      const char* ElementSetter(TClass* cl, const char* membername, char* endch);
      
      TString GetMemberTypeName(TDataMember* member);
      TString GetBasicTypeName(TStreamerElement* el);
      TString GetBasicTypeReaderMethodName(Int_t type, const char* realname);
      void ProduceStreamerSource(ostream& fs, TClass* cl, TList* cllist);
      
      void ReadSTLarg(ostream& fs, TString& argname, int argtyp, Bool_t isargptr, TClass* argcl, TString& tname, TString& ifcond);
      void WriteSTLarg(ostream& fs, const char* accname, int argtyp, Bool_t isargptr, TClass* argcl);
      Bool_t ProduceSTLstreamer(ostream& fs, TClass* cl, TStreamerSTL* el, Bool_t isWriting);
      
      TString fGetterName;                   //!  buffer for name of getter method
      TString fSetterName;                   //!  buffer for name of setter method
      TXMLSetup fXmlSetup;                   //!  buffer for xml names convertion

   ClassDef(TXMLPlayer,1) // Generation of external xml streamers
};

#endif
