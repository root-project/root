// @(#)root/xml:$Name:  $:$Id: TXMLDtdGenerator.h,v 1.2 2004/05/10 23:50:27 rdm Exp $
// Author: Sergey Linev  10.05.2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXMLDtdGenerator
#define ROOT_TXMLDtdGenerator

#ifndef ROOT_TXMLSetup
#include "TXMLSetup.h"
#endif
#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif


class TCollection;
class TClass;
class TStreamerElement;
class TStreamerInfo;

class TXMLDtdGenerator : public TXMLSetup {
   public:
      TXMLDtdGenerator();
      TXMLDtdGenerator(const char* setup);
      TXMLDtdGenerator(const TXMLSetup& setup);
      virtual ~TXMLDtdGenerator();

      void Produce(const char* fname, TClass* onlyclass = 0);

      void AddClassSpace(TCollection* col = 0);

      void AddInstrumentedClass(TStreamerInfo* info);

      void AddBlackClass(TClass* cl);

      void AddUsedClass(TClass * cl);

   protected:
      enum {dtd_none, dtd_attr, dtd_elem, dtd_charstar, dtd_base, dtd_object, dtd_objptr,
            dtd_fixarray, dtd_array, dtd_fastobj1, dtd_fastobj2,
            dtd_everyobj, dtd_stlp, dtd_objects, dtd_any};

      enum { MaxBaseTypeNum = 21};

      Int_t dtdType(TStreamerElement* el);

      const char* dtdBaseTypeName(int typ);
      const char* dtdUseBaseType(TStreamerElement* el);

      void ProduceDtdForItem(ofstream& fs, const char* name);

      void ProduceDtdForInstrumentedClass(ofstream& fs, TStreamerInfo* info);

      void ProduceDtdForBlackClass(ofstream& fs, TClass* cl);

      void ProduceObjectElement(ofstream& fs, const char* name, TClass* cl, Bool_t isPointer = kFALSE);

      void ProduceGeneralDtd(ofstream& fs, TClass* onlyclass = 0);
      void ProduceSpecificDtd(ofstream& fs, TClass* onlyclass = 0);

      TObjArray  fClassSpace;
      TObjArray  fInstrumentedClasses;
      TObjArray  fBlackClasses;

      TObjArray  fUsedClasses;

      TString fDtdBuf;
      Bool_t  fUsedBaseTypes[MaxBaseTypeNum];

   ClassDef(TXMLDtdGenerator,1) // The XML DTD generator
};


#endif

