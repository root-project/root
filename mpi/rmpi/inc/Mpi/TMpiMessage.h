// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2016 http://oproject.org 
#ifndef ROOT_TMpiMessage
#define ROOT_TMpiMessage

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#include<TClass.h>
#include<TMessage.h>
#include<TROOT.h>

namespace ROOT {
   namespace Mpi {
      class TMpiMessage: public  TMessage {
	TString fDataTypeName;		//Datatype name encapsulate in this message
	UInt_t  fOrigin;                //Rank(Process ID) of origin of this message 
	UInt_t  fDestination;           //Rank(Process ID) of destination of this message 
      public:
         using TMessage::WriteObject;
         TMpiMessage(Char_t *buffer, Int_t size);
         TMpiMessage(UInt_t what = kMESS_ANY, Int_t bufsiz = TBuffer::kInitialSize);
         virtual ~TMpiMessage() {}
         template<class ClassType>  void WriteObject(ClassType *obj);
         template<class ClassType>  void WriteObject(ClassType &obj);
      private:
         ClassDef(TMpiMessage, 1);
      };
      
      template<class ClassType> void TMpiMessage::WriteObject(ClassType *obj)
      {
	 const std::type_info& type = typeid(*obj);
	 fDataTypeName=type.name();
	 TClass *cl=gROOT->GetClass(type); 
         WriteObjectAny(obj, cl);
      }
      template<class ClassType> void TMpiMessage::WriteObject(ClassType &obj)
      {
	WriteObject<ClassType>(&obj);
      }
   }
}

#endif
