#ifndef ROOT_TXMLPlayer
#define ROOT_TXMLPlayer

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TList
#include "TList.h"
#endif

class TStreamerInfo;
class TStreamerElement;
class TDataMember;

class TXMLPlayer : public TObject {
   public:
      TXMLPlayer();
      virtual ~TXMLPlayer();
      
      Int_t ProduceCode(TList* cllist, const char* filename);
      
   protected:
   
      TString GetStreamerName(TClass* cl);
      
      const char* ElementGetter(TClass* cl, const char* membername, int specials = 0);
      const char* ElementSetter(TClass* cl, const char* membername, char* endch);
      
      TString GetElemName(TStreamerElement* el);
      
      TString GetMemberTypeName(TDataMember* member);
      TString GetBasicTypeName(TStreamerElement* el);
      TString GetBasicTypeReaderMethodName(TStreamerElement* el);
      void ProduceStreamerSource(ostream& fs, TClass* cl, TList* cllist);
      
      TString fGetterName;
      TString fSetterName;

   ClassDef(TXMLPlayer,1) // generator of external XML reader/writers
};




#endif
