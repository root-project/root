// @(#)root/core:$Id$
// author: Lukasz Janyst <ljanyst@cern.ch>

#ifndef ROOT_TSchemaRuleSet
#define ROOT_TSchemaRuleSet

class TClass;

#include "TObject.h"
#include "TObjArray.h"
#include "Rtypes.h"
#include "TString.h"
#include "TClassRef.h"

namespace ROOT {

   class TSchemaRule;

   class TSchemaMatch: public TObjArray
   {
      public:
         virtual ~TSchemaMatch() {};
         const TSchemaRule* GetRuleWithSource( const TString& name ) const;
         const TSchemaRule* GetRuleWithTarget( const TString& name ) const;
               Bool_t       HasRuleWithSource( const TString& name ) const;
               Bool_t       HasRuleWithTarget( const TString& name ) const;
   };

   class TSchemaRuleSet: public TObject
   {
      public:

         TSchemaRuleSet();
         virtual ~TSchemaRuleSet();

         Bool_t              AddRule( TSchemaRule* rule, Bool_t checkConsistency = kTRUE );
         Bool_t              HasRuleWithSourceClass( const TString &source) const;
         const TObjArray*    FindRules( const TString &source ) const;
         const TSchemaMatch* FindRules( const TString &source, Int_t version ) const;
         const TSchemaMatch* FindRules( const TString &source, UInt_t checksum ) const;
         TClass*             GetClass();
         UInt_t              GetClassCheckSum() const;
         TString             GetClassName() const;
         Int_t               GetClassVersion() const;
         const TObjArray*    GetRules() const;
         const TObjArray*    GetPersistentRules() const;
         void                RemoveRule( TSchemaRule* rule );
         void                RemoveRules( TObjArray* rules );
         void                SetClass( TClass* cls );

         ClassDef( TSchemaRuleSet, 1 )

      private:
         TObjArray*                             fPersistentRules; //  Array of the rules that will be embeded in the file
         TObjArray*                             fRemainingRules;  //! Array of non-persisten rules - just for cleanup purposes - owns the elements
         TObjArray*                             fAllRules;        //! Array of all rules
         TClassRef                              fClass;           //! Target class pointer (for consistency checking)
         TString                                fClassName;       //  Target class name
         Int_t                                  fVersion;         //  Target class version
         UInt_t                                 fCheckSum;        //  Target class checksum
   };

} // End of Namespace ROOT 

#endif // ROOT_TSchemaRuleSet
