// @(#)root/core:$Id$
// author: Lukasz Janyst <ljanyst@cern.ch>

#include "TSchemaRuleSet.h"
#include "TSchemaRule.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TClass.h"
#include "TROOT.h"
#include "Riostream.h"

ClassImp(TSchemaRule)

using namespace ROOT;

//------------------------------------------------------------------------------
TSchemaRuleSet::TSchemaRuleSet(): fPersistentRules( 0 ), fRemainingRules( 0 ),
                                  fAllRules( 0 ), fVersion(-3), fCheckSum( 0 )
{
   fPersistentRules = new TObjArray();
   fRemainingRules  = new TObjArray();
   fAllRules        = new TObjArray();
   fAllRules->SetOwner( kTRUE );
}

//------------------------------------------------------------------------------
TSchemaRuleSet::~TSchemaRuleSet()
{
   delete fPersistentRules;
   delete fRemainingRules;
   delete fAllRules;
}

//------------------------------------------------------------------------------
void TSchemaRuleSet::ls(Option_t *) const
{
   // The ls function lists the contents of a class on stdout. Ls output
   // is typically much less verbose then Dump().
   
   TROOT::IndentLevel();
   cout << "TSchemaRuleSet for " << fClassName << ":\n";
   TROOT::IncreaseDirLevel();
   TObject *object = 0;
   TIter next(fPersistentRules);
   while ((object = next())) {
      object->ls(fClassName);
   }
   TROOT::DecreaseDirLevel();   
}

//------------------------------------------------------------------------------
Bool_t TSchemaRuleSet::AddRules( TSchemaRuleSet* /* rules */, EConsistencyCheck /* checkConsistency */, TString * /* errmsg */ )
{
   return kFALSE;
}

//------------------------------------------------------------------------------
Bool_t TSchemaRuleSet::AddRule( TSchemaRule* rule, EConsistencyCheck checkConsistency, TString *errmsg )
{
   // The consistency check always fails if the TClass object was not set!
   // if checkConsistency is:
   //   kNoCheck: no check is done, register the rule as is
   //   kCheckConflict: check only for conflicting rules
   //   kCheckAll: check for conflict and check for rule about members that are not in the current class layout.
   // return kTRUE if the layout is accepted, in which case we take ownership of
   // the rule object.
   // return kFALSE if the rule failed one of the test, the rule now needs to be deleted by the caller.

   //---------------------------------------------------------------------------
   // Cannot verify the consistency if the TClass object is not present
   //---------------------------------------------------------------------------
   if( (checkConsistency != kNoCheck) && !fClass )
      return kFALSE;

   if( !rule->IsValid() )
      return kFALSE;

   //---------------------------------------------------------------------------
   // If we don't check the consistency then we should just add the object
   //---------------------------------------------------------------------------
   if( checkConsistency == kNoCheck ) {
      if( rule->GetEmbed() )
         fPersistentRules->Add( rule );
      else
         fRemainingRules->Add( rule );
      fAllRules->Add( rule );
      return kTRUE;
   }

   //---------------------------------------------------------------------------
   // Check if all of the target data members specified in the rule are
   // present int the target class
   //---------------------------------------------------------------------------
   TObject* obj;
   // Check only if we have some information about the class, otherwise we have
   // nothing to check against
   if( rule->GetTarget()  && !(fClass->TestBit(TClass::kIsEmulation) && (fClass->GetStreamerInfos()==0 || fClass->GetStreamerInfos()->GetEntries()==0)) ) {
      TObjArrayIter titer( rule->GetTarget() );
      while( (obj = titer.Next()) ) {
         TObjString* str = (TObjString*)obj;
         if( !fClass->GetDataMember( str->GetString() ) && !fClass->GetBaseClass( str->GetString() ) ) {
            if (checkConsistency == kCheckAll) {
               if (errmsg) {
                  errmsg->Form("the target member (%s) is unknown",str->GetString().Data());
               }
               return kFALSE;
            } else {
               // We ignore the rules that do not apply ...
               delete rule;
               return kTRUE;
            }
         }
      }
   }

   //---------------------------------------------------------------------------
   // Check if there is a rule conflicting with this one
   //---------------------------------------------------------------------------
   const TObjArray* rules = FindRules( rule->GetSourceClass() );
   TObjArrayIter it( rules );
   TSchemaRule *r;

   while( (obj = it.Next()) ) {
      r = (TSchemaRule *) obj;
      if( rule->Conflicts( r ) ) {
         delete rules;
         if ( *r == *rule) {
            // The rules are duplicate from each other,
            // just ignore the new ones.
            if (errmsg) {
               *errmsg = "it conflicts with one of the other rules";
            }
            delete rule;
            return kTRUE;
         }
         if (errmsg) {
            *errmsg = "The existing rule is:\n   ";
            r->AsString(*errmsg,"s");
            *errmsg += "\nand the ignored rule is:\n   ";
            rule->AsString(*errmsg);
            *errmsg += ".\n";
         }
         return kFALSE;
      }
   }
   delete rules;

   //---------------------------------------------------------------------------
   // No conflicts - insert the rules
   //---------------------------------------------------------------------------
   if( rule->GetEmbed() )
      fPersistentRules->Add( rule );
   else
      fRemainingRules->Add( rule );
   fAllRules->Add( rule );

   return kTRUE;
}

//------------------------------------------------------------------------------
void TSchemaRuleSet::AsString(TString &out) const
{
   // Fill the string 'out' with the string representation of the rule.
   
   TObjArrayIter it( fAllRules );
   TSchemaRule *rule;
   while( (rule = (TSchemaRule*)it.Next()) ) {
      rule->AsString(out);
      out += "\n";
   }
}

//------------------------------------------------------------------------------
Bool_t TSchemaRuleSet::HasRuleWithSourceClass( const TString &source ) const
{
   // Return True if we have any rule whose source class is 'source'.
   
   TObjArrayIter it( fAllRules );
   TObject *obj;
   while( (obj = it.Next()) ) {
      TSchemaRule* rule = (TSchemaRule*)obj;
      if( rule->GetSourceClass() == source )
         return kTRUE;
   }
   return kFALSE;
}

//------------------------------------------------------------------------------
const TObjArray* TSchemaRuleSet::FindRules( const TString &source ) const
{
   // Return all the rules that are about the given 'source' class.
   // User has to delete the returned array
   TObject*      obj;
   TObjArrayIter it( fAllRules );
   TObjArray*    arr = new TObjArray();
   arr->SetOwner( kFALSE );
   
   while( (obj = it.Next()) ) {
      TSchemaRule* rule = (TSchemaRule*)obj;
      if( rule->GetSourceClass() == source )
         arr->Add( rule );
   }
   return arr;
}

//------------------------------------------------------------------------------
const TSchemaMatch* TSchemaRuleSet::FindRules( const TString &source, Int_t version ) const
{
   // Return all the rules that applies to the specified version of the given 'source' class.
   // User has to delete the returned array

   TObject*      obj;
   TObjArrayIter it( fAllRules );
   TSchemaMatch* arr = new TSchemaMatch();
   arr->SetOwner( kFALSE );

   while( (obj = it.Next()) ) {
      TSchemaRule* rule = (TSchemaRule*)obj;
      if( rule->GetSourceClass() == source && rule->TestVersion( version ) )
         arr->Add( rule );
   }

   if( arr->GetEntriesFast() )
      return arr;
   else {
      delete arr;
      return 0;
   }
}

//------------------------------------------------------------------------------
const TSchemaMatch* TSchemaRuleSet::FindRules( const TString &source, UInt_t checksum ) const
{
   // Return all the rules that applies to the specified checksum of the given 'source' class.
   // User has to delete the returned array

   TObject*      obj;
   TObjArrayIter it( fAllRules );
   TSchemaMatch* arr = new TSchemaMatch();
   arr->SetOwner( kFALSE );

   while( (obj = it.Next()) ) {
      TSchemaRule* rule = (TSchemaRule*)obj;
      if( rule->GetSourceClass() == source && rule->TestChecksum( checksum ) )
         arr->Add( rule );
   }

   if( arr->GetEntriesFast() )
      return arr;
   else {
      delete arr;
      return 0;
   }
}

//------------------------------------------------------------------------------
const TSchemaMatch* TSchemaRuleSet::FindRules( const TString &source, Int_t version, UInt_t checksum ) const
{
   // Return all the rules that applies to the specified version OR checksum of the given 'source' class.
   // User has to delete the returned array

   TObject*      obj;
   TObjArrayIter it( fAllRules );
   TSchemaMatch* arr = new TSchemaMatch();
   arr->SetOwner( kFALSE );

   while( (obj = it.Next()) ) {
      TSchemaRule* rule = (TSchemaRule*)obj;
      if( rule->GetSourceClass() == source && ( rule->TestVersion( version ) || rule->TestChecksum( checksum ) ) )
         arr->Add( rule );
   }

   if( arr->GetEntriesFast() )
      return arr;
   else {
      delete arr;
      return 0;
   }
}

//------------------------------------------------------------------------------
TClass* TSchemaRuleSet::GetClass()
{
   return fClass;
}

//------------------------------------------------------------------------------
UInt_t TSchemaRuleSet::GetClassCheckSum() const
{
   if (fCheckSum == 0 && fClass) {
      const_cast<TSchemaRuleSet*>(this)->fCheckSum = fClass->GetCheckSum();
   }
   return fCheckSum;
}

//------------------------------------------------------------------------------
TString TSchemaRuleSet::GetClassName() const
{
   return fClassName;
}

//------------------------------------------------------------------------------
Int_t TSchemaRuleSet::GetClassVersion() const
{
   return fVersion;
}

//------------------------------------------------------------------------------
const TObjArray* TSchemaRuleSet::GetRules() const
{
   return fAllRules;
}

//------------------------------------------------------------------------------
const TObjArray* TSchemaRuleSet::GetPersistentRules() const
{
   return fPersistentRules;
}

//------------------------------------------------------------------------------
void TSchemaRuleSet::RemoveRule( TSchemaRule* rule )
{
   // Remove given rule from the set - the rule is not being deleted!
   fPersistentRules->Remove( rule );
   fRemainingRules->Remove( rule );
   fAllRules->Remove( rule );
}

//------------------------------------------------------------------------------
void TSchemaRuleSet::RemoveRules( TObjArray* rules )
{
   // remove given array of rules from the set - the rules are not being deleted!
   TObject*      obj;
   TObjArrayIter it( rules );

   while( (obj = it.Next()) ) {
      fPersistentRules->Remove( obj );
      fRemainingRules->Remove( obj );
      fAllRules->Remove( obj );
   }
}

//------------------------------------------------------------------------------
void TSchemaRuleSet::SetClass( TClass* cls )
{
   fClass     = cls;
   fClassName = cls->GetName();
   fVersion   = cls->GetClassVersion();
}


//------------------------------------------------------------------------------
const TSchemaRule* TSchemaMatch::GetRuleWithSource( const TString& name ) const
{
   for( Int_t i = 0; i < GetEntries(); ++i ) {
      TSchemaRule* rule = (ROOT::TSchemaRule*)At(i);
      if( rule->HasSource( name ) ) return rule;
   }
   return 0;
}

//------------------------------------------------------------------------------
const TSchemaRule* TSchemaMatch::GetRuleWithTarget( const TString& name ) const
{
   for( Int_t i=0; i<GetEntries(); ++i) {
      ROOT::TSchemaRule *rule = (ROOT::TSchemaRule*)At(i);
      if( rule->HasTarget( name ) ) return rule;
   }
   return 0;
}

//------------------------------------------------------------------------------
Bool_t TSchemaMatch::HasRuleWithSource( const TString& name, Bool_t needingAlloc ) const
{
   // Return true if the set of rules has at least one rule that has the data
   // member named 'name' as a source.
   // If needingAlloc is true, only the rule that requires the data member to 
   // be cached will be taken in consideration.
   
   for( Int_t i = 0; i < GetEntries(); ++i ) {
      TSchemaRule* rule = (ROOT::TSchemaRule*)At(i);
      if( rule->HasSource( name ) ) {
         if (needingAlloc) {
            const TObjArray *targets = rule->GetTarget();
            if (targets && (targets->GetEntries() > 1 || targets->GetEntries()==0) ) {
               return kTRUE;
            }
            if (targets && name != targets->UncheckedAt(0)->GetName() ) {
               return kTRUE;
            }
            // If the rule has the same source and target and does not
            // have any actions, then it does not need allocation.
            if (rule->GetReadFunctionPointer() || rule->GetReadRawFunctionPointer()) {
               return kTRUE;
            }
         } else {
            return kTRUE;
         }
      }
   }
   return kFALSE;
}

//------------------------------------------------------------------------------
Bool_t TSchemaMatch::HasRuleWithTarget( const TString& name, Bool_t willset ) const
{
   // Return true if the set of rules has at least one rule that has the data
   // member named 'name' as a target.
   // If willset is true, only the rule that will set the value of the data member.
   
   for( Int_t i=0; i<GetEntries(); ++i) {
      ROOT::TSchemaRule *rule = (ROOT::TSchemaRule*)At(i);
      if( rule->HasTarget( name ) ) {
         if (willset) {
            const TObjArray *targets = rule->GetTarget();
            if (targets && (targets->GetEntries() > 1 || targets->GetEntries()==0) ) {
               return kTRUE;
            }
            const TObjArray *sources = rule->GetSource();
            if (sources && (sources->GetEntries() > 1 || sources->GetEntries()==0) ) {
               return kTRUE;
            }
            if (sources && name != sources->UncheckedAt(0)->GetName() ) {
               return kTRUE;
            }            
            // If the rule has the same source and target and does not
            // have any actions, then it will not directly set the value.
            if (rule->GetReadFunctionPointer() || rule->GetReadRawFunctionPointer()) {
               return kTRUE;
            }
         } else {
            return kTRUE;
         }
      }
   }
   return kFALSE;
}

//______________________________________________________________________________
void TSchemaRuleSet::Streamer(TBuffer &R__b)
{
   // Stream an object of class ROOT::TSchemaRuleSet.
   
   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(ROOT::TSchemaRuleSet::Class(),this);
      fAllRules->Clear(); 
      fAllRules->AddAll(fPersistentRules);
   } else {
      GetClassCheckSum();
      R__b.WriteClassBuffer(ROOT::TSchemaRuleSet::Class(),this);
   }
}

