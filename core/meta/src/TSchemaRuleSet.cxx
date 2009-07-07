// @(#)root/core:$Id$
// author: Lukasz Janyst <ljanyst@cern.ch>

#include "TSchemaRuleSet.h"
#include "TSchemaRule.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TClass.h"

ClassImp(TSchemaRule)

using namespace ROOT;

//------------------------------------------------------------------------------
TSchemaRuleSet::TSchemaRuleSet(): fPersistentRules( 0 ), fRemainingRules( 0 ),
                                  fAllRules( 0 ), fVersion(-3), fCheckSum( 0 )
{
   fPersistentRules = new TObjArray();
   fRemainingRules  = new TObjArray();
   fAllRules        = new TObjArray();
   fAllRules->SetOwner( kFALSE );
}

//------------------------------------------------------------------------------
TSchemaRuleSet::~TSchemaRuleSet()
{
   delete fPersistentRules;
   delete fRemainingRules;
   delete fAllRules;
}

//------------------------------------------------------------------------------
Bool_t TSchemaRuleSet::AddRule( TSchemaRule* rule, Bool_t checkConsistency )
{
   // The consistency check always fails if the TClass object was not set!

   //---------------------------------------------------------------------------
   // Cannot verify the consistency if the TClass object is not present
   //---------------------------------------------------------------------------
   if( checkConsistency && !fClass )
      return kFALSE;

   if( !rule->IsValid() )
      return kFALSE;

   //---------------------------------------------------------------------------
   // If we don't check the consistency then we should just add the object
   //---------------------------------------------------------------------------
   if( !checkConsistency ) {
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
   if( rule->GetTarget() ) {
      TObjArrayIter titer( rule->GetTarget() );
      while( (obj = titer.Next()) ) {
         TObjString* str = (TObjString*)obj;
         if( !fClass->GetDataMember( str->GetString() ) && !fClass->GetBaseClass( str->GetString() ) )
            return kFALSE;
 
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
Bool_t TSchemaMatch::HasRuleWithSource( const TString& name ) const
{
   for( Int_t i = 0; i < GetEntries(); ++i ) {
      TSchemaRule* rule = (ROOT::TSchemaRule*)At(i);
      if( rule->HasSource( name ) ) return kTRUE;
   }
   return kFALSE;
}

//------------------------------------------------------------------------------
Bool_t TSchemaMatch::HasRuleWithTarget( const TString& name ) const
{
   for( Int_t i=0; i<GetEntries(); ++i) {
      ROOT::TSchemaRule *rule = (ROOT::TSchemaRule*)At(i);
      if( rule->HasTarget( name ) ) return kTRUE;
   }
   return kFALSE;
}

//______________________________________________________________________________
void TSchemaRuleSet::Streamer(TBuffer &R__b)
{
   // Stream an object of class ROOT::TSchemaRuleSet.
   
   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(ROOT::TSchemaRuleSet::Class(),this);
   } else {
      GetClassCheckSum();
      R__b.WriteClassBuffer(ROOT::TSchemaRuleSet::Class(),this);
   }
}

