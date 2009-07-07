// @(#)root/core:$Id$
// author: Lukasz Janyst <ljanyst@cern.ch>

#include "TSchemaRule.h"
#include "TSchemaRuleProcessor.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TNamed.h"
#include <utility>
#include <iostream>
#include <vector>
#include <list>
#include <string>
#include <cstdlib>
#include "TROOT.h"
#include "Riostream.h"

#include "RConversionRuleParser.h"

ClassImp(TSchemaRule)

using namespace ROOT;

//------------------------------------------------------------------------------
TSchemaRule::TSchemaRule(): fVersionVect( 0 ), fChecksumVect( 0 ),
                            fTargetVect( 0 ), fSourceVect( 0 ),
                            fIncludeVect( 0 ), fEmbed( kTRUE ), 
                            fReadFuncPtr( 0 ), fReadRawFuncPtr( 0 ),
                            fRuleType( kNone )
{
   // Default Constructor.
   
}

//------------------------------------------------------------------------------
TSchemaRule::~TSchemaRule()
{
   // Destructor.
   
   delete fVersionVect;
   delete fChecksumVect;
   delete fTargetVect;
   delete fSourceVect;
   delete fIncludeVect;
}

//------------------------------------------------------------------------------
TSchemaRule::TSchemaRule( const TSchemaRule& rhs ): TObject( rhs )
{
   // Copy Constructor.
   *this = rhs;
}

//------------------------------------------------------------------------------
TSchemaRule& TSchemaRule::operator = ( const TSchemaRule& rhs )
{
   // Copy operator.
   
   if( this != &rhs ) {
      fVersion        = rhs.fVersion;
      fChecksum       = rhs.fChecksum;
      fSourceClass    = rhs.fSourceClass;
      fTarget         = rhs.fTarget;
      fSource         = rhs.fSource;
      fInclude        = rhs.fInclude;
      fCode           = rhs.fCode;
      fEmbed          = rhs.fEmbed;
      fReadFuncPtr    = rhs.fReadFuncPtr;
      fReadRawFuncPtr = rhs.fReadRawFuncPtr;
      fRuleType       = rhs.fRuleType;
   }
   return *this;
}

//------------------------------------------------------------------------------
void TSchemaRule::ls(Option_t *targetname) const
{
   // The ls function lists the contents of a class on stdout. Ls output
   // is typically much less verbose then Dump().
   
   TROOT::IndentLevel();
   cout <<"Schema Evolution Rule: \n";
   TROOT::IndentLevel();
   cout << "sourceClass=\"" << fSourceClass << "\" "
   << "version=\"" << fVersion << "\" "
   << "checksum=\"" << fChecksum << "\" ";
   if (targetname && targetname[0]) cout << "targetClass=\"" << targetname << "\" ";
   cout << "\n";
   TROOT::IndentLevel();
   cout << "source=\"" << fSource << "\" ";
   cout << "target=\"" << fTarget << "\" ";
   cout << "\n";
   TROOT::IndentLevel();
   cout << "include=\"" << fInclude << "\" "
   << "\n";
   TROOT::IndentLevel();
   cout << "code=\"" << fCode << "\" "
   << "\n";
}

//------------------------------------------------------------------------------
Bool_t TSchemaRule::SetVersion( const TString& version )
{
   // Set the version string - returns kFALSE if the format is incorrect

   fVersion = "";
   Bool_t ret = ProcessVersion( version );
   if( ret )
      fVersion = version;
   return ret;
}

//------------------------------------------------------------------------------
Bool_t TSchemaRule::TestVersion( Int_t version ) const
{
   // Check if given version number is defined in this rule

   if( fVersion == "" )
      return kFALSE;

   if( !fVersionVect )
      ProcessVersion( fVersion ); // At this point the version string should always be correct

   std::vector<std::pair<Int_t, Int_t> >::iterator it;
   for( it = fVersionVect->begin(); it != fVersionVect->end(); ++it ) {
      if( version >= it->first && version <= it->second )
         return kTRUE;
   }
   return kFALSE;
}

//------------------------------------------------------------------------------
Bool_t TSchemaRule::SetChecksum( const TString& checksum )
{
   // Set the checksum string - returns kFALSE if the format is incorrect
   fChecksum = "";
   Bool_t ret = ProcessChecksum( checksum );
   if( ret )
      fChecksum = checksum;
   return ret;
}

//------------------------------------------------------------------------------
Bool_t TSchemaRule::TestChecksum( UInt_t checksum ) const
{
   // Check if given checksum is defined in this rule

   if( fChecksum == "" )
      return kFALSE;

   if( !fChecksumVect )
      ProcessChecksum( fChecksum ); // At this point the checksum string should always be correct

   std::vector<UInt_t>::iterator it;
   for( it = fChecksumVect->begin(); it != fChecksumVect->end(); ++it ) {
      if( checksum == *it )
         return kTRUE;
   }
   return kFALSE;
}

//------------------------------------------------------------------------------
void TSchemaRule::SetSourceClass( const TString& classname )
{
   // Set the source class of this rule (i.e. the onfile class).

   fSourceClass = classname;
}

//------------------------------------------------------------------------------
TString TSchemaRule::GetSourceClass() const
{
   // Set the source class of this rule (i.e. the onfile class).

   return fSourceClass;
}

//------------------------------------------------------------------------------
void TSchemaRule::SetTarget( const TString& target )
{
   // Set the target class of this rule (i.e. the in memory class).

   fTarget = target;

   if( target == "" ) {
      delete fTargetVect;
      fTargetVect = 0;
      return;
   }

   if( !fTargetVect )
      fTargetVect = new TObjArray();

   ProcessList( fTargetVect, target );
}

//------------------------------------------------------------------------------
const TObjArray*  TSchemaRule::GetTarget() const
{
   // Get the target class of this rule (i.e. the in memory class).

   if( fTarget == "" )
      return 0;

   if( !fTargetVect ) {
      fTargetVect = new TObjArray();
      ProcessList( fTargetVect, fTarget );
   }

   return fTargetVect;
}

//------------------------------------------------------------------------------
void TSchemaRule::SetSource( const TString& source )
{
   // Set the list of source members.  This should be in the form of a declaration:
   //     Int_t fOldMember; TNamed fName;
   
   fSource = source;

   if( source == "" ) {
      delete fSourceVect;
      fSourceVect = 0;
      return;
   }

   if( !fSourceVect )
      fSourceVect = new TObjArray();

   ProcessDeclaration( fSourceVect, source );
}

//------------------------------------------------------------------------------
const TObjArray* TSchemaRule::GetSource() const
{
   // Get the list of source members as a TObjArray of TNamed object,
   // with the name being the member name and the title being its type.
   
   if( fSource == "" )
      return 0;

   if( !fSourceVect ) {
      fSourceVect = new TObjArray();
      ProcessDeclaration( fSourceVect, fSource );
   }
   return fSourceVect;
}

//------------------------------------------------------------------------------
void TSchemaRule::SetInclude( const TString& incl )
{
   // Set the comma separated list of header files to include to be able
   // to compile this rule.
   
   fInclude = incl;

   if( incl == "" ) {
      delete fIncludeVect;
      fIncludeVect = 0;
      return;
   }

   if( !fIncludeVect )
      fIncludeVect = new TObjArray();

   ProcessList( fIncludeVect, incl );
}

//------------------------------------------------------------------------------
const TObjArray* TSchemaRule::GetInclude() const
{
   // Return the list of header files to include to be able to
   // compile this rule as a TObjArray of TObjString
   
   if( fInclude == "" )
      return 0;

   if( !fIncludeVect ) {
      fIncludeVect = new TObjArray();
      ProcessList( fIncludeVect, fInclude );
   }

   return fIncludeVect;
}

//------------------------------------------------------------------------------
void TSchemaRule::SetEmbed( Bool_t embed )
{
   // Set whether this rule should be save in the ROOT file (if true)
   
   fEmbed = embed;
}

//------------------------------------------------------------------------------
Bool_t TSchemaRule::GetEmbed() const
{
   // Return true if this rule should be saved in the ROOT File.
   
   return fEmbed;
}

//------------------------------------------------------------------------------
Bool_t TSchemaRule::IsValid() const
{
   // Return kTRUE if this rule is valid.
   
   return (fVersionVect || fChecksumVect) && (fSourceClass.Length() != 0); 
}

//------------------------------------------------------------------------------
void TSchemaRule::SetCode( const TString& code )
{
   // Set the source code of this rule.
   
   fCode = code;
}

//------------------------------------------------------------------------------
TString TSchemaRule::GetCode() const
{
   // Get the source code of this rule.
   
   return fCode;
}

//------------------------------------------------------------------------------
Bool_t TSchemaRule::HasTarget( const TString& target ) const
{
   // Return true if one of the rule's data member target  is 'target'.
   
   if( !fTargetVect )
      return kFALSE;

   TObject*      obj;
   TObjArrayIter it( fTargetVect );
   while( (obj = it.Next()) ) {
      TObjString* str = (TObjString*)obj;
      if( str->GetString() == target )
         return kTRUE;
   }
   return kFALSE;
}

//------------------------------------------------------------------------------
Bool_t TSchemaRule::HasSource( const TString& source ) const
{
   // Return true if one of the rule's data member source is 'source'
   if( !fSourceVect )
      return kFALSE;

   TObject*      obj;
   TObjArrayIter it( fSourceVect );
   while( (obj = it.Next()) ) {
      TNamed* var = (TNamed*)obj;
      if( var->GetName() == source )
         return kTRUE;
   }
   return kFALSE;
}

//------------------------------------------------------------------------------
void TSchemaRule::SetReadFunctionPointer( TSchemaRule::ReadFuncPtr_t ptr )
{
   // Set the pointer to the function to be run for the rule (if it is a read rule).
   
   fReadFuncPtr = ptr;
}

//------------------------------------------------------------------------------
TSchemaRule::ReadFuncPtr_t TSchemaRule::GetReadFunctionPointer() const
{
   // Get the pointer to the function to be run for the rule (if it is a read rule).

   return fReadFuncPtr;
}

//------------------------------------------------------------------------------
void TSchemaRule::SetReadRawFunctionPointer( TSchemaRule::ReadRawFuncPtr_t ptr )
{
   // Set the pointer to the function to be run for the rule (if it is a raw read rule).

   fReadRawFuncPtr = ptr;
}

//------------------------------------------------------------------------------
TSchemaRule::ReadRawFuncPtr_t TSchemaRule::GetReadRawFunctionPointer() const
{
   // Get the pointer to the function to be run for the rule (if it is a raw read rule).

   return fReadRawFuncPtr;
}

//------------------------------------------------------------------------------
void TSchemaRule::SetRuleType( TSchemaRule::RuleType_t type )
{
   // Set the type of the rule.
   
   fRuleType = type;
}

//------------------------------------------------------------------------------
Bool_t TSchemaRule::IsAliasRule() const
{
   // Return kTRUE if the rule is a strict renaming of one of the data member of the class.

   return fSourceClass != "" && (fVersion != "" || fChecksum != "") && fTarget == "" && fSource == "" && fInclude == "" && fCode == "";
}

//------------------------------------------------------------------------------
Bool_t TSchemaRule::IsRenameRule() const
{
   // Return kTRUE if the rule is a strict renaming of the class to a new name.

   return fSourceClass != "" && (fVersion != "" || fChecksum != "") && fTarget != "" && fSource != "" && fInclude == "" && fCode == "";
}

//------------------------------------------------------------------------------
TSchemaRule::RuleType_t TSchemaRule::GetRuleType() const
{
   // Return the type of the rule.
   
   return fRuleType;
}

//------------------------------------------------------------------------------
Bool_t TSchemaRule::Conflicts( const TSchemaRule* rule ) const
{
   // Check if this rule conflicts with the given one.

   //---------------------------------------------------------------------------
   // If the rules have different sources then the don't conflict
   //---------------------------------------------------------------------------
   if( GetSourceClass() != rule->GetSourceClass() )
      return kFALSE;

   //---------------------------------------------------------------------------
   // Check if the rules have common target
   //---------------------------------------------------------------------------
   if( !rule->GetTarget() )
      return kFALSE;

   Bool_t         haveCommonTargets = kFALSE;
   TObjArrayIter  titer( rule->GetTarget() );
   TObjString    *str;
   TObject       *obj;

   while( (obj = titer.Next() ) ) {
      str = (TObjString*)obj;
      if( HasTarget( str->String() ) )
         haveCommonTargets = kTRUE;
   }

   if( !haveCommonTargets )
      return kFALSE;

   //---------------------------------------------------------------------------
   // Check if there are conflicting checksums
   //---------------------------------------------------------------------------
   if( fChecksumVect ) {
      std::vector<UInt_t>::iterator it;
      for( it = fChecksumVect->begin(); it != fChecksumVect->end(); ++it )
         if( rule->TestChecksum( *it ) )
            return kTRUE;
   }

   //---------------------------------------------------------------------------
   // Check if there are conflicting versions
   //---------------------------------------------------------------------------
   if( fVersionVect && rule->fVersionVect )
   {
      std::vector<std::pair<Int_t, Int_t> >::iterator it1;
      std::vector<std::pair<Int_t, Int_t> >::iterator it2;
      for( it1 = fVersionVect->begin(); it1 != fVersionVect->end(); ++it1 ) {
         for( it2 = rule->fVersionVect->begin();
              it2 != rule->fVersionVect->end(); ++it2 ) {
            //------------------------------------------------------------------
            // the rules conflict it their version ranges intersect
            //------------------------------------------------------------------
            if( it1->first >= it2->first && it1->first <= it2->second )
               return kTRUE;

            if( it1->first < it2->first && it1->second >= it2->first )
               return kTRUE;
         }
      }
   }
   return kFALSE;
}

//------------------------------------------------------------------------------
Bool_t TSchemaRule::ProcessVersion( const TString& version ) const
{
   // Check if specified version string is correct and build version vector.

   //---------------------------------------------------------------------------
   // Check if we have valid list
   //---------------------------------------------------------------------------
   if( version[0] != '[' || version[version.Length()-1] != ']' )
      return kFALSE;
   std::string ver = version.Data();

   std::list<std::string> versions;
   ROOT::TSchemaRuleProcessor::SplitList( ver.substr( 1, ver.size()-2), versions );

   if( versions.empty() )
   {
      delete fVersionVect;
      fVersionVect = 0;
      return kFALSE;
   }

   if( !fVersionVect )
      fVersionVect = new std::vector<std::pair<Int_t, Int_t> >;
   fVersionVect->clear();

   //---------------------------------------------------------------------------
   // Check the validity of each list element
   //---------------------------------------------------------------------------
   std::list<std::string>::iterator it;
   for( it = versions.begin(); it != versions.end(); ++it ) {
      std::pair<Int_t, Int_t> verpair;
      if( !ROOT::TSchemaRuleProcessor::ProcessVersion( *it, verpair ) )
      {
         delete fVersionVect;
         fVersionVect = 0;
         return kFALSE;
      }
      fVersionVect->push_back( verpair );
   }
   return kTRUE;
}

//------------------------------------------------------------------------------
Bool_t TSchemaRule::ProcessChecksum( const TString& checksum ) const
{
   // Check if specified checksum string is correct and build checksum vector.
   
   //---------------------------------------------------------------------------
   // Check if we have valid list
   //---------------------------------------------------------------------------
   std::string chk = (const char*)checksum;
   if( chk[0] != '[' || chk[chk.size()-1] != ']' )
      return kFALSE;

   std::list<std::string> checksums;
   ROOT::TSchemaRuleProcessor::SplitList( chk.substr( 1, chk.size()-2), checksums );

   if( checksums.empty() ) {
      delete fChecksumVect;
      fChecksumVect = 0;
      return kFALSE; 
   }

   if( !fChecksumVect )
      fChecksumVect = new std::vector<UInt_t>;
   fChecksumVect->clear();

   //---------------------------------------------------------------------------
   // Check the validity of each list element
   //---------------------------------------------------------------------------
   std::list<std::string>::iterator it;
   for( it = checksums.begin(); it != checksums.end(); ++it ) {
      if( !ROOT::TSchemaRuleProcessor::IsANumber( *it ) ) {
         delete fChecksumVect;
         fChecksumVect = 0;
         return kFALSE;
      }
      fChecksumVect->push_back( atoi( it->c_str() ) );
   }
   return kTRUE;
}

//------------------------------------------------------------------------------
void TSchemaRule::ProcessList( TObjArray* array, const TString& list )
{
   // Split the list as a comma separated list into a TObjArray of TObjString.

   std::list<std::string>           elems;
   std::list<std::string>::iterator it;
   ROOT::TSchemaRuleProcessor::SplitList( (const char*)list, elems );

   array->Clear();

   if( elems.empty() )
      return;

   for( it = elems.begin(); it != elems.end(); ++it ) {
      TObjString *str = new TObjString;
      *str = it->c_str();
      array->Add( str );
   }
}

//------------------------------------------------------------------------------
void TSchemaRule::ProcessDeclaration( TObjArray* array, const TString& list )
{
   // Split the list as a declaration into as a TObjArray of TNamed(name,type).

   std::list<std::pair<std::string,std::string> >           elems;
   std::list<std::pair<std::string,std::string> >::iterator it;
   ROOT::TSchemaRuleProcessor::SplitDeclaration( (const char*)list, elems );

   array->Clear();

   if( elems.empty() )
      return;

   for( it = elems.begin(); it != elems.end(); ++it ) {
      TNamed *type = new TNamed( it->second.c_str(), it->first.c_str() ) ;
      array->Add( type );
   }
}

#if 0
//------------------------------------------------------------------------------
Bool_t TSchemaRule::GenerateFor( TStreamerInfo *info )
{
   // Generate the actual function for the rule.

   String funcname = fSourceClass + "_to_" + fTargetClass;
   if (info) funcname += "_v" + info->GetClassVersion();
   TString names = fSource + "_" + fTarget; 
   name.ReplaceAll(',','_');
   name.ReplaceAll(':','_');
   funcname += "_" + name;

   String filename = funcname + ".C";
   if (!false) {
      filename += '+';
   }

   std::ofstream fileout(filename);


      ROOT::WriteReadRawRuleFunc( *rIt, 0, mappedname, nameTypeMap, fileout );
      ROOT::WriteReadRuleFunc( *rIt, 0, mappedname, nameTypeMap, fileout );

   gROOT->LoadMacro(filename);
}

#endif
