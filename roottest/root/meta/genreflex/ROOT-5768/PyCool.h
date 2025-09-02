// $Id: PyCool_headers.h,v 1.110 2013-04-30 14:09:28 avalassi Exp $
#ifndef DICT_PYCOOL_HEADERS_H
#define DICT_PYCOOL_HEADERS_H 1

// First of all, enable or disable the COOL290 API extensions (bug #92204)
#include "CoolKernel/VersionInfo.h"

// Port to ROOT6 (bug #102630) - remove Reflex dependency
#include "RVersion.h"

// Disable vector payload in PyCool for ROOT6 (workaround for bug #103017)
//#if ROOT_VERSION_CODE >= ROOT_VERSION(5,99,0)
//#undef COOL290VP
//#endif

// Disable inline ASM in Boost
//#if ROOT_VERSION_CODE >= ROOT_VERSION(5,99,0)
//#define BOOST_SP_USE_SPINLOCK 1
//#endif

// Workaround to avoid "_mingw.h: No such file or directory" warning from
// boost/config/platform/win32.hpp (and possiblyavoid any related problems)
#ifdef _WIN32
#ifdef __MINGW32__
#undef __MINGW32__
#endif
#endif

// Workaround for SEAL IntTraits 'integer constant too large for "long" type'
// (gccxml emulation of VC++ needs long long literal constants ending in LL)
// See http://savannah.cern.ch/bugs/?func=detailitem&item_id=9704
//#ifdef _WIN32
//#define LONG_LONG_MAX 0x7fffffffffffffffLL  // max signed __int64 value
//#define LONG_LONG_MIN 0x8000000000000000LL  // min signed __int64 value
//#define ULONG_LONG_MAX 0xffffffffffffffffLL // max unsigned __int64 value
//#endif

// Workaround for 'conversion from void* to const char*' errors
// (the problem occurs within asserts, hence disable all asserts!)
// See http://www.gccxml.org/Bug/bug.php?op=show&bugid=895
// See also SealBase/DebugAids.h:83 (NDEBUG is defined as 1 else problems)
//#ifdef _WIN32
//#define NDEBUG 1
//#include <assert.h>
//#endif

// Disable compilation warning C4345 ('behavior change: an object of POD type
// constructed with an initializer of the form () will be default-initialized')
#ifdef _WIN32
#pragma warning ( disable : 4345 )
#endif

// Disable icc warnings from Reflex code
// Disable icc warning #177: variable was declared but never referenced
// Disable icc warning #279: controlling expression is constant (bug #101369)
// Disable icc warning #444: destructor for base class is not virtual
// Disable icc warning #522: function redeclared "inline" after being called
#ifdef __ICC
#pragma warning ( disable: 177 )
#pragma warning ( disable: 279 )
#pragma warning ( disable: 444 )
#pragma warning ( disable: 522 )
#endif

// Workaround for '__int128' was not declared in this scope' error (bug #94232)
// Add -D__STRICT_ANSI__ for gcc >= 4.7
//#if defined __GNUC__
//#define GCC_VERSION (__GNUC__*10000+__GNUC_MINOR__*100+__GNUC_PATCHLEVEL__)
//#if GCC_VERSION >= 40700
//#ifndef __STRICT_ANSI__
//#define __STRICT_ANSI__
//#endif
//#endif
//#endif

// Standard headers
#include <typeinfo>
#include <vector>
#include <stdexcept>
#include <sstream>

// Boost headers (this is already in pointers.h if needed - task #48846)
//#include "boost/shared_ptr.hpp"

// TODO: (MCl) re-introduce the Python bindings for Coral Date and TimeStamp.
//       They have been removed because causing problems with GCCXML after
//       they have been moved to Boost. (bug #35712)
// CoralBase headers
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeException.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/AttributeListException.h"
#include "CoralBase/AttributeListSpecification.h"
#include "CoralBase/AttributeSpecification.h"
//#include "CoralBase/Date.h"
#include "CoralBase/Exception.h"
//#include "CoralBase/TimeStamp.h"

// CoolKernel headers
#include "CoolKernel/ChannelId.h"
#include "CoolKernel/ChannelSelection.h"
#include "CoolKernel/CompositeSelection.h"
#include "CoolKernel/ConstRecordAdapter.h"
#include "CoolKernel/DatabaseId.h"
#include "CoolKernel/Exception.h"
#include "CoolKernel/FieldSelection.h"
#include "CoolKernel/FieldSpecification.h"
#include "CoolKernel/FolderSpecification.h"
#include "CoolKernel/FolderVersioning.h"
#include "CoolKernel/HvsTagLock.h"
#include "CoolKernel/IDatabase.h"
#include "CoolKernel/IDatabaseSvc.h"
#include "CoolKernel/IField.h"
#include "CoolKernel/IFieldSpecification.h"
#include "CoolKernel/IFolder.h"
#include "CoolKernel/IFolderSet.h"
#include "CoolKernel/IFolderSpecification.h"
#include "CoolKernel/IHvsNode.h"
#include "CoolKernel/IHvsNodeRecord.h"
#include "CoolKernel/IObject.h"
#include "CoolKernel/IObjectIterator.h"
#include "CoolKernel/IRecord.h"
#ifdef COOL290VP
#include "CoolKernel/IRecordIterator.h"
#endif
#include "CoolKernel/IRecordSelection.h"
#include "CoolKernel/IRecordSpecification.h"
#include "CoolKernel/ITime.h"
#ifdef COOL300
#include "CoolKernel/ITransaction.h"
#endif
#ifdef COOL290VP
#include "CoolKernel/PayloadMode.h"
#endif
#include "CoolKernel/Record.h"
#include "CoolKernel/RecordException.h"
#include "CoolKernel/RecordSpecification.h"
#include "CoolKernel/Time.h"
#include "CoolKernel/ValidityKey.h"
#include "CoolKernel/pointers.h"
#include "CoolKernel/types.h"

// CoolApplication headers
#include "CoolApplication/Application.h"
#include "CoolApplication/DatabaseSvcFactory.h"
#include "CoolApplication/IApplication.h"
#include "CoolApplication/MessageLevels.h"

// RelationalCool helpers for PyCool
#include "RelationalCool/src/PyCool_helpers.h"

// Pythonize a subset of CORAL RelationalAccess too
#include "RelationalAccess/IConnectionService.h"
#include "RelationalAccess/IConnectionServiceConfiguration.h"
#include "RelationalAccess/IWebCacheControl.h"
#include "RelationalAccess/IWebCacheInfo.h"

// Workaround for bug #44524 (forward declaration is not enough with ROOT 5.22)
#ifdef _WIN32
#include "RelationalAccess/IConnectionServiceConfiguration.h"
#include "RelationalAccess/IMonitoringReporter.h"
#endif

/*
// Needed to be able to cast Reflex objects (e.g. constants) to python
#include "Reflex/Object.h"
// Reflex changed namespace between ROOT 5.18 and 5.19
#if ROOT_VERSION_CODE <= ROOT_VERSION(5,18,0)
namespace Reflex = ROOT::Reflex;
#endif
*/

// Disable vector payload in PyCool for ROOT6 (workaround for bug #103017)
//#if ROOT_VERSION_CODE >= ROOT_VERSION(5,99,0)
//#ifdef COOL290VP
//#error "COOL290VP should not be defined here?"
//#endif
//#endif

// Disable inline ASM in Boost
//#if ROOT_VERSION_CODE >= ROOT_VERSION(5,99,0)
//#ifndef BOOST_SP_USE_SPINLOCK
//#error "BOOST_SP_USE_SPINLOCK should be defined here?"
//#endif
//#endif

//----------------------------------------------------------------------------

namespace dummy
{
  // Force instantiation for dictionary inclusion
  class _instantiations
  {

  public:

    std::type_info                *aTypeInfo;

    coral::Blob aBlob;
    //coral::Date                   aDate;
    //coral::TimeStamp              aTimeStamp;

    cool::FolderSpecification aFolderSpecification;
    cool::IDatabasePtr aIDatabasePtr;
    cool::IFolderPtr aIFolderPtr;
    cool::IFolderSetPtr aIFolderSetPtr;

    cool::IObjectPtr aIObjectPtr;
    cool::IObjectIteratorPtr aIObjectIteratorPtr;
    cool::IObjectVector aIObjectVector;
    cool::IObjectVector::iterator aIObjectVectorIterator;
    cool::IObjectVectorPtr aIObjectVectorPtr;
    cool::IRecordPtr aIRecordPtr;
    cool::IRecordVector aIRecordVector;
    cool::IRecordVector::iterator aIRecordVectorIterator;
    cool::IRecordVectorPtr aIRecordVectorPtr;
#ifdef COOL300
    cool::ITransactionPtr aITransactionPtr;
#endif
    cool::Record aRecord;
    cool::RecordSpecification aRecordSpecification;
    cool::Time aTime;

    std::vector<cool::ChannelId>  aChannelIdVector;
    /// Marco's workaround for ROOT bug #22513: vector dictionary is not
    /// generated correctly if the vector iterator is included!! See also
    /// Wim's comment in bug #99488, do not generate vector iterator dict!
    //std::vector<cool::ChannelId>::iterator aChannelIdVectorIterator;

    std::map<cool::ChannelId, std::string> aChannelNameMap;
    /// Fix bug #99488: generate dictionaries for map iterators
    std::map<cool::ChannelId, std::string>::iterator aChannelNameMapIt;
    std::map<cool::ChannelId, std::string>::const_iterator aChannelNameMapCIt;

    cool::Application aApplication;

    // Add dictionary for vector<string>: this becomes necessary after
    // removing the Reflex/Object.h header (ROOT6 port - see bug #102630)
    std::vector<std::string> aVectorOfStrings;

  };

  /*
  not used any more, doesn't seem to work
  // This is needed to declare all FieldSelection ctors to genreflex
  template<typename T> void dummyMethod( const T& refValue )
  {
    cool::FieldSelection dummy( "",
                                cool::StorageType::Bool,
                                cool::FieldSelection::EQ,
                                refValue );
  }
  */

}

//----------------------------------------------------------------------------

namespace cool
{
  namespace PyCool
  {
    namespace Helpers
    {

      // Helper for shared pointer to a record
      cool::IRecordPtr IRecordPtr( const cool::IRecordSpecification& spec )
      {
        cool::IRecordPtr recPtr( new cool::Record( spec ) );
        return recPtr;
      }

      cool::IRecordPtr IRecordPtr( const cool::IRecord& rec )
      {
        cool::IRecordPtr recPtr( new cool::Record( rec ) );
        return recPtr;
      }

      // Helper for IObject
      cool::IObjectPtr IObjectPtr( cool::IObject* obj )
      {
        cool::IObjectPtr objPtr( obj );
        return objPtr;
      }

      // Helpers for coral::Blob
      unsigned char getBlobByte(coral::Blob &b, long pos)
      {
        if ( pos < b.size() ) {
          return ((unsigned char*)b.startingAddress())[pos];
        }
        else {
          throw std::out_of_range("coral::Blob index out of range");
        }
        return 0;
      }

      void setBlobByte(coral::Blob &b, long pos, unsigned char val)
      {
        if ( pos < b.size() ) {
          ((unsigned char*)b.startingAddress())[pos] = val;
        }
        else {
          throw std::out_of_range("coral::Blob index out of range");
        }
      }

      // Helper to convert IRecord to its string representation
      std::string toString(cool::IRecord &r)
      {
        std::ostringstream s;
        r.print(s);
        return s.str();
      }

      /*
      // Helper to instantiate a FieldSelection
      // WARNING: this does not show up in dir(gbl.cool.PyCool.Helpers)
      template<typename T> FieldSelection*
      createFieldSelection( const std::string& name,
                            const StorageType::TypeId typeId,
                            const FieldSelection::Relation relation,
                            const T& refValue )
      {
        return new FieldSelection( name, typeId, relation, refValue );
      }
      */

      // Port to ROOT6 (bug #102630) - remove Reflex dependency
      /*
      template <typename T>
      class ReflexCast {
      public:
        static T& Cast(const Reflex::Object& obj){
          return *(reinterpret_cast<T*>(obj.Address()));
        }
      };
      */

      /*
      // Marco: This does not work with ROOT 5.18
      //        It looks like a templated member functions cannot have
      //        "_" or numbers in the name.
      struct Typedefs {
#define TD(X) typedef cool::X X; template <typename T> inline T function_##X() const {return T();}
        TD(Bool);
        TD(UChar);
        //TD(SChar);
        TD(Int16);
        TD(UInt16);
        TD(Int32);
        TD(UInt32);
        TD(UInt63);
        TD(Int64);
        TD(UInt64);
        TD(Float);
        TD(Double);
        TD(String255);
        TD(String4k);
        TD(String64k);
        TD(String16M);
        TD(Blob64k);
        TD(Blob16M);

        TD(ValidityKey);
        TD(ChannelId);
#undef TD
      };
      */

      /* COOLCPPCLEAN-NOINDENT-START */
      struct Typedefs
      {
#define TD(N, X) typedef cool::X X; \
  inline void type_id_##N##__##X () {} \
  template <typename T> inline T function##N() const {return T();}
        TD(a, Bool)
        TD(b, UChar)
        //TD(c, SChar)
        TD(d, Int16)
        TD(e, UInt16)
        TD(f, Int32)
        TD(g, UInt32)
        TD(h, UInt63)
        TD(i, Int64)
        TD(j, UInt64)
        TD(k, Float)
        TD(l, Double)
        TD(m, String255)
        TD(n, String4k)
        TD(o, String64k)
        TD(p, String16M)
        TD(q, Blob64k)
        TD(r, Blob16M)
        TD(s, ValidityKey)
        TD(t, ChannelId)
#undef TD
      };
      /* COOLCPPCLEAN-NOINDENT-END */

      // Port to ROOT6 (bug #102630) - remove Reflex dependency
      // However in ROOT6 this should not be needed at all (bug #102651)
      // Note that ROOT6 beta is called 5.99! (see ROOT-5577)
#if ROOT_VERSION_CODE < ROOT_VERSION(5,99,0)
      const cool::Record& coolMinMax()
      {
        static cool::Record rec;
        if ( rec.size() == 0 )
        {
          // Define a macro to add Min and Max for each type
#define COOL_RECMINMAX_EXTEND2(recminmax,type,type2)                    \
          cool::RecordSpecification spec##type;                         \
          std::string name##type = std::string( #type );                \
          spec##type.extend( name##type + "Min", cool::StorageType::type2 ); \
          spec##type.extend( name##type + "Max", cool::StorageType::type2 ); \
          cool::Record rec##type( spec##type );                         \
          rec##type[ name##type + "Min" ].setValue( cool::type##Min );  \
          try{ rec##type[ name##type + "Max" ].setValue( cool::type##Max ); } \
          catch( ... ) {}                                               \
          recminmax.extend( rec##type )
#define COOL_RECMINMAX_EXTEND1(recminmax,type)        \
          COOL_RECMINMAX_EXTEND2(recminmax,type,type)
          // Add all COOL storage types
          COOL_RECMINMAX_EXTEND1( rec, UChar );
          COOL_RECMINMAX_EXTEND1( rec, Int16 );
          COOL_RECMINMAX_EXTEND1( rec, UInt16 );
          COOL_RECMINMAX_EXTEND1( rec, Int32 );
          COOL_RECMINMAX_EXTEND1( rec, UInt32 );
          COOL_RECMINMAX_EXTEND1( rec, Int64 );
          COOL_RECMINMAX_EXTEND1( rec, UInt63 );
          COOL_RECMINMAX_EXTEND2( rec, UInt64, UInt63 ); // NB sets UInt64Max=0
          COOL_RECMINMAX_EXTEND2( rec, ValidityKey, UInt63 );
        }
        return rec;
      }
#endif

    }
  }
}

//----------------------------------------------------------------------------

// Instantiate all known coral::Attribute types
// NB This is pythonizing CORAL, not COOL!
// ==> Must include _all_ types supported by _CORAL_ (COOL is irrelevant here!)
/* COOLCPPCLEAN-NOINDENT-START */
#define INST_CORAL_ATTR_FUNCT(t) \
  template t& coral::Attribute::data<t>(); \
  template const t & coral::Attribute::data<t>() const; \
  template void coral::Attribute::setValue<t>(const t &)
/* COOLCPPCLEAN-NOINDENT-END */

// Include all types declared in CoralBase/src/AttributeSpecification.cpp
INST_CORAL_ATTR_FUNCT(bool); // cool::Bool
INST_CORAL_ATTR_FUNCT(char); // [NO cool::Char]
// INST_CORAL_ATTR_FUNCT(signed char); // cool::SChar
INST_CORAL_ATTR_FUNCT(unsigned char); // cool::UChar
INST_CORAL_ATTR_FUNCT(short); // cool::Int16
INST_CORAL_ATTR_FUNCT(unsigned short); // cool::UInt16
INST_CORAL_ATTR_FUNCT(int); // cool::Int32
INST_CORAL_ATTR_FUNCT(unsigned int); // cool::UInt32
INST_CORAL_ATTR_FUNCT(long); // [NO cool equivalent]
INST_CORAL_ATTR_FUNCT(unsigned long); // [NO cool equivalent]
INST_CORAL_ATTR_FUNCT(long long); // cool::Int64
INST_CORAL_ATTR_FUNCT(unsigned long long); // cool::SInt64
INST_CORAL_ATTR_FUNCT(float); // cool::Float
INST_CORAL_ATTR_FUNCT(double); // cool::Double
INST_CORAL_ATTR_FUNCT(long double); // [NO cool equivalent]
INST_CORAL_ATTR_FUNCT(std::string); // cool::String255/4k/64k/16M
INST_CORAL_ATTR_FUNCT(coral::Blob); // cool::Blob64k/16M
//INST_CORAL_ATTR_FUNCT(coral::Date); // [NO cool equivalent]
//INST_CORAL_ATTR_FUNCT(coral::TimeStamp); // [NO cool equivalent]

//----------------------------------------------------------------------------

// Instantiate all known cool IField types
// NB This is pythonizing COOL, not CORAL!
// ==> Must include _all_ types supported by _COOL_ (CORAL is irrelevant here!)
/* COOLCPPCLEAN-NOINDENT-START */
#define INST_COOL_IFIELD_FUNCT(t) \
  namespace dummy { \
    cool::FieldSelection* createFieldSelection##t( const std::string& name, \
                                                   const cool::StorageType::TypeId type, \
                                                   const cool::FieldSelection::Relation rel, \
                                                   const cool::t& value ) \
    { return new cool::FieldSelection( name, type, rel, value ); } \
  } \
  template const cool::t& cool::IField::data<cool::t>() const; \
  template void cool::IField::setValue<cool::t>( const cool::t& )
//template class cool::PyCool::Helpers::ReflexCast<cool::t>
/* COOLCPPCLEAN-NOINDENT-END */

// Include all types declared in CoolKernel/CoolKernel/StorageType.h
// (and in CoolKernel/CoolKernel/types.h), avoiding duplicates
INST_COOL_IFIELD_FUNCT(Bool);
// INST_COOL_IFIELD_FUNCT(SChar);
INST_COOL_IFIELD_FUNCT(UChar);
INST_COOL_IFIELD_FUNCT(Int16);
INST_COOL_IFIELD_FUNCT(UInt16);
INST_COOL_IFIELD_FUNCT(Int32);
INST_COOL_IFIELD_FUNCT(UInt32);
//INST_COOL_IFIELD_FUNCT(UInt63);     // error - duplicate
INST_COOL_IFIELD_FUNCT(Int64);
INST_COOL_IFIELD_FUNCT(UInt64);
INST_COOL_IFIELD_FUNCT(Float);
INST_COOL_IFIELD_FUNCT(Double);
INST_COOL_IFIELD_FUNCT(String255);
//INST_COOL_IFIELD_FUNCT(String4k);   // error - duplicate
//INST_COOL_IFIELD_FUNCT(String64k);  // error - duplicate
//INST_COOL_IFIELD_FUNCT(String16M);  // error - duplicate
INST_COOL_IFIELD_FUNCT(Blob64k);
//INST_COOL_IFIELD_FUNCT(Blob16M);    // error - duplicate

// Instantiate all known cool typedefs. This is needed to be able to produce
// a correct mapping between COOL type name and template signature.
/* COOLCPPCLEAN-NOINDENT-START */
#define INST_COOL_TYPEDEFS_HELPER(N, T) \
  template cool::T cool::PyCool::Helpers::Typedefs::function ## N<cool::T>() const
/* COOLCPPCLEAN-NOINDENT-END */
INST_COOL_TYPEDEFS_HELPER(a, Bool);
INST_COOL_TYPEDEFS_HELPER(b, UChar);
// INST_COOL_TYPEDEFS_HELPER(c, SChar);
INST_COOL_TYPEDEFS_HELPER(d, Int16);
INST_COOL_TYPEDEFS_HELPER(e, UInt16);
INST_COOL_TYPEDEFS_HELPER(f, Int32);
INST_COOL_TYPEDEFS_HELPER(g, UInt32);
INST_COOL_TYPEDEFS_HELPER(h, UInt63);
INST_COOL_TYPEDEFS_HELPER(i, Int64);
INST_COOL_TYPEDEFS_HELPER(j, UInt64);
INST_COOL_TYPEDEFS_HELPER(k, Float);
INST_COOL_TYPEDEFS_HELPER(l, Double);
INST_COOL_TYPEDEFS_HELPER(m, String255);
INST_COOL_TYPEDEFS_HELPER(n, String4k);
INST_COOL_TYPEDEFS_HELPER(o, String64k);
INST_COOL_TYPEDEFS_HELPER(p, String16M);
INST_COOL_TYPEDEFS_HELPER(q, Blob64k);
INST_COOL_TYPEDEFS_HELPER(r, Blob16M);
INST_COOL_TYPEDEFS_HELPER(s, ValidityKey);
INST_COOL_TYPEDEFS_HELPER(t, ChannelId);

//----------------------------------------------------------------------------

//#endif
#endif // DICT_PYCOOL_HEADERS_H
