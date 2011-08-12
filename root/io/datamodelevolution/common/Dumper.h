//--------------------------------------------------------------------*- C++ -*-
// file:   Dumper.h
// author: Lukasz Janyst <ljanyst@cern.ch>
//------------------------------------------------------------------------------

#ifndef DUMPER_H
#define DUMPER_H

#include <TClass.h>
#include <TList.h>
#include <TDataMember.h>
#include <TVirtualCollectionProxy.h>
#include <TBaseClass.h>
#include <typeinfo>
#include <string>
#include <iostream>
#include <ostream>
#include <../common/Demangler.h>

//------------------------------------------------------------------------------
// Dump wrapper
//------------------------------------------------------------------------------
template <typename Type>
void dump( Type* obj, std::ostream& out )
{
   const std::type_info& ti( typeid( Type ) );
   TClass*  cl         = TClass :: GetClass( ti );

   if( !cl )
   {
      std :: cout << "[!] Could not get dictionry for class: ";
      std :: cout << getName( ti ) << std :: endl;
      return;
   }

   if( !obj )
   {
      std :: cout << "[!] A null pointer ro object of class: " << cl->GetName();
      std :: cout << std :: endl;
      return;
   }

   TClass*  actClass   = cl->GetActualClass( obj );
   unsigned baseOffset = actClass->GetBaseClassOffset( cl );
   obj = (Type*)(((char *)obj) - baseOffset);
   dump( obj, actClass, out );
}

template <class Type>
void dump( Type *obj, const char *prefix, const char *test_number, unsigned int varnumber, const char *version_number, const char *split)
{
   ofstream out( TString::Format("../logs/%s/%stest%02d_%s%s.log",test_number,prefix,varnumber,version_number,split) );
   dump(obj,out);
}

class Dumper {
public:
   TString fPrefix;
   TString fTestNumber;
   TString fVersionNumber;
   
   Dumper(const char *prefix, const char *test, const char *version) : fPrefix(prefix), fTestNumber(test), fVersionNumber(version) {}
   
   template <class Type>
   void dump( Type *obj, unsigned int varnumber, const char *split)
   {
     ofstream out( TString::Format("../logs/%s/%stest%02d_%s%s.log",fTestNumber.Data(),fPrefix.Data(),varnumber,fVersionNumber.Data(),split) );
      ::dump(obj,out);
   }
};

//------------------------------------------------------------------------------
// Dump all the primitive members of the class
//------------------------------------------------------------------------------
void dump( void* obj, TClass* cl, std::ostream& out, std :: string prefix = "" )
{
   using namespace std;

   const char* clName = cl->GetName();
   //--------------------------------------------------------------------------
   // If the object is a nulle pointer then ignore it
   //--------------------------------------------------------------------------
   if( !obj )
   {
      out << "[i] A null pointer to object of class: " << clName<< endl;
      return;
   }

   out << prefix << "Processing object of class: " << clName << endl;

   //---------------------------------------------------------------------------
   // Check if we're dealing with a collection
   //---------------------------------------------------------------------------
   TVirtualCollectionProxy *proxy;
   if( (proxy = cl->GetCollectionProxy()) )
   {
      //------------------------------------------------------------------------
      // A collection of objects?
      //------------------------------------------------------------------------
      TClass *vcl;
      if( (vcl = proxy->GetValueClass()) )
      {
         TVirtualCollectionProxy::TPushPop helper( proxy, obj );
         out << prefix << "Length = " << proxy->Size() << endl;
         //---------------------------------------------------------------------
         // Deal with pointers
         //---------------------------------------------------------------------
         if( proxy->HasPointers() )
         {
            for( UInt_t i = 0; i < proxy->Size(); ++i )
            {
               out << prefix << "[ele] [" << i << "] =";
               if( *(char**)proxy->At(i) == 0 )
                  out << " 0" << endl;
               else
               {
                  out << endl;
                  char* ele = *(char**)proxy->At(i);
                  TClass* actClass = vcl->GetActualClass( ele );
                  ele -= actClass->GetBaseClassOffset( vcl );
                  dump( ele, actClass, out, prefix + "  " );
               }
               
            }
         }
         //---------------------------------------------------------------------
         // Deal with ordinary objects
         //---------------------------------------------------------------------
         else
            for( UInt_t i = 0; i < proxy->Size(); ++i )
            {
               out << prefix << "[ele] [" << i << "] = " << endl;
               dump( proxy->At( i ), proxy->GetValueClass(), out,
                     prefix + "  " );
            }
      }
      else
      {
         TVirtualCollectionProxy::TPushPop helper( proxy, obj );
         out << prefix << "Length = " << proxy->Size() << endl;
         switch( proxy->GetType() )
         {
            case kInt_t:
               for( UInt_t i = 0; i < proxy->Size(); ++i )
               {
                  out << prefix << "[ele] [" << i << "] = ";
                  out << *((int*)proxy->At(i)) << endl;
               }
               break;

           case kShort_t:
               for( UInt_t i = 0; i < proxy->Size(); ++i )
               {
                  out << prefix << "[ele] [" << i << "] = ";
                  out << *((short*)proxy->At(i)) << endl;
               }
               break;

           case kDouble_t:
               for( UInt_t i = 0; i < proxy->Size(); ++i )
               {
                  out << prefix << "[ele] [" << i << "] = ";
                  out << *((double*)proxy->At(i)) << endl;
               }
               break;

           case kFloat_t:
               for( UInt_t i = 0; i < proxy->Size(); ++i )
               {
                  out << prefix << "[ele] [" << i << "] = ";
                  out << *((float*)proxy->At(i)) << endl;
               }
               break;
           default:
               cout << "Case not handled in test dump(): " << proxy->GetType() << std::endl;

         }
      }
      out << prefix << "End of " << clName << endl; 
      return;
   }

   //---------------------------------------------------------------------------
   // Process base classes
   //---------------------------------------------------------------------------
   if( !cl->GetListOfBases()->IsEmpty() )
   {
      out << prefix << "Processing bases of: " << clName << endl;
      TIter next( cl->GetListOfBases() );
      TBaseClass* base;

      //------------------------------------------------------------------------
      // Loop over bases
      //------------------------------------------------------------------------
      while( (base = (TBaseClass*)next()) )
      {
         TClass* baseCl  = base->GetClassPointer();
         void*   basePtr = ((char*)obj) + cl->GetBaseClassOffset( baseCl );
         dump( basePtr, baseCl, out, prefix + "  " );
      }
      out << prefix << "End of bases of: " << clName << endl;
   }

   //---------------------------------------------------------------------------
   // Get list of data members
   //---------------------------------------------------------------------------
   TIter next( cl->GetListOfDataMembers() );
   TDataMember* member;
   while( (member = (TDataMember*)next()) )
   {
      out << prefix << "[mem] ";
      if( member->IsPersistent() ) out << "p ";
      else out << "t ";
      out << member->GetTrueTypeName() << " " << member->GetName();

      TClass *mcl = TClass :: GetClass( member->GetTrueTypeName() );
      out << " = ";

      //------------------------------------------------------------------------
      // Process basic types
      //------------------------------------------------------------------------
      if( member->IsBasic() )
      {
         Int_t datatype = member->GetDataType()->GetType();
         const char *type = member->GetTrueTypeName();
         switch( datatype )
         {
            case kInt_t:
               out << *(int *)(((char *)obj) + member->GetOffset());
               break;
            case kUInt_t:
               out << *(unsigned int *)(((char *)obj) + member->GetOffset());
               break;
            case kDouble_t:
            case kDouble32_t:
               out << *(double *)(((char *)obj) + member->GetOffset());
               break;
            case kFloat_t:
            case kFloat16_t:
               out << *(float *)(((char *)obj) + member->GetOffset());
               break;
            case kShort_t:
               out << *(short *)(((char *)obj) + member->GetOffset());
               break;
            case kBool_t:
               out << *(bool *)(((char *)obj) + member->GetOffset()); 
               //debug: << ' ' << (void*)obj << ' ' << member->GetOffset() << ' ' << (void*)(((char *)obj) + member->GetOffset());
               break;
            default:
               out << "unsupported type: " << type << " (" << datatype << ")";
         }
         out << endl;
      }
      //------------------------------------------------------------------------
      // Deal with objects
      //------------------------------------------------------------------------
      else if( mcl )
      {
         char *ptr   = ((char *)obj) + member->GetOffset();
         if( member->IsaPointer() && *((char**)ptr) == 0 )
            out << " = 0" << endl;
         else
         {
            out << endl;

            if( member->IsaPointer() )
            {
               char*   mem      = *(char**)ptr;
               TClass* actClass = mcl->GetActualClass( mem );
               mem -= actClass->GetBaseClassOffset( mcl );
               dump( mem, actClass, out, prefix + "  " );
            }
            else
               dump( ptr, mcl, out, prefix + "  " );
         }
      }
   }
   out << prefix << "End of " << clName << endl; 
}

#endif // DUMPER_H
