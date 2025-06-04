//--------------------------------------------------------------------*- C++ -*-
// file:   Generator.h
// date:   26.05.2008
// author: Lukasz Janyst <ljanyst@cern.ch>
//------------------------------------------------------------------------------

#ifndef GENERATOR_H
#define GENERATOR_H

#include <TClass.h>
#include <TList.h>
#include <TDataMember.h>
#include <TBaseClass.h>
#include <TVirtualCollectionProxy.h>
#include <typeinfo>
#include <iostream>
#include <cstdlib>
#include <TROOT.h>
#include <TClassTable.h>
#include <TRandom.h>
#include "TError.h"

#include "Demangler.h"

//------------------------------------------------------------------------------
// Random number generators
//------------------------------------------------------------------------------
double genDouble()
{
   return gRandom->Rndm(10); // (10.0 * (random() / ((double)RAND_MAX + 1.0)));
}

float genFloat()
{
   return genDouble();
}


bool genBool( )
{
   return gRandom->Integer(2) >= 1; // 1+(int)(max * (random() / ((double)RAND_MAX + 1.0)));
}

int genInt( int max = 30 )
{
   return gRandom->Integer(max) + 1; // 1+(int)(max * (random() / ((double)RAND_MAX + 1.0)));
}

unsigned int genUInt( int max = 30 )
{
   return gRandom->Integer(max) + 1; // 1+(int)(max * (random() / ((double)RAND_MAX + 1.0)));
}

short genShort()
{
   return genInt();
}

//------------------------------------------------------------------------------
// Create a list of concrete classes deriving from given class
//------------------------------------------------------------------------------
std::vector<TClass*> getConcreteClasses( TClass *base )
{
   Int_t oldlevel = gErrorIgnoreLevel;
   // Hide the warning about the missing pair dictionary.
   gErrorIgnoreLevel = kError;

   std::vector<TClass*> ret;
   for( int i = 0; i < gClassTable->Classes(); ++i )
   {
      //------------------------------------------------------------------------
      // Eliminate the ROOT classes
      //------------------------------------------------------------------------
      if( *gClassTable->At(i) == 'T' || *gClassTable->At(i) == 'R' )
         continue;

      //------------------------------------------------------------------------
      // Check the base
      //------------------------------------------------------------------------
      TClass* cl = TClass :: GetClass( gClassTable->At( i ) );
      if( cl && !(cl->Property() & kIsAbstract) && cl->GetBaseClass( base->GetName() ) )
         ret.push_back( cl );
   }
   gErrorIgnoreLevel = oldlevel;
   return ret;
}

//------------------------------------------------------------------------------
// Dump wrapper
//------------------------------------------------------------------------------
template <typename Type>
void generate( Type*& obj )
{
   using namespace std;
   const type_info& ti( typeid( Type ) );
   TClass *cl = TClass :: GetClass( ti );
   if( !cl )
   {
      cerr << "[!] Unable to find dictionary for: " << getName( ti ) << endl;
      return;
   }

   //---------------------------------------------------------------------------
   // Create an object
   //---------------------------------------------------------------------------
   obj = (Type*)cl->New();
   generate( obj, cl );
}

//------------------------------------------------------------------------------
// Dump all the primitive members of the class
//------------------------------------------------------------------------------
void generate( void* obj, TClass* cl )
{
   using namespace std;

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
         proxy->Allocate( genInt(), false );

         //---------------------------------------------------------------------
         // A collection of pointers?
         //---------------------------------------------------------------------
         if( proxy->HasPointers() )
         {
            //------------------------------------------------------------------
            // Get all the concrete classes deriving from the contained class
            //------------------------------------------------------------------
            vector<TClass*> cls = getConcreteClasses( vcl );
            for( UInt_t i = 0; i < proxy->Size(); ++i )
            {
               //---------------------------------------------------------------
               // Choose the class
               //---------------------------------------------------------------
               int ind = genInt( cls.size()+1 ) - 1;

               //---------------------------------------------------------------
               // Zero pointer
               //---------------------------------------------------------------
               if( ind == 0 )
               {
                  *((void**)proxy->At(i)) = 0;
                  continue;
               }
               //---------------------------------------------------------------
               // Generate one of the base classes
               //---------------------------------------------------------------
               --ind;
               TClass* rcl = cls[ind];
               char* newObj = (char*)rcl->New();
               *((void**)proxy->At(i)) = newObj + rcl->GetBaseClassOffset( vcl );
               generate( newObj, rcl );
            }

         }
         //---------------------------------------------------------------------
         // Collection of non-pointer
         //---------------------------------------------------------------------
         else
            for( UInt_t i = 0; i < proxy->Size(); ++i )
               generate( proxy->At( i ), proxy->GetValueClass() );
      }
      //------------------------------------------------------------------------
      // Collection of base types
      //------------------------------------------------------------------------
      else
      {
         TVirtualCollectionProxy::TPushPop helper( proxy, obj );
         proxy->Allocate( genInt(), false );
         switch( proxy->GetType() )
         {
            case kInt_t:
               for( UInt_t i = 0; i < proxy->Size(); ++i )
                  *((int*)proxy->At(i)) = genInt();
               break;

           case kShort_t:
               for( UInt_t i = 0; i < proxy->Size(); ++i )
                  *((short*)proxy->At(i)) = genShort();
               break;

           case kDouble_t:
               for( UInt_t i = 0; i < proxy->Size(); ++i )
                  *((double*)proxy->At(i)) = genDouble();
               break;

           case kFloat_t:
               for( UInt_t i = 0; i < proxy->Size(); ++i )
                  *((float*)proxy->At(i)) = genFloat();
               break;
           default:
               std::cout << "Case not handled in test generate():" << proxy->GetType() << std::endl;
         }
      }
      return;
   }

   //---------------------------------------------------------------------------
   // Process base classes
   //---------------------------------------------------------------------------
   if( !cl->GetListOfBases()->IsEmpty() )
   {
      TIter next( cl->GetListOfBases() );
      TBaseClass* base;

      //------------------------------------------------------------------------
      // Loop over bases
      //------------------------------------------------------------------------
      while( ( base = (TBaseClass*)next() ) )
      {
         TClass* baseCl  = base->GetClassPointer();
         void*   basePtr = ((char*)obj) + cl->GetBaseClassOffset( baseCl );
         generate( basePtr, baseCl );
      }
   }

   //---------------------------------------------------------------------------
   // Get list of data members
   //---------------------------------------------------------------------------
   TIter next( cl->GetListOfDataMembers() );
   TDataMember* member;
   while( ( member = (TDataMember*)next() ) )
   {
      TClass *mcl = TClass :: GetClass( member->GetTrueTypeName() );

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
               *(int *)(((char *)obj) + member->GetOffset()) = genInt();
               break;
            case kUInt_t:
               *(unsigned int *)(((char *)obj) + member->GetOffset()) = genUInt();
               break;
            case kDouble_t:
            case kDouble32_t:
               *(double *)(((char *)obj) + member->GetOffset()) = genDouble();
               break;
            case kFloat_t:
            case kFloat16_t:
               *(float *)(((char *)obj) + member->GetOffset()) = genFloat();
               break;
            case kShort_t:
               *(short *)(((char *)obj) + member->GetOffset()) = genShort();
               break;
            case kBool_t: // Don't change these values.
               *(bool *)(((char *)obj) + member->GetOffset()) = genBool();
               break;
            default:
               std::cout << "Case not handled in test generate():" << type << " in " << cl->GetName() << std::endl;
               break;
         }
      }
      //------------------------------------------------------------------------
      // Deal with objects
      //------------------------------------------------------------------------
      else if( mcl )
      {
         char *ptr   = ((char *)obj) + member->GetOffset();

         //---------------------------------------------------------------------
         // A pointer?
         //---------------------------------------------------------------------
         if( member->IsaPointer() )
         {
            vector<TClass*> cls = getConcreteClasses( mcl );
            int ind = genInt( cls.size() + 1 ) - 1;
            if( ind == 0 )
            {
               *((void**)ptr) = 0;
               continue;
            }
            --ind;
            TClass* rcl = cls[ind];
            char* newObj = (char*)rcl->New();
            *((void**)ptr) = newObj + rcl->GetBaseClassOffset( mcl );
            generate( newObj, rcl );
         }
         //---------------------------------------------------------------------
         // Non-pointer
         //---------------------------------------------------------------------
         else
            generate( ptr, mcl );
      }
   }
}

#endif // GENERATOR_H
