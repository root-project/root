// @(#)root/reflex:$Name:  $:$Id: FunctionMember.cxx,v 1.10 2006/08/03 16:49:21 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef REFLEX_BUILD
#define REFLEX_BUILD
#endif

#include "FunctionMember.h"

#include "Reflex/Scope.h"
#include "Reflex/Object.h"
#include "Reflex/internal/OwnedMember.h"
#include "Reflex/DictionaryGenerator.h"

#include "Function.h"
#include "Reflex/Tools.h"


//-------------------------------------------------------------------------------
ROOT::Reflex::FunctionMember::FunctionMember( const char *  nam,
                                              const Type &  typ,
                                              StubFunction  stubFP,
                                              void*         stubCtx,
                                              const char *  parameters,
                                              unsigned int  modifiers,
                                              TYPE          memType )
//-------------------------------------------------------------------------------
   : MemberBase( nam, typ, memType, modifiers ),
     fStubFP( stubFP ), 
     fStubCtx( stubCtx ),
     fParameterNames( std::vector<std::string>()),
     fParameterDefaults( std::vector<std::string>()),
     fReqParameters( 0 )
{
   // Obtain the names and default values of the function parameters
   // The "real" number of parameters is obtained from the function type
   size_t numDefaultParams = 0;
   size_t type_npar = typ.FunctionParameterSize();
   std::vector<std::string> params;
   if ( parameters ) Tools::StringSplit(params, parameters, ";");
   size_t npar = std::min(type_npar,params.size());
   for ( size_t i = 0; i < npar ; ++i ) {
      size_t pos = params[i].find( "=" );
      fParameterNames.push_back(params[i].substr(0,pos));
      if ( pos != std::string::npos ) {
         fParameterDefaults.push_back(params[i].substr(pos+1));
         ++numDefaultParams;
      }
      else {
         fParameterDefaults.push_back("");
      }
   }
   // padding with blanks
   for ( size_t i = npar; i < type_npar; ++i ) {
      fParameterNames.push_back("");
      fParameterDefaults.push_back("");
   }
   fReqParameters = type_npar - numDefaultParams;
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::FunctionMember::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
// Construct the qualified (if requested) name of the function member.
   std::string s = "";

   if ( 0 != ( mod & ( QUALIFIED | Q ))) {
      if ( IsPublic())          { s += "public ";    }
      if ( IsProtected())       { s += "protected "; }
      if ( IsPrivate())         { s += "private ";   }  
      if ( IsExtern())          { s += "extern ";    }
      if ( IsStatic())          { s += "static ";    }
      if ( IsInline())          { s += "inline ";    }
      if ( IsVirtual())         { s += "virtual ";   }
      if ( IsExplicit())        { s += "explicit ";  }
   }

   s += MemberBase::Name( mod ); 

   return s;
}


/*/-------------------------------------------------------------------------------
  ROOT::Reflex::Object
  ROOT::Reflex::FunctionMember::Invoke( const Object & obj,
  const std::vector < Object > & paramList ) const {
//-----------------------------------------------------------------------------
  if ( paramList.size() < FunctionParameterSize(true)) {
  throw RuntimeError("Not enough parameters given to function ");
  return Object();
  }
  void * mem = CalculateBaseObject( obj );
  std::vector < void * > paramValues;
  // needs more checking FIXME
  for (std::vector<Object>::const_iterator it = paramList.begin();
  it != paramList.end(); ++it ) paramValues.push_back(it->Address());
  return Object(TypeOf().ReturnType(), fStubFP( mem, paramValues, fStubCtx ));
  }
*/


//-------------------------------------------------------------------------------
ROOT::Reflex::Object
ROOT::Reflex::FunctionMember::Invoke( const Object & obj,
                                      const std::vector < void * > & paramList ) const {
//-----------------------------------------------------------------------------
// Invoke this function member with object obj. 
   if ( paramList.size() < FunctionParameterSize(true)) {
      throw RuntimeError("Not enough parameters given to function ");
      return Object();
   }
   void * mem = CalculateBaseObject( obj );
   // parameters need more checking FIXME
   return Object(TypeOf().ReturnType(), fStubFP( mem, paramList, fStubCtx ));
}


/*/-------------------------------------------------------------------------------
  ROOT::Reflex::Object
  ROOT::Reflex::FunctionMember::Invoke( const std::vector < Object > & paramList ) const {
//-------------------------------------------------------------------------------
  std::vector < void * > paramValues;
  // needs more checking FIXME
  for (std::vector<Object>::const_iterator it = paramList.begin();
  it != paramList.end(); ++it ) paramValues.push_back(it->Address());
  return Object(TypeOf().ReturnType(), fStubFP( 0, paramValues, fStubCtx ));
  }
*/


//-------------------------------------------------------------------------------
ROOT::Reflex::Object
ROOT::Reflex::FunctionMember::Invoke( const std::vector < void * > & paramList ) const {
//-------------------------------------------------------------------------------
// Call static function 
   // parameters need more checking FIXME
   return Object(TypeOf().ReturnType(), fStubFP( 0, paramList, fStubCtx ));
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::FunctionMember::FunctionParameterSize( bool required ) const {
//-------------------------------------------------------------------------------
// Return number of function parameters. If required = true return number without default params.
   if ( required ) return fReqParameters;
   else            return TypeOf().FunctionParameterSize();
}



//-------------------------------------------------------------------------------
void ROOT::Reflex::FunctionMember::GenerateDict( DictionaryGenerator & generator ) const {
//-------------------------------------------------------------------------------
// Generate Dictionary information about itself.   

   std::string mName = Name();

   if ( mName != "__getNewDelFunctions"  && mName != "__getBasesTable" ) {

      // The return type
      const Type & retT = TypeOf().ReturnType();

      // Get function return type, register that to "used types"  
      std::string returntype = generator.GetTypeNumber( TypeOf().ReturnType() );
    
      // Prevents __getNewDelFunctions getting into AddFunctionMember twice
      //if(generator.IsNewType( TypeOf() ) && Name()!="__getNewDelFunctions" ) {
    
      if( IsPrivate() ) {
         generator.AddIntoShadow ( Name(SCOPED) + "();\n");
      }
              
      // Get a number for the function type
      //std::string number = generator.GetTypeNumber( TypeOf() );
          
      std::stringstream temp;
      temp<< generator.fMethodCounter;
      std::string number = temp.str();
   
      ++generator.fMethodCounter;
      
      // Get current Namespace location
      std::string namespc =  DeclaringScope().Name(SCOPED);
        
      std::stringstream tempcounter;
      tempcounter<<generator.fMethodCounter;
            
      // Free function, shall be added into Instances-field only
      if(DeclaringScope().IsNamespace() ) {
         generator.AddIntoInstances("      Type t" + tempcounter.str() + " = FunctionTypeBuilder(type_" + returntype  );
        
      } else { // "normal" function, inside a class
         
         generator.AddIntoFree(".AddFunctionMember(FunctionTypeBuilder(type_" + returntype  );
      }
            
      
      // Get the parameters for function
      for (Type_Iterator params = TypeOf().FunctionParameter_Begin();
           params != TypeOf().FunctionParameter_End(); ++params) {
         
         
         if(DeclaringScope().IsNamespace() ) {
            generator.AddIntoInstances(", type_" + generator.GetTypeNumber( (*params) )  );
               
         } else {
            generator.AddIntoFree(", type_" + generator.GetTypeNumber( (*params) )  );
         }
         
      }


      
      if(DeclaringScope().IsNamespace() ) {
         generator.AddIntoInstances(");  FunctionBuilder(t" + tempcounter.str() + ", \"" 
                                    + Name() + "\", function_" + number ); //function name
      }
   
                   
      else {  // normal function
           
         generator.AddIntoFree("), \"" + Name() + "\"" ); //function name
      }

      if      ( IsConstructor() ) generator.AddIntoFree(", constructor_");
      else if ( IsDestructor()  ) generator.AddIntoFree(", destructor_");

      
           
      if ( IsConstructor() ) {
         generator.AddIntoClasses("static void* constructor_" );
              
      } else if ( IsDestructor() ) {
         generator.AddIntoClasses("static void* destructor_" );
              
      } else {
              
         if(! (DeclaringScope().IsNamespace()) ) {
            generator.AddIntoFree(", method_");
                   
            generator.AddIntoClasses("\nstatic void* method_");

         }
              
         else {
            // free function
            generator.AddIntoClasses("\nstatic void* function_");
         }
      }
           
      if(!(DeclaringScope().IsNamespace()) ) generator.AddIntoFree(number); //method_n
      
            
      generator.AddIntoClasses(number);
    
      
      if( IsConstructor() ) {  // these have parameters

         generator.AddIntoClasses("(void* mem, const std::vector<void*>&");  
         
         if( FunctionParameterSize()) generator.AddIntoClasses(" arg");
         
         generator.AddIntoClasses(", void*)\n{");
         generator.AddIntoClasses("\n  return ::new(mem) " + namespc );
         generator.AddIntoClasses("(");
       
      }//is constructor/destructor with parameters

      else if ( IsDestructor()) {
         generator.AddIntoClasses("(void * o, const std::vector<void*>&, void *) {\n");
         generator.AddIntoClasses("  ((" + namespc + "*)o)->~" + DeclaringScope().Name() + "(");
      }
        
      else {
         // method function with parameters
       
         if( DeclaringScope().IsNamespace() ) {
            generator.AddIntoClasses(" (void*, const std::vector<void*>&");// arg, void*)\n{");
                
         } else {
            generator.AddIntoClasses(" (void* o, const std::vector<void*>&");// arg, void*)\n{");
         }

         if(FunctionParameterSize()>0) generator.AddIntoClasses(" arg");

         generator.AddIntoClasses(", void*)\n{");

         if ( retT.Name() != "void" ) {

            if ( retT.IsFundamental() ) {
               generator.AddIntoClasses("static " + retT.Name(SCOPED) + " ret;\n");
               generator.AddIntoClasses("ret = ");
            }
            else if ( retT.IsReference() || retT.IsPointer()) {
               generator.AddIntoClasses("return (void*)");
               if ( retT.IsReference()) generator.AddIntoClasses("&");
            }
            else { // compound type
               generator.AddIntoClasses("return new " + retT.Name(SCOPED) + "(");
            }
         }
         
         if( DeclaringScope().IsNamespace() ) {
            generator.AddIntoClasses(Name() + "( "); 
                
         } else {
            generator.AddIntoClasses("((" + namespc + "*)o)->"+ Name() + "( ");   
         }

      }

      
      // Add to Stub Functions with some parameters
      if( FunctionParameterSize()>0 ) {
       
         unsigned args = 0;
       
         // Get all parameters
         for (Type_Iterator methpara = TypeOf().FunctionParameter_Begin();
              methpara != TypeOf().FunctionParameter_End(); ++methpara) {

            // get params for the function, can include pointers or references
            std::string param = generator.GetParams( *methpara );
                  
            std::stringstream temp2;
            temp2<<args;
                           
            // We de-deference parameter only, if it's not a pointer
            if (! methpara->IsPointer() ) generator.AddIntoClasses("*");
          
            generator.AddIntoClasses("(");
                    
            //if( methpara->IsConst()) generator.AddIntoClasses("const ");
            
            std::string paraT = methpara->Name(SCOPED|QUALIFIED);
            if ( methpara->IsReference()) paraT = paraT.substr(0,paraT.length()-1);

            generator.AddIntoClasses( paraT );

            if ( ! methpara->IsPointer()) generator.AddIntoClasses("*");

            generator.AddIntoClasses(") arg[" + temp2.str() + "]" );
                        
            //still parameters left
            if( (args+1) < FunctionParameterSize() )  generator.AddIntoClasses(", ");
                  
            ++args;
            //fundam
                  
         }
             
      } // funct. params!=0
                 
                  
      if( IsConstructor()) {
         generator.AddIntoClasses(");\n} \n");
    
      } else {

         generator.AddIntoClasses(")");

         if ( retT.Name() == "void" ) {
            generator.AddIntoClasses(";\n  return 0;\n");
         }
         else if ( retT.IsFundamental()) {
            generator.AddIntoClasses(";\n  return & ret;\n");
         }
         else if ( retT.IsPointer() || retT.IsReference()) {
            generator.AddIntoClasses(";\n");
         }
         else { // compound type
            generator.AddIntoClasses(");\n");
         }

         generator.AddIntoClasses("\n} \n"); //);
      }    
      
   
      
        
      if(DeclaringScope().IsNamespace() ) {
         generator.AddIntoInstances(", 0");
      }
      else {
         generator.AddIntoFree(", 0");
      }
      
        
      if( DeclaringScope().IsNamespace() ) generator.AddIntoInstances(", \"");
      else                                 generator.AddIntoFree(", \"");     
           
         
      // Get the names of the function param.types (like MyInt)
      if( TypeOf().FunctionParameterSize() ) {
         
         unsigned dot = 0;
         Type_Iterator params;
         StdString_Iterator parnam;

         

         for (params = TypeOf().FunctionParameter_Begin(), parnam = FunctionParameterName_Begin();
              params != TypeOf().FunctionParameter_End(), parnam != FunctionParameterName_End(); 
              ++params, ++parnam) {
            
            // THESE SHOULD ALSO INCLUDE DEFAULT VALUES,
            // LIKE int i=5 FunctionParameterDefault_Begin(), _End
            //
            
            if(DeclaringScope().IsNamespace() ) {
               generator.AddIntoInstances( (*parnam) );
               if((dot+1) < FunctionParameterSize()) {
                  generator.AddIntoInstances(";");
               }
            }
            else {
               generator.AddIntoFree( (*parnam) );
               if((dot+1) < FunctionParameterSize()) {
                  generator.AddIntoFree(";");
               }
            }
            ++dot;
         }

      }
    
      if( DeclaringScope().IsNamespace() ) generator.AddIntoInstances("\"");
      else                                 generator.AddIntoFree("\"");
   
    
      if(DeclaringScope().IsNamespace() )  { // free func
         generator.AddIntoInstances(", ");
           
         if( IsPublic() )         generator.AddIntoInstances("PUBLIC");
         else if( IsPrivate() )   generator.AddIntoInstances("PRIVATE");
         else if( IsProtected() ) generator.AddIntoInstances("PROTECTED");
           
         if( IsArtificial())  generator.AddIntoInstances(" | ARTIFICIAL");
           
         generator.AddIntoInstances(");\n");
         
         ++generator.fMethodCounter;
           
      } else {
         generator.AddIntoFree(", ");
           
         if( IsPublic() )         generator.AddIntoFree("PUBLIC");
         else if( IsPrivate() )   generator.AddIntoFree("PRIVATE");
         else if( IsProtected() ) generator.AddIntoFree("PROTECTED");
           
         if( IsVirtual() )    generator.AddIntoFree(" | VIRTUAL");
         if( IsArtificial())  generator.AddIntoFree(" | ARTIFICIAL");
         if( IsConstructor()) generator.AddIntoFree(" | CONSTRUCTOR");
         if( IsDestructor())  generator.AddIntoFree(" | DESTRUCTOR");
           
         generator.AddIntoFree(")\n");
      }
   }
}


