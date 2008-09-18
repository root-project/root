// @(#)root/core:$Id$
// author: Lukasz Janyst <ljanyst@cern.ch>

#include "RConversionRuleParser.h"
#include "TSchemaRuleProcessor.h"

#include <iostream>
#include <string>
#include <utility>
#include <map>
#include <sstream>

namespace ROOT
{
   //--------------------------------------------------------------------------
   // Allocate global variables
   //--------------------------------------------------------------------------
   SchemaRuleClassMap_t G__ReadRules;
   SchemaRuleClassMap_t G__ReadRawRules;

   //--------------------------------------------------------------------------
   static void parse( std::string command,
                      std::map<std::string, std::string> &result )
   {
      // Parse the schema rule as specified in the LinkDef file

      std::string::size_type l;
      command = TSchemaRuleProcessor::Trim( command );

      //-----------------------------------------------------------------------
      // Remove the semicolon from the end if declared
      //-----------------------------------------------------------------------
      if( command[command.size()-1] == ';' )
         command = command.substr( 0, command.size()-1 );

      //-----------------------------------------------------------------------
      // Process the input until there are no characters left
      //-----------------------------------------------------------------------
      while( !command.empty() ) {

         //--------------------------------------------------------------------
         // Find key token
         //--------------------------------------------------------------------
         std::string::size_type pos = command.find( '=' );

         //--------------------------------------------------------------------
         // No equality sign found - no keys left
         //--------------------------------------------------------------------
         if( pos == std::string::npos ) {
            std::cerr << "Parsing error, no key found!" << std::endl;
            break;
         }

         //--------------------------------------------------------------------
         // The key was found - process the arguments
         //--------------------------------------------------------------------
         std::string key = TSchemaRuleProcessor::Trim( command.substr( 0, pos ) );
         command = TSchemaRuleProcessor::Trim( command.substr( pos+1 ) );

         //--------------------------------------------------------------------
         // Nothing left to be processed
         //--------------------------------------------------------------------
         if( command.size() < 2 ) {
            std::cerr << "Parsing error, wrond or no value specified for key: ";
            std::cerr << key << std::endl;
            break;
         }

         if( command[0] != '"' ) {
            std::cerr << "Parsing error while processing key: " << key << std::endl;
            std::cerr << "Expected \" at the beginning of the value" << std::endl;
            break;
         }

         //--------------------------------------------------------------------
         // Processing code tag: "{ code }"
         //--------------------------------------------------------------------
         if( key == "code" ) {
            if( command[1] != '{' ) {
               std::cerr << "Parsing error while processing key: code" << std::endl;
               std::cerr << "Expected \"{ at the beginning of the value" << std::endl;
               break;
            }
            l = command.find( "}\"" );
            if( l == std::string::npos ) {
               std::cerr << "Parsing error while processing key: \"" << key << "\"" << std::endl;
               std::cerr << "Expected }\" at the end of the value" << std::endl;
               break;
            }
            result[key] = command.substr( 2, l-2 );
            ++l;
         }
         //--------------------------------------------------------------------
         // Processing normal tag: "value"
         //--------------------------------------------------------------------
         else {
            l = command.find( '"', 1 );
            if( l == std::string::npos ) {
               std::cerr << "Parsing error while processing key: \"" << key << "\"" << std::endl;
               std::cerr << "Expected \" at the end of the value" << std::endl;
               break;
            }
            result[key] = command.substr( 1, l-1 );
         }

         //--------------------------------------------------------------------
         // Everything went ok
         //--------------------------------------------------------------------
         if( l == command.size() )
            break;
         command = command.substr( l+1 );
      }
   }

   //--------------------------------------------------------------------------
   static bool valid( const std::map<std::string, std::string>& rule )
   {
      // Validate if the user specified rules are correct

      //-----------------------------------------------------------------------
      // Check if we have target class name
      //-----------------------------------------------------------------------
      std::map<std::string, std::string>::const_iterator it1, it2;
      std::list<std::string>                             lst;
      std::list<std::string>::iterator                   lsIt;

      it1 = rule.find( "targetClass" );
      if( it1 == rule.end() ) {
         std::cout << "WARNING: You always have to specify the targetClass ";
         std::cout << "when specyfying an IO rule" << std::endl;
         return false;
      }

      std::string className = TSchemaRuleProcessor::Trim( it1->second );
      std::string warning = "WARNING: IO rule for class " + className;

      //-----------------------------------------------------------------------
      // Check if we have the source tag
      //-----------------------------------------------------------------------
      it1 = rule.find( "sourceClass" );
      if( it1 == rule.end())
      {
         std::cout << warning << " - sourceClass parameter is missing";
         std::cout << std::endl;
         return false;
      }

      //-----------------------------------------------------------------------
      // Check if we have either version or checksum specified
      //-----------------------------------------------------------------------
      it1 = rule.find( "version" );
      it2 = rule.find( "checksum" );
      if( it1 == rule.end() && it2 == rule.end() ) {
         std::cout << warning << " - you need to specify either version or ";
         std::cout << "checksum" << std::endl;
         return false;
      }

      //-----------------------------------------------------------------------
      // Check if the checksum has been set to right value
      //-----------------------------------------------------------------------
      if( it2 != rule.end() ) {
         if( it2->second.size() < 2 || it2->second[0] != '[' ||
             it2->second[it2->second.size()-1] != ']' ) {
            std::cout << warning << " - a comma separated list of ints ";
            std::cout << "enclosed in square brackets expected";
            std::cout << "as a value of checksum parameter" << std::endl;
            return false;
         }

         TSchemaRuleProcessor::SplitList( it2->second.substr( 1, it2->second.size()-2 ),
                                          lst );
         if( lst.empty() ) {
            std::cout << warning << " - the list of checksums is empty";
            std::cout << std::endl;
         }

         for( lsIt = lst.begin(); lsIt != lst.end(); ++lsIt )
            if( !TSchemaRuleProcessor::IsANumber( *lsIt ) ) {
               std::cout << warning << " - " << *lsIt<< " is not a valid value";
               std::cout << " of checksum parameter - an integer expected";
               std:: cout << std:: endl;
               return false;
            }
      }

      //-----------------------------------------------------------------------
      // Check if the version is correct
      //-----------------------------------------------------------------------
      std::pair<Int_t, Int_t> ver;
      if( it1 != rule.end() ) {
         if( it1->second.size() < 2 || it1->second[0] != '[' ||
             it1->second[it1->second.size()-1] != ']' ) {
            std::cout << warning << " - a comma separated list of version specifiers ";
            std::cout << "enclosed in square brackets expected";
            std::cout << "as a value of version parameter" << std::endl;
            return false;
         }

         TSchemaRuleProcessor::SplitList( it1->second.substr( 1, it1->second.size()-2 ),
                                          lst );
         if( lst.empty() ) {
            std::cout << warning << " - the list of versions is empty";
            std::cout << std::endl;
         }

         for( lsIt = lst.begin(); lsIt != lst.end(); ++lsIt )
            if( !TSchemaRuleProcessor::ProcessVersion( *lsIt, ver ) ) {
               std::cout << warning << " - " << *lsIt<< " is not a valid value";
               std::cout << " of version parameter" << std:: endl;
               return false;
            }
      }

      //-----------------------------------------------------------------------
      // Check if we're dealing with renameing declaration - sourceClass,
      // targetClass and either version or checksum required
      //-----------------------------------------------------------------------
      if( rule.size() == 3 || (rule.size() == 4 && it1 != rule.end() && it2 != rule.end()) )
         return true;

      //-----------------------------------------------------------------------
      // Check if we have all the keys we need
      //-----------------------------------------------------------------------
      std::string keys[] = {"target", "source"};
      for( int i = 0; i < 2; ++i ) {
         it1 = rule.find( keys[i] );
         if( it1 == rule.end() ) {
            std::cout << warning << " - required parameter is missing: ";
            std::cout << keys[i] << std::endl;
            return false;
         }
      }

      //-----------------------------------------------------------------------
      // Check if we have an embed aparameter and if so if it has been set to
      // the right value
      //-----------------------------------------------------------------------
      it1 = rule.find( "embed" );
      if( it1 != rule.end() ) {
         std::string emValue = TSchemaRuleProcessor::Trim( it1->second );
         if( emValue != "true" && emValue != "false" ) {
            std::cout << warning << " - true or false expected as a value ";
            std::cout << "of embed parameter" << std::endl;
            return false;
         }
      }

      //-----------------------------------------------------------------------
      // Check if the include list is not empty
      //-----------------------------------------------------------------------
      it1 = rule.find( "include" );
      if( it1 != rule.end() ) {
         if( it1->second.empty() ) {
            std::cout << warning << " - the include list is empty" << std::endl;
            return false;
         }
      }

      return true;
   }

   //--------------------------------------------------------------------------
   void CreateNameTypeMap( G__ClassInfo &cl, MembersMap_t& nameType )
   {
      // Create the data member name-type map for given class

      G__DataMemberInfo member( cl );
      while( member.Next() ) {
         nameType[member.Name()] = member.Type()->Name();
      }

      G__BaseClassInfo base( cl );
      while( base.Next() ) {
         nameType[base.Name()] = base.Name();
      }
   }

   //---------------------------------------------------------------------------
   bool HasValidDataMembers( SchemaRuleMap_t& rule,
                             MembersMap_t& members )
   {
      // Check if given rule contains references to valid data members
      std::list<std::string>           mem;
      std::list<std::string>::iterator it;
      MembersMap_t::iterator           rIt;

      TSchemaRuleProcessor::SplitList( rule["target"], mem );

      //-----------------------------------------------------------------------
      // Loop over the data members
      //-----------------------------------------------------------------------
      for( it = mem.begin(); it != mem.end(); ++it ) {
         if( members.find( *it ) == members.end() ) {
            std::cout << "WARNING: IO rule for class " + rule["targetClass"];
            std::cout << " data member: " << *it << " was specified as a ";
            std::cout << "target in the rule but doesn't seem to appear in ";
            std::cout << "target class" << std::endl;
            return false;
         }
      }
      return true;
   }

   //--------------------------------------------------------------------------
   static void WriteAutoVariables( const std::list<std::string>& target,
                                   const std::list<std::pair<std::string,std::string> >& source,
                                   MembersMap_t& members,
                                   std::string& className, std::string& mappedName,
                                   std::ostream& output )
   {
      //-----------------------------------------------------------------------
      // Write down the sources
      //-----------------------------------------------------------------------
      if (!source.empty()) {
         bool start = true;
         std::list<std::pair<std::string,std::string> >::const_iterator it;

         //--------------------------------------------------------------------
         // Write IDs and check if we should generate the onfile structure
         // this is done if the type was declared
         //--------------------------------------------------------------------
         bool generateOnFile = false;
         output << "#if 0" << std::endl; // this is to be removed later
         for( it = source.begin(); it != source.end(); ++it ) {
            output << "      ";
            output << "static Int_t id_" << it->second << " = oldObj->GetId(";
            output << "\"" << it->second << "\");" << std::endl;

            if( it->first != "" )
               generateOnFile = true;
         }
         output << "#endif" << std::endl; // this is to be removed later

         //--------------------------------------------------------------------
         // Declare the on-file structure - if needed
         //--------------------------------------------------------------------
         if( generateOnFile ) {
            std::string onfileStructName = mappedName + "_Onfile";
            output << "      ";
            output << "struct " << onfileStructName << " {\n";

            //-----------------------------------------------------------------
            // List the data members with non-empty type declarations
            //-----------------------------------------------------------------
            for( it = source.begin(); it != source.end(); ++it ) {
               if( it->first != "" ) {
                  output << "         ";
                  output << it->first << " &" << it->second << ";\n"; 
               }
            }

            //-----------------------------------------------------------------
            // Generate the constructor
            //-----------------------------------------------------------------
            output << "         " << onfileStructName << "(";
            for( start = true, it = source.begin(); it != source.end(); ++it ) {
               if( it->first == "" )
                  continue;

               if( !start )
                  output << ", ";
               else
                  start = false;

               output << it->first << " &onfile_" << it->second;
            }
            output << " ): ";

            //-----------------------------------------------------------------
            // Generate the constructor's initializer list
            //-----------------------------------------------------------------
            for( start = true, it = source.begin(); it != source.end(); ++it ) {
               if( it->first == "" )
                  continue;

               if( !start )
                  output << ", ";
               else
                  start = false;

               output << it->second << "(onfile_" << it->second << ")";
            }
            output << " {}\n";
            output << "      " << "};\n";

            //-----------------------------------------------------------------
            // Initialize the structure - to be changed later
            //-----------------------------------------------------------------
            for( it = source.begin(); it != source.end(); ++it ) {
               output << "      ";
               output << "static Long_t offset_Onfile_" << mappedName;
               output << "_" << it->second << " = oldObj->GetClass()->GetDataMemberOffset(\"";
               output << it->second << "\");\n";
            }
            output << "      " << "char *onfile_add = (char*)oldObj->GetObject();\n";
            output << "      " << mappedName << "_Onfile onfile(\n";

            for( start = true, it = source.begin(); it != source.end(); ++it ) {
               if( it->first == "" )
                  continue;

               if( !start )
                  output << ",\n";

               else
                  start = false;

               output << "         ";
               output << "*(" << it->first << "*)(onfile_add+offset_Onfile_";
               output << mappedName << "_" << it->second << ")";  
            }
            output << " );\n\n";
         }
      }

      //-----------------------------------------------------------------------
      // Write down the targets
      //-----------------------------------------------------------------------
      if( !target.empty() ) {
         output << "      static TClassRef cls(\"";
         output << className << "\");" << std::endl;

         std::list<std::string>::const_iterator it;
         for( it = target.begin(); it != target.end(); ++it ) {
            std::string memData = members[*it];
            output << "      static Long_t offset_" << *it << " = ";
            output << "cls->GetDataMemberOffset(\"" << *it << "\");";
            output << std::endl;
            output << "      " << memData << "& " << *it << " = ";
            output << "*(" << memData << "*)(target+offset_" << *it;
            output << ");" << std::endl;
         }
      }
   }

   //--------------------------------------------------------------------------
   void WriteReadRuleFunc( SchemaRuleMap_t& rule, int index,
                           std::string& mappedName, MembersMap_t& members,
                           std::ostream& output )
   {
      // Write the conversion function for Read rule, the function name
      // is being written to rule["funcname"]

      std::string className = rule["targetClass"]; 

      //-----------------------------------------------------------------------
      // Create the function name
      //-----------------------------------------------------------------------
      std::ostringstream func;
      func << "read_" << mappedName << "_" << index;
      rule["funcname"] = func.str();

      //-----------------------------------------------------------------------
      // Write the header
      //-----------------------------------------------------------------------
      output << "   static void " << func.str();
      output << "( char* target, TVirtualObject *oldObj )" << std::endl;
      output << "   {" << std::endl;
      output << "      //--- Automatically generated variables ---" << std::endl;

      //-----------------------------------------------------------------------
      // Write the automatically generated variables
      //-----------------------------------------------------------------------
      std::list<std::pair<std::string,std::string> > source;
      std::list<std::string> target;
      TSchemaRuleProcessor::SplitDeclaration( rule["source"], source );
      TSchemaRuleProcessor::SplitList( rule["target"], target );

      WriteAutoVariables( target, source, members, className, mappedName, output );
      output << "      " << className << "* newObj = (" << className;
      output << "*)target;" << std::endl << std::endl;

      //-----------------------------------------------------------------------
      // Write the user's code
      //-----------------------------------------------------------------------
      output << "      //--- User's code ---" << std::endl;
      output << "     " << rule["code"] << std::endl;
      output << "   }" << std::endl;
   }


   //--------------------------------------------------------------------------
   void WriteReadRawRuleFunc( SchemaRuleMap_t& rule, int index,
                              std::string& mappedName, MembersMap_t& members,
                              std::ostream& output )
   {
      // Write the conversion function for ReadRaw rule, the function name
      // is being written to rule["funcname"]

      std::string className = rule["targetClass"]; 

      //-----------------------------------------------------------------------
      // Create the function name
      //-----------------------------------------------------------------------
      std::ostringstream func;
      func << "readraw_" << mappedName << "_" << index;
      rule["funcname"] = func.str();

      //-----------------------------------------------------------------------
      // Write the header
      //-----------------------------------------------------------------------
      output << "   static void " << func.str();
      output << "( char* target, TBuffer &b )" << std::endl;
      output << "   {" << std::endl;
      output << "#if 0" << std::endl;
      output << "      //--- Automatically generated variables ---" << std::endl;

      //-----------------------------------------------------------------------
      // Write the automatically generated variables
      //-----------------------------------------------------------------------
      std::list<std::pair<std::string,std::string> > source;
      std::list<std::string> target;
      TSchemaRuleProcessor::SplitList( rule["target"], target );

      WriteAutoVariables( target, source, members, className, mappedName, output );
      output << "      " << className << "* newObj = (" << className;
      output << "*)target;" << std::endl << std::endl;

      //-----------------------------------------------------------------------
      // Write the user's code
      //-----------------------------------------------------------------------
      output << "      //--- User's code ---" << std::endl;
      output << rule["code"] << std::endl;
      output << "#endif" << std::endl;
      output << "   }" << std::endl;
   }

   //--------------------------------------------------------------------------
   static void StrReplace( std::string& proc, const std::string& pat,
                           const std::string& tr )
   {
      // Replace all accurances of given string with other string
      std::string::size_type it = 0;
      std::string::size_type s  = pat.size();

      if( s == 0 ) return;

      while( 1 ) {
         it = proc.find( pat, it );
         if( it == std::string::npos )
            break;

         proc.replace( it, s, tr );
         ++it;
      }
   }

   //--------------------------------------------------------------------------
   void WriteSchemaList( std::list<SchemaRuleMap_t>& rules,
                         const std::string& listName, std::ostream& output )
   {
      // Write schema rules
      std::list<SchemaRuleMap_t>::iterator it;
      int                                  i = 0;

      //-----------------------------------------------------------------------
      // Loop over the rules
      //-----------------------------------------------------------------------
      for( it = rules.begin(); it != rules.end(); ++it ) {
         output << "      rule = &" << listName << "[" << i++;
         output << "];" << std::endl;

         //--------------------------------------------------------------------
         // Write down the mandatory fields
         //--------------------------------------------------------------------
         output << "      rule->fSourceClass = \"" << (*it)["sourceClass"];
         output << "\";" << std::endl;

         if( it->find( "target" ) != it->end() ) {
            output << "      rule->fTarget      = \"" << (*it)["target"];
            output << "\";" << std::endl;
         }

         if( it->find( "source" ) != it->end() ) {
            output << "      rule->fSource      = \"" << (*it)["source"];
            output << "\";" << std::endl;
         }

         //--------------------------------------------------------------------
         // Deal with nonmandatory keys
         //--------------------------------------------------------------------
         if( it->find( "funcname" ) != it->end() ) {
            std::string code = (*it)["code"];
            StrReplace( code, "\n", "\\n" );

            output << "      rule->fFunctionPtr = (void *)";
            output << (*it)["funcname"] << ";" << std::endl;
            output << "      rule->fCode        = \"" << code;
            output << "\";" << std::endl;
         }
 
         if( it->find( "version" ) != it->end() ) {
            output << "      rule->fVersion     = \"" << (*it)["version"];
            output << "\";" << std::endl;
         }

         if( it->find( "checksum" ) != it->end() ) {
            output << "      rule->fChecksum    = \"" << (*it)["checksum"];
            output << "\";" << std::endl;
         }

         if( it->find( "embed" ) != it->end() ) {
            output << "      rule->fEmbed       = " <<  (*it)["embed"];
            output << ";" << std::endl;
         }

         if( it->find( "include" ) != it->end() ) {
            output << "      rule->fInclude     = \"" << (*it)["include"];
            output << "\";" << std::endl;
         }
      }
   }

   //--------------------------------------------------------------------------
   void GetRuleIncludes( std::list<std::string> &result )
   {
      // Get the list of includes specified in the shema rules
      std::list<std::string>               tmp;
      std::list<SchemaRuleMap_t>::iterator rule;
      SchemaRuleMap_t::iterator            attr;
      SchemaRuleClassMap_t::iterator       it;

      //-----------------------------------------------------------------------
      // Processing read rules
      //-----------------------------------------------------------------------
      for( it = G__ReadRules.begin(); it != G__ReadRules.end(); ++it ) {
         for( rule = it->second.begin(); rule != it->second.end(); ++rule ) {
            attr = rule->find( "include" );
            if( attr == rule->end() ) continue;
            TSchemaRuleProcessor::SplitList( attr->second, tmp );
            result.splice( result.begin(), tmp, tmp.begin(), tmp.end() );
         }
      }

      //-----------------------------------------------------------------------
      // Processing read raw rules
      //-----------------------------------------------------------------------
      for( it = G__ReadRawRules.begin(); it != G__ReadRawRules.end(); ++it ) {
         for( rule = it->second.begin(); rule != it->second.end(); ++rule ) {
            attr = rule->find( "include" );
            if( attr == rule->end() ) continue;
            TSchemaRuleProcessor::SplitList( attr->second, tmp );
            result.splice( result.begin(), tmp, tmp.begin(), tmp.end() );
         }
      }

      //-----------------------------------------------------------------------
      // Removing duplicates
      //-----------------------------------------------------------------------
      result.sort();
      result.unique();
   }

   //--------------------------------------------------------------------------
   void ProcessReadPragma( char* args )
   {
      // I am being called when a read pragma is encountered

      //-----------------------------------------------------------------------
      // Parse the rule and check it's validity
      //-----------------------------------------------------------------------
      std::map<std::string, std::string> rule;
      parse( args, rule );
      if( !valid( rule ) ) {
         std::cout << "The rule has been omited!" << std::endl;
         return;
      }

      //-----------------------------------------------------------------------
      // Append the rule to the list
      //-----------------------------------------------------------------------
      SchemaRuleClassMap_t::iterator it;
      std::string                    targetClass = rule["targetClass"];
      it = G__ReadRules.find( targetClass );
      if( it == G__ReadRules.end() ) {
         std::list<SchemaRuleMap_t> lst;
         lst.push_back( rule );
         G__ReadRules[targetClass] = lst;
      }
      else
         it->second.push_back( rule );
   }

   //--------------------------------------------------------------------------
   void ProcessReadRawPragma( char* args )
   {
      // I am being called then a readraw pragma is encountered

      //-----------------------------------------------------------------------
      // Parse the rule and check it's validity
      //-----------------------------------------------------------------------
      std::map<std::string, std::string> rule;
      parse( args, rule );
      if( !valid( rule ) ) {
         std::cout << "The rule has been omited!" << std::endl;
         return;
      }

      //-----------------------------------------------------------------------
      // Append the rule to the list
      //-----------------------------------------------------------------------
      SchemaRuleClassMap_t::iterator it;
      std::string                    targetClass = rule["targetClass"];
      it = G__ReadRawRules.find( targetClass );
      if( it == G__ReadRawRules.end() ) {
         std::list<SchemaRuleMap_t> lst;
         lst.push_back( rule );
         G__ReadRawRules[targetClass] = lst;
      }
      else
         it->second.push_back( rule );
   }
}
