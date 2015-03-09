// @(#)root/core:$Id$
// author: Lukasz Janyst <ljanyst@cern.ch>

#include "RConversionRuleParser.h"
#include "TSchemaRuleProcessor.h"
#include "TMetaUtils.h"

#include <iostream>
#include <algorithm>
#include <string>
#include <utility>
#include <map>
#include <sstream>

namespace {
   static void RemoveEscapeSequences(std::string& rawString)
   {
      const std::vector<std::pair<std::string, std::string>> subPairs { {"\\\\","\\"},
                                                                        {"\\\"","\""},
                                                                        {"\\\'","\'"}};
      size_t start_pos = 0;
      for (auto const & subPair : subPairs){
         start_pos = 0;
         auto from = subPair.first;
         auto to = subPair.second;
         while((start_pos = rawString.find(from, start_pos)) != std::string::npos) {
            rawString.replace(start_pos, from.length(), to);
            start_pos += to.length();
         }
      }
   }
}

namespace ROOT
{
   typedef std::list<std::pair<ROOT::TSchemaType,std::string> > SourceTypeList_t;

   //--------------------------------------------------------------------------
   // Allocate global variables
   //--------------------------------------------------------------------------
   SchemaRuleClassMap_t gReadRules;
   SchemaRuleClassMap_t gReadRawRules;

   static Bool_t ValidateRule( const std::map<std::string, std::string>& rule, std::string &error_string );

   static std::string::size_type FindEndSymbol(std::string &command)
   {
      // Find the end of a symbol.

      if (command.length() == 0) return std::string::npos;
      std::string::size_type cursor;
      unsigned int level = 0;
      for (cursor = 0 ; cursor < command.length(); ++cursor)
      {
         switch( command[cursor] ) {
            case ' ':
            case '\t':
            case '\r':
            case '=': if (level==0) {
               std::string::size_type sub_cursor = cursor;
               while( isspace(command[sub_cursor]) ) {
                  ++sub_cursor;
               }
               if ( command[sub_cursor] == '=' ) {
                  return sub_cursor;
               } else {
                  return cursor;
               }
            } else {
               break;
            }
            case '<': ++level; break;
            case '>': if (level==0) { return std::string::npos; }
                      --level; break;
            default: {
               // nothing to do
            }
         };
      }
      return cursor;
   }


   //--------------------------------------------------------------------------
   Bool_t ParseRule( std::string command,
                     std::map<std::string, std::string> &result,
                     std::string &error_string )
   {
      // Parse the schema rule as specified in the LinkDef file

      std::string::size_type l=0;
      command = TSchemaRuleProcessor::Trim( command );

      //-----------------------------------------------------------------------
      // Remove the semicolon from the end if declared
      //-----------------------------------------------------------------------
      if( command[command.size()-1] == ';' )
         command = command.substr( 0, command.size()-1 );

      //-----------------------------------------------------------------------
      // If the first symbol does not end is not followed by equal then it
      // defaults to being the sourceClass.
      //-----------------------------------------------------------------------
      {
         std::string::size_type endsymbol = FindEndSymbol( command );
         if ( endsymbol == command.length() || command[endsymbol] == ' ' || command[endsymbol] == '\t' ) {

//         std::string::size_type space_pos = command.find( ' ' );
//         std::string::size_type equal_pos = command.find( '=' );
//         if ( space_pos < equal_pos) {
            std::string value = TSchemaRuleProcessor::Trim( command.substr( 0, endsymbol ) );
            result["sourceClass"] = value;
            result["targetClass"] = value;
            if (endsymbol < command.length()) {
               command = TSchemaRuleProcessor::Trim( command.substr( endsymbol+1 ) );
            } else {
               command.clear();
            }

            //-----------------------------------------------------------------------
            // If the first symbol is the targetClass then the 2nd symbol can be
            // the source data member name.
            //-----------------------------------------------------------------------
//            space_pos = command.find( ' ' );
//            equal_pos = command.find( '=' );
//            if ( space_pos < equal_pos ) {
            endsymbol = FindEndSymbol( command );
            if ( endsymbol == command.length() || command[endsymbol] == ' ' || command[endsymbol] == '\t' ) {
               std::string membervalue = TSchemaRuleProcessor::Trim( command.substr( 0, endsymbol ) );
               result["source"] = membervalue;
               result["target"] = membervalue;
               command = TSchemaRuleProcessor::Trim( command.substr( endsymbol+1 ) );
            }
         }
      }

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
            error_string = "Parsing error, no key found!";
            return false;
         }

         //--------------------------------------------------------------------
         // The key was found - process the arguments
         //--------------------------------------------------------------------
         std::string key = TSchemaRuleProcessor::Trim( command.substr( 0, pos ) );
         command = TSchemaRuleProcessor::Trim( command.substr( pos+1 ) );

         //--------------------------------------------------------------------
         // Nothing left to be processed
         //--------------------------------------------------------------------
         if( command.size() < 1 ) {
            error_string = "Parsing error, wrond or no value specified for key: " + key;
            return false;
         }

         Bool_t hasquote = command[0] == '"';

         //--------------------------------------------------------------------
         // Processing code tag: "{ code }"
         //--------------------------------------------------------------------
         if( key == "code" ) {
            if( command[1] != '{' ) {
               error_string = "Parsing error while processing key: code\n";
               error_string += "Expected \"{ at the beginning of the value.";
               return false;
            }
            l = command.find( "}\"" );
            if( l == std::string::npos ) {
               error_string = "Parsing error while processing key: \"" + key + "\"\n";
               error_string += "Expected }\" at the end of the value.";
               return false;
            }
            auto rawCode = command.substr( 2, l-2 );
            RemoveEscapeSequences(rawCode);
            result[key] = rawCode;
            ++l;
         }
         //--------------------------------------------------------------------
         // Processing normal tag: "value"
         //--------------------------------------------------------------------
         else {

            if( hasquote) {
               l = command.find( '"', 1 );
               if (l == std::string::npos ) {
                  error_string = "\nParsing error while processing key: \"" + key + "\"\n";
                  error_string += "Expected \" at the end of the value.";
                  return false;
               }
               result[key] = command.substr( 1, l-1 );
            } else {
               l = command.find(' ', 1);
               if (l == std::string::npos) l = command.size();
               result[key] = command.substr( 0, l );
            }
         }

         //--------------------------------------------------------------------
         // Everything went ok
         //--------------------------------------------------------------------
         if( l == command.size() )
            break;
         command = command.substr( l+1 );
      }
      std::map<std::string, std::string>::const_iterator it1;
      it1 = result.find("oldtype");
      if ( it1 != result.end() ) {
         std::map<std::string, std::string>::const_iterator it2;
         it2 = result.find("source");
         if ( it2 != result.end() ) {
            result["source"] = it1->second + " " + it2->second;
         }
      }
      if ( result.find("version") == result.end() && result.find("checksum") == result.end() ) {
         result["version"] = "[1-]";
      }

      //------------------------------------------------------------------------
      // "include" tag. Replace ";" with "," for backwards compatibility with
      // ROOT5
      //------------------------------------------------------------------------
      auto const includeKeyName = "include";
      auto includeTag = result.find(includeKeyName);
      if (includeTag != result.end()){
         auto & includeTagValue = includeTag->second;
         std::replace_if (includeTagValue.begin(),
                          includeTagValue.end(),
                          [](char c){ return c == ';';},
                          ',');
         result[includeKeyName] = includeTagValue;
      }

      return ValidateRule( result, error_string);
   }

   //--------------------------------------------------------------------------
   static Bool_t ValidateRule( const std::map<std::string, std::string>& rule, std::string &error_string )
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
         error_string = "WARNING: You always have to specify the targetClass ";
         error_string += "when specyfying an IO rule";
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
         error_string = warning + " - sourceClass parameter is missing";
         return false;
      }

      //-----------------------------------------------------------------------
      // Check if we have either version or checksum specified
      //-----------------------------------------------------------------------
      it1 = rule.find( "version" );
      it2 = rule.find( "checksum" );
      if( it1 == rule.end() && it2 == rule.end() ) {
         error_string = warning + " - you need to specify either version or ";
         error_string += "checksum";
         return false;
      }

      //-----------------------------------------------------------------------
      // Check if the checksum has been set to right value
      //-----------------------------------------------------------------------
      if( it2 != rule.end() ) {
         if( it2->second.size() < 2 || it2->second[0] != '[' ||
             it2->second[it2->second.size()-1] != ']' ) {
            error_string = warning + " - a comma separated list of ints";
            error_string += " enclosed in square brackets expected";
            error_string += " as a value of checksum parameter";
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
               error_string = warning + " - " + *lsIt + " is not a valid value";
               error_string += " of checksum parameter - an integer expected";
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
            error_string = warning + " - a comma separated list of version specifiers ";
            error_string += "enclosed in square brackets expected";
            error_string += "as a value of version parameter";
            return false;
         }

         TSchemaRuleProcessor::SplitList( it1->second.substr( 1, it1->second.size()-2 ),
                                          lst );
         if( lst.empty() ) {
            error_string = warning + " - the list of versions is empty";
         }

         for( lsIt = lst.begin(); lsIt != lst.end(); ++lsIt )
            if( !TSchemaRuleProcessor::ProcessVersion( *lsIt, ver ) ) {
               error_string = warning + " - " + *lsIt + " is not a valid value";
               error_string += " of version parameter";
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
            error_string = warning + " - required parameter is missing: ";
            error_string += keys[i];
            return false;
         }
      }

      //-----------------------------------------------------------------------
      // Check the source contains proper declarations.
      //-----------------------------------------------------------------------
      it1 = rule.find("code");
      if (it1 != rule.end() && it1->second != "") {
         SourceTypeList_t source;
         TSchemaRuleProcessor::SplitDeclaration( rule.find("source")->second, source );
         SourceTypeList_t::const_iterator it;
         for( it = source.begin(); it != source.end(); ++it ) {
            if ( ( it->first.fType == "" && it->second != "") ) {
               error_string = warning + " - type required when listing a rule's source: ";
               error_string += "source=\""+ rule.find("source")->second +"\"";
               return false;
            }
         }
      }

      //-----------------------------------------------------------------------
      // Check if we have an embed parameter and if so if it has been set to
      // the right value
      //-----------------------------------------------------------------------
      it1 = rule.find( "embed" );
      if( it1 != rule.end() ) {
         std::string emValue = TSchemaRuleProcessor::Trim( it1->second );
         if( emValue != "true" && emValue != "false" ) {
            error_string = warning + " - true or false expected as a value ";
            error_string += "of embed parameter";
            return false;
         }
      }

      //-----------------------------------------------------------------------
      // Check if the include list is not empty
      //-----------------------------------------------------------------------
      it1 = rule.find( "include" );
      if( it1 != rule.end() ) {
         if( it1->second.empty() ) {
            error_string = warning + " - the include list is empty";
            return false;
         }
      }

      return true;
   }

   //---------------------------------------------------------------------------
   Bool_t HasValidDataMembers(SchemaRuleMap_t& rule,
                              MembersTypeMap_t& members )
   {
      // Check if given rule contains references to valid data members
      std::list<std::string>           mem;
      std::list<std::string>::iterator it;
      // MembersMap_t::iterator           rIt;

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
                                   const SourceTypeList_t& source,
                                   MembersTypeMap_t& members,
                                   std::string& className, std::string& mappedName,
                                   std::ostream& output )
   {
      //-----------------------------------------------------------------------
      // Write down the sources
      //-----------------------------------------------------------------------
      if (!source.empty()) {
         Bool_t start = true;
         SourceTypeList_t::const_iterator it;

         //--------------------------------------------------------------------
         // Write IDs and check if we should generate the onfile structure
         // this is done if the type was declared
         //--------------------------------------------------------------------
         Bool_t generateOnFile = false;
         output << "#if 0" << std::endl; // this is to be removed later
         for( it = source.begin(); it != source.end(); ++it ) {
            output << "      ";
            output << "static Int_t id_" << it->second << " = oldObj->GetId(";
            output << "\"" << it->second << "\");" << std::endl;

            if( it->first.fType != "" )
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
               // fprintf(stderr, "Seeing %s %s %s\n", it->first.fType.c_str(), it->second.c_str(), it->first.fDimensions.c_str());
               if( it->first.fType.size() ) {
                  if ( it->first.fDimensions.size() ) {
                     output << "         typedef " << it->first.fType;
                     output << " onfile_" << it->second << "_t" << it->first.fDimensions << ";\n";
                     output << "         ";
                     output << "onfile_" << it->second << "_t &" << it->second << ";\n";

                  } else {
                     output << "         ";
                     output << it->first.fType << " &" << it->second << ";\n";
                  }
               }
            }

            //-----------------------------------------------------------------
            // Generate the constructor
            //-----------------------------------------------------------------
            output << "         " << onfileStructName << "(";
            for( start = true, it = source.begin(); it != source.end(); ++it ) {
               if( it->first.fType.size() == 0)
                  continue;

               if( !start )
                  output << ", ";
               else
                  start = false;

               if (it->first.fDimensions.size() == 0) {
                  output << it->first.fType << " &onfile_" << it->second;
               } else {
                  output << " onfile_" << it->second << "_t" << " &onfile_" << it->second;
               }
            }
            output << " ): ";

            //-----------------------------------------------------------------
            // Generate the constructor's initializer list
            //-----------------------------------------------------------------
            for( start = true, it = source.begin(); it != source.end(); ++it ) {
               if( it->first.fType == "" )
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
               if( it->first.fType == "" )
                  continue;

               if( !start )
                  output << ",\n";

               else
                  start = false;

               output << "         ";
               output << "*(";
               if (it->first.fDimensions.size() == 0) {
                  output << it->first.fType;
               } else {
                  output << mappedName << "_Onfile::onfile_" << it->second << "_t";
               }
               output << "*)(onfile_add+offset_Onfile_";
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
            TSchemaType memData = members[*it];
            output << "      static Long_t offset_" << *it << " = ";
            output << "cls->GetDataMemberOffset(\"" << *it << "\");";
            output << std::endl;
            if (memData.fDimensions.size()) {
               output << "      typedef " << memData.fType << " " << *it << "_t" << memData.fDimensions << ";" << std::endl;
               output << "      " << *it << "_t& " << *it << " = ";
               output << "*(" << *it << "_t *)(target+offset_" << *it;
               output << ");" << std::endl;
            } else {
               output << "      " << memData.fType << "& " << *it << " = ";
               output << "*(" << memData.fType << "*)(target+offset_" << *it;
               output << ");" << std::endl;
            }
         }
      }
   }

   //--------------------------------------------------------------------------
   void WriteReadRuleFunc( SchemaRuleMap_t& rule, int index,
                           std::string& mappedName, MembersTypeMap_t& members,
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
      std::list<std::pair<ROOT::TSchemaType,std::string> > source;
      std::list<std::string> target;
      TSchemaRuleProcessor::SplitDeclaration( rule["source"], source );
      TSchemaRuleProcessor::SplitList( rule["target"], target );

      WriteAutoVariables( target, source, members, className, mappedName, output );
      output << "      " << className << "* newObj = (" << className;
      output << "*)target;" << std::endl;
      output << "      // Supress warning message.\n";
      output << "      " << "if (oldObj) {}\n\n";
      output << "      " << "if (newObj) {}\n\n";

      //-----------------------------------------------------------------------
      // Write the user's code
      //-----------------------------------------------------------------------
      output << "      //--- User's code ---" << std::endl;
      output << "     " << rule["code"] << std::endl;
      output << "   }" << std::endl;
   }


   //--------------------------------------------------------------------------
   void WriteReadRawRuleFunc( SchemaRuleMap_t& rule, int index,
                              std::string& mappedName, MembersTypeMap_t& members,
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
      std::list<std::pair<ROOT::TSchemaType,std::string> > source;
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
      std::string::size_type tr_len= tr.size();

      if( s == 0 ) return;

      while( 1 ) {
         it = proc.find( pat, it );
         if( it == std::string::npos )
            break;

         proc.replace( it, s, tr );
         it += tr_len;
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
         // Deal with non mandatory keys
         //--------------------------------------------------------------------
         if( it->find( "funcname" ) != it->end() ) {
            std::string code = (*it)["code"];
            StrReplace( code, "\n", "\\n" );
            StrReplace( code, "\"", "\\\"");

            output << "      rule->fFunctionPtr = (void *)TFunc2void( ";
            output << (*it)["funcname"] << ");" << std::endl;
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

         if( it->find( "attributes" ) != it->end() ) {
            output << "      rule->fAttributes   = \"" << (*it)["attributes"];
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
      for( it = gReadRules.begin(); it != gReadRules.end(); ++it ) {
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
      for( it = gReadRawRules.begin(); it != gReadRawRules.end(); ++it ) {
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
   void ProcessReadPragma( const char* args )
   {
      // I am being called when a read pragma is encountered

      //-----------------------------------------------------------------------
      // Parse the rule and check it's validity
      //-----------------------------------------------------------------------
      std::map<std::string, std::string> rule;
      std::string error_string;
      if( !ParseRule( args, rule, error_string ) ) {
         std::cout << error_string << '\n';
         std::cout << "The following rule has been omitted:" << std::endl;
         std::cout << "   read " << args << std::endl;
         return;
      }

      //-----------------------------------------------------------------------
      // Append the rule to the list
      //-----------------------------------------------------------------------
      SchemaRuleClassMap_t::iterator it;
      std::string                    targetClass = rule["targetClass"];
      it = gReadRules.find( targetClass );
      if( it == gReadRules.end() ) {
         std::list<SchemaRuleMap_t> lst;
         lst.push_back( rule );
         gReadRules[targetClass] = lst;
      }
      else
         it->second.push_back( rule );
   }

   //--------------------------------------------------------------------------
   void ProcessReadRawPragma( const char* args )
   {
      // I am being called then a readraw pragma is encountered

      //-----------------------------------------------------------------------
      // Parse the rule and check it's validity
      //-----------------------------------------------------------------------
      std::map<std::string, std::string> rule;
      std::string error_string;
      if( !ParseRule( args, rule, error_string ) ) {
         std::cout << error_string << '\n';
         std::cout << "The following rule has been omitted:" << std::endl;
         std::cout << "   readraw " << args << std::endl;
         return;
      }

      //-----------------------------------------------------------------------
      // Append the rule to the list
      //-----------------------------------------------------------------------
      SchemaRuleClassMap_t::iterator it;
      std::string                    targetClass = rule["targetClass"];
      it = gReadRawRules.find( targetClass );
      if( it == gReadRawRules.end() ) {
         std::list<SchemaRuleMap_t> lst;
         lst.push_back( rule );
         gReadRawRules[targetClass] = lst;
      }
      else
         it->second.push_back( rule );
   }


}
