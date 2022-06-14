/// \file
/// \ingroup tutorial_regexp
/// A regular expression, often called a pattern, is an expression that describes a set of
/// strings. They are usually used to give a concise description of a set, without having to
/// list all elements.
/// The Unix utilities like sed and grep make extensive use of regular expressions. Scripting
/// languages like Perl have regular expression engines built directly into their syntax .
///
/// Extensive documentation about Regular expressions in Perl can be
/// found at: http://perldoc.perl.org/perlre.html
///
/// ROOT has this capability through the use of the P(erl) C(ompatible) R(egular) E(xpression)
///  - library, PCRE, see http://www.pcre.org
///
/// Its functionality can be accessed through the TPRegexp and TString class .
/// Note that in patterns taken from Perl all backslash character have to be replaced in the
/// C/C++ strings by two backslashes .
///
/// This macro shows several ways how to use the Match/Substitute capabilities of the
/// the TPRegexp class . It can be run as follows :
/// ~~~
///     .x regexp.C
/// ~~~
///
/// \macro_output
/// \macro_code
///
/// \author Eddy Offermann

#include "Riostream.h"
#include "TString.h"
#include "TPRegexp.h"
#include "TClonesArray.h"
#include "TObjString.h"



void regexp()
{
   // Substitute example :
   // Find a word that starts with "peper" and ends with "koek" .

   TString s1("lekkere pepernotenkoek");
   TPRegexp r1("\\bpeper(\\w+)koek\\b");

   // Note that the TString class gives access to some of the simpler TPRegexp functionality .
   // The following command returns the fully matched string .
   cout << s1(r1) << endl;

   // In the "Substitute" command, keep the middle part (indicated in the regexp by "(\\w+)"
   // and the substitute string by "$1") and sandwich it between "wal" and "boom" .
   r1.Substitute(s1,"wal$1boom");
   cout << s1 << endl;

   // Substitute example :
   // Swap first two words in a string

   TString s2("one two three");
   TPRegexp("^([^ ]+) +([^ ]+)").Substitute(s2,"$2 $1");
   cout << s2 << endl;

   // Substitute example :
   // $1, $2, and so on, in the substitute string are equivalent to whatever the corresponding set
   // of parentheses match in the regexp string, counting opening parentheses from left to right .
   // In the following example, we are trying to catch a date MMDDYYYY in a string and rearrange
   // it to DDMMYYY . "(\\d{1,2}) matches only 1 or 2 digits etc .

   TString s3("on 09/24/1959 the world stood still");
   TPRegexp("\\b(\\d{1,2})/(\\d{1,2})/(\\d{4})\\b").Substitute(s3,"$2-$1-$3");
   cout << s3 << endl;

   // Match Example :
   // The following example shows how to extract a protocol and port number from an URL string .
   // Note again the parentheses in the regexp string : "(\\w+)" requires a non-empty
   // alphanumeric string while "(\\d+)" wants a pure digital string .
   // The matched substrings together with the full matched string are returned in a
   // TObjArray . The first entry is the full string while next entries are the substrings
   // in the order as listed in the regexp string .
   //
   // Note that there is also a Match(..) command that returns the positions of the
   // substrings in the input string .

   TString s4("http://fink.sourceforge.net:8080/index/readme.html");
   TObjArray *subStrL = TPRegexp("^(\\w+)://[^/]+:(\\d+)/$").MatchS(s4);
   const Int_t nrSubStr = subStrL->GetLast()+1;
   if (nrSubStr > 2) {
     const TString proto = ((TObjString *)subStrL->At(1))->GetString();
     const TString port  = ((TObjString *)subStrL->At(2))->GetString();
     cout << "protocol: " << proto << "  port: " << port << endl;
   }

   // Match Example :
   // This example returns kTRUE if the email address is valid . For that it has to fulfil the following
   // criteria:
   // 1) It should be of the form string1@string2 . The "^" and "$" ensure that we compare the complete
   //    email string
   // 2) ([\\w-\\.]+)  :
   //    string1 is only allowed to be composed out of the alphanumeric characters, "-" and "." .
   //    The "+" ensures that string1 can not be empty .
   // 3) string2 is matched against three different parts :
   //    a. ((\\[[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\.)|(([\\w-]+\\.)+))  :
   //       This regular expression ensures that EITHER the string starts with "[" followed by three groups
   //       of numbers, separated by "." , where each group has 1 to 3 numbers, OR alphanumeric strings,
   //       possibly containing "-" characters, separated by "." .
   //    b. ([a-zA-Z]{2,4}|[0-9]{1,3})  :
   //       This part contains EITHER 2 to 4 alpha characters OR 1 to 3 numbers
   //    c. (\\]?)  :
   //       At most one "]" character .

   TString s5("fons.rademakers@cern.ch");
   TPRegexp r5("^([\\w-\\.]+)@((\\[[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\.)|(([\\w-]+\\.)+))([a-zA-Z]{2,4}|[0-9]{1,3})(\\]?)$");
   cout << "Check if the email address \"" << s5 << "\" is valid: " << (r5.MatchB(s5) ? "TRUE" : "FALSE") << endl;

   // Substitute Example with pattern modifier :
   // Like in Perl, Substitute/Match commands accept modifier arguments . For instance a "g" modifier causes to
   // match the regexp globally . In the example below, all words starting and ending with the character "n"
   // are replaced by the word neutrino .

   TString s6("neutron proton electron neutron");
   TPRegexp("(n\\w+n)").Substitute(s6,"neutrino","g");
   cout << s6 << endl;
}
