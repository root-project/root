// @(#)root/test:$Name:  $:$Id: tstring.cxx,v 1.4 2002/01/24 11:39:31 rdm Exp $
// Author: Fons Rademakers   19/08/96

#include <stdlib.h>

#include "Riostream.h"
#include "TString.h"
#include "TRegexp.h"


void Ok(int i, int b)
{
   cout << "Test " << i;
   if (b)
      cout << "   ok" << endl;
   else
      cout << "   NOT ok" << endl;
}



int main()
{
   // create base string
   TString s = "aap noot mies";
   cout << s << endl;

   // use different constructors and excercise +, += and == operators
   TString s1 = TString("aa") + "p ";
   s1 += TString("noot ") + "mi" + "es";
   cout << s1 << endl;
   Ok(1, s1==s);

   // use single char constructor and excercise again += and == operators
   TString s2 = 'a';
   s2 += "ap ";
   s2 += "noot ";
   s2 += "mies";
   cout << s2 << endl;
   Ok(2, s2==s);

   // get and set a subrange (via TSubString)
   TString s3 = s(4,9);
   s3(0,4) = "mama papa";
   cout << s3 << endl;
   Ok(3, s3=="mama papa mies");

   // create a regular expression, make search in string and replace
   // matched substring
   TRegexp re(" n.*t ");
   TString s4 = s;
   s4(re) = " pipo ";
   cout << s4 << endl;
   Ok(4, s4=="aap pipo mies");

   // use "const char*" operator and the Data() member function to access
   // the string as a const char*
   const char *a = (const char*)s;
   cout << a << endl;
   Ok(5, !strcmp(a, s.Data()));

   // return 13th character and replace it by 't', getting 14th character
   // should result in an error message since operator[] uses range checking
   TString s6 = s;
   s6[12] = 't';
   cout << s6 << endl;
   cout << "*** Error message ok, accessing intentionaly out-of-bounds" << endl;
   char b = s6[13];
   cout << b << endl;
   Ok(6, s6=="aap noot miet");

   // return 13th character and replace it by 'p', getting 14th character
   // should NOT result in an error message since operator() does not use
   // range checking
   TString s7 = s;
   s7(12) = 'p';
   cout << s7 << endl;
   char c = s7(13);
   cout << c << endl;
   Ok(7, s7=="aap noot miep");

   // use Append, Remove, Prepend, Replace and Insert
   TString s8 = s;
   s8.Append(" tante ");
   s8.Append(s7(9,4));
   cout << s8 << endl;

   s8.Remove(0,14);
   cout << s8 << endl;
   s8.Prepend("oom jan");
   cout << s8 << endl;
   s8.Insert(7," en ");
   cout << s8 << endl;
   s8.Replace(4,3,"jaap");
   cout << s8 << endl;
   Ok(8, s8=="oom jaap en tante miep");

   // use CompareTo to compare char * and TString
   TString s9 = s;
   Ok(9,  !s9.CompareTo(s));
   Ok(10, !s9.CompareTo("AAP NOOT MIES", TString::kIgnoreCase));

   // use Contains to find if "string" is contained
   Ok(11, s9.Contains("NooT", TString::kIgnoreCase));
   Ok(12, !s9.Contains("pipo"));

   // return the index to the first and last character 'c'
   Ok(13, s.First('o')==5);
   Ok(14, s.First('z')==kNPOS);
   Ok(15, s.Last('a')==1);
   Ok(16, s.Last('z')==kNPOS);

   // returns the index of the start of match
   Ok(17, s.Index("mies")==9);
   Ok(18, s.Index("aa",1)==kNPOS);

   // returns length of string
   Ok(19, s.Length()==13);

   // test IsNull
   TString s10;

   Ok(20, s10.IsNull());
   Ok(21, !s.IsNull());

   // test IsAscii
   s10 = "¹";

   Ok(22, !s10.IsAscii());
   Ok(23, s.IsAscii());

   // test Resize and Strip
   s9.Prepend("   ");
   cout << s9 << endl;

   s9.Resize(50);
   cout << s9 << "<<ends here" << endl;

   cout << s9.Strip(TString::kBoth) << "<<ends here" << endl;

   Printf("Using Print: %s (%d)\n", (const char*) s9, s9.Length());

   return 0;
}
