//-------------------------------------------------------------------------------------------
//
// class TPMERegexp - API similar to PME - PCRE Made Easy
// Tries to be as close as possible to PERL syntax and functionality.
//
// Extension of TPRegexp class, see also macro 'regexp.C'.
//
//-------------------------------------------------------------------------------------------

void regexp_pme()
{
   static const char *underline =
      "----------------------------------------------------------------\n";


   // Match tests

   {
      printf("Global matching\n%s", underline);
      TPMERegexp re("ba[rz]", "g");
      TString m("foobarbaz");
      while (re.Match(m))
         re.Print("all");
      printf("\n");

      printf("Global matching with back-refs\n%s", underline);
      TPMERegexp re("(ba[rz])", "g");
      TString m("foobarbaz");
      while (re.Match(m))
         re.Print("all");
      printf("\n");

      printf("Matching with nested back-refs\n%s", underline);
      TPMERegexp re("([\\w\\.-]+)@((\\d+)\\.(\\d+)\\.(\\d+)\\.(\\d+))");
      TString m("matevz.tadel@137.138.170.210");
      re.Match(m);
      re.Print("all");
      printf("\n");
   }


   // Split tests

   {
      printf("Split\n%s", underline);
      TPMERegexp re(":");
      TString m("root:x:0:0:root:/root:/bin/bash");
      re.Split(m);
      re.Print("all");
      printf("\n");

      printf("Split with maxfields=5\n%s", underline);
      re.Split(m, 5);
      re.Print("all");
      printf("\n");

      printf("Split with empty elements in the middle and at the end\n"
             "maxfields=0, so trailing empty elements are dropped\n%s", underline);
      m = "root::0:0:root:/root::";
      re.Split(m);
      re.Print("all");
      printf("\n");

      printf("Split with empty elements at the beginning and end\n"
             "maxfields=-1, so trailing empty elements are kept\n%s", underline);
      m = ":x:0:0:root::";
      re.Split(m, -1);
      re.Print("all");
      printf("\n");

      printf("Split with no pattern in string\n%s", underline);
      m = "A dummy line of text.";
      re.Split(m);
      re.Print("all");
      printf("\n");
   }

   {
      printf("Split with regexp potentially matching a null string \n%s", underline);
      TPMERegexp re(" *");
      TString m("hi there");
      re.Split(m);
      re.Print("all");
      printf("\n");
   }

   {
      printf("Split on patteren with back-refs\n%s", underline);
      TPMERegexp re("([,-])");
      TString m("1-10,20");
      re.Split(m);
      re.Print("all");
      printf("\n");
   }


   // Substitute tests

   {
      printf("Substitute\n%s", underline);
      TPMERegexp re("(\\d+)\\.(\\d+)\\.(\\d+)\\.(\\d+)");
      TString m("137.138.170.210");
      TString r("$4.$3.$2.$1");
      TString s(m); re.Substitute(s, r);
      re.Print();
      printf("Substitute '%s','%s' => '%s'\n", m.Data(), r.Data(), s.Data());
      printf("\n");
   }

   {
      printf("Global substitute\n%s", underline);
      TPMERegexp re("(\\w+)\\.(\\w+)@[\\w\\.-]+", "g");
      TString m("rene.brun@cern.ch, philippe.canal@fnal.gov, fons.rademakers@cern.ch");
      TString r("\\u$1 \\U$2\\E");
      TString s(m); re.Substitute(s, r);
      re.Print();
      printf("Substitute '%s','%s' => '%s'\n", m.Data(), r.Data(), s.Data());
      printf("\n");
   }
}
