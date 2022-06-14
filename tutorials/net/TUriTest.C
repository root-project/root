/// \file
/// \ingroup tutorial_net
/// Rudimentary TUri test macro.
///
/// \macro_code
///
/// \author Gerhard E. Bruckner 2007-10-18

#include <TUri.h>


Bool_t TestResolutionHelper(TUri reference, TUri nominal, TUri &base)
{
   TUri actual = TUri::Transform(reference, base);
   if (!(nominal == actual))
      Printf("\tERROR: %s => %s (should read: %s)", reference.GetUri().Data(), actual.GetUri().Data(), nominal.GetUri().Data());
   return (nominal == actual);
}


Bool_t TestResolution()
{
   TUri base = TUri("http://a/b/c/d;p?q");
   Bool_t success = kTRUE;

   // 5.4.1. Normal Examples
   success &= TestResolutionHelper("g:h", "g:h", base);
   success &= TestResolutionHelper("g", "http://a/b/c/g", base);
   success &= TestResolutionHelper("./g", "http://a/b/c/g", base);
   success &= TestResolutionHelper("g/", "http://a/b/c/g/", base);
   success &= TestResolutionHelper("/g", "http://a/g", base);
   success &= TestResolutionHelper("//g", "http://g", base);
   success &= TestResolutionHelper("?y", "http://a/b/c/d;p?y", base);
   success &= TestResolutionHelper("g?y", "http://a/b/c/g?y", base);
   success &= TestResolutionHelper("#s", "http://a/b/c/d;p?q#s", base);
   success &= TestResolutionHelper("g#s", "http://a/b/c/g#s", base);
   success &= TestResolutionHelper("g?y#s", "http://a/b/c/g?y#s", base);
   success &= TestResolutionHelper(";x", "http://a/b/c/;x", base);
   success &= TestResolutionHelper("g;x", "http://a/b/c/g;x", base);
   success &= TestResolutionHelper("g;x?y#s", "http://a/b/c/g;x?y#s", base);
   success &= TestResolutionHelper("", "http://a/b/c/d;p?q", base);
   success &= TestResolutionHelper(".", "http://a/b/c/", base);
   success &= TestResolutionHelper("./", "http://a/b/c/", base);
   success &= TestResolutionHelper("..", "http://a/b/", base);
   success &= TestResolutionHelper("../", "http://a/b/", base);
   success &= TestResolutionHelper("../g", "http://a/b/g", base);
   success &= TestResolutionHelper("../..", "http://a/", base);
   success &= TestResolutionHelper("../../", "http://a/", base);
   success &= TestResolutionHelper("../../g", "http://a/g", base);
   //  5.4.2. Abnormal Examples
   success &= TestResolutionHelper("../../../g", "http://a/g", base);
   success &= TestResolutionHelper("../../../../g", "http://a/g", base);
   success &= TestResolutionHelper("/./g", "http://a/g", base);
   success &= TestResolutionHelper("/../g", "http://a/g", base);
   success &= TestResolutionHelper("g.", "http://a/b/c/g.", base);
   success &= TestResolutionHelper(".g", "http://a/b/c/.g", base);
   success &= TestResolutionHelper("g..", "http://a/b/c/g..", base);
   success &= TestResolutionHelper("..g", "http://a/b/c/..g", base);
   success &= TestResolutionHelper("./../g", "http://a/b/g", base);
   success &= TestResolutionHelper("./g/.", "http://a/b/c/g/", base);
   success &= TestResolutionHelper("g/./h", "http://a/b/c/g/h", base);
   success &= TestResolutionHelper("g/../h", "http://a/b/c/h", base);
   success &= TestResolutionHelper("g;x=1/./y", "http://a/b/c/g;x=1/y", base);
   success &= TestResolutionHelper("g;x=1/../y", "http://a/b/c/y", base);
   success &= TestResolutionHelper("g?y/./x", "http://a/b/c/g?y/./x", base);
   success &= TestResolutionHelper("g?y/../x", "http://a/b/c/g?y/../x", base);
   success &= TestResolutionHelper("g#s/./x", "http://a/b/c/g#s/./x", base);
   success &= TestResolutionHelper("g#s/../x", "http://a/b/c/g#s/../x", base);
   success &= TestResolutionHelper("http:g", "http:g", base);
   return(success);
}


Bool_t TestPct()
{
   TString errors = "";
   for (char i = 0; i < 127; i++) {
      if (TUri::PctDecode(TUri::PctEncode(i)) != i) {
         char buffer[10];
         sprintf(buffer, "0x%02x, ", i);
         errors = errors + buffer;
      }
   }
   if (!errors.IsNull())
      Printf("\tERROR at %s", errors.Data());
   else
      Printf("\tOK");
   return errors.IsNull();
}

Bool_t TestComposition()
{
   TString composed = "http://user:pass@host.org/some/path/file.avi?key1=value1#anchor3";
   TUri uri;
   uri.SetScheme("http");
   uri.SetUserInfo("user:pass");
   uri.SetHost("host.org");
   uri.SetPath("/some/path/file.avi");
   uri.SetQuery("key1=value1");
   uri.SetFragment("anchor3");
   return uri.GetUri() == composed;
}

void Answer(Bool_t success)
{
   if (success)
      Printf("---> SUCCESS\n");
   else
      Printf("---> F A I L E D   F A I L E D   F A I L E D\n");
}

Bool_t TestValidation()
{
   // validating examples from RFC chapter 1.1.2
   Bool_t valid = kTRUE;
   valid &= TUri("ftp://ftp.is.co.za/rfc/rfc1808.txt").IsUri();
   valid &= TUri("http://www.ietf.org/rfc/rfc2396.txt").IsUri();
   // IPV6 example excluded
   valid &= TUri("mailto:John.Doe@example.com").IsUri();
   valid &= TUri("news:comp.infosystems.www.servers.unix").IsUri();
   valid &= TUri("tel:+1-816-555-1212").IsUri();
   valid &= TUri("telnet://192.0.2.16:80/").IsUri();
   valid &= TUri("urn:oasis:names:specification:docbook:dtd:xml:4.1.2").IsUri();
   return valid;
}


void TUriTest()
{
   Printf("\n\nTUri test macro ...");
   Printf("---> Validation");
   Answer(TestValidation());
   Printf("---> Reference Resolution");
   Answer(TestResolution());
   Printf("---> PCT Conversion");
   Answer(TestPct());
   Printf("---> Equivalence and Normalisation");
   Answer(TUri("example://a/b/c/%7Bfoo%7D") == TUri("eXAMPLE://a/./b/../b/%63/%7bfoo%7d"));
   Printf("---> Composition");
   Answer(TestComposition());
}

