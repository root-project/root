#ifndef ROOT7_REveTypes
#define ROOT7_REveTypes

#include "GuiTypes.h" // For Pixel_t only, to be changed.

#include "TString.h"

class TGeoManager;

namespace ROOT {
namespace Experimental {

typedef unsigned int ElementId_t;


//==============================================================================
// Exceptions, string functions
//==============================================================================

bool operator==(const TString &t, const std::string &s);
bool operator==(const std::string &s, const TString &t);

class REveException : public std::exception, public TString
{
public:
   REveException() {}
   REveException(const TString &s) : TString(s) {}
   REveException(const char *s) : TString(s) {}
   REveException(const std::string &s);

   virtual ~REveException() noexcept {}

   virtual const char *what() const noexcept { return Data(); }

   ClassDef(REveException, 1); // Exception-type thrown by Eve classes.
};

REveException operator+(const REveException &s1, const std::string &s2);
REveException operator+(const REveException &s1, const TString &s2);
REveException operator+(const REveException &s1, const char *s2);
REveException operator+(const REveException &s1, ElementId_t x);

} // namespace Experimental
} // namespace ROOT

#endif
