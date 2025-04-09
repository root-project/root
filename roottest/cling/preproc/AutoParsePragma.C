#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wsign-conversion"

namespace boost { namespace mpl {

// Commenting the next line make the assert failure go away
struct TTUBE {};

}}

#pragma GCC diagnostic pop

int AutoParsePragma() {
   return 0;
}
