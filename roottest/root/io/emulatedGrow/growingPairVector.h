#ifndef GROWINGPAIRVECTOR_H
#define GROWINGPAIRVECTOR_H

#include <string>
#include <utility>
#include <vector>

// The number of entries and their sizes are chosen so that the collection grows
// on every entry, forcing the reader's buffer to reallocate several times.
const int kNEntries = 12;

inline int NElements(int entry)
{
   return 8 * (entry + 1);
}

// Alternate between a string that fits in the small-string-optimization buffer
// and one that does not, so both layouts get relocated.
inline std::string ElementString(int entry, int i)
{
   std::string s = std::to_string(entry) + "_" + std::to_string(i);
   if (i % 3 == 0)
      s += std::string(40, 'x');
   return s;
}

inline double ElementValue(int entry, int i)
{
   return entry * 1000.0 + i;
}

// A class of our own on purpose: read back without this dictionary, its
// vector<pair<string,double>> member goes through TEmulatedCollectionProxy.
class GrowingPairVector {
public:
   std::vector<std::pair<std::string, double>> fData;

   void Fill(int entry)
   {
      fData.clear();
      for (int i = 0; i < NElements(entry); ++i)
         fData.emplace_back(ElementString(entry, i), ElementValue(entry, i));
   }
};

#endif
