#include "NdbMF.h"

ClassImp(NdbMF);

#include <Riostream.h>

/* -------- Section -------- */
NdbMT *
NdbMF::Section(Int_t id)
{
   TObjArrayIter   iter(&aSection);
   while (iter()) {
      NdbMT   *mt = (NdbMT*)(iter.Next());
      if (mt->MT() == id)
         return mt;
   }
   return NULL;
} // Section

/* -------- Section -------- */
NdbMT *
NdbMF::Section(const char *name )
{
   TObjArrayIter   iter(&aSection);
   while (iter()) {
      NdbMT   *mt = (NdbMT*)(iter.Next());
      if (mt->Description().CompareTo(name))
         return mt;
   }
   return NULL;
} // Section

/* ---------- Add ---------- */
void
NdbMF::Add(NdbMT& )
{
   std::cout << "NdbMF::add(sec)" << std::endl;
} // Add

/* ---------- EnumerateENDFType ---------- */
/* Enumerates all the available sections inside the ENDF file
 * @param sec   Find the next section after <B>sec</B>
 * @return   Next available section in ENDF file, -1 if EOF
 */
Int_t
NdbMF::EnumerateENDFType( Int_t )
{
   std::cout << "NdbMF ::enumerateENDFSection" << std::endl;
   return 0;
} // EnumerateENDFType
