#include <stdlib.h>
#include <string.h>

#include "NdbDefs.h"
#include "NdbEndfIO.h"
#include "NdbMTDir.h"

ClassImp(NdbMTDir);

/* ========= NdbMTDir ========  */
NdbMTDir::~NdbMTDir( )
{
   if (ZSYMAM)   free(ZSYMAM);
   if (ALAB)   free(ALAB);
   if (EDATE)   free(EDATE);
   if (AUTH)   free(AUTH);
   if (REF)   free(REF);
   if (DDATE)   free(DDATE);
   if (RDATE)   free(RDATE);
   if (ENDATE)   free(ENDATE);
} // ~NdbMTDir

/* -------- LoadENDF -------- */
Bool_t
NdbMTDir::LoadENDF( const char *filename )
{
   Bool_t      error;

   NdbEndfIO   endf(filename,TENDF_READ);

   if (!endf.IsOpen()) return kTRUE;

   endf.FindMFMT(1,MT());

   ZA   = endf.ReadReal(&error);
   AWR   = endf.ReadReal(&error);
   LRP   = endf.ReadInt(&error);
   LFI   = endf.ReadInt(&error);
   NLIB   = endf.ReadInt(&error);
   NMOD   = endf.ReadInt(&error);

   ELIS   = endf.ReadReal(&error);
   STA   = endf.ReadInt(&error);
   LIS   = endf.ReadInt(&error);
   LISO   = endf.ReadInt(&error);
   endf.ReadInt(&error);   // Skip one number
   NFOR   = endf.ReadInt(&error);

   AWI   = endf.ReadReal(&error);
   endf.ReadReal(&error);   // Skip three number
   endf.ReadInt(&error);
   endf.ReadInt(&error);
   NSUB   = endf.ReadInt(&error);
   NVER   = endf.ReadInt(&error);

   TEMP   = endf.ReadReal(&error);
   endf.ReadReal(&error);
   LDRV   = endf.ReadInt(&error);
   endf.ReadInt(&error);   // Skip one number
   NWD   = endf.ReadInt(&error);
   NXC   = endf.ReadInt(&error);

   //
   // Author, Dates, etc, skip 'em for the moment
   //
   endf.ReadLine();
   ZSYMAM   = strdup(endf.Substr(0,11));
   ALAB   = strdup(endf.Substr(11,10));
   EDATE   = strdup(endf.Substr(22,10));
   AUTH   = strdup(endf.Substr(33,33));
   endf.ReadLine();
   REF   = strdup(endf.Substr(1,20));
   DDATE   = strdup(endf.Substr(22,10));
   RDATE   = strdup(endf.Substr(33,10));
   ENDATE   = strdup(endf.Substr(55,6));

   // Skip comments
   for (int i=0; i<NWD-2; i++) {
      endf.ReadLine();
      INFO.Append(endf.Substr(0,66));
      INFO.Append("\n");
   }
   //printf("ZSYMAM=\"%s\"\n",ZSYMAM);
   //printf("ALAB=\"%s\"\n",ALAB);
   //printf("EDATE=\"%s\"\n",EDATE);
   //printf("AUTH=\"%s\"\n",AUTH);
   //printf("REF=\"%s\"\n",REF);
   //printf("DDATE=\"%s\"\n",DDATE);
   //printf("RDATE=\"%s\"\n",RDATE);
   //printf("ENDATE=\"%s\"\n",ENDATE);
   //printf("INFO=\"%s\"\n",INFO.Data());

   dir_mf.Set(NXC);
   dir_mt.Set(NXC);
   dir_mc.Set(NXC);
   dir_mod.Set(NXC);

   endf.ReadLine();
   for (int i=0; i<NXC; i++) {
      dir_mf.AddAt(endf.ReadInt(&error,2), i);
      dir_mt.AddAt(endf.ReadInt(&error), i);
      dir_mc.AddAt(endf.ReadInt(&error), i);
      dir_mod.AddAt(endf.ReadInt(&error), i);
      endf.ReadLine();
      //printf("MF=%d MT=%d MC=%d\n",dir_mf[i], dir_mt[i], dir_mc[i]);
   }
   return kFALSE;
} // LoadENDF
