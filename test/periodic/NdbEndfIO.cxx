#include <stdlib.h>

#include "NdbEndfIO.h"

#define NUMBER_SIZE   11

char   NdbEndfIO::_str[100];      // define static _str var

ClassImp(NdbEndfIO);

/* ============ NdbEndfIO ============== */
NdbEndfIO::NdbEndfIO( const char *filename, Int_t mode )
{
   f = fopen(filename,mode==TENDF_READ?"r":"w");
   matStart = 0;
   mfStart = 0;
   mtStart = 0;
   iMAT = 0;
   iMF = 0;
   iMT = 0;
   lineNum = 0;
   lineTxt[0] = 0;
} // NdbEndfIO


/* -------- FindMAT --------- */
// Will search for the specific material
Bool_t
NdbEndfIO::FindMAT( Int_t mat, Bool_t rewind )
{
   if (rewind)
      fseek(f, 0L, SEEK_SET);

   while (ReadLine())
      if (iMAT == mat)
         return TRUE;

   return FALSE;
} // FindMAT

/* -------- FindMATMF --------- */
// Will search for the specific material - mf
Bool_t
NdbEndfIO::FindMATMF( Int_t mat, Int_t mf, Bool_t rewind )
{
   FindMAT(mat,rewind);

   do {
      if (iMF == mf)
         return TRUE;
   } while (ReadLine());

   return FALSE;
} // FindMATMF

/* -------- FindMATMFMT --------- */
// Will search for the specific material - mf - mt
Bool_t
NdbEndfIO::FindMATMFMT( Int_t mat, Int_t mf, Int_t mt, Bool_t rewind )
{
   FindMATMF(mat,mf,rewind);

   do {
      if (iMT == mt)
         return TRUE;
   } while (ReadLine());

   return FALSE;
} // FindMATMFMT

/* ------------ FindMFMT ------------- */
Bool_t
NdbEndfIO::FindMFMT( Int_t mf, Int_t mt )
{
   // If current MFMT is higher than what we search go to the beggining
   if (mf*1000+mt >= iMF*1000+iMT) {
      RewindMAT();
      if (!ReadLine())
         return FALSE;
   }

   while (mf*1000+mt > iMF*1000+iMT)
      if (!ReadLine())
         return FALSE;
   return TRUE;
} // FindMFMT

/* -------- Substr ---------- */
char *
NdbEndfIO::Substr(Int_t start, Int_t length)
{
   if (start + length > lineLen) {
//      error(ERR_INVALID_RECORD);
      return NULL;
   }
   memcpy(_str, lineTxt+start, length);
   _str[length] = 0;

   return _str;
} // Substr

/* -------- NextNumber ---------- */
/* Advance lastNumber Point to the correct number
 * so to read 6 numbers of width 11 in each line
 */
Bool_t
NdbEndfIO::NextNumber(Int_t pos)
{
   if (pos<0 || pos>5*NUMBER_SIZE) {
      lastNumPos += NUMBER_SIZE;
      if (lineTxt[0]=='\0' || lastNumPos>5*NUMBER_SIZE) {
         Int_t   mf = iMF;   // Remember current MF, MT
         Int_t   mt = iMT;
         if (!ReadLine() || (mf*1000+mt != iMF*1000+iMT))
            return TRUE;
         lastNumPos = 0;
      }
   } else {
      lastNumPos = pos * NUMBER_SIZE;
   }
   return FALSE;
} // NextNumber

/* -------- SubReadInt ---------- */
Int_t
NdbEndfIO::SubReadInt(Int_t start, Int_t length)
{
   NdbEndfIO::Substr(start,length);
   return atoi(_str);
} // SubReadInt

/* ---------- ReadInt ------------- */
/* Read one by one the int numbers with length 11 from the current line.
 * If the this is the last one then it prompts for an new line
 * only if it stays inside the same MF:MT type
 */
Int_t
NdbEndfIO::ReadInt(Bool_t *error, Int_t pos)
{
   if (NextNumber(pos)) {
      *error = TRUE;
      return 0;
   }
   *error = FALSE;
   return SubReadInt(lastNumPos,NUMBER_SIZE);
} // ReadInt

/* -------- SubReadReal ---------- */
Float_t
NdbEndfIO::SubReadReal(Int_t start, Int_t length)
{
   char   numstr[20];
   char   *dst, *src;

   NdbEndfIO::Substr(start,length);

   // Real numbers are in the following formats
   //   +/- N.NNNNNN+/-N
   //   +/- N.NNNNN+/-bN   (b=Blank)
   //   +/- N.NNNNN+/-NN

   dst = numstr;
   src = _str;

   /* skip blanks */
   while (*src==' ') src++;

   /* copy sign */
   if (*src=='+' || *src=='-') *dst++ = *src++;

   /* copy realpart */
   while (1) {
      if (memchr(".0123456789",*src,11))
         *dst++ = *src++;
      else
         break;
   }

   /* append the exponent */
   *dst++ = 'E';

   /* copy sign */
   if (*src=='+' || *src=='-') *dst++ = *src++;

   /* skip blanks */
   while (*src==' ') src++;

   /* copy exponent */
   while (1) {
      if (memchr("0123456789",*src,10))
         *dst++ = *src++;
      else
         break;
   }

   /* close the destination */
   *dst = 0;

   /* if something is still in source then we have an error number */
   if (*src) {
//      error(ERR_INVALID_REAL_NUMBER);
      return 0.0;
   }

   return atof(numstr);
} // SubReadReal

/* ---------- ReadReal ------------- */
/* Read one by one the real numbers with length 11 from the current line.
 * If the this is the last one then it prompts for an new line
 * only if it stays inside the same MF:MT type
 */
Float_t
NdbEndfIO::ReadReal(Bool_t *error, Int_t pos)
{
   if (NextNumber(pos)) {
      *error = TRUE;
      return 0.0;
   }
   *error = FALSE;
   return SubReadReal(lastNumPos,NUMBER_SIZE);
} // ReadReal

/* -------- ReadLine -------- */
Bool_t
NdbEndfIO::ReadLine()
{
   Long_t   current_pos;

   // --- Force the next number reading to be from the beggining of line
   lastNumPos = -NUMBER_SIZE;

   // --- Have we reached the end of file
   if (feof(f)) {
      lineTxt[0] = 0;
      return FALSE;
   }

   // --- Remember the position of start of line ---
   current_pos = ftell(f);

   // --- Read one line ---
   fgets(lineTxt,sizeof(lineTxt),f);

   lineLen = strlen(lineTxt);

   // Chop the trailing newline char
   if (lineTxt[lineLen-1] == '\n')
      lineTxt[--lineLen] = 0;

   // --- Remember previous values ---
   Int_t   oldMAT   = iMAT;
   Int_t   oldMF   = iMF;
   Int_t   oldMT   = iMT;

   // --- Read the material, mf, mt and line number ---
   iMAT    = SubReadInt(66, 4);
   iMF     = SubReadInt(70, 2);
   iMT     = SubReadInt(72, 3);
   lineNum = SubReadInt(75, 5);

   // --- Update pointers if necessary ---
   if (iMAT && iMAT != oldMAT)
      matStart = current_pos;

   if (iMF && iMF != oldMF)
      mfStart = current_pos;

   if (iMT && iMT != oldMT)
      mtStart = current_pos;

   return TRUE;
} // ReadLine
