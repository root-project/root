/* @(#)root/x11:$Id$ */
/* Author: E.Chernyaev   19/01/94
 * C++ interface: S.Linev 20/04/2026 */

#include "gifencode.h"

#include <vector>


#define BITS     12                     /* largest code size */
#define THELIMIT 4096                   /* NEVER generate this */
#define SHIFT    4                      /* shift for hashing */


void TGifEncode::put_byte(unsigned char b)
{
   if (fOut && (ferror(fOut) == 0)) {
      fputc(b, fOut);
      fNbyte++;
   }
}

bool TGifEncode::OpenFile(const char *fname)
{
   fOut = fopen(fname, "w+");
   return fOut != nullptr;
}

void TGifEncode::CloseFile()
{
   if (fOut)
      fclose(fOut);
   fOut = nullptr;
}


void TGifEncode::char_init()
{
   a_count = 0;
   cur_accum = 0;
   cur_bits  = 0;
}

void TGifEncode::char_out(unsigned char c)
{
   accum[a_count++] = c;
   if (a_count >= 254)
      char_flush();
}

void TGifEncode::char_flush()
{
  if (a_count == 0) return;
  put_byte(a_count);
  for (int i=0; i<a_count; i++) {
    put_byte(accum[i]);
  }
  a_count = 0;
}

void TGifEncode::put_short(int word)
{
  put_byte(word & 0xFF);
  put_byte((word>>8) & 0xFF);
}

/***************************************************************
 *                                                             *
 * Name: output                                 Date: 02.10.92 *
 *                                                             *
 * Function: output GIF code                                   *
 *                                                             *
 * Input: code - GIF code                                      *
 *                                                             *
 ***************************************************************/
void TGifEncode::output(int code)
{
   /*   O U T P U T   C O D E   */

   static unsigned long masks[] = { 0x0000,
                                    0x0001, 0x0003, 0x0007, 0x000F,
                                    0x001F, 0x003F, 0x007F, 0x00FF,
                                    0x01FF, 0x03FF, 0x07FF, 0x0FFF,
                                    0x1FFF, 0x3FFF, 0x7FFF, 0xFFFF };

   cur_accum &= masks[cur_bits];
   if (cur_bits > 0)
     cur_accum |= ((long)code << cur_bits);
   else
     cur_accum = code;
   cur_bits += CurCodeSize;
   while( cur_bits >= 8 ) {
     char_out(cur_accum & 0xFF);
     cur_accum >>= 8;
     cur_bits -= 8;
   }

  /*   R E S E T   */

  if (code == ClearCode ) {
    memset((char *) HashTab, -1, sizeof(HashTab));
    FreeCode = ClearCode + 2;
    CurCodeSize = IniCodeSize;
    CurMaxCode  = (1 << (IniCodeSize)) - 1;
  }

  /*   I N C R E A S E   C O D E   S I Z E   */

  if (FreeCode > CurMaxCode ) {
      CurCodeSize++;
      if ( CurCodeSize == BITS )
        CurMaxCode = THELIMIT;
      else
        CurMaxCode = (1 << (CurCodeSize)) - 1;
   }

  /*   E N D   O F   F I L E :  write the rest of the buffer  */

  if( code == EOFCode ) {
    while( cur_bits > 0 ) {
      char_out(cur_accum & 0xff);
      cur_accum >>= 8;
      cur_bits -= 8;
    }
    char_flush();
  }
}



/***********************************************************************
 *                                                                     *
 * Name: GIFencode                                   Date:    02.10.92 *
 * Author: E.Chernyaev (IHEP/Protvino)               Revised:          *
 *                                                                     *
 * Function: GIF compression of the image                              *
 *                                                                     *
 * Input: Width      - image width  (must be >= 8)                     *
 *        Height     - image height (must be >= 8)                     *
 *        Ncol       - number of colors                                *
 *        R[]        - red components                                  *
 *        G[]        - green components                                *
 *        B[]        - blue components                                 *
 *        ScLine[]   - array for scan line (byte per pixel)            *
 *        get_scline - user routine to read scan line:                 *
 *                       get_scline(y, Width, ScLine)                  *
 *        pb         - user routine for "put_byte": pb(b)              *
 *                                                                     *
 * Return: size of GIF                                                 *
 *                                                                     *
 ***********************************************************************/
long TGifEncode::GIFencode(int Width, int Height, int Ncol, unsigned char *R, unsigned char *G, unsigned char *B)
//               void(*get_scline) ARGS((int, int, byte *)), void(*pb) ARGS((byte)))
{
  long          CodeK;
  int           ncol, i, x, y, disp, Code, K;

  std::vector<unsigned char> ScLine(Width);

  /*   C H E C K   P A R A M E T E R S   */

  Code = 0;
  if (Width <= 0 || Width > 4096 || Height <= 0 || Height > 4096) {
    fprintf(stderr,
            "\nGIFencode: incorrect image size: %d x %d\n", Width, Height);
    return 0;
  }

  if (Ncol <= 0 || Ncol > 256) {
    fprintf(stderr,"\nGIFencode: wrong number of colors: %d\n", Ncol);
    return 0;
  }

  /*   I N I T I A L I S A T I O N   */

  fNbyte  = 0;
  char_init();                          /* initialise "char_..." routines */

  /*   F I N D   #   O F   B I T S   P E R    P I X E L   */

  BitsPixel = 1;
  if (Ncol > 2)   BitsPixel = 2;
  if (Ncol > 4)   BitsPixel = 3;
  if (Ncol > 8)   BitsPixel = 4;
  if (Ncol > 16)  BitsPixel = 5;
  if (Ncol > 32)  BitsPixel = 6;
  if (Ncol > 64)  BitsPixel = 7;
  if (Ncol > 128) BitsPixel = 8;

  ncol  = 1 << BitsPixel;
  IniCodeSize = BitsPixel;
  if (BitsPixel <= 1) IniCodeSize = 2;

  /*   W R I T E   H E A D E R  */

  put_byte('G');                        /* magic number: GIF87a */
  put_byte('I');
  put_byte('F');
  put_byte('8');
  put_byte('7');
  put_byte('a');

  put_short(Width);                     /* screen size */
  put_short(Height);

  K  = 0x80;                            /* yes, there is a color map */
  K |= (8-1)<<4;                        /* OR in the color resolution */
  K |= (BitsPixel - 1);                 /* OR in the # of bits per pixel */
  put_byte(K);

  put_byte(0);                          /* background color */
  put_byte(0);                          /* future expansion byte */

  for (i=0; i<Ncol; i++) {              /* global colormap */
    put_byte(R[i]);
    put_byte(G[i]);
    put_byte(B[i]);
  }
  for (; i<ncol; i++) {
    put_byte(0);
    put_byte(0);
    put_byte(0);
  }

  put_byte(',');                        /* image separator */
  put_short(0);                         /* left offset of image */
  put_short(0);                         /* top offset of image */
  put_short(Width);                     /* image size */
  put_short(Height);
  put_byte(0);                          /* no local colors, no interlace */
  put_byte(IniCodeSize);                /* initial code size */

  /*   L W Z   C O M P R E S S I O N   */

  CurCodeSize = ++IniCodeSize;
  CurMaxCode  = (1 << (IniCodeSize)) - 1;
  ClearCode   = (1 << (IniCodeSize - 1));
  EOFCode     = ClearCode + 1;
  FreeCode    = ClearCode + 2;
  output(ClearCode);
  for (y=0; y<Height; y++) {
    get_scline(y, Width, ScLine.data());
    x     = 0;
    if (y == 0)
      Code  = ScLine[x++];
    while(x < Width) {
      K     = ScLine[x++];              /* next symbol */
      CodeK = ((long) K << BITS) + Code;  /* set full code */
      i     = (K << SHIFT) ^ Code;      /* xor hashing */

      if (HashTab[i] == CodeK) {        /* full code found */
        Code = CodeTab[i];
        continue;
      }
      else if (HashTab[i] < 0 )         /* empty slot */
        goto NOMATCH;

      disp  = HSIZE - i;                /* secondary hash */
      if (i == 0) disp = 1;

PROBE:
      if ((i -= disp) < 0)
        i  += HSIZE;

      if (HashTab[i] == CodeK) {        /* full code found */
        Code = CodeTab[i];
        continue;
      }

      if (HashTab[i] > 0)               /* try again */
        goto PROBE;

NOMATCH:
      output(Code);                     /* full code not found */
      Code = K;

      if (FreeCode < THELIMIT) {
        CodeTab[i] = FreeCode++;        /* code -> hashtable */
        HashTab[i] = CodeK;
      }
      else
        output(ClearCode);
    }
  }
   /*   O U T P U T   T H E   R E S T  */

  output(Code);
  output(EOFCode);
  put_byte(0);                          /* zero-length packet (EOF) */
  put_byte(';');                        /* GIF file terminator */

  return fNbyte;
}

