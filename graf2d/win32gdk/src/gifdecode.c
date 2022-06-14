/* @(#)root/win32gdk:$Id$ */
/* Author: Rene Brun   11/06/97*/
#include <stdio.h>
#include <string.h>


#define BITS    12                      /* largest code size */
#define TSIZE   4096                    /* tables size */

typedef unsigned char byte;

static int      Prefix[TSIZE];          /* prefix table */
static byte     Suffix[TSIZE];          /* suffix table */
static byte     OutCode[TSIZE];         /* output stack */

static byte     *ptr1,                  /* pointer to GIF array */
                *ptr2;                  /* pointer to PIX array */

static int      CurCodeSize,            /* current number of bits per code */
                CurMaxCode;             /* maximum code, given CurCodeSize */

static long     CurBit;                 /* current bit in GIF image data */

/***************************************************************
 *                                                             *
 ***************************************************************/
static int ReadCode()
{
  static long   b3[3], CurByte;
  static byte   lblk;
  int           shift, nbyte;
  long          OldByte;

  if (CurBit == -1) {
    lblk = 0;
    CurByte = -1;
  }

  CurBit += CurCodeSize;
  OldByte = CurByte;
  CurByte = CurBit/8;
  nbyte   = CurByte - OldByte;
  shift   = 17 + (CurBit%8) - CurCodeSize;
  while (nbyte-- > 0) {
    if (lblk == 0) {
      lblk = *ptr1++;
      if (lblk == 0) return -1;
    }
    b3[0] = b3[1];
    b3[1] = b3[2];
    b3[2] = *ptr1++;
    lblk--;
  }
  return (((b3[0]+0x100*b3[1]+0x10000*b3[2])>>shift) & (CurMaxCode-1));
}

/***************************************************************
 *                                                             *
 ***************************************************************/
static void OutPixel(byte pix)
{
  *ptr2++ = pix;
}

/***************************************************************
 *                                                             *
 * Name: GIFinfo                                Date: 03.10.94 *
 *                                                             *
 * Function: Get information on GIF image                      *
 *                                                             *
 * Input: GIFarr[] - compressed image in GIF format            *
 *                                                             *
 * Output: Width    - image width                              *
 *         Height   - image height                             *
 *         Ncols    - number of colors                         *
 *         return   - 0 - if O.K.                              *
 *                    1 - if error                             *
 *                                                             *
 ***************************************************************/
int GIFinfo(byte *GIFarr, int *Width, int *Height, int *Ncols)
{
  byte          b;

  ptr1 = GIFarr;

  /*   R E A D   H E A D E R   */

  if (strncmp((char *)GIFarr,"GIF87a",6) && strncmp((char *)GIFarr,"GIF89a",6))
  {
    fprintf(stderr,"\nGIFinfo: not a GIF\n");
    return 1;
  }

  ptr1 += 6;

  ptr1 += 2;                            /* screen width ... ignore */
  ptr1 += 2;                            /* screen height ... ignore */

  b         = *ptr1++;
  *Ncols    = 1 << ((b & 7) + 1);
  if ((b & 0x80) == 0) {                /* is there color map? */
    fprintf(stderr,"\nGIFinfo: warning! no color map\n");
    *Ncols = 0;
  }

  ++ptr1;                               /* background color ... ignore */
  b      = *ptr1++;                     /* supposed to be NULL */
  if (b) {
    fprintf(stderr,"\nGIFdecode: bad screen descriptor\n");
    return 1;
  }

  ptr1 += (*Ncols) * 3;                 /* skip color map */

  b      = *ptr1++;                     /* image separator */
  if (b != ',') {
    fprintf(stderr,"\nGIFinfo: no image separator\n");
    return 1;
  }

  ptr1   += 2;                          /* left offset ... ignore */
  ptr1   += 2;                          /* top offset ... ignore */
  b       = *ptr1++;                    /* image width */
  *Width  = b + 0x100*(*ptr1++);
  b       = *ptr1++;                    /* image height */
  *Height = b + 0x100*(*ptr1++);
  return 0;
}

/***************************************************************
 *                                                             *
 * Name: GIFdecode                              Date: 06.10.92 *
 *                                                             *
 * Function: Decode image from GIF array                       *
 *                                                             *
 * Input: GIFarr[] - compressed image in GIF format            *
 *                                                             *
 * Output: PIXarr[] - image (byte per pixel)                   *
 *         Width    - image width                              *
 *         Height   - image height                             *
 *         Ncols    - number of colors                         *
 *         R[]      - red components                           *
 *         G[]      - green components                         *
 *         B[]      - blue components                          *
 *         return   - 0 - if O.K.                              *
 *                    1 - if error                             *
 *                                                             *
 ***************************************************************/
int GIFdecode(byte *GIFarr, byte *PIXarr, int *Width, int *Height, int *Ncols, byte *R, byte *G, byte *B)
{
  byte          b,                      /* working variable */
                FinChar;                /* final character */

  int           i,                      /* working variable for loops */
                BitsPixel,              /* number of bits per pixel */
                IniCodeSize,            /* initial number of bits per code */
                ClearCode,              /* reset code */
                EOFCode,                /* end of file code */
                FreeCode,               /* first unused entry */
                CurCode,                /* current code */
                InCode,                 /* input code */
                OldCode,                /* previous code */
                PixMask,                /* mask for pixel */
                OutCount;               /* output stack counter */

  long          Npix;                   /* number of pixels */

  ptr1    = GIFarr;
  ptr2    = PIXarr;
  OldCode = 0;
  FinChar = 0;

  /*   R E A D   H E A D E R   */
  if (strncmp((char *)GIFarr,"GIF87a",6) && strncmp((char *)GIFarr,"GIF89a",6))
  {
    fprintf(stderr,"\nGIFinfo: not a GIF\n");
    return 1;
  }

  ptr1 += 6;

  ptr1 += 2;                            /* screen width ... ignore */
  ptr1 += 2;                            /* screen height ... ignore */

  b         = *ptr1++;
  BitsPixel = (b & 7) + 1;              /* # of bits per pixel */
  *Ncols    = 1 << BitsPixel;
  PixMask   = (*Ncols) - 1;             /* mask for pixel code */
  if ((b & 0x80) == 0) {                /* is there color map? */
    fprintf(stderr,"\nGIFdecode: warning! no color map\n");
    *Ncols = 0;
  }

  ++ptr1;                               /* background color ... ignore */
  b      = *ptr1++;                     /* supposed to be NULL */
  if (b) {
    fprintf(stderr,"\nGIFdecode: bad screen descriptor\n");
    return 1;
  }

  for (i=0; i<(*Ncols); i++) {          /* global color map */
    R[i] = *ptr1++;
    G[i] = *ptr1++;
    B[i] = *ptr1++;
  }

  b      = *ptr1++;                     /* image separator */
  if (b != ',') {
    fprintf(stderr,"\nGIFdecode: no image separator\n");
    return 1;
  }

  ptr1   += 2;                          /* left offset ... ignore */
  ptr1   += 2;                          /* top offset ... ignore */
  b       = *ptr1++;                    /* image width */
  *Width  = b + 0x100*(*ptr1++);
  b       = *ptr1++;                    /* image height */
  *Height = b + 0x100*(*ptr1++);

  b       = *ptr1++;                    /* local colors, interlace */
  if ((b & 0xc0) != 0) {
    fprintf(stderr,
            "\nGIFdecode: unexpected item (local colors or interlace)\n");
    return 1;
  }

  IniCodeSize = *ptr1++;
  CurCodeSize = ++IniCodeSize;
  CurMaxCode  = (1 << IniCodeSize);
  ClearCode   = (1 << (IniCodeSize - 1));
  EOFCode     = ClearCode + 1;
  FreeCode    = ClearCode + 2;

  /*   D E C O D E    I M A G E   */

  Npix     =(long) (*Width) * (*Height);
  OutCount = 0;
  CurBit   = -1;
  CurCode  = ReadCode();
  while (Npix > 0) {

    if (CurCode < 0) {
      fprintf(stderr,"\nGIFdecode: corrupted GIF (zero block length)\n");
      return 1;
    }

    if (CurCode == EOFCode) {
      fprintf(stderr,"\nGIFdecode: corrupted GIF (unexpected EOF)\n");
      return 1;
    }

    if (CurCode == ClearCode) {         /* clear code ... reset */

      CurCodeSize = IniCodeSize;
      CurMaxCode  = (1 << IniCodeSize);
      FreeCode    = ClearCode + 2;
      OldCode     = CurCode = ReadCode();
      FinChar     = CurCode;
      OutPixel(FinChar);
      Npix--;

    } else {                            /* image code */

      InCode = CurCode;
      if (CurCode >= FreeCode) {
        CurCode = OldCode;
        OutCode[OutCount++] = FinChar;
      }
      while (CurCode > PixMask) {       /* build output pixel chain */
        if (OutCount >= TSIZE) {
          fprintf(stderr,"\nGIFdecode: corrupted GIF (big output count)\n");
          return 1;
        }
      OutCode[OutCount++] = Suffix[CurCode];
      CurCode = Prefix[CurCode];
      }
      FinChar = CurCode;
      OutCode[OutCount++] = FinChar;

      for (i=OutCount-1; i>=0; i--) {   /* put out pixel chain */
        OutPixel(OutCode[i]);
        Npix--;
      }
      OutCount = 0;

      Prefix[FreeCode] = OldCode;       /* build the tables */
      Suffix[FreeCode] = FinChar;
      OldCode = InCode;

      FreeCode++;                       /* move pointer */
      if (FreeCode >= CurMaxCode) {
        if (CurCodeSize < BITS) {
          CurCodeSize++;
          CurMaxCode *= 2;
        }
      }
    }
    CurCode = ReadCode();
  }
  return 0;
}
