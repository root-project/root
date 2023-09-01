/******************************************************************************
*   "Gif-Lib" - Yet another gif library.				      *
*									      *
* Written by:  Gershon Elber			IBM PC Ver 1.1,	Aug. 1990     *
*******************************************************************************
* The kernel of the GIF Decoding process can be found here.		      *
*******************************************************************************
* History:								      *
* 16 Jun 89 - Version 1.0 by Gershon Elber.				      *
*  3 Sep 90 - Version 1.1 by Gershon Elber (Support for Gif89, Unique names). *
******************************************************************************/

#ifdef _WIN32
#include "../win32/config.h"
#else
#include "../config.h"
#endif

#if defined (__MSDOS__) && !defined(__DJGPP__) && !defined(__GNUC__)
#include <alloc.h>
#include <sys\stat.h>
#else
#include <sys/types.h>
#include <sys/stat.h>
#endif /* __MSDOS__ */

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include "gif_lib.h"
#include "gif_lib_private.h"

#define COMMENT_EXT_FUNC_CODE	0xfe /* Extension function code for comment. */

/* avoid extra function call in case we use fread (TVT) */
#define READ(_gif,_buf,_len)                                     \
  (((GifFilePrivateType*)_gif->Private)->Read ?                   \
    ((GifFilePrivateType*)_gif->Private)->Read(_gif,_buf,_len) : \
    fread(_buf,1,_len,((GifFilePrivateType*)_gif->Private)->File))

static int DGifGetWord(GifFileType *GifFile, int *Word);
static int DGifSetupDecompress(GifFileType *GifFile);
static int DGifDecompressLine(GifFileType *GifFile, GifPixelType *Line,
								int LineLen);
static int DGifGetPrefixChar(unsigned int *Prefix, int Code, int ClearCode);
static int DGifDecompressInput(GifFileType *GifFile, int *Code);
static int DGifBufferedInput(GifFileType *GifFile, GifByteType *Buf,
						     GifByteType *NextByte);

/******************************************************************************
* GifFileType constructor with user supplied input function (TVT)             *
*                                                                             *
******************************************************************************/
GifFileType *DGifOpen( void* userData, InputFunc readFunc )
{
    unsigned char Buf[GIF_STAMP_LEN+1];
    GifFileType *GifFile;
    GifFilePrivateType *Private;


    if ((GifFile = (GifFileType *) malloc(sizeof(GifFileType))) == NULL) {
	  _GifError = D_GIF_ERR_NOT_ENOUGH_MEM;
	  return NULL;
    }

    memset(GifFile, '\0', sizeof(GifFileType));

    if (!(Private = (GifFilePrivateType*) malloc(sizeof(GifFilePrivateType)))){
	  _GifError = D_GIF_ERR_NOT_ENOUGH_MEM;
	  free((char *) GifFile);
	  return NULL;
    }

    GifFile->Private = (VoidPtr) Private;
    Private->FileHandle = 0;
    Private->File = 0;
    Private->FileState = FILE_STATE_READ;

	Private->Read     = readFunc;  /* TVT */
	GifFile->UserData = userData;    /* TVT */

    /* Lets see if this is a GIF file: */
    if ( READ( GifFile, Buf, GIF_STAMP_LEN) != GIF_STAMP_LEN) {
	  _GifError = D_GIF_ERR_READ_FAILED;
	  free((char *) Private);
	  free((char *) GifFile);
	  return NULL;
    }

    /* The GIF Version number is ignored at this time. Maybe we should do    */
    /* something more useful with it.					     */
    Buf[GIF_STAMP_LEN] = 0;
    if (strncmp(GIF_STAMP, (char*)Buf, GIF_VERSION_POS) != 0) {
	  _GifError = D_GIF_ERR_NOT_GIF_FILE;
	  free((char *) Private);
	  free((char *) GifFile);
	  return NULL;
    }

    if (DGifGetScreenDesc(GifFile) == GIF_ERROR) {
	  free((char *) Private);
	  free((char *) GifFile);
	  return NULL;
    }

    _GifError = 0;

    return GifFile;
}

/******************************************************************************
*   This routine should be called before any other DGif calls. Note that      *
* this routine is called automatically from DGif file open routines.	      *
******************************************************************************/
int DGifGetScreenDesc(GifFileType *GifFile)
{
    int i, BitsPerPixel;
    GifByteType Buf[3];
    GifFilePrivateType *Private = (GifFilePrivateType *) GifFile->Private;

    if (!IS_READABLE(Private)) {
	/* This file was NOT open for reading: */
	_GifError = D_GIF_ERR_NOT_READABLE;
	return GIF_ERROR;
    }

    /* Put the screen descriptor into the file: */
    if (DGifGetWord(GifFile, &GifFile->SWidth) == GIF_ERROR ||
	DGifGetWord(GifFile, &GifFile->SHeight) == GIF_ERROR)
	return GIF_ERROR;

    if (READ( GifFile, Buf, 3) != 3) {
	_GifError = D_GIF_ERR_READ_FAILED;
	return GIF_ERROR;
    }
    GifFile->SColorResolution = (((Buf[0] & 0x70) + 1) >> 4) + 1;
    BitsPerPixel = (Buf[0] & 0x07) + 1;
    GifFile->SBackGroundColor = Buf[1];
    if (Buf[0] & 0x80) {		     /* Do we have global color map? */

	GifFile->SColorMap = MakeMapObject(1 << BitsPerPixel, NULL);

	/* Get the global color map: */
	for (i = 0; i < GifFile->SColorMap->ColorCount; i++) {
	    if (READ(GifFile, Buf, 3) != 3) {
		_GifError = D_GIF_ERR_READ_FAILED;
		return GIF_ERROR;
	    }
	    GifFile->SColorMap->Colors[i].Red = Buf[0];
	    GifFile->SColorMap->Colors[i].Green = Buf[1];
	    GifFile->SColorMap->Colors[i].Blue = Buf[2];
	}
    }

    return GIF_OK;
}

/******************************************************************************
*   This routine should be called before any attemp to read an image.         *
******************************************************************************/
int DGifGetRecordType(GifFileType *GifFile, GifRecordType *Type)
{
    GifByteType Buf;
    GifFilePrivateType *Private = (GifFilePrivateType *) GifFile->Private;

    if (!IS_READABLE(Private)) {
	/* This file was NOT open for reading: */
	_GifError = D_GIF_ERR_NOT_READABLE;
	return GIF_ERROR;
    }

    if (READ( GifFile, &Buf, 1) != 1) {
	_GifError = D_GIF_ERR_READ_FAILED;
	return GIF_ERROR;
    }

    switch (Buf) {
	case ',':
	    *Type = IMAGE_DESC_RECORD_TYPE;
	    break;
	case '!':
	    *Type = EXTENSION_RECORD_TYPE;
	    break;
	case ';':
	    *Type = TERMINATE_RECORD_TYPE;
	    break;
	default:
	    *Type = UNDEFINED_RECORD_TYPE;
	    _GifError = D_GIF_ERR_WRONG_RECORD;
	    return GIF_ERROR;
    }

    return GIF_OK;
}

/******************************************************************************
*   This routine should be called before any attemp to read an image.         *
*   Note it is assumed the Image desc. header (',') has been read.	      *
******************************************************************************/
int DGifGetImageDesc(GifFileType *GifFile)
{
    int i, BitsPerPixel;
    GifByteType Buf[3];
    GifFilePrivateType *Private = (GifFilePrivateType *) GifFile->Private;
	SavedImage	*sp;

    if (!IS_READABLE(Private)) {
	/* This file was NOT open for reading: */
	_GifError = D_GIF_ERR_NOT_READABLE;
	return GIF_ERROR;
    }
    if (DGifGetWord(GifFile, &GifFile->Image.Left) == GIF_ERROR ||
	DGifGetWord(GifFile, &GifFile->Image.Top) == GIF_ERROR ||
	DGifGetWord(GifFile, &GifFile->Image.Width) == GIF_ERROR ||
	DGifGetWord(GifFile, &GifFile->Image.Height) == GIF_ERROR)
	return GIF_ERROR;
    if (READ(GifFile,Buf, 1) != 1) {
	_GifError = D_GIF_ERR_READ_FAILED;
	return GIF_ERROR;
    }
    BitsPerPixel = (Buf[0] & 0x07) + 1;
    GifFile->Image.Interlace = (Buf[0] & 0x40);
    if (Buf[0] & 0x80) {	    /* Does this image have local color map? */

	if (GifFile->Image.ColorMap && GifFile->SavedImages == NULL)
	    FreeMapObject(GifFile->Image.ColorMap);

	GifFile->Image.ColorMap = MakeMapObject(1 << BitsPerPixel, NULL);
	/* Get the image local color map: */
	for (i = 0; i < GifFile->Image.ColorMap->ColorCount; i++) {
	    if (READ(GifFile,Buf, 3) != 3) {
		_GifError = D_GIF_ERR_READ_FAILED;
		return GIF_ERROR;
	    }
	    GifFile->Image.ColorMap->Colors[i].Red = Buf[0];
	    GifFile->Image.ColorMap->Colors[i].Green = Buf[1];
	    GifFile->Image.ColorMap->Colors[i].Blue = Buf[2];
	}
    }
    if (GifFile->SavedImages) {
	    if ((GifFile->SavedImages = (SavedImage *)realloc(GifFile->SavedImages,
		     sizeof(SavedImage) * (GifFile->ImageCount + 1))) == NULL) {
	        _GifError = D_GIF_ERR_NOT_ENOUGH_MEM;
	        return GIF_ERROR;
	    }
    } else {
        if ((GifFile->SavedImages =
             (SavedImage *)malloc(sizeof(SavedImage))) == NULL) {
            _GifError = D_GIF_ERR_NOT_ENOUGH_MEM;
            return GIF_ERROR;
        }
    }

	sp = &GifFile->SavedImages[GifFile->ImageCount];
	memcpy(&sp->ImageDesc, &GifFile->Image, sizeof(GifImageDesc));
    if (GifFile->Image.ColorMap != NULL) {
        sp->ImageDesc.ColorMap = MakeMapObject(
                GifFile->Image.ColorMap->ColorCount,
                GifFile->Image.ColorMap->Colors);
    }
	sp->RasterBits = (char *)NULL;
	sp->ExtensionBlockCount = 0;
	sp->ExtensionBlocks = (ExtensionBlock *)NULL;

    GifFile->ImageCount++;

    Private->PixelCount = (long) GifFile->Image.Width *
			    (long) GifFile->Image.Height;

    DGifSetupDecompress(GifFile);  /* Reset decompress algorithm parameters. */

    return GIF_OK;
}

/******************************************************************************
*  Get one full scanned line (Line) of length LineLen from GIF file.	      *
******************************************************************************/
int DGifGetLine(GifFileType *GifFile, GifPixelType *Line, int LineLen)
{
    GifByteType *Dummy;
    GifFilePrivateType *Private = (GifFilePrivateType *) GifFile->Private;

    if (!IS_READABLE(Private)) {
	/* This file was NOT open for reading: */
	_GifError = D_GIF_ERR_NOT_READABLE;
	return GIF_ERROR;
    }

    if (!LineLen) LineLen = GifFile->Image.Width;

#if defined(__MSDOS__) || defined(__GNUC__)
    if ((Private->PixelCount -= LineLen) > 0xffff0000UL) {
#else
    if ((Private->PixelCount -= LineLen) > 0xffff0000) {
#endif /* __MSDOS__ */
	_GifError = D_GIF_ERR_DATA_TOO_BIG;
	return GIF_ERROR;
    }

    if (DGifDecompressLine(GifFile, Line, LineLen) == GIF_OK) {
	if (Private->PixelCount == 0) {
	    /* We probably would not be called any more, so lets clean 	     */
	    /* everything before we return: need to flush out all rest of    */
	    /* image until empty block (size 0) detected. We use GetCodeNext.*/
	    do if (DGifGetCodeNext(GifFile, &Dummy) == GIF_ERROR)
		return GIF_ERROR;
	    while (Dummy != NULL);
	}
	return GIF_OK;
    }
    else
	return GIF_ERROR;
}

/******************************************************************************
*   Get an extension block (see GIF manual) from gif file. This routine only  *
* returns the first data block, and DGifGetExtensionNext shouldbe called      *
* after this one until NULL extension is returned.			      *
*   The Extension should NOT be freed by the user (not dynamically allocated).*
*   Note it is assumed the Extension desc. header ('!') has been read.	      *
******************************************************************************/
int DGifGetExtension(GifFileType *GifFile, int *ExtCode,
						    GifByteType **Extension)
{
    GifByteType Buf;
    GifFilePrivateType *Private = (GifFilePrivateType *) GifFile->Private;

    if (!IS_READABLE(Private)) {
	/* This file was NOT open for reading: */
	_GifError = D_GIF_ERR_NOT_READABLE;
	return GIF_ERROR;
    }

    if (READ(GifFile,&Buf, 1) != 1) {
	_GifError = D_GIF_ERR_READ_FAILED;
	return GIF_ERROR;
    }
    *ExtCode = Buf;

    return DGifGetExtensionNext(GifFile, Extension);
}

/******************************************************************************
*   Get a following extension block (see GIF manual) from gif file. This      *
* routine should be called until NULL Extension is returned.		      *
*   The Extension should NOT be freed by the user (not dynamically allocated).*
******************************************************************************/
int DGifGetExtensionNext(GifFileType *GifFile, GifByteType **Extension)
{
    GifByteType Buf;
    GifFilePrivateType *Private = (GifFilePrivateType *) GifFile->Private;

    if (READ(GifFile,&Buf, 1) != 1) {
	_GifError = D_GIF_ERR_READ_FAILED;
	return GIF_ERROR;
    }
    if (Buf > 0) {
	*Extension = Private->Buf;           /* Use private unused buffer. */
	(*Extension)[0] = Buf;  /* Pascal strings notation (pos. 0 is len.). */
	if (READ(GifFile,&((*Extension)[1]), Buf) != Buf) {
	    _GifError = D_GIF_ERR_READ_FAILED;
	    return GIF_ERROR;
	}
    }
    else
	*Extension = NULL;

    return GIF_OK;
}

/******************************************************************************
*   This routine should be called last, to close the GIF file.		      *
******************************************************************************/
int DGifCloseFile(GifFileType *GifFile)
{
    GifFilePrivateType *Private;
    FILE *File;
	int ret = GIF_OK;

    if (GifFile == NULL) return GIF_ERROR;

    Private = (GifFilePrivateType *) GifFile->Private;

    if (!IS_READABLE(Private)) {
		/* This file was NOT open for reading: */
		_GifError = D_GIF_ERR_NOT_READABLE;
		ret = GIF_ERROR; /* we have to free everything regardless */
    }

    File = Private->File;

    if (GifFile->Image.ColorMap)
    {
		FreeMapObject(GifFile->Image.ColorMap);
        GifFile->Image.ColorMap = NULL;
    }

    if (GifFile->SColorMap)
    {
		FreeMapObject(GifFile->SColorMap);
		GifFile->SColorMap = NULL;
    }

    if (Private)
    {
		free((char *) Private);
		GifFile->Private = NULL;
    }

    if (GifFile->SavedImages)
    {
		FreeSavedImages(GifFile);
		GifFile->SavedImages = NULL;
    }

    free(GifFile);

    if ( File && (fclose(File) != 0)) 
	{
	  	_GifError = D_GIF_ERR_CLOSE_FAILED;
		ret = GIF_ERROR;
    }
    return ret;
}

/******************************************************************************
*   Get 2 bytes (word) from the given file:				      *
******************************************************************************/
static int DGifGetWord(GifFileType *GifFile, int *Word)
{
    unsigned char c[2];

    if (READ(GifFile,c, 2) != 2) {
	_GifError = D_GIF_ERR_READ_FAILED;
	return GIF_ERROR;
    }

    *Word = (((unsigned int) c[1]) << 8) + c[0];
    return GIF_OK;
}

/******************************************************************************
*   Get the image code in compressed form.  This routine can be called if the  *
* information needed to be piped out as is. Obviously this is much faster     *
* than decoding and encoding again. This routine should be followed by calls  *
* to DGifGetCodeNext, until NULL block is returned.			      *
*   The block should NOT be freed by the user (not dynamically allocated).    *
******************************************************************************/
int DGifGetCode(GifFileType *GifFile, int *CodeSize, GifByteType **CodeBlock)
{
    GifFilePrivateType *Private = (GifFilePrivateType *) GifFile->Private;

    if (!IS_READABLE(Private)) {
	/* This file was NOT open for reading: */
	_GifError = D_GIF_ERR_NOT_READABLE;
	return GIF_ERROR;
    }

    *CodeSize = Private->BitsPerPixel;

    return DGifGetCodeNext(GifFile, CodeBlock);
}

/******************************************************************************
*   Continue to get the image code in compressed form. This routine should be *
* called until NULL block is returned.					      *
*   The block should NOT be freed by the user (not dynamically allocated).    *
******************************************************************************/
int DGifGetCodeNext(GifFileType *GifFile, GifByteType **CodeBlock)
{
    GifByteType Buf;
    GifFilePrivateType *Private = (GifFilePrivateType *) GifFile->Private;
    if (READ(GifFile,&Buf, 1) != 1) {
	_GifError = D_GIF_ERR_READ_FAILED;
	return GIF_ERROR;
    }
    if (Buf > 0) {
		int ret ; 
	*CodeBlock = Private->Buf;	       /* Use private unused buffer. */
	(*CodeBlock)[0] = Buf;  /* Pascal strings notation (pos. 0 is len.). */
	ret = READ(GifFile,&((*CodeBlock)[1]), Buf);
	if ( ret != Buf) 
	{
		if( Buf == 59 ) 
		{
			fseek( GifFile->UserData, -1, SEEK_END );	
			*CodeBlock = NULL ; 
			return GIF_OK;
		}	 
	    _GifError = D_GIF_ERR_READ_FAILED;
	    return GIF_ERROR;
	}
    }
    else {
	*CodeBlock = NULL;
	Private->Buf[0] = 0;		   /* Make sure the buffer is empty! */
	Private->PixelCount = 0;   /* And local info. indicate image read. */
    }

    return GIF_OK;
}

/******************************************************************************
*   Setup the LZ decompression for this image:				      *
******************************************************************************/
static int DGifSetupDecompress(GifFileType *GifFile)
{
    int i, BitsPerPixel;
    GifByteType CodeSize;
    unsigned int *Prefix;
    GifFilePrivateType *Private = (GifFilePrivateType *) GifFile->Private;

    READ(GifFile,&CodeSize, 1);    /* Read Code size from file. */
    BitsPerPixel = CodeSize;

    Private->Buf[0] = 0;			      /* Input Buffer empty. */
    Private->BitsPerPixel = BitsPerPixel;
    Private->ClearCode = (1 << BitsPerPixel);
    Private->EOFCode = Private->ClearCode + 1;
    Private->RunningCode = Private->EOFCode + 1;
    Private->RunningBits = BitsPerPixel + 1;	 /* Number of bits per code. */
    Private->MaxCode1 = 1 << Private->RunningBits;     /* Max. code + 1. */
    Private->StackPtr = 0;		    /* No pixels on the pixel stack. */
    Private->LastCode = NO_SUCH_CODE;
    Private->CrntShiftState = 0;	/* No information in CrntShiftDWord. */
    Private->CrntShiftDWord = 0;

    Prefix = Private->Prefix;
    for (i = 0; i <= LZ_MAX_CODE; i++) Prefix[i] = NO_SUCH_CODE;

    return GIF_OK;
}

/******************************************************************************
*   The LZ decompression routine:					      *
*   This version decompress the given gif file into Line of length LineLen.   *
*   This routine can be called few times (one per scan line, for example), in *
* order the complete the whole image.					      *
******************************************************************************/
static int DGifDecompressLine(GifFileType *GifFile, GifPixelType *Line,
								int LineLen)
{
    int i = 0, j, CrntCode, EOFCode, ClearCode, CrntPrefix, LastCode, StackPtr;
    GifByteType *Stack, *Suffix;
    unsigned int *Prefix;
    GifFilePrivateType *Private = (GifFilePrivateType *) GifFile->Private;

    StackPtr = Private->StackPtr;
    Prefix = Private->Prefix;
    Suffix = Private->Suffix;
    Stack = Private->Stack;
    EOFCode = Private->EOFCode;
    ClearCode = Private->ClearCode;
    LastCode = Private->LastCode;

    if (StackPtr != 0) {
	/* Let pop the stack off before continueing to read the gif file: */
	while (StackPtr != 0 && i < LineLen) Line[i++] = Stack[--StackPtr];
    }

    while (i < LineLen) {			    /* Decode LineLen items. */
	if (DGifDecompressInput(GifFile, &CrntCode) == GIF_ERROR)
    	    return GIF_ERROR;

	if (CrntCode == EOFCode) {
	    /* Note however that usually we will not be here as we will stop */
	    /* decoding as soon as we got all the pixel, or EOF code will    */
	    /* not be read at all, and DGifGetLine/Pixel clean everything.   */
	    if (i != LineLen - 1 || Private->PixelCount != 0) {
		_GifError = D_GIF_ERR_EOF_TOO_SOON;
		return GIF_ERROR;
	    }
	    i++;
	}
	else if (CrntCode == ClearCode) {
	    /* We need to start over again: */
	    for (j = 0; j <= LZ_MAX_CODE; j++) Prefix[j] = NO_SUCH_CODE;
	    Private->RunningCode = Private->EOFCode + 1;
	    Private->RunningBits = Private->BitsPerPixel + 1;
	    Private->MaxCode1 = 1 << Private->RunningBits;
	    LastCode = Private->LastCode = NO_SUCH_CODE;
	}
	else {
	    /* Its regular code - if in pixel range simply add it to output  */
	    /* stream, otherwise trace to codes linked list until the prefix */
	    /* is in pixel range:					     */
	    if (CrntCode < ClearCode) {
		/* This is simple - its pixel scalar, so add it to output:   */
		Line[i++] = CrntCode;
	    }
	    else {
		/* Its a code to needed to be traced: trace the linked list  */
		/* until the prefix is a pixel, while pushing the suffix     */
		/* pixels on our stack. If we done, pop the stack in reverse */
		/* (thats what stack is good for!) order to output.	     */
		if (Prefix[CrntCode] == NO_SUCH_CODE) {
		    /* Only allowed if CrntCode is exactly the running code: */
		    /* In that case CrntCode = XXXCode, CrntCode or the	     */
		    /* prefix code is last code and the suffix char is	     */
		    /* exactly the prefix of last code!			     */
		    if (CrntCode == Private->RunningCode - 2) {
			CrntPrefix = LastCode;
			Suffix[Private->RunningCode - 2] =
			Stack[StackPtr++] = DGifGetPrefixChar(Prefix,
							LastCode, ClearCode);
		    }
		    else {
			/*fprintf( stderr, "error at line %d, CrntCode = %d,  Private->RunningCode - 2 = %d\n", __LINE__, CrntCode, Private->RunningCode - 2 ); */
			_GifError = D_GIF_ERR_IMAGE_DEFECT;
			return GIF_ERROR;
		    }
		}
		else
		    CrntPrefix = CrntCode;

		/* Now (if image is O.K.) we should not get an NO_SUCH_CODE  */
		/* During the trace. As we might loop forever, in case of    */
		/* defective image, we count the number of loops we trace    */
		/* and stop if we got LZ_MAX_CODE. obviously we can not      */
		/* loop more than that.					     */
		j = 0;
		while (j++ <= LZ_MAX_CODE &&
		       CrntPrefix > ClearCode &&
		       CrntPrefix <= LZ_MAX_CODE) {
		    Stack[StackPtr++] = Suffix[CrntPrefix];
		    CrntPrefix = Prefix[CrntPrefix];
		}
		if (j >= LZ_MAX_CODE || CrntPrefix > LZ_MAX_CODE) {
			/* fprintf( stderr,  "error at line %d, j = %d, CrntPrefix = %d\n", __LINE__, j, CrntPrefix ); */
		    _GifError = D_GIF_ERR_IMAGE_DEFECT;
		    return GIF_ERROR;
		}
		/* Push the last character on stack: */
		Stack[StackPtr++] = CrntPrefix;

		/* Now lets pop all the stack into output: */
		while (StackPtr != 0 && i < LineLen)
		    Line[i++] = Stack[--StackPtr];
	    }
	    if (LastCode != NO_SUCH_CODE) {
		Prefix[Private->RunningCode - 2] = LastCode;

		if (CrntCode == Private->RunningCode - 2) {
		    /* Only allowed if CrntCode is exactly the running code: */
		    /* In that case CrntCode = XXXCode, CrntCode or the	     */
		    /* prefix code is last code and the suffix char is	     */
		    /* exactly the prefix of last code!			     */
		    Suffix[Private->RunningCode - 2] =
			DGifGetPrefixChar(Prefix, LastCode, ClearCode);
		}
		else {
		    Suffix[Private->RunningCode - 2] =
			DGifGetPrefixChar(Prefix, CrntCode, ClearCode);
		}
	    }
	    LastCode = CrntCode;
	}
    }

    Private->LastCode = LastCode;
    Private->StackPtr = StackPtr;

    return GIF_OK;
}

/******************************************************************************
* Routine to trace the Prefixes linked list until we get a prefix which is    *
* not code, but a pixel value (less than ClearCode). Returns that pixel value.*
* If image is defective, we might loop here forever, so we limit the loops to *
* the maximum possible if image O.k. - LZ_MAX_CODE times.		      *
******************************************************************************/
static int DGifGetPrefixChar(unsigned int *Prefix, int Code, int ClearCode)
{
    int i = 0;

    while (Code > ClearCode && i++ <= LZ_MAX_CODE) Code = Prefix[Code];
    return Code;
}

/******************************************************************************
*   The LZ decompression input routine:					      *
*   This routine is responsable for the decompression of the bit stream from  *
* 8 bits (bytes) packets, into the real codes.				      *
*   Returns GIF_OK if read successfully.					      *
******************************************************************************/
static int DGifDecompressInput(GifFileType *GifFile, int *Code)
{
    GifFilePrivateType *Private = (GifFilePrivateType *)GifFile->Private;

    GifByteType NextByte;
    static unsigned int CodeMasks[] = {
	0x0000, 0x0001, 0x0003, 0x0007,
	0x000f, 0x001f, 0x003f, 0x007f,
	0x00ff, 0x01ff, 0x03ff, 0x07ff,
	0x0fff
    };

    while (Private->CrntShiftState < Private->RunningBits) {
	/* Needs to get more bytes from input stream for next code: */
	if (DGifBufferedInput(GifFile, Private->Buf, &NextByte)
	    == GIF_ERROR) {
	    return GIF_ERROR;
	}
	Private->CrntShiftDWord |=
		((unsigned long) NextByte) << Private->CrntShiftState;
	Private->CrntShiftState += 8;
    }
    *Code = Private->CrntShiftDWord & CodeMasks[Private->RunningBits];

    Private->CrntShiftDWord >>= Private->RunningBits;
    Private->CrntShiftState -= Private->RunningBits;

    /* If code cannt fit into RunningBits bits, must raise its size. Note */
    /* however that codes above 4095 are used for special signaling.      */
    if (++Private->RunningCode > Private->MaxCode1 &&
	Private->RunningBits < LZ_BITS) {
	Private->MaxCode1 <<= 1;
	Private->RunningBits++;
    }
    return GIF_OK;
}

/******************************************************************************
*   This routines read one gif data block at a time and buffers it internally *
* so that the decompression routine could access it.			      *
*   The routine returns the next byte from its internal buffer (or read next  *
* block in if buffer empty) and returns GIF_OK if successful.		      *
******************************************************************************/
static int DGifBufferedInput(GifFileType *GifFile, GifByteType *Buf,
						      GifByteType *NextByte)
{
    if (Buf[0] == 0) {
	/* Needs to read the next buffer - this one is empty: */
	if (READ(GifFile, Buf, 1) != 1)
	{
	    _GifError = D_GIF_ERR_READ_FAILED;
	    return GIF_ERROR;
	}
	if (READ(GifFile,&Buf[1], Buf[0]) != Buf[0])
	{
	    _GifError = D_GIF_ERR_READ_FAILED;
	    return GIF_ERROR;
	}
	*NextByte = Buf[1];
	Buf[1] = 2;	   /* We use now the second place as last char read! */
	Buf[0]--;
    }
    else {
	*NextByte = Buf[Buf[1]++];
	Buf[0]--;
    }

    return GIF_OK;
}


