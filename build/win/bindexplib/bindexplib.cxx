/*
 *----------------------------------------------------------------------
 * Program:  dumpexts.exe
 * Author:   Gordon Chaffee
 *
 * History:  The real functionality of this file was written by
 *           Matt Pietrek in 1993 in his pedump utility.  I've
 *           modified it to dump the externals in a bunch of object
 *           files to create a .def file.
 *
 * Notes:    Visual C++ puts an underscore before each exported symbol.
 *           This file removes them.  I don't know if this is a problem
 *           this other compilers.  If VISUAL_CPLUSCPLUS is defined,
 *           the underscore is removed.  If not, it isn't.  To get a
 *           full dump of an object file, use the -f option.  This can
 *           help determine the something that may be different with a
 *           compiler other than Visual C++.
 *   ======================================
 * Corrections (Valery Fine 23/02/98):
 *
 *           The "(vector) deleting destructor" MUST not be exported
 *           To recognize it the following test are introduced:
 *  "@@UAEPAXI@Z"  scalar deleting dtor
 *  "@@QAEPAXI@Z"  vector deleting dtor
 *  "AEPAXI@Z"     vector deleting dtor with thunk adjustor
 *   ======================================
 * Corrections (Valery Fine 12/02/97):
 *
 *    It created a wrong EXPORTS for the global pointers and constants.
 *    The Section Header has been involved to discover the missing information
 *    Now the pointers are correctly supplied  supplied with "DATA" descriptor
 *        the constants  with no extra descriptor.
 *
 * Corrections (Valery Fine 16/09/96):
 *
 *     It didn't work for C++ code with global variables and class definitons
 *     The DumpExternalObject function has been introduced to generate .DEF file
 *
 * Author:   Valery Fine 16/09/96  (E-mail: fine@vxcern.cern.ch)
 *----------------------------------------------------------------------
 */

static char sccsid[] = "@(#) winDumpExts.c 1.2 95/10/03 15:27:34";

static int fort = 0;
static const int kMaxSymbolSize = 2048;

#include <windows.h>
#include <stdio.h>
#include <string.h>

/*
 *----------------------------------------------------------------------
 * GetArgcArgv --
 *
 *      Break up a line into argc argv
 *----------------------------------------------------------------------
 */
int
GetArgcArgv(char *s, char **argv)
{
    int quote = 0;
    int argc = 0;
    char *bp;

    bp = s;
    while (1) {
        while (isspace(*bp)) {
            bp++;
        }
        if (*bp == '\n' || *bp == '\0') {
            *bp = '\0';
            return argc;
        }
        if (*bp == '\"') {
            quote = 1;
            bp++;
        }
        argv[argc++] = bp;

        while (*bp != '\0') {
            if (quote) {
                if (*bp == '\"') {
                    quote = 0;
                    *bp = '\0';
                    bp++;
                    break;
                }
                bp++;
                continue;
            }
            if (isspace(*bp)) {
                *bp = '\0';
                bp++;
                break;
            }
            bp++;
        }
    }
}

/*
 *  The names of the first group of possible symbol table storage classes
 */
char * SzStorageClass1[] = {
    "NULL","AUTOMATIC","EXTERNAL","STATIC","REGISTER","EXTERNAL_DEF","LABEL",
    "UNDEFINED_LABEL","MEMBER_OF_STRUCT","ARGUMENT","STRUCT_TAG",
    "MEMBER_OF_UNION","UNION_TAG","TYPE_DEFINITION","UNDEFINED_STATIC",
    "ENUM_TAG","MEMBER_OF_ENUM","REGISTER_PARAM","BIT_FIELD"
};

/*
 * The names of the second group of possible symbol table storage classes
 */
char * SzStorageClass2[] = {
    "BLOCK","FUNCTION","END_OF_STRUCT","FILE","SECTION","WEAK_EXTERNAL"
};

/*
 *----------------------------------------------------------------------
 * GetSZStorageClass --
 *
 *      Given a symbol storage class value, return a descriptive
 *      ASCII string
 *----------------------------------------------------------------------
 */
PSTR
GetSZStorageClass(BYTE storageClass)
{
        if ( storageClass <= IMAGE_SYM_CLASS_BIT_FIELD )
                return SzStorageClass1[storageClass];
        else if ( (storageClass >= IMAGE_SYM_CLASS_BLOCK)
                      && (storageClass <= IMAGE_SYM_CLASS_WEAK_EXTERNAL) )
                return SzStorageClass2[storageClass-IMAGE_SYM_CLASS_BLOCK];
        else
                return "???";
}

/*
 *----------------------------------------------------------------------
 * GetSectionName --
 *
 *      Used by DumpSymbolTable, it gives meaningful names to
 *      the non-normal section number.
 *
 * Results:
 *      A name is returned in buffer
 *----------------------------------------------------------------------
 */
void
GetSectionName(PIMAGE_SYMBOL pSymbolTable, PSTR buffer, unsigned cbBuffer)
{
    char tempbuffer[10];
    DWORD section;

    section = pSymbolTable->SectionNumber;

    switch ( (SHORT)section )
    {
      case IMAGE_SYM_UNDEFINED: if (pSymbolTable->Value) strcpy(tempbuffer, "COMM"); else strcpy(tempbuffer, "UNDEF"); break;
      case IMAGE_SYM_ABSOLUTE:  strcpy(tempbuffer, "ABS  "); break;
      case IMAGE_SYM_DEBUG:       strcpy(tempbuffer, "DEBUG"); break;
      default: sprintf(tempbuffer, "%-5X", section);
    }

    strncpy(buffer, tempbuffer, cbBuffer-1);
}

/*
 *----------------------------------------------------------------------
 * GetSectionCharacteristics --
 *
 *      Converts the Characteristics field of IMAGE_SECTION_HEADER
 *      to print.
 *
 *  Results:
 *       A definiton of the section symbol type
 *----------------------------------------------------------------------
 */
void
GetSectionCharacteristics(PIMAGE_SECTION_HEADER pSectionHeaders, int nSectNum, PSTR buffer)
{
  DWORD SectChar;
  char TempBuf[100];
  memset(buffer,'\0',30);
  if (nSectNum > 0) {
    SectChar = pSectionHeaders[nSectNum-1].Characteristics;

    sprintf(buffer," %x", SectChar);
    if       (SectChar & IMAGE_SCN_CNT_CODE)                strcat(buffer," Code");
    else if  (SectChar & IMAGE_SCN_CNT_INITIALIZED_DATA)    strcat(buffer," Init. data");
    else if  (SectChar & IMAGE_SCN_CNT_UNINITIALIZED_DATA ) strcat(buffer," UnInit data");
    else                                                    strcat(buffer," Unknow type");

    if   (SectChar & IMAGE_SCN_MEM_READ)  {
              strcat(buffer," Read");
         if (SectChar & IMAGE_SCN_MEM_WRITE)
              strcat(buffer," and Write");
         else strcat(buffer," only");
    }
    else if (SectChar & IMAGE_SCN_MEM_WRITE)
              strcat(buffer," Write only");

  }
}

/*
 *----------------------------------------------------------------------
 * DumpSymbolTable --
 *
 *      Dumps a COFF symbol table from an EXE or OBJ.  We only use
 *      it to dump tables from OBJs.
 *----------------------------------------------------------------------
 */
void
DumpSymbolTable(PIMAGE_SYMBOL pSymbolTable, PIMAGE_SECTION_HEADER pSectionHeaders, FILE *fout, unsigned cSymbols)
{
    unsigned i;
    PSTR stringTable;
    char sectionName[10];
    char sectionCharacter[40];
    int iSectNum;

    fprintf(fout, "Symbol Table - %X entries  (* = auxillary symbol)\n",
            cSymbols);

    fprintf(fout,
     "Indx Name                 Value    Section    cAux  Type    Storage  Character\n"
     "---- -------------------- -------- ---------- ----- ------- -------- ---------\n");

    /*
     * The string table apparently starts right after the symbol table
     */
    stringTable = (PSTR)&pSymbolTable[cSymbols];

    for ( i=0; i < cSymbols; i++ ) {
        fprintf(fout, "%04X ", i);
        if ( pSymbolTable->N.Name.Short != 0 )
            fprintf(fout, "%-20.8s", pSymbolTable->N.ShortName);
        else
            fprintf(fout, "%-20s", stringTable + pSymbolTable->N.Name.Long);

        fprintf(fout, " %08X", pSymbolTable->Value);

        iSectNum = pSymbolTable->SectionNumber;
        GetSectionName(pSymbolTable, sectionName,
                       sizeof(sectionName));
        fprintf(fout, " sect:%s aux:%X type:%02X st:%s",
               sectionName,
               pSymbolTable->NumberOfAuxSymbols,
               pSymbolTable->Type,
               GetSZStorageClass(pSymbolTable->StorageClass) );

        GetSectionCharacteristics(pSectionHeaders,iSectNum,sectionCharacter);
        fprintf(fout," hc: %s \n",sectionCharacter);
#if 0
        if ( pSymbolTable->NumberOfAuxSymbols )
            DumpAuxSymbols(pSymbolTable);
#endif

        /*
         * Take into account any aux symbols
         */
        i += pSymbolTable->NumberOfAuxSymbols;
        pSymbolTable += pSymbolTable->NumberOfAuxSymbols;
        pSymbolTable++;
    }
}

/*
 *----------------------------------------------------------------------
 * DumpExternals --
 *
 *      Dumps a COFF symbol table from an EXE or OBJ.  We only use
 *      it to dump tables from OBJs.
 *----------------------------------------------------------------------
 */
void
DumpExternals(PIMAGE_SYMBOL pSymbolTable, FILE *fout, unsigned cSymbols)
{
    unsigned i;
    PSTR stringTable;
    char *s, *f;
    char symbol[kMaxSymbolSize];

    /*
     * The string table apparently starts right after the symbol table
     */
    stringTable = (PSTR)&pSymbolTable[cSymbols];

    for ( i=0; i < cSymbols; i++ ) {
        if (pSymbolTable->SectionNumber > 0 && pSymbolTable->Type == 0x20) {
            if (pSymbolTable->StorageClass == IMAGE_SYM_CLASS_EXTERNAL) {
                if (pSymbolTable->N.Name.Short != 0) {
                    strncpy(symbol, (const char *)(pSymbolTable->N.ShortName), 8);
                    symbol[8] = 0;
                } else {
                    s = stringTable + pSymbolTable->N.Name.Long;
                    strcpy(symbol, s);
                }
                s = symbol;
                f = strchr(s, '@');
                if (f) {
                    *f = 0;
                }
#ifndef VISUAL_CPLUSPLUS
                fprintf(fout, "\t%s\n", symbol);
#else
                fprintf(fout, "\t%s\n", &symbol[1]);
#endif
            }
        }

        /*
         * Take into account any aux symbols
         */
        i += pSymbolTable->NumberOfAuxSymbols;
        pSymbolTable += pSymbolTable->NumberOfAuxSymbols;
        pSymbolTable++;
    }
}

/*
 *----------------------------------------------------------------------
 * DumpExternalsObjects --
 *
 *      Dumps a COFF symbol table from an EXE or OBJ.  We only use
 *      it to dump tables from OBJs.
 *----------------------------------------------------------------------
 */
void
DumpExternalsObjects(PIMAGE_SYMBOL pSymbolTable, PIMAGE_SECTION_HEADER pSectionHeaders, FILE *fout, unsigned cSymbols)
{
    unsigned i;
    PSTR stringTable;
    char *s, *f;
    char symbol[kMaxSymbolSize];
    DWORD SectChar;
    static int fImportFlag = -1;  /*  The status is nor defined yet */

    /*
     * The string table apparently starts right after the symbol table
     */
    stringTable = (PSTR)&pSymbolTable[cSymbols];

    for ( i=0; i < cSymbols; i++ ) {
        if (pSymbolTable->SectionNumber > 0 && ( pSymbolTable->Type == 0x20 || pSymbolTable->Type == 0x0)) {
          if (pSymbolTable->StorageClass == IMAGE_SYM_CLASS_EXTERNAL) {
/*
 *    The name of the Function entry points
 */
            if (pSymbolTable->N.Name.Short != 0) {
                strncpy(symbol, (const char *)pSymbolTable->N.ShortName, 8);
                symbol[8] = 0;
            } else {
                s = stringTable + pSymbolTable->N.Name.Long;
                strcpy(symbol, s);
            }

            s = symbol;
            while (isspace(*s))  s++;
#ifdef VISUAL_CPLUSPLUS
            if (*s == '_') s++;
            if (fort) {
               f = strchr(s, '@');
               if (f)
                      *f = 0;
            }
#endif
            if (fImportFlag) {
                fImportFlag = 0;
                fprintf(fout,"EXPORTS \n");
            }
/*
  Check whether it is "Scalar deleting destructor" and
                      "Vector deleting destructor"
 */
/*
            if (!strstr(s,"@@UAEPAXI@Z") && !strstr(s,"@@QAEPAXI@Z") &&
                !strstr(s,"@AEPAXI@Z")   && !strstr(s,"AEPAXI@Z")    &&
                !strstr(s,"real@"))
*/
              const char *scalarPrefix = "??_G";
              const char *vectorPrefix = "??_E";
              if (strncmp(s,scalarPrefix,strlen(scalarPrefix))!=0 && 
                  strncmp(s,vectorPrefix,strlen(vectorPrefix))!=0 &&
                 !strstr(s,"real@"))
            {
              SectChar = pSectionHeaders[pSymbolTable->SectionNumber-1].Characteristics;
              if (!pSymbolTable->Type  && (SectChar & IMAGE_SCN_MEM_WRITE)) {
                 // Read only (i.e. constants) must be excluded
                 fprintf(fout, "\t%s", s);
                 fprintf(fout, " \t DATA");
                 fprintf(fout, "\n");
              } else {
                 if ( pSymbolTable->Type  || !(SectChar & IMAGE_SCN_MEM_READ)) {
                    fprintf(fout, "\t%s", s);
                    fprintf(fout, "\n");
                 } else {
//                    printf(" strange symbol: %s \n",s);
                 }
              }
            }
          }
        }
        else if (pSymbolTable->SectionNumber == IMAGE_SYM_UNDEFINED && !pSymbolTable->Type && 0){
/*
 *    The IMPORT global variable entry points
 */
              if (pSymbolTable->StorageClass == IMAGE_SYM_CLASS_EXTERNAL) {
                      s = stringTable + pSymbolTable->N.Name.Long;
                      strcpy(symbol, s);
                      s = symbol;
              while (isspace(*s))  s++;
                      if (*s == '_') s++;
              if (!fImportFlag) {
                 fImportFlag = 1;
                 fprintf(fout,"IMPORTS \n");
              }
                      fprintf(fout, "\t%s DATA \n", &symbol[1]);
          }
        }

        /*
         * Take into account any aux symbols
         */
        i += pSymbolTable->NumberOfAuxSymbols;
        pSymbolTable += pSymbolTable->NumberOfAuxSymbols;
        pSymbolTable++;
    }
}

/*
 *----------------------------------------------------------------------
 * DumpObjFile --
 *
 *      Dump an object file--either a full listing or just the exported
 *      symbols.
 *----------------------------------------------------------------------
 */
void
DumpObjFile(PIMAGE_FILE_HEADER pImageFileHeader, FILE *fout, int full)
{
    PIMAGE_SYMBOL PCOFFSymbolTable;
    PIMAGE_SECTION_HEADER PCOFFSectionHeaders;
    DWORD COFFSymbolCount;

    PCOFFSymbolTable = (PIMAGE_SYMBOL)
        ((DWORD)pImageFileHeader + pImageFileHeader->PointerToSymbolTable);
    COFFSymbolCount = pImageFileHeader->NumberOfSymbols;

    PCOFFSectionHeaders = (PIMAGE_SECTION_HEADER)
                          ((DWORD)pImageFileHeader          +
                                   IMAGE_SIZEOF_FILE_HEADER +
                                   pImageFileHeader->SizeOfOptionalHeader);


    if (full) {
        DumpSymbolTable(PCOFFSymbolTable, PCOFFSectionHeaders, fout, COFFSymbolCount);
    } else {
/*      DumpExternals(PCOFFSymbolTable, fout, COFFSymbolCount); */
        DumpExternalsObjects(PCOFFSymbolTable, PCOFFSectionHeaders, fout, COFFSymbolCount);
    }
}

/*
 *----------------------------------------------------------------------
 * DumpFile --
 *
 *      Open up a file, memory map it, and call the appropriate
 *      dumping routine
 *----------------------------------------------------------------------
 */
void
DumpFile(LPSTR filename, FILE *fout, int full)
{
    HANDLE hFile;
    HANDLE hFileMapping;
    LPVOID lpFileBase;
    PIMAGE_DOS_HEADER dosHeader;

    hFile = CreateFile(filename, GENERIC_READ, FILE_SHARE_READ, NULL,
                       OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);

    if (hFile == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "Couldn't open file with CreateFile()\n");
        return;
    }

    hFileMapping = CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (hFileMapping == 0) {
        CloseHandle(hFile);
        fprintf(stderr, "Couldn't open file mapping with CreateFileMapping()\n");
        return;
    }

    lpFileBase = MapViewOfFile(hFileMapping, FILE_MAP_READ, 0, 0, 0);
    if (lpFileBase == 0) {
        CloseHandle(hFileMapping);
        CloseHandle(hFile);
        fprintf(stderr, "Couldn't map view of file with MapViewOfFile()\n");
        return;
    }

    dosHeader = (PIMAGE_DOS_HEADER)lpFileBase;
    if (dosHeader->e_magic == IMAGE_DOS_SIGNATURE) {
#if 0
        DumpExeFile( dosHeader );
#else
        fprintf(stderr, "File is an executable.  I don't dump those.\n");
        return;
#endif
    }
    /* Does it look like a i386 COFF OBJ file??? */
    else if ((dosHeader->e_magic == 0x014C) && (dosHeader->e_sp == 0)) {
        /*
         * The two tests above aren't what they look like.  They're
         * really checking for IMAGE_FILE_HEADER.Machine == i386 (0x14C)
         * and IMAGE_FILE_HEADER.SizeOfOptionalHeader == 0;
         */
        DumpObjFile((PIMAGE_FILE_HEADER) lpFileBase, fout, full);
    } else {
        printf("unrecognized file format\n");
    }
    UnmapViewOfFile(lpFileBase);
    CloseHandle(hFileMapping);
    CloseHandle(hFile);
}

void
main(int argc, char **argv)
{
    char *fargv[1000];
    char cmdline[10000];
    int i, arg;
    FILE *fargs, *fout;
    int pos;
    int full = 0;
    char *dllname = "";
    char *outfile = NULL;

    if (argc < 3) {
      Usage:
        fprintf(stderr, "Usage: %s ?-o outfile? ?-f(ull)? <dllname> <object filenames> ..\n", argv[0]);
        exit(1);
    }

    fargs = NULL;
    arg = 1;
    while (argv[arg][0] == '-') {
        if (strcmp(argv[arg], "--") == 0) {
            arg++;
            break;
        } else if (strcmp(argv[arg], "-f") == 0) {
            full = 1;
        } else if (strcmp(argv[arg], "-x") == 0) {
            fort = 1;
        } else if (strcmp(argv[arg], "-o") == 0) {
            arg++;
            if (arg == argc) {
                goto Usage;
            }
            outfile = argv[arg];
        }
        arg++;
    }
    if (arg == argc) {
        goto Usage;
    }

    if (outfile) {
        fout = fopen(outfile, "w+");
        if (fout == NULL) {
            fprintf(stderr, "Unable to open \'%s\' for writing:\n",
                    argv[arg]);
            perror("");
            exit(1);
        }
    } else {
        fout = stdout;
    }

    if (! full) {
        dllname = argv[arg];
        arg++;
        if (arg == argc) {
            goto Usage;
        }
        fprintf(fout, "LIBRARY    %s\n", dllname);
#ifndef _X86_
        fprintf(fout, "CODE PRELOAD MOVEABLE DISCARDABLE\n");
        fprintf(fout, "DATA PRELOAD MOVEABLE MULTIPLE\n\n");
#endif
    }

    for (; arg < argc; arg++) {
    WIN32_FIND_DATA FindFileData;
    HANDLE SearchFile;
        if (argv[arg][0] == '@') {
            fargs = fopen(&argv[arg][1], "r");
            if (fargs == NULL) {
                fprintf(stderr, "Unable to open \'%s\' for reading:\n",
                        argv[arg]);
                perror("");
                exit(1);
            }
            pos = 0;
            for (i = 0; i < arg; i++) {
                strcpy(&cmdline[pos], argv[i]);
                pos += strlen(&cmdline[pos]) + 1;
                fargv[i] = argv[i];
            }
            fgets(&cmdline[pos], sizeof(cmdline), fargs);
            fprintf(stderr, "%s\n", &cmdline[pos]);
            fclose(fargs);
            i += GetArgcArgv(&cmdline[pos], &fargv[i]);
            argc = i;
            argv = fargv;
        }
/*
 *  Argument can contain the wildcard names
 */
       SearchFile = FindFirstFile(argv[arg],&FindFileData);
           if (SearchFile == INVALID_HANDLE_VALUE){
                  fprintf(stderr, "Unable to find \'%s\' for reading:\n",
           argv[arg]);
              exit(1);
           }
           else  {
/*
 *  Since WIN32_FIND_DATA has no path information one has to extract it himself.
 */
             TCHAR *filename = argv[arg];
             TCHAR path[2048];
             int i = strlen(filename);
             i--;
             while( filename[i] != '\\' && filename[i] != '/'  && i >=0) i--;
             do
             {
               if (i >= 0) strncpy( path, filename, i+1); /* Generate the 'path' info */
               path[i+1] = '\0';
               DumpFile(strcat(path, FindFileData.cFileName), fout, full);
             } while (FindNextFile(SearchFile,&FindFileData));


             FindClose(SearchFile);
           }
    }
    exit(0);
}
