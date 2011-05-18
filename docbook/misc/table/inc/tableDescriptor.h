/* @(#)root/table:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
/* tableDescriptor.h */
#ifndef TABLEDESCRIPTOR_H
#define TABLEDESCRIPTOR_H

#ifdef NORESTRICTIONS
# define TABLEDESCRIPTOR_SPEC   \
 "struct tableDescriptor {      \
    char         *fColumnName; \
    unsigned int *fIndexArray; \
    unsigned int fOffset;      \
    unsigned int fSize;        \
    unsigned int fTypeSize;    \
    unsigned int fDimensions;  \
    EColumnType  fType;        \
};"
#else
# define TABLEDESCRIPTOR_SPEC      \
 "struct tableDescriptor {         \
    char         fColumnName[20]; \
    unsigned int fIndexArray[2];  \
    unsigned int fOffset;         \
    unsigned int fSize;           \
    unsigned int fTypeSize;       \
    unsigned int fDimensions;     \
    EColumnType  fType;           \
};"
#endif

/*   this is a name clas with ROOT
 * enum EColumnType {kNAN, kFloat, kInt, kLong, kShort, kDouble, kUInt
 *                     ,kULong, kUShort, kUChar, kChar };
 */

/*  This is to introduce an artificial restriction demanded by STAR database group
 *
 *    1. the name may be 31 symbols at most
 *    2. the number of the dimensions is 3 at most
 *
 *  To lift this restriction one has to provide -DNORESTRICTIONS CPP symbol and
 *  recompile code.
 */
//______________________________________________________________________________
//
// Table descriptor tableDescriptor_st is internal descriptor of TTableDescriptor class
//
// One should not use it directly.
// To access the TTable descriptor information use TTableDescriptor object instead
//______________________________________________________________________________

typedef struct tableDescriptor_st {
#ifdef NORESTRICTIONS
    char         *fColumnName; /* The name of this data-member                                         */
    unsigned int *fIndexArray; /* The array of the sizes for each dimensions fIndexArray[fDimensions]  */
#else
    char         fColumnName[32];  /* The name of this data-member: see dstype.h for dsl compatible mode */
    unsigned int fIndexArray[3];   /* The array of the sizes for each dimensions fIndexArray[fDimensions]*/
#endif
    unsigned int fOffset;      /* The first byte in the row of this column                              */
    unsigned int fSize;        /* The full size of the selected column in bytes                         */
    unsigned int fTypeSize;    /* The type size of the selected column in bytes                         */
    unsigned int fDimensions;  /* The number of the dimensions for array                                */
    int          fType;        /* The data type of the selected column                                  */
} TABLEDESCRIPTOR_ST;
// $Log: tableDescriptor.h,v $
// Revision 1.2  2003/01/27 20:41:36  brun
// New version of the Table package by Valeri Fine.
// New classes TIndexTable TResponseIterator TResponseTable TTableMap
//
// Revision 1.1.1.1  2002/05/28 12:32:02  fisyak
//
//
// Revision 1.1  2002/05/27 16:26:59  rdm
// rename star to table.
//
// Revision 1.1.1.1  2000/05/16 17:00:49  rdm
// Initial import of ROOT into CVS
//
// Revision 1.6  2000/01/12 18:07:25  fine
//  cvs symbols have been added and copyright class introduced
//"
#endif /* TABLEDESCRIPTOR_H */
