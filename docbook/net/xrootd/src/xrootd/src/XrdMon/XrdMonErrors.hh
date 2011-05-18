/*****************************************************************************/
/*                                                                           */
/*                              XrdMonErrors.hh                              */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#ifndef XRDMONERRORS_HH
#define XRDMONERRORS_HH

typedef int err_t;

const err_t ERR_CANNOTOPENFILE  =    1;
const err_t ERR_DICTIDINCACHE   =   10;
const err_t ERR_FILENOTCLOSED   =   20;
const err_t ERR_FILENOTOPEN     =   30;
const err_t ERR_FILEOPEN        =   40;
const err_t ERR_INTERNALERR     =   50;
const err_t ERR_INVALIDADDR     =   60;
const err_t ERR_INVALIDARG      =   70;
const err_t ERR_INVALIDFNAME    =   80;
const err_t ERR_INVALIDINFOTYPE =   90;
const err_t ERR_INVALIDSEQNO    =  100;
const err_t ERR_INVALIDTIME     =  110;
const err_t ERR_INVDICTSTRING   =  120;
const err_t ERR_INVPACKETLEN    =  130;
const err_t ERR_INVPACKETTYPE   =  140;
const err_t ERR_NEGATIVEOFFSET  =  150;
const err_t ERR_NOAVAILLOG      =  160;
const err_t ERR_NODICTIDINCACHE =  170;
const err_t ERR_NODIR           =  180;
const err_t ERR_NOMEM           =  190;
const err_t ERR_NOTATIMEWINDOW  =  200;
const err_t ERR_OUTOFMEMORY     =  210;
const err_t ERR_PTHREADCREATE   =  220;
const err_t ERR_PUBLISHFAILED   =  230;
const err_t ERR_RECEIVE         =  240;
const err_t ERR_RENAME          =  250;
const err_t ERR_SENDERNOTREG    =  260;
const err_t ERR_TOOMANYLOST     =  270;
const err_t ERR_UNKNOWN         =  280;
const err_t ERR_USERIDINCACHE   =  290;

const err_t SIG_SHUTDOWNNOW     = 9000;

#endif /* XRDMONERRORS_HH */
