#ifndef __OUC_CHAIN__
#define __OUC_CHAIN__
/******************************************************************************/
/*                                                                            */
/*                        X r d O u c C h a i n . h h                         */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

template<class T>
class XrdOucQSItem
{
public:
XrdOucQSItem<T>  *nextelem;
T                *dataitem;
                  XrdOucQSItem(T *item) {dataitem = item; nextelem = 0;}
                 ~XrdOucQSItem()        {}
};
  
template<class T>
class XrdOucStack
{
public:

int    isEmpty() {return anchor == 0;}

T     *Pop() {XrdOucQSItem<T> *cp;
              if (!(cp = anchor)) return (T *)0;
              anchor = anchor->nextelem;
              cp->nextelem = 0;
              return cp->dataitem;
             }

void   Push(XrdOucQSItem<T> *item) {item->nextelem = anchor; anchor = item;}

       XrdOucStack() {anchor = 0;}
      ~XrdOucStack() {}

private:
XrdOucQSItem<T>    *anchor;
};

template<class T>
class XrdOucQueue
{
public:

void   Add(XrdOucQSItem<T> *item) 
             {item->nextelem = 0;
              if (lastelem) {lastelem->nextelem = item;
                             lastelem = item;
                            }
                 else        anchor = lastelem  = item;
             }

int    isEmpty() {return anchor == 0;}

T     *Remove() {XrdOucQSItem<T> *qp;
                 if (!(qp = anchor)) return (T *)0;
                 if (!(anchor = anchor->nextelem)) lastelem = 0;
                 return qp->dataitem;
                }

       XrdOucQueue() {anchor = lastelem = 0;}
      ~XrdOucQueue() {}

private:
XrdOucQSItem<T> *anchor;
XrdOucQSItem<T> *lastelem;
};
#endif
