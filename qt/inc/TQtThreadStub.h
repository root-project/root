/****************************************************************************
** $Id: TQtThreadStub.h,v 1.7 2004/07/21 21:55:42 fine Exp $
**
** Copyright (C) 2003 by Valeri Fine. Brookhaven National Laboratory.
**                                    All rights reserved.
**
** This file may be distributed under the terms of the Q Public License
** as defined by Trolltech AS of Norway and appearing in the file
** LICENSE.QPL included in the packaging of this file.
**
*****************************************************************************/
#ifndef ROOT_TQtThreadStub
#define ROOT_TQtThreadStub

#ifndef  BASECLASS
#error BASECLASS macro has not been defined yet !!!
#endif

#ifndef  THREADCLASS 
#error THREADCLASS macro has not been defined yet !!!
#endif


#ifndef PROXYPOINTER
#  define PROXYPOINTER this
#  ifdef ISPROXY
#    undef ISPROXY 
#  endif
#else 
#  define ISPROXY 1
#endif

#ifdef ISPROXY
#  define PROXYCLASS BASECLASS
#  define PROXYDIRECTCLASS _NAME1_(PROXYPOINTER)->_NAME1_(BASECLASS)
#else
#  define PROXYCLASS THREADCLASS
#  define PROXYDIRECTCLASS BASECLASS
#endif

#include "TQtApplication.h"
#include "TQtEvent.h"
#include "TWaitCondition.h"

#define RETURNACTION0(type,method)         \
type _NAME1_(THREADCLASS)::method()        \
{                                          \
 if (!TQtApplication::IsThisGuiThread())   \
  {                                        \
    class ev : public TQtEvent {           \
    public:                                \
      ev(QObject *obj): TQtEvent(0) {setData(obj);} \
      void ExecuteCB()                     \
      {                                    \
        SetResult((void *)((_NAME1_(PROXYCLASS) *)data())->_NAME1_(BASECLASS)::method()); \
      }                                    \
    };                                     \
    ev *e = new ev(_NAME1_(PROXYPOINTER));                  \
    TWaitCondition w;                      \
    void *result=0;                        \
    e->SetWait(w,result);                  \
    TGQt::PostQtEvent(this,e);            \
    w.wait();                              \
    return (type)result;                   \
  } else {                                 \
    return PROXYDIRECTCLASS::method();            \
  }                                        \
}
#define VOIDACTION0(method)                \
void _NAME1_(THREADCLASS)::method()        \
{                                          \
 if (!TQtApplication::IsThisGuiThread())   \
  {                                        \
    class ev : public TQtEvent {           \
    public:                                \
      ev(QObject *obj): TQtEvent(0) {setData(obj);} \
      void ExecuteCB()                     \
      {                                    \
        ((_NAME1_(PROXYCLASS) *)data())->_NAME1_(BASECLASS)::method(); \
        SetResult();                       \
      }                                    \
    };                                     \
    ev *e = new ev(_NAME1_(PROXYPOINTER));                  \
    TWaitCondition w;                      \
    e->SetWait(w);                         \
    TGQt::PostQtEvent(this,e);            \
    w.wait();                              \
  } else {                                 \
    _NAME1_(PROXYDIRECTCLASS)::method();          \
  }                                        \
}

#define SENDACTION0(method)                \
void _NAME1_(THREADCLASS)::method()        \
{                                          \
 if (!TQtApplication::IsThisGuiThread())   \
  {                                        \
    class ev : public TQtEvent {           \
    public:                                \
            ev(QObject *obj):TQtEvent(0){setData(obj);} \
      void ExecuteCB()                     \
      {                                    \
        ((_NAME1_(PROXYCLASS) *)data())->_NAME1_(BASECLASS)::method(); \
      }                                    \
    };                                     \
    ev *e = new ev(_NAME1_(PROXYPOINTER));                  \
    TGQt::PostQtEvent(this,e);            \
  } else {                                 \
    _NAME1_(PROXYDIRECTCLASS)::method();          \
  }                                        \
}

#define RETURNACTION1(type,method,type1,par1)\
type _NAME1_(THREADCLASS)::method(type1 par1)\
{                                          \
 if (!TQtApplication::IsThisGuiThread())   \
  {                                        \
    class ev : public TQtEvent {           \
    public:                                \
          type1 par1;                      \
      ev(QObject *obj,type1 p1): TQtEvent(0), par1(p1) {setData(obj);} \
      void ExecuteCB()                     \
      {                                    \
        SetResult((void *)((_NAME1_(PROXYCLASS) *)data())->_NAME1_(BASECLASS)::method(par1)); \
      }                                    \
    };                                     \
    ev *e = new ev(_NAME1_(PROXYPOINTER),par1);             \
    TWaitCondition w;                      \
    void *result=0;                        \
    e->SetWait(w,result);                  \
    TGQt::PostQtEvent(this,e);            \
    w.wait();                              \
    return (type)result;                   \
  } else {                                 \
    return PROXYDIRECTCLASS::method(par1); \
  }                                        \
}
#define SENDACTION1(method,type1,par1)     \
void _NAME1_(THREADCLASS)::method(type1 par1) \
{                                          \
 if (!TQtApplication::IsThisGuiThread())   \
  {                                        \
    class ev : public TQtEvent {           \
    public:                                \
          type1 par1;                      \
      ev(QObject *obj,type1 p1): TQtEvent(0), par1(p1) {setData(obj);}     \
      void ExecuteCB()                     \
      {                                    \
        ((_NAME1_(PROXYCLASS) *)data())->_NAME1_(BASECLASS)::method(par1); \
      }                                    \
    };                                     \
    ev *e = new ev(_NAME1_(PROXYPOINTER),par1); \
    TGQt::PostQtEvent(this,e);            \
  } else {                                 \
    _NAME1_(PROXYDIRECTCLASS)::method(par1);    \
  }                                        \
}

#define VOIDACTION1(method,type1,par1)     \
void _NAME1_(THREADCLASS)::method(type1 par1) \
{                                          \
 if (!TQtApplication::IsThisGuiThread())   \
  {                                        \
    class ev : public TQtEvent {           \
    public:                                \
          type1 par1;                      \
      ev(QObject *obj,type1 p1): TQtEvent(0),par1(p1) {setData(obj);}      \
      void ExecuteCB()                     \
      {                                    \
        ((_NAME1_(PROXYCLASS) *)data())->_NAME1_(BASECLASS)::method(par1); \
        SetResult();                       \
      }                                    \
    };                                     \
    ev *e = new ev(_NAME1_(PROXYPOINTER),par1);  \
    TWaitCondition w;                      \
    e->SetWait(w);                         \
    TGQt::PostQtEvent(this,e);            \
    w.wait();                              \
  } else {                                 \
    _NAME1_(PROXYDIRECTCLASS)::method(par1);     \
  }                                        \
}

#define RETURNACTION2(type,method,type1,par1,type2,par2) \
type _NAME1_(THREADCLASS)::method(type1 par1,type2 par2) \
{                                          \
   if (!TQtApplication::IsThisGuiThread()) \
   {                                       \
    class ev : public TQtEvent {           \
    public:                                \
           type1 par1;                     \
           type2 par2;                     \
      ev(QObject *obj,type1 p1,type2 p2):TQtEvent(0), par1(p1), par2(p2) {setData(obj);} \
      void ExecuteCB()                     \
      {                                    \
        SetResult((void *)((_NAME1_(PROXYCLASS) *)data())->_NAME1_(BASECLASS)::method(par1,par2)); \
      }                                    \
    };                                     \
    ev *e = new ev(_NAME1_(PROXYPOINTER),par1,par2);        \
    TWaitCondition w;                      \
    void *result=0;                        \
    e->SetWait(w,result);                  \
    TGQt::PostQtEvent(this,e);            \
    w.wait();                              \
    return (type)result;                   \
  } else {                                 \
    return _NAME1_(PROXYDIRECTCLASS)::method(par1,par2);  \
  }                                        \
}

#define VOIDACTION2(method,type1,par1,type2,par2)        \
void _NAME1_(THREADCLASS)::method(type1 par1,type2 par2) \
{                                          \
 if (!TQtApplication::IsThisGuiThread())   \
  {                                        \
    class ev : public TQtEvent {           \
    public:                                \
          type1 par1;                      \
          type2 par2;                      \
      ev(QObject *obj,type1 p1,type2 p2):TQtEvent(0),par1(p1), par2(p2){setData(obj);} \
      void ExecuteCB()                     \
      {                                    \
        ((_NAME1_(PROXYCLASS) *)data())->_NAME1_(BASECLASS)::method(par1,par2); \
        SetResult();                       \
      }                                    \
    };                                     \
    ev *e = new ev(_NAME1_(PROXYPOINTER),par1,par2);        \
    TWaitCondition w;                      \
    e->SetWait(w);                         \
    TGQt::PostQtEvent(this,e);            \
    w.wait();                              \
  } else {                                 \
    _NAME1_(PROXYDIRECTCLASS)::method(par1,par2); \
  }                                        \
}

#define SENDACTION2(method,type1,par1,type2,par2)        \
void _NAME1_(THREADCLASS)::method(type1 par1,type2 par2) \
{                                          \
 if (!TQtApplication::IsThisGuiThread())   \
  {                                        \
    class ev : public TQtEvent {           \
    public:                                \
          type1 par1;                      \
          type2 par2;                      \
      ev(QObject *obj,type1 p1,type2 p2):TQtEvent(0), par1(p1), par2(p2) {setData(obj);} \
      void ExecuteCB()                     \
      {                                    \
        ((_NAME1_(PROXYCLASS) *)data())->_NAME1_(BASECLASS)::method(par1,par2); \
      }                                    \
    };                                     \
    ev *e = new ev(_NAME1_(PROXYPOINTER),par1,par2); \
    TGQt::PostQtEvent(this,e);            \
  } else {                                 \
    _NAME1_(PROXYDIRECTCLASS)::method(par1,par2);    \
  }                                        \
}

#define RETURNACTION3(type,method,type1,par1,type2,par2,type3,par3) \
type _NAME1_(THREADCLASS)::method(type1 par1,type2 par2,type3 par3) \
{                                          \
   if (!TQtApplication::IsThisGuiThread()) \
   {                                       \
    class ev : public TQtEvent {           \
    public:                                \
           type1 par1;                     \
           type2 par2;                     \
           type3 par3;                     \
      ev(QObject *obj,type1 p1,type2 p2,type3 p3):TQtEvent(0),par1(p1),par2(p2),par3(p3){setData(obj);} \
      void ExecuteCB()                     \
      {                                    \
        SetResult((void *)((_NAME1_(PROXYCLASS) *)data())->_NAME1_(BASECLASS)::method(par1,par2,par3)); \
      }                                    \
    };                                     \
    ev *e = new ev(_NAME1_(PROXYPOINTER),par1,par2,par3);   \
    TWaitCondition w;                      \
    void *result=0;                        \
    e->SetWait(w,result);                  \
    TGQt::PostQtEvent(this,e);            \
    w.wait();                              \
    return (type)result;                   \
  } else {                                 \
    return _NAME1_(PROXYDIRECTCLASS)::method(par1,par2,par3);  \
  }                                        \
}

#define VOIDACTION3(method,type1,par1,type2,par2,type3,par3)        \
void _NAME1_(THREADCLASS)::method(type1 par1,type2 par2,type3 par3) \
{                                          \
   if (!TQtApplication::IsThisGuiThread()) \
   {                                       \
    class ev : public TQtEvent {           \
    public:                                \
           type1 par1;                     \
           type2 par2;                     \
           type3 par3;                     \
      ev(QObject *obj,type1 p1,type2 p2,type3 p3):TQtEvent(0),par1(p1),par2(p2),par3(p3) {setData(obj);} \
      void ExecuteCB()                     \
      {                                    \
        ((_NAME1_(PROXYCLASS) *)data())->_NAME1_(BASECLASS)::method(par1,par2,par3); \
        SetResult();                       \
      }                                    \
    };                                     \
    ev *e = new ev(_NAME1_(PROXYPOINTER),par1,par2,par3); \
    TWaitCondition w;                      \
    e->SetWait(w);                         \
    TGQt::PostQtEvent(this,e);            \
    w.wait();                              \
    return;                                \
  } else {                                 \
    _NAME1_(PROXYDIRECTCLASS)::method(par1,par2,par3);   \
  }                                        \
}

#define SENDACTION3(method,type1,par1,type2,par2,type3,par3)        \
void _NAME1_(THREADCLASS)::method(type1 par1,type2 par2,type3 par3) \
{                                          \
 if (!TQtApplication::IsThisGuiThread())   \
  {                                        \
    class ev : public TQtEvent {           \
    public:                                \
          type1 par1;                      \
          type2 par2;                      \
          type3 par3;                      \
      ev(QObject *obj,type1 p1,type2 p2,type3 p3):TQtEvent(0),par1(p1),par2(p2),par3(p3) {setData(obj);} \
      void ExecuteCB()                     \
      {                                    \
        ((_NAME1_(PROXYCLASS) *)data())->_NAME1_(BASECLASS)::method(par1,par2,par3); \
      }                                    \
    };                                     \
    ev *e = new ev(_NAME1_(PROXYPOINTER),par1,par2,par3); \
    TGQt::PostQtEvent(this,e);            \
  } else {                                 \
    _NAME1_(PROXYDIRECTCLASS)::method(par1,par2,par3);    \
  }                                        \
}

#define VOIDACTION4(method,type1,par1,type2,par2,type3,par3,type4,par4)        \
void _NAME1_(THREADCLASS)::method(type1 par1,type2 par2,type3 par3,type4 par4) \
{                                          \
 if (!TQtApplication::IsThisGuiThread())   \
  {                                        \
    class ev : public TQtEvent {           \
    public:                                \
          type1 par1;                      \
          type2 par2;                      \
          type3 par3;                      \
          type4 par4;                      \
      ev(QObject *obj,type1 p1,type2 p2,type3 p3,type4 p4):TQtEvent(0),par1(p1),par2(p2),par3(p3),par4(p4) {setData(obj);} \
      void ExecuteCB()                     \
      {                                    \
        ((_NAME1_(PROXYCLASS) *)data())->_NAME1_(BASECLASS)::method(par1,par2,par3,par4); \
        SetResult();                       \
      }                                    \
    };                                     \
    ev *e = new ev(_NAME1_(PROXYPOINTER),par1,par2,par3,par4); \
    TWaitCondition w;                      \
    e->SetWait(w);                         \
    TGQt::PostQtEvent(this,e);            \
    w.wait();                              \
  } else {                                 \
    _NAME1_(PROXYDIRECTCLASS)::method(par1,par2,par3,par4);    \
  }                                        \
}

#define RETURNACTION4(type,method,type1,par1,type2,par2,type3,par3,type4,par4) \
type _NAME1_(THREADCLASS)::method(type1 par1,type2 par2,type3 par3,type4 par4) \
{                                          \
   if (!TQtApplication::IsThisGuiThread()) \
   {                                       \
    class ev : public TQtEvent {           \
    public:                                \
           type1 par1;                     \
           type2 par2;                     \
           type3 par3;                     \
           type4 par4;                     \
      ev(QObject *obj,type1 p1,type2 p2,type3 p3,type4 p4):TQtEvent(0), par1(p1), par2(p2),par3(p3),par4(p4) {setData(obj);} \
      void ExecuteCB()                     \
      {                                    \
        SetResult((void *)((_NAME1_(PROXYCLASS) *)data())->_NAME1_(BASECLASS)::method(par1,par2,par3,par4)); \
      }                                    \
    };                                     \
    ev *e = new ev(_NAME1_(PROXYPOINTER),par1,par2,par3,par4);  \
    TWaitCondition w;                      \
    void *result=0;                        \
    e->SetWait(w,result);                  \
    TGQt::PostQtEvent(this,e);            \
    w.wait();                              \
    return (type)result;                   \
  } else {                                 \
    return _NAME1_(PROXYDIRECTCLASS)::method(par1,par2,par3,par4);  \
  }                                        \
}


#define SENDACTION4(method,type1,par1,type2,par2,type3,par3,type4,par4)        \
void _NAME1_(THREADCLASS)::method(type1 par1,type2 par2,type3 par3,type4 par4) \
{                                          \
   if (!TQtApplication::IsThisGuiThread()) \
   {                                       \
    class ev : public TQtEvent {           \
    public:                                \
          type1 par1;                      \
          type2 par2;                      \
          type3 par3;                      \
          type4 par4;                      \
      ev(QObject *obj,type1 p1,type2 p2,type3 p3,type4 p4):TQtEvent(0),par1(p1),par2(p2),par3(p3),par4(p4){setData(obj);} \
      void ExecuteCB()                     \
      {                                    \
        ((_NAME1_(PROXYCLASS) *)data())->_NAME1_(BASECLASS)::method(par1,par2,par3,par4); \
      }                                    \
    };                                     \
    ev *e = new ev(_NAME1_(PROXYPOINTER),par1,par2,par3,par4); \
    TGQt::PostQtEvent(this,e);            \
  } else {                                 \
    _NAME1_(PROXYDIRECTCLASS)::method(par1,par2,par3,par4);    \
  }                                        \
}

#define SENDACTION5(method,type1,par1,type2,par2,type3,par3,type4,par4,type5,par5)        \
void _NAME1_(THREADCLASS)::method(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5) \
{                                          \
   if (!TQtApplication::IsThisGuiThread()) \
   {                                       \
    class ev : public TQtEvent {           \
    public:                                \
          type1 par1;                      \
          type2 par2;                      \
          type3 par3;                      \
          type4 par4;                      \
          type5 par5;                      \
      ev(QObject *obj                      \
      ,type1 p1,type2 p2,type3 p3,type4 p4,type5 p5): TQtEvent(0), par1(p1),par2(p2),par3(p3),par4(p4),par5(p5) \
      {setData(obj);}                      \
      void ExecuteCB()                     \
      {                                    \
        ((_NAME1_(PROXYCLASS) *)data())->_NAME1_(BASECLASS)::method(par1,par2,par3,par4,par5); \
      }                                    \
    };                                     \
    ev *e = new ev(_NAME1_(PROXYPOINTER),par1,par2,par3,par4,par5); \
    TGQt::PostQtEvent(this,e);            \
  } else {                                 \
    _NAME1_(PROXYDIRECTCLASS)::method(par1,par2,par3,par4,par5);    \
  }                                        \
}

#define VOIDACTION5(method,type1,par1,type2,par2,type3,par3,type4,par4,type5,par5)        \
void _NAME1_(THREADCLASS)::method(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5) \
{                                          \
   if (!TQtApplication::IsThisGuiThread()) \
   {                                       \
    class ev : public TQtEvent {           \
    public:                                \
          type1 par1;                      \
          type2 par2;                      \
          type3 par3;                      \
          type4 par4;                      \
          type5 par5;                      \
      ev(QObject *obj                      \
      ,type1 p1,type2 p2,type3 p3,type4 p4,type5 p5): TQtEvent(0), par1(p1),par2(p2),par3(p3),par4(p4),par5(p5) \
      {setData(obj);}                      \
      void ExecuteCB()                     \
      {                                    \
        ((_NAME1_(PROXYCLASS) *)data())->_NAME1_(BASECLASS)::method(par1,par2,par3,par4,par5); \
        SetResult();                       \
     }                                     \
    };                                     \
    ev *e = new ev(_NAME1_(PROXYPOINTER),par1,par2,par3,par4,par5); \
    TWaitCondition w;                      \
    e->SetWait(w);                         \
    TGQt::PostQtEvent(this,e);            \
    w.wait();                              \
  } else {                                 \
    _NAME1_(PROXYDIRECTCLASS)::method(par1,par2,par3,par4,par5);    \
  }                                        \
}

#define SENDACTION6(method,type1,par1,type2,par2,type3,par3,type4,par4,type5,par5,type6,par6)        \
void _NAME1_(THREADCLASS)::method(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6) \
{                                          \
   if (!TQtApplication::IsThisGuiThread()) \
   {                                       \
    class ev : public TQtEvent {           \
    public:                                \
          type1 par1;                      \
          type2 par2;                      \
          type3 par3;                      \
          type4 par4;                      \
          type5 par5;                      \
          type6 par6;                      \
      ev(QObject *obj                      \
      ,type1 p1,type2 p2,type3 p3,type4 p4,type5 p5,type6 p6):TQtEvent(0), par1(p1),par2(p2),par3(p3),par4(p4),par5(p5),par6(p6)\
      {setData(obj);}                      \
      void ExecuteCB()                     \
      {                                    \
        ((_NAME1_(PROXYCLASS) *)data())->_NAME1_(BASECLASS)::method(par1,par2,par3,par4,par5,par6); \
      }                                    \
    };                                     \
    ev *e = new ev(_NAME1_(PROXYPOINTER),par1,par2,par3,par4,par5,par6);        \
    TGQt::PostQtEvent(this,e);            \
  } else {                                 \
    _NAME1_(PROXYDIRECTCLASS)::method(par1,par2,par3,par4,par5,par6); \
  }                                        \
}

#define VOIDACTION6(method,type1,par1,type2,par2,type3,par3,type4,par4,type5,par5,type6,par6)        \
void _NAME1_(THREADCLASS)::method(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6) \
{                                          \
   if (!TQtApplication::IsThisGuiThread()) \
   {                                       \
    class ev : public TQtEvent {           \
    public:                                \
          type1 par1;                      \
          type2 par2;                      \
          type3 par3;                      \
          type4 par4;                      \
          type5 par5;                      \
          type6 par6;                      \
      ev(QObject *obj                      \
      ,type1 p1,type2 p2,type3 p3,type4 p4,type5 p5,type6 p6):TQtEvent(0), par1(p1),par2(p2),par3(p3),par4(p4),par5(p5),par6(p6)\
      {setData(obj);}                      \
      void ExecuteCB()                     \
      {                                    \
        ((_NAME1_(PROXYCLASS) *)data())->_NAME1_(BASECLASS)::method(par1,par2,par3,par4,par5,par6); \
        SetResult();                       \
      }                                    \
    };                                     \
    ev *e = new ev(_NAME1_(PROXYPOINTER),par1,par2,par3,par4,par5,par6); \
    TWaitCondition w;                      \
    e->SetWait(w);                         \
    TGQt::PostQtEvent(this,e);            \
    w.wait();                              \
  } else {                                 \
    _NAME1_(PROXYDIRECTCLASS)::method(par1,par2,par3,par4,par5,par6);    \
  }                                        \
}

#define SENDACTION7(method,type1,par1,type2,par2,type3,par3,type4,par4,type5,par5,type6,par6,type7,par7)        \
void _NAME1_(THREADCLASS)::method(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6,type7 par7) \
{                                          \
   if (!TQtApplication::IsThisGuiThread()) \
   {                                       \
    class ev : public TQtEvent {           \
    public:                                \
          type1 par1;                      \
          type2 par2;                      \
          type3 par3;                      \
          type4 par4;                      \
          type5 par5;                      \
          type6 par6;                      \
          type7 par7;                      \
      ev(QObject *obj                      \
      ,type1 p1,type2 p2,type3 p3,type4 p4,type5 p5,type6 p6,type7 p7):TQtEvent(0), par1(p1),par2(p2),par3(p3),par4(p4),par5(p5),par6(p6),par7(p7)\
      {setData(obj);}                      \
      void ExecuteCB()                     \
      {                                    \
        ((_NAME1_(PROXYCLASS) *)data())->_NAME1_(BASECLASS)::method(par1,par2,par3,par4,par5,par6,par7); \
      }                                    \
    };                                     \
    ev *e = new ev(_NAME1_(PROXYPOINTER),par1,par2,par3,par4,par5,par6,par7);        \
    TGQt::PostQtEvent(this,e);            \
  } else {                                 \
    _NAME1_(PROXYDIRECTCLASS)::method(par1,par2,par3,par4,par5,par6,par7); \
  }                                        \
}

#define VOIDACTION7(method,type1,par1,type2,par2,type3,par3,type4,par4,type5,par5,type6,par6,type7,par7)        \
void _NAME1_(THREADCLASS)::method(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6,type7 par7) \
{                                          \
   if (!TQtApplication::IsThisGuiThread()) \
   {                                       \
    class ev : public TQtEvent {           \
    public:                                \
          type1 par1;                      \
          type2 par2;                      \
          type3 par3;                      \
          type4 par4;                      \
          type5 par5;                      \
          type6 par6;                      \
          type7 par7;                      \
      ev(QObject *obj                      \
      ,type1 p1,type2 p2,type3 p3,type4 p4,type5 p5,type6 p6,type7 p7):TQtEvent(0), par1(p1),par2(p2),par3(p3),par4(p4),par5(p5),par6(p6),par7(p7)\
      {setData(obj);}                      \
      void ExecuteCB()                     \
      {                                    \
        ((_NAME1_(PROXYCLASS) *)data())->_NAME1_(BASECLASS)::method(par1,par2,par3,par4,par5,par6,par7); \
        SetResult();                       \
      }                                    \
    };                                     \
    ev *e = new ev(_NAME1_(PROXYPOINTER),par1,par2,par3,par4,par5,par6,par7);        \
    TWaitCondition w;                      \
    e->SetWait(w);                         \
    TGQt::PostQtEvent(this,e);            \
    w.wait();                              \
  } else {                                 \
    _NAME1_(PROXYDIRECTCLASS)::method(par1,par2,par3,par4,par5,par6,par7); \
  }                                        \
}
#define RETURNACTION7(type,method,type1,par1,type2,par2,type3,par3,type4,par4,type5,par5,type6,par6,type7,par7) \
type _NAME1_(THREADCLASS)::method(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6,type7 par7) \
{                                          \
   if (!TQtApplication::IsThisGuiThread()) \
   {                                       \
    class ev : public TQtEvent {           \
    public:                                \
           type1  par1;                     \
           type2  par2;                     \
           type3  par3;                     \
           type4  par4;                     \
           type5  par5;                     \
           type6  par6;                     \
           type7  par7;                     \
      ev(QObject *obj,type1 p1,type2 p2,type3 p3,type4 p4,type5 p5,type6 p6,type7 p7):TQtEvent(0)\
           , par1(p1), par2(p2),par3(p3),par4(p4),par5(p5),par6(p6),par7(p7) {setData(obj);} \
      void ExecuteCB()                     \
      {                                    \
        SetResult((void *)((_NAME1_(PROXYCLASS) *)data())->_NAME1_(BASECLASS)::\
        method(par1,par2,par3,par4,par5,par6,par7)); \
      }                                    \
    };                                     \
    ev *e = new ev(_NAME1_(PROXYPOINTER),par1,par2,par3,par4,par5,par6,par7);  \
    TWaitCondition w;                      \
    void *result=0;                        \
    e->SetWait(w,result);                  \
    TGQt::PostQtEvent(this,e);            \
    w.wait();                              \
    return (type)result;                   \
  } else {                                 \
    return _NAME1_(PROXYDIRECTCLASS)::method(par1,par2,par3,par4,par5,par6,par7);  \
  }                                        \
}
#define VOIDACTION8(method,type1,par1,type2,par2,type3,par3,type4,par4,type5,par5,type6,par6,type7,par7,type8,par8)        \
void _NAME1_(THREADCLASS)::method(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6,type7 par7,type8 par8) \
{                                          \
   if (!TQtApplication::IsThisGuiThread()) \
   {                                       \
    class ev : public TQtEvent {           \
    public:                                \
          type1 par1;                      \
          type2 par2;                      \
          type3 par3;                      \
          type4 par4;                      \
          type5 par5;                      \
          type6 par6;                      \
          type7 par7;                      \
          type8 par8;                      \
      ev(QObject *obj                      \
      ,type1 p1,type2 p2,type3 p3,type4 p4,type5 p5,type6 p6,type7 p7,type8 p8):TQtEvent(0), par1(p1),par2(p2),par3(p3),par4(p4),par5(p5),par6(p6),par7(p7),par8(p8)\
      {setData(obj);}                      \
      void ExecuteCB()                     \
      {                                    \
        ((_NAME1_(PROXYCLASS) *)data())->_NAME1_(BASECLASS)::method(par1,par2,par3,par4,par5,par6,par7,par8); \
        SetResult();                       \
      }                                    \
    };                                     \
    ev *e = new ev(_NAME1_(PROXYPOINTER),par1,par2,par3,par4,par5,par6,par7,par8);        \
    TWaitCondition w;                      \
    e->SetWait(w);                         \
    TGQt::PostQtEvent(this,e);            \
    w.wait();                              \
  } else {                                 \
    _NAME1_(PROXYDIRECTCLASS)::method(par1,par2,par3,par4,par5,par6,par7,par8); \
  }                                        \
}
#define VOIDACTION9(method,type1,par1,type2,par2,type3,par3,type4,par4,type5,par5,type6,par6,type7,par7,type8,par8,type9,par9)        \
void _NAME1_(THREADCLASS)::method(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6,type7 par7,type8 par8,type9 par9) \
{                                          \
   if (!TQtApplication::IsThisGuiThread()) \
   {                                       \
    class ev : public TQtEvent {           \
    public:                                \
          type1 par1;                      \
          type2 par2;                      \
          type3 par3;                      \
          type4 par4;                      \
          type5 par5;                      \
          type6 par6;                      \
          type7 par7;                      \
          type8 par8;                      \
          type9 par9;                      \
      ev(QObject *obj                      \
      ,type1 p1,type2 p2,type3 p3,type4 p4,type5 p5,type6 p6,type7 p7,type8 p8,type9 p9):TQtEvent(0), par1(p1),par2(p2),par3(p3),par4(p4),par5(p5),par6(p6),par7(p7),par8(p8),par9(p9)\
      {setData(obj);}                      \
      void ExecuteCB()                     \
      {                                    \
        ((_NAME1_(PROXYCLASS) *)data())->_NAME1_(BASECLASS)::method(par1,par2,par3,par4,par5,par6,par7,par8,par9); \
        SetResult();                       \
      }                                    \
    };                                     \
    ev *e = new ev(_NAME1_(PROXYPOINTER),par1,par2,par3,par4,par5,par6,par7,par8,par9);        \
    TWaitCondition w;                      \
    e->SetWait(w);                         \
    TGQt::PostQtEvent(this,e);            \
    w.wait();                              \
  } else {                                 \
    _NAME1_(PROXYDIRECTCLASS)::method(par1,par2,par3,par4,par5,par6,par7,par8,par9); \
  }                                        \
}
#define SENDACTION9(method,type1,par1,type2,par2,type3,par3,type4,par4,type5,par5,type6,par6,type7,par7,type8,par8,type9,par9)        \
void _NAME1_(THREADCLASS)::method(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6,type7 par7,type8 par8,type9 par9) \
{                                          \
   if (!TQtApplication::IsThisGuiThread()) \
   {                                       \
    class ev : public TQtEvent {           \
    public:                                \
          type1 par1;                      \
          type2 par2;                      \
          type3 par3;                      \
          type4 par4;                      \
          type5 par5;                      \
          type6 par6;                      \
          type7 par7;                      \
          type8 par8;                      \
          type9 par9;                      \
      ev(QObject *obj                      \
      ,type1 p1,type2 p2,type3 p3,type4 p4,type5 p5,type6 p6,type7 p7,type8 p8,type9 p9)\
        :TQtEvent(0), par1(p1),par2(p2),par3(p3),par4(p4),par5(p5),par6(p6),par7(p7),par8(p8),par9(p9)\
      {setData(obj);}                      \
      void ExecuteCB()                     \
      {                                    \
        ((_NAME1_(PROXYCLASS) *)data())->_NAME1_(BASECLASS)::method(par1,par2,par3,par4,par5,par6,par7,par8,par9); \
      }                                    \
    };                                     \
    ev *e = new ev(_NAME1_(PROXYPOINTER),par1,par2,par3,par4,par5,par6,par7,par8,par9);        \
    TGQt::PostQtEvent(this,e);            \
  } else {                                 \
    _NAME1_(PROXYDIRECTCLASS)::method(par1,par2,par3,par4,par5,par6,par7,par8,par9); \
  }                                        \
}
#define RETURNACTION11(type,method,type1,par1,type2,par2,type3,par3,type4,par4,type5,par5,type6,par6,type7,par7,type8,par8,type9,par9,type10,par10,type11,par11) \
type _NAME1_(THREADCLASS)::method(type1 par1,type2 par2,type3 par3,type4 par4,type5 par5,type6 par6,type7 par7,type8 par8,type9 par9,type10 par10,type11 par11)  \
{                                          \
   if (!TQtApplication::IsThisGuiThread()) \
   {                                       \
    class ev : public TQtEvent {           \
    public:                                \
           type1  par1;                     \
           type2  par2;                     \
           type3  par3;                     \
           type4  par4;                     \
           type5  par5;                     \
           type6  par6;                     \
           type7  par7;                     \
           type8  par8;                     \
           type9  par9;                     \
           type10 par10;                    \
           type11 par11;                    \
      ev(QObject *obj,type1 p1,type2 p2,type3 p3,type4 p4,type5 p5,type6 p6,type7 p7,type8 p8,type9 p9,type10 p10,type11 p11):TQtEvent(0)\
           , par1(p1), par2(p2),par3(p3),par4(p4),par5(p5),par6(p6),par7(p7),par8(p8),par9(p9),par10(p10),par11(p11) {setData(obj);} \
      void ExecuteCB()                     \
      {                                    \
        SetResult((void *)((_NAME1_(PROXYCLASS) *)data())->_NAME1_(BASECLASS)::\
        method(par1,par2,par3,par4,par5,par6,par7,par8,par9,par10,par11)); \
      }                                    \
    };                                     \
    ev *e = new ev(_NAME1_(PROXYPOINTER),par1,par2,par3,par4,par5,par6,par7,par8,par9,par10,par11);  \
    TWaitCondition w;                      \
    void *result=0;                        \
    e->SetWait(w,result);                  \
    TGQt::PostQtEvent(this,e);            \
    w.wait();                              \
    return (type)result;                   \
  } else {                                 \
    return _NAME1_(PROXYDIRECTCLASS)::method(par1,par2,par3,par4,par5,par6,par7,par8,par9,par10,par11);  \
  }                                        \
}
#endif
