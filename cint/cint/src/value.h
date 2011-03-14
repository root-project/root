#ifndef G__INCLUDE_VALUE_H
#define G__INCLUDE_VALUE_H

template<typename T>
inline T G__convertT(const G__value* buf)
{
   // NOTE that because of branch prediction optimization in x86
   // using the following is actually faster than a switch (but
   // might not work when _SUNPRO_CC is true
   //   return (('f'==buf->type||'d'==buf->type) ? buf->obj.d :
   //           ('k'==buf->type||'h'==buf->type) ? (double)(buf->obj.ulo) :
   //           ('m'==buf->type) ? (double)(G__int64)(buf->obj.ull) :
   //           ('n'==buf->type) ? (double)(buf->obj.ll) :
   //           (double)(buf->obj.i) );
 
   const int buftype = buf->type;
   // this is by far the most common case; move in front of expensive switch:
   if (buftype == 'i') return (T) buf->obj.in;

   switch(buftype) {
   //case 'i': return (T) buf->obj.in;
   case 'c': return (T) buf->obj.ch;
   case 's': return (T) buf->obj.sh;
   case 'd': /* double */
      // return (T) buf->obj.d;
   case 'f': /* float */
      return (T) buf->obj.d;
   case 'w': /* logic */
   case 'r': /* unsigned short */
      return (T) buf->obj.ush;
   case 'h': /* unsigned int */
      return (T) buf->obj.uin;
#ifndef G__BOOL4BYTE
   case 'g': /* bool */
#endif
   case 'b': /* unsigned char */
      return (T) buf->obj.uch;
   case 'k': /* unsigned long */
      return (T) buf->obj.ulo;
   case 'n': return (T) buf->obj.ll;
   case 'm': return (T) buf->obj.ull;
   case 'q': return (T) buf->obj.ld;
   default: ;
   }
   return (T) buf->obj.i;
}

template<typename T> T& G__value_ref(G__value &buf);

template <> inline long double &G__value_ref<long double>(G__value &buf){ return buf.obj.ld; }
template <> inline double &G__value_ref<double>(G__value &buf){ return buf.obj.d; }
// The union's "f" member is never set, so we need to do it when resuting a ref.
// This will still fail when we assume it points to the "d" member,
// but at least the value might be correct.
template <> inline float &G__value_ref<float>(G__value &buf)
{ buf.obj.fl = (float)buf.obj.d; return (float&) buf.obj.fl; }

template <> inline unsigned char & G__value_ref<unsigned char>(G__value  & buf){ return buf.obj.uch;}
template <> inline unsigned short & G__value_ref<unsigned short>(G__value  & buf){ return buf.obj.ush;}
template <> inline unsigned int & G__value_ref<unsigned int>(G__value  & buf){ return buf.obj.uin;}
template <> inline unsigned long & G__value_ref<unsigned long>(G__value  & buf){ return buf.obj.ulo;}
template <> inline unsigned long long & G__value_ref<unsigned long long>(G__value  & buf){ return buf.obj.ull;}

template <> inline char & G__value_ref<char>(G__value  & buf){ return buf.obj.ch;}
template <> inline short & G__value_ref<short>(G__value  & buf){ return buf.obj.sh;}
template <> inline int & G__value_ref<int>(G__value  & buf){ return buf.obj.in;}
template <> inline long & G__value_ref<long>(G__value  & buf){ return buf.obj.i;}
template <> inline long long & G__value_ref<long long>(G__value  & buf){ return buf.obj.ll;}

#if defined(__GNUC__) && __GNUC__ >= 4 && ((__GNUC_MINOR__ == 2 && __GNUC_PATCHLEVEL__ >= 1) || (__GNUC_MINOR__ >= 3)) && !__INTEL_COMPILER
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif // __GNUC__ && __GNUC__ > 3 && __GNUC_MINOR__ > 1
template <> inline bool & G__value_ref<bool>(G__value  & buf)
#ifdef G__BOOL4BYTE
{ return (bool&)buf.obj.i; }
#else
{ return (bool&)buf.obj.uch; }
#endif
#if defined(__GNUC__) && __GNUC__ >= 4 && ((__GNUC_MINOR__ == 2 && __GNUC_PATCHLEVEL__ >= 1) || (__GNUC_MINOR__ >= 3)) && !__INTEL_COMPILER
#pragma GCC diagnostic warning "-Wstrict-aliasing"
#endif // __GNUC__ && __GNUC__ > 3 && __GNUC_MINOR__ > 1

template <typename T>
inline void G__setvalue(G__value* pbuf, const T& value) { pbuf->obj.i = (long) value; }

template <> inline void G__setvalue(G__value* pbuf, const bool& value)
#ifdef G__BOOL4BYTE
{ pbuf->obj.i = value ? 1 : 0; }
#else
{ pbuf->obj.uch = value ? 1 : 0; }
#endif
template <> inline void G__setvalue(G__value* pbuf, const unsigned char& value)
{ pbuf->obj.uch = value; }
template <> inline void G__setvalue(G__value* pbuf, const char& value)
{ pbuf->obj.ch = value; }
template <> inline void G__setvalue(G__value* pbuf, const unsigned short& value)
{ pbuf->obj.ush = value; }
template <> inline void G__setvalue(G__value* pbuf, const short& value)
{ pbuf->obj.sh = value; }
template <> inline void G__setvalue(G__value* pbuf, const unsigned int& value)
{ pbuf->obj.uin = value; }
template <> inline void G__setvalue(G__value* pbuf, const int& value)
{ pbuf->obj.i = value; } // should be ".in", but too many cases where "int" should really be "long"
template <> inline void G__setvalue(G__value* pbuf, const unsigned long& value)
{ pbuf->obj.ulo = value; }
template <> inline void G__setvalue(G__value* pbuf, const long& value)
{ pbuf->obj.i = value; }
template <> inline void G__setvalue(G__value* pbuf, const G__uint64& value)
{ pbuf->obj.ull = value; }
template <> inline void G__setvalue(G__value* pbuf, const G__int64& value)
{ pbuf->obj.ll = value; }
template <> inline void G__setvalue(G__value* pbuf, const float& value)
{ pbuf->obj.d = value; }
template <> inline void G__setvalue(G__value* pbuf, const double& value)
{ pbuf->obj.d = value; }
template <> inline void G__setvalue(G__value* pbuf, const long double& value)
{ pbuf->obj.ld = value; }

template <typename T> inline char G__gettypechar() { return 0; }
template<> inline char G__gettypechar<bool>() { return 'g'; }
template<> inline char G__gettypechar<unsigned char>() { return 'b'; }
template<> inline char G__gettypechar<char>() { return 'c'; }
template<> inline char G__gettypechar<unsigned short>() { return 'r'; }
template<> inline char G__gettypechar<short>() { return 's'; }
template<> inline char G__gettypechar<unsigned int>() { return 'h'; }
template<> inline char G__gettypechar<int>() { return 'i'; }
template<> inline char G__gettypechar<unsigned long>() { return 'k'; }
template<> inline char G__gettypechar<long>() { return 'l'; }
template<> inline char G__gettypechar<G__uint64>() { return 'm'; }
template<> inline char G__gettypechar<G__int64>() { return 'n'; }
template<> inline char G__gettypechar<float>() { return 'f'; }
template<> inline char G__gettypechar<double>() { return 'd'; }
template<> inline char G__gettypechar<long double>() { return 'q'; }

#endif
