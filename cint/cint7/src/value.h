#ifndef G__INCLUDE_VALUE_H
#define G__INCLUDE_VALUE_H

namespace Cint {
namespace Internal {

//______________________________________________________________________________
template<class T> inline T G__convertT(const G__value* buf)
{
   char type = G__get_type(*buf);
   switch (type) {
      case 'b': // unsigned char
         return (T) buf->obj.uch;
      case 'c': // char
         return (T) buf->obj.ch;
      case 'd': // double
         return (T) buf->obj.d;
      case 'f': // float
         return (T) buf->obj.d;
#ifdef G__BOOL4BYTE
      case 'g': // bool
         return (T) buf->obj.i;
#else // G__BOOL4BYTE
      case 'g': // bool
         return (T) buf->obj.uch;
#endif // G__BOOL4BYTE
      case 'h': // unsigned int
         return (T) buf->obj.uin;
      case 'i': // int
         return (T) buf->obj.in;
      case 'k': // unsigned long
         return (T) buf->obj.ulo;
      case 'w': // logic
         return (T) buf->obj.ush;
      case 'r': // unsigned short
         return (T) buf->obj.ush;
      case 'm': // unsigned long long
         return (T) buf->obj.ull;
      case 'n': // long long
         return (T) buf->obj.ll;
      case 'q': // long double
         return (T) buf->obj.ld;
      case 's': // short
         return (T) buf->obj.sh;
   }
   return (T) buf->obj.i;
}

//______________________________________________________________________________
template<class T> T& G__value_ref(G__value& buf);

//______________________________________________________________________________
template<> inline long double& G__value_ref<long double>(G__value& buf)
{
   return buf.obj.ld;
}

//______________________________________________________________________________
template<> inline double& G__value_ref<double>(G__value& buf)
{
   return buf.obj.d;
}

//______________________________________________________________________________
template<> inline float& G__value_ref<float>(G__value& buf)
{
   // The union's "f" member is never set, so we need to do it ourselves.
   // This will still fail when we assume it points to the "d" member,
   // but at least the value might be correct.
   buf.obj.fl = (float)buf.obj.d;
   return (float&) buf.obj.fl;
}

//______________________________________________________________________________
template<> inline unsigned char& G__value_ref<unsigned char>(G__value& buf)
{
   return buf.obj.uch;
};

//______________________________________________________________________________
template<> inline unsigned short& G__value_ref<unsigned short>(G__value& buf)
{
   return buf.obj.ush;
};

//______________________________________________________________________________
template<> inline unsigned int& G__value_ref<unsigned int>(G__value& buf)
{
   return buf.obj.uin;
};

//______________________________________________________________________________
template<> inline unsigned long& G__value_ref<unsigned long>(G__value& buf)
{
   return buf.obj.ulo;
};

//______________________________________________________________________________
template<> inline unsigned long long& G__value_ref<unsigned long long>(G__value& buf)
{
   return buf.obj.ull;
};

//______________________________________________________________________________
template<> inline char& G__value_ref<char>(G__value& buf)
{
   return buf.obj.ch;
};

//______________________________________________________________________________
template<> inline short& G__value_ref<short>(G__value& buf)
{
   return buf.obj.sh;
};

//______________________________________________________________________________
template<> inline int& G__value_ref<int>(G__value& buf)
{
   return buf.obj.in;
};

//______________________________________________________________________________
template<> inline long& G__value_ref<long>(G__value& buf)
{
   return buf.obj.i;
};

//______________________________________________________________________________
template<> inline long long& G__value_ref<long long>(G__value& buf)
{
   return buf.obj.ll;
};

#if defined(__GNUC__) && (__GNUC__ > 4 || ((__GNUC__ == 4) && (__GNUC_MINOR__ > 1)))
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif // __GNUC__ && __GNUC__ > 3 && __GNUC_MINOR__ > 1
#ifdef G__BOOL4BYTE

//______________________________________________________________________________
template<> inline bool& G__value_ref<bool>(G__value& buf)
{
   return (bool&)buf.obj.i;
}
#else // G__BOOL4BYTE
template<> inline bool& G__value_ref<bool>(G__value& buf)
{
   return (bool&)buf.obj.uch;
}
#endif // G__BOOL4BYTE

//______________________________________________________________________________
template<class T> inline void G__setvalue(G__value* pbuf, const T& value)
{
   pbuf->obj.i = (long) value;
}

//______________________________________________________________________________
#ifdef G__BOOL4BYTE
template<> inline void G__setvalue(G__value* pbuf, const bool& value)
{
   pbuf->obj.i = value ? 1 : 0;
}
#else // G__BOOL4BYTE
template<> inline void G__setvalue(G__value* pbuf, const bool& value)
{
   pbuf->obj.uch = value ? 1 : 0;
}
#endif // G__BOOL4BYTE

//______________________________________________________________________________
template<> inline void G__setvalue(G__value* pbuf, const unsigned char& value)
{
   pbuf->obj.uch = value;
}

//______________________________________________________________________________
template<> inline void G__setvalue(G__value* pbuf, const char& value)
{
   pbuf->obj.ch = value;
}

//______________________________________________________________________________
template<> inline void G__setvalue(G__value* pbuf, const unsigned short& value)
{
   pbuf->obj.ush = value;
}

//______________________________________________________________________________
template<> inline void G__setvalue(G__value* pbuf, const short& value)
{
   pbuf->obj.sh = value;
}

//______________________________________________________________________________
template<> inline void G__setvalue(G__value* pbuf, const unsigned int& value)
{
   pbuf->obj.uin = value;
}

//______________________________________________________________________________
template<> inline void G__setvalue(G__value* pbuf, const int& value)
{
   pbuf->obj.i = value;
} // should be ".in", but too many cases where "int" should really be "long"

//______________________________________________________________________________
template<> inline void G__setvalue(G__value* pbuf, const unsigned long& value)
{
   pbuf->obj.ulo = value;
}

//______________________________________________________________________________
template<> inline void G__setvalue(G__value* pbuf, const long& value)
{
   pbuf->obj.i = value;
}

//______________________________________________________________________________
template<> inline void G__setvalue(G__value* pbuf, const G__uint64& value)
{
   pbuf->obj.ull = value;
}

//______________________________________________________________________________
template<> inline void G__setvalue(G__value* pbuf, const G__int64& value)
{
   pbuf->obj.ll = value;
}

//______________________________________________________________________________
template<> inline void G__setvalue(G__value* pbuf, const float& value)
{
   pbuf->obj.d = value;
}

//______________________________________________________________________________
template<> inline void G__setvalue(G__value* pbuf, const double& value)
{
   pbuf->obj.d = value;
}

//______________________________________________________________________________
template<> inline void G__setvalue(G__value* pbuf, const long double& value)
{
   pbuf->obj.ld = value;
}

//______________________________________________________________________________
template<class T> inline char G__gettypechar()
{
   return 0;
}

//______________________________________________________________________________
template<> inline char G__gettypechar<bool>()
{
   return 'g';
}

//______________________________________________________________________________
template<> inline char G__gettypechar<unsigned char>()
{
   return 'b';
}

//______________________________________________________________________________
template<> inline char G__gettypechar<char>()
{
   return 'c';
}

//______________________________________________________________________________
template<> inline char G__gettypechar<unsigned short>()
{
   return 'r';
}

//______________________________________________________________________________
template<> inline char G__gettypechar<short>()
{
   return 's';
}

//______________________________________________________________________________
template<> inline char G__gettypechar<unsigned int>()
{
   return 'h';
}

//______________________________________________________________________________
template<> inline char G__gettypechar<int>()
{
   return 'i';
}

//______________________________________________________________________________
template<> inline char G__gettypechar<unsigned long>()
{
   return 'k';
}

//______________________________________________________________________________
template<> inline char G__gettypechar<long>()
{
   return 'l';
}

//______________________________________________________________________________
template<> inline char G__gettypechar<G__uint64>()
{
   return 'm';
}

//______________________________________________________________________________
template<> inline char G__gettypechar<G__int64>()
{
   return 'n';
}

//______________________________________________________________________________
template<> inline char G__gettypechar<float>()
{
   return 'f';
}

//______________________________________________________________________________
template<> inline char G__gettypechar<double>()
{
   return 'd';
}

//______________________________________________________________________________
template<> inline char G__gettypechar<long double>()
{
   return 'q';
}

} // namespace Internal
} // namespace Cint

#endif // G__INCLUDE_VALUE_H
