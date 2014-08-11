// JSROOTIO.core.js
//
// core methods for Javascript ROOT IO.
//

var kBase = 0, kOffsetL = 20, kOffsetP = 40, kCounter = 6, kCharStar = 7,
    kChar = 1, kShort = 2, kInt = 3, kLong = 4, kFloat = 5,
    kDouble = 8, kDouble32 = 9, kLegacyChar = 10, kUChar = 11, kUShort = 12,
    kUInt = 13, kULong = 14, kBits = 15, kLong64 = 16, kULong64 = 17, kBool = 18,
    kFloat16 = 19,
    kObject = 61, kAny = 62, kObjectp = 63, kObjectP = 64, kTString = 65,
    kTObject = 66, kTNamed = 67, kAnyp = 68, kAnyP = 69, kAnyPnoVT = 70,
    kSTLp = 71,
    kSkip = 100, kSkipL = 120, kSkipP = 140,
    kConv = 200, kConvL = 220, kConvP = 240,
    kSTL = 300, kSTLstring = 365,
    kStreamer = 500, kStreamLoop = 501;

var kMapOffset = 2;
var kByteCountMask = 0x40000000;
var kNewClassTag = 0xFFFFFFFF;
var kClassMask = 0x80000000;

(function(){

   if (typeof JSROOTIO == "object"){
      var e1 = new Error("JSROOTIO is already defined");
      e1.source = "JSRootIOEvolution.js";
      throw e1;
   }

   var Z_DEFLATED = 8;
   var HDRSIZE = 9;
   var kByteCountMask = 0x40000000;

   JSROOTIO = {};

   JSROOTIO.version = "2.9 2014/05/12";

   JSROOTIO.debug = false;

   JSROOTIO.fUserStreamers = null; // map of user-streamer function like func(buf,obj,prop,streamerinfo)

   JSROOTIO.addUserStreamer = function(type, user_streamer)
   {
      if (this.fUserStreamers == null) this.fUserStreamers = {};
      this.fUserStreamers[type] = user_streamer;
   }

   JSROOTIO.BIT = function(bits, index) {
      var mask = 1 << index;
      return (bits & mask);
   };

   JSROOTIO.R__unzip_header = function(str, off, noalert) {
      // Reads header envelope, and determines target size.

      if (off + HDRSIZE > str.length) {
         if (!noalert) alert("Error R__unzip_header: header size exceeds buffer size");
         return -1;
      }

      /*   C H E C K   H E A D E R   */
      if (!(str.charAt(off) == 'Z' && str.charAt(off+1) == 'L' && str.charCodeAt(off+2) == Z_DEFLATED) &&
          !(str.charAt(off) == 'C' && str.charAt(off+1) == 'S' && str.charCodeAt(off+2) == Z_DEFLATED) &&
          !(str.charAt(off) == 'X' && str.charAt(off+1) == 'Z' && str.charCodeAt(off+2) == 0)) {
         if (!noalert) alert("Error R__unzip_header: error in header");
         return -1;
      }
      return HDRSIZE + ((str.charCodeAt(off+3) & 0xff) |
                       ((str.charCodeAt(off+4) & 0xff) << 8) |
                       ((str.charCodeAt(off+5) & 0xff) << 16));
   };

   JSROOTIO.R__unzip = function(srcsize, str, off, noalert) {

      /*   C H E C K   H E A D E R   */
      if (srcsize < HDRSIZE) {
         if (!noalert) alert("R__unzip: too small source");
         return null;
      }

      /*   C H E C K   H E A D E R   */
      if (!(str.charAt(off) == 'Z' && str.charAt(off+1) == 'L' && str.charCodeAt(off+2) == Z_DEFLATED) &&
          !(str.charAt(off) == 'C' && str.charAt(off+1) == 'S' && str.charCodeAt(off+2) == Z_DEFLATED) &&
          !(str.charAt(off) == 'X' && str.charAt(off+1) == 'Z' && str.charCodeAt(off+2) == 0)) {
         if (!noalert) alert("Error R__unzip: error in header");
         return null;
      }
      var ibufcnt = ((str.charCodeAt(off+3) & 0xff) |
                    ((str.charCodeAt(off+4) & 0xff) << 8) |
                    ((str.charCodeAt(off+5) & 0xff) << 16));
      if (ibufcnt + HDRSIZE != srcsize) {
         if (!noalert) alert("R__unzip: discrepancy in source length");
         return null;
      }

      /*   D E C O M P R E S S   D A T A  */
      if (str.charAt(off) == 'Z' && str.charAt(off+1) == 'L') {
         /* New zlib format */
         var data = str.substr(off + HDRSIZE + 2, srcsize);
         return RawInflate.inflate(data);
      }
      /* Old zlib format */
      else {
         if (!noalert) alert("R__unzip: Old zlib format is not supported!");
         return null;
      }
      return null;
   };

   JSROOTIO.Print = function(str, what) {
      what = typeof(what) === 'undefined' ? 'info' : what;
      if ( (window['console'] !== undefined) ) {
         if (console[what] !== undefined) console[what](str + '\n');
      }
   };

})();

/// JSROOTIO.core.js ends



// JSROOTIO.TBuffer

(function(){

   JSROOTIO.TBuffer = function(_str, _o, _file) {
      if (! (this instanceof arguments.callee) ) {
         var error = new Error("you must use new to instantiate this class");
         error.source = "JSROOTIO.TBuffer.ctor";
         throw error;
      }

      JSROOTIO.TBuffer.prototype.locate = function(pos) {
         this.o = pos;
      }

      JSROOTIO.TBuffer.prototype.shift = function(cnt) {
         this.o += cnt;
      }

      JSROOTIO.TBuffer.prototype.ntou1 = function() {
         return (this.b.charCodeAt(this.o++) & 0xff) >>> 0;
      }

      JSROOTIO.TBuffer.prototype.ntou2 = function() {
         // convert (read) two bytes of buffer b into a UShort_t
         var n  = ((this.b.charCodeAt(this.o)   & 0xff) << 8) >>> 0;
             n +=  (this.b.charCodeAt(this.o+1) & 0xff) >>> 0;
         this.o += 2;
         return n;
      };

      JSROOTIO.TBuffer.prototype.ntou4 = function() {
         // convert (read) four bytes of buffer b into a UInt_t
         var n  = ((this.b.charCodeAt(this.o)   & 0xff) << 24) >>> 0;
             n += ((this.b.charCodeAt(this.o+1) & 0xff) << 16) >>> 0;
             n += ((this.b.charCodeAt(this.o+2) & 0xff) << 8)  >>> 0;
             n +=  (this.b.charCodeAt(this.o+3) & 0xff) >>> 0;
         this.o += 4;
         return n;
      };

      JSROOTIO.TBuffer.prototype.ntou8 = function() {
         // convert (read) eight bytes of buffer b into a ULong_t
         var n  = ((this.b.charCodeAt(this.o)   & 0xff) << 56) >>> 0;
             n += ((this.b.charCodeAt(this.o+1) & 0xff) << 48) >>> 0;
             n += ((this.b.charCodeAt(this.o+2) & 0xff) << 40) >>> 0;
             n += ((this.b.charCodeAt(this.o+3) & 0xff) << 32) >>> 0;
             n += ((this.b.charCodeAt(this.o+4) & 0xff) << 24) >>> 0;
             n += ((this.b.charCodeAt(this.o+5) & 0xff) << 16) >>> 0;
             n += ((this.b.charCodeAt(this.o+6) & 0xff) << 8) >>> 0;
             n +=  (this.b.charCodeAt(this.o+7) & 0xff) >>> 0;
         this.op += 8;
         return n;
      };

      JSROOTIO.TBuffer.prototype.ntoi1 = function() {
         return (this.b.charCodeAt(this.o++) & 0xff);
      }

      JSROOTIO.TBuffer.prototype.ntoi2 = function() {
         // convert (read) two bytes of buffer b into a Short_t
         var n  = (this.b.charCodeAt(this.o)   & 0xff) << 8;
             n += (this.b.charCodeAt(this.o+1) & 0xff);
         this.o += 2;
         return n;
      };

      JSROOTIO.TBuffer.prototype.ntoi4 = function() {
         // convert (read) four bytes of buffer b into a Int_t
         var n  = (this.b.charCodeAt(this.o)   & 0xff) << 24;
             n += (this.b.charCodeAt(this.o+1) & 0xff) << 16;
             n += (this.b.charCodeAt(this.o+2) & 0xff) << 8;
             n += (this.b.charCodeAt(this.o+3) & 0xff);
         this.o += 4;
         return n;
      };

      JSROOTIO.TBuffer.prototype.ntoi8 = function(b, o) {
         // convert (read) eight bytes of buffer b into a Long_t
         var n  = (this.b.charCodeAt(this.o)   & 0xff) << 56;
             n += (this.b.charCodeAt(this.o+1) & 0xff) << 48;
             n += (this.b.charCodeAt(this.o+2) & 0xff) << 40;
             n += (this.b.charCodeAt(this.o+3) & 0xff) << 32;
             n += (this.b.charCodeAt(this.o+4) & 0xff) << 24;
             n += (this.b.charCodeAt(this.o+5) & 0xff) << 16;
             n += (this.b.charCodeAt(this.o+6) & 0xff) << 8;
             n += (this.b.charCodeAt(this.o+7) & 0xff);
         this.o += 8;
         return n;
      };

      JSROOTIO.TBuffer.prototype.ntof = function() {
         // IEEE-754 Floating-Point Conversion (single precision - 32 bits)
         var inString = this.b.substring(this.o, this.o + 4); this.o+=4;
         if (inString.length < 4) return Number.NaN;
         var bits = "";
         for (var i=0; i<4; i++) {
            var curByte = (inString.charCodeAt(i) & 0xff).toString(2);
            var byteLen = curByte.length;
            if (byteLen < 8) {
               for (var bit=0; bit<(8-byteLen); bit++)
                  curByte = '0' + curByte;
            }
            bits = bits + curByte;
         }
         //var bsign = parseInt(bits[0]) ? -1 : 1;
         var bsign = (bits.charAt(0) == '1') ? -1 : 1;
         var bexp = parseInt(bits.substring(1, 9), 2) - 127;
         var bman;
         if (bexp == -127)
            bman = 0;
         else {
            bman = 1;
            for (i=0; i<23; i++) {
               if (parseInt(bits.substr(9+i, 1)) == 1)
                  bman = bman + 1 / Math.pow(2, i+1);
            }
         }
         return (bsign * Math.pow(2, bexp) * bman);
      };

      JSROOTIO.TBuffer.prototype.ntod = function() {
         // IEEE-754 Floating-Point Conversion (double precision - 64 bits)
         var inString = this.b.substring(this.o, this.o + 8); this.o+=8;
         if (inString.length < 8) return Number.NaN;
         var bits = "";
         for (var i=0; i<8; i++) {
            var curByte = (inString.charCodeAt(i) & 0xff).toString(2);
            var byteLen = curByte.length;
            if (byteLen < 8) {
               for (var bit=0; bit<(8-byteLen); bit++)
                  curByte = '0' + curByte;
            }
            bits = bits + curByte;
         }
         //var bsign = parseInt(bits[0]) ? -1 : 1;
         var bsign = (bits.charAt(0) == '1') ? -1 : 1;
         var bexp = parseInt(bits.substring(1, 12), 2) - 1023;
         var bman;
         if (bexp == -127)
            bman = 0;
         else {
            bman = 1;
            for (i=0; i<52; i++) {
               if (parseInt(bits.substr(12+i, 1)) == 1)
                  bman = bman + 1 / Math.pow(2, i+1);
            }
         }
         return (bsign * Math.pow(2, bexp) * bman);
      };


      JSROOTIO.TBuffer.prototype.ReadFastArray = function(n, array_type) {
         // read array of n integers from the I/O buffer
         var array = new Array();
         switch (array_type) {
            case 'D':
               for (var i = 0; i < n; ++i) {
                  array[i] = this.ntod();
                  if (Math.abs(array[i]) < 1e-300) array[i] = 0.0;
               }
               break;
            case 'F':
               for (var i = 0; i < n; ++i) {
                  array[i] = this.ntof();
                  if (Math.abs(array[i]) < 1e-300) array[i] = 0.0;
               }
               break;
            case 'L':
               for (var i = 0; i < n; ++i)
                  array[i] = this.ntoi8();
               break;
            case 'LU':
               for (var i = 0; i < n; ++i)
                  array[i] = this.ntou8();
               break;
            case 'I':
               for (var i = 0; i < n; ++i)
                  array[i] = this.ntoi4();
               break;
            case 'U':
               for (var i = 0; i < n; ++i)
                  array[i] = this.ntou4();
               break;
            case 'S':
               for (var i = 0; i < n; ++i)
                  array[i] = this.ntoi2();
               break;
            case 'C':
               for (var i = 0; i < n; ++i)
                  array[i] = this.b.charCodeAt(this.o++) & 0xff;
               break;
            case 'TString':
               for (var i = 0; i < n; ++i)
                  array[i] = this.ReadTString();
               break;
            default:
               for (var i = 0; i < n; ++i)
                  array[i] = this.ntou4();
            break;
         }
         return array;
      };


      JSROOTIO.TBuffer.prototype.ReadBasicPointer = function(len, array_type) {
         var isArray = this.b.charCodeAt(this.o++) & 0xff;
         if (isArray)
            return this.ReadFastArray(len, array_type);

         if (len==0) return new Array();

         this.o--;
         return this.ReadFastArray(len, array_type);
      };


      JSROOTIO.TBuffer.prototype.ReadString = function(max_len) {
         // stream a string from buffer
         max_len = typeof(max_len) != 'undefined' ? max_len : 0;
         var len = 0;
         var pos0 = this.o;
         while ((max_len==0) || (len<max_len)) {
            if ((this.b.charCodeAt(this.o++) & 0xff) == 0) break;
            len++;
         }

         return (len == 0) ? "" : this.b.substring(pos0, pos0 + len);
      };

      JSROOTIO.TBuffer.prototype.ReadTString = function() {
         // stream a TString object from buffer
         var len = this.b.charCodeAt(this.o++) & 0xff;
         // large strings
         if (len == 255) len = this.ntou4();

         var pos = this.o;
         this.o += len;

         return (this.b.charCodeAt(pos) == 0) ? '' : this.b.substring(pos, pos + len);
      };


      JSROOTIO.TBuffer.prototype.GetMappedObject = function(tag) {
         return this.fObjectMap[tag];
      };

      JSROOTIO.TBuffer.prototype.MapObject = function(tag, obj) {
         if (obj==null) return;
         this.fObjectMap[tag] = obj;
      };

      JSROOTIO.TBuffer.prototype.MapClass = function(tag, classname) {
         this.fClassMap[tag] = classname;
      };

      JSROOTIO.TBuffer.prototype.GetMappedClass = function(tag) {
         if (tag in this.fClassMap) return this.fClassMap[tag];
         return -1;
      };


      JSROOTIO.TBuffer.prototype.ClearObjectMap = function() {
         this.fObjectMap = {};
         this.fClassMap = {};
         this.fObjectMap[0] = null;
      };

      JSROOTIO.TBuffer.prototype.ReadVersion = function() {
         // read class version from I/O buffer
         var version = {};
         var bytecnt = this.ntou4(); // byte count
         if (bytecnt & kByteCountMask)
            version['bytecnt'] = bytecnt - kByteCountMask - 2; // one can check between Read version and end of streamer
         version['val'] = this.ntou2();
         version['off'] = this.o;
         return version;
      };

      JSROOTIO.TBuffer.prototype.CheckBytecount = function(ver, where) {
         if (('bytecnt' in ver) && (ver['off'] + ver['bytecnt'] != this.o)) {
            if (where!=null)
               alert("Missmatch in " + where + " bytecount expected = " + ver['bytecnt'] + "  got = " + (this.o-ver['off']));
            this.o = ver['off'] + ver['bytecnt'];
            return false;
         }
         return true;
      }

      JSROOTIO.TBuffer.prototype.ReadTObject = function(tobj) {
         this.o += 2; // skip version
         if ((!'_typename' in tobj) || (tobj['_typename'] == ''))
            tobj['_typename'] = "JSROOTIO.TObject";

         tobj['fUniqueID'] = this.ntou4();
         tobj['fBits'] = this.ntou4();
         return true;
      }

      JSROOTIO.TBuffer.prototype.ReadTNamed = function(tobj) {
         // read a TNamed class definition from I/O buffer
         var ver = this.ReadVersion();
         this.ReadTObject(tobj);
         tobj['fName'] = this.ReadTString();
         tobj['fTitle'] = this.ReadTString();
         return this.CheckBytecount(ver, "ReadTNamed");
      };

      JSROOTIO.TBuffer.prototype.ReadTObjString = function(tobj) {
         // read a TObjString definition from I/O buffer
         var ver = this.ReadVersion();
         this.ReadTObject(tobj);
         tobj['fString'] = this.ReadTString();
         return this.CheckBytecount(ver, "ReadTObjString");
      };

      JSROOTIO.TBuffer.prototype.ReadTList = function(list) {
         // stream all objects in the list from the I/O buffer
         list['_typename'] = "JSROOTIO.TList";
         list['name'] = "";
         list['arr'] = new Array;
         list['opt'] = new Array;
         var ver = this.ReadVersion();
         if (ver['val'] > 3) {
            this.ReadTObject(list);
            list['name'] = this.ReadTString();
            var nobjects = this.ntou4();
            for (var i = 0; i < nobjects; ++i) {

               var obj = this.ReadObjectAny();
               list['arr'].push(obj);

               var opt = this.ReadTString();
               list['opt'].push(opt);
            }
         }

         return this.CheckBytecount(ver);
      };

      JSROOTIO.TBuffer.prototype.ReadTObjArray = function(list) {
         list['_typename'] = "JSROOTIO.TObjArray";
         list['name'] = "";
         list['arr'] = new Array();
         var ver = this.ReadVersion();
         if (ver['val'] > 2)
            this.ReadTObject(list);
         if (ver['val'] > 1)
            list['name'] = this.ReadTString();
         var nobjects = this.ntou4();
         var lowerbound = this.ntou4();
         for (var i = 0; i < nobjects; i++) {
            var obj = this.ReadObjectAny();
            list['arr'].push(obj);
         }
         return this.CheckBytecount(ver, "ReadTObjArray");
      };

      JSROOTIO.TBuffer.prototype.ReadTClonesArray = function(list) {
         list['_typename'] = "JSROOTIO.TClonesArray";
         list['name'] = "";
         list['arr'] = new Array();
         var ver = this.ReadVersion();
         if (ver['val'] > 2)
            this.ReadTObject(list);
         if (ver['val'] > 1)
            list['name'] = this.ReadTString();
         var s = this.ReadTString();
         var classv = s;
         var clv = 0;
         var pos = s.indexOf(";");
         if (pos != -1) {
            classv = s.slice(0, pos);
            s = s.slice(pos+1, s.length()-pos-1);
            clv = parseInt(s);
         }
         var nobjects = this.ntou4();
         if (nobjects < 0) nobjects = -nobjects;  // for backward compatibility
         var lowerbound = this.ntou4();
         for (var i = 0; i < nobjects; i++) {
            var obj = {};

            this.ClassStreamer(obj, classv);

            list['arr'].push(obj);
         }
         return this.CheckBytecount(ver, "ReadTClonesArray");
      };

      JSROOTIO.TBuffer.prototype.ReadTCollection = function(list, str, o) {
         list['_typename'] = "JSROOTIO.TCollection";
         list['name'] = "";
         list['arr'] = new Array();
         var ver = this.ReadVersion();
         if (ver['val'] > 2)
            this.ReadTObject(list);
         if (ver['val'] > 1)
            list['name'] = this.ReadTString();
         var nobjects = this.ntou4();
         for (var i = 0; i < nobjects; i++) {
            o += 10; // skip object bits & unique id
            list['arr'].push(null);
         }
         return this.CheckBytecount(ver,"ReadTCollection");
      };

      JSROOTIO.TBuffer.prototype.ReadTStreamerInfo = function(streamerinfo) {
         // stream an object of class TStreamerInfo from the I/O buffer

         var R__v = this.ReadVersion();
         if (R__v['val'] > 1) {
            this.ReadTNamed(streamerinfo);
            streamerinfo['name'] = streamerinfo['fName'];
            streamerinfo['title'] = streamerinfo['fTitle'];

            // console.log("name = " + streamerinfo['name']);

            streamerinfo['fCheckSum'] = this.ntou4();
            streamerinfo['fClassVersion'] = this.ntou4();

            streamerinfo['fElements'] = this.ReadObjectAny();
         }
         return this.CheckBytecount(R__v, "ReadTStreamerInfo");
      };

      JSROOTIO.TBuffer.prototype.ReadStreamerElement = function(element) {
         // stream an object of class TStreamerElement

         var R__v = this.ReadVersion();
         this.ReadTNamed(element);
         element['name'] = element['fName']; // TODO - should be removed
         element['title'] = element['fTitle']; // TODO - should be removed
         element['type'] = this.ntou4();
         element['size'] = this.ntou4();
         element['length'] = this.ntou4();
         element['dim'] = this.ntou4();
         if (R__v['val'] == 1) {
            var n = this.ntou4();
            element['maxindex'] = this.ReadFastArray(n, 'U');
         } else {
            element['maxindex'] = this.ReadFastArray(5, 'U');
         }
         element['fTypeName'] = this.ReadTString();
         element['typename'] = element['fTypeName']; // TODO - should be removed
         if ((element['type'] == 11) && (element['typename'] == "Bool_t" ||
              element['typename'] == "bool"))
            element['type'] = 18;
         if (R__v['val'] > 1) {
            element['uuid'] = 0;
         }
         if (R__v['val'] <= 2) {
            // In TStreamerElement v2, fSize was holding the size of
            // the underlying data type.  In later version it contains
            // the full length of the data member.
         }
         if (R__v['val'] == 3) {
            element['xmin'] = this.ntou4();
            element['xmax'] = this.ntou4();
            element['factor'] = this.ntou4();
            //if (element['factor'] > 0) SetBit(kHasRange);
         }
         if (R__v['val'] > 3) {
            //if (TestBit(kHasRange)) GetRange(GetTitle(),fXmin,fXmax,fFactor);
         }
         return this.CheckBytecount(R__v, "ReadStreamerElement");
      };


      JSROOTIO.TBuffer.prototype.ReadStreamerBase = function(streamerbase) {
         // stream an object of class TStreamerBase

         var R__v = this.ReadVersion();
         this.ReadStreamerElement(streamerbase);
         if (R__v['val'] > 2) {
            streamerbase['baseversion'] = this.ntou4();
         }
         return this.CheckBytecount(R__v, "ReadStreamerBase");
      };

      JSROOTIO.TBuffer.prototype.ReadStreamerBasicType = function(streamerbase) {
         // stream an object of class TStreamerBasicType
         var R__v = this.ReadVersion();
         if (R__v['val'] > 1) {
            this.ReadStreamerElement(streamerbase);
         }
         return this.CheckBytecount(R__v, "ReadStreamerBasicType");
      };

      JSROOTIO.TBuffer.prototype.ReadStreamerBasicPointer = function(streamerbase) {
         // stream an object of class TStreamerBasicPointer
         var R__v = this.ReadVersion();
         if (R__v['val'] > 1) {
            this.ReadStreamerElement(streamerbase);
            streamerbase['countversion'] = this.ntou4();
            streamerbase['countName'] = this.ReadTString();
            streamerbase['countClass'] = this.ReadTString();
         }
         return this.CheckBytecount(R__v, "ReadStreamerBasicPointer");
      };

      JSROOTIO.TBuffer.prototype.ReadStreamerSTL = function(streamerSTL) {
         // stream an object of class TStreamerSTL

         var R__v = this.ReadVersion();
         if (R__v['val'] > 2) {
            this.ReadStreamerElement(streamerSTL);
            streamerSTL['stltype'] = this.ntou4();
            streamerSTL['ctype'] = this.ntou4();
         }
         return this.CheckBytecount(R__v, "ReadStreamerSTL");
      };

      JSROOTIO.TBuffer.prototype.ReadTStreamerObject = function(streamerbase) {
         // stream an object of class TStreamerObject
         var R__v = this.ReadVersion();
         if (R__v['val'] > 1) {
            this.ReadStreamerElement(streamerbase);
         }
         return this.CheckBytecount(R__v, "ReadTStreamerObject");
      };


      JSROOTIO.TBuffer.prototype.ReadClass = function() {
         // read class definition from I/O buffer
         var classInfo = {};
         classInfo['name'] = -1;
         var tag = 0;
         var bcnt = this.ntou4();

         var startpos = this.o;
         if (!(bcnt & kByteCountMask) || (bcnt == kNewClassTag)) {
            tag = bcnt;
            bcnt = 0;
         } else {
            // classInfo['fVersion'] = 1;
            tag = this.ntou4();
         }
         if (!(tag & kClassMask)) {
            classInfo['objtag'] = tag; // indicate that we have deal with objects tag
            return classInfo;
         }
         if (tag == kNewClassTag) {
            // got a new class description followed by a new object
            classInfo['name'] = this.ReadString();

            if (this.GetMappedClass(this.fTagOffset + startpos + kMapOffset)==-1)
               this.MapClass(this.fTagOffset + startpos + kMapOffset, classInfo['name']);
         }
         else {
            // got a tag to an already seen class
            var clTag = (tag & ~kClassMask);
            classInfo['name'] = this.GetMappedClass(clTag);

            if (classInfo['name']==-1) {
               alert("Did not found class with tag " + clTag);
            }

         }
         // classInfo['cnt'] = (bcnt & ~kByteCountMask);

         return classInfo;
      };

      JSROOTIO.TBuffer.prototype.ReadObjectAny = function() {
         var startpos = this.o;
         var clRef = this.ReadClass();

         // class identified as object and should be handled so
         if ('objtag' in clRef)
            return this.GetMappedObject(clRef['objtag']);

         if (clRef['name'] == -1) return null;

         var obj = {};

         this.MapObject(this.fTagOffset + startpos + kMapOffset, obj);

         this.ClassStreamer(obj, clRef['name']);

         return obj;
      };

      JSROOTIO.TBuffer.prototype.ClassStreamer = function(obj, classname) {
         //if (!'_typename' in obj)
         //   obj['_typename'] = 'JSROOTIO.' + classname;

         // console.log("Start streaming of class " + classname);

         if (classname == 'TObject' || classname == 'TMethodCall') {
            this.ReadTObject(obj);
         }
         else if (classname == 'TQObject') {
            // skip TQObject
         }
         else if (classname == 'TObjString') {
            this.ReadTObjString(obj);
         }
         else if (classname == 'TObjArray') {
            this.ReadTObjArray(obj);
         }
         else if (classname == 'TClonesArray') {
            this.ReadTClonesArray(obj);
         }
         else if ((classname == 'TList') || (classname == 'THashList')) {
            this.ReadTList(obj);
         }
         else if (classname == 'TCollection') {
            this.ReadTCollection(obj);
            alert("Trying to read TCollection - wrong!!!");
         }
         else if (classname == 'TCanvas') {
            var ver = this.ReadVersion();
            this.ClassStreamer(obj, "TPad");
            // we repair here correct position - no warning to outside
            this.CheckBytecount(ver);
         }
         else if (classname == "TStreamerInfo") {
            this.ReadTStreamerInfo(obj);
         }
         else if (classname == "TStreamerBase") {
            this.ReadStreamerBase(obj);
         }
         else if (classname == "TStreamerBasicType") {
            this.ReadStreamerBasicType(obj);
         }
         else if ((classname == "TStreamerBasicPointer") || (classname == "TStreamerLoop")) {
            this.ReadStreamerBasicPointer(obj);
         } else if (classname == "TStreamerSTL") {
            this.ReadStreamerSTL(obj);
         } else if (classname == "TStreamerObject" ||
                    classname == "TStreamerObjectAny" ||
                    classname == "TStreamerString" ||
                    classname == "TStreamerObjectPointer" ) {
            this.ReadTStreamerObject(obj);
         }
         else {
            var streamer = this.fFile.GetStreamer(classname);
            if (streamer != null)
               streamer.Stream(obj, this);
            else
               console.log("Did not found streamer for class " + classname);
         }

         // TODO: check how typename set
         obj['_typename'] = 'JSROOTIO.' + classname;

         JSROOTCore.addMethods(obj);
      }

      this.b = _str;
      this.o = (_o==null) ? 0 : _o;
      this.fFile = _file;
      this.ClearObjectMap();
      this.fTagOffset = 0;
   }

})();

// JSROOTIO.TStreamer.js
//
// A TStreamer base class.
// Depends on the JSROOTIO library functions.
//

(function(){

   var version = "2.8 2014/03/24";


   // ctor
   JSROOTIO.TStreamer = function(file) {
      if (! (this instanceof arguments.callee) ) {
         var error = new Error("you must use new to instantiate this class");
         error.source = "JSROOTIO.TStreamer.ctor";
         throw error;
      }

      this.fFile = file;
      this._version = version;
      this._typename = "JSROOTIO.TStreamer";

      JSROOTIO.TStreamer.prototype.ReadBasicType = function(buf, obj, prop) {

         // read basic types (known from the streamer info)
         switch (this[prop]['type']) {
            case kBase:
               break;
            case kOffsetL:
               break;
            case kOffsetP:
               break;
            case kCharStar:
               var n_el = obj[this[prop]['cntname']];
               obj[prop] = buf.ReadBasicPointer(n_el, 'C');
               break;
            case kChar:
            case kLegacyChar:
               obj[prop] = buf.b.charCodeAt(buf.o++) & 0xff;
               break;
            case kShort:
               obj[prop] = buf.ntoi2();
               break;
            case kInt:
            case kCounter:
               obj[prop] = buf.ntoi4();
               break;
            case kLong:
               obj[prop] = buf.ntoi8();
               break;
            case kFloat:
            case kDouble32:
               obj[prop] = buf.ntof();
               if (Math.abs(obj[prop]) < 1e-300) obj[prop] = 0.0;
               break;
            case kDouble:
               obj[prop] = buf.ntod();
               if (Math.abs(obj[prop]) < 1e-300) obj[prop] = 0.0;
               break;
            case kUChar:
               obj[prop] = (buf.b.charCodeAt(buf.o++) & 0xff) >>> 0;
               break;
            case kUShort:
               obj[prop] = buf.ntou2();
               break;
            case kUInt:
               obj[prop] = buf.ntou4();
               break;
            case kULong:
               obj[prop] = buf.ntou8();
               break;
            case kBits:
               alert('failed to stream ' + prop + ' (' + this[prop]['typename'] + ')');
               break;
            case kLong64:
               obj[prop] = buf.ntoi8();
               break;
            case kULong64:
               obj[prop] = buf.ntou8();
               break;
            case kBool:
               obj[prop] = (buf.b.charCodeAt(buf.o++) & 0xff) != 0;
               break;
            case kFloat16:
               obj[prop] = 0;
               buf.o += 2;
               break;
            case kAny:
            case kAnyp:
            case kObjectp:
            case kObject:
               var classname = this[prop]['typename'];
               if (classname.endsWith("*"))
                  classname = classname.substr(0, classname.length - 1);

               obj[prop] = {};
               buf.ClassStreamer(obj[prop], classname);
               break;

            case kAnyP:
            case kObjectP:
               obj[prop] = buf.ReadObjectAny();
               break;
            case kTString:
               obj[prop] = buf.ReadTString();
               break;
            case kTObject:
               buf.ReadTObject(obj);
               break;
            case kTNamed:
               buf.ReadTNamed(obj);
               break;
            case kAnyPnoVT:
            case kSTLp:
            case kSkip:
            case kSkipL:
            case kSkipP:
            case kConv:
            case kConvL:
            case kConvP:
            case kSTL:
            case kSTLstring:
            case kStreamer:
            case kStreamLoop:
               alert('failed to stream ' + prop + ' (' + this[prop]['typename'] + ')');
               break;
            case kOffsetL+kShort:
            case kOffsetL+kUShort:
               alert("Strange code was here????"); // var n_el = str.charCodeAt(o) & 0xff;
               var n_el  = this[prop]['length'];
               obj[prop] = buf.ReadFastArray(n_el, 'S');
               break;
            case kOffsetL+kInt:
               var n_el  = this[prop]['length'];
               obj[prop] = buf.ReadFastArray(n_el, 'I');
               break;
            case kOffsetL+kUInt:
               var n_el  = this[prop]['length'];
               obj[prop] = buf.ReadFastArray(n_el, 'U');
               break;
            case kOffsetL+kULong:
            case kOffsetL+kULong64:
               var n_el  = this[prop]['length'];
               obj[prop] = buf.ReadFastArray(n_el, 'LU');
               break;
            case kOffsetL+kLong:
            case kOffsetL+kLong64:
               var n_el  = this[prop]['length'];
               obj[prop] = buf.ReadFastArray(n_el, 'L');
               break;
            case kOffsetL+kFloat:
            case kOffsetL+kDouble32:
               //var n_el = str.charCodeAt(o) & 0xff;
               var n_el  = this[prop]['length'];
               obj[prop] = buf.ReadFastArray(n_el, 'F');
               break;
            case kOffsetL+kDouble:
               //var n_el = str.charCodeAt(o) & 0xff;
               var n_el  = this[prop]['length'];
               obj[prop] = buf.ReadFastArray(n_el, 'D');
               break;
            case kOffsetP+kChar:
               var n_el = obj[this[prop]['cntname']];
               obj[prop] = buf.ReadBasicPointer(n_el, 'C');
               break;
            case kOffsetP+kShort:
            case kOffsetP+kUShort:
               var n_el = obj[this[prop]['cntname']];
               obj[prop] = buf.ReadBasicPointer(n_el, 'S');
               break;
            case kOffsetP+kInt:
               var n_el = obj[this[prop]['cntname']];
               obj[prop] = buf.ReadBasicPointer(n_el, 'I');
               break;
            case kOffsetP+kUInt:
               var n_el = obj[this[prop]['cntname']];
               obj[prop] = buf.ReadBasicPointer(n_el, 'U');
               break;
            case kOffsetP+kULong:
            case kOffsetP+kULong64:
               var n_el = obj[this[prop]['cntname']];
               obj[prop] = buf.ReadBasicPointer(n_el, 'LU');
               break;
            case kOffsetP+kLong:
            case kOffsetP+kLong64:
               var n_el = obj[this[prop]['cntname']];
               obj[prop] = buf.ReadBasicPointer(n_el, 'L');
               break;
            case kOffsetP+kFloat:
            case kOffsetP+kDouble32:
               var n_el = obj[this[prop]['cntname']];
               obj[prop] = buf.ReadBasicPointer(n_el, 'F');
               break;
            case kOffsetP+kDouble:
               var n_el = obj[this[prop]['cntname']];
               obj[prop] = buf.ReadBasicPointer(n_el, 'D');
               break;
            default:
               alert('failed to stream ' + prop + ' (' + this[prop]['typename'] + ')');
               break;
         }
      };

      JSROOTIO.TStreamer.prototype.Stream = function(obj, buf) {

         var ver = buf.ReadVersion();

         // first base classes
         for (var prop in this) {
            if (!this[prop] || typeof(this[prop]) === "function")
               continue;
            if (this[prop]['typename'] === 'BASE') {
               var clname = this[prop]['class'];
               if (this[prop]['class'].indexOf("TArray") == 0) {
                  var array_type = this[prop]['class'].charAt(6);
                  obj['fN'] = buf.ntou4();
                  obj['fArray'] = buf.ReadFastArray(obj['fN'], array_type);
               } else {
                  buf.ClassStreamer(obj, this[prop]['class']);
               }
            }
         }
         // then class members
         for (var prop in this) {

            if (!this[prop] || typeof(this[prop]) === "function") continue;

            var prop_typename = this[prop]['typename'];

            if (typeof(prop_typename) === "undefined" || prop_typename === "BASE") continue;

            if (JSROOTIO.fUserStreamers !== null) {
               var user_func = JSROOTIO.fUserStreamers[prop_typename];

               if (user_func !== undefined) {
                  user_func(buf, obj, prop, this);
                  continue;
               }
            }

            // special classes (custom streamers)
            switch (prop_typename) {
               case "TString*":
                  // TODO: check how and when it used
                  var r__v = buf.ReadVersion();
                  obj[prop] = new Array();
                  for (var i = 0; i<obj[this[prop]['cntname']]; ++i )
                     obj[prop][i] = buf.ReadTString();
                  buf.CheckBytecount(r__v, "TString* array");
                  break;
               case "TArrayC":
               case "TArrayD":
               case "TArrayF":
               case "TArrayI":
               case "TArrayL":
               case "TArrayL64":
               case "TArrayS":
                  var array_type = this[prop]['typename'].charAt(6);
                  var n = buf.ntou4();
                  obj[prop] = buf.ReadFastArray(n, array_type);
                  break;
               case "TObject":
                  // TODO: check why it is here
                  buf.ReadTObject(obj);
                  break;
               case "TQObject":
                  // TODO: check why it is here
                  // skip TQObject...
                  break;
               default:
                  // basic types and standard streamers
                  this.ReadBasicType(buf, obj, prop);
                  break;
            }
         }
         if (('fBits' in obj) && !('TestBit' in obj)) {
            obj['TestBit'] = function (f) {
               return ((obj['fBits'] & f) != 0);
            };
         }

         buf.CheckBytecount(ver, "TStreamer.Stream");

         return buf.o;

      };

      return this;
   };

   JSROOTIO.TStreamer.Version = version;

})();

// JSROOTIO.TStreamer.js ends


// JSROOTIO.TDirectory.js
//
// A class that reads a TDirectory from a buffer.
// Depends on the JSROOTIO library functions.
//

(function(){

   var version = "2.8 2014/03/18";

   // ctor
   JSROOTIO.TDirectory = function(file, dirname, cycle) {
      if (! (this instanceof arguments.callee) ) {
         var error = new Error("you must use new to instantiate this class");
         error.source = "JSROOTIO.TDirectory.ctor";
         throw error;
      }

      this.fFile = file;
      this._version = version;
      this._typename = "JSROOTIO.TDirectory";
      this['dirname'] = dirname;
      this['cycle'] = cycle;

      JSROOTIO.TDirectory.prototype.GetKey = function(keyname, cycle) {
         // retrieve a key by its name and cycle in the list of keys
         for (var i=0; i<this.fKeys.length; ++i) {
            if (this.fKeys[i]['name'] == keyname && this.fKeys[i]['cycle'] == cycle)
               return this.fKeys[i];
         }
         return null;
      }


      JSROOTIO.TDirectory.prototype.ReadKeys = function(cycle, dir_id) {

         var thisdir = this;

         var callback2 = function(file, _buffer) {
            //headerkey->ReadKeyBuffer(buffer);
            var buf = new JSROOTIO.TBuffer(_buffer, 0, thisdir.fFile);

            var key = thisdir.fFile.ReadKey(buf);

            var nkeys = buf.ntoi4();
            for (var i = 0; i < nkeys; i++) {
               key = thisdir.fFile.ReadKey(buf);
               thisdir.fKeys.push(key);
            }
            thisdir.fFile.fDirectories.push(thisdir);

            JSROOTPainter.displayListOfKeys(thisdir.fKeys, '#status', dir_id);
            delete buf;
         };

         var callback1 = function(file, buffer) {
            var buf = new JSROOTIO.TBuffer(buffer, thisdir.fNbytesName, thisdir.fFile);

            thisdir.StreamHeader(buf);

            //*-*---------read TKey::FillBuffer info
            buf.locate(4); // Skip NBytes;
            var keyversion = buf.ntoi2();
            // Skip ObjLen, DateTime, KeyLen, Cycle, SeekKey, SeekPdir
            if (keyversion > 1000) buf.shift(28); // Large files
                              else buf.shift(20);
            buf.ReadTString();
            buf.ReadTString();
            thisdir.fTitle = buf.ReadTString();
            if (thisdir.fNbytesName < 10 || thisdir.fNbytesName > 10000) {
               throw "Init : cannot read directory info of file " + thisdir.fURL;
            }
            //*-* -------------Read keys of the top directory

            if ( thisdir.fSeekKeys >  0) {
               thisdir.fFile.Seek(thisdir.fSeekKeys, thisdir.fFile.ERelativeTo.kBeg);
               thisdir.fFile.ReadBuffer(thisdir.fNbytesKeys, callback2);
            }
         };

         //*-*-------------Read directory info
         var nbytes = this.fNbytesName + 22;
         nbytes += 4;  // fDatimeC.Sizeof();
         nbytes += 4;  // fDatimeM.Sizeof();
         nbytes += 18; // fUUID.Sizeof();
         // assume that the file may be above 2 Gbytes if file version is > 4
         if (this.fFile.fVersion >= 40000) nbytes += 12;

         this.fFile.Seek(this.fSeekDir, this.fFile.ERelativeTo.kBeg);
         this.fFile.ReadBuffer(nbytes, callback1);
      };

      JSROOTIO.TDirectory.prototype.StreamHeader = function(buf) {
         var version = buf.ntou2();
         var versiondir = version%1000;
         buf.shift(8); // skip fDatimeC and fDatimeM ReadBuffer()
         this.fNbytesKeys = buf.ntou4();
         this.fNbytesName = buf.ntou4();
         this.fSeekDir = (version > 1000) ? buf.ntou8() : buf.ntou4();
         this.fSeekParent = (version > 1000) ? buf.ntou8() : buf.ntou4();
         this.fSeekKeys = (version > 1000) ? buf.ntou8() : buf.ntou4();
         if (versiondir > 2) buf.shift(18); // skip fUUID.ReadBuffer(buffer);
      };

      this.fKeys = new Array();
      return this;
   };

   JSROOTIO.TDirectory.Version = version;

})();

// JSROOTIO.TDirectory.js ends

// JSROOTIO.RootFile.js
//
// A class that reads ROOT files.
// Depends on the JSROOTIO library functions.
//
////////////////////////////////////////////////////////////////////////////////
// A ROOT file is a suite of consecutive data records (TKey's) with
// the following format (see also the TKey class). If the key is
// located past the 32 bit file limit (> 2 GB) then some fields will
// be 8 instead of 4 bytes:
//    1->4            Nbytes    = Length of compressed object (in bytes)
//    5->6            Version   = TKey version identifier
//    7->10           ObjLen    = Length of uncompressed object
//    11->14          Datime    = Date and time when object was written to file
//    15->16          KeyLen    = Length of the key structure (in bytes)
//    17->18          Cycle     = Cycle of key
//    19->22 [19->26] SeekKey   = Pointer to record itself (consistency check)
//    23->26 [27->34] SeekPdir  = Pointer to directory header
//    27->27 [35->35] lname     = Number of bytes in the class name
//    28->.. [36->..] ClassName = Object Class Name
//    ..->..          lname     = Number of bytes in the object name
//    ..->..          Name      = lName bytes with the name of the object
//    ..->..          lTitle    = Number of bytes in the object title
//    ..->..          Title     = Title of the object
//    ----->          DATA      = Data bytes associated to the object
//

(function(){

   var version = "1.8 2013/07/03";

   if (typeof JSROOTCore != "object") {
      var e1 = new Error("This extension requires JSROOTCore.js");
      e1.source = "JSROOTIO.RootFile.js";
      throw e1;
   }

   if (typeof JSROOTIO != "object") {
      var e1 = new Error("This extension requires JSROOTIO.core.js");
      e1.source = "JSROOTIO.RootFile.js";
      throw e1;
   }

   // ctor
   JSROOTIO.RootFile = function(url) {
      if (! (this instanceof arguments.callee) ) {
         var error = new Error("you must use new to instantiate this class");
         error.source = "JSROOTIO.RootFile.ctor";
         throw error;
      }

      this._version = version;
      this._typename = "JSROOTIO.RootFile";
      this.fOffset = 0;
      this.fArchiveOffset = 0;
      this.fEND = 0;
      this.fURL = url;
      this.fLogMsg = "";
      this.fAcceptRanges = true;
      this.fFullFileContent = ""; // this will full content of the file

      this.ERelativeTo = {
         kBeg : 0,
         kCur : 1,
         kEnd : 2
      };

      JSROOTIO.RootFile.prototype.GetSize = function(url) {
         // Return maximum file size.
         var xhr = new XMLHttpRequest();
         xhr.open('HEAD', url+"?"+-1, false);
         xhr.send(null);
         if (xhr.status == 200 || xhr.status == 0) {
            var header = xhr.getResponseHeader("Content-Length");
            var accept_ranges = xhr.getResponseHeader("Accept-Ranges");
            if (!accept_ranges) this.fAcceptRanges = false;
            return parseInt(header);
         }

         xhr = null;
         return -1;
      }

      JSROOTIO.RootFile.prototype.ReadBuffer = function(len, callback) {

         // Read specified byte range from remote file
         var ie9 = function(url, pos, len, file) {
            // IE9 Fallback
            var xhr = new XMLHttpRequest();
            xhr.onreadystatechange = function() {
               if (this.readyState == 4 && (this.status == 200 || this.status == 206)) {
                  var filecontent = new String("");
                  var array = new VBArray(this.responseBody).toArray();
                  for (var i = 0; i < array.length; i++) {
                     filecontent = filecontent + String.fromCharCode(array[i]);
                  }

                  if (!file.fAcceptRanges && (filecontent.length != len) &&
                      (file.fEND == filecontent.length)) {
                     // $('#report').append("<br> seems to be, we get full file");
                     file.fFullFileContent = filecontent;
                     filecontent = file.fFullFileContent.substr(pos, len);
                  }

                  callback(file, filecontent); // Call callback func with data
                  delete filecontent;
               }
               else if (this.readyState == 4 && this.status == 404) {
                  alert("Error 404: File not found!");
               }
            }
            xhr.open('GET', url, true);
            var xhr_header = "bytes=" + pos + "-" + (pos + len);
            xhr.setRequestHeader("Range", xhr_header);
            xhr.setRequestHeader("If-Modified-Since", "Wed, 31 Dec 1980 00:00:00 GMT");
            xhr.send(null);
            xhr = null;
         }
         var other = function(url, pos, len, file) {
            var xhr = new XMLHttpRequest();
            xhr.onreadystatechange = function() {
               if (this.readyState == 4 && (this.status == 0 || this.status == 200 ||
                   this.status == 206)) {
                  var HasArrayBuffer = ('ArrayBuffer' in window && 'Uint8Array' in window);
                  if (HasArrayBuffer && 'mozResponse' in this) {
                     Buf = this.mozResponse;
                  } else if (HasArrayBuffer && this.mozResponseArrayBuffer) {
                     Buf = this.mozResponseArrayBuffer;
                  } else if ('responseType' in this) {
                     Buf = this.response;
                  } else {
                     Buf = this.responseText;
                     HasArrayBuffer = false;
                  }
                  if (HasArrayBuffer) {
                     var filecontent = new String("");
                     var bLen = Buf.byteLength;
                     var u8Arr = new Uint8Array(Buf, 0, bLen);
                     for (var i = 0; i < u8Arr.length; i++) {
                        filecontent = filecontent + String.fromCharCode(u8Arr[i]);
                     }
                  } else {
                     var filecontent = Buf;
                  }

                  if (!file.fAcceptRanges && (filecontent.length != len) &&
                      (file.fEND == filecontent.length)) {
                    file.fFullFileContent = filecontent;
                    filecontent = file.fFullFileContent.substr(pos, len);
                  }

                  callback(file, filecontent); // Call callback func with data
                  delete filecontent;
               }
            };
            xhr.open('GET', url, true);
            var xhr_header = "bytes=" + pos + "-" + (pos + len);
            xhr.setRequestHeader("Range", xhr_header);
            // next few lines are there to make Safari working with byte ranges...
            xhr.setRequestHeader("If-Modified-Since", "Wed, 31 Dec 1980 00:00:00 GMT");

            var HasArrayBuffer = ('ArrayBuffer' in window && 'Uint8Array' in window);
            if (HasArrayBuffer && 'mozResponseType' in xhr) {
               xhr.mozResponseType = 'arraybuffer';
            } else if (HasArrayBuffer && 'responseType' in xhr) {
               xhr.responseType = 'arraybuffer';
            } else {
               //XHR binary charset opt by Marcus Granado 2006 [http://mgran.blogspot.com]
               xhr.overrideMimeType("text/plain; charset=x-user-defined");
            }
            xhr.send(null);
            xhr = null;
         }

         if (!this.fAcceptRanges && (this.fFullFileContent.length>0)) {
            var res = this.fFullFileContent.substr(this.fOffset,len);
            callback(this, res);
         }
         else
         // Multi-browser support
         if (typeof ActiveXObject == "function")
            return ie9(this.fURL, this.fOffset, len, this);
         else
            return other(this.fURL, this.fOffset, len, this);
      };

      JSROOTIO.RootFile.prototype.Seek = function(offset, pos) {
         // Set position from where to start reading.
         switch (pos) {
            case this.ERelativeTo.kBeg:
               this.fOffset = offset + this.fArchiveOffset;
               break;
            case this.ERelativeTo.kCur:
               this.fOffset += offset;
               break;
            case this.ERelativeTo.kEnd:
               // this option is not used currently in the ROOT code
               if (this.fArchiveOffset)
                  throw  "Seek : seeking from end in archive is not (yet) supported";
               this.fOffset = this.fEND - offset;  // is fEND really EOF or logical EOF?
               break;
            default:
               throw  "Seek : unknown seek option (" + pos + ")";
               break;
         }
      };

      JSROOTIO.RootFile.prototype.Log = function(s, i) {
         // format html log information
         if (!i) i = '';
         for (var e in s) {
            if (s[e] != null && typeof(s[e]) == 'object') {
               this.fLogMsg += i + e + ':<br>\n';
               this.fLogMsg += '<ul type="circle">\n';
               this.Log(s[e], '<li>');
            }
            else {
               if ((i == '<li>') || (i == '<li> '))
                  this.fLogMsg += i + e + ' = ' + s[e] + '</li>\n';
               else
                  this.fLogMsg += i + e + ' = ' + s[e] + '<br>\n';
            }
         }
         if (i == '<li>') this.fLogMsg += '</ul>\n';
      };

      JSROOTIO.RootFile.prototype.ReadHeader = function(str) {
         // read the Root header file informations
         if (str.substring(0, 4) != "root") {
            alert("NOT A ROOT FILE!");
            return null;
         }
         var header = {};

         var buf = new JSROOTIO.TBuffer(str, 4, this); // skip root
         header['version'] = buf.ntou4();
         header['begin'] = buf.ntou4();
         var largeFile = header['version'] >= 1000000;
         header['end'] = largeFile ? buf.ntou8() : buf.ntou4();
         header['seekFree'] = largeFile ? buf.ntou8() : buf.ntou4();
         buf.shift(12); // skip fNBytesFree, nfree, fNBytesName
         header['units'] = buf.ntoi1();
         header['fCompress'] = buf.ntou4();
         header['seekInfo'] = largeFile ? buf.ntou8() : buf.ntou4();
         header['nbytesInfo'] = buf.ntou4();

         if (!header['seekInfo'] && !header['nbytesInfo']) {
            // empty file
            return null;
         }
         this.fSeekInfo = header['seekInfo'];
         this.fNbytesInfo = header['nbytesInfo'];
         this.Log(header);
         return header;
      };

      JSROOTIO.RootFile.prototype.ReadKey = function(buf) {
         // read key from buffer
         var key = {};
         key['offset'] = buf.o;
         var nbytes = buf.ntoi4();
         key['nbytes'] = Math.abs(nbytes);
         var largeKey = buf.o + nbytes > 2 * 1024 * 1024 * 1024 /*2G*/;

         buf.shift(2);

         key['objLen'] = buf.ntou4();
         var datime = buf.ntou4();
         key['datime'] = {
            year : (datime >>> 26) + 1995,
            month : (datime << 6) >>> 28,
            day : (datime << 10) >>> 27,
            hour : (datime << 15) >>> 27,
            min : (datime << 20) >>> 26,
            sec : (datime << 26) >>> 26
         };
         key['keyLen'] = buf.ntou2();
         key['cycle'] = buf.ntou2();
         if (largeKey) {
            key['seekKey'] = buf.ntou8();
            buf.shift(8); // skip seekPdir
         } else {
            key['seekKey'] = buf.ntou4();
            buf.shift(4); // skip seekPdir
         }
         key['className'] = buf.ReadTString();
         key['name'] = buf.ReadTString();
         key['title'] = buf.ReadTString();
         key['dataoffset'] = key['seekKey'] + key['keyLen'];
         key['name'] = key['name'].replace(/['"]/g,''); // get rid of quotes
         // should we do it here ???
         //buf.locate(key['offset'] + key['keyLen']);

         // remember offset
         if (key['className'] != "" && key['name'] != "")
            key['offset'] = buf.o;

         return key;
      };

      JSROOTIO.RootFile.prototype.GetDir = function(dirname) {
         for (var j=0; j<this.fDirectories.length;++j) {
            if (this.fDirectories[j]['dirname'] == dirname)
               return this.fDirectories[j];
         }
         return null;
      }

      JSROOTIO.RootFile.prototype.GetKey = function(keyname, cycle) {
         // retrieve a key by its name and cycle in the list of keys
         for (var i=0; i<this.fKeys.length; ++i) {
            if (this.fKeys[i]['name'] == keyname && this.fKeys[i]['cycle'] == cycle)
               return this.fKeys[i];
         }

         var n = keyname.lastIndexOf("/");
         if (n<=0) return null;

         var dir = this.GetDir(keyname.substr(0, n));
         if (dir==null) return null;

         return dir.GetKey(keyname.substr(n+1), cycle);
      };

      JSROOTIO.RootFile.prototype.ReadObjBuffer = function(key, callback) {
         // read and inflate object buffer described by its key
         var callback1 = function(file, buffer) {
            var buf = null;

            if (key['objLen'] <= key['nbytes']-key['keyLen']) {
               buf = new JSROOTIO.TBuffer(buffer, 0, file);
            } else {
               var hdrsize = JSROOTIO.R__unzip_header(buffer, 0);
               if (hdrsize<0) return;
               var objbuf = JSROOTIO.R__unzip(hdrsize, buffer, 0);
               buf = new JSROOTIO.TBuffer(objbuf, 0, file);
            }

            buf.fTagOffset = key.keyLen;
            callback(file, buf);
            delete buf;
         };

         this.Seek(key['dataoffset'], this.ERelativeTo.kBeg);
         this.ReadBuffer(key['nbytes'] - key['keyLen'], callback1);
      };

      JSROOTIO.RootFile.prototype.ReadObject = function(obj_name, cycle, node_id) {
         // read any object from a root file

         if (findObject(obj_name+cycle)) return;
         var key = this.GetKey(obj_name, cycle);
         if (key == null) return;

         var callback = function(file, buf) {
            if (!buf) return;
            var obj = {};
            obj['_typename'] = 'JSROOTIO.' + key['className'];

            buf.MapObject(1, obj); // workaround - tag first object with id1
            buf.ClassStreamer(obj, key['className']);

            if (key['className'] == 'TFormula') {
               JSROOTCore.addFormula(obj);
            }
            else if (key['className'] == 'TNtuple' || key['className'] == 'TTree') {
               displayTree(obj, cycle, node_id);
            }
            else if (key['className'] == 'TList' || key['className'] == 'TObjArray' || key['className'] == 'TClonesArray') {
               displayCollection(obj_name, cycle, node_id, obj);
               obj_list.push(obj_name+cycle);
               obj_index++;
            }
            else {
               if (obj['fName'] == "") obj['fName'] = obj_name;
               displayObject(obj, cycle, obj_index);
               obj_list.push(obj_name+cycle);
               obj_index++;
            }
         };

         this.ReadObjBuffer(key, callback);
      };

      JSROOTIO.RootFile.prototype.ExtractStreamerInfos = function(buf)
      {
         if (!buf) return;

         var lst = {};
         lst['_typename'] = "JSROOTIO.TList";

         buf.MapObject(1, lst);
         buf.ClassStreamer(lst, 'TList');

         for (var i=0;i<lst['arr'].length;i++) {
            this.fStreamerInfos[lst.arr[i].name] = lst.arr[i];
         }

         delete lst;
      }

      JSROOTIO.RootFile.prototype.ReadStreamerInfos = function() {

         if (this.fSeekInfo == 0 || this.fNbytesInfo == 0) return;
         this.Seek(this.fSeekInfo, this.ERelativeTo.kBeg);
         var callback1 = function(file, _buffer) {
            var buf = new JSROOTIO.TBuffer(_buffer, 0, file);
            var key = file.ReadKey(buf);
            if (key == null) return;
            file.fKeys.push(key);
            var callback2 = function(file, buf) {

               file.ExtractStreamerInfos(buf);

               for (i=0;i<file.fKeys.length;++i) {
                  if (file.fKeys[i]['className'] == 'TFormula') {
                     file.ReadObject(file.fKeys[i]['name'], file.fKeys[i]['cycle']);
                  }
               }
               if (typeof(userCallback) == 'function')
                  userCallback(file);
            };
            file.ReadObjBuffer(key, callback2);
            
            JSROOTPainter.displayListOfKeys(file.fKeys, '#status');
         };
         this.ReadBuffer(this.fNbytesInfo, callback1);
      };

      JSROOTIO.RootFile.prototype.ReadKeys = function() {
         // read keys only in the root file

         var callback1 = function(file, buffer) {
            var header = file.ReadHeader(buffer);
            if (header == null) {
               delete buffer;
               buffer = null;
               return;
            }

            var callback2 = function(file, str) {

               var buf = new JSROOTIO.TBuffer(str, 4, file); // skip the "root" file identifier
               file.fVersion = buf.ntou4();
               var headerLength = buf.ntou4();
               file.fBEGIN = headerLength;
               if (file.fVersion < 1000000) { //small file
                  file.fEND = buf.ntou4();
                  file.fSeekFree = buf.ntou4();
                  file.fNbytesFree = buf.ntou4();
                  var nfree = buf.ntoi4();
                  file.fNbytesName = buf.ntou4();
                  file.fUnits = buf.ntou1();
                  file.fCompress = buf.ntou4();
                  file.fSeekInfo = buf.ntou4();
                  file.fNbytesInfo = buf.ntou4();
               } else { // new format to support large files
                  file.fEND = buf.ntou8();
                  file.fSeekFree = buf.ntou8();
                  file.fNbytesFree = buf.ntou4();
                  var nfree = buf.ntou4();
                  file.fNbytesName = buf.ntou4();
                  file.fUnits = buf.ntou1();
                  file.fCompress = buf.ntou4();
                  file.fSeekInfo = buf.ntou8();
                  file.fNbytesInfo = buf.ntou4();
               }
               file.fSeekDir = file.fBEGIN;

               //*-*-------------Read directory info

               var nbytes = file.fNbytesName + 22;
               nbytes += 4;  // fDatimeC.Sizeof();
               nbytes += 4;  // fDatimeM.Sizeof();
               nbytes += 18; // fUUID.Sizeof();
               // assume that the file may be above 2 Gbytes if file version is > 4
               if (file.fVersion >= 40000) nbytes += 12;

               file.Seek(file.fBEGIN, file.ERelativeTo.kBeg);

               var callback3 = function(file, str) {

                  var buf = new JSROOTIO.TBuffer(str, file.fNbytesName, file);

                  var version = buf.ntou2();
                  var versiondir = version%1000;
                  buf.shift(8); // skip fDatimeC and fDatimeM ReadBuffer()
                  file.fNbytesKeys = buf.ntou4();
                  file.fNbytesName = buf.ntou4();
                  if (version > 1000) {
                     file.fSeekDir = buf.ntou8();
                     file.fSeekParent = buf.ntou8();
                     file.fSeekKeys = buf.ntou8();
                  } else {
                     file.fSeekDir = buf.ntou4();
                     file.fSeekParent = buf.ntou4();
                     file.fSeekKeys = buf.ntou4();
                  }
                  if (versiondir > 1) buf.o += 18; // skip fUUID.ReadBuffer(buffer);

                  //*-*---------read TKey::FillBuffer info
                  buf.o = 4; // Skip NBytes;
                  var keyversion = buf.ntoi2();
                  // Skip ObjLen, DateTime, KeyLen, Cycle, SeekKey, SeekPdir
                  if (keyversion > 1000) buf.shift(28); // Large files
                                    else buf.shift(20);
                  buf.ReadTString();
                  buf.ReadTString();
                  file.fTitle = buf.ReadTString();
                  if (file.fNbytesName < 10 || this.fNbytesName > 10000) {
                     throw "Init : cannot read directory info of file " + file.fURL;
                  }
                  //*-* -------------Read keys of the top directory

                  if ( file.fSeekKeys >  0) {
                     file.Seek(file.fSeekKeys, file.ERelativeTo.kBeg);

                     var callback4 = function(file, _buffer) {
                        //headerkey->ReadKeyBuffer(buffer);

                        var buf = new JSROOTIO.TBuffer(_buffer, 0, file);

                        var key = file.ReadKey(buf);

                        var nkeys = buf.ntoi4();
                        for (var i = 0; i < nkeys; i++) {
                           key = file.ReadKey(buf);
                           file.fKeys.push(key);
                        }
                        file.ReadStreamerInfos();
                        delete buf;
                     };
                     file.ReadBuffer(file.fNbytesKeys, callback4);
                  }
                  delete str;
                  str = null;
               };
               file.ReadBuffer(Math.max(300, nbytes), callback3);
               delete str;
               str = null;
            };
            file.ReadBuffer(300, callback2);
            delete buffer;
            buffer = null;
         };
         this.ReadBuffer(256, callback1);
      };

      JSROOTIO.RootFile.prototype.ReadDirectory = function(dir_name, cycle, dir_id) {
         // read the directory content from  a root file
         // do not read directory if it is already exists

         var dir = this.GetDir(dir_name);
         if (dir!=null) return;

         var key = this.GetKey(dir_name, cycle);
         if (key == null) return null;

         var callback = function(file, buf) {
            if (!buf) return;

            var directory = new JSROOTIO.TDirectory(file, dir_name, cycle);
            directory.StreamHeader(buf);
            if (directory.fSeekKeys) directory.ReadKeys(cycle, dir_id);
         };
         this.ReadObjBuffer(key, callback);
      };

      JSROOTIO.RootFile.prototype.Init = function(fileurl) {
         // init members of a Root file from given url
         this.fURL = fileurl;
         this.fLogMsg = "";
         if (fileurl) {
            this.fEND = this.GetSize(fileurl);
         }
      };

      JSROOTIO.RootFile.prototype.GetStreamer = function(clname) {
         // return the streamer for the class 'clname', from the list of streamers
         // or generate it from the streamer infos and add it to the list

         var streamer = this.fStreamers[clname];
         if (typeof(streamer) != 'undefined') return streamer;

         var s_i = this.fStreamerInfos[clname];
         if (typeof(s_i) === 'undefined') return null;

         this.fStreamers[clname] = new JSROOTIO.TStreamer(this);
         if (typeof(s_i['fElements']) != 'undefined') {
            var n_el = s_i['fElements']['arr'].length;
            for (var j=0;j<n_el;++j) {
               var element = s_i['fElements']['arr'][j];
               if (element['typename'] === 'BASE') {
                  // generate streamer for the base classes
                  this.GetStreamer(element['name']);
               }
            }
         }
         if (typeof(s_i['fElements']) != 'undefined') {
            var n_el = s_i['fElements']['arr'].length;
            for (var j=0;j<n_el;++j) {
               // extract streamer info for each class member
               var element = s_i['fElements']['arr'][j];
               var streamer = {};
               streamer['typename'] = element['typename'];
               streamer['class']    = element['name'];
               streamer['cntname']  = s_i['fElements']['arr'][j]['countName'];
               streamer['type']     = element['type'];
               streamer['length']   = element['length'];

               this.fStreamers[clname][element['name']] = streamer;
            }
         }
         return this.fStreamers[clname];
      };

      JSROOTIO.RootFile.prototype.Delete = function() {
         if (this.fDirectories) this.fDirectories.splice(0, this.fDirectories.length);
         this.fDirectories = null;
         if (this.fKeys) this.fKeys.splice(0, this.fKeys.length);
         this.fKeys = null;
         if (this.fStreamers) this.fStreamers.splice(0, this.fStreamers.length);
         this.fStreamers = null;
         this.fSeekInfo = 0;
         this.fNbytesInfo = 0;
         this.fTagOffset = 0;
      };

      this.fDirectories = new Array();
      this.fKeys = new Array();
      this.fSeekInfo = 0;
      this.fNbytesInfo = 0;
      this.fTagOffset = 0;
      this.fStreamers = 0;
      this.fStreamerInfos = {};
      if (this.fURL) {
         this.fEND = this.GetSize(this.fURL);
         this.ReadKeys();
      }
      this.fStreamers = new Array;

      return this;
   };

   JSROOTIO.RootFile.Version = version;

})();

// JSROOTIO.RootFile.js ends

