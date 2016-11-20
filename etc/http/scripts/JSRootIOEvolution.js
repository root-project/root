/// @file JSRootIOEvolution.js
/// I/O methods of JavaScript ROOT

(function( factory ) {
   if ( typeof define === "function" && define.amd ) {
      // AMD. Register as an anonymous module.
      define( ['JSRootCore', 'rawinflate'], factory );
   } else {
      if (typeof JSROOT == 'undefined')
         throw new Error("This extension requires JSRootCore.js", "JSRootIOEvolution.js");

      if (typeof JSROOT.IO == "object")
         throw new Error("This JSROOT IO already loaded", "JSRootIOEvolution.js");

      factory(JSROOT);
   }
} (function(JSROOT) {

   JSROOT.IO = {
         kBase : 0, kOffsetL : 20, kOffsetP : 40,
         kChar : 1, kShort : 2, kInt : 3, kLong : 4, kFloat : 5, kCounter : 6, kCharStar : 7,
         kDouble : 8, kDouble32 : 9, kLegacyChar : 10, kUChar : 11, kUShort : 12,
         kUInt : 13, kULong : 14, kBits : 15, kLong64 : 16, kULong64 : 17, kBool : 18, kFloat16 : 19,
         kObject : 61, kAny : 62, kObjectp : 63, kObjectP : 64, kTString : 65,
         kTObject : 66, kTNamed : 67, kAnyp : 68, kAnyP : 69, kAnyPnoVT : 70,
         kSTLp : 71,
         kSkip : 100, kSkipL : 120, kSkipP : 140,
         kConv : 200, kConvL : 220, kConvP : 240,
         kSTL : 300, kSTLstring : 365,
         kStreamer : 500, kStreamLoop : 501,
         kMapOffset : 2,
         kByteCountMask : 0x40000000,
         kNewClassTag : 0xFFFFFFFF,
         kClassMask : 0x80000000,
         Z_DEFLATED : 8,
         Z_HDRSIZE : 9,
         Mode : "array", // could be string or array, enable usage of ArrayBuffer in http requests
         NativeArray : true, // when true, native arrays like Int32Array or Float64Array are used
         IsInteger : function(typ) { return ((typ>=this.kChar) && (typ<=this.kLong)) ||
                                             (typ===this.kCounter) ||
                                            ((typ>=this.kLegacyChar) && (typ<=this.kBool)); },
         IsNumeric : function(typ) { return (typ>0) && (typ<=this.kBool) && (typ!==this.kCharStar); }

   };



// map of user-streamer function like func(buf,obj)
   JSROOT.fUserStreamers = {};

   JSROOT.addUserStreamer = function(type, user_streamer) {
      JSROOT.fUserStreamers[type] = user_streamer;
   }

   JSROOT.R__unzip = function(str, tgtsize, noalert, src_shift) {
      // Reads header envelope, determines zipped size and unzip content

      var isarr = (typeof str != 'string') && ('byteLength' in str),
          totallen = isarr ? str.byteLength : str.length,
          curr = 0, fullres = 0, tgtbuf = null;

      if (src_shift!==undefined) curr = src_shift;

      function getChar(o) {
         return isarr ? String.fromCharCode(str.getUint8(o)) : str.charAt(o);
      }

      function getCode(o) {
         return isarr ? str.getUint8(o) : str.charCodeAt(o);
      }

      while (fullres < tgtsize) {

         if (curr + JSROOT.IO.Z_HDRSIZE >= totallen) {
            if (!noalert) alert("Error R__unzip: header size exceeds buffer size");
            return null;
         }

         /*   C H E C K   H E A D E R   */
         if (!((getChar(curr) == 'Z' && getChar(curr+1) == 'L' && getCode(curr+2) == JSROOT.IO.Z_DEFLATED))) {
            if (!noalert) alert("R__unzip: Old zlib format is not supported!");
            return null;
         }

         var srcsize = JSROOT.IO.Z_HDRSIZE +
                         ((getCode(curr+3) & 0xff) | ((getCode(curr+4) & 0xff) << 8) | ((getCode(curr+5) & 0xff) << 16));

         if (isarr) {
            // portion of packed data to process
            var uint8arr = new Uint8Array(str.buffer, str.byteOffset + curr + JSROOT.IO.Z_HDRSIZE + 2, str.byteLength - curr - JSROOT.IO.Z_HDRSIZE - 2);

            //  place for unpacking
            if (tgtbuf===null) tgtbuf = new ArrayBuffer(tgtsize);

            var reslen = window.RawInflate.arr_inflate(uint8arr, new Uint8Array(tgtbuf, fullres));
            if (reslen<=0) break;

            fullres += reslen;
         } else {
            // old code using String for unpacking, keep for compativility
            var unpacked = window.RawInflate.inflate(str.substr(JSROOT.IO.Z_HDRSIZE + 2 + curr, srcsize));
            if ((unpacked === null) || (unpacked.length===0)) break;
            if (tgtbuf===null) tgtbuf = unpacked; else tgtbuf += unpacked;
            fullres += unpacked.length;
         }

         curr += srcsize;
      }

      if (fullres !== tgtsize) {
         if (!noalert) alert("R__unzip: fail to unzip data expacts " + tgtsize + " , got " + fullres);
         return null;
      }

      return isarr ? new DataView(tgtbuf) : tgtbuf;
   }

   // =================================================================================

   JSROOT.TBuffer = function(_o, _file) {
      this._typename = "TBuffer";
      this.o = (_o !== undefined) ? _o : 0;
      this.length = 0;
      this.fFile = _file;
      this.ClearObjectMap();
      this.fTagOffset = 0;
      this.last_read_version = 0;
      return this;
   }

   JSROOT.TBuffer.prototype.locate = function(pos) {
      this.o = pos;
   }

   JSROOT.TBuffer.prototype.shift = function(cnt) {
      this.o += cnt;
   }

   JSROOT.TBuffer.prototype.remain = function() {
      return this.length - this.o;
   }

   JSROOT.TBuffer.prototype.GetMappedObject = function(tag) {
      return this.fObjectMap[tag];
   }

   JSROOT.TBuffer.prototype.MapObject = function(tag, obj) {
      if (obj!==null)
         this.fObjectMap[tag] = obj;
   }

   JSROOT.TBuffer.prototype.MapClass = function(tag, classname) {
      this.fClassMap[tag] = classname;
   }

   JSROOT.TBuffer.prototype.GetMappedClass = function(tag) {
      if (tag in this.fClassMap) return this.fClassMap[tag];
      return -1;
   }

   JSROOT.TBuffer.prototype.ClearObjectMap = function() {
      this.fObjectMap = {};
      this.fClassMap = {};
      this.fObjectMap[0] = null;
   }

   JSROOT.TBuffer.prototype.ReadVersion = function() {
      // read class version from I/O buffer
      var ver = {};
      var bytecnt = this.ntou4(); // byte count
      if (bytecnt & JSROOT.IO.kByteCountMask)
         ver.bytecnt = bytecnt - JSROOT.IO.kByteCountMask - 2; // one can check between Read version and end of streamer
      else
         this.o -= 4; // rollback read bytes, this is old buffer without bytecount

      this.last_read_version = ver.val = this.ntoi2();
      ver.off = this.o;

      if ((ver.val <= 0) && ver.bytecnt && (ver.bytecnt>=6)) {
         // this is foreign class, extra stored checksum
         ver.checksum = this.ntou4();
      }

      return ver;
   }

   JSROOT.TBuffer.prototype.CheckBytecount = function(ver, where) {
      if (('bytecnt' in ver) && (ver.off + ver.bytecnt !== this.o)) {
         if (where!=null)
            alert("Missmatch in " + where + " bytecount expected = " + ver['bytecnt'] + "  got = " + (this.o-ver['off']));
         this.o = ver.off + ver.bytecnt;
         return false;
      }
      return true;
   }

   JSROOT.TBuffer.prototype.ReadString = function() {
      // read a null-terminated string from buffer
      var pos0 = this.o;
      while (this.o < this.length) {
         if (this.codeAt(this.o++) == 0) break;
      }
      return (this.o > pos0) ? this.substring(pos0, this.o-1) : "";
   }

   JSROOT.TBuffer.prototype.ReadTString = function() {
      // stream a TString object from buffer
      var len = this.ntou1();
      // large strings
      if (len == 255) len = this.ntou4();
      if (len==0) return "";

      var pos = this.o;
      this.o += len;

      return (this.codeAt(pos) == 0) ? '' : this.substring(pos, pos + len);
   }

   JSROOT.TBuffer.prototype.ReadFastArray = function(n, array_type) {
      // read array of n values from the I/O buffer

      var array = null;
      switch (array_type) {
         case JSROOT.IO.kDouble:
            array = JSROOT.IO.NativeArray ? new Float64Array(n) : new Array(n);
            for (var i = 0; i < n; ++i)
               array[i] = this.ntod();
            break;
         case JSROOT.IO.kFloat:
         case JSROOT.IO.kDouble32:
            array = JSROOT.IO.NativeArray ? new Float32Array(n) : new Array(n);
            for (var i = 0; i < n; ++i)
               array[i] = this.ntof();
            break;
         case JSROOT.IO.kLong:
         case JSROOT.IO.kLong64:
            array = JSROOT.IO.NativeArray ? new Float64Array(n) : new Array(n);
            for (var i = 0; i < n; ++i)
               array[i] = this.ntoi8();
            break;
         case JSROOT.IO.kULong:
         case JSROOT.IO.kULong64:
            array = JSROOT.IO.NativeArray ? new Float64Array(n) : new Array(n);
            for (var i = 0; i < n; ++i)
               array[i] = this.ntou8();
            break;
         case JSROOT.IO.kInt:
            array = JSROOT.IO.NativeArray ? new Int32Array(n) : new Array(n);
            for (var i = 0; i < n; ++i)
               array[i] = this.ntoi4();
            break;
         case JSROOT.IO.kBits:
         case JSROOT.IO.kUInt:
            array = JSROOT.IO.NativeArray ? new Uint32Array(n) : new Array(n);
            for (var i = 0; i < n; ++i)
               array[i] = this.ntou4();
            break;
         case JSROOT.IO.kShort:
            array = JSROOT.IO.NativeArray ? new Int16Array(n) : new Array(n);
            for (var i = 0; i < n; ++i)
               array[i] = this.ntoi2();
            break;
         case JSROOT.IO.kUShort:
            array = JSROOT.IO.NativeArray ? new Uint16Array(n) : new Array(n);
            for (var i = 0; i < n; ++i)
               array[i] = this.ntou2();
            break;
         case JSROOT.IO.kChar:
            array = JSROOT.IO.NativeArray ? new Int8Array(n) : new Array(n);
            for (var i = 0; i < n; ++i)
               array[i] = this.ntoi1();
            break;
         case JSROOT.IO.kBool:
         case JSROOT.IO.kUChar:
            array = JSROOT.IO.NativeArray ? new Uint8Array(n) : new Array(n);
            for (var i = 0; i < n; ++i)
               array[i] = this.ntou1();
            break;
         case JSROOT.IO.kTString:
            array = new Array(n);
            for (var i = 0; i < n; ++i)
               array[i] = this.ReadTString();
            break;
         default:
            array = new Array(n);
            for (var i = 0; i < n; ++i)
               array[i] = this.ntou4();
         break;
      }
      return array;
   }

   JSROOT.TBuffer.prototype.can_extract = function(place) {
      for (var n=0;n<place.length;n+=2)
        if (place[n] + place[n+1] > this.length) return false;
      return true;
   }

   JSROOT.IO.GetArrayKind = function(type_name) {
      // returns type of array
      // 0 - if TString (or equivalent)
      // -1 - if any other kind
      if ((type_name === "TString") || (JSROOT.fUserStreamers[type_name] === 'TString')) return 0;
      if ((type_name.length < 7) || (type_name.indexOf("TArray")!==0)) return -1;
      if (type_name.length == 7)
         switch (type_name.charAt(6)) {
            case 'I': return JSROOT.IO.kInt;
            case 'D': return JSROOT.IO.kDouble;
            case 'F': return JSROOT.IO.kFloat;
            case 'S': return JSROOT.IO.kShort;
            case 'C': return JSROOT.IO.kChar;
            case 'L': return JSROOT.IO.kLong;
            default: return -1;
         }

      return  type_name == "TArrayL64" ? JSROOT.IO.kLong64 : -1;
   }

   JSROOT.TBuffer.prototype.ReadTDate = function() {
      var datime = this.ntou4();
      var res = new Date();
      res.setFullYear((datime >>> 26) + 1995);
      res.setMonth((datime << 6) >>> 28);
      res.setDate((datime << 10) >>> 27);
      res.setHours((datime << 15) >>> 27);
      res.setMinutes((datime << 20) >>> 26);
      res.setSeconds((datime << 26) >>> 26);
      res.setMilliseconds(0);
      return res;
   }


   JSROOT.TBuffer.prototype.ReadTKey = function(key) {
      if (!key) key = {};
      key.fNbytes = this.ntoi4();
      key.fVersion = this.ntoi2();
      key.fObjlen = this.ntou4();
      key.fDatime = this.ReadTDate();
      key.fKeylen = this.ntou2();
      key.fCycle = this.ntou2();
      if (key.fVersion > 1000) {
         key.fSeekKey = this.ntou8();
         this.shift(8); // skip seekPdir
      } else {
         key.fSeekKey = this.ntou4();
         this.shift(4); // skip seekPdir
      }
      key.fClassName = this.ReadTString();
      key.fName = this.ReadTString();
      key.fTitle = this.ReadTString();

      var name = key.fName.replace(/['"]/g,'');

      if (name !== key.fName) {
         key.fRealName = key.fName;
         key.fName = name;
      }

      return key;
   }

   JSROOT.TBuffer.prototype.ReadTDirectory = function(dir) {

      var version = this.ntou2();
      dir.fDatimeC = this.ReadTDate();
      dir.fDatimeM = this.ReadTDate();
      dir.fNbytesKeys = this.ntou4();
      dir.fNbytesName = this.ntou4();
      dir.fSeekDir = (version > 1000) ? this.ntou8() : this.ntou4();
      dir.fSeekParent = (version > 1000) ? this.ntou8() : this.ntou4();
      dir.fSeekKeys = (version > 1000) ? this.ntou8() : this.ntou4();

      // if ((version % 1000) > 2) buf.shift(18); // skip fUUID
   }


   JSROOT.TBuffer.prototype.ReadTBasket = function(obj) {
      this.ReadTKey(obj);
      var ver = this.ReadVersion();
      obj.fBufferSize = this.ntoi4();
      obj.fNevBufSize = this.ntoi4();
      obj.fNevBuf = this.ntoi4();
      obj.fLast = this.ntoi4();
      var flag = this.ntoi1();
      // here we implement only data skipping, no real I/O for TBasket is performed

      if ((flag!==0) && ((flag % 10) != 2)) {
         var sz = this.ntoi4();
         // obj.fEntryOffset = this.ReadFastArray(sz, JSROOT.IO.kInt);
         this.shift(sz*4);

         if (flag>40) {
            sz = this.ntoi4();
            //   obj.fDisplacement = this.ReadFastArray(sz, JSROOT.IO.kInt);
            this.shift(sz*4);
         }
      }

      if ((flag === 1) || (flag > 10)) {
         var sz = obj.fLast;
         if (ver.val <= 1) sz = this.ntoi4();
         this.o += sz; // fBufferRef
      }

      return this.CheckBytecount(ver,"ReadTBasket");
   }

   JSROOT.TBuffer.prototype.ReadClass = function() {
      // read class definition from I/O buffer
      var classInfo = { name: -1 };
      var tag = 0;
      var bcnt = this.ntou4();

      var startpos = this.o;
      if (!(bcnt & JSROOT.IO.kByteCountMask) || (bcnt == JSROOT.IO.kNewClassTag)) {
         tag = bcnt;
         bcnt = 0;
      } else {
         tag = this.ntou4();
      }
      if (!(tag & JSROOT.IO.kClassMask)) {
         classInfo.objtag = tag; // indicate that we have deal with objects tag
         return classInfo;
      }
      if (tag == JSROOT.IO.kNewClassTag) {
         // got a new class description followed by a new object
         classInfo.name = this.ReadString();

         if (this.GetMappedClass(this.fTagOffset + startpos + JSROOT.IO.kMapOffset) === -1)
            this.MapClass(this.fTagOffset + startpos + JSROOT.IO.kMapOffset, classInfo.name);
      }  else {
         // got a tag to an already seen class
         var clTag = (tag & ~JSROOT.IO.kClassMask);
         classInfo.name = this.GetMappedClass(clTag);

         if (classInfo.name === -1) {
            alert("Did not found class with tag " + clTag);
         }
      }

      return classInfo;
   }

   JSROOT.TBuffer.prototype.ReadObjectAny = function() {
      var objtag = this.fTagOffset + this.o + JSROOT.IO.kMapOffset;

      var clRef = this.ReadClass();

      // class identified as object and should be handled so
      if ('objtag' in clRef)
         return this.GetMappedObject(clRef.objtag);

      if (clRef.name === -1) return null;

      var arrkind = JSROOT.IO.GetArrayKind(clRef.name);

      var obj = {};

      if (arrkind > 0) {
         // reading array, can map array only afterwards
         obj = this.ReadFastArray(this.ntou4(), arrkind);
         this.MapObject(objtag, obj);
      } else {
         // reading normal object, should map before to
         this.MapObject(objtag, obj);
         this.ClassStreamer(obj, clRef.name);
      }

      return obj;
   }

   JSROOT.TBuffer.prototype.ClassStreamer = function(obj, classname) {

      if (! ('_typename' in obj)) obj._typename = classname;

      var streamer = this.fFile.GetStreamer(classname);

      if (streamer !== null) {

         var ver = this.ReadVersion();

         for (var n = 0; n < streamer.length; ++n)
            streamer[n].func(this, obj);

         this.CheckBytecount(ver, classname);

      }
      else if (classname == 'TQObject') {
         // skip TQObject
      }
      else if (classname == "TBasket") {
         this.ReadTBasket(obj);
         JSROOT.addMethods(obj);
      } else {
         // just skip bytes belonging to not-recognized object
         // console.warn('skip object ', classname);

         var ver = this.ReadVersion();
         this.CheckBytecount(ver);
         JSROOT.addMethods(obj);
      }

      return obj;
   }

   // =================================================================================

   JSROOT.TStrBuffer = function(str, pos, file, length) {
      JSROOT.TBuffer.call(this, pos, file);
      this.b = str;
      if (length!==undefined) this.length = length; else
      if (str) this.length = str.length;
   }

   JSROOT.TStrBuffer.prototype = Object.create(JSROOT.TBuffer.prototype);

   JSROOT.TStrBuffer.prototype.extract = function(place) {
      var res = this.b.substr(place[0], place[1]);
      if (place.length===2) return res;
      res = [res];
      for (var n=2;n<place.length;n+=2)
         res.push(this.b.substr(place[n], place[n+1]));
      return res; // return array of strings for each part of the request
   }

   JSROOT.TStrBuffer.prototype.ntou1 = function() {
      return (this.b.charCodeAt(this.o++) & 0xff) >>> 0;
   }

   JSROOT.TStrBuffer.prototype.ntou2 = function() {
      // convert (read) two bytes of buffer b into a UShort_t
      var n = ((this.b.charCodeAt(this.o++) & 0xff) << 8) >>> 0;
         n += (this.b.charCodeAt(this.o++) & 0xff) >>> 0;
      return n;
   }

   JSROOT.TStrBuffer.prototype.ntou4 = function() {
      // convert (read) four bytes of buffer b into a UInt_t
      var n  = ((this.b.charCodeAt(this.o++) & 0xff) << 24) >>> 0;
      n += ((this.b.charCodeAt(this.o++) & 0xff) << 16) >>> 0;
      n += ((this.b.charCodeAt(this.o++) & 0xff) << 8)  >>> 0;
      n +=  (this.b.charCodeAt(this.o++) & 0xff) >>> 0;
      return n;
   }

   JSROOT.TStrBuffer.prototype.ntou8 = function() {
      // convert (read) eight bytes of buffer b into a ULong_t
      var n = ((this.b.charCodeAt(this.o++) & 0xff) << 56) >>> 0;
      n += ((this.b.charCodeAt(this.o++) & 0xff) << 48) >>> 0;
      n += ((this.b.charCodeAt(this.o++) & 0xff) << 40) >>> 0;
      n += ((this.b.charCodeAt(this.o++) & 0xff) << 32) >>> 0;
      n += ((this.b.charCodeAt(this.o++) & 0xff) << 24) >>> 0;
      n += ((this.b.charCodeAt(this.o++) & 0xff) << 16) >>> 0;
      n += ((this.b.charCodeAt(this.o++) & 0xff) << 8) >>> 0;
      n +=  (this.b.charCodeAt(this.o++) & 0xff) >>> 0;
      return n;
   }

   JSROOT.TStrBuffer.prototype.ntoi1 = function() {
      return (this.b.charCodeAt(this.o++) & 0xff);
   }

   JSROOT.TStrBuffer.prototype.ntoi2 = function() {
      // convert (read) two bytes of buffer b into a Short_t
      var n = ((this.b.charCodeAt(this.o++) & 0xff) << 8);
      n += ((this.b.charCodeAt(this.o++) & 0xff));
      return (n < 0x8000) ? n : -1 - (~n &0xFFFF);
   }

   JSROOT.TStrBuffer.prototype.ntoi4 = function() {
      // convert (read) four bytes of buffer b into a Int_t
      var n = ((this.b.charCodeAt(this.o++) & 0xff) << 24);
      n +=  ((this.b.charCodeAt(this.o++) & 0xff) << 16);
      n += ((this.b.charCodeAt(this.o++) & 0xff) << 8);
      n += ((this.b.charCodeAt(this.o++) & 0xff));
      return n;
   }

   JSROOT.TStrBuffer.prototype.ntoi8 = function(b, o) {
      // convert (read) eight bytes of buffer b into a Long_t
      var n = (this.b.charCodeAt(this.o++) & 0xff) << 56;
      n += (this.b.charCodeAt(this.o++) & 0xff) << 48;
      n += (this.b.charCodeAt(this.o++) & 0xff) << 40;
      n += (this.b.charCodeAt(this.o++) & 0xff) << 32;
      n += (this.b.charCodeAt(this.o++) & 0xff) << 24;
      n += (this.b.charCodeAt(this.o++) & 0xff) << 16;
      n += (this.b.charCodeAt(this.o++) & 0xff) << 8;
      n += (this.b.charCodeAt(this.o++) & 0xff);
      return n;
   }

   JSROOT.TStrBuffer.prototype.ntof = function() {
      // IEEE-754 Floating-Point Conversion (single precision - 32 bits)
      var inString = this.b.substring(this.o, this.o + 4); this.o+=4;
      if (inString.length < 4) return Number.NaN;
      var bits = "";
      for (var i=0; i<4; ++i) {
         var curByte = (inString.charCodeAt(i) & 0xff).toString(2);
         var byteLen = curByte.length;
         if (byteLen < 8) {
            for (var bit=0; bit<(8-byteLen); ++bit)
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
         for (var i=0; i<23; ++i) {
            if (parseInt(bits.substr(9+i, 1)) == 1)
               bman = bman + 1 / Math.pow(2, i+1);
         }
      }
      var res = bsign * Math.pow(2, bexp) * bman;
      return (Math.abs(res) < 1e-300) ? 0.0 : res;
   }

   JSROOT.TStrBuffer.prototype.ntod = function() {
      // IEEE-754 Floating-Point Conversion (double precision - 64 bits)
      var inString = this.b.substring(this.o, this.o + 8); this.o+=8;
      if (inString.length < 8) return Number.NaN;
      var bits = "";
      for (var i=0; i<8; ++i) {
         var curByte = (inString.charCodeAt(i) & 0xff).toString(2);
         var byteLen = curByte.length;
         if (byteLen < 8) {
            for (var bit=0; bit<(8-byteLen); ++bit)
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
         for (var i=0; i<52; ++i) {
            if (parseInt(bits.substr(12+i, 1)) == 1)
               bman = bman + 1 / Math.pow(2, i+1);
         }
      }
      var res = (bsign * Math.pow(2, bexp) * bman);
      return (Math.abs(res) < 1e-300) ? 0.0 : res;
   }

   JSROOT.TStrBuffer.prototype.codeAt = function(pos) {
      return this.b.charCodeAt(pos) & 0xff;
   }

   JSROOT.TStrBuffer.prototype.substring = function(beg, end) {
      return this.b.substring(beg, end);
   }

   // =======================================================================

   JSROOT.TArrBuffer = function(arr, pos, file, length) {
      // buffer should work with DataView as first argument
      JSROOT.TBuffer.call(this, pos, file, length);
      this.arr = arr;
      if (length!==undefined) this.length = length; else
      if (arr && arr.buffer) this.length = arr.buffer.byteLength;
   }

   JSROOT.TArrBuffer.prototype = Object.create(JSROOT.TBuffer.prototype);

   JSROOT.TArrBuffer.prototype.ReadFastArray = function(n, array_type) {
      // read array of n values from the I/O buffer

      var array, i = 0, o = this.o, view = this.arr;
      switch (array_type) {
         case JSROOT.IO.kDouble:
            array = new Float64Array(n);
            for (; i < n; ++i, o+=8)
               array[i] = view.getFloat64(o);
            break;
         case JSROOT.IO.kFloat:
         case JSROOT.IO.kDouble32:
            array = new Float32Array(n);
            for (; i < n; ++i, o+=4)
               array[i] = view.getFloat32(o);
            break;
         case JSROOT.IO.kLong:
         case JSROOT.IO.kLong64:
            array = new Float64Array(n);
            for (; i < n; ++i)
               array[i] = this.ntoi8();
            return array; // exit here to avoid conflicts
         case JSROOT.IO.kULong:
         case JSROOT.IO.kULong64:
            array = new Float64Array(n);
            for (; i < n; ++i)
               array[i] = this.ntou8();
            return array; // exit here to avoid conflicts
         case JSROOT.IO.kInt:
            array = new Int32Array(n);
            for (; i < n; ++i, o+=4)
               array[i] = view.getInt32(o);
            break;
         case JSROOT.IO.kBits:
         case JSROOT.IO.kUInt:
            array = new Uint32Array(n);
            for (; i < n; ++i, o+=4)
               array[i] = view.getUint32(o);
            break;
         case JSROOT.IO.kShort:
            array = new Int16Array(n);
            for (; i < n; ++i, o+=2)
               array[i] = view.getInt16(o);
            break;
         case JSROOT.IO.kUShort:
            array = new Uint16Array(n);
            for (; i < n; ++i, o+=2)
               array[i] = view.getUint16(o);
            break;
         case JSROOT.IO.kChar:
            array = new Int8Array(n);
            for (; i < n; ++i)
               array[i] = view.getInt8(o++);
            break;
         case JSROOT.IO.kBool:
         case JSROOT.IO.kUChar:
            array = new Uint8Array(n);
            for (; i < n; ++i)
               array[i] = view.getUint8(o++);
            break;
         case JSROOT.IO.kTString:
            array = new Array(n);
            for (; i < n; ++i)
               array[i] = this.ReadTString();
            return array; // exit here to avoid conflicts
         default:
            array = new Uint32Array(n);
            for (; i < n; ++i, o+=4)
               array[i] = view.getUint32(o);
            break;
      }

      this.o = o;

      return array;
   }

   JSROOT.TArrBuffer.prototype.extract = function(place) {
      if (!this.arr || !this.arr.buffer || !this.can_extract(place)) return null;
      if (place.length===2) return new DataView(this.arr.buffer, place[0], place[1]);

      var res = [];

      for (var n=0;n<place.length;n+=2)
         res.push(new DataView(this.arr.buffer, place[n], place[n+1]));

      return res; // return array of buffers
   }

   JSROOT.TArrBuffer.prototype.codeAt = function(pos) {
      return this.arr.getUint8(pos);
   }

   JSROOT.TArrBuffer.prototype.substring = function(beg, end) {
      var res = "";
      for (var n=beg;n<end;++n)
         res += String.fromCharCode(this.arr.getUint8(n));
      return res;
   }

   JSROOT.TArrBuffer.prototype.ntou1 = function() {
      return this.arr.getUint8(this.o++);
   }

   JSROOT.TArrBuffer.prototype.ntou2 = function() {
      var o = this.o; this.o+=2;
      return this.arr.getUint16(o);
   }

   JSROOT.TArrBuffer.prototype.ntou4 = function() {
      var o = this.o; this.o+=4;
      return this.arr.getUint32(o);
   }

   JSROOT.TArrBuffer.prototype.ntou8 = function() {
      var high = this.arr.getUint32(this.o); this.o+=4;
      var low = this.arr.getUint32(this.o); this.o+=4;
      return high * 0x100000000 + low;
   }

   JSROOT.TArrBuffer.prototype.ntoi1 = function() {
      return this.arr.getInt8(this.o++);
   }

   JSROOT.TArrBuffer.prototype.ntoi2 = function() {
      var o = this.o; this.o+=2;
      return this.arr.getInt16(o);
   }

   JSROOT.TArrBuffer.prototype.ntoi4 = function() {
      var o = this.o; this.o+=4;
      return this.arr.getInt32(o);
   }

   JSROOT.TArrBuffer.prototype.ntoi8 = function() {
      var high = this.arr.getUint32(this.o); this.o+=4;
      var low = this.arr.getUint32(this.o); this.o+=4;
      if (high < 0x80000000) return high * 0x100000000 + low;
      return -1 - ((~high) * 0x100000000 + ~low);
   }

   JSROOT.TArrBuffer.prototype.ntof = function() {
      var o = this.o; this.o+=4;
      return this.arr.getFloat32(o);
   }

   JSROOT.TArrBuffer.prototype.ntod = function() {
      var o = this.o; this.o+=8;
      return this.arr.getFloat64(o);
   }

   // =======================================================================

   JSROOT.CreateTBuffer = function(blob, pos, file, length) {
      if ((blob==null) || (typeof(blob) == 'string'))
         return new JSROOT.TStrBuffer(blob, pos, file, length);

      return new JSROOT.TArrBuffer(blob, pos, file, length);
   }

   JSROOT.ReconstructObject = function(class_name, obj_rawdata, sinfo_rawdata) {
      // method can be used to reconstruct ROOT object from binary buffer
      // Buffer can be requested from online server with request like:
      //   http://localhost:8080/Files/job1.root/hpx/root.bin
      // One also requires buffer with streamer infos, reqeusted with command
      //   http://localhost:8080/StreamerInfo/root.bin
      // And one should provide class name of the object
      //
      // Method provided for convenience only to see how binary JSROOT.IO works.
      // It is strongly recommended to use JSON representation:
      //   http://localhost:8080/Files/job1.root/hpx/root.json

      var file = new JSROOT.TFile;
      var buf = JSROOT.CreateTBuffer(sinfo_rawdata, 0, file);
      file.ExtractStreamerInfos(buf);

      var obj = {};

      buf = JSROOT.CreateTBuffer(obj_rawdata, 0, file);
      buf.MapObject(obj, 1);
      buf.ClassStreamer(obj, class_name);

      return obj;
   }

   // ==============================================================================

   // A class that reads a TDirectory from a buffer.

   // ctor
   JSROOT.TDirectory = function(file, dirname, cycle) {
      if (! (this instanceof arguments.callee) )
         throw new Error("you must use new to instantiate this class", "JSROOT.TDirectory.ctor");

      this.fFile = file;
      this._typename = "TDirectory";
      this.dir_name = dirname;
      this.dir_cycle = cycle;
      this.fKeys = [];
      return this;
   }

   JSROOT.TDirectory.prototype.GetKey = function(keyname, cycle, call_back) {
      // retrieve a key by its name and cycle in the list of keys
      for (var i=0; i < this.fKeys.length; ++i) {
         if (this.fKeys[i].fName == keyname && this.fKeys[i].fCycle == cycle) {
            JSROOT.CallBack(call_back, this.fKeys[i]);
            return this.fKeys[i];
         }
      }

      var pos = keyname.lastIndexOf("/");
      // try to handle situation when object name contains slashed (bad practice anyway)
      while (pos > 0) {
         var dirname = keyname.substr(0, pos);
         var subname = keyname.substr(pos+1);

         var dirkey = this.GetKey(dirname, 1);
         if ((dirkey!==null) && (typeof call_back == 'function') &&
              (dirkey.fClassName.indexOf("TDirectory")==0)) {

            this.fFile.ReadObject(this.dir_name + "/" + dirname, 1, function(newdir) {
               if (newdir) newdir.GetKey(subname, cycle, call_back);
            });
            return null;
         }

         pos = keyname.lastIndexOf("/", pos-1);
      }

      JSROOT.CallBack(call_back, null);
      return null;
   }

   JSROOT.TDirectory.prototype.ReadKeys = function(objbuf, readkeys_callback) {

      objbuf.ReadTDirectory(this);

      if ((this.fSeekKeys <= 0) || (this.fNbytesKeys <= 0))
         return JSROOT.CallBack(readkeys_callback, this);

      var dir = this, file = this.fFile;

      file.ReadBuffer([this.fSeekKeys, this.fNbytesKeys], function(blob) {
         if (!blob) return JSROOT.CallBack(readkeys_callback,null);

         //*-* -------------Read keys of the top directory

         var buf = JSROOT.CreateTBuffer(blob, 0, file);

         buf.ReadTKey();
         var nkeys = buf.ntoi4();

         for (var i = 0; i < nkeys; ++i)
            dir.fKeys.push(buf.ReadTKey());

         file.fDirectories.push(dir);

         delete buf;

         JSROOT.CallBack(readkeys_callback, dir);
      });
   }

   // ==============================================================================
   // A class that reads ROOT files.
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

   // ctor
   JSROOT.TFile = function(url, newfile_callback) {
      if (! (this instanceof arguments.callee) )
         throw new Error("you must use new to instantiate this class", "JSROOT.TFile.ctor");

      this._typename = "TFile";
      this.fEND = 0;
      this.fFullURL = url;
      this.fURL = url;
      this.fAcceptRanges = true; // when disabled ('+' at the end of file name), complete file content read with single operation
      this.fUseStampPar = new Date; // use additional time stamp parameter for file name to avoid browser caching problem
      this.fFileContent = null; // this can be full or parial content of the file (if ranges are not supported or if 1K header read from file)
                                // stored as TBuffer instance
      this.fMultiRanges = true; // true when server supports multirange requests
      this.fDirectories = [];
      this.fKeys = [];
      this.fSeekInfo = 0;
      this.fNbytesInfo = 0;
      this.fTagOffset = 0;
      this.fStreamers = 0;
      this.fStreamerInfos = null;
      this.fFileName = "";
      this.fStreamers = [];

      if (typeof this.fURL != 'string') return this;

      if (this.fURL.charAt(this.fURL.length-1) == "+") {
         this.fURL = this.fURL.substr(0, this.fURL.length-1);
         this.fAcceptRanges = false;
      }

      var pos = Math.max(this.fURL.lastIndexOf("/"), this.fURL.lastIndexOf("\\"));
      this.fFileName = pos>=0 ? this.fURL.substr(pos+1) : this.fURL;

      if (!this.fAcceptRanges) {
         this.ReadKeys(newfile_callback);
      } else {
         var file = this;

         JSROOT.NewHttpRequest(this.fURL, "head", function(res) {
            if (res==null)
               return JSROOT.CallBack(newfile_callback, null);

            var accept_ranges = res.getResponseHeader("Accept-Ranges");
            if (!accept_ranges) file.fAcceptRanges = false;
            var len = res.getResponseHeader("Content-Length");
            if (len) file.fEND = parseInt(len);
                else file.fAcceptRanges = false;
            file.ReadKeys(newfile_callback);
         }).send(null);
      }

      return this;
   }

   JSROOT.TFile.prototype.ReadBuffer = function(place, callback) {

      if ((this.fFileContent!==null) && (!this.fAcceptRanges || this.fFileContent.can_extract(place)))
         return callback(this.fFileContent.extract(place));

      var file = this;
      if ((place.length > 2) && !file.fMultiRanges) {
         var arg = { file: file, place: place, arr: [], callback: callback };

         function workaround_callback(res) {
            if (res!==undefined) this.arr.push(res);

            if (this.place.length===0)
               return JSROOT.CallBack(this.callback, this.arr);

            this.file.ReadBuffer([this.place.shift(), this.place.shift()], workaround_callback.bind(this));
         }

         return workaround_callback.bind(arg)();
      }

      var url = this.fURL, ranges = "bytes=";
      for (var n=0;n<place.length;n+=2) {
         if (n>0) ranges+=","
         ranges += (place[n] + "-" + (place[n] + place[n+1] - 1));
      }

      if (this.fUseStampPar) {
         // try to avoid browser caching by adding stamp parameter to URL
         if (url.indexOf('?')>0) url+="&stamp="; else url += "?stamp=";
         url += this.fUseStampPar.getTime();
      }

      function read_callback(res) {

         if (!res && file.fUseStampPar && (place[0]===0) && (place.length===2)) {
            // if fail to read file with stamp parameter, try once again without it
            file.fUseStampPar = false;
            var xhr = JSROOT.NewHttpRequest(file.fURL, ((JSROOT.IO.Mode == "array") ? "buf" : "bin"), read_callback);
            if (file.fAcceptRanges) xhr.setRequestHeader("Range", ranges);
            return xhr.send(null);
         }

         if (res && (place[0]===0) && (place.length===2) && !file.fFileContent) {
            // special case - keep content of first request (could be complete file) in memory

            file.fFileContent = JSROOT.CreateTBuffer((typeof res == 'string') ? res : new DataView(res));

            if (!file.fAcceptRanges)
               file.fEND = file.fFileContent.length;

            return callback(file.fFileContent.extract(place));
         }

         if ((res === null) || (res === undefined)) return callback(res);

         var isstr = (typeof res == 'string');

         // if only single segment requested, return result as is
         if (place.length===2) return callback(isstr ? res : new DataView(res));

         // object to access response data
         var arr = [], o = 0,
             hdr = this.getResponseHeader('Content-Type'),
             ismulti = hdr && (hdr.indexOf('multipart')>=0),
             view = isstr ? { getUint8: function(pos) { return res.charCodeAt(pos);  }, byteLength: res.length }
                       : new DataView(res);


         if (!ismulti) {
            // server may returns simple buffer

            var hdr_range = this.getResponseHeader('Content-Range'), segm_start = 0, segm_last = -1;

            if (hdr_range && hdr_range.indexOf("bytes")>=0) {
               var parts = hdr_range.substr(hdr_range.indexOf("bytes") + 6).split(/[\s-\/]+/);
               if (parts.length===3) {
                  segm_start = parseInt(parts[0]);
                  segm_last = parseInt(parts[1]);
                  if (isNaN(segm_start) || isNaN(segm_last) || (segm_start > segm_last)) {
                     segm_start = 0; segm_last = -1;
                  }
               }
            }

            var canbe_single_segment = segm_start<=segm_last;
            for(var n=0;n<place.length;n+=2)
               if ((place[n]<segm_start) || (place[n] + place[n+1] -1 > segm_last))
                  canbe_single_segment = false;

            if (canbe_single_segment) {
               for (var n=0;n<place.length;n+=2)
                  arr.push(isstr ? res.substr(place[n]-segm_start, place[n+1]) : new DataView(res, place[n]-segm_start, place[n+1]));
               return callback(arr);
            }

            console.error('Server returns normal response when multipart was requested, disable multirange support');
            file.fMultiRanges = false;
            return file.ReadBuffer(place, callback);
         }

         // multipart messages requires special handling

         var indx = hdr.indexOf("boundary="), boundary = "";
         if (indx > 0) boundary = "--" + hdr.substr(indx+9);
                  else console.error('Did not found boundary id in the response header');

         var n = 0;

         while (n<place.length) {

            var code1, code2 = view.getUint8(o), nline = 0, line = "",
                finish_header = false, segm_start = 0, segm_last = -1;

            while((o < view.byteLength-1) && !finish_header && (nline<5)) {
               code1 = code2;
               code2 = view.getUint8(o+1);

               if ((code1==13) && (code2==10)) {
                  if ((line.length>2) && (line.substr(0,2)=='--') && (line !== boundary)) {
                     console.error('Expact boundary ' + boundary + '  got ' + line);
                  }

                  line = line.toLowerCase();

                  if ((line.indexOf("content-range")>=0) && (line.indexOf("bytes") > 0)) {
                     var parts = line.substr(line.indexOf("bytes") + 6).split(/[\s-\/]+/);
                     if (parts.length===3) {
                        segm_start = parseInt(parts[0]);
                        segm_last = parseInt(parts[1]);
                        if (isNaN(segm_start) || isNaN(segm_last) || (segm_start > segm_last)) {
                           segm_start = 0; segm_last = -1;
                        }
                     } else {
                        console.error('Fail to decode content-range', line, parts);
                     }
                  }

                  if ((nline > 1) && (line.length===0)) finish_header = true;

                  o++; nline++; line = "";
                  code2 = view.getUint8(o+1);
               } else {
                  line += String.fromCharCode(code1);
               }
               o++;
            }

            if (!finish_header) {
               console.error('Cannot decode header in multipart message ');
               return callback(null);
            }

            if (segm_start > segm_last) {
               // fall-back solution, believe that segments same as requested
               arr.push(isstr ? res.substr(o, place[n+1]) : new DataView(res, o, place[n+1]));
               o += place[n+1];
               n += 2;
            } else {
               //var mycnt = 0;
               // segments may be merged by server
               while ((n<place.length) && (place[n] >= segm_start) && (place[n] + place[n+1] - 1 <= segm_last)) {
                  arr.push(isstr ? res.substr(o + place[n] - segm_start, place[n+1]) :
                                   new DataView(res, o + place[n] - segm_start, place[n+1]));
                  n += 2;
                  //mycnt++;
               }
               //if (mycnt>1) console.log('MERGE segments', mycnt);

               o += (segm_last-segm_start+1);
            }
         }

         callback(arr);
      }

      var xhr = JSROOT.NewHttpRequest(url, ((JSROOT.IO.Mode == "array") ? "buf" : "bin"), read_callback);
      if (this.fAcceptRanges) xhr.setRequestHeader("Range", ranges);
      xhr.send(null);
   }

   JSROOT.TFile.prototype.ReadBaskets = function(places, call_back) {
      // read basket with tree data

      var file = this;

      this.ReadBuffer(places, function(blobs) {

         if (!blobs) JSROOT.CallBack(call_back, null);

         var baskets = [];

         for (var n=0;n<places.length;n+=2) {

            var basket = {}, blob = (places.length > 2) ? blobs[n/2] : blobs;

            var buf = JSROOT.CreateTBuffer(blob);

            buf.ReadTBasket(basket);

//            console.log('buf.o', buf.o,'keylen', basket.fKeylen);

            if (basket.fNbytes !== places[n+1]) console.log('mismatch in basket sizes', basket.fNbytes, places[n+1]);

            if (basket.fKeylen + basket.fObjlen === basket.fNbytes) {
               // use data from original blob
               basket.raw = buf;
            } else {
               // unpack data and create new blob
               var objblob = JSROOT.R__unzip(blob, basket.fObjlen, false, buf.o);

               if (objblob) basket.raw = JSROOT.CreateTBuffer(objblob, 0, file);
            }

            baskets.push(basket);
         }

         JSROOT.CallBack(call_back, baskets);
      });
   }


   JSROOT.TFile.prototype.GetDir = function(dirname, cycle) {
      // check first that directory with such name exists

      if ((cycle==null) && (typeof dirname == 'string')) {
         var pos = dirname.lastIndexOf(';');
         if (pos>0) { cycle = dirname.substr(pos+1); dirname = dirname.substr(0,pos); }
      }

      for (var j=0; j < this.fDirectories.length; ++j) {
         var dir = this.fDirectories[j];
         if (dir.dir_name != dirname) continue;
         if ((cycle !== undefined) && (dir.dir_cycle !== cycle)) continue;
         return dir;
      }
      return null;
   }

   JSROOT.TFile.prototype.GetKey = function(keyname, cycle, getkey_callback) {
      // retrieve a key by its name and cycle in the list of keys
      // one should call_back when keys must be read first from the directory

      for (var i=0; i < this.fKeys.length; ++i) {
         if (this.fKeys[i].fName === keyname && this.fKeys[i].fCycle === cycle) {
            JSROOT.CallBack(getkey_callback, this.fKeys[i]);
            return this.fKeys[i];
         }
      }

      var pos = keyname.lastIndexOf("/");
      // try to handle situation when object name contains slashed (bad practice anyway)
      while (pos > 0) {
         var dirname = keyname.substr(0, pos);
         var subname = keyname.substr(pos+1);

         var dir = this.GetDir(dirname);
         if (dir!=null) return dir.GetKey(subname, cycle, getkey_callback);

         var dirkey = this.GetKey(dirname, 1);
         if ((dirkey !== null) && (getkey_callback != null) &&
             (dirkey.fClassName.indexOf("TDirectory")==0)) {

            this.ReadObject(dirname, function(newdir) {
               if (newdir) newdir.GetKey(subname, cycle, getkey_callback);
            });
            return null;
         }

         pos = keyname.lastIndexOf("/", pos-1);
      }

      JSROOT.CallBack(getkey_callback, null);
      return null;
   }

   JSROOT.TFile.prototype.ReadObjBuffer = function(key, callback) {
      // read and inflate object buffer described by its key

      var file = this;

      this.ReadBuffer([key.fSeekKey + key.fKeylen, key.fNbytes - key.fKeylen], function(blob1) {

         if (blob1==null) callback(null);

         var buf = null;

         if (key.fObjlen <= key.fNbytes - key.fKeylen) {
            buf = JSROOT.CreateTBuffer(blob1, 0, file);
         } else {
            var objbuf = JSROOT.R__unzip(blob1, key.fObjlen);
            if (objbuf==null) return callback(null);
            buf = JSROOT.CreateTBuffer(objbuf, 0, file);
         }

         buf.fTagOffset = key.fKeylen;

         callback(buf);
      });
   }

   JSROOT.TFile.prototype.ReadObject = function(obj_name, cycle, user_call_back) {
      // Read any object from a root file
      // One could specify cycle number in the object name or as separate argument
      // Last argument should be callback function, while data reading from file is asynchron

      if (typeof cycle == 'function') { user_call_back = cycle; cycle = 1; }

      var pos = obj_name.lastIndexOf(";");
      if (pos>0) {
         cycle = parseInt(obj_name.slice(pos+1));
         obj_name = obj_name.slice(0, pos);
      }

      if ((typeof cycle != 'number') || (cycle<0)) cycle = 1;
      // remove leading slashes
      while ((obj_name.length>0) && (obj_name[0] == "/")) obj_name = obj_name.substr(1);

      var file = this;

      // we use callback version while in some cases we need to
      // read sub-directory to get list of keys
      // in such situation calls are asynchrone
      this.GetKey(obj_name, cycle, function(key) {

         if (key == null)
            return JSROOT.CallBack(user_call_back, null);

         if ((obj_name=="StreamerInfo") && (key.fClassName=="TList"))
            return file.fStreamerInfos;

         var isdir = false;
         if ((key.fClassName == 'TDirectory' || key.fClassName == 'TDirectoryFile')) {
            isdir = true;
            var dir = file.GetDir(obj_name, cycle);
            if (dir) return JSROOT.CallBack(user_call_back, dir);
         }

         file.ReadObjBuffer(key, function(buf) {
            if (!buf) return JSROOT.CallBack(user_call_back, null);

            if (isdir) {
               var dir = new JSROOT.TDirectory(file, obj_name, cycle);
               dir.fTitle = key.fTitle;
               return dir.ReadKeys(buf, user_call_back);
            }

            var obj = {};
            buf.MapObject(1, obj); // tag object itself with id==1
            buf.ClassStreamer(obj, key.fClassName);

            if (key.fClassName==='TF1')
               return file.ReadFormulas(obj, user_call_back, -1);

            JSROOT.CallBack(user_call_back, obj);
         }); // end of ReadObjBuffer callback
      }); // end of GetKey callback
   }

   JSROOT.TFile.prototype.ReadFormulas = function(tf1, user_call_back, cnt) {

      var indx = cnt;
      while (++indx < this.fKeys.length) {
         if (this.fKeys[indx].fClassName == 'TFormula') break;
      }

      if (indx >= this.fKeys.length)
         return JSROOT.CallBack(user_call_back, tf1);

      var file = this;

      this.ReadObject(this.fKeys[indx].fName, this.fKeys[indx].fCycle, function(formula) {
          tf1.addFormula(formula);
          file.ReadFormulas(tf1, user_call_back, indx);
      });
   }

   JSROOT.TFile.prototype.ExtractStreamerInfos = function(buf) {
      if (!buf) return;

      var lst = {};
      buf.MapObject(1, lst);
      buf.ClassStreamer(lst, 'TList');

      lst._typename = "TStreamerInfoList";

      this.fStreamerInfos = lst;

      if (typeof JSROOT.addStreamerInfos === 'function')
         JSROOT.addStreamerInfos(lst);
   }


   JSROOT.TFile.prototype.ReadKeys = function(readkeys_callback) {
      // read keys only in the root file

      var file = this;

      // with the first readbuffer we read bigger amount to create header cache
      this.ReadBuffer([0, 1024], function(blob) {
         if (!blob) return JSROOT.CallBack(readkeys_callback, null);

         var buf = JSROOT.CreateTBuffer(blob, 0, file);

         if (buf.substring(0, 4) !== 'root') {
            alert("NOT A ROOT FILE! " + file.fURL);
            return JSROOT.CallBack(readkeys_callback, null);
         }
         buf.shift(4);

         file.fVersion = buf.ntou4();
         file.fBEGIN = buf.ntou4();
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

         // empty file
         if (!file.fSeekInfo || !file.fNbytesInfo)
            return JSROOT.CallBack(readkeys_callback, null);

         // extra check to prevent reading of corrupted data
         if (!file.fNbytesName || this.fNbytesName > 100000) {
            JSROOT.console("Init : cannot read directory info of file " + file.fURL);
            return JSROOT.CallBack(readkeys_callback, null);
         }

         //*-*-------------Read directory info
         var nbytes = file.fNbytesName + 22;
         nbytes += 4;  // fDatimeC.Sizeof();
         nbytes += 4;  // fDatimeM.Sizeof();
         nbytes += 18; // fUUID.Sizeof();
         // assume that the file may be above 2 Gbytes if file version is > 4
         if (file.fVersion >= 40000) nbytes += 12;

         // this part typically read from the header, no need to optimize
         file.ReadBuffer([file.fBEGIN, Math.max(300, nbytes)], function(blob3) {
            if (!blob3) return JSROOT.CallBack(readkeys_callback, null);

            var buf3 = JSROOT.CreateTBuffer(blob3, 0, file);

            // keep only title from TKey data
            file.fTitle = buf3.ReadTKey().fTitle;

            buf3.locate(file.fNbytesName);

            // we read TDirectory part of TFile
            buf3.ReadTDirectory(file);

            if (!file.fSeekKeys) {
               JSROOT.console("Empty keys list in " + file.fURL);
               return JSROOT.CallBack(readkeys_callback, null);
            }

            // read with same request keys and streamer infos
            file.ReadBuffer([file.fSeekKeys, file.fNbytesKeys, file.fSeekInfo, file.fNbytesInfo], function(blobs) {

               if (!blobs) return JSROOT.CallBack(readkeys_callback, null);

               var buf4 = JSROOT.CreateTBuffer(blobs[0], 0, file);

               buf4.ReadTKey(); //
               var nkeys = buf4.ntoi4();
               for (var i = 0; i < nkeys; ++i)
                  file.fKeys.push(buf4.ReadTKey());

               var buf5 = JSROOT.CreateTBuffer(blobs[1], 0, file);
               var si_key = buf5.ReadTKey();
               if (!si_key) return JSROOT.CallBack(readkeys_callback, null);

               file.fKeys.push(si_key);
               file.ReadObjBuffer(si_key, function(blob6) {
                  if (blob6) file.ExtractStreamerInfos(blob6);

                  return JSROOT.CallBack(readkeys_callback, file);
               });

               delete buf5;
               delete buf4;
            });
            delete buf3;
         });
         delete buf;
      });
   };

   JSROOT.TFile.prototype.ReadDirectory = function(dir_name, cycle, readdir_callback) {
      // read the directory content from  a root file
      // do not read directory if it is already exists

      return this.ReadObject(dir_name, cycle, readdir_callback);
   };

   JSROOT.TFile.prototype.AddMethods = function(clname, streamer) {
      // create additional entries in the streamer, which sets all methods of the class

      if (streamer === null) return streamer;

      var methods = JSROOT.getMethods(clname);
      if (methods !== null)
         for (var key in methods)
            if ((typeof methods[key] === 'function') || (key.indexOf("_")==0))
               streamer.push({
                 name: key,
                 method: methods[key],
                 func: function(buf,obj) { obj[this.name] = this.method; }
               });

      return streamer;
   }

   JSROOT.TFile.prototype.FindStreamerInfo = function(clname, clversion) {
      if (this.fStreamerInfos)
         for (var i=0; i < this.fStreamerInfos.arr.length; ++i)
            if (this.fStreamerInfos.arr[i].fName === clname)
               if ((clversion===undefined) || (this.fStreamerInfos.arr[i].fClassVersion===clversion))
                  return this.fStreamerInfos.arr[i];

      return null;
   }

   JSROOT.TFile.prototype.GetStreamer = function(clname) {
      // return the streamer for the class 'clname', from the list of streamers
      // or generate it from the streamer infos and add it to the list

      var streamer = this.fStreamers[clname];
      if (streamer !== undefined) return streamer;

      // check element in streamer infos, one can have special cases
      var s_i = this.FindStreamerInfo(clname);

      if (clname == 'TQObject' || clname == "TBasket") {
         // these are special cases, which are handled separately
         this.fStreamers[clname] = null;
         return null;
      }

      this.fStreamers[clname] = streamer = new Array;

      if (clname == 'TObject'|| clname == 'TMethodCall') {
         streamer.push({ func: function(buf,obj) {
            obj.fUniqueID = buf.ntou4();
            obj.fBits = buf.ntou4();
         } });
         return this.AddMethods(clname, streamer);
      }

      if (clname == 'TNamed') {
         streamer.push({ func : function(buf,obj) {
            buf.ReadVersion(); // ignore TObject version
            obj.fUniqueID = buf.ntou4();
            obj.fBits = buf.ntou4();
            obj.fName = buf.ReadTString();
            obj.fTitle = buf.ReadTString();
         } });
         return this.AddMethods(clname, streamer);
      }

      if ((clname == 'TList') || (clname == 'THashList')) {
         streamer.push({ classname: clname,
                         func : function(buf, obj) {
            // stream all objects in the list from the I/O buffer
            obj._typename = this.classname;
            obj.name = "";
            obj.arr = new Array;
            obj.opt = new Array;
            if (buf.last_read_version > 3) {
               buf.ClassStreamer(obj, "TObject");
               obj.name = buf.ReadTString();
               var nobjects = buf.ntou4();
               for (var i = 0; i < nobjects; ++i) {
                  obj.arr.push(buf.ReadObjectAny());
                  obj.opt.push(buf.ReadTString());
               }
            }
         } });
         return this.AddMethods(clname, streamer);
      }

      if (clname == 'TClonesArray') {
         streamer.push({ func : function(buf, list) {
            list._typename = "TClonesArray";
            list.name = "";
            list.arr = new Array();
            var ver = buf.last_read_version;
            if (ver > 2)
               buf.ClassStreamer(list, "TObject");
            if (ver > 1)
               list.name = buf.ReadTString();
            var s = buf.ReadTString();
            var classv = s;
            var clv = 0;
            var pos = s.indexOf(";");
            if (pos != -1) {
               classv = s.slice(0, pos);
               s = s.slice(pos+1, s.length-pos-1);
               clv = parseInt(s);
            }

            var nobjects = buf.ntou4();
            if (nobjects < 0) nobjects = -nobjects;  // for backward compatibility
            var lowerbound = buf.ntou4();

            // TO BE DONE - READING OF CLONES ARRAY!!!

            //for (var i = 0; i < nobjects; ++i) {
            //   var obj = buf.ClassStreamer({}, classv);
            //   list['arr'].push(obj);
            //}
         }});
         return this.AddMethods(clname, streamer);
      }

      if (clname == 'TCanvas') {
         streamer.push({ func : function(buf, obj) {

            obj._typename = "TCanvas";

            buf.ClassStreamer(obj, "TPad");

            obj.fDISPLAY = buf.ReadTString();
            obj.fDoubleBuffer = buf.ntoi4();
            obj.fRetained = (buf.ntou1() !== 0);
            obj.fXsizeUser = buf.ntoi4();
            obj.fYsizeUser = buf.ntoi4();
            obj.fXsizeReal = buf.ntoi4();
            obj.fYsizeReal = buf.ntoi4();
            obj.fWindowTopX = buf.ntoi4();
            obj.fWindowTopY = buf.ntoi4();
            obj.fWindowWidth = buf.ntoi4();
            obj.fWindowHeight = buf.ntoi4();
            obj.fCw = buf.ntou4();
            obj.fCh = buf.ntou4();

            obj.fCatt = buf.ClassStreamer({}, "TAttCanvas");

            buf.ntou1(); // ignore b << TestBit(kMoveOpaque);
            buf.ntou1(); // ignore b << TestBit(kResizeOpaque);
            obj.fHighLightColor = buf.ntoi2();
            obj.fBatch = (buf.ntou1() !== 0);
            buf.ntou1();   // ignore b << TestBit(kShowEventStatus);
            buf.ntou1();   // ignore b << TestBit(kAutoExec);
            buf.ntou1();   // ignore b << TestBit(kMenuBar);
         }});
         return this.AddMethods(clname, streamer);
      }

      if (clname == 'TObjArray')  {
         streamer.push({ func : function(buf, list) {
            list._typename = "TObjArray";
            list.name = "";
            list.arr = new Array();
            var ver = buf.last_read_version;
            if (ver > 2)
               buf.ClassStreamer(list, "TObject");
            if (ver > 1)
               list.name = buf.ReadTString();
            var nobjects = buf.ntou4();
            var lowerbound = buf.ntou4();
            for (var i = 0; i < nobjects; ++i)
               list.arr.push(buf.ReadObjectAny());
         }});
         return this.AddMethods(clname, streamer);
      }

      if (clname == 'TPolyMarker3D') {
         streamer.push({ func : function(buf, marker) {
            var ver = buf.last_read_version;

            buf.ClassStreamer(marker, "TObject");

            buf.ClassStreamer(marker, "TAttMarker");

            marker.fN = buf.ntoi4();

            marker.fP = buf.ReadFastArray(marker.fN*3, JSROOT.IO.kFloat);

            marker.fOption = buf.ReadTString();

            if (ver > 1)
               marker.fName = buf.ReadTString();
            else
               marker.fName = "TPolyMarker3D";
         }});
         return this.AddMethods(clname, streamer);
      }

      if ((clname == 'TObjString') && !s_i) {
         // special case when TObjString was stored inside streamer infos,
         // than streamer cannot be normally generated
         streamer.push({ func : function(buf, obj) {
            obj._typename = "TObjString";
            buf.ClassStreamer(obj, "TObject");
            obj.fString = buf.ReadTString();
         }});
         return this.AddMethods(clname, streamer);
      }

      if (clname == "TStreamerInfo") {
         streamer.push({ func : function(buf, streamerinfo) {
            // stream an object of class TStreamerInfo from the I/O buffer
            if (buf.last_read_version > 1) {
               buf.ClassStreamer(streamerinfo, "TNamed");

               streamerinfo.fCheckSum = buf.ntou4();
               streamerinfo.fClassVersion = buf.ntou4();
               streamerinfo.fElements = buf.ReadObjectAny();
            }
         }});
         return this.AddMethods(clname, streamer);
      }

      if (clname == "TStreamerElement") {
         streamer.push({ func : function(buf, element) {
            // stream an object of class TStreamerElement

            var ver = buf.last_read_version;
            buf.ClassStreamer(element, "TNamed");
            element.fType = buf.ntou4();
            element.fSize = buf.ntou4();
            element.fArrayLength = buf.ntou4();
            element.fArrayDim = buf.ntou4();
            element.fMaxIndex = buf.ReadFastArray((ver == 1) ? buf.ntou4() : 5, JSROOT.IO.kUInt);
            element.fTypeName = buf.ReadTString();

            if ((element.fType == JSROOT.IO.kUChar) && (element.fTypeName == "Bool_t" ||
                  element.fTypeName == "bool"))
               element.fType = JSROOT.IO.kBool;
            if (ver > 1) {
            }
            if (ver <= 2) {
               // In TStreamerElement v2, fSize was holding the size of
               // the underlying data type.  In later version it contains
               // the full length of the data member.
            }
            if (ver == 3) {
               element.fXmin = buf.ntod();
               element.fXmax = buf.ntod();
               element.fFactor = buf.ntod();
            }
            if (ver > 3) {
            }
         }});
         return this.AddMethods(clname, streamer);
      }

      if (clname == "TStreamerBase") {
         streamer.push({ func : function(buf, elem) {
            // stream an object of class TStreamerBase

            var ver = buf.last_read_version;
            buf.ClassStreamer(elem, "TStreamerElement");
            if (ver > 2) {
               elem.fBaseVersion = buf.ntou4();
            }
         }});
         return this.AddMethods(clname, streamer);
      }

      if ((clname == "TStreamerBasicPointer") || (clname == "TStreamerLoop")) {
         streamer.push({ func : function(buf,elem) {
            // stream an object of class TStreamerBasicPointer
            if (buf.last_read_version > 1) {
               buf.ClassStreamer(elem, "TStreamerElement");
               elem.fCountVersion = buf.ntou4();
               elem.fCountName = buf.ReadTString();
               elem.fCountClass = buf.ReadTString();
            }
         }});
         return this.AddMethods(clname, streamer);
      }

      if (clname == "TStreamerSTL") {
         streamer.push({ func : function(buf, elem) {
            if (buf.last_read_version > 1) {
               buf.ClassStreamer(elem, "TStreamerElement");
               elem.fSTLtype = buf.ntou4();
               elem.fCtype = buf.ntou4();
            }
         }});
         return streamer;
      }

      if (clname == "TStreamerObject" || clname == "TStreamerBasicType" ||
            clname == "TStreamerObjectAny" || clname == "TStreamerString" ||
            clname == "TStreamerObjectPointer") {
         streamer.push({ func: function(buf, elem) {
            if (buf.last_read_version > 1)
               buf.ClassStreamer(elem, "TStreamerElement");
         }});
         return this.AddMethods(clname, streamer);
      }

      if (clname == "TStreamerObjectAnyPointer") {
         streamer.push({ func: function(buf, elem) {
            if (buf.last_read_version > 0)
               buf.ClassStreamer(elem, "TStreamerElement");
         }});
         return this.AddMethods(clname, streamer);
      }

      if (s_i == null) {
         delete this.fStreamers[clname];
         // console.warn('did not find streamer for ', clname);
         return null;
      }

      if (s_i.fElements === null)
         return this.AddMethods(clname, streamer);

      for (var j=0; j<s_i.fElements.arr.length; ++j) {
         // extract streamer info for each class member
         var element = s_i.fElements.arr[j];

         var member = { name: element.fName, type: element.fType };

         if (element.fTypeName === 'BASE') {
            if (JSROOT.IO.GetArrayKind(member.name) > 0) {
               // this is workaround for arrays as base class
               // we create 'fArray' member, which read as any other data member
               member.name = 'fArray';
               member.type = JSROOT.IO.kAny;
            } else {
               // create streamer for base class
               member.type = JSROOT.IO.kBase;
               this.GetStreamer(element.fName);
            }
         }

         switch (member.type) {
            case JSROOT.IO.kBase:
               member.func =  function(buf, obj) {
                  buf.ClassStreamer(obj, this.name);
               };
               break;
            case JSROOT.IO.kTString:
               member.func = function(buf,obj) { obj[this.name] = buf.ReadTString(); }; break;
            case JSROOT.IO.kAnyP:
            case JSROOT.IO.kObjectP:
               member.func = function(buf,obj) { obj[this.name] = buf.ReadObjectAny(); }; break;
            case JSROOT.IO.kOffsetL+JSROOT.IO.kBool:
            case JSROOT.IO.kOffsetL+JSROOT.IO.kInt:
            case JSROOT.IO.kOffsetL+JSROOT.IO.kDouble:
            case JSROOT.IO.kOffsetL+JSROOT.IO.kShort:
            case JSROOT.IO.kOffsetL+JSROOT.IO.kUShort:
            case JSROOT.IO.kOffsetL+JSROOT.IO.kBits:
            case JSROOT.IO.kOffsetL+JSROOT.IO.kUInt:
            case JSROOT.IO.kOffsetL+JSROOT.IO.kULong:
            case JSROOT.IO.kOffsetL+JSROOT.IO.kULong64:
            case JSROOT.IO.kOffsetL+JSROOT.IO.kLong:
            case JSROOT.IO.kOffsetL+JSROOT.IO.kLong64:
            case JSROOT.IO.kOffsetL+JSROOT.IO.kFloat:
            case JSROOT.IO.kOffsetL+JSROOT.IO.kDouble32:
               if (element.fArrayDim === 1) {
                  member.arrlength = element.fArrayLength;
                  member.func = function(buf, obj) {
                     obj[this.name] = buf.ReadFastArray(this.arrlength, this.type - JSROOT.IO.kOffsetL);
                  };
               } else
               if (element.fArrayDim === 2) {
                  member.arrlength = element.fMaxIndex[1];
                  member.maxindx = element.fMaxIndex[0];
                  member.func = function(buf, obj) {
                     obj[this.name] = [];
                     for (var n=0;n<this.maxindx;++n)
                        obj[this.name].push(buf.ReadFastArray(this.arrlength, this.type - JSROOT.IO.kOffsetL));
                  };
               } else {
                  member.maxdim = element.fArrayDim - 1;
                  member.maxindx = element.fMaxIndex;
                  member.arrlength = element.fArrayLength;
                  member.func = function(buf, obj) {
                     var tmp = buf.ReadFastArray(this.arrlength, this.type - JSROOT.IO.kOffsetL),
                         indx = [], arr = [], i, k;
                     for (i=0; i<=this.maxdim; ++i) { indx[i] = 0; arr[i] = []; }
                     for (i=0;i<tmp.length;++i) {
                        arr[this.maxdim].push(tmp[i]);
                        ++indx[this.maxdim];
                        k = this.maxdim;
                        while ((indx[k] === this.maxindx[k]) && (k>0)) {
                           indx[k] = 0;
                           arr[k-1].push(arr[k]);
                           arr[k] = [];
                           ++indx[--k];
                        }
                     }
                     obj[this.name] = arr[0];
                  };
               }
               break;
            case JSROOT.IO.kOffsetP+JSROOT.IO.kBool:
            case JSROOT.IO.kOffsetP+JSROOT.IO.kInt:
            case JSROOT.IO.kOffsetP+JSROOT.IO.kDouble:
            case JSROOT.IO.kOffsetP+JSROOT.IO.kUChar:
            case JSROOT.IO.kOffsetP+JSROOT.IO.kChar:
            case JSROOT.IO.kOffsetP+JSROOT.IO.kShort:
            case JSROOT.IO.kOffsetP+JSROOT.IO.kUShort:
            case JSROOT.IO.kOffsetP+JSROOT.IO.kBits:
            case JSROOT.IO.kOffsetP+JSROOT.IO.kUInt:
            case JSROOT.IO.kOffsetP+JSROOT.IO.kULong:
            case JSROOT.IO.kOffsetP+JSROOT.IO.kULong64:
            case JSROOT.IO.kOffsetP+JSROOT.IO.kLong:
            case JSROOT.IO.kOffsetP+JSROOT.IO.kLong64:
            case JSROOT.IO.kOffsetP+JSROOT.IO.kFloat:
            case JSROOT.IO.kOffsetP+JSROOT.IO.kDouble32:
               member.cntname = element.fCountName;
               member.func = function(buf, obj) {
                  if (buf.ntou1() === 1)
                     obj[this.name] = buf.ReadFastArray(obj[this.cntname], this.type - JSROOT.IO.kOffsetP);
                  else
                     obj[this.name] = new Array();
               };
               break;
            case JSROOT.IO.kAny:
            case JSROOT.IO.kAnyp:
            case JSROOT.IO.kObjectp:
            case JSROOT.IO.kObject:
               var classname = (element.fTypeName === 'BASE') ? element.fName : element.fTypeName;
               if (classname.charAt(classname.length-1) == "*")
                  classname = classname.substr(0, classname.length - 1);

               var arrkind = JSROOT.IO.GetArrayKind(classname);

               if (arrkind > 0) {
                  member.arrkind = arrkind;
                  member.func = function(buf, obj) {
                     obj[this.name] = buf.ReadFastArray(buf.ntou4(), this.arrkind);
                  };
               } else
               if (arrkind === 0) {
                  member.func = function(buf,obj) { obj[this.name] = buf.ReadTString(); };
               } else {
                  member.classname = classname;

                  if (element.fArrayLength>1) {
                     member.arrlen = element.fArrayLength;
                     member.func = function(buf, obj) {
                        obj[this.name] = [];
                        for (var k=0;k<this.arrlen;++k)
                           obj[this.name].push(buf.ClassStreamer({}, this.classname));
                     };
                  } else {
                     member.func = function(buf, obj) {
                        obj[this.name] = buf.ClassStreamer({}, this.classname);
                     };
                  }
               }
               break;
            case JSROOT.IO.kOffsetL + JSROOT.IO.kObject:
            case JSROOT.IO.kOffsetL + JSROOT.IO.kAny:
            case JSROOT.IO.kOffsetL + JSROOT.IO.kAnyp:
            case JSROOT.IO.kOffsetL + JSROOT.IO.kObjectp:
               member.arrlength = element.fArrayLength;
               var classname = element.fTypeName;
               if (classname.charAt(classname.length-1) == "*")
                  classname = classname.substr(0, classname.length - 1);

               var arrkind = JSROOT.IO.GetArrayKind(classname);

               if (arrkind > 0) {
                  member.arrkind = arrkind;
                  member.func = function(buf, obj) {
                     obj[this.name] = [];
                     for (var k=0;k<this.arrlength;++k)
                        obj[this.name].push(buf.ReadFastArray(buf.ntou4(), this.arrkind));
                  };
               } else
               if (arrkind === 0) {
                  member.func = function(buf, obj) {
                     obj[this.name] = [];
                     for (var k=0;k<this.arrlength;++k)
                        obj[this.name].push(buf.ReadTString());
                  }
               } else {
                  member.classname = classname;
                  member.func = function(buf, obj) {
                     obj[this.name] = [];
                     for (var k=0;k<this.arrlength;++k)
                        obj[this.name].push(buf.ClassStreamer({}, this.classname));
                  };
               }
               break;
            case JSROOT.IO.kChar:
               member.func = function(buf,obj) { obj[this.name] = buf.ntoi1(); }; break;
            case JSROOT.IO.kCharStar:
               member.func = function(buf,obj) {
                  obj[this.name] = "";
                  var n = buf.ntoi4();
                  for (var i=0;i<n;++i) {
                     obj[this.name] += String.fromCharCode(buf.ntou1());
                  }
               };
               break;
            case JSROOT.IO.kShort:
               member.func = function(buf,obj) { obj[this.name] = buf.ntoi2(); }; break;
            case JSROOT.IO.kInt:
            case JSROOT.IO.kCounter:
               member.func = function(buf,obj) { obj[this.name] = buf.ntoi4(); }; break;
            case JSROOT.IO.kLong:
            case JSROOT.IO.kLong64:
               member.func = function(buf,obj) { obj[this.name] = buf.ntoi8(); }; break;
            case JSROOT.IO.kDouble:
               member.func = function(buf,obj) { obj[this.name] = buf.ntod(); }; break;
            case JSROOT.IO.kFloat:
            case JSROOT.IO.kDouble32:
               member.func = function(buf,obj) { obj[this.name] = buf.ntof(); }; break;
            case JSROOT.IO.kLegacyChar:
            case JSROOT.IO.kUChar:
               member.func = function(buf,obj) { obj[this.name] = buf.ntou1(); }; break;
            case JSROOT.IO.kUShort:
               member.func = function(buf,obj) { obj[this.name] = buf.ntou2(); }; break;
            case JSROOT.IO.kBits:
            case JSROOT.IO.kUInt:
               member.func = function(buf,obj) { obj[this.name] = buf.ntou4(); }; break;
            case JSROOT.IO.kULong64:
            case JSROOT.IO.kULong:
               member.func = function(buf,obj) { obj[this.name] = buf.ntou8(); }; break;
            case JSROOT.IO.kBool:
               member.func = function(buf,obj) { obj[this.name] = buf.ntou1() != 0; }; break;
            case JSROOT.IO.kStreamLoop:
            case JSROOT.IO.kStreamer:
               member.cntname = element.fCountName;
               member.typename = element.fTypeName;
               member.func = function(buf,obj) {
                  var ver = buf.ReadVersion();
                  var res = null;

                  if (this.typename == "TString*") {
                     var cnt = obj[this.cntname];
                     res = new Array(cnt);
                     for (var i = 0; i < cnt; ++i )
                        res[i] = buf.ReadTString();
                  } else
                  if (this.typename == "TList*") {
                     var cnt = obj[this.cntname];
                     res = new Array(cnt);
                     for (var i = 0; i < cnt; ++i)
                        res[i] = buf.ClassStreamer({}, "TList");
                  } else
                  if (this.typename == "vector<double>") res = buf.ReadFastArray(buf.ntoi4(), JSROOT.IO.kDouble); else
                  if (this.typename == "vector<int>") res = buf.ReadFastArray(buf.ntoi4(), JSROOT.IO.kInt); else
                  if (this.typename == "vector<float>") res = buf.ReadFastArray(buf.ntoi4(), JSROOT.IO.kFloat); else
                  if (this.typename == "vector<TObject*>") {
                     var n = buf.ntoi4();
                     res = [];
                     for (var i=0;i<n;++i) res.push(buf.ReadObjectAny());
                  }  else
                  if (this.typename.indexOf("map<TString,int")==0) {
                     var n = buf.ntoi4();
                     res = [];
                     for (var i=0;i<n;++i) {
                        var str = buf.ReadTString();
                        var val = buf.ntoi4();
                        res.push({ first: str, second: val});
                     }
                  } else {
                     JSROOT.console('failed to stream element of type ' + this.typename);
                  }

                  if (!buf.CheckBytecount(ver, this.typename)) res = null;

                  obj[this.name] = res;
               };
               break;
            default:
               if (JSROOT.fUserStreamers !== null)
                  member.func = JSROOT.fUserStreamers[element.fTypeName];

               if (typeof member.func !== 'function') {
                  JSROOT.console('fail to provide function for ' + element.fName + ' (' + element.fTypeName + ')  typ = ' + element.fType);
                  member.func = function(buf,obj) {};  // do nothing, fix in the future
               } else {
                  member.element = element; // one can use element in the custom function
               }
         }

         streamer.push(member);
      }

      return this.AddMethods(clname, streamer);
   };

   JSROOT.TFile.prototype.Delete = function() {
      this.fDirectories = null;
      this.fKeys = null;
      this.fStreamers = null;
      this.fSeekInfo = 0;
      this.fNbytesInfo = 0;
      this.fTagOffset = 0;
   };

   (function() {
      var iomode = JSROOT.GetUrlOption("iomode");
      if ((iomode=="str") || (iomode=="string")) JSROOT.IO.Mode = "string"; else
      if ((iomode=="bin") || (iomode=="arr") || (iomode=="array")) JSROOT.IO.Mode = "array";
      JSROOT.IO.NativeArray = ('Float64Array' in window);
   })();

   return JSROOT;

}));


// JSRootIOEvolution.js ends

