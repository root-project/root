/// @file JSRootIOEvolution.js
/// I/O methods of JavaScript ROOT

(function(){

   if (typeof JSROOT != "object") {
      var e1 = new Error("This extension requires JSRootCore.js");
      e1.source = "JSRootIOEvolution.js";
      throw e1;
   }

   if (typeof JSROOT.IO == "object") {
      var e1 = new Error("This JSROOT IO already loaded");
      e1.source = "JSRootIOEvolution.js";
      throw e1;
   }

   JSROOT.IO = {
         kBase : 0, kOffsetL : 20, kOffsetP : 40, kCounter : 6, kCharStar : 7,
         kChar : 1, kShort : 2, kInt : 3, kLong : 4, kFloat : 5,
         kDouble : 8, kDouble32 : 9, kLegacyChar : 10, kUChar : 11, kUShort : 12,
         kUInt : 13, kULong : 14, kBits : 15, kLong64 : 16, kULong64 : 17, kBool : 18,
         kFloat16 : 19,
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
         Z_HDRSIZE : 9
   };

   JSROOT.fUserStreamers = null; // map of user-streamer function like func(buf,obj,prop,streamerinfo)

   JSROOT.addUserStreamer = function(type, user_streamer) {
      if (JSROOT.fUserStreamers == null) JSROOT.fUserStreamers = {};
      JSROOT.fUserStreamers[type] = user_streamer;
   }

   JSROOT.R__unzip_header = function(str, off, noalert) {
      // Reads header envelope, and determines target size.

      if (off + JSROOT.IO.Z_HDRSIZE > str.length) {
         if (!noalert) alert("Error R__unzip_header: header size exceeds buffer size");
         return -1;
      }

      /*   C H E C K   H E A D E R   */
      if (!(str.charAt(off) == 'Z' && str.charAt(off+1) == 'L' && str.charCodeAt(off+2) == JSROOT.IO.Z_DEFLATED) &&
          !(str.charAt(off) == 'C' && str.charAt(off+1) == 'S' && str.charCodeAt(off+2) == JSROOT.IO.Z_DEFLATED) &&
          !(str.charAt(off) == 'X' && str.charAt(off+1) == 'Z' && str.charCodeAt(off+2) == 0)) {
         if (!noalert) alert("Error R__unzip_header: error in header");
         return -1;
      }
      return JSROOT.IO.Z_HDRSIZE +
                ((str.charCodeAt(off+3) & 0xff) |
                 ((str.charCodeAt(off+4) & 0xff) << 8) |
                 ((str.charCodeAt(off+5) & 0xff) << 16));
   }

   JSROOT.R__unzip = function(srcsize, str, off, noalert) {

      /*   C H E C K   H E A D E R   */
      if (srcsize < JSROOT.IO.Z_HDRSIZE) {
         if (!noalert) alert("R__unzip: too small source");
         return null;
      }

      /*   C H E C K   H E A D E R   */
      if (!(str.charAt(off) == 'Z' && str.charAt(off+1) == 'L' && str.charCodeAt(off+2) == JSROOT.IO.Z_DEFLATED) &&
          !(str.charAt(off) == 'C' && str.charAt(off+1) == 'S' && str.charCodeAt(off+2) == JSROOT.IO.Z_DEFLATED) &&
          !(str.charAt(off) == 'X' && str.charAt(off+1) == 'Z' && str.charCodeAt(off+2) == 0)) {
         if (!noalert) alert("Error R__unzip: error in header");
         return null;
      }
      var ibufcnt = ((str.charCodeAt(off+3) & 0xff) |
                    ((str.charCodeAt(off+4) & 0xff) << 8) |
                    ((str.charCodeAt(off+5) & 0xff) << 16));
      if (ibufcnt + JSROOT.IO.Z_HDRSIZE != srcsize) {
         if (!noalert) alert("R__unzip: discrepancy in source length");
         return null;
      }

      /*   D E C O M P R E S S   D A T A  */
      if (str.charAt(off) == 'Z' && str.charAt(off+1) == 'L') {
         /* New zlib format */
         var data = str.substr(off + JSROOT.IO.Z_HDRSIZE + 2, srcsize);
         return RawInflate.inflate(data);
      }
      /* Old zlib format */
      else {
         if (!noalert) alert("R__unzip: Old zlib format is not supported!");
         return null;
      }
      return null;
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
      var buf = new JSROOT.TBuffer(sinfo_rawdata, 0, file);
      file.ExtractStreamerInfos(buf);

      var obj = {};

      buf = new JSROOT.TBuffer(obj_rawdata, 0, file);
      buf.MapObject(obj, 1);
      buf.ClassStreamer(obj, class_name);

      return obj;
   }

   JSROOT.TBuffer = function(_str, _o, _file) {
      this._typename = "TBuffer";
      this.b = _str;
      this.o = (_o==null) ? 0 : _o;
      this.fFile = _file;
      this.ClearObjectMap();
      this.fTagOffset = 0;
      return this;
   }

   JSROOT.TBuffer.prototype.locate = function(pos) {
      this.o = pos;
   }

   JSROOT.TBuffer.prototype.shift = function(cnt) {
      this.o += cnt;
   }

   JSROOT.TBuffer.prototype.ntou1 = function() {
      return (this.b.charCodeAt(this.o++) & 0xff) >>> 0;
   }

   JSROOT.TBuffer.prototype.ntou2 = function() {
      // convert (read) two bytes of buffer b into a UShort_t
      var n = ((this.b.charCodeAt(this.o) & 0xff) << 8) >>> 0;
      n += (this.b.charCodeAt(this.o+1) & 0xff) >>> 0;
      this.o += 2;
      return n;
   }

   JSROOT.TBuffer.prototype.ntou4 = function() {
      // convert (read) four bytes of buffer b into a UInt_t
      var n  = ((this.b.charCodeAt(this.o) & 0xff) << 24) >>> 0;
      n += ((this.b.charCodeAt(this.o+1) & 0xff) << 16) >>> 0;
      n += ((this.b.charCodeAt(this.o+2) & 0xff) << 8)  >>> 0;
      n +=  (this.b.charCodeAt(this.o+3) & 0xff) >>> 0;
      this.o += 4;
      return n;
   }

   JSROOT.TBuffer.prototype.ntou8 = function() {
      // convert (read) eight bytes of buffer b into a ULong_t
      var n = ((this.b.charCodeAt(this.o) & 0xff) << 56) >>> 0;
      n += ((this.b.charCodeAt(this.o+1) & 0xff) << 48) >>> 0;
      n += ((this.b.charCodeAt(this.o+2) & 0xff) << 40) >>> 0;
      n += ((this.b.charCodeAt(this.o+3) & 0xff) << 32) >>> 0;
      n += ((this.b.charCodeAt(this.o+4) & 0xff) << 24) >>> 0;
      n += ((this.b.charCodeAt(this.o+5) & 0xff) << 16) >>> 0;
      n += ((this.b.charCodeAt(this.o+6) & 0xff) << 8) >>> 0;
      n +=  (this.b.charCodeAt(this.o+7) & 0xff) >>> 0;
      this.o += 8;
      return n;
   }

   JSROOT.TBuffer.prototype.ntoi1 = function() {
      return (this.b.charCodeAt(this.o++) & 0xff);
   }

   JSROOT.TBuffer.prototype.ntoi2 = function() {
      // convert (read) two bytes of buffer b into a Short_t
      var n  = (this.b.charCodeAt(this.o)   & 0xff) << 8;
      n += (this.b.charCodeAt(this.o+1) & 0xff);
      this.o += 2;
      return n;
   }

   JSROOT.TBuffer.prototype.ntoi4 = function() {
      // convert (read) four bytes of buffer b into a Int_t
      var n  = ((this.b.charCodeAt(this.o) & 0xff) << 24) +
               ((this.b.charCodeAt(this.o+1) & 0xff) << 16) +
               ((this.b.charCodeAt(this.o+2) & 0xff) << 8) +
               ((this.b.charCodeAt(this.o+3) & 0xff));
      this.o += 4;
      return n;
   }

   JSROOT.TBuffer.prototype.ntoi8 = function(b, o) {
      // convert (read) eight bytes of buffer b into a Long_t
      var n = (this.b.charCodeAt(this.o) & 0xff) << 56;
      n += (this.b.charCodeAt(this.o+1) & 0xff) << 48;
      n += (this.b.charCodeAt(this.o+2) & 0xff) << 40;
      n += (this.b.charCodeAt(this.o+3) & 0xff) << 32;
      n += (this.b.charCodeAt(this.o+4) & 0xff) << 24;
      n += (this.b.charCodeAt(this.o+5) & 0xff) << 16;
      n += (this.b.charCodeAt(this.o+6) & 0xff) << 8;
      n += (this.b.charCodeAt(this.o+7) & 0xff);
      this.o += 8;
      return n;
   }

   JSROOT.TBuffer.prototype.ntof = function() {
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
         for (var i=0; i<23; i++) {
            if (parseInt(bits.substr(9+i, 1)) == 1)
               bman = bman + 1 / Math.pow(2, i+1);
         }
      }
      return (bsign * Math.pow(2, bexp) * bman);
   }

   JSROOT.TBuffer.prototype.ntod = function() {
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
         for (var i=0; i<52; i++) {
            if (parseInt(bits.substr(12+i, 1)) == 1)
               bman = bman + 1 / Math.pow(2, i+1);
         }
      }
      return (bsign * Math.pow(2, bexp) * bman);
   }

   JSROOT.TBuffer.prototype.ReadFastArray = function(n, array_type) {
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
   }

   JSROOT.TBuffer.prototype.ReadBasicPointer = function(len, array_type) {
      var isArray = this.b.charCodeAt(this.o++) & 0xff;
      if (isArray)
         return this.ReadFastArray(len, array_type);

      if (len==0) return new Array();

      this.o--;
      return this.ReadFastArray(len, array_type);
   }

   JSROOT.TBuffer.prototype.ReadString = function(max_len) {
      // stream a string from buffer
      max_len = typeof(max_len) != 'undefined' ? max_len : 0;
      var len = 0;
      var pos0 = this.o;
      while ((max_len==0) || (len<max_len)) {
         if ((this.b.charCodeAt(this.o++) & 0xff) == 0) break;
         len++;
      }

      return (len == 0) ? "" : this.b.substring(pos0, pos0 + len);
   }

   JSROOT.TBuffer.prototype.ReadTString = function() {
      // stream a TString object from buffer
      var len = this.b.charCodeAt(this.o++) & 0xff;
      // large strings
      if (len == 255) len = this.ntou4();

      var pos = this.o;
      this.o += len;

      return (this.b.charCodeAt(pos) == 0) ? '' : this.b.substring(pos, pos + len);
   }

   JSROOT.TBuffer.prototype.GetMappedObject = function(tag) {
      return this.fObjectMap[tag];
   }

   JSROOT.TBuffer.prototype.MapObject = function(tag, obj) {
      if (obj==null) return;
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
         ver['bytecnt'] = bytecnt - JSROOT.IO.kByteCountMask - 2; // one can check between Read version and end of streamer
      else
         this.o -= 4; // rollback read bytes, this is old buffer without bytecount
      ver['val'] = this.ntou2();
      ver['off'] = this.o;
      return ver;
   }

   JSROOT.TBuffer.prototype.CheckBytecount = function(ver, where) {
      if (('bytecnt' in ver) && (ver['off'] + ver['bytecnt'] != this.o)) {
         if (where!=null)
            alert("Missmatch in " + where + " bytecount expected = " + ver['bytecnt'] + "  got = " + (this.o-ver['off']));
         this.o = ver['off'] + ver['bytecnt'];
         return false;
      }
      return true;
   }

   JSROOT.TBuffer.prototype.ReadTObject = function(tobj) {
      this.o += 2; // skip version
      if ((!'_typename' in tobj) || (tobj['_typename'] == ''))
         tobj['_typename'] = "TObject";

      tobj['fUniqueID'] = this.ntou4();
      tobj['fBits'] = this.ntou4();
      return true;
   }

   JSROOT.TBuffer.prototype.ReadTNamed = function(tobj) {
      // read a TNamed class definition from I/O buffer
      var ver = this.ReadVersion();
      this.ReadTObject(tobj);
      tobj['fName'] = this.ReadTString();
      tobj['fTitle'] = this.ReadTString();
      return this.CheckBytecount(ver, "ReadTNamed");
   }

   JSROOT.TBuffer.prototype.ReadTObjString = function(tobj) {
      // read a TObjString definition from I/O buffer
      var ver = this.ReadVersion();
      this.ReadTObject(tobj);
      tobj['fString'] = this.ReadTString();
      return this.CheckBytecount(ver, "ReadTObjString");
   }

   JSROOT.TBuffer.prototype.ReadTList = function(list) {
      // stream all objects in the list from the I/O buffer
      list['_typename'] = "TList";
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
   }

   JSROOT.TBuffer.prototype.ReadTObjArray = function(list) {
      list['_typename'] = "TObjArray";
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
   }

   JSROOT.TBuffer.prototype.ReadTClonesArray = function(list) {
      list['_typename'] = "TClonesArray";
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
   }

   JSROOT.TBuffer.prototype.ReadTPolyMarker3D = function(marker) {
      var ver = this.ReadVersion();

      this.ReadTObject(marker);

      this.ClassStreamer(marker, "TAttMarker");

      marker['fN'] = this.ntoi4();

      marker['fP'] = this.ReadFastArray(marker['fN']*3, 'F');

      marker['fOption'] = this.ReadTString();

      if (ver['val'] > 1)
         marker['fName'] = this.ReadTString();
      else
         marker['fName'] = "TPolyMarker3D";

      return this.CheckBytecount(ver, "ReadTPolyMarker3D");
   }

   JSROOT.TBuffer.prototype.ReadTCollection = function(list, str, o) {
      list['_typename'] = "TCollection";
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
   }

   JSROOT.TBuffer.prototype.ReadTKey = function(key) {
      key['fNbytes'] = this.ntoi4();
      key['fVersion'] = this.ntoi2();
      key['fObjlen'] = this.ntou4();
      var datime = this.ntou4();
      key['fDatime'] = new Date();
      key['fDatime'].setFullYear((datime >>> 26) + 1995);
      key['fDatime'].setMonth((datime << 6) >>> 28);
      key['fDatime'].setDate((datime << 10) >>> 27);
      key['fDatime'].setHours((datime << 15) >>> 27);
      key['fDatime'].setMinutes((datime << 20) >>> 26);
      key['fDatime'].setSeconds((datime << 26) >>> 26);
      key['fDatime'].setMilliseconds(0);
      key['fKeylen'] = this.ntou2();
      key['fCycle'] = this.ntou2();
      if (key['fVersion'] > 1000) {
         key['fSeekKey'] = this.ntou8();
         this.shift(8); // skip seekPdir
      } else {
         key['fSeekKey'] = this.ntou4();
         this.shift(4); // skip seekPdir
      }
      key['fClassName'] = this.ReadTString();
      key['fName'] = this.ReadTString();
      key['fTitle'] = this.ReadTString();

      var name = key['fName'].replace(/['"]/g,'');

      if (name != key['fName']) {
         key['fRealName'] = key['fName'];
         key['fName'] = name;
      }

      return true;
   }

   JSROOT.TBuffer.prototype.ReadTBasket = function(obj) {
      this.ReadTKey(obj);
      var ver = this.ReadVersion();
      obj['fBufferSize'] = this.ntoi4();
      obj['fNevBufSize'] = this.ntoi4();
      obj['fNevBuf'] = this.ntoi4();
      obj['fLast'] = this.ntoi4();
      var flag = this.ntoi1();
      // here we implement only data skipping, no real I/O for TBasket is performed
      if ((flag % 10) != 2) {
         var sz = this.ntoi4(); this.o += sz*4; // fEntryOffset
         if (flag>40) { sz = this.ntoi4(); this.o += sz*4; } // fDisplacement
      }

      if (flag == 1 || flag > 10) {
         var sz = obj['fLast'];
         if (ver['val'] <=1) sz = this.ntoi4();
         this.o += sz; // fBufferRef
      }
      return this.CheckBytecount(ver,"ReadTBasket");
   }


   JSROOT.TBuffer.prototype.ReadTCanvas = function(obj) {
      // stream all objects in the list from the I/O buffer
      var ver = this.ReadVersion();

      this.ClassStreamer(obj, "TPad");

      obj['fDISPLAY'] = this.ReadTString();
      obj['fDoubleBuffer'] = this.ntoi4();
      obj['fRetained'] = this.ntou1()!=0;
      obj['fXsizeUser'] = this.ntoi4();
      obj['fYsizeUser'] = this.ntoi4();
      obj['fXsizeReal'] = this.ntoi4();
      obj['fYsizeReal'] = this.ntoi4();
      obj['fWindowTopX'] = this.ntoi4();
      obj['fWindowTopY'] = this.ntoi4();
      obj['fWindowWidth'] = this.ntoi4();
      obj['fWindowHeight'] = this.ntoi4();
      obj['fCw'] = this.ntou4();
      obj['fCh'] = this.ntou4();

      obj['fCatt'] = {};
      this.ClassStreamer(obj['fCatt'], "TAttCanvas");
      this.ntou1(); // ignore b << TestBit(kMoveOpaque);
      this.ntou1(); // ignore b << TestBit(kResizeOpaque);
      obj['fHighLightColor'] = this.ntoi2();
      obj['fBatch'] = this.ntou1()!=0;
      this.ntou1();   // ignore b << TestBit(kShowEventStatus);
      this.ntou1();   // ignore b << TestBit(kAutoExec);
      this.ntou1();   // ignore b << TestBit(kMenuBar);

      // now TCanvas streamer should be complete - verify that bytecount is correct
      return this.CheckBytecount(ver, "TCanvas");
   }

   JSROOT.TBuffer.prototype.ReadTStreamerInfo = function(streamerinfo) {
      // stream an object of class TStreamerInfo from the I/O buffer

      var R__v = this.ReadVersion();
      if (R__v['val'] > 1) {
         this.ReadTNamed(streamerinfo);

         streamerinfo['fCheckSum'] = this.ntou4();
         streamerinfo['fClassVersion'] = this.ntou4();

         streamerinfo['fElements'] = this.ReadObjectAny();
      }
      return this.CheckBytecount(R__v, "ReadTStreamerInfo");
   }

   JSROOT.TBuffer.prototype.ReadStreamerElement = function(element) {
      // stream an object of class TStreamerElement

      var R__v = this.ReadVersion();
      this.ReadTNamed(element);
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
   }

   JSROOT.TBuffer.prototype.ReadStreamerBase = function(streamerbase) {
      // stream an object of class TStreamerBase

      var R__v = this.ReadVersion();
      this.ReadStreamerElement(streamerbase);
      if (R__v['val'] > 2) {
         streamerbase['baseversion'] = this.ntou4();
      }
      return this.CheckBytecount(R__v, "ReadStreamerBase");
   }

   JSROOT.TBuffer.prototype.ReadStreamerBasicType = function(streamerbase) {
      // stream an object of class TStreamerBasicType
      var R__v = this.ReadVersion();
      if (R__v['val'] > 1) {
         this.ReadStreamerElement(streamerbase);
      }
      return this.CheckBytecount(R__v, "ReadStreamerBasicType");
   }

   JSROOT.TBuffer.prototype.ReadStreamerBasicPointer = function(streamerbase) {
      // stream an object of class TStreamerBasicPointer
      var R__v = this.ReadVersion();
      if (R__v['val'] > 1) {
         this.ReadStreamerElement(streamerbase);
         streamerbase['countversion'] = this.ntou4();
         streamerbase['countName'] = this.ReadTString();
         streamerbase['countClass'] = this.ReadTString();
      }
      return this.CheckBytecount(R__v, "ReadStreamerBasicPointer");
   }

   JSROOT.TBuffer.prototype.ReadStreamerSTL = function(streamerSTL) {
      // stream an object of class TStreamerSTL

      var R__v = this.ReadVersion();
      if (R__v['val'] > 1) {
         this.ReadStreamerElement(streamerSTL);
         streamerSTL['stltype'] = this.ntou4();
         streamerSTL['ctype'] = this.ntou4();
      }
      return this.CheckBytecount(R__v, "ReadStreamerSTL");
   }

   JSROOT.TBuffer.prototype.ReadTStreamerObject = function(streamerbase) {
      // stream an object of class TStreamerObject
      var R__v = this.ReadVersion();
      if (R__v['val'] > 1) {
         this.ReadStreamerElement(streamerbase);
      }
      return this.CheckBytecount(R__v, "ReadTStreamerObject");
   }

   JSROOT.TBuffer.prototype.ReadClass = function() {
      // read class definition from I/O buffer
      var classInfo = {};
      classInfo['name'] = -1;
      var tag = 0;
      var bcnt = this.ntou4();

      var startpos = this.o;
      if (!(bcnt & JSROOT.IO.kByteCountMask) || (bcnt == JSROOT.IO.kNewClassTag)) {
         tag = bcnt;
         bcnt = 0;
      } else {
         // classInfo['fVersion'] = 1;
         tag = this.ntou4();
      }
      if (!(tag & JSROOT.IO.kClassMask)) {
         classInfo['objtag'] = tag; // indicate that we have deal with objects tag
         return classInfo;
      }
      if (tag == JSROOT.IO.kNewClassTag) {
         // got a new class description followed by a new object
         classInfo['name'] = this.ReadString();

         if (this.GetMappedClass(this.fTagOffset + startpos + JSROOT.IO.kMapOffset)==-1)
            this.MapClass(this.fTagOffset + startpos + JSROOT.IO.kMapOffset, classInfo['name']);
      }
      else {
         // got a tag to an already seen class
         var clTag = (tag & ~JSROOT.IO.kClassMask);
         classInfo['name'] = this.GetMappedClass(clTag);

         if (classInfo['name']==-1) {
            alert("Did not found class with tag " + clTag);
         }

      }
      // classInfo['cnt'] = (bcnt & ~JSROOT.IO.kByteCountMask);

      return classInfo;
   }

   JSROOT.TBuffer.prototype.ReadObjectAny = function() {
      var startpos = this.o;
      var clRef = this.ReadClass();

      // class identified as object and should be handled so
      if ('objtag' in clRef)
         return this.GetMappedObject(clRef['objtag']);

      if (clRef['name'] == -1) return null;

      var obj = {};

      this.MapObject(this.fTagOffset + startpos + JSROOT.IO.kMapOffset, obj);

      this.ClassStreamer(obj, clRef['name']);

      return obj;
   }

   JSROOT.TBuffer.prototype.ClassStreamer = function(obj, classname) {
      if (! ('_typename' in obj))  obj['_typename'] = classname;

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
         this.ReadTCanvas(obj);
      }
      else if (classname == 'TPolyMarker3D') {
         this.ReadTPolyMarker3D(obj);
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
      }
      else if (classname == "TStreamerSTL") {
         this.ReadStreamerSTL(obj);
      }
      else if (classname == "TStreamerObject" ||
            classname == "TStreamerObjectAny" ||
            classname == "TStreamerString" ||
            classname == "TStreamerObjectPointer" ) {
         this.ReadTStreamerObject(obj);
      }
      else if (classname == "TBasket") {
         this.ReadTBasket(obj);
      }
      else {
         var streamer = this.fFile.GetStreamer(classname);
         if (streamer != null)
            streamer.Stream(obj, this);
         else {
            JSROOT.console("Did not found streamer for class " + classname + " try to skip data");
            var ver = this.ReadVersion();
            this.CheckBytecount(ver);
         }
      }

      JSROOT.addMethods(obj);
   }


   // ==============================================================================

   // ctor
   JSROOT.TStreamer = function(file) {
      this.fFile = file;
      this._typename = "TStreamer";
      return this;
   }

   JSROOT.TStreamer.prototype.ReadBasicType = function(buf, obj, prop) {
      // read basic types (known from the streamer info)
      switch (this[prop]['type']) {
         case JSROOT.IO.kBase:
            break;
         case JSROOT.IO.kOffsetL:
            break;
         case JSROOT.IO.kOffsetP:
            break;
         case JSROOT.IO.kCharStar:
            var n_el = obj[this[prop]['cntname']];
            obj[prop] = buf.ReadBasicPointer(n_el, 'C');
            break;
         case JSROOT.IO.kChar:
         case JSROOT.IO.kLegacyChar:
            obj[prop] = buf.b.charCodeAt(buf.o++) & 0xff;
            break;
         case JSROOT.IO.kShort:
            obj[prop] = buf.ntoi2();
            break;
         case JSROOT.IO.kInt:
         case JSROOT.IO.kCounter:
            obj[prop] = buf.ntoi4();
            break;
         case JSROOT.IO.kLong:
            obj[prop] = buf.ntoi8();
            break;
         case JSROOT.IO.kFloat:
         case JSROOT.IO.kDouble32:
            obj[prop] = buf.ntof();
            if (Math.abs(obj[prop]) < 1e-300) obj[prop] = 0.0;
            break;
         case JSROOT.IO.kDouble:
            obj[prop] = buf.ntod();
            if (Math.abs(obj[prop]) < 1e-300) obj[prop] = 0.0;
            break;
         case JSROOT.IO.kUChar:
            obj[prop] = (buf.b.charCodeAt(buf.o++) & 0xff) >>> 0;
            break;
         case JSROOT.IO.kUShort:
            obj[prop] = buf.ntou2();
            break;
         case JSROOT.IO.kUInt:
            obj[prop] = buf.ntou4();
            break;
         case JSROOT.IO.kULong:
            obj[prop] = buf.ntou8();
            break;
         case JSROOT.IO.kBits:
            alert('failed to stream ' + prop + ' (' + this[prop]['typename'] + ')');
            break;
         case JSROOT.IO.kLong64:
            obj[prop] = buf.ntoi8();
            break;
         case JSROOT.IO.kULong64:
            obj[prop] = buf.ntou8();
            break;
         case JSROOT.IO.kBool:
            obj[prop] = (buf.b.charCodeAt(buf.o++) & 0xff) != 0;
            break;
         case JSROOT.IO.kFloat16:
            obj[prop] = 0;
            buf.o += 2;
            break;
         case JSROOT.IO.kAny:
         case JSROOT.IO.kAnyp:
         case JSROOT.IO.kObjectp:
         case JSROOT.IO.kObject:
            var classname = this[prop]['typename'];
            if (classname.charAt(classname.length-1) == "*")
               classname = classname.substr(0, classname.length - 1);

            obj[prop] = {};
            buf.ClassStreamer(obj[prop], classname);
            break;

         case JSROOT.IO.kAnyP:
         case JSROOT.IO.kObjectP:
            obj[prop] = buf.ReadObjectAny();
            break;
         case JSROOT.IO.kTString:
            obj[prop] = buf.ReadTString();
            break;
         case JSROOT.IO.kTObject:
            buf.ReadTObject(obj);
            break;
         case JSROOT.IO.kTNamed:
            buf.ReadTNamed(obj);
            break;
         case JSROOT.IO.kAnyPnoVT:
         case JSROOT.IO.kSTLp:
         case JSROOT.IO.kSkip:
         case JSROOT.IO.kSkipL:
         case JSROOT.IO.kSkipP:
         case JSROOT.IO.kConv:
         case JSROOT.IO.kConvL:
         case JSROOT.IO.kConvP:
         case JSROOT.IO.kSTL:
         case JSROOT.IO.kSTLstring:
         case JSROOT.IO.kStreamer:
         case JSROOT.IO.kStreamLoop:
            alert('failed to stream ' + prop + ' (' + this[prop]['typename'] + ')');
            break;
         case JSROOT.IO.kOffsetL+JSROOT.IO.kShort:
         case JSROOT.IO.kOffsetL+JSROOT.IO.kUShort:
            alert("Strange code was here????"); // var n_el = str.charCodeAt(o) & 0xff;
            var n_el  = this[prop]['length'];
            obj[prop] = buf.ReadFastArray(n_el, 'S');
            break;
         case JSROOT.IO.kOffsetL+JSROOT.IO.kInt:
            var n_el  = this[prop]['length'];
            obj[prop] = buf.ReadFastArray(n_el, 'I');
            break;
         case JSROOT.IO.kOffsetL+JSROOT.IO.kUInt:
            var n_el  = this[prop]['length'];
            obj[prop] = buf.ReadFastArray(n_el, 'U');
            break;
         case JSROOT.IO.kOffsetL+JSROOT.IO.kULong:
         case JSROOT.IO.kOffsetL+JSROOT.IO.kULong64:
            var n_el  = this[prop]['length'];
            obj[prop] = buf.ReadFastArray(n_el, 'LU');
            break;
         case JSROOT.IO.kOffsetL+JSROOT.IO.kLong:
         case JSROOT.IO.kOffsetL+JSROOT.IO.kLong64:
            var n_el  = this[prop]['length'];
            obj[prop] = buf.ReadFastArray(n_el, 'L');
            break;
         case JSROOT.IO.kOffsetL+JSROOT.IO.kFloat:
         case JSROOT.IO.kOffsetL+JSROOT.IO.kDouble32:
            //var n_el = str.charCodeAt(o) & 0xff;
            var n_el  = this[prop]['length'];
            obj[prop] = buf.ReadFastArray(n_el, 'F');
            break;
         case JSROOT.IO.kOffsetL+JSROOT.IO.kDouble:
            //var n_el = str.charCodeAt(o) & 0xff;
            var n_el  = this[prop]['length'];
            obj[prop] = buf.ReadFastArray(n_el, 'D');
            break;
         case JSROOT.IO.kOffsetP+JSROOT.IO.kChar:
            var n_el = obj[this[prop]['cntname']];
            obj[prop] = buf.ReadBasicPointer(n_el, 'C');
            break;
         case JSROOT.IO.kOffsetP+JSROOT.IO.kShort:
         case JSROOT.IO.kOffsetP+JSROOT.IO.kUShort:
            var n_el = obj[this[prop]['cntname']];
            obj[prop] = buf.ReadBasicPointer(n_el, 'S');
            break;
         case JSROOT.IO.kOffsetP+JSROOT.IO.kInt:
            var n_el = obj[this[prop]['cntname']];
            obj[prop] = buf.ReadBasicPointer(n_el, 'I');
            break;
         case JSROOT.IO.kOffsetP+JSROOT.IO.kUInt:
            var n_el = obj[this[prop]['cntname']];
            obj[prop] = buf.ReadBasicPointer(n_el, 'U');
            break;
         case JSROOT.IO.kOffsetP+JSROOT.IO.kULong:
         case JSROOT.IO.kOffsetP+JSROOT.IO.kULong64:
            var n_el = obj[this[prop]['cntname']];
            obj[prop] = buf.ReadBasicPointer(n_el, 'LU');
            break;
         case JSROOT.IO.kOffsetP+JSROOT.IO.kLong:
         case JSROOT.IO.kOffsetP+JSROOT.IO.kLong64:
            var n_el = obj[this[prop]['cntname']];
            obj[prop] = buf.ReadBasicPointer(n_el, 'L');
            break;
         case JSROOT.IO.kOffsetP+JSROOT.IO.kFloat:
         case JSROOT.IO.kOffsetP+JSROOT.IO.kDouble32:
            var n_el = obj[this[prop]['cntname']];
            obj[prop] = buf.ReadBasicPointer(n_el, 'F');
            break;
         case JSROOT.IO.kOffsetP+JSROOT.IO.kDouble:
            var n_el = obj[this[prop]['cntname']];
            obj[prop] = buf.ReadBasicPointer(n_el, 'D');
            break;
         default:
            alert('failed to stream ' + prop + ' (' + this[prop]['typename'] + ')');
            break;
      }
   }

   JSROOT.TStreamer.prototype.Stream = function(obj, buf) {

      var ver = buf.ReadVersion();

      // first base classes
      for (var prop in this) {
         if (!this[prop] || typeof(this[prop]) === "function")
            continue;
         if (this[prop]['typename'] === 'BASE') {
            var clname = this[prop]['class'];
            if (this[prop]['class'].indexOf("TArray") == 0) {
               var array_type = this[prop]['class'].charAt(6);
               var len = buf.ntou4();
               obj['fArray'] = buf.ReadFastArray(len, array_type);
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

         if (JSROOT.fUserStreamers !== null) {
            var user_func = JSROOT.fUserStreamers[prop_typename];

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
   }


   // ==============================================================================

   // A class that reads a TDirectory from a buffer.

   // ctor
   JSROOT.TDirectory = function(file, dirname, cycle) {
      if (! (this instanceof arguments.callee) ) {
         var error = new Error("you must use new to instantiate this class");
         error.source = "JSROOT.TDirectory.ctor";
         throw error;
      }

      this.fFile = file;
      this._typename = "TDirectory";
      this['dir_name'] = dirname;
      this['dir_cycle'] = cycle;
      this.fKeys = new Array();
      return this;
   }

   JSROOT.TDirectory.prototype.GetKey = function(keyname, cycle, call_back) {
      // retrieve a key by its name and cycle in the list of keys
      for (var i in this.fKeys) {
         if (this.fKeys[i]['fName'] == keyname && this.fKeys[i]['fCycle'] == cycle) {
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
         if ((dirkey!=null) && (typeof call_back == 'function') &&
              (dirkey['fClassName'].indexOf("TDirectory")==0)) {

            this.fFile.ReadObject(this['dir_name'] + "/" + dirname, 1, function(newdir) {
               if (newdir) newdir.GetKey(subname, cycle, call_back);
            });
            return null;
         }

         pos = keyname.lastIndexOf("/", pos-1);
      }


      JSROOT.CallBack(call_back, null);
      return null;
   }

   JSROOT.TDirectory.prototype.ReadKeys = function(readkeys_callback) {
      var thisdir = this;
      var file = this.fFile;

      //*-*-------------Read directory info
      var nbytes = this.fNbytesName + 22;
      nbytes += 4;  // fDatimeC.Sizeof();
      nbytes += 4;  // fDatimeM.Sizeof();
      nbytes += 18; // fUUID.Sizeof();
      // assume that the file may be above 2 Gbytes if file version is > 4
      if (file.fVersion >= 40000) nbytes += 12;

      file.Seek(this.fSeekDir, this.fFile.ERelativeTo.kBeg);
      file.ReadBuffer(nbytes, function(blob1) {
         if (blob1==null) return JSROOT.CallBack(readkeys_callback,null);
         var buf = new JSROOT.TBuffer(blob1, thisdir.fNbytesName, file);

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
            JSROOT.console("Cannot read directory info of file " + file.fURL);
            return JSROOT.CallBack(readkeys_callback, null);
         }
         //*-* -------------Read keys of the top directory

         if (thisdir.fSeekKeys <=0)
            return JSROOT.CallBack(readkeys_callback, null);

         file.Seek(thisdir.fSeekKeys, file.ERelativeTo.kBeg);
         file.ReadBuffer(thisdir.fNbytesKeys, function(blob2) {
            if (blob2 == null) return JSROOT.CallBack(readkeys_callback, null);

            var buf = new JSROOT.TBuffer(blob2, 0, file);

            var key = file.ReadKey(buf);

            var nkeys = buf.ntoi4();
            for (var i = 0; i < nkeys; i++) {
               key = file.ReadKey(buf);
               thisdir.fKeys.push(key);
            }
            file.fDirectories.push(thisdir);
            delete buf;

            JSROOT.CallBack(readkeys_callback, thisdir);
         });

         delete buf;
      });
   }

   JSROOT.TDirectory.prototype.StreamHeader = function(buf) {
      var version = buf.ntou2();
      var versiondir = version % 1000;
      buf.shift(8); // skip fDatimeC and fDatimeM
      this.fNbytesKeys = buf.ntou4();
      this.fNbytesName = buf.ntou4();
      this.fSeekDir = (version > 1000) ? buf.ntou8() : buf.ntou4();
      this.fSeekParent = (version > 1000) ? buf.ntou8() : buf.ntou4();
      this.fSeekKeys = (version > 1000) ? buf.ntou8() : buf.ntou4();
      if (versiondir > 2) buf.shift(18); // skip fUUID
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
      if (! (this instanceof arguments.callee) ) {
         var error = new Error("you must use new to instantiate this class");
         error.source = "JSROOT.TFile.ctor";
         throw error;
      }

      this._typename = "TFile";
      this.fOffset = 0;
      this.fEND = 0;
      this.fFullURL = url;
      this.fURL = url;
      this.fAcceptRanges = true; // when disabled ('+' at the end of file name), complete file content read with single operation
      this.fUseStampPar = true;  // use additional stamp parameter for file name to avoid browser caching problem
      this.fFileContent = ""; // this can be full or parial content of the file (if ranges are not supported or if 1K header read from file)

      this.ERelativeTo = { kBeg : 0, kCur : 1, kEnd : 2 };
      this.fDirectories = new Array();
      this.fKeys = new Array();
      this.fSeekInfo = 0;
      this.fNbytesInfo = 0;
      this.fTagOffset = 0;
      this.fStreamers = 0;
      this.fStreamerInfos = null;
      this.fFileName = "";
      this.fStreamers = new Array;

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

         var xhr = JSROOT.NewHttpRequest(this.fURL, "head", function(res) {
            if (res==null)
               return JSROOT.CallBack(newfile_callback, null);

            var accept_ranges = res.getResponseHeader("Accept-Ranges");
            if (accept_ranges==null) file.fAcceptRanges = false;
            var len = res.getResponseHeader("Content-Length");
            if (len!=null) file.fEND = parseInt(len);
            else file.fAcceptRanges = false;
            file.ReadKeys(newfile_callback);
         });

         xhr.send(null);
      }

      return this;
   }

   JSROOT.TFile.prototype.ReadBuffer = function(len, callback) {

      if ((this.fFileContent.length>0) && (!this.fAcceptRanges || (this.fOffset+len <= this.fFileContent.length)))
         return callback(this.fFileContent.substr(this.fOffset, len));

      var file = this;

      var url = this.fURL;
      if (this.fUseStampPar) {
         // try to avoid browser caching by adding stamp parameter to URL
         if (url.indexOf('?')>0) url+="&stamp="; else url += "?stamp=";
         var d = new Date;
         url += d.getTime();
      }

      function read_callback(res) {
         if ((res==null) && file.fUseStampPar && (file.fOffset==0)) {
            // if fail to read file with stamp parameter, try once to avoid it
            file.fUseStampPar = false;
            var xhr2 = JSROOT.NewHttpRequest(this.fURL, "bin", read_callback);
            if (this.fAcceptRanges)
               xhr2.setRequestHeader("Range", "bytes=" + this.fOffset + "-" + (this.fOffset + len - 1));
            xhr2.send(null);
            return;
         } else
         if ((res!=null) && (file.fOffset==0) && (file.fFileContent.length == 0)) {
            // special case - read content all at once
            file.fFileContent = res;
            if (!this.fAcceptRanges) {
               file.fEND = res.length;
               res = file.fFileContent.substr(file.fOffset, len);
            }
         }

         callback(res);
      }

      var xhr = JSROOT.NewHttpRequest(url, "bin", read_callback);
      if (this.fAcceptRanges)
         xhr.setRequestHeader("Range", "bytes=" + this.fOffset + "-" + (this.fOffset + len - 1));
      xhr.send(null);
   }

   JSROOT.TFile.prototype.Seek = function(offset, pos) {
      // Set position from where to start reading.
      switch (pos) {
         case this.ERelativeTo.kBeg:
            this.fOffset = offset;
            break;
         case this.ERelativeTo.kCur:
            this.fOffset += offset;
            break;
         case this.ERelativeTo.kEnd:
            // this option is not used currently in the ROOT code
            if (this.fEND == 0)
               throw  "Seek : seeking from end in file with fEND==0 is not supported";
            this.fOffset = this.fEND - offset;
            break;
         default:
            throw  "Seek : unknown seek option (" + pos + ")";
            break;
      }
   }

   JSROOT.TFile.prototype.ReadKey = function(buf) {
      // read key from buffer
      var key = {};
      buf.ReadTKey(key);
      return key;
   }

   JSROOT.TFile.prototype.GetDir = function(dirname, cycle) {
      // check first that directory with such name exists

      if ((cycle==null) && (typeof dirname == 'string')) {
         var pos = dirname.lastIndexOf(';');
         if (pos>0) { cycle = dirname.substr(pos+1); dirname = dirname.substr(0,pos); }
      }

      for (var j in this.fDirectories) {
         var dir = this.fDirectories[j];
         if (dir['dir_name'] != dirname) continue;
         if ((cycle!=null) && (dir['dir_cycle']!=cycle)) continue;
         return dir;
      }
      return null;
   }

   JSROOT.TFile.prototype.GetKey = function(keyname, cycle, getkey_callback) {
      // retrieve a key by its name and cycle in the list of keys
      // one should call_back when keys must be read first from the directory

      for (var i in this.fKeys) {
         if (this.fKeys[i]['fName'] == keyname && this.fKeys[i]['fCycle'] == cycle) {
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
         if ((dirkey!=null) && (getkey_callback != null) &&
             (dirkey['fClassName'].indexOf("TDirectory")==0)) {

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

      this.Seek(key['fSeekKey'] + key['fKeylen'], this.ERelativeTo.kBeg);
      this.ReadBuffer(key['fNbytes'] - key['fKeylen'], function(blob1) {

         if (blob1==null) callback(null);

         var buf = null;

         if (key['fObjlen'] <= key['fNbytes']-key['fKeylen']) {
            buf = new JSROOT.TBuffer(blob1, 0, file);
         } else {
            var hdrsize = JSROOT.R__unzip_header(blob1, 0);
            if (hdrsize<0) return callback(null);
            var objbuf = JSROOT.R__unzip(hdrsize, blob1, 0);
            buf = new JSROOT.TBuffer(objbuf, 0, file);
         }

         buf.fTagOffset = key.fKeylen;
         callback(buf);
         delete buf;
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

         if ((obj_name=="StreamerInfo") && (key['fClassName']=="TList"))
            return file.fStreamerInfos;

         var isdir = false;
         if ((key['fClassName'] == 'TDirectory' || key['fClassName'] == 'TDirectoryFile')) {
            isdir = true;
            var dir = file.GetDir(obj_name, cycle);
            if (dir!=null)
               return JSROOT.CallBack(user_call_back, dir);
         }

         file.ReadObjBuffer(key, function(buf) {
            if (!buf) return JSROOT.CallBack(user_call_back, null);

            if (isdir) {
               var dir = new JSROOT.TDirectory(file, obj_name, cycle);
               dir.StreamHeader(buf);
               if (dir.fSeekKeys) {
                  dir.ReadKeys(user_call_back);
               } else {
                  JSROOT.CallBack(user_call_back,dir);
               }

               return;
            }

            var obj = {};
            buf.MapObject(1, obj); // tag object itself with id==1
            buf.ClassStreamer(obj, key['fClassName']);

            JSROOT.CallBack(user_call_back, obj);
         }); // end of ReadObjBuffer callback
      }); // end of GetKey callback
   }

   JSROOT.TFile.prototype.ExtractStreamerInfos = function(buf)
   {
      if (!buf) return;

      var lst = {};
      buf.MapObject(1, lst);
      buf.ClassStreamer(lst, 'TList');

      lst['_typename'] = "TStreamerInfoList";

      this.fStreamerInfos = lst;
   }

   JSROOT.TFile.prototype.ReadFormulas = function()
   {
      for (var i in this.fKeys)
        if (this.fKeys[i]['fClassName'] == 'TFormula')
          this.ReadObject(this.fKeys[i]['fName'], this.fKeys[i]['fCycle'], function(obj) {
               JSROOT.addFormula(obj);
         });
   }

   JSROOT.TFile.prototype.ReadStreamerInfos = function(si_callback)
   {
      if (this.fSeekInfo == 0 || this.fNbytesInfo == 0) return si_callback(null);
      this.Seek(this.fSeekInfo, this.ERelativeTo.kBeg);

      var file = this;

      file.ReadBuffer(file.fNbytesInfo, function(blob1) {
         var buf = new JSROOT.TBuffer(blob1, 0, file);
         var key = file.ReadKey(buf);
         if (key == null) return si_callback(null);
         file.fKeys.push(key);

         file.ReadObjBuffer(key, function(blob2) {
            if (blob2==null) return si_callback(null);
            file.ExtractStreamerInfos(blob2);
            file.ReadFormulas();
            si_callback(file);
         });
      });
   }

   JSROOT.TFile.prototype.ReadKeys = function(readkeys_callback) {
      // read keys only in the root file

      var file = this;

      // with the first readbuffer we read bigger amount to create header cache
      this.ReadBuffer(1024, function(blob1) {
         if (blob1==null) return JSROOT.CallBack(readkeys_callback, null);

         if (blob1.substring(0, 4)!='root') {
            alert("NOT A ROOT FILE! " + file.fURL);
            return JSROOT.CallBack(readkeys_callback, null);
         }

         var buf = new JSROOT.TBuffer(blob1, 4, file); // skip the "root" file identifier
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
         if (!file.fSeekInfo && !file.fNbytesInfo)
            return JSROOT.CallBack(readkeys_callback, null);

         //*-*-------------Read directory info
         var nbytes = file.fNbytesName + 22;
         nbytes += 4;  // fDatimeC.Sizeof();
         nbytes += 4;  // fDatimeM.Sizeof();
         nbytes += 18; // fUUID.Sizeof();
         // assume that the file may be above 2 Gbytes if file version is > 4
         if (file.fVersion >= 40000) nbytes += 12;

         file.Seek(file.fBEGIN, file.ERelativeTo.kBeg);

         file.ReadBuffer(Math.max(300, nbytes), function(blob3) {
            if (blob3==null) return JSROOT.CallBack(readkeys_callback, null);

            var buf3 = new JSROOT.TBuffer(blob3, file.fNbytesName, file);

            // we call TDirectory method while TFile is just derived class
            JSROOT.TDirectory.prototype.StreamHeader.call(file, buf3);

            //*-*---------read TKey::FillBuffer info
            buf3.o = 4; // Skip NBytes;
            var keyversion = buf3.ntoi2();
            // Skip ObjLen, DateTime, KeyLen, Cycle, SeekKey, SeekPdir
            if (keyversion > 1000) buf3.shift(28); // Large files
                              else buf3.shift(20);
            buf3.ReadTString();
            buf3.ReadTString();
            file.fTitle = buf3.ReadTString();
            if (file.fNbytesName < 10 || this.fNbytesName > 10000) {
               JSROOT.console("Init : cannot read directory info of file " + file.fURL);
               return JSROOT.CallBack(readkeys_callback, null);
            }
            //*-* -------------Read keys of the top directory

            if (file.fSeekKeys <= 0) {
               JSROOT.console("Empty keys list - not supported" + file.fURL);
               return JSROOT.CallBack(readkeys_callback, null);
            }

            file.Seek(file.fSeekKeys, file.ERelativeTo.kBeg);
            file.ReadBuffer(file.fNbytesKeys, function(blob4) {
               if (blob4==null) return JSROOT.CallBack(readkeys_callback, null);

               var buf4 = new JSROOT.TBuffer(blob4, 0, file);

               var key = file.ReadKey(buf4);

               var nkeys = buf4.ntoi4();
               for (var i = 0; i < nkeys; i++) {
                  key = file.ReadKey(buf4);
                  file.fKeys.push(key);
               }
               file.ReadStreamerInfos(readkeys_callback);
               delete buf4;
            });
            delete buf3;
         });
         delete buf;
      });
   }

   JSROOT.TFile.prototype.ReadDirectory = function(dir_name, cycle, readdir_callback) {
      // read the directory content from  a root file
      // do not read directory if it is already exists

      return this.ReadObject(dir_name, cycle, readdir_callback);
   }

   JSROOT.TFile.prototype.GetStreamer = function(clname) {
      // return the streamer for the class 'clname', from the list of streamers
      // or generate it from the streamer infos and add it to the list

      var streamer = this.fStreamers[clname];
      if (typeof(streamer) != 'undefined') return streamer;

      var s_i;

      if (this.fStreamerInfos)
         for (var i in this.fStreamerInfos.arr)
            if (this.fStreamerInfos.arr[i].fName == clname)  {
               s_i = this.fStreamerInfos.arr[i];
               break;
            }
      if (typeof s_i == 'undefined') return null;

      this.fStreamers[clname] = new JSROOT.TStreamer(this);
      if (typeof(s_i['fElements']) != 'undefined') {
         var n_el = s_i['fElements']['arr'].length;
         for (var j=0;j<n_el;++j) {
            var element = s_i['fElements']['arr'][j];
            if (element['typename'] === 'BASE') {
               // generate streamer for the base classes
               this.GetStreamer(element['fName']);
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
            streamer['class']    = element['fName'];
            streamer['cntname']  = element['countName'];
            streamer['type']     = element['type'];
            streamer['length']   = element['length'];

            this.fStreamers[clname][element['fName']] = streamer;
         }
      }
      return this.fStreamers[clname];
   }

   JSROOT.TFile.prototype.Delete = function() {
      if (this.fDirectories) this.fDirectories.splice(0, this.fDirectories.length);
      this.fDirectories = null;
      if (this.fKeys) this.fKeys.splice(0, this.fKeys.length);
      this.fKeys = null;
      if (this.fStreamers) this.fStreamers.splice(0, this.fStreamers.length);
      this.fStreamers = null;
      this.fSeekInfo = 0;
      this.fNbytesInfo = 0;
      this.fTagOffset = 0;
   }

})();

// JSRootIOEvolution.js ends

