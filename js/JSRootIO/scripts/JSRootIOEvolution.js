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

(function(){

   if (typeof JSROOTIO == "object"){
      var e1 = new Error("JSROOTIO is already defined");
      e1.source = "JSROOTIO.core.js";
      throw e1;
   }

   var kByteCountMask = 0x40000000;

   JSROOTIO = {};

   JSROOTIO.version = "1.8 2012/11/28";

   JSROOTIO.BIT = function(bits, index) {
      var mask = 1 << index;
      return (bits & mask);
   };

   JSROOTIO.ntou2 = function(b, o) {
      // convert (read) two bytes of buffer b into a UShort_t
      var n  = ((b.charCodeAt(o)   & 0xff) << 8) >>> 0;
          n +=  (b.charCodeAt(o+1) & 0xff) >>> 0;
      return n;
   };

   JSROOTIO.ntou4 = function(b, o) {
      // convert (read) four bytes of buffer b into a UInt_t
      var n  = ((b.charCodeAt(o)   & 0xff) << 24) >>> 0;
          n += ((b.charCodeAt(o+1) & 0xff) << 16) >>> 0;
          n += ((b.charCodeAt(o+2) & 0xff) << 8)  >>> 0;
          n +=  (b.charCodeAt(o+3) & 0xff) >>> 0;
      return n;
   };

   JSROOTIO.ntou8 = function(b, o) {
      // convert (read) eight bytes of buffer b into a ULong_t
      var n  = ((b.charCodeAt(o)   & 0xff) << 56) >>> 0;
          n += ((b.charCodeAt(o+1) & 0xff) << 48) >>> 0;
          n += ((b.charCodeAt(o+2) & 0xff) << 40) >>> 0;
          n += ((b.charCodeAt(o+3) & 0xff) << 32) >>> 0;
          n += ((b.charCodeAt(o+4) & 0xff) << 24) >>> 0;
          n += ((b.charCodeAt(o+5) & 0xff) << 16) >>> 0;
          n += ((b.charCodeAt(o+6) & 0xff) << 8) >>> 0;
          n +=  (b.charCodeAt(o+7) & 0xff) >>> 0;
      return n;
   };

   JSROOTIO.ntoi2 = function(b, o) {
      // convert (read) two bytes of buffer b into a Short_t
      var n  = (b.charCodeAt(o)   & 0xff) << 8;
          n += (b.charCodeAt(o+1) & 0xff);
      return n;
   };

   JSROOTIO.ntoi4 = function(b, o) {
      // convert (read) four bytes of buffer b into a Int_t
      var n  = (b.charCodeAt(o)   & 0xff) << 24;
          n += (b.charCodeAt(o+1) & 0xff) << 16;
          n += (b.charCodeAt(o+2) & 0xff) << 8;
          n += (b.charCodeAt(o+3) & 0xff);
      return n;
   };

   JSROOTIO.ntoi8 = function(b, o) {
      // convert (read) eight bytes of buffer b into a Long_t
      var n  = (b.charCodeAt(o)   & 0xff) << 56;
          n += (b.charCodeAt(o+1) & 0xff) << 48;
          n += (b.charCodeAt(o+2) & 0xff) << 40;
          n += (b.charCodeAt(o+3) & 0xff) << 32;
          n += (b.charCodeAt(o+4) & 0xff) << 24;
          n += (b.charCodeAt(o+5) & 0xff) << 16;
          n += (b.charCodeAt(o+6) & 0xff) << 8;
          n += (b.charCodeAt(o+7) & 0xff);
      return n;
   };

   JSROOTIO.ntof = function(b, o) {
      // IEEE-754 Floating-Point Conversion (single precision - 32 bits)
      var inString = b.substring(o, o + 4);
      inString = inString.toString();
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

   JSROOTIO.ntod = function(b, o) {
      // IEEE-754 Floating-Point Conversion (double precision - 64 bits)
      var inString = b.substring(o, o + 8);
      inString = inString.toString();
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

   JSROOTIO.ReadArray = function(str, o, array_type) {
      // Read array of floats from the I/O buffer
      var array = {}
      array['array'] = new Array();
      var n = JSROOTIO.ntou4(str, o); o += 4;
      if (array_type == 'D') {
         for (var i = 0; i < n; ++i) {
            array['array'][i] = JSROOTIO.ntod(str, o); o += 8;
         }
      }
      if (array_type == 'F') {
         for (var i = 0; i < n; ++i) {
            array['array'][i] = JSROOTIO.ntof(str, o); o += 4;
         }
      }
      if (array_type == 'L') {
         for (var i = 0; i < n; ++i) {
            array['array'][i] = JSROOTIO.ntoi8(str, o); o += 8;
         }
      }
      if (array_type == 'I') {
         for (var i = 0; i < n; ++i) {
            array['array'][i] = JSROOTIO.ntoi4(str, o); o += 4;
         }
      }
      if (array_type == 'S') {
         for (var i = 0; i < n; ++i) {
            array['array'][i] = JSROOTIO.ntoi2(str, o); o += 2;
         }
      }
      if (array_type == 'C') {
         for (var i = 0; i < n; ++i) {
            array['array'][i] = str.charCodeAt(o) & 0xff; o++;
         }
      }
      array['off'] = o;
      return array;
   };

   JSROOTIO.ReadStaticArray = function(str, o) {
      // read array of integers from the I/O buffer
      var array = {}
      array['array'] = new Array();
      var n = JSROOTIO.ntou4(str, o); o += 4;
      for (var i = 0; i < n; ++i) {
         array['array'][i] = JSROOTIO.ntou4(str, o); o += 4;
      }
      array['off'] = o;
      return array;
   };

   JSROOTIO.ReadFastArray = function(str, o, n, array_type) {
      // read array of n integers from the I/O buffer
      var array = {};
      array['array'] = new Array();
      if (array_type == 'D') {
         for (var i = 0; i < n; ++i) {
            array['array'][i] = JSROOTIO.ntod(str, o); o += 8;
         }
      }
      else if (array_type == 'F') {
         for (var i = 0; i < n; ++i) {
            array['array'][i] = JSROOTIO.ntof(str, o); o += 4;
         }
      }
      else if (array_type == 'L') {
         for (var i = 0; i < n; ++i) {
            array['array'][i] = JSROOTIO.ntoi8(str, o); o += 8;
         }
      }
      else if (array_type == 'I') {
         for (var i = 0; i < n; ++i) {
            array['array'][i] = JSROOTIO.ntoi4(str, o); o += 4;
         }
      }
      else if (array_type == 'S') {
         for (var i = 0; i < n; ++i) {
            array['array'][i] = JSROOTIO.ntoi2(str, o); o += 2;
         }
      }
      else if (array_type == 'C') {
         for (var i = 0; i < n; ++i) {
            array['array'][i] = str.charCodeAt(o) & 0xff; o++;
         }
      }
      else if (array_type == 'TString') {
         for (var i = 0; i < n; ++i) {
            var so = JSROOTIO.ReadTString(str, o); o = so['off'];
            array['array'][i] = so['str'];
         }
      }
      else {
         for (var i = 0; i < n; ++i) {
            array['array'][i] = JSROOTIO.ntou4(str, o); o += 4;
         }
      }
      array['off'] = o;
      return array;
   };

   JSROOTIO.ReadBasicPointerElem = function(str, o, n, array_type) {
      var isArray = str.charCodeAt(o) & 0xff; o++;
      if (isArray) {
         var array = JSROOTIO.ReadFastArray(str, o, n, array_type);
         return array;
      }
      else {
         o--;
         var array = JSROOTIO.ReadFastArray(str, o, n, array_type);
         if (n == 0) array['off']++;
         return array;
      }
      return null;
   };

   JSROOTIO.ReadBasicPointer = function(str, o, len, array_type) {
      return JSROOTIO.ReadBasicPointerElem(str, o, len, array_type);
   };

   JSROOTIO.ReadTString = function(str, off) {
      // stream a TString object from buffer
      var len = str.charCodeAt(off) & 0xff;
      off++;
      if (len == 255) {
         // large string
         len = JSROOTIO.ntou4(str, off);
         off += 4;
      }
      return {
         'off' : off + len,
         'str' : (str.charCodeAt(off) == 0) ? '' : str.substring(off, off + len)
      };
   };

   JSROOTIO.ReadString = function(str, off, max_len) {
      // stream a string from buffer
      max_len = typeof(max_len) != 'undefined' ? max_len : 0;
      var len = 0;
      while ((str.charCodeAt(off + len) & 0xff) != 0) {
         len++;
         if ((max_len > 0) && (len >= max_len))
            break;
      }
      return {
         'off' : off + len + 1,
         'str' : (str.charCodeAt(off) == 0) ? '' : str.substring(off, off + len)
      };
   };

   JSROOTIO.ReadObjectAny = function(str, o, previous) {
      var obj = {};
      var class_name = previous;
      var startpos = o;
      var clRef = gFile.fStreamerInfo.ReadClass(str, o);
      o = clRef['off'];
      // use clRef['name'] and use previous name if == -1
      if (clRef['name'] && clRef['name'] != -1) {
         class_name = clRef['name'];
      }
      else if (!clRef['name'] && clRef['tag']) {
         class_name = gFile.fStreamerInfo.GetClassMap(clRef['tag']);
         if (class_name != -1)
            obj['_typename'] = 'JSROOTIO.' + class_name;
         class_name = 0;
      }
      if (class_name && class_name != '' && class_name != -1) {
         if (class_name == 'TObject' || class_name == 'TMethodCall') {
            o += 2; // skip version
            o += 4; // skip unique id
            obj['fBits'] = JSROOTIO.ntou4(str, o); o += 4;
         }
         else if (class_name == 'TObjArray') {
            var array = JSROOTIO.ReadTObjArray(str, o);
            obj = array['array'];
            o = array['off'];
         }
         else if (JSROOTIO.GetStreamer(class_name)) {
            o = JSROOTIO.GetStreamer(class_name).Stream(obj, str, o);
         }
         obj['_typename'] = 'JSROOTIO.' + class_name;
         JSROOTCore.addMethods(obj);
         if (clRef['tag'])
            gFile.MapObject(obj, clRef['tag']);
         gFile.MapObject(obj, (gFile.fTagOffset + startpos + 2) | 0x01);
      }
      else if (clRef['name'] === 0 && clRef['tag'] != 0) {
         // already seen (and read) object...
         var ro = gFile.GetMappedObject(clRef['tag']+4);
         if (ro) {
            obj = JSROOTCore.clone(ro);
            class_name = obj['_typename'];
            class_name = class_name.replace('JSROOTIO.', '');
         }
      }
      else {
         // simply skip empty object...
         //o += 2; // skip version
         //o += 4; // skip unique id
         //obj['fBits'] = JSROOTIO.ntou4(str, o); o += 4;
      }
      return {
         'cln' : class_name,
         'off' : o,
         'obj' : obj
      };
   };

   JSROOTIO.ReadTList = function(str, o) {
      // stream all objects in the list from the I/O buffer
      var list = {};
      list['name'] = "";
      list['array'] = new Array();
      var ver = JSROOTIO.ReadVersion(str, o);
      o = ver['off'];
      if (ver['val'] > 3) {
         o += 2; // skip version
         o += 8; // object bits & unique id
         var so = JSROOTIO.ReadTString(str, o); o = so['off'];
         list['name'] = so['str'];
         var nobjects = JSROOTIO.ntou4(str, o); o += 4;
         var class_name = '';
         var idx = 0;
         for (var i = 0; i < nobjects; ++i) {
            obj = this.ReadObjectAny(str, o, class_name);
            o = obj['off'];
            if (obj['cln'] && obj['cln'] != '' && obj['cln'] != -1) {
               class_name = obj['cln'];
               list['array'][idx] = obj['obj'];
               ++idx;
            }

            var nch = str.charCodeAt(o) & 0xff; o++;
            if (ver['val'] > 4 && nch == 255)  {
               nbig= JSROOTIO.ntou4(str, o); o += 4;
            } else {
               nbig = nch;
            }
            if (nbig > 0) {
               var readOption = JSROOTIO.ReadString(str, o, nbig);
               // add the drawing option somewhere...
               //obj['obj']['_options'] = readOption['str'];
               obj['obj']['fOption'] = readOption['str'];
               o += nbig;
            }
         }
      }
      list['off'] = o;
      return list;
   };

   JSROOTIO.ReadTObjArray = function(str, o) {
      var list = {};
      list['name'] = "";
      list['array'] = new Array();
      var ver = JSROOTIO.ReadVersion(str, o);
      o = ver['off'];
      if (ver['val'] > 2)
         o += 10; // skip object bits & unique id
      if (ver['val'] > 1) {
         var so = JSROOTIO.ReadTString(str, o); o = so['off'];
         list['name'] = so['str'];
      }
      var nobjects = JSROOTIO.ntou4(str, o); o += 4;
      var lowerbound = JSROOTIO.ntou4(str, o); o += 4;
      var class_name = '';
      for (var i = 0; i < nobjects; i++) {
         obj = this.ReadObjectAny(str, o, class_name);
         o = obj['off'];
         list['array'][i] = 0;
         if (obj['cln'] && obj['cln'] != '' && obj['cln'] != -1) {
            class_name = obj['cln'];
            list['array'][i] = obj['obj'];
         }
      }
      list['off'] = o;
      return list;
   };

   JSROOTIO.ReadTClonesArray = function(str, o) {
      var list = {};
      list['name'] = "";
      list['array'] = new Array();
      var ver = JSROOTIO.ReadVersion(str, o);
      o = ver['off'];
      if (ver['val'] > 2)
         o += 10; // skip object bits & unique id
      if (ver['val'] > 1) {
         var so = JSROOTIO.ReadTString(str, o); o = so['off'];
         list['name'] = so['str'];
      }
      var so = JSROOTIO.ReadTString(str, o); o = so['off'];
      var s = so['str'];
      var classv = s;
      var clv = 0;
      var pos = s.indexOf(";");
      if (pos != -1) {
         classv = s.slice(0, pos);
         s = s.slice(pos+1, s.length()-pos-1);
         clv = parseInt(s);
      }
      var nobjects = JSROOTIO.ntou4(str, o); o += 4;
      if (nobjects < 0) nobjects = -nobjects;  // for backward compatibility
      var lowerbound = JSROOTIO.ntou4(str, o); o += 4;
      for (var i = 0; i < nobjects; i++) {
         if (classv == 'TObject' || classv == 'TMethodCall') {
            o += 10; // skip object bits & unique id
         }
         else {
            var obj = {};
            obj['_typename'] = 'JSROOTIO.' + classv;
            if (JSROOTIO.GetStreamer(classv))
               o = JSROOTIO.GetStreamer(classv).Stream(obj, str, o);
            list['array'][i] = obj;
            JSROOTCore.addMethods(obj);
         }
      }
      list['off'] = o;
      return list;
   };

   JSROOTIO.ReadTCollection = function(str, o) {
      var list = {};
      list['name'] = "";
      list['array'] = new Array();
      var ver = JSROOTIO.ReadVersion(str, o);
      o = ver['off'];
      if (ver['val'] > 2)
         o += 10; // skip object bits & unique id
      if (ver['val'] > 1) {
         var so = JSROOTIO.ReadTString(str, o); o = so['off'];
         list['name'] = so['str'];
      }
      var nobjects = JSROOTIO.ntou4(str, o); o += 4;
      for (var i = 0; i < nobjects; i++) {
         o += 10; // skip object bits & unique id
      }
      list['off'] = o;
      return list;
   };

   JSROOTIO.ReadTHashList = function(str, o) {
      var list = JSROOTIO.ReadTList(str, o);
      return list;
   };

   JSROOTIO.ReadVersion = function(str, o) {
      // read class version from I/O buffer
      var version = {};
      o += 4; // skip byte count
      version['val'] = JSROOTIO.ntou2(str, o); o += 2;
      version['off'] = o;
      return version;
   };

   JSROOTIO.ReadTNamed = function(str, o) {
      // read a TNamed class definition from I/O buffer
      var named = {};
      var ver = JSROOTIO.ReadVersion(str, o);
      o = ver['off'];
      o += 2; // skip version
      o += 4; // skip unique id
      named['fBits'] = JSROOTIO.ntou4(str, o); o += 4;
      var so = JSROOTIO.ReadTString(str, o); o = so['off'];
      named['name'] = so['str'];
      so = JSROOTIO.ReadTString(str, o); o = so['off'];
      named['title'] = so['str'];
      named['off'] = o;
      return named;
   };

   JSROOTIO.ReadTCanvas = function(str, o) {
      var ver = JSROOTIO.ReadVersion(str, o);
      o = ver['off'];
      var obj = {};
      obj['_typename'] = 'JSROOTIO.TCanvas';
      gFile.ClearObjectMap();
      if (JSROOTIO.GetStreamer('TPad')) {
         o = JSROOTIO.GetStreamer('TPad').Stream(obj, str, o);
      }
      return obj;
   };

   JSROOTIO.GetStreamer = function(clname) {
      // return the streamer for the class 'clname', from the list of streamers
      // or generate it from the streamer infos and add it to the list
      var i, j, n_el;
      if (typeof(gFile.fStreamers[clname]) != 'undefined')
         return gFile.fStreamers[clname];
      var s_i = gFile.fStreamerInfo.fStreamerInfos[clname];
      if (typeof(s_i) === 'undefined')
         return 0;
      gFile.fStreamers[clname] = new JSROOTIO.TStreamer(gFile);
      if (typeof(s_i['elements']) != 'undefined') {
         n_el = s_i['elements']['array'].length;
         for (j=0;j<n_el;++j) {
            var element = s_i['elements']['array'][j];
            if (element['typename'] === 'BASE') {
               // generate streamer for the base classes
               JSROOTIO.GetStreamer(element['name']);
            }
         }
      }
      if (typeof(s_i['elements']) != 'undefined') {
         n_el = s_i['elements']['array'].length;
         for (j=0;j<n_el;++j) {
            // extract streamer info for each class member
            var element = s_i['elements']['array'][j];
            gFile.fStreamers[clname][element['name']] = {};
            gFile.fStreamers[clname][element['name']]['typename'] = element['typename'];
            gFile.fStreamers[clname][element['name']]['class']    = element['name'];
            gFile.fStreamers[clname][element['name']]['cntname']  = s_i['elements']['array'][j]['countName'];
            gFile.fStreamers[clname][element['name']]['type']     = element['type'];
            gFile.fStreamers[clname][element['name']]['length']   = element['length'];
         }
      }
      return gFile.fStreamers[clname];
   };

   JSROOTIO.Print = function(str, what) {
      what = typeof(what) === 'undefined' ? 'info' : what;
      if ( (window['console'] !== undefined) ) {
         if (console[what] !== undefined) console[what](str + '\n');
      }
   };

})();

/// JSROOTIO.core.js ends


// JSROOTIO.TStreamer.js
//
// A TStreamer base class.
// Depends on the JSROOTIO library functions.
//

(function(){

   var version = "1.10 2012/11/28";

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

      JSROOTIO.TStreamer.prototype.ReadBasicType = function(str, o, obj, prop) {

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
               var array = JSROOTIO.ReadBasicPointer(str, o, n_el, 'C');
               obj[prop] = array['array'];
               o = array['off'];
               break;
            case kChar:
            case kLegacyChar:
               obj[prop] = str.charCodeAt(o) & 0xff; o++;
               break;
            case kShort:
               obj[prop] = JSROOTIO.ntoi2(str, o); o += 2;
               break;
            case kInt:
            case kCounter:
               obj[prop] = JSROOTIO.ntoi4(str, o); o += 4;
               break;
            case kLong:
               obj[prop] = JSROOTIO.ntoi8(str, o); o += 8;
               break;
            case kFloat:
            case kDouble32:
               obj[prop] = JSROOTIO.ntof(str, o); o += 4;
               break;
            case kDouble:
               obj[prop] = JSROOTIO.ntod(str, o); o += 8;
               break;
            case kUChar:
               obj[prop] = (str.charCodeAt(o) & 0xff) >>> 0; o++;
               break;
            case kUShort:
               obj[prop] = JSROOTIO.ntou2(str, o); o += 2;
               break;
            case kUInt:
               obj[prop] = JSROOTIO.ntou4(str, o); o += 4;
               break;
            case kULong:
               obj[prop] = JSROOTIO.ntou8(str, o); o += 8;
               break;
            case kBits:
               alert('failed to stream ' + prop + ' (' + this[prop]['typename'] + ')');
               break;
            case kLong64:
               obj[prop] = JSROOTIO.ntoi8(str, o); o += 8;
               break;
            case kULong64:
               obj[prop] = JSROOTIO.ntou8(str, o); o += 8;
               break;
            case kBool:
               obj[prop] = str.charCodeAt(o) & 0xff; o++;
               break;
            case kFloat16:
               //obj[prop] = JSROOTIO.ntof(str, o); o += 4;
               o += 2;
               break;
            case kObject:
               classname = this[prop]['typename'];
               if (JSROOTIO.GetStreamer(classname)) {
                  var clRef = gFile.fStreamerInfo.ReadClass(str, o);
                  if (clRef && clRef['name']) o = clRef['off'];
                  obj[prop] = new Object();
                  obj[prop]['_typename'] = 'JSROOTIO.' + classname;
                  o = JSROOTIO.GetStreamer(classname).Stream(obj[prop], str, o);
                  JSROOTCore.addMethods(obj[prop]);
               }
               break;
            case kAny:
               break;
            case kObjectp:
            case kObjectP:
            case kAnyp:
            case kAnyP:
               classname = this[prop]['typename'];
               if (classname == "TObject*") {
                  obj[prop] = JSROOTIO.ntou4(str, o);
                  o += 4;
                  break;
               }
               if (classname.endsWith("*")) {
                  classname = classname.substr(0, classname.length - 1);
                  pval = JSROOTIO.ntou4(str, o);
                  if (pval == 0) {
                     o += 4;
                     break;
                  }
                  if (pval < 1000000) {
                     o += 4;
                     break;
                  }
               }
               if (JSROOTIO.GetStreamer(classname)) {
                  var clRef = gFile.fStreamerInfo.ReadClass(str, o);
                  if (clRef && clRef['name']) {
                     o = clRef['off'];
                     classname = clRef['name'];
                  }
                  obj[prop] = new Object();
                  obj[prop]['_typename'] = 'JSROOTIO.' + classname;
                  o = JSROOTIO.GetStreamer(classname).Stream(obj[prop], str, o);
                  JSROOTCore.addMethods(obj[prop]);
               }
               break;
            case kTString:
               var so = JSROOTIO.ReadTString(str, o); o = so['off'];
               obj[prop] = so['str'];
               break;
            case kTObject:
               o += 2; // skip version
               o += 4; // skip unique id
               obj['fBits'] = JSROOTIO.ntou4(str, o); o += 4;
               break;
            case kTNamed:
               var named = JSROOTIO.ReadTNamed(str, o);
               o = named['off'];
               obj['fName'] = named['name'];
               obj['fTitle'] = named['title'];
               break;
            case kAnyPnoVT:
               alert('failed to stream ' + prop + ' (' + this[prop]['typename'] + ')');
               break;
            case kSTLp:
               alert('failed to stream ' + prop + ' (' + this[prop]['typename'] + ')');
               break;
            case kSkip:
               alert('failed to stream ' + prop + ' (' + this[prop]['typename'] + ')');
               break;
            case kSkipL:
               alert('failed to stream ' + prop + ' (' + this[prop]['typename'] + ')');
               break;
            case kSkipP:
               alert('failed to stream ' + prop + ' (' + this[prop]['typename'] + ')');
               break;
            case kConv:
               alert('failed to stream ' + prop + ' (' + this[prop]['typename'] + ')');
               break;
            case kConvL:
               alert('failed to stream ' + prop + ' (' + this[prop]['typename'] + ')');
               break;
            case kConvP:
               alert('failed to stream ' + prop + ' (' + this[prop]['typename'] + ')');
               break;
            case kSTL:
               alert('failed to stream ' + prop + ' (' + this[prop]['typename'] + ')');
               break;
            case kSTLstring:
               alert('failed to stream ' + prop + ' (' + this[prop]['typename'] + ')');
               break;
            case kStreamer:
               alert('failed to stream ' + prop + ' (' + this[prop]['typename'] + ')');
               break;
            case kStreamLoop:
               alert('failed to stream ' + prop + ' (' + this[prop]['typename'] + ')');
               break;
            case kOffsetL+kShort:
            case kOffsetL+kUShort:
               var n_el = str.charCodeAt(o) & 0xff;
               var array = JSROOTIO.ReadFastArray(str, o, n_el, 'S');
               obj[prop] = array['array'];
               o = array['off'];
               break;
            case kOffsetL+kInt:
            case kOffsetL+kUInt:
               //var n_el = str.charCodeAt(o) & 0xff;
               var n_el  = this[prop]['length'];
               var array = JSROOTIO.ReadFastArray(str, o, n_el, 'I');
               obj[prop] = array['array'];
               o = array['off'];
               break;
            case kOffsetL+kLong:
            case kOffsetL+kULong:
            case kOffsetL+kLong64:
            case kOffsetL+kULong64:
               //var n_el = str.charCodeAt(o) & 0xff;
               var n_el  = this[prop]['length'];
               var array = JSROOTIO.ReadFastArray(str, o, n_el, 'L');
               obj[prop] = array['array'];
               o = array['off'];
               break;
            case kOffsetL+kFloat:
            case kOffsetL+kDouble32:
               //var n_el = str.charCodeAt(o) & 0xff;
               var n_el  = this[prop]['length'];
               var array = JSROOTIO.ReadFastArray(str, o, n_el, 'F');
               obj[prop] = array['array'];
               o = array['off'];
               break;
            case kOffsetL+kDouble:
               //var n_el = str.charCodeAt(o) & 0xff;
               var n_el  = this[prop]['length'];
               var array = JSROOTIO.ReadFastArray(str, o, n_el, 'D');
               obj[prop] = array['array'];
               o = array['off'];
               break;
            case kOffsetP+kChar:
               var n_el = obj[this[prop]['cntname']];
               var array = JSROOTIO.ReadBasicPointer(str, o, n_el, 'C');
               obj[prop] = array['array'];
               o = array['off'];
               break;
            case kOffsetP+kShort:
            case kOffsetP+kUShort:
               var n_el = obj[this[prop]['cntname']];
               var array = JSROOTIO.ReadBasicPointer(str, o, n_el, 'S');
               obj[prop] = array['array'];
               o = array['off'];
               break;
            case kOffsetP+kInt:
            case kOffsetP+kUInt:
               var n_el = obj[this[prop]['cntname']];
               var array = JSROOTIO.ReadBasicPointer(str, o, n_el, 'I');
               obj[prop] = array['array'];
               o = array['off'];
               break;
            case kOffsetP+kLong:
            case kOffsetP+kULong:
            case kOffsetP+kLong64:
            case kOffsetP+kULong64:
               var n_el = obj[this[prop]['cntname']];
               var array = JSROOTIO.ReadBasicPointer(str, o, n_el, 'L');
               obj[prop] = array['array'];
               o = array['off'];
               break;
            case kOffsetP+kFloat:
            case kOffsetP+kDouble32:
               var n_el = obj[this[prop]['cntname']];
               var array = JSROOTIO.ReadBasicPointer(str, o, n_el, 'F');
               obj[prop] = array['array'];
               o = array['off'];
               break;
            case kOffsetP+kDouble:
               var n_el = obj[this[prop]['cntname']];
               var array = JSROOTIO.ReadBasicPointer(str, o, n_el, 'D');
               obj[prop] = array['array'];
               o = array['off'];
               break;
            default:
               alert('failed to stream ' + prop + ' (' + this[prop]['typename'] + ')');
               break;
         }
         return o;
      };

      JSROOTIO.TStreamer.prototype.Stream = function(obj, str, o) {

         var pval;
         var ver = JSROOTIO.ReadVersion(str, o);
         o = ver['off'];

         // first base classes
         for (prop in this) {
            if (!this[prop] || typeof(this[prop]) === "function")
               continue;
            if (this[prop]['typename'] === 'BASE') {
               var clname = this[prop]['class'];
               if (this[prop]['class'].indexOf("TArray") == 0) {
                  var array_type = this[prop]['class'].charAt(this[prop]['class'].length-1)
                  //o++; // ?????
                  obj['fN'] = JSROOTIO.ntou4(str, o);
                  var array = JSROOTIO.ReadArray(str, o, array_type);
                  obj['fArray'] = array['array'];
                  o = array['off'];
               }
               else if (this[prop]['class'] === 'TObject') {
                  o += 2; // skip version
                  o += 4; // skip unique id
                  obj['fBits'] = JSROOTIO.ntou4(str, o); o += 4;
               }
               else if (this[prop]['class'] === 'TQObject') {
                  // skip TQObject
               }
               else {
                  if (JSROOTIO.GetStreamer(prop))
                     o = JSROOTIO.GetStreamer(prop).Stream(obj, str, o);
               }
            }
         }
         // then class members
         for (prop in this) {
            if (!this[prop] || typeof(this[prop]) === "function" ||
                typeof(this[prop]['typename']) === "undefined" ||
                this[prop]['typename'] === "BASE")
               continue;
            var prop_name = prop;
            // special classes (custom streamers)
            switch (this[prop]['typename']) {
               case "TString*":
                  var r__v = JSROOTIO.ReadVersion(str, o);
                  o = r__v['off'];
                  obj[prop_name] = new Array();
                  for (var i = 0; i<obj[this[prop]['cntname']]; ++i ) {
                     var so = JSROOTIO.ReadTString(str, o); o = so['off'];
                     obj[prop_name][i] = so['str'];
                  }
                  break;
               case "TList*":
                  pval = JSROOTIO.ntou4(str, o);
                  if (pval == 0) {
                     o += 4; // skip NULL pointer
                     break;
                  }
                  var clRef = gFile.fStreamerInfo.ReadClass(str, o);
                  if (clRef && clRef['name']) o = clRef['off'];
               case "TList":
                  var list = JSROOTIO.ReadTList(str, o);
                  obj[prop_name] = list['array'];
                  o = list['off'];
                  break;
               case "TObjArray*":
                  pval = JSROOTIO.ntou4(str, o);
                  if (pval == 0) {
                     o += 4; // skip NULL pointer
                     break;
                  }
                  var clRef = gFile.fStreamerInfo.ReadClass(str, o);
                  if (clRef && clRef['name']) o = clRef['off'];
               case "TObjArray":
                  var list = JSROOTIO.ReadTObjArray(str, o);
                  obj[prop_name] = list['array'];
                  o = list['off'];
                  break;
               case "TClonesArray*":
                  pval = JSROOTIO.ntou4(str, o);
                  if (pval == 0) {
                     o += 4; // skip NULL pointer
                     break;
                  }
                  var clRef = gFile.fStreamerInfo.ReadClass(str, o);
                  if (clRef && clRef['name']) o = clRef['off'];
               case "TClonesArray":
                  var list = JSROOTIO.ReadTClonesArray(str, o);
                  obj[prop_name] = list['array'];
                  o = list['off'];
                  break;
               case "TCollection*":
                  pval = JSROOTIO.ntou4(str, o);
                  if (pval == 0) {
                     o += 4; // skip NULL pointer
                     break;
                  }
                  var clRef = gFile.fStreamerInfo.ReadClass(str, o);
                  if (clRef && clRef['name']) o = clRef['off'];
               case "TCollection":
                  var list = JSROOTIO.ReadTCollection(str, o);
                  obj[prop_name] = list['array'];
                  o = list['off'];
                  break;
               case "THashList*":
                  pval = JSROOTIO.ntou4(str, o);
                  if (pval == 0) {
                     o += 4; // skip NULL pointer
                     break;
                  }
                  var clRef = gFile.fStreamerInfo.ReadClass(str, o);
                  if (clRef && clRef['name']) o = clRef['off'];
               case "THashList":
                  var list = JSROOTIO.ReadTHashList(str, o);
                  obj[prop_name] = list['array'];
                  o = list['off'];
                  break;
               case "TArrayI":
                  var array = JSROOTIO.ReadArray(str, o, 'I');
                  obj[prop] = array['array'];
                  o = array['off'];
                  break;
               case "TArrayF":
                  var array = JSROOTIO.ReadArray(str, o, 'F');
                  obj[prop] = array['array'];
                  o = array['off'];
                  break;
               case "TArrayD":
                  var array = JSROOTIO.ReadArray(str, o, 'D');
                  obj[prop] = array['array'];
                  o = array['off'];
                  break;
               case "TObject":
                  o += 2; // skip version
                  o += 4; // skip unique id
                  obj['fBits'] = JSROOTIO.ntou4(str, o); o += 4;
                  break;
               case "TQObject":
                  // skip TQObject...
                  break;
               default:
                  if (this[prop]['type'] == kAny && typeof(streamUserType) == 'function') {
                     o = streamUserType(this, str, o, obj, prop);
                  }
                  else {
                     // basic types and standard streamers
                     o = this.ReadBasicType(str, o, obj, prop);
                  }
                  break;
            }
         }
         if (obj['fBits'] && typeof(obj['fBits']) != "undefined") {
            obj['TestBit'] = function (f) {
               return ((obj['fBits'] & f) != 0);
            };
         }
         return o;
      };

      return this;
   };

   JSROOTIO.TStreamer.Version = version;

})();

// JSROOTIO.TStreamer.js ends

// StreamerInfo.js
//
// A class that reads TStreamerInfo inside ROOT files.
// Depends on the JSROOTIO library functions.
//

(function(){

   if (typeof JSROOTIO != "object") {
      var e1 = new Error("This extension requires JSROOTIO.core.js");
      e1.source = "JSROOTIO.StreamerInfo.js";
      throw e1;
   }

   var version = "1.6 2012/02/24";

   var kMapOffset = 2;
   var kByteCountMask = 0x40000000;
   var kNewClassTag = 0xFFFFFFFF;
   var kClassMask = 0x80000000;

   // ctor
   JSROOTIO.StreamerInfo = function(buffer, callback) {
      if (! (this instanceof arguments.callee) ) {
         var error = new Error("you must use new to instantiate this class");
         error.source = "JSROOTIO.StreamerInfo.ctor";
         throw error;
      }

      this._version = version;
      this._typename = "JSROOTIO.StreamerInfo";

      JSROOTIO.StreamerInfo.prototype.GetClassMap = function(clTag) {
         // find the tag 'clTag' in the list and return the class name
         clTag |= 0x01;
         for (var i=0; i<gFile['fStreamerInfo'].fClassIndex; ++i) {
            if (gFile['fStreamerInfo'].fClassMap[i]['tag'] == clTag)
               return gFile['fStreamerInfo'].fClassMap[i]['name'];
         }
         return -1;
      };

      JSROOTIO.StreamerInfo.prototype.ReadClass = function(str, o) {
         // read class definition from I/O buffer
         var classInfo = {};
         classInfo['name'] = "";
         classInfo['cnt'] = 0;
         classInfo['off'] = o;
         classInfo['tag'] = 0;
         var tag = 0;
         var bcnt = JSROOTIO.ntou4(str, o); o += 4;
         var startpos = o;
         if (!(bcnt & kByteCountMask) ||
              (bcnt == kNewClassTag)) {
            tag = bcnt;
            bcnt = 0;
         } else {
            clVersion = 1;
            tag = JSROOTIO.ntou4(str, o); o+= 4;
         }
         if (!(tag & kClassMask)) {
            classInfo['name'] = 0;
            classInfo['tag'] = tag;
            classInfo['off'] = o;//startpos;
            return classInfo;
         }
         if (tag == kNewClassTag) {
            // got a new class description followed by a new object
            var so = JSROOTIO.ReadString(str, o); // class name
            o = so['off'];
            classInfo['name'] = so['str'];
            //if (gFile.fTagOffset == 0) gFile.fTagOffset = 68;
            classInfo['tag'] = gFile.fTagOffset + startpos + kMapOffset;
         }
         else {
            // got a tag to an already seen class
            var clTag = (tag & ~kClassMask);
            classInfo['name'] = this.GetClassMap(clTag);
         }
         classInfo['cnt'] = (bcnt & ~kByteCountMask);
         classInfo['off'] = o;
         if (tag == kNewClassTag) {
            // add class to fClassMap for later reference
            classInfo['tag'] |= 0x01;
            this.fClassMap[this.fClassIndex] = classInfo;
            this.fClassIndex++;
         }
         return classInfo;
      };

      JSROOTIO.StreamerInfo.prototype.ReadObjArray = function(str, o) {
         // read a TObjArray class definition from I/O buffer
         // and stream all objects in the array
         var objarray = {};
         objarray['name'] = "";
         objarray['array'] = new Array();
         var ver = JSROOTIO.ReadVersion(str, o);
         o = ver['off'];
         if (ver['val'] > 2) {
            o += 2; // skip version
            o += 8; // object bits & unique id
         }
         if (ver['val'] > 1) {
            var so = JSROOTIO.ReadTString(str, o); o = so['off'];
            objarray['name'] = so['str'];
         }
         var nobjects = JSROOTIO.ntou4(str, o); o += 4;
         o += 4; // fLowerBound
         for (var i = 0; i < nobjects; ++i) {
            objarray['array'][i] = this.ReadObject(str, o);
            o = objarray['array'][i]['off'];
         }
         objarray['off'] = o;
         return objarray;
      };

      JSROOTIO.StreamerInfo.prototype.ReadElements = function(str, o) {
         // stream all the elements in the array from the I/O buffer
         var clRef = this.ReadClass(str, o);
         o = clRef['off'];
         return this.ReadObjArray(str, o);
      };

      JSROOTIO.StreamerInfo.prototype.ReadStreamerInfo = function(str, o) {
         // stream an object of class TStreamerInfo from the I/O buffer
         var streamerinfo = {};
         var R__v = JSROOTIO.ReadVersion(str, o);
         o = R__v['off'];
         if (R__v['val'] > 1) {
            var named = JSROOTIO.ReadTNamed(str, o);
            o = named['off'];
            streamerinfo['name'] = named['name'];
            streamerinfo['title'] = named['title'];
            streamerinfo['checksum'] = JSROOTIO.ntou4(str, o); o += 4;
            streamerinfo['classversion'] = JSROOTIO.ntou4(str, o); o += 4;
            streamerinfo['elements'] = this.ReadElements(str, o);
            o = streamerinfo['elements']['off'];
         }
         streamerinfo['off'] = o;
         return streamerinfo;
      };

      JSROOTIO.StreamerInfo.prototype.ReadStreamerElement = function(str, o) {
         // stream an object of class TStreamerElement
         var element = {};
         var R__v = JSROOTIO.ReadVersion(str, o);
         o = R__v['off'];
         var named = JSROOTIO.ReadTNamed(str, o);
         o = named['off'];
         element['name'] = named['name'];
         element['title'] = named['title'];
         element['type'] = JSROOTIO.ntou4(str, o); o += 4;
         element['size'] = JSROOTIO.ntou4(str, o); o += 4;
         element['length'] = JSROOTIO.ntou4(str, o); o += 4;
         element['dim'] = JSROOTIO.ntou4(str, o); o += 4;
         if (R__v['val'] == 1) {
            var array = JSROOTIO.ReadStaticArray(str, o);
            element['maxindex'] = array['array'];
            o = array['off'];
         }
         else {
            var array = JSROOTIO.ReadFastArray(str, o, 5);
            element['maxindex'] = array['array'];
            o = array['off'];
         }
         var so = JSROOTIO.ReadTString(str, o); o = so['off'];
         element['typename'] = so['str'];
         GetExecID = function(str, o) {
            if (element['typename'] == "TRef") return 0;
         }
         if ((element['type'] == 11) && (element['typename'] == "Bool_t" ||
              element['typename'] == "bool"))
            element['type'] = 18;
         if (R__v['val'] > 1) {
            element['uuid'] = 0;
            // check if element is a TRef or TRefArray
            GetExecID();
         }
         if (R__v['val'] <= 2) {
            // In TStreamerElement v2, fSize was holding the size of
            // the underlying data type.  In later version it contains
            // the full length of the data member.
         }
         if (R__v['val'] == 3) {
            element['xmin'] = JSROOTIO.ntou4(str, o); o += 4;
            element['xmax'] = JSROOTIO.ntou4(str, o); o += 4;
            element['factor'] = JSROOTIO.ntou4(str, o); o += 4;
            //if (element['factor'] > 0) SetBit(kHasRange);
         }
         if (R__v['val'] > 3) {
            //if (TestBit(kHasRange)) GetRange(GetTitle(),fXmin,fXmax,fFactor);
         }
         element['off'] = o;
         return element;
      };

      JSROOTIO.StreamerInfo.prototype.ReadStreamerBase = function(str, o) {
         // stream an object of class TStreamerBase
         var streamerbase = {};
         var R__v = JSROOTIO.ReadVersion(str, o);
         o = R__v['off'];
         streamerbase = this.ReadStreamerElement(str, o);
         o = streamerbase['off'];
         if (R__v['val'] > 2) {
            streamerbase['baseversion'] = JSROOTIO.ntou4(str, o); o += 4;
         }
         streamerbase['off'] = o;
         return streamerbase;
      };

      JSROOTIO.StreamerInfo.prototype.ReadBuffer = function(str, o) {
         // read a streamer element from a buffer
         return this.ReadStreamerElement(str, o);
      };

      JSROOTIO.StreamerInfo.prototype.GenericReadAction = function(str, o) {
         // generic read from a buffer
         return this.ReadBuffer(str, o);
      };

      JSROOTIO.StreamerInfo.prototype.ReadClassBuffer = function(str, o) {
         // deserialize class information from a buffer
         return this.GenericReadAction(str, o);
      };

      JSROOTIO.StreamerInfo.prototype.ReadStreamerBasicType = function(str, o) {
         // stream an object of class TStreamerBasicType
         var streamerbase = {};
         var R__v = JSROOTIO.ReadVersion(str, o);
         o = R__v['off'];
         if (R__v['val'] > 1) {
            streamerbase = this.ReadClassBuffer(str, o);
            o = streamerbase['off'];
         }
         else {
         }
         streamerbase['off'] = o;
         return streamerbase;
      };

      JSROOTIO.StreamerInfo.prototype.ReadStreamerObject = function(str, o) {
         // stream an object of class TStreamerObject
         var streamerbase = {};
         var R__v = JSROOTIO.ReadVersion(str, o);
         o = R__v['off'];
         if (R__v['val'] > 1) {
            streamerbase = this.ReadClassBuffer(str, o);
            o = streamerbase['off'];
         }
         else {
            //====process old versions before automatic schema evolution
            //return ReadStreamerElement(str, o);
         }
         streamerbase['off'] = o;
         return streamerbase;
      };

      JSROOTIO.StreamerInfo.prototype.ReadStreamerBasicPointer = function(str, o) {
         // stream an object of class TStreamerBasicPointer
         var streamerbase = {};
         var R__v = JSROOTIO.ReadVersion(str, o);
         o = R__v['off'];
         if (R__v['val'] > 1) {
            streamerbase = this.ReadClassBuffer(str, o);
            o = streamerbase['off'];
            streamerbase['countversion'] = JSROOTIO.ntou4(str, o); o += 4;
            var so = JSROOTIO.ReadTString(str, o); o = so['off'];
            streamerbase['countName'] = so['str'];
            so = JSROOTIO.ReadTString(str, o); o = so['off'];
            streamerbase['countClass'] = so['str'];
         }
         else {
         }
         streamerbase['off'] = o;
         return streamerbase;
      };

      JSROOTIO.StreamerInfo.prototype.ReadStreamerSTL = function(str, o) {
         // stream an object of class TStreamerSTL
         var streamerSTL = {};
         var R__v = JSROOTIO.ReadVersion(str, o);
         o = R__v['off'];
         if (R__v['val'] > 2) {
            streamerSTL = this.ReadClassBuffer(str, o);
            o = streamerSTL['off'];
            streamerSTL['stltype'] = JSROOTIO.ntou4(str, o); o += 4;
            streamerSTL['ctype'] = JSROOTIO.ntou4(str, o); o += 4;
         }
         streamerSTL['off'] = o;
         return streamerSTL;
      };

      JSROOTIO.StreamerInfo.prototype.ReadTList = function(str, o) {
         // stream all objects in the collection from the I/O buffer
         var list = {};
         list['name'] = "";
         list['array'] = new Array();
         var ver = JSROOTIO.ReadVersion(str, o);
         o = ver['off'];
         if (ver['val'] > 3) {
            o += 2; // skip version
            o += 8; // object bits & unique id
            var so = JSROOTIO.ReadTString(str, o); o = so['off'];
            list['name'] = so['str'];
            var nobjects = JSROOTIO.ntou4(str, o); o += 4;
            for (var i = 0; i < nobjects; ++i) {
               list['array'][i] = this.ReadObject(str, o);
               o = list['array'][i]['off'];
               var nch = str.charCodeAt(o) & 0xff; o++;
               if (ver['val'] > 4 && nch == 255)  {
                  nbig= JSROOTIO.ntou4(str, o); o += 4;
               } else {
                  nbig = nch;
               }
               if (nbig > 0) {
                  var readOption = JSROOTIO.ReadString(str, o, nbig);
                  o += nbig;
               }
            }
         }
         list['off'] = o;
         return list;
      };

      JSROOTIO.StreamerInfo.prototype.ReadObjString = function(str, o) {
         // stream an object of class TObjString
         var version = JSROOTIO.ReadVersion(str, o);
         o = version['off'];
         o += 2; // skip version
         o += 8; // object bits & unique id
         return JSROOTIO.ReadTString(str, o);
      };

      JSROOTIO.StreamerInfo.prototype.ReadObject = function(str, o) {
         // read object from I/O buffer
         var clRef = this.ReadClass(str, o);
         o = clRef['off'];
         if (clRef['name'] == "TStreamerInfo")
            return this.ReadStreamerInfo(str, o);
         if (clRef['name'] == "TStreamerBase")
            return this.ReadStreamerBase(str, o);
         if (clRef['name'] == "TStreamerBasicType")
            return this.ReadStreamerBasicType(str, o);
         if ((clRef['name'] == "TStreamerBasicPointer") ||
             (clRef['name'] == "TStreamerLoop"))
            return this.ReadStreamerBasicPointer(str, o);
         if (clRef['name'] == "TStreamerSTL")
            return this.ReadStreamerSTL(str, o);
         if (clRef['name'] == "TList")
            return this.ReadTList(str, o);
         if (clRef['name'] == "TObjString")
            return this.ReadObjString(str, o);
         // default:
         return this.ReadStreamerObject(str, o);
      };

      JSROOTIO.StreamerInfo.prototype.ExtractStreamerInfo = function(str) {
         // extract the list of streamer infos from the buffer (file)
         var o = 0;
         var version = JSROOTIO.ReadVersion(str, o);
         o = version['off'];
         if (version['val'] > 3) {
            o += 10; // TObject's uniqueID, bits
            var so = JSROOTIO.ReadTString(str, o); o = so['off'];
            var nobjects = JSROOTIO.ntou4(str, o); o += 4;

            for (var i = 0; i < nobjects; ++i) {
               var obj = this.ReadObject(str, o);
               if (obj) {
                  o = obj['off'];
                  this.fStreamerInfos[obj['name']] = obj;
                  this.fStreamerIndex++;
               }
               var nch = str.charCodeAt(o) & 0xff; o++;
               if (version['val'] > 4 && nch == 255)  {
                  nbig= JSROOTIO.ntou4(str, o); o += 4;
               } else {
                  nbig = nch;
               }
               if (nbig > 0) {
                  var readOption = JSROOTIO.ReadString(str, o, nbig);
                  o += nbig;
               }
            }
         }
      };

      this.fStreamerInfos = new Array();
      this.fStreamerIndex = 0;

      this.fClassMap = new Array();
      this.fClassIndex = 0;

      return this;
   };

   JSROOTIO.StreamerInfo.Version = version;

})();

// JSROOTIO.StreamerInfo.js ends

// JSROOTIO.TDirectory.js
//
// A class that reads a TDirectory from a buffer.
// Depends on the JSROOTIO library functions.
//

(function(){

   var version = "1.5 2012/02/21";

   // ctor
   JSROOTIO.TDirectory = function(file, classname) {
      if (! (this instanceof arguments.callee) ) {
         var error = new Error("you must use new to instantiate this class");
         error.source = "JSROOTIO.TDirectory.ctor";
         throw error;
      }

      this.fFile = file;
      this._classname = classname;
      this._version = version;
      this._typename = "JSROOTIO.TDirectory";

      JSROOTIO.TDirectory.prototype.ReadKeys = function(cycle, dir_id) {
         //*-*-------------Read directory info
         var nbytes = this.fNbytesName + 22;
         nbytes += 4;  // fDatimeC.Sizeof();
         nbytes += 4;  // fDatimeM.Sizeof();
         nbytes += 18; // fUUID.Sizeof();
         // assume that the file may be above 2 Gbytes if file version is > 4
         if (this.fFile.fVersion >= 40000) nbytes += 12;

         this.fFile.Seek(this.fSeekDir, this.fFile.ERelativeTo.kBeg);
         var callback1 = function(file, buffer, _dir) {
            var str = new String(buffer);
            o = _dir.fNbytesName;

            var version = JSROOTIO.ntou2(str, o); o += 2;
            var versiondir = version%1000;
            o += 8; // skip fDatimeC and fDatimeM ReadBuffer()
            _dir.fNbytesKeys = JSROOTIO.ntou4(str, o); o += 4;
            _dir.fNbytesName = JSROOTIO.ntou4(str, o); o += 4;
            if (version > 1000) {
               _dir.fSeekDir = JSROOTIO.ntou8(str, o); o += 8;
               _dir.fSeekParent = JSROOTIO.ntou8(str, o); o += 8;
               _dir.fSeekKeys = JSROOTIO.ntou8(str, o); o += 8;
            } else {
               _dir.fSeekDir = JSROOTIO.ntou4(str, o); o += 4;
               _dir.fSeekParent = JSROOTIO.ntou4(str, o); o += 4;
               _dir.fSeekKeys = JSROOTIO.ntou4(str, o); o += 4;
            }
            if (versiondir > 1) o += 18; // skip fUUID.ReadBuffer(buffer);

            //*-*---------read TKey::FillBuffer info
            var kl = 4; // Skip NBytes;
            var keyversion = JSROOTIO.ntoi2(buffer, kl); kl += 2;
            // Skip ObjLen, DateTime, KeyLen, Cycle, SeekKey, SeekPdir
            if (keyversion > 1000) kl += 28; // Large files
            else kl += 20;
            var so = JSROOTIO.ReadTString(buffer, kl); kl = so['off'];
            so = JSROOTIO.ReadTString(buffer, kl); kl = so['off'];
            so = JSROOTIO.ReadTString(buffer, kl); kl = so['off'];
            _dir.fTitle = so['str'];
            if (_dir.fNbytesName < 10 || _dir.fNbytesName > 10000) {
               throw "Init : cannot read directory info of file " + _dir.fURL;
            }
            //*-* -------------Read keys of the top directory

            if ( _dir.fSeekKeys >  0) {
               _dir.fFile.Seek(_dir.fSeekKeys, _dir.fFile.ERelativeTo.kBeg);
               var callback2 = function(file, buffer, _dir) {
                  //headerkey->ReadKeyBuffer(buffer);
                  var key = _dir.fFile.ReadKey(buffer, 0);
                  var offset = key['keyLen']; // 113
                  if (key['className'] != "" && key['name'] != "") {
                     key['offset'] = offset;
                  }
                  var nkeys = JSROOTIO.ntoi4(buffer, offset); offset += 4;
                  for (var i = 0; i < nkeys; i++) {
                     key = _dir.fFile.ReadKey(buffer, offset);
                     offset += key['keyLen'];
                     if (key['className'] != "" && key['name'] != "") {
                        key['offset'] = offset;
                     }
                     _dir.fKeys.push(key);
                  }
                  _dir.fFile.fDirectories.push(_dir);
                  displayDirectory(_dir, cycle, dir_id);
                  delete buffer;
                  buffer = null;
               };
               _dir.fFile.ReadBuffer(_dir.fNbytesKeys, callback2, _dir);
            }
            delete buffer;
            buffer = null;
         };
         this.fFile.ReadBuffer(nbytes, callback1, this);
      };

      JSROOTIO.TDirectory.prototype.Stream = function(str, o, cycle, dir_id) {
         var version = JSROOTIO.ntou2(str, o); o += 2;
         var versiondir = version%1000;
         o += 8; // skip fDatimeC and fDatimeM ReadBuffer()
         this.fNbytesKeys = JSROOTIO.ntou4(str, o); o += 4;
         this.fNbytesName = JSROOTIO.ntou4(str, o); o += 4;
         if (version > 1000) {
            this.fSeekDir = JSROOTIO.ntou8(str, o); o += 8;
            this.fSeekParent = JSROOTIO.ntou8(str, o); o += 8;
            this.fSeekKeys = JSROOTIO.ntou8(str, o); o += 8;
         } else {
            this.fSeekDir = JSROOTIO.ntou4(str, o); o += 4;
            this.fSeekParent = JSROOTIO.ntou4(str, o); o += 4;
            this.fSeekKeys = JSROOTIO.ntou4(str, o); o += 4;
         }
         if (versiondir > 2) o += 18; // skip fUUID.ReadBuffer(buffer);
         if (this.fSeekKeys) this.ReadKeys(cycle, dir_id);
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
/*
   function TH1(name, tile, nbinsx, xlow, xup)
   {
      this.fName = name;
      this.fTitle = title;
      ...
   }

   ReadBasicType = function(str, o, type) {
      ['off'] = o;
      switch (type) {
         case "Char_t":
            ['val'] = str.charCodeAt(o) & 0xff; ['off']++;
            break;
         case "UChar_t":
            ['val'] = (str.charCodeAt(o) & 0xff) >>> 0; ['off']++;
            break;
         case "Short_t":
            ['val']  = (str.charCodeAt(o)   & 0xff) << 8;
            ['val'] += (str.charCodeAt(o+1) & 0xff);
            ['off'] += 2;
            break;
         case "UShort_t":
            ['val']  = ((str.charCodeAt(o)   & 0xff) << 8) >>> 0;
            ['val'] +=  (str.charCodeAt(o+1) & 0xff) >>> 0;
            ['off'] += 2;
            break;
         case "Int_t":
            ['val']  = (b.charCodeAt(o)   & 0xff) << 24;
            ['val'] += (b.charCodeAt(o+1) & 0xff) << 16;
            ['val'] += (b.charCodeAt(o+2) & 0xff) << 8;
            ['val'] += (b.charCodeAt(o+3) & 0xff);
            ['off'] += 4;
            break;
         case "UInt_t":
            ['val']  = ((b.charCodeAt(o)   & 0xff) << 24) >>> 0;
            ['val'] += ((b.charCodeAt(o+1) & 0xff) << 16) >>> 0;
            ['val'] += ((b.charCodeAt(o+2) & 0xff) << 8)  >>> 0;
            ['val'] +=  (b.charCodeAt(o+3) & 0xff) >>> 0;
            ['off'] += 4;
            break;
         case "Long_t":
            ['val']  = (b.charCodeAt(o)   & 0xff) << 56;
            ['val'] += (b.charCodeAt(o+1) & 0xff) << 48;
            ['val'] += (b.charCodeAt(o+2) & 0xff) << 40;
            ['val'] += (b.charCodeAt(o+3) & 0xff) << 32;
            ['val'] += (b.charCodeAt(o+4) & 0xff) << 24;
            ['val'] += (b.charCodeAt(o+5) & 0xff) << 16;
            ['val'] += (b.charCodeAt(o+6) & 0xff) << 8;
            ['val'] += (b.charCodeAt(o+7) & 0xff);
            ['off'] += 8;
            break;
         case "ULong_t":
            ['val']  = ((b.charCodeAt(o)   & 0xff) << 56) >>> 0;
            ['val'] += ((b.charCodeAt(o+1) & 0xff) << 48) >>> 0;
            ['val'] += ((b.charCodeAt(o+2) & 0xff) << 40) >>> 0;
            ['val'] += ((b.charCodeAt(o+3) & 0xff) << 32) >>> 0;
            ['val'] += ((b.charCodeAt(o+4) & 0xff) << 24) >>> 0;
            ['val'] += ((b.charCodeAt(o+5) & 0xff) << 16) >>> 0;
            ['val'] += ((b.charCodeAt(o+6) & 0xff) << 8) >>> 0;
            ['val'] +=  (b.charCodeAt(o+7) & 0xff) >>> 0;
            ['off'] += 8;
            break;
      }
   }

   function ReadBasicTypeElem(name, index)
   {
      name *x=(name*)(arr[index]+ioffset);
      b >> *x;
   }

   TStreamerInfo.ReadBuffer = function(buffer, o, obj) {
      for (element in streamerinfo) {
         element.Read(buffer, o, obj);
      }
   }

   element.Read = function(buffer, o, obj) {
      obj[obj['member_name']] = ReadString(buffer, o);
   }

   element.AddMember = function(obj, member_name, val) {
      obj[member_name] = val;
   }

   element.AddMethod = function(obj, method) {
      obj[obj['method_name']] = method;
   }

   element.AddMethod = function(obj, type) {
      switch (type) {
         case "Char_t":
            obj['Read'] = readChar(str, o);
            break;
         ...
   }
*/

(function(){

   var version = "1.7 2012/11/28";

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

   if (typeof JSROOTIO.StreamerInfo != "function") {
      var e1 = new Error("This extension requires JSROOTIO.StreamerInfo.js");
      e1.source = "JSROOTIO.RootFile.js";
      throw e1;
   }

   var logobj = null;
   var Z_DEFLATED = 8;
   var HDRSIZE = 9;

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
            return parseInt(header);
         }
         xhr = null;
         return -1;
      }

      JSROOTIO.RootFile.prototype.ReadBuffer = function(len, callback, object) {
         // Read specified byte range from remote file
         var ie9 = function(url, pos, len, file, callbk, obj) {
            // IE9 Fallback
            var xhr = new XMLHttpRequest();
            xhr.onreadystatechange = function() {
               if (this.readyState == 4 && (this.status == 200 || this.status == 206)) {
                  var filecontent = new String("");
                  var array = new VBArray(this.responseBody).toArray();
                  for (var i = 0; i < array.length; i++) {
                     filecontent = filecontent + String.fromCharCode(array[i]);
                  }
                  callbk(file, filecontent, obj); // Call callback func with data
                  delete filecontent;
                  filecontent = null;
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
         var other = function(url, pos, len, file, callbk, obj) {
            //
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
                  callbk(file, filecontent, obj); // Call callback func with data
                  delete filecontent;
                  filecontent = null;
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
         // Multi-browser support
         if (typeof ActiveXObject == "function")
            return ie9(this.fURL, this.fOffset, len, this, callback, object);
         else
            return other(this.fURL, this.fOffset, len, this, callback, object);
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
         header['version'] = JSROOTIO.ntou4(str, 4);
         var largeFile = header['version'] >= 1000000;
         var ntohoff = largeFile ? JSROOTIO.ntou8 : JSROOTIO.ntou4;
         header['begin'] = JSROOTIO.ntou4(str, 8);
         header['end'] = ntohoff(str, 12);
         header['units'] = str.charCodeAt(largeFile ? 40 : 32) & 0xff;
         header['seekInfo'] = ntohoff(str, largeFile ? 45 : 37);
         header['nbytesInfo'] = ntohoff(str, largeFile ? 53 : 41);
         if (!header['seekInfo'] && !header['nbytesInfo']) {
            // empty file
            return null;
         }
         this.fSeekInfo = header['seekInfo'];
         this.fNbytesInfo = header['nbytesInfo'];
         this.Log(header);
         return header;
      };

      JSROOTIO.RootFile.prototype.ReadKey = function(str, o) {
         // read key from buffer
         var key = {};
         key['offset'] = o;
         var nbytes = JSROOTIO.ntoi4(str, o);
         key['nbytes'] = Math.abs(nbytes);
         var largeKey = o + nbytes > 2 * 1024 * 1024 * 1024 /*2G*/;
         var ntohoff = largeKey ? JSROOTIO.ntou8 : JSROOTIO.ntou4;
         key['objLen'] = JSROOTIO.ntou4(str, o + 6);
         var datime = JSROOTIO.ntou4(str, o + 10);
         key['datime'] = {
            year : (datime >>> 26) + 1995,
            month : (datime << 6) >>> 28,
            day : (datime << 10) >>> 27,
            hour : (datime << 15) >>> 27,
            min : (datime << 20) >>> 26,
            sec : (datime << 26) >>> 26
         };
         key['keyLen'] = JSROOTIO.ntou2(str, o + 14);
         key['cycle'] = JSROOTIO.ntou2(str, o + 16);
         o += 18;
         if (largeKey) {
            key['seekKey'] = JSROOTIO.ntou8(str, o); o += 8;
            o += 8; // skip seekPdir
         }
         else {
            key['seekKey'] = JSROOTIO.ntou4(str, o); o += 4;
            o += 4; // skip seekPdir
         }
         var so = JSROOTIO.ReadTString(str, o); o = so['off'];
         key['className'] = so['str'];
         so = JSROOTIO.ReadTString(str, o); o = so['off'];
         key['name'] = so['str'];
         so = JSROOTIO.ReadTString(str, o); o = so['off'];
         key['title'] = so['str'];
         key['dataoffset'] = key['seekKey'] + key['keyLen'];
         key['name'] = key['name'].replace(/['"]/g,''); // get rid of quotes
         return key;
      };

      JSROOTIO.RootFile.prototype.R__unzip_header = function(str, off) {
         // Reads header envelope, and determines target size.

         var header = {};
         header['srcsize'] = HDRSIZE;
         header['tgtsize'] = 0;

         /*   C H E C K   H E A D E R   */
         if (!(str.charAt(off) == 'Z' && str.charAt(off+1) == 'L' && str.charCodeAt(off+2) == Z_DEFLATED) &&
             !(str.charAt(off) == 'C' && str.charAt(off+1) == 'S' && str.charCodeAt(off+2) == Z_DEFLATED) &&
             !(str.charAt(off) == 'X' && str.charAt(off+1) == 'Z' && str.charCodeAt(off+2) == 0)) {
            alert("Error R__unzip_header: error in header");
            return null;
         }
         header['srcsize'] += ((str.charCodeAt(off+3) & 0xff) |
                              ((str.charCodeAt(off+4) & 0xff) << 8) |
                              ((str.charCodeAt(off+5) & 0xff) << 16));
         header['tgtsize'] += ((str.charCodeAt(off+6) & 0xff) |
                              ((str.charCodeAt(off+7) & 0xff) << 8) |
                              ((str.charCodeAt(off+8) & 0xff) << 16));
         return header;
      };

      JSROOTIO.RootFile.prototype.R__unzip = function(srcsize, str, off, tgtsize) {

         var obj_buf = {};
         obj_buf['irep'] = 0;
         obj_buf['unzipdata'] = 0;

         /*   C H E C K   H E A D E R   */
         if (srcsize < HDRSIZE) {
            alert("R__unzip: too small source");
            return null;
         }

         /*   C H E C K   H E A D E R   */
         if (!(str.charAt(off) == 'Z' && str.charAt(off+1) == 'L' && str.charCodeAt(off+2) == Z_DEFLATED) &&
             !(str.charAt(off) == 'C' && str.charAt(off+1) == 'S' && str.charCodeAt(off+2) == Z_DEFLATED) &&
             !(str.charAt(off) == 'X' && str.charAt(off+1) == 'Z' && str.charCodeAt(off+2) == 0)) {
            alert("Error R__unzip: error in header");
            return null;
         }
         var ibufcnt = ((str.charCodeAt(off+3) & 0xff) |
                       ((str.charCodeAt(off+4) & 0xff) << 8) |
                       ((str.charCodeAt(off+5) & 0xff) << 16));
         var isize   = ((str.charCodeAt(off+6) & 0xff) |
                       ((str.charCodeAt(off+7) & 0xff) << 8) |
                       ((str.charCodeAt(off+8) & 0xff) << 16));

         if (tgtsize < isize) {
            alert("R__unzip: too small target");
            return null;
         }
         if (ibufcnt + HDRSIZE != srcsize) {
            alert("R__unzip: discrepancy in source length");
            return null;
         }

         /*   D E C O M P R E S S   D A T A  */
         if (str.charAt(off) == 'Z' && str.charAt(off+1) == 'L') {
            /* New zlib format */
            var data = str.substr(off + HDRSIZE + 2, srcsize);
            var unzipdata = RawInflate.inflate(data);
            if (typeof(unzipdata) != 'undefined') {
               obj_buf['unzipdata'] = unzipdata;
               obj_buf['irep'] = unzipdata.length;
            }
         }
         /* Old zlib format */
         else {
            alert("R__unzip: Old zlib format is not supported!");
            return null;
         }
         return obj_buf;
      };

      JSROOTIO.RootFile.prototype.GetKey = function(keyname, cycle) {
         // retrieve a key by its name and cycle in the list of keys
         var i, j;
         for (i=0; i<this.fKeys.length; ++i) {
            if (this.fKeys[i]['name'] == keyname && this.fKeys[i]['cycle'] == cycle)
               return this.fKeys[i];
         }
         for (j=0; j<this.fDirectories.length;++j) {
            for (i=0; i<this.fDirectories[j].fKeys.length; ++i) {
               if (this.fDirectories[j].fKeys[i]['name'] == keyname &&
                   this.fDirectories[j].fKeys[i]['cycle'] == cycle)
                  return this.fDirectories[j].fKeys[i];
            }
         }
         return null;
      };

      JSROOTIO.RootFile.prototype.ReadObjBuffer = function(key, callback) {
         // read and inflate object buffer described by its key
         this.Seek(key['dataoffset'], this.ERelativeTo.kBeg);
         this.fTagOffset = key.keyLen;
         var callback1 = function(file, buffer) {
            var noutot = 0;
            var objbuf = 0;

            if (key['objLen'] > key['nbytes']-key['keyLen']) {
               var hdr = file.R__unzip_header(buffer, 0);
               if (hdr == null) {
                  delete buffer;
                  buffer = null;
                  return;
               }
               objbuf = file.R__unzip(hdr['srcsize'], buffer, 0, hdr['tgtsize']);
            }
            else {
               var obj_buf = {};
               obj_buf['unzipdata'] = buffer;
               obj_buf['irep'] = buffer.length;
               objbuf = obj_buf;
            }
            delete buffer;
            buffer = null;
            callback(file, objbuf);
         };
         this.ReadBuffer(key['nbytes'] - key['keyLen'], callback1);
      };

      JSROOTIO.RootFile.prototype.ReadObject = function(obj_name, cycle, node_id) {
         // read any object from a root file
         if (findObject(obj_name+cycle)) return;
         var key = this.GetKey(obj_name, cycle);
         if (key == null) return;
         this.fTagOffset = key.keyLen;
         var callback = function(file, objbuf) {
            if (objbuf && objbuf['unzipdata']) {
               if (key['className'] == 'TCanvas') {
                  var canvas = JSROOTIO.ReadTCanvas(objbuf['unzipdata'], 0);
                  if (canvas && canvas['fPrimitives']) {
                     if(canvas['fName'] == "") canvas['fName'] = obj_name;
                     displayObject(canvas, cycle, obj_index);
                     obj_list.push(obj_name+cycle);
                     obj_index++;
                  }
               }
               else if (JSROOTIO.GetStreamer(key['className'])) {
                  var obj = {};
                  obj['_typename'] = 'JSROOTIO.' + key['className'];
                  JSROOTIO.GetStreamer(key['className']).Stream(obj, objbuf['unzipdata'], 0);
                  if (key['className'] == 'TFormula') {
                     JSROOTCore.addFormula(obj);
                  }
                  else if (key['className'] == 'TNtuple' || key['className'] == 'TTree') {
                     displayTree(obj, cycle, node_id);
                  }
                  else {
                     JSROOTCore.addMethods(obj);
                     displayObject(obj, cycle, obj_index);
                     obj_list.push(obj_name+cycle);
                     obj_index++;
                  }
               }
               delete objbuf['unzipdata'];
               objbuf['unzipdata'] = null;
            }
         };
         this.ReadObjBuffer(key, callback);
      };

      JSROOTIO.RootFile.prototype.ReadStreamerInfo = function() {

         if (this.fSeekInfo == 0 || this.fNbytesInfo == 0) return;
         this.Seek(this.fSeekInfo, this.ERelativeTo.kBeg);
         var callback1 = function(file, buffer) {
            var key = file.ReadKey(buffer, 0);
            this.fTagOffset = key.keyLen;
            if (key == 0) return;
            file.fKeys.push(key);
            var callback2 = function(file, objbuf) {
               if (objbuf && objbuf['unzipdata']) {
                  file.fStreamerInfo.ExtractStreamerInfo(objbuf['unzipdata']);
                  //JSROOTIO.GenerateStreamers(file);
                  delete objbuf['unzipdata'];
                  objbuf['unzipdata'] = null;
               }
               for (i=0;i<file.fKeys.length;++i) {
                  if (file.fKeys[i]['className'] == 'TFormula') {
                     file.ReadObject(file.fKeys[i]['name'], file.fKeys[i]['cycle']);
                  }
               }
            };
            file.ReadObjBuffer(key, callback2);
            JSROOTPainter.displayListOfKeys(file.fKeys, '#status');
            delete buffer;
            buffer = null;
            // the next two lines are for debugging/info purpose
            //$("#status").append("file header: " + file.fLogMsg  + "<br/>");
            //JSROOTPainter.displayListOfKeyDetails(file.fKeys, '#status');
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
               var o = 4; // skip the "root" file identifier
               file.fVersion = JSROOTIO.ntou4(str, o); o += 4;
               var headerLength = JSROOTIO.ntou4(str, o); o += 4;
               file.fBEGIN = headerLength;
               if (file.fVersion < 1000000) { //small file
                  file.fEND = JSROOTIO.ntou4(str, o); o += 4;
                  file.fSeekFree = JSROOTIO.ntou4(str, o); o += 4;
                  file.fNbytesFree = JSROOTIO.ntou4(str, o); o += 4;
                  var nfree = JSROOTIO.ntoi4(str, o); o += 4;
                  file.fNbytesName = JSROOTIO.ntou4(str, o); o += 4;
                  file.fUnits = str.charCodeAt(o) & 0xff; o++;
                  file.fCompress = JSROOTIO.ntou4(str, o); o += 4;
                  file.fSeekInfo = JSROOTIO.ntou4(str, o); o += 4;
                  file.fNbytesInfo = JSROOTIO.ntou4(str, o); o += 4;
               } else { // new format to support large files
                  file.fEND = JSROOTIO.ntou8(str, o); o += 8;
                  file.fSeekFree = JSROOTIO.ntou8(str, o); o += 8;
                  file.fNbytesFree = JSROOTIO.ntou4(str, o); o += 4;
                  var nfree = JSROOTIO.ntou4(str, o); o += 4;
                  file.fNbytesName = JSROOTIO.ntou4(str, o); o += 4;
                  file.fUnits = str.charCodeAt(o) & 0xff; o++;
                  file.fCompress = JSROOTIO.ntou4(str, o); o += 4;
                  file.fSeekInfo = JSROOTIO.ntou8(str, o); o += 8;
                  file.fNbytesInfo = JSROOTIO.ntou4(str, o); o += 4;
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
                  var buffer_keyloc = new String(str);
                  o = file.fNbytesName;

                  var version = JSROOTIO.ntou2(str, o); o += 2;
                  var versiondir = version%1000;
                  o += 8; // skip fDatimeC and fDatimeM ReadBuffer()
                  file.fNbytesKeys = JSROOTIO.ntou4(str, o); o += 4;
                  file.fNbytesName = JSROOTIO.ntou4(str, o); o += 4;
                  if (version > 1000) {
                     file.fSeekDir = JSROOTIO.ntou8(str, o); o += 8;
                     file.fSeekParent = JSROOTIO.ntou8(str, o); o += 8;
                     file.fSeekKeys = JSROOTIO.ntou8(str, o); o += 8;
                  } else {
                     file.fSeekDir = JSROOTIO.ntou4(str, o); o += 4;
                     file.fSeekParent = JSROOTIO.ntou4(str, o); o += 4;
                     file.fSeekKeys = JSROOTIO.ntou4(str, o); o += 4;
                  }
                  if (versiondir > 1) o += 18; // skip fUUID.ReadBuffer(buffer);

                  //*-*---------read TKey::FillBuffer info
                  var kl = 4; // Skip NBytes;
                  var keyversion = JSROOTIO.ntoi2(buffer_keyloc, kl); kl += 2;
                  // Skip ObjLen, DateTime, KeyLen, Cycle, SeekKey, SeekPdir
                  if (keyversion > 1000) kl += 28; // Large files
                  else kl += 20;
                  var so = JSROOTIO.ReadTString(buffer_keyloc, kl); kl = so['off'];
                  so = JSROOTIO.ReadTString(buffer_keyloc, kl); kl = so['off'];
                  so = JSROOTIO.ReadTString(buffer_keyloc, kl); kl = so['off'];
                  file.fTitle = so['str'];
                  if (file.fNbytesName < 10 || this.fNbytesName > 10000) {
                     throw "Init : cannot read directory info of file " + file.fURL;
                  }
                  //*-* -------------Read keys of the top directory

                  if ( file.fSeekKeys >  0) {
                     file.Seek(file.fSeekKeys, file.ERelativeTo.kBeg);

                     var callback4 = function(file, buffer) {
                        //headerkey->ReadKeyBuffer(buffer);
                        var key = file.ReadKey(buffer, 0);
                        var offset = key['keyLen']; // 113
                        if (key['className'] != "" && key['name'] != "") {
                           key['offset'] = offset;
                        }
                        var nkeys = JSROOTIO.ntoi4(buffer, offset); offset += 4;
                        for (var i = 0; i < nkeys; i++) {
                           key = file.ReadKey(buffer, offset);
                           offset += key['keyLen'];
                           if (key['className'] != "" && key['name'] != "") {
                              key['offset'] = offset;
                           }
                           file.fKeys.push(key);
                        }
                        file.ReadStreamerInfo();
                        delete buffer;
                        buffer = null;
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
         var key = this.GetKey(dir_name, cycle);
         if (key == null) return null;

         var callback = function(file, objbuf) {
            if (objbuf && objbuf['unzipdata']) {
               var directory = new JSROOTIO.TDirectory(file, key['className']);
               directory.Stream(objbuf['unzipdata'], 0, cycle, dir_id);
               delete objbuf['unzipdata'];
               objbuf['unzipdata'] = null;
            }
         };
         this.ReadObjBuffer(key, callback);
      };

      JSROOTIO.RootFile.prototype.Init = function(fileurl) {
         // init members of a Root file from given url
         this.fURL = fileurl;
         this.fLogMsg = "";
         this.fEND = this.GetSize(fileurl);
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
         this.fStreamerInfo = null;
         if (this.fObjectMap) this.fObjectMap.splice(0, this.fObjectMap.length);
         this.fObjectMap = null;
      };

      JSROOTIO.RootFile.prototype.GetMappedObject = function(tag) {
         // find the tag 'clTag' in the list and return the class name
         tag |= 0x01;
         for (var i=0; i<this['fObjectMap'].length; ++i) {
            if (this['fObjectMap'][i]['tag'] == tag)
               return this['fObjectMap'][i]['obj'];
         }
         return null;
      };

      JSROOTIO.RootFile.prototype.MapObject = function(obj, tag) {
         if (this['fObjectMap'].indexOf({tag: tag, obj: obj}) == -1)
            this['fObjectMap'].push({tag: tag, obj: obj});
      };

      JSROOTIO.RootFile.prototype.ClearObjectMap = function() {
         if (this.fObjectMap) this.fObjectMap.splice(0, this.fObjectMap.length);
      };

      this.fDirectories = new Array();
      this.fKeys = new Array();
      this.fSeekInfo = 0;
      this.fNbytesInfo = 0;
      this.fTagOffset = 0;
      this.fStreamers = 0;
      this.fStreamerInfo = new JSROOTIO.StreamerInfo();
      this.fEND = this.GetSize(this.fURL);
      this.ReadKeys();
      this.fStreamers = new Array();
      this.fObjectMap = new Array();
      //this.ReadStreamerInfo();

      return this;
   };

   JSROOTIO.RootFile.Version = version;

})();

// JSROOTIO.RootFile.js ends

