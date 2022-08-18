/// I/O methods of JavaScript ROOT

import { httpRequest, createHttpRequest, BIT, loadScript, internals, settings,
         create, getMethods, addMethods, isNodeJs } from './core.mjs';

const clTObject = 'TObject', clTNamed = 'TNamed', clTObjString = 'TObjString', clTString = 'TString',
      clTList = 'TList', clTStreamerElement = "TStreamerElement", clTStreamerObject = 'TStreamerObject',

      kChar = 1, kShort = 2, kInt = 3, kLong = 4, kFloat = 5, kCounter = 6,
      kCharStar = 7, kDouble = 8, kDouble32 = 9, kLegacyChar = 10,
      kUChar = 11, kUShort = 12, kUInt = 13, kULong = 14, kBits = 15,
      kLong64 = 16, kULong64 = 17, kBool = 18, kFloat16 = 19,

      kBase = 0, kOffsetL = 20, kOffsetP = 40,
      kObject = 61, kAny = 62, kObjectp = 63, kObjectP = 64, kTString = 65,
      kTObject = 66, kTNamed = 67, kAnyp = 68, kAnyP = 69,

      /* kAnyPnoVT: 70, */
      kSTLp = 71,
      /* kSkip = 100, kSkipL = 120, kSkipP = 140, kConv = 200, kConvL = 220, kConvP = 240, */

      kSTL = 300, /* kSTLstring = 365, */

      kStreamer = 500, kStreamLoop = 501,

      kMapOffset = 2, kByteCountMask = 0x40000000, kNewClassTag = 0xFFFFFFFF, kClassMask = 0x80000000,

      // constants of bits in version
      kStreamedMemberWise = BIT(14),

      // constants used for coding type of STL container
      kNotSTL = 0, kSTLvector = 1, kSTLlist = 2, kSTLdeque = 3, kSTLmap = 4, kSTLmultimap = 5,
      kSTLset = 6, kSTLmultiset = 7, kSTLbitset = 8,
      // kSTLforwardlist = 9, kSTLunorderedset = 10, kSTLunorderedmultiset = 11, kSTLunorderedmap = 12,
      // kSTLunorderedmultimap = 13, kSTLend = 14

      // name of base IO types
      BasicTypeNames = ["BASE", "char", "short", "int", "long", "float", "int", "const char*", "double", "Double32_t",
                        "char", "unsigned  char", "unsigned short", "unsigned", "unsigned long", "unsigned", "Long64_t", "ULong64_t", "bool", "Float16_t"],

      // names of STL containers
      StlNames = ["", "vector", "list", "deque", "map", "multimap", "set", "multiset", "bitset"],

      // TObject bits
      kIsReferenced = BIT(4), kHasUUID = BIT(5);


/** @summary Custom streamers for root classes
  * @desc map of user-streamer function like func(buf,obj)
  * or alias (classname) which can be used to read that function
  * or list of read functions
  * @private */
const CustomStreamers = {
   TObject(buf, obj) {
      obj.fUniqueID = buf.ntou4();
      obj.fBits = buf.ntou4();
      if (obj.fBits & kIsReferenced) buf.ntou2(); // skip pid
   },

   TNamed: [ {
      basename: clTObject, base: 1, func(buf, obj) {
         if (!obj._typename) obj._typename = clTNamed;
         buf.classStreamer(obj, clTObject);
      }
     },
     { name: 'fName', func(buf, obj) { obj.fName = buf.readTString(); } },
     { name: 'fTitle', func(buf, obj) { obj.fTitle = buf.readTString(); } }
   ],

   TObjString: [ {
      basename: clTObject, base: 1, func(buf, obj) {
         if (!obj._typename) obj._typename = clTObjString;
         buf.classStreamer(obj, clTObject);
      }
     },
     { name: 'fString', func(buf, obj) { obj.fString = buf.readTString(); } }
   ],

   TClonesArray(buf, list) {
      if (!list._typename) list._typename = "TClonesArray";
      list.$kind = "TClonesArray";
      list.name = "";
      const ver = buf.last_read_version;
      if (ver > 2) buf.classStreamer(list, clTObject);
      if (ver > 1) list.name = buf.readTString();
      let classv = buf.readTString(), clv = 0,
          pos = classv.lastIndexOf(";");

      if (pos > 0) {
         clv = parseInt(classv.slice(pos + 1));
         classv = classv.slice(0, pos);
      }

      let nobjects = buf.ntou4();
      if (nobjects < 0) nobjects = -nobjects;  // for backward compatibility

      list.arr = new Array(nobjects);
      list.fLast = nobjects - 1;
      list.fLowerBound = buf.ntou4();

      let streamer = buf.fFile.getStreamer(classv, { val: clv });
      streamer = buf.fFile.getSplittedStreamer(streamer);

      if (!streamer) {
         console.log('Cannot get member-wise streamer for', classv, clv);
      } else {
         // create objects
         for (let n = 0; n < nobjects; ++n)
            list.arr[n] = { _typename: classv };

         // call streamer for all objects member-wise
         for (let k = 0; k < streamer.length; ++k)
            for (let n = 0; n < nobjects; ++n)
               streamer[k].func(buf, list.arr[n]);
      }
   },

   TMap(buf, map) {
      if (!map._typename) map._typename = "TMap";
      map.name = "";
      map.arr = [];
      const ver = buf.last_read_version;
      if (ver > 2) buf.classStreamer(map, clTObject);
      if (ver > 1) map.name = buf.readTString();

      const nobjects = buf.ntou4();
      // create objects
      for (let n = 0; n < nobjects; ++n) {
         let obj = { _typename: "TPair" };
         obj.first = buf.readObjectAny();
         obj.second = buf.readObjectAny();
         if (obj.first) map.arr.push(obj);
      }
   },

   TTreeIndex(buf, obj) {
      const ver = buf.last_read_version;
      obj._typename = "TTreeIndex";
      buf.classStreamer(obj, "TVirtualIndex");
      obj.fMajorName = buf.readTString();
      obj.fMinorName = buf.readTString();
      obj.fN = buf.ntoi8();
      obj.fIndexValues = buf.readFastArray(obj.fN, kLong64);
      if (ver > 1) obj.fIndexValuesMinor = buf.readFastArray(obj.fN, kLong64);
      obj.fIndex = buf.readFastArray(obj.fN, kLong64);
   },

   TRefArray(buf, obj) {
      obj._typename = "TRefArray";
      buf.classStreamer(obj, clTObject);
      obj.name = buf.readTString();
      const nobj = buf.ntoi4();
      obj.fLast = nobj - 1;
      obj.fLowerBound = buf.ntoi4();
      /*const pidf = */ buf.ntou2();
      obj.fUIDs = buf.readFastArray(nobj, kUInt);
   },

   TCanvas(buf, obj) {
      obj._typename = "TCanvas";
      buf.classStreamer(obj, "TPad");
      obj.fDISPLAY = buf.readTString();
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
      obj.fCatt = buf.classStreamer({}, "TAttCanvas");
      buf.ntou1(); // ignore b << TestBit(kMoveOpaque);
      buf.ntou1(); // ignore b << TestBit(kResizeOpaque);
      obj.fHighLightColor = buf.ntoi2();
      obj.fBatch = (buf.ntou1() !== 0);
      buf.ntou1();   // ignore b << TestBit(kShowEventStatus);
      buf.ntou1();   // ignore b << TestBit(kAutoExec);
      buf.ntou1();   // ignore b << TestBit(kMenuBar);
   },

   TObjArray(buf, list) {
      if (!list._typename) list._typename = "TObjArray";
      list.$kind = "TObjArray";
      list.name = "";
      const ver = buf.last_read_version;
      if (ver > 2)
         buf.classStreamer(list, clTObject);
      if (ver > 1)
         list.name = buf.readTString();
      const nobjects = buf.ntou4();
      let i = 0;
      list.arr = new Array(nobjects);
      list.fLast = nobjects - 1;
      list.fLowerBound = buf.ntou4();
      while (i < nobjects)
         list.arr[i++] = buf.readObjectAny();
   },

   TPolyMarker3D(buf, marker) {
      const ver = buf.last_read_version;
      buf.classStreamer(marker, clTObject);
      buf.classStreamer(marker, "TAttMarker");
      marker.fN = buf.ntoi4();
      marker.fP = buf.readFastArray(marker.fN * 3, kFloat);
      marker.fOption = buf.readTString();
      marker.fName = (ver > 1) ? buf.readTString() : "TPolyMarker3D";
   },

   TPolyLine3D(buf, obj) {
      buf.classStreamer(obj, clTObject);
      buf.classStreamer(obj, "TAttLine");
      obj.fN = buf.ntoi4();
      obj.fP = buf.readFastArray(obj.fN * 3, kFloat);
      obj.fOption = buf.readTString();
   },

   TStreamerInfo(buf, obj) {
      buf.classStreamer(obj, clTNamed);
      obj.fCheckSum = buf.ntou4();
      obj.fClassVersion = buf.ntou4();
      obj.fElements = buf.readObjectAny();
   },

   TStreamerElement(buf, element) {
      const ver = buf.last_read_version;
      buf.classStreamer(element, clTNamed);
      element.fType = buf.ntou4();
      element.fSize = buf.ntou4();
      element.fArrayLength = buf.ntou4();
      element.fArrayDim = buf.ntou4();
      element.fMaxIndex = buf.readFastArray((ver == 1) ? buf.ntou4() : 5, kUInt);
      element.fTypeName = buf.readTString();

      if ((element.fType === kUChar) && ((element.fTypeName == "Bool_t") || (element.fTypeName == "bool")))
         element.fType = kBool;

      element.fXmin = element.fXmax = element.fFactor = 0;
      if (ver === 3) {
         element.fXmin = buf.ntod();
         element.fXmax = buf.ntod();
         element.fFactor = buf.ntod();
      } else if ((ver > 3) && (element.fBits & BIT(6))) { // kHasRange

         let p1 = element.fTitle.indexOf("[");
         if ((p1 >= 0) && (element.fType > kOffsetP)) p1 = element.fTitle.indexOf("[", p1 + 1);
         let p2 = element.fTitle.indexOf("]", p1 + 1);

         if ((p1 >= 0) && (p2 >= p1 + 2)) {

            let arr = element.fTitle.slice(p1+1, p2).split(","), nbits = 32;
            if (!arr || arr.length < 2)
               throw new Error(`Problem to decode range setting from streamer element title ${element.fTitle}`);

            if (arr.length === 3) nbits = parseInt(arr[2]);
            if (!Number.isInteger(nbits) || (nbits < 2) || (nbits > 32)) nbits = 32;

            let parse_range = val => {
               if (!val) return 0;
               if (val.indexOf("pi") < 0) return parseFloat(val);
               val = val.trim();
               let sign = 1.;
               if (val[0] == "-") { sign = -1; val = val.slice(1); }
               switch (val) {
                  case "2pi":
                  case "2*pi":
                  case "twopi": return sign * 2 * Math.PI;
                  case "pi/2": return sign * Math.PI / 2;
                  case "pi/4": return sign * Math.PI / 4;
               }
               return sign * Math.PI;
            };

            element.fXmin = parse_range(arr[0]);
            element.fXmax = parse_range(arr[1]);

            // avoid usage of 1 << nbits, while only works up to 32 bits
            let bigint = ((nbits >= 0) && (nbits < 32)) ? Math.pow(2, nbits) : 0xffffffff;
            if (element.fXmin < element.fXmax)
               element.fFactor = bigint / (element.fXmax - element.fXmin);
            else if (nbits < 15)
               element.fXmin = nbits;
         }
      }
   },

   TStreamerBase(buf, elem) {
      const ver = buf.last_read_version;
      buf.classStreamer(elem, clTStreamerElement);
      if (ver > 2) elem.fBaseVersion = buf.ntou4();
   },

   TStreamerSTL(buf, elem) {
      buf.classStreamer(elem, clTStreamerElement);
      elem.fSTLtype = buf.ntou4();
      elem.fCtype = buf.ntou4();

      if ((elem.fSTLtype === kSTLmultimap) &&
         ((elem.fTypeName.indexOf("std::set") === 0) ||
            (elem.fTypeName.indexOf("set") === 0))) elem.fSTLtype = kSTLset;

      if ((elem.fSTLtype === kSTLset) &&
         ((elem.fTypeName.indexOf("std::multimap") === 0) ||
            (elem.fTypeName.indexOf("multimap") === 0))) elem.fSTLtype = kSTLmultimap;
   },

   TStreamerSTLstring(buf, elem) {
      if (buf.last_read_version > 0)
         buf.classStreamer(elem, "TStreamerSTL");
   },

   TList(buf, obj) {
      // stream all objects in the list from the I/O buffer
      if (!obj._typename) obj._typename = this.typename;
      obj.$kind = clTList; // all derived classes will be marked as well
      if (buf.last_read_version > 3) {
         buf.classStreamer(obj, clTObject);
         obj.name = buf.readTString();
         const nobjects = buf.ntou4();
         obj.arr = new Array(nobjects);
         obj.opt = new Array(nobjects);
         for (let i = 0; i < nobjects; ++i) {
            obj.arr[i] = buf.readObjectAny();
            obj.opt[i] = buf.readTString();
         }
      } else {
         obj.name = "";
         obj.arr = [];
         obj.opt = [];
      }
   },

   THashList: clTList,

   TStreamerLoop(buf, elem)  {
      if (buf.last_read_version > 1) {
         buf.classStreamer(elem, clTStreamerElement);
         elem.fCountVersion = buf.ntou4();
         elem.fCountName = buf.readTString();
         elem.fCountClass = buf.readTString();
      }
   },

   TStreamerBasicPointer: 'TStreamerLoop',

   TStreamerObject(buf, elem) {
      if (buf.last_read_version > 1)
         buf.classStreamer(elem, clTStreamerElement);
   },

   TStreamerBasicType: clTStreamerObject,
   TStreamerObjectAny: clTStreamerObject,
   TStreamerString: clTStreamerObject,
   TStreamerObjectPointer: clTStreamerObject,

   TStreamerObjectAnyPointer(buf, elem) {
      if (buf.last_read_version > 0)
         buf.classStreamer(elem, clTStreamerElement);
   },

   TTree: {
      name: '$file',
      func(buf, obj) { obj.$kind = "TTree"; obj.$file = buf.fFile; }
   },

   RooRealVar(buf, obj) {
      const v = buf.last_read_version;
      buf.classStreamer(obj, "RooAbsRealLValue");
      if (v == 1) { buf.ntod(); buf.ntod(); buf.ntoi4(); } // skip fitMin, fitMax, fitBins
      obj._error = buf.ntod();
      obj._asymErrLo = buf.ntod();
      obj._asymErrHi = buf.ntod();
      if (v >= 2) obj._binning = buf.readObjectAny();
      if (v == 3) obj._sharedProp = buf.readObjectAny();
      if (v >= 4) obj._sharedProp = buf.classStreamer({}, "RooRealVarSharedProperties");
   },

   RooAbsBinning(buf, obj) {
      buf.classStreamer(obj, (buf.last_read_version == 1) ? clTObject : clTNamed);
      buf.classStreamer(obj, "RooPrintable");
   },

   RooCategory(buf, obj) {
      const v = buf.last_read_version;
      buf.classStreamer(obj, "RooAbsCategoryLValue");
      obj._sharedProp = (v === 1) ? buf.readObjectAny() : buf.classStreamer({}, "RooCategorySharedProperties");
   },

   'RooWorkspace::CodeRepo': (buf /*, obj*/) => {
      const sz = (buf.last_read_version == 2) ? 3 : 2;
      for (let i = 0; i < sz; ++i) {
         let cnt = buf.ntoi4() * ((i == 0) ? 4 : 3);
         while (cnt--) buf.readTString();
      }
   },

   RooLinkedList(buf, obj) {
      const v = buf.last_read_version;
      buf.classStreamer(obj, clTObject);
      let size = buf.ntoi4();
      obj.arr = create(clTList);
      while (size--)
         obj.arr.Add(buf.readObjectAny());
      if (v > 1) obj._name = buf.readTString();
   },

   TImagePalette: [
      {
         basename: clTObject, base: 1, func(buf, obj) {
            if (!obj._typename) obj._typename = 'TImagePalette';
            buf.classStreamer(obj, clTObject);
         }
      },
      { name: 'fNumPoints', func(buf, obj) { obj.fNumPoints = buf.ntou4(); } },
      { name: 'fPoints', func(buf, obj) { obj.fPoints = buf.readFastArray(obj.fNumPoints, kDouble); } },
      { name: 'fColorRed', func(buf, obj) { obj.fColorRed = buf.readFastArray(obj.fNumPoints, kUShort); } },
      { name: 'fColorGreen', func(buf, obj) { obj.fColorGreen = buf.readFastArray(obj.fNumPoints, kUShort); } },
      { name: 'fColorBlue', func(buf, obj) { obj.fColorBlue = buf.readFastArray(obj.fNumPoints, kUShort); } },
      { name: 'fColorAlpha', func(buf, obj) { obj.fColorAlpha = buf.readFastArray(obj.fNumPoints, kUShort); } }
   ],

   TAttImage: [
      { name: 'fImageQuality', func(buf, obj) { obj.fImageQuality = buf.ntoi4(); } },
      { name: 'fImageCompression', func(buf, obj) { obj.fImageCompression = buf.ntou4(); } },
      { name: 'fConstRatio', func(buf, obj) { obj.fConstRatio = (buf.ntou1() != 0); } },
      { name: 'fPalette', func(buf, obj) { obj.fPalette = buf.classStreamer({}, "TImagePalette"); } }
   ],

   TASImage(buf, obj) {
      if ((buf.last_read_version == 1) && (buf.fFile.fVersion > 0) && (buf.fFile.fVersion < 50000))
         return console.warn("old TASImage version - not yet supported");

      buf.classStreamer(obj, clTNamed);

      if (buf.ntou1() != 0) {
         const size = buf.ntoi4();
         obj.fPngBuf = buf.readFastArray(size, kUChar);
      } else {
         buf.classStreamer(obj, "TAttImage");
         obj.fWidth = buf.ntoi4();
         obj.fHeight = buf.ntoi4();
         obj.fImgBuf = buf.readFastArray(obj.fWidth * obj.fHeight, kDouble);
      }
   },

   TMaterial(buf, obj) {
      const v = buf.last_read_version;
      buf.classStreamer(obj, clTNamed);
      obj.fNumber = buf.ntoi4();
      obj.fA = buf.ntof();
      obj.fZ = buf.ntof();
      obj.fDensity = buf.ntof();
      if (v > 2) {
         buf.classStreamer(obj, "TAttFill");
         obj.fRadLength = buf.ntof();
         obj.fInterLength = buf.ntof();
      } else {
         obj.fRadLength = obj.fInterLength = 0;
      }
   },

   TMixture(buf, obj) {
      buf.classStreamer(obj, "TMaterial");
      obj.fNmixt = buf.ntoi4();
      obj.fAmixt = buf.readFastArray(buf.ntoi4(), kFloat);
      obj.fZmixt = buf.readFastArray(buf.ntoi4(), kFloat);
      obj.fWmixt = buf.readFastArray(buf.ntoi4(), kFloat);
   },

   TVirtualPerfStats: clTObject, // use directly TObject streamer

   TMethodCall: clTObject
};


/** @summary Add custom streamer
  * @public */
function addUserStreamer(type, user_streamer) {
   CustomStreamers[type] = user_streamer;
}


/** @summary these are streamers which do not handle version regularly
  * @desc used for special classes like TRef or TBasket
  * @private */
const DirectStreamers = {
   // do nothing for these classes
   TQObject() {},
   TGraphStruct() {},
   TGraphNode() {},
   TGraphEdge() {},

   TDatime(buf, obj) {
      obj.fDatime = buf.ntou4();
      //  obj.GetDate = function() {
      //  let res = new Date();
      //  res.setFullYear((this.fDatime >>> 26) + 1995);
      //  res.setMonth((this.fDatime << 6) >>> 28);
      //  res.setDate((this.fDatime << 10) >>> 27);
      //  res.setHours((this.fDatime << 15) >>> 27);
      //  res.setMinutes((this.fDatime << 20) >>> 26);
      //  res.setSeconds((this.fDatime << 26) >>> 26);
      //  res.setMilliseconds(0);
      //  return res;
      //  }
   },

   TKey(buf, key) {
      key.fNbytes = buf.ntoi4();
      key.fVersion = buf.ntoi2();
      key.fObjlen = buf.ntou4();
      key.fDatime = buf.classStreamer({}, 'TDatime');
      key.fKeylen = buf.ntou2();
      key.fCycle = buf.ntou2();
      if (key.fVersion > 1000) {
         key.fSeekKey = buf.ntou8();
         buf.shift(8); // skip seekPdir
      } else {
         key.fSeekKey = buf.ntou4();
         buf.shift(4); // skip seekPdir
      }
      key.fClassName = buf.readTString();
      key.fName = buf.readTString();
      key.fTitle = buf.readTString();
   },

   TDirectory(buf, dir) {
      const version = buf.ntou2();
      dir.fDatimeC = buf.classStreamer({}, 'TDatime');
      dir.fDatimeM = buf.classStreamer({}, 'TDatime');
      dir.fNbytesKeys = buf.ntou4();
      dir.fNbytesName = buf.ntou4();
      dir.fSeekDir = (version > 1000) ? buf.ntou8() : buf.ntou4();
      dir.fSeekParent = (version > 1000) ? buf.ntou8() : buf.ntou4();
      dir.fSeekKeys = (version > 1000) ? buf.ntou8() : buf.ntou4();
      // if ((version % 1000) > 2) buf.shift(18); // skip fUUID
   },

   TBasket(buf, obj) {
      buf.classStreamer(obj, 'TKey');
      const ver = buf.readVersion();
      obj.fBufferSize = buf.ntoi4();
      obj.fNevBufSize = buf.ntoi4();
      obj.fNevBuf = buf.ntoi4();
      obj.fLast = buf.ntoi4();
      if (obj.fLast > obj.fBufferSize) obj.fBufferSize = obj.fLast;
      const flag = buf.ntoi1();

      if (flag === 0) return;

      if ((flag % 10) != 2) {
         if (obj.fNevBuf) {
            obj.fEntryOffset = buf.readFastArray(buf.ntoi4(), kInt);
            if ((20 < flag) && (flag < 40))
               for (let i = 0, kDisplacementMask = 0xFF000000; i < obj.fNevBuf; ++i)
                  obj.fEntryOffset[i] &= ~kDisplacementMask;
         }

         if (flag > 40)
            obj.fDisplacement = buf.readFastArray(buf.ntoi4(), kInt);
      }

      if ((flag === 1) || (flag > 10)) {
         // here is reading of raw data
         const sz = (ver.val <= 1) ? buf.ntoi4() : obj.fLast;

         if (sz > obj.fKeylen) {
            // buffer includes again complete TKey data - exclude it
            let blob = buf.extract([buf.o + obj.fKeylen, sz - obj.fKeylen]);

            obj.fBufferRef = new TBuffer(blob, 0, buf.fFile, sz - obj.fKeylen);
            obj.fBufferRef.fTagOffset = obj.fKeylen;
         }

         buf.shift(sz);
      }
   },

   TRef(buf, obj) {
      buf.classStreamer(obj, clTObject);
      if (obj.fBits & kHasUUID)
         obj.fUUID = buf.readTString();
      else
         obj.fPID = buf.ntou2();
   },

   'TMatrixTSym<float>': (buf, obj) => {
      buf.classStreamer(obj, "TMatrixTBase<float>");
      obj.fElements = new Float32Array(obj.fNelems);
      let arr = buf.readFastArray((obj.fNrows * (obj.fNcols + 1)) / 2, kFloat), cnt = 0;
      for (let i = 0; i < obj.fNrows; ++i)
         for (let j = i; j < obj.fNcols; ++j)
            obj.fElements[j * obj.fNcols + i] = obj.fElements[i * obj.fNcols + j] = arr[cnt++];
   },

   'TMatrixTSym<double>': (buf, obj) => {
      buf.classStreamer(obj, "TMatrixTBase<double>");
      obj.fElements = new Float64Array(obj.fNelems);
      let arr = buf.readFastArray((obj.fNrows * (obj.fNcols + 1)) / 2, kDouble), cnt = 0;
      for (let i = 0; i < obj.fNrows; ++i)
         for (let j = i; j < obj.fNcols; ++j)
            obj.fElements[j * obj.fNcols + i] = obj.fElements[i * obj.fNcols + j] = arr[cnt++];
   }
};




/** @summary Returns type id by its name
  * @private */
function getTypeId(typname, norecursion) {
   switch (typname) {
      case "bool":
      case "Bool_t": return kBool;
      case "char":
      case "signed char":
      case "Char_t": return kChar;
      case "Color_t":
      case "Style_t":
      case "Width_t":
      case "short":
      case "Short_t": return kShort;
      case "int":
      case "EErrorType":
      case "Int_t": return kInt;
      case "long":
      case "Long_t": return kLong;
      case "float":
      case "Float_t": return kFloat;
      case "double":
      case "Double_t": return kDouble;
      case "unsigned char":
      case "UChar_t": return kUChar;
      case "unsigned short":
      case "UShort_t": return kUShort;
      case "unsigned":
      case "unsigned int":
      case "UInt_t": return kUInt;
      case "unsigned long":
      case "ULong_t": return kULong;
      case "int64_t":
      case "long long":
      case "Long64_t": return kLong64;
      case "uint64_t":
      case "unsigned long long":
      case "ULong64_t": return kULong64;
      case "Double32_t": return kDouble32;
      case "Float16_t": return kFloat16;
      case "char*":
      case "const char*":
      case "const Char_t*": return kCharStar;
   }

   if (!norecursion) {
      let replace = CustomStreamers[typname];
      if (typeof replace === "string") return getTypeId(replace, true);
   }

   return -1;
}

/** @summary create element of the streamer
  * @private  */
function createStreamerElement(name, typename, file) {
   const elem = {
      _typename: 'TStreamerElement', fName: name, fTypeName: typename,
      fType: 0, fSize: 0, fArrayLength: 0, fArrayDim: 0, fMaxIndex: [0, 0, 0, 0, 0],
      fXmin: 0, fXmax: 0, fFactor: 0
   };

   if (typeof typename === "string") {
      elem.fType = getTypeId(typename);
      if ((elem.fType < 0) && file && file.fBasicTypes[typename])
         elem.fType = file.fBasicTypes[typename];
   } else {
      elem.fType = typename;
      typename = elem.fTypeName = BasicTypeNames[elem.fType] || "int";
   }

   if (elem.fType > 0) return elem; // basic type

   // check if there are STL containers
   let stltype = kNotSTL, pos = typename.indexOf("<");
   if ((pos > 0) && (typename.indexOf(">") > pos + 2))
      for (let stl = 1; stl < StlNames.length; ++stl)
         if (typename.slice(0, pos) === StlNames[stl]) {
            stltype = stl; break;
         }

   if (stltype !== kNotSTL) {
      elem._typename = 'TStreamerSTL';
      elem.fType = kStreamer;
      elem.fSTLtype = stltype;
      elem.fCtype = 0;
      return elem;
   }

   const isptr = (typename.lastIndexOf("*") === typename.length - 1);

   if (isptr)
      elem.fTypeName = typename = typename.slice(0, typename.length - 1);

   if (getArrayKind(typename) == 0) {
      elem.fType = kTString;
      return elem;
   }

   elem.fType = isptr ? kAnyP : kAny;

   return elem;
}


/** @summary Function creates streamer for std::pair object
  * @private */
function getPairStreamer(si, typname, file) {
   if (!si) {
      if (typname.indexOf("pair") !== 0) return null;

      si = file.findStreamerInfo(typname);

      if (!si) {
         let p1 = typname.indexOf("<"), p2 = typname.lastIndexOf(">");
         function GetNextName() {
            let res = "", p = p1 + 1, cnt = 0;
            while ((p < p2) && (cnt >= 0)) {
               switch (typname[p]) {
                  case "<": cnt++; break;
                  case ",": if (cnt === 0) cnt--; break;
                  case ">": cnt--; break;
               }
               if (cnt >= 0) res += typname[p];
               p++;
            }
            p1 = p - 1;
            return res.trim();
         }
         si = { _typename: 'TStreamerInfo', fVersion: 1, fName: typname, fElements: create(clTList) };
         si.fElements.Add(createStreamerElement("first", GetNextName(), file));
         si.fElements.Add(createStreamerElement("second", GetNextName(), file));
      }
   }

   let streamer = file.getStreamer(typname, null, si);

   if (!streamer) return null;

   if (streamer.length !== 2) {
      console.error(`Streamer for pair class contains ${streamer.length} elements`);
      return null;
   }

   for (let nn = 0; nn < 2; ++nn)
      if (streamer[nn].readelem && !streamer[nn].pair_name) {
         streamer[nn].pair_name = (nn == 0) ? "first" : "second";
         streamer[nn].func = function(buf, obj) {
            obj[this.pair_name] = this.readelem(buf);
         };
      }

   return streamer;
}


/** @summary create member entry for streamer element
  * @desc used for reading of data
  * @private */
function createMemberStreamer(element, file) {
   let member = {
      name: element.fName, type: element.fType,
      fArrayLength: element.fArrayLength,
      fArrayDim: element.fArrayDim,
      fMaxIndex: element.fMaxIndex
   };

   if (element.fTypeName === 'BASE') {
      if (getArrayKind(member.name) > 0) {
         // this is workaround for arrays as base class
         // we create 'fArray' member, which read as any other data member
         member.name = 'fArray';
         member.type = kAny;
      } else {
         // create streamer for base class
         member.type = kBase;
         // this.getStreamer(element.fName);
      }
   }

   switch (member.type) {
      case kBase:
         member.base = element.fBaseVersion; // indicate base class
         member.basename = element.fName; // keep class name
         member.func = function(buf, obj) { buf.classStreamer(obj, this.basename); };
         break;
      case kShort:
         member.func = function(buf, obj) { obj[this.name] = buf.ntoi2(); }; break;
      case kInt:
      case kCounter:
         member.func = function(buf, obj) { obj[this.name] = buf.ntoi4(); }; break;
      case kLong:
      case kLong64:
         member.func = function(buf, obj) { obj[this.name] = buf.ntoi8(); }; break;
      case kDouble:
         member.func = function(buf, obj) { obj[this.name] = buf.ntod(); }; break;
      case kFloat:
         member.func = function(buf, obj) { obj[this.name] = buf.ntof(); }; break;
      case kLegacyChar:
      case kUChar:
         member.func = function(buf, obj) { obj[this.name] = buf.ntou1(); }; break;
      case kUShort:
         member.func = function(buf, obj) { obj[this.name] = buf.ntou2(); }; break;
      case kBits:
      case kUInt:
         member.func = function(buf, obj) { obj[this.name] = buf.ntou4(); }; break;
      case kULong64:
      case kULong:
         member.func = function(buf, obj) { obj[this.name] = buf.ntou8(); }; break;
      case kBool:
         member.func = function(buf, obj) { obj[this.name] = buf.ntou1() != 0; }; break;
      case kOffsetL + kBool:
      case kOffsetL + kInt:
      case kOffsetL + kCounter:
      case kOffsetL + kDouble:
      case kOffsetL + kUChar:
      case kOffsetL + kShort:
      case kOffsetL + kUShort:
      case kOffsetL + kBits:
      case kOffsetL + kUInt:
      case kOffsetL + kULong:
      case kOffsetL + kULong64:
      case kOffsetL + kLong:
      case kOffsetL + kLong64:
      case kOffsetL + kFloat:
         if (element.fArrayDim < 2) {
            member.arrlength = element.fArrayLength;
            member.func = function(buf, obj) {
               obj[this.name] = buf.readFastArray(this.arrlength, this.type - kOffsetL);
            };
         } else {
            member.arrlength = element.fMaxIndex[element.fArrayDim - 1];
            member.minus1 = true;
            member.func = function(buf, obj) {
               obj[this.name] = buf.readNdimArray(this, (buf, handle) =>
                  buf.readFastArray(handle.arrlength, handle.type - kOffsetL));
            };
         }
         break;
      case kOffsetL + kChar:
         if (element.fArrayDim < 2) {
            member.arrlength = element.fArrayLength;
            member.func = function(buf, obj) {
               obj[this.name] = buf.readFastString(this.arrlength);
            };
         } else {
            member.minus1 = true; // one dimension used for char*
            member.arrlength = element.fMaxIndex[element.fArrayDim - 1];
            member.func = function(buf, obj) {
               obj[this.name] = buf.readNdimArray(this, (buf, handle) =>
                  buf.readFastString(handle.arrlength));
            };
         }
         break;
      case kOffsetP + kBool:
      case kOffsetP + kInt:
      case kOffsetP + kDouble:
      case kOffsetP + kUChar:
      case kOffsetP + kShort:
      case kOffsetP + kUShort:
      case kOffsetP + kBits:
      case kOffsetP + kUInt:
      case kOffsetP + kULong:
      case kOffsetP + kULong64:
      case kOffsetP + kLong:
      case kOffsetP + kLong64:
      case kOffsetP + kFloat:
         member.cntname = element.fCountName;
         member.func = function(buf, obj) {
            obj[this.name] = (buf.ntou1() === 1) ? buf.readFastArray(obj[this.cntname], this.type - kOffsetP) : [];
         };
         break;
      case kOffsetP + kChar:
         member.cntname = element.fCountName;
         member.func = function(buf, obj) {
            obj[this.name] = (buf.ntou1() === 1) ? buf.readFastString(obj[this.cntname]) : null;
         };
         break;
      case kDouble32:
      case kOffsetL + kDouble32:
      case kOffsetP + kDouble32:
         member.double32 = true;
      case kFloat16:
      case kOffsetL + kFloat16:
      case kOffsetP + kFloat16:
         if (element.fFactor !== 0) {
            member.factor = 1. / element.fFactor;
            member.min = element.fXmin;
            member.read = function(buf) { return buf.ntou4() * this.factor + this.min; };
         } else
            if ((element.fXmin === 0) && member.double32) {
               member.read = function(buf) { return buf.ntof(); };
            } else {
               member.nbits = Math.round(element.fXmin);
               if (member.nbits === 0) member.nbits = 12;
               member.dv = new DataView(new ArrayBuffer(8), 0); // used to cast from uint32 to float32
               member.read = function(buf) {
                  let theExp = buf.ntou1(), theMan = buf.ntou2();
                  this.dv.setUint32(0, (theExp << 23) | ((theMan & ((1 << (this.nbits + 1)) - 1)) << (23 - this.nbits)));
                  return ((1 << (this.nbits + 1) & theMan) ? -1 : 1) * this.dv.getFloat32(0);
               };
            }

         member.readarr = function(buf, len) {
            let arr = this.double32 ? new Float64Array(len) : new Float32Array(len);
            for (let n = 0; n < len; ++n) arr[n] = this.read(buf);
            return arr;
         }

         if (member.type < kOffsetL) {
            member.func = function(buf, obj) { obj[this.name] = this.read(buf); }
         } else
            if (member.type > kOffsetP) {
               member.cntname = element.fCountName;
               member.func = function(buf, obj) {
                  obj[this.name] = (buf.ntou1() === 1) ? this.readarr(buf, obj[this.cntname]) : null;
               };
            } else
               if (element.fArrayDim < 2) {
                  member.arrlength = element.fArrayLength;
                  member.func = function(buf, obj) { obj[this.name] = this.readarr(buf, this.arrlength); };
               } else {
                  member.arrlength = element.fMaxIndex[element.fArrayDim - 1];
                  member.minus1 = true;
                  member.func = function(buf, obj) {
                     obj[this.name] = buf.readNdimArray(this, (buf, handle) => handle.readarr(buf, handle.arrlength));
                  };
               }
         break;

      case kAnyP:
      case kObjectP:
         member.func = function(buf, obj) {
            obj[this.name] = buf.readNdimArray(this, buf => buf.readObjectAny());
         };
         break;

      case kAny:
      case kAnyp:
      case kObjectp:
      case kObject: {
         let classname = (element.fTypeName === 'BASE') ? element.fName : element.fTypeName;
         if (classname[classname.length - 1] == "*")
            classname = classname.slice(0, classname.length - 1);

         const arrkind = getArrayKind(classname);

         if (arrkind > 0) {
            member.arrkind = arrkind;
            member.func = function(buf, obj) { obj[this.name] = buf.readFastArray(buf.ntou4(), this.arrkind); };
         } else if (arrkind === 0) {
            member.func = function(buf, obj) { obj[this.name] = buf.readTString(); };
         } else {
            member.classname = classname;

            if (element.fArrayLength > 1) {
               member.func = function(buf, obj) {
                  obj[this.name] = buf.readNdimArray(this, (buf, handle) => buf.classStreamer({}, handle.classname));
               };
            } else {
               member.func = function(buf, obj) {
                  obj[this.name] = buf.classStreamer({}, this.classname);
               };
            }
         }
         break;
      }
      case kOffsetL + kObject:
      case kOffsetL + kAny:
      case kOffsetL + kAnyp:
      case kOffsetL + kObjectp: {
         let classname = element.fTypeName;
         if (classname[classname.length - 1] == "*")
            classname = classname.slice(0, classname.length - 1);

         member.arrkind = getArrayKind(classname);
         if (member.arrkind < 0) member.classname = classname;
         member.func = function(buf, obj) {
            obj[this.name] = buf.readNdimArray(this, (buf, handle) => {
               if (handle.arrkind > 0) return buf.readFastArray(buf.ntou4(), handle.arrkind);
               if (handle.arrkind === 0) return buf.readTString();
               return buf.classStreamer({}, handle.classname);
            });
         }
         break;
      }
      case kChar:
         member.func = function(buf, obj) { obj[this.name] = buf.ntoi1(); }; break;
      case kCharStar:
         member.func = function(buf, obj) {
            const len = buf.ntoi4();
            obj[this.name] = buf.substring(buf.o, buf.o + len);
            buf.o += len;
         };
         break;
      case kTString:
         member.func = function(buf, obj) { obj[this.name] = buf.readTString(); };
         break;
      case kTObject:
      case kTNamed:
         member.typename = element.fTypeName;
         member.func = function(buf, obj) { obj[this.name] = buf.classStreamer({}, this.typename); };
         break;
      case kOffsetL + kTString:
      case kOffsetL + kTObject:
      case kOffsetL + kTNamed:
         member.typename = element.fTypeName;
         member.func = function(buf, obj) {
            const ver = buf.readVersion();
            obj[this.name] = buf.readNdimArray(this, (buf, handle) => {
               if (handle.typename === clTString) return buf.readTString();
               return buf.classStreamer({}, handle.typename);
            });
            buf.checkByteCount(ver, this.typename + "[]");
         };
         break;
      case kStreamLoop:
      case kOffsetL + kStreamLoop:
         member.typename = element.fTypeName;
         member.cntname = element.fCountName;

         if (member.typename.lastIndexOf("**") > 0) {
            member.typename = member.typename.slice(0, member.typename.lastIndexOf("**"));
            member.isptrptr = true;
         } else {
            member.typename = member.typename.slice(0, member.typename.lastIndexOf("*"));
            member.isptrptr = false;
         }

         if (member.isptrptr) {
            member.readitem = function(buf) { return buf.readObjectAny(); }
         } else {
            member.arrkind = getArrayKind(member.typename);
            if (member.arrkind > 0)
               member.readitem = function(buf) { return buf.readFastArray(buf.ntou4(), this.arrkind); }
            else if (member.arrkind === 0)
               member.readitem = function(buf) { return buf.readTString(); }
            else
               member.readitem = function(buf) { return buf.classStreamer({}, this.typename); }
         }

         if (member.readitem !== undefined) {
            member.read_loop = function(buf, cnt) {
               return buf.readNdimArray(this, (buf2, member2) => {
                  let itemarr = new Array(cnt);
                  for (let i = 0; i < cnt; ++i)
                     itemarr[i] = member2.readitem(buf2);
                  return itemarr;
               });
            }

            member.func = function(buf, obj) {
               const ver = buf.readVersion();
               let res = this.read_loop(buf, obj[this.cntname]);
               if (!buf.checkByteCount(ver, this.typename)) res = null;
               obj[this.name] = res;
            }
            member.branch_func = function(buf, obj) {
               // this is special functions, used by branch in the STL container

               const ver = buf.readVersion(), sz0 = obj[this.stl_size];
               let res = new Array(sz0);

               for (let loop0 = 0; loop0 < sz0; ++loop0) {
                  let cnt = obj[this.cntname][loop0];
                  res[loop0] = this.read_loop(buf, cnt);
               }
               if (!buf.checkByteCount(ver, this.typename)) res = null;
               obj[this.name] = res;
            }

            member.objs_branch_func = function(buf, obj) {
               // special function when branch read as part of complete object
               // objects already preallocated and only appropriate member must be set
               // see code in JSRoot.tree.js for reference

               const ver = buf.readVersion();
               let arr = obj[this.name0]; // objects array where reading is done

               for (let loop0 = 0; loop0 < arr.length; ++loop0) {
                  let obj1 = this.get(arr, loop0), cnt = obj1[this.cntname];
                  obj1[this.name] = this.read_loop(buf, cnt);
               }

               buf.checkByteCount(ver, this.typename);
            }

         } else {
            console.error(`fail to provide function for ${element.fName} (${element.fTypeName})  typ = ${element.fType}`);
            member.func = function(buf, obj) {
               const ver = buf.readVersion();
               buf.checkByteCount(ver);
               obj[this.name] = null;
            };
         }

         break;

      case kStreamer: {
         member.typename = element.fTypeName;

         let stl = (element.fSTLtype || 0) % 40;

         if ((element._typename === 'TStreamerSTLstring') ||
            (member.typename == "string") || (member.typename == "string*")) {
            member.readelem = buf => buf.readTString();
         } else if ((stl === kSTLvector) || (stl === kSTLlist) ||
                    (stl === kSTLdeque) || (stl === kSTLset) || (stl === kSTLmultiset)) {
            let p1 = member.typename.indexOf("<"),
               p2 = member.typename.lastIndexOf(">");

            member.conttype = member.typename.slice(p1 + 1, p2).trim();

            member.typeid = getTypeId(member.conttype);
            if ((member.typeid < 0) && file.fBasicTypes[member.conttype]) {
               member.typeid = file.fBasicTypes[member.conttype];
               console.log('!!! Reuse basic type', member.conttype, 'from file streamer infos');
            }

            // check
            if (element.fCtype && (element.fCtype < 20) && (element.fCtype !== member.typeid)) {
               console.warn('Contained type', member.conttype, 'not recognized as basic type', element.fCtype, 'FORCE');
               member.typeid = element.fCtype;
            }

            if (member.typeid > 0) {
               member.readelem = function(buf) {
                  return buf.readFastArray(buf.ntoi4(), this.typeid);
               };
            } else {
               member.isptr = false;

               if (member.conttype.lastIndexOf("*") === member.conttype.length - 1) {
                  member.isptr = true;
                  member.conttype = member.conttype.slice(0, member.conttype.length - 1);
               }

               if (element.fCtype === kObjectp) member.isptr = true;

               member.arrkind = getArrayKind(member.conttype);

               member.readelem = readVectorElement;

               if (!member.isptr && (member.arrkind < 0)) {

                  let subelem = createStreamerElement("temp", member.conttype);

                  if (subelem.fType === kStreamer) {
                     subelem.$fictional = true;
                     member.submember = createMemberStreamer(subelem, file);
                  }
               }
            }
         } else if ((stl === kSTLmap) || (stl === kSTLmultimap)) {
            const p1 = member.typename.indexOf("<"),
                  p2 = member.typename.lastIndexOf(">");

            member.pairtype = "pair<" + member.typename.slice(p1 + 1, p2) + ">";

            // remember found streamer info from the file -
            // most probably it is the only one which should be used
            member.si = file.findStreamerInfo(member.pairtype);

            member.streamer = getPairStreamer(member.si, member.pairtype, file);

            if (!member.streamer || (member.streamer.length !== 2)) {
               console.error(`Fail to build streamer for pair ${member.pairtype}`);
               delete member.streamer;
            }

            if (member.streamer) member.readelem = readMapElement;
         } else if (stl === kSTLbitset) {
            member.readelem = (buf/*, obj*/) => buf.readFastArray(buf.ntou4(), kBool);
         }

         if (!member.readelem) {
            console.error(`'failed to create streamer for element ${member.typename} ${member.name} element ${element._typename} STL type ${element.fSTLtype}`);
            member.func = function(buf, obj) {
               const ver = buf.readVersion();
               buf.checkByteCount(ver);
               obj[this.name] = null;
            }
         } else
            if (!element.$fictional) {

               member.read_version = function(buf, cnt) {
                  if (cnt === 0) return null;
                  const ver = buf.readVersion();
                  this.member_wise = ((ver.val & kStreamedMemberWise) !== 0);

                  this.stl_version = undefined;
                  if (this.member_wise) {
                     ver.val = ver.val & ~kStreamedMemberWise;
                     this.stl_version = { val: buf.ntoi2() };
                     if (this.stl_version.val <= 0) this.stl_version.checksum = buf.ntou4();
                  }
                  return ver;
               }

               member.func = function(buf, obj) {
                  const ver = this.read_version(buf);

                  let res = buf.readNdimArray(this, (buf2, member2) => member2.readelem(buf2));

                  if (!buf.checkByteCount(ver, this.typename)) res = null;
                  obj[this.name] = res;
               }

               member.branch_func = function(buf, obj) {
                  // special function to read data from STL branch
                  const cnt = obj[this.stl_size],
                        ver = this.read_version(buf, cnt);
                  let arr = new Array(cnt);

                  for (let n = 0; n < cnt; ++n)
                     arr[n] = buf.readNdimArray(this, (buf2, member2) => member2.readelem(buf2));

                  if (ver) buf.checkByteCount(ver, "branch " + this.typename);

                  obj[this.name] = arr;
               }
               member.split_func = function(buf, arr, n) {
                  // function to read array from member-wise streaming
                  const ver = this.read_version(buf);
                  for (let i = 0; i < n; ++i)
                     arr[i][this.name] = buf.readNdimArray(this, (buf2, member2) => member2.readelem(buf2));
                  buf.checkByteCount(ver, this.typename);
               }
               member.objs_branch_func = function(buf, obj) {
                  // special function when branch read as part of complete object
                  // objects already preallocated and only appropriate member must be set
                  // see code in JSRoot.tree.js for reference

                  let arr = obj[this.name0]; // objects array where reading is done

                  const ver = this.read_version(buf, arr.length);

                  for (let n = 0; n < arr.length; ++n) {
                     let obj1 = this.get(arr, n);
                     obj1[this.name] = buf.readNdimArray(this, (buf2, member2) => member2.readelem(buf2));
                  }

                  if (ver) buf.checkByteCount(ver, "branch " + this.typename);
               }
            }
         break;
      }

      default:
         console.error(`fail to provide function for ${element.fName} (${element.fTypeName})  typ = ${element.fType}`);

         member.func = function(/*buf, obj*/) { };  // do nothing, fix in the future
   }

   return member;
}


/** @summary Analyze and returns arrays kind
  * @returns 0 if TString (or equivalent), positive value - some basic type, -1 - any other kind
  * @private */
function getArrayKind(type_name) {
   if ((type_name === clTString) || (type_name === "string") ||
      (CustomStreamers[type_name] === clTString)) return 0;
   if ((type_name.length < 7) || (type_name.indexOf("TArray") !== 0)) return -1;
   if (type_name.length == 7)
      switch (type_name[6]) {
         case 'I': return kInt;
         case 'D': return kDouble;
         case 'F': return kFloat;
         case 'S': return kShort;
         case 'C': return kChar;
         case 'L': return kLong;
         default: return -1;
      }

   return type_name == "TArrayL64" ? kLong64 : -1;
}

/** @summary Let directly assign methods when doing I/O
  * @private */
function addClassMethods(clname, streamer) {
   if (streamer === null) return streamer;

   let methods = getMethods(clname);

   if (methods)
      for (let key in methods)
         if ((typeof methods[key] === 'function') || (key.indexOf("_") == 0))
            streamer.push({
               name: key,
               method: methods[key],
               func(buf, obj) { obj[this.name] = this.method; }
            });

   return streamer;
}


/* Copyright (C) 1999 Masanao Izumo <iz@onicos.co.jp>
 * Version: 1.0.0.1
 * LastModified: Dec 25 1999
 * original: http://www.onicos.com/staff/iz/amuse/javascript/expert/inflate.txt
 */

/* constant parameters */
const zip_WSIZE = 32768;       // Sliding Window size

/* constant tables (inflate) */
const zip_MASK_BITS = [
   0x0000,
   0x0001, 0x0003, 0x0007, 0x000f, 0x001f, 0x003f, 0x007f, 0x00ff,
   0x01ff, 0x03ff, 0x07ff, 0x0fff, 0x1fff, 0x3fff, 0x7fff, 0xffff],

// Tables for deflate from PKZIP's appnote.txt.
   zip_cplens = [ // Copy lengths for literal codes 257..285
   3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
   35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258, 0, 0],

/* note: see note #13 above about the 258 in this list. */
   zip_cplext = [ // Extra bits for literal codes 257..285
   0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
   3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0, 99, 99], // 99==invalid

   zip_cpdist = [ // Copy offsets for distance codes 0..29
   1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
   257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
   8193, 12289, 16385, 24577],

   zip_cpdext = [ // Extra bits for distance codes
   0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
   7, 7, 8, 8, 9, 9, 10, 10, 11, 11,
   12, 12, 13, 13],

   zip_border = [  // Order of the bit length code lengths
   16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

//    zip_STORED_BLOCK = 0,
//    zip_STATIC_TREES = 1,
//    zip_DYN_TREES    = 2,

/* for inflate */
//    zip_lbits = 9,            // bits in base literal/length lookup table
//    zip_dbits = 6,            // bits in base distance lookup table
//    zip_INBUFSIZ = 32768,     // Input buffer size
//    zip_INBUF_EXTRA = 64,     // Extra buffer

function ZIP_inflate(arr, tgt) {

   /* variables (inflate) */
   let zip_slide = new Array(2 * zip_WSIZE),
       zip_wp = 0,                // current position in slide
       zip_fixed_tl = null,      // inflate static
       zip_fixed_td,             // inflate static
       zip_fixed_bl, zip_fixed_bd,   // inflate static
       zip_bit_buf = 0,            // bit buffer
       zip_bit_len = 0,           // bits in bit buffer
       zip_method = -1,
       zip_eof = false,
       zip_copy_leng = 0,
       zip_copy_dist = 0,
       zip_tl = null, zip_td,    // literal/length and distance decoder tables
       zip_bl, zip_bd,           // number of bits decoded by tl and td
       zip_inflate_data = arr,
       zip_inflate_datalen = arr.byteLength,
       zip_inflate_pos = 0;

   function zip_NEEDBITS(n) {
      while (zip_bit_len < n) {
         if (zip_inflate_pos < zip_inflate_datalen)
            zip_bit_buf |= zip_inflate_data[zip_inflate_pos++] << zip_bit_len;
         zip_bit_len += 8;
      }
   }

   function zip_GETBITS(n) {
      return zip_bit_buf & zip_MASK_BITS[n];
   }

   function zip_DUMPBITS(n) {
      zip_bit_buf >>= n;
      zip_bit_len -= n;
   }

   /* objects (inflate) */
   function zip_HuftBuild(b,     // code lengths in bits (all assumed <= BMAX)
                          n,     // number of codes (assumed <= N_MAX)
                          s,     // number of simple-valued codes (0..s-1)
                          d,     // list of base values for non-simple codes
                          e,     // list of extra bits for non-simple codes
                          mm ) { // maximum lookup bits

      this.status = 0;     // 0: success, 1: incomplete table, 2: bad input
      this.root = null;    // (zip_HuftList) starting table
      this.m = 0;          // maximum lookup bits, returns actual

   /* Given a list of code lengths and a maximum table size, make a set of
      tables to decode that set of codes. Return zero on success, one if
      the given code set is incomplete (the tables are still built in this
      case), two if the input is invalid (all zero length codes or an
      oversubscribed set of lengths), and three if not enough memory.
      The code with value 256 is special, and the tables are constructed
      so that no bits beyond that code are fetched when that code is
      decoded. */

      const BMAX = 16,      // maximum bit length of any code
            N_MAX = 288;    // maximum number of codes in any set
      let c = Array(BMAX+1).fill(0),  // bit length count table
          lx = Array(BMAX+1).fill(0), // stack of bits per table
          u = Array(BMAX).fill(null), // zip_HuftNode[BMAX][]  table stack
          v = Array(N_MAX).fill(0), // values in order of bit length
          x = Array(BMAX+1).fill(0),// bit offsets, then code stack
          r = { e: 0, b: 0, n: 0, t: null }, // new zip_HuftNode(), // table entry for structure assignment
          rr = null, // temporary variable, use in assignment
          el = (n > 256) ? b[256] : BMAX, // set length of EOB code, if any
          a,         // counter for codes of length k
          f,         // i repeats in table every f entries
          g,         // maximum code length
          h,         // table level
          j,         // counter
          k,         // number of bits in current code
          p = b,     // pointer into c[], b[], or v[]
          pidx = 0,  // index of p
          q,         // (zip_HuftNode) points to current table
          w,
          xp,        // pointer into x or c
          y,         // number of dummy codes added
          z,         // number of entries in current table
          o,
          tail = this.root = null, // (zip_HuftList)
          i = n;         // counter, current code

      // Generate counts for each bit length
      do {
         c[p[pidx++]]++; // assume all entries <= BMAX
      } while (--i > 0);

      if (c[0] == n) {   // null input--all zero length codes
         this.root = null;
         this.m = 0;
         this.status = 0;
         return this;
      }

      // Find minimum and maximum length, bound *m by those
      for (j = 1; j <= BMAX; ++j)
         if (c[j] != 0)
            break;
      k = j;         // minimum code length
      if (mm < j)
         mm = j;
      for (i = BMAX; i != 0; --i)
         if (c[i] != 0)
            break;
      g = i;         // maximum code length
      if (mm > i)
         mm = i;

      // Adjust last length count to fill out codes, if needed
      for (y = 1 << j; j < i; ++j, y <<= 1) {
         if ((y -= c[j]) < 0) {
            this.status = 2;  // bad input: more codes than bits
            this.m = mm;
            return this;
         }
      }
      if ((y -= c[i]) < 0) {
         this.status = 2;
         this.m = mm;
         return this;
      }
      c[i] += y;

      // Generate starting offsets into the value table for each length
      x[1] = j = 0;
      p = c;
      pidx = 1;
      xp = 2;
      while (--i > 0)    // note that i == g from above
         x[xp++] = (j += p[pidx++]);

      // Make a table of values in order of bit lengths
      p = b; pidx = 0;
      i = 0;
      do {
         if ((j = p[pidx++]) != 0)
            v[x[j]++] = i;
      } while (++i < n);
      n = x[g];         // set n to length of v

      // Generate the Huffman codes and for each, make the table entries
      x[0] = i = 0;     // first Huffman code is zero
      p = v; pidx = 0;     // grab values in bit order
      h = -1;        // no tables yet--level -1
      w = lx[0] = 0;    // no bits decoded yet
      q = null;         // ditto
      z = 0;         // ditto

      // go through the bit lengths (k already is bits in shortest code)
      for (; k <= g; ++k) {
         a = c[k];
         while (a-- > 0) {
            // here i is the Huffman code of length k bits for value p[pidx]
            // make tables up to required level
            while (k > w + lx[1 + h]) {
               w += lx[1 + h++]; // add bits already decoded

               // compute minimum size table less than or equal to *m bits
               z = (z = g - w) > mm ? mm : z; // upper limit
               if ((f = 1 << (j = k - w)) > a + 1) { // try a k-w bit table
                  // too few codes for k-w bit table
                  f -= a + 1; // deduct codes from patterns left
                  xp = k;
                  while (++j < z) { // try smaller tables up to z bits
                     if ((f <<= 1) <= c[++xp])
                        break;   // enough codes to use up j bits
                     f -= c[xp];   // else deduct codes from patterns
                  }
               }
               if (w + j > el && w < el)
                  j = el - w; // make EOB code end at table
               z = 1 << j;   // table entries for j-bit table
               lx[1 + h] = j; // set table size in stack

               // allocate and link in new table
               q = new Array(z);
               for (o = 0; o < z; ++o)
                  q[o] = { e: 0, b: 0, n: 0, t: null }; // new zip_HuftNode

               if (tail == null)
                  tail = this.root = { next: null, list: null }; // new zip_HuftList();
               else
                  tail = tail.next = { next: null, list: null }; // new zip_HuftList();
               tail.next = null;
               tail.list = q;
               u[h] = q;  // table starts after link

               /* connect to last table, if there is one */
               if (h > 0) {
                  x[h] = i;      // save pattern for backing up
                  r.b = lx[h];   // bits to dump before this table
                  r.e = 16 + j;  // bits in this table
                  r.t = q;    // pointer to this table
                  j = (i & ((1 << w) - 1)) >> (w - lx[h]);
                  rr = u[h-1][j];
                  rr.e = r.e;
                  rr.b = r.b;
                  rr.n = r.n;
                  rr.t = r.t;
               }
            }

            // set up table entry in r
            r.b = k - w;
            if (pidx >= n)
               r.e = 99;     // out of values--invalid code
            else if (p[pidx] < s) {
               r.e = (p[pidx] < 256 ? 16 : 15); // 256 is end-of-block code
               r.n = p[pidx++]; // simple code is just the value
            } else {
               r.e = e[p[pidx] - s];  // non-simple--look up in lists
               r.n = d[p[pidx++] - s];
            }

            // fill code-like entries with r //
            f = 1 << (k - w);
            for (j = i >> w; j < z; j += f) {
               rr = q[j];
               rr.e = r.e;
               rr.b = r.b;
               rr.n = r.n;
               rr.t = r.t;
            }

            // backwards increment the k-bit code i
            for (j = 1 << (k - 1); (i & j) != 0; j >>= 1)
               i ^= j;
            i ^= j;

            // backup over finished tables
            while ((i & ((1 << w) - 1)) != x[h]) {
               w -= lx[h--];      // don't need to update q
            }
         }
      }

      /* return actual size of base table */
      this.m = lx[1];

      /* Return true (1) if we were given an incomplete table */
      this.status = ((y != 0 && g != 1) ? 1 : 0);
     /* end of constructor */

      return this;
   }

   /* routines (inflate) */

   function zip_inflate_codes(buff, off, size) {
      if (size == 0) return 0;

      /* inflate (decompress) the codes in a deflated (compressed) block.
         Return an error code or zero if it all goes ok. */

      let e,     // table entry flag/number of extra bits
          t,     // (zip_HuftNode) pointer to table entry
          n = 0;

      // inflate the coded data
      for (;;) {        // do until end of block
         zip_NEEDBITS(zip_bl);
         t = zip_tl.list[zip_GETBITS(zip_bl)];
         e = t.e;
         while (e > 16) {
            if (e == 99)
               return -1;
            zip_DUMPBITS(t.b);
            e -= 16;
            zip_NEEDBITS(e);
            t = t.t[zip_GETBITS(e)];
            e = t.e;
         }
         zip_DUMPBITS(t.b);

         if (e == 16) {     // then it's a literal
            zip_wp &= zip_WSIZE - 1;
            buff[off + n++] = zip_slide[zip_wp++] = t.n;
            if (n == size)
               return size;
            continue;
         }

         // exit if end of block
         if (e == 15)
            break;

         // it's an EOB or a length

         // get length of block to copy
         zip_NEEDBITS(e);
         zip_copy_leng = t.n + zip_GETBITS(e);
         zip_DUMPBITS(e);

         // decode distance of block to copy
         zip_NEEDBITS(zip_bd);
         t = zip_td.list[zip_GETBITS(zip_bd)];
         e = t.e;

         while (e > 16) {
            if (e == 99)
               return -1;
            zip_DUMPBITS(t.b);
            e -= 16;
            zip_NEEDBITS(e);
            t = t.t[zip_GETBITS(e)];
            e = t.e;
         }
         zip_DUMPBITS(t.b);
         zip_NEEDBITS(e);
         zip_copy_dist = zip_wp - t.n - zip_GETBITS(e);
         zip_DUMPBITS(e);

         // do the copy
         while (zip_copy_leng > 0 && n < size) {
            --zip_copy_leng;
            zip_copy_dist &= zip_WSIZE - 1;
            zip_wp &= zip_WSIZE - 1;
            buff[off + n++] = zip_slide[zip_wp++] = zip_slide[zip_copy_dist++];
         }

         if (n == size)
            return size;
      }

      zip_method = -1; // done
      return n;
   }

   function zip_inflate_stored(buff, off, size) {
      /* "decompress" an inflated type 0 (stored) block. */

      // go to byte boundary
      let n = zip_bit_len & 7;
      zip_DUMPBITS(n);

      // get the length and its complement
      zip_NEEDBITS(16);
      n = zip_GETBITS(16);
      zip_DUMPBITS(16);
      zip_NEEDBITS(16);
      if (n != ((~zip_bit_buf) & 0xffff))
         return -1;        // error in compressed data
      zip_DUMPBITS(16);

      // read and output the compressed data
      zip_copy_leng = n;

      n = 0;
      while (zip_copy_leng > 0 && n < size) {
         --zip_copy_leng;
         zip_wp &= zip_WSIZE - 1;
         zip_NEEDBITS(8);
         buff[off + n++] = zip_slide[zip_wp++] = zip_GETBITS(8);
         zip_DUMPBITS(8);
      }

      if (zip_copy_leng == 0)
         zip_method = -1; // done
      return n;
   }

   function zip_inflate_fixed(buff, off, size) {
      /* decompress an inflated type 1 (fixed Huffman codes) block.  We should
         either replace this with a custom decoder, or at least precompute the
         Huffman tables. */

      // if first time, set up tables for fixed blocks
      if (zip_fixed_tl == null) {

         // literal table
         let l = Array(288).fill(8, 0, 144).fill(9, 144, 256).fill(7, 256, 280).fill(8, 280, 288);
         // make a complete, but wrong code set
         zip_fixed_bl = 7;

         let h = new zip_HuftBuild(l, 288, 257, zip_cplens, zip_cplext, zip_fixed_bl);
         if (h.status != 0)
            throw new Error("HufBuild error: " + h.status);
         zip_fixed_tl = h.root;
         zip_fixed_bl = h.m;

         // distance table
         l.fill(5, 0, 30); // make an incomplete code set
         zip_fixed_bd = 5;

         h = new zip_HuftBuild(l, 30, 0, zip_cpdist, zip_cpdext, zip_fixed_bd);
         if (h.status > 1) {
            zip_fixed_tl = null;
            throw new Error("HufBuild error: "+h.status);
         }
         zip_fixed_td = h.root;
         zip_fixed_bd = h.m;
      }

      zip_tl = zip_fixed_tl;
      zip_td = zip_fixed_td;
      zip_bl = zip_fixed_bl;
      zip_bd = zip_fixed_bd;
      return zip_inflate_codes(buff, off, size);
   }

   function zip_inflate_dynamic(buff, off, size) {
      // decompress an inflated type 2 (dynamic Huffman codes) block.
      let i,j,    // temporary variables
          l,     // last length
          n,     // number of lengths to get
          t,     // (zip_HuftNode) literal/length code table
          h,     // (zip_HuftBuild)
          ll = new Array(286+30).fill(0); // literal/length and distance code lengths

      // read in table lengths
      zip_NEEDBITS(5);
      const nl = 257 + zip_GETBITS(5);   // number of literal/length codes
      zip_DUMPBITS(5);
      zip_NEEDBITS(5);
      const nd = 1 + zip_GETBITS(5);  // number of distance codes
      zip_DUMPBITS(5);
      zip_NEEDBITS(4);
      const nb = 4 + zip_GETBITS(4);  // number of bit length codes
      zip_DUMPBITS(4);
      if (nl > 286 || nd > 30)
         return -1;     // bad lengths

      // read in bit-length-code lengths
      for (j = 0; j < nb; ++j) {
         zip_NEEDBITS(3);
         ll[zip_border[j]] = zip_GETBITS(3);
         zip_DUMPBITS(3);
      }
      for (; j < 19; ++j)
         ll[zip_border[j]] = 0;

      // build decoding table for trees--single level, 7 bit lookup
      zip_bl = 7;
      h = new zip_HuftBuild(ll, 19, 19, null, null, zip_bl);
      if (h.status != 0)
         return -1;  // incomplete code set

      zip_tl = h.root;
      zip_bl = h.m;

      // read in literal and distance code lengths
      n = nl + nd;
      i = l = 0;
      while (i < n) {
         zip_NEEDBITS(zip_bl);
         t = zip_tl.list[zip_GETBITS(zip_bl)];
         j = t.b;
         zip_DUMPBITS(j);
         j = t.n;
         if (j < 16) // length of code in bits (0..15)
            ll[i++] = l = j; // save last length in l
         else if (j == 16) {   // repeat last length 3 to 6 times
            zip_NEEDBITS(2);
            j = 3 + zip_GETBITS(2);
            zip_DUMPBITS(2);
            if (i + j > n)
               return -1;
            while (j-- > 0)
               ll[i++] = l;
         } else if (j == 17) { // 3 to 10 zero length codes
            zip_NEEDBITS(3);
            j = 3 + zip_GETBITS(3);
            zip_DUMPBITS(3);
            if (i + j > n)
               return -1;
            while (j-- > 0)
               ll[i++] = 0;
            l = 0;
         } else {    // j == 18: 11 to 138 zero length codes
            zip_NEEDBITS(7);
            j = 11 + zip_GETBITS(7);
            zip_DUMPBITS(7);
            if (i + j > n)
               return -1;
            while (j-- > 0)
               ll[i++] = 0;
            l = 0;
         }
      }

      // build the decoding tables for literal/length and distance codes
      zip_bl = 9; // zip_lbits;
      h = new zip_HuftBuild(ll, nl, 257, zip_cplens, zip_cplext, zip_bl);
      if (zip_bl == 0)  // no literals or lengths
         h.status = 1;
      if (h.status != 0) {
         // if (h.status == 1); // **incomplete literal tree**
         return -1;     // incomplete code set
      }
      zip_tl = h.root;
      zip_bl = h.m;

      for (i = 0; i < nd; ++i)
         ll[i] = ll[i + nl];
      zip_bd = 6; // zip_dbits;
      h = new zip_HuftBuild(ll, nd, 0, zip_cpdist, zip_cpdext, zip_bd);
      zip_td = h.root;
      zip_bd = h.m;

      if (zip_bd == 0 && nl > 257) {   // lengths but no distances
         // **incomplete distance tree**
         return -1;
      }

      //if (h.status == 1); // **incomplete distance tree**

      if (h.status != 0)
         return -1;

      // decompress until an end-of-block code
      return zip_inflate_codes(buff, off, size);
   }

   function zip_inflate_internal(buff, off, size) {
      // decompress an inflated entry
      let n = 0, i;

      while (n < size) {
         if (zip_eof && zip_method == -1)
            return n;

         if (zip_copy_leng > 0) {
            if (zip_method != 0 /*zip_STORED_BLOCK*/) {
               // STATIC_TREES or DYN_TREES
               while (zip_copy_leng > 0 && n < size) {
                  --zip_copy_leng;
                  zip_copy_dist &= zip_WSIZE - 1;
                  zip_wp &= zip_WSIZE - 1;
                  buff[off + n++] = zip_slide[zip_wp++] =
                  zip_slide[zip_copy_dist++];
               }
            } else {
               while (zip_copy_leng > 0 && n < size) {
                  --zip_copy_leng;
                  zip_wp &= zip_WSIZE - 1;
                  zip_NEEDBITS(8);
                  buff[off + n++] = zip_slide[zip_wp++] = zip_GETBITS(8);
                  zip_DUMPBITS(8);
               }
               if (zip_copy_leng == 0)
                  zip_method = -1; // done
            }
            if (n == size)
               return n;
         }

         if (zip_method == -1) {
            if (zip_eof)
               break;

            // read in last block bit
            zip_NEEDBITS(1);
            if (zip_GETBITS(1) != 0)
               zip_eof = true;
            zip_DUMPBITS(1);

            // read in block type
            zip_NEEDBITS(2);
            zip_method = zip_GETBITS(2);
            zip_DUMPBITS(2);
            zip_tl = null;
            zip_copy_leng = 0;
         }

         switch (zip_method) {
            case 0: // zip_STORED_BLOCK
               i = zip_inflate_stored(buff, off + n, size - n);
               break;

            case 1: // zip_STATIC_TREES
               if (zip_tl != null)
                  i = zip_inflate_codes(buff, off + n, size - n);
               else
                  i = zip_inflate_fixed(buff, off + n, size - n);
               break;

            case 2: // zip_DYN_TREES
               if (zip_tl != null)
                  i = zip_inflate_codes(buff, off + n, size - n);
               else
                  i = zip_inflate_dynamic(buff, off + n, size - n);
               break;

            default: // error
               i = -1;
               break;
         }

         if (i == -1)
            return zip_eof ? 0 : -1;
         n += i;
      }
      return n;
   }

   let i, cnt = 0;
   while ((i = zip_inflate_internal(tgt, cnt, Math.min(1024, tgt.byteLength-cnt))) > 0) {
      cnt += i;
   }

   return cnt;
} // function ZIP_inflate

/**
 * https://github.com/pierrec/node-lz4/blob/master/lib/binding.js
 *
 * LZ4 based compression and decompression
 * Copyright (c) 2014 Pierre Curto
 * MIT Licensed
 */

/**
 * Decode a block. Assumptions: input contains all sequences of a
 * chunk, output is large enough to receive the decoded data.
 * If the output buffer is too small, an error will be thrown.
 * If the returned value is negative, an error occured at the returned offset.
 *
 * @param input {Buffer} input data
 * @param output {Buffer} output data
 * @returns {Number} number of decoded bytes
 * @private */
function LZ4_uncompress(input, output, sIdx, eIdx) {
   sIdx = sIdx || 0;
   eIdx = eIdx || (input.length - sIdx);
   // Process each sequence in the incoming data
   for (let i = sIdx, n = eIdx, j = 0; i < n;) {
      let token = input[i++];

      // Literals
      let literals_length = (token >> 4);
      if (literals_length > 0) {
         // length of literals
         let l = literals_length + 240;
         while (l === 255) {
            l = input[i++];
            literals_length += l;
         }

         // Copy the literals
         let end = i + literals_length;
         while (i < end) output[j++] = input[i++];

         // End of buffer?
         if (i === n) return j;
      }

      // Match copy
      // 2 bytes offset (little endian)
      const offset = input[i++] | (input[i++] << 8);

      // 0 is an invalid offset value
      if (offset === 0 || offset > j) return -(i-2);

      // length of match copy
      let match_length = (token & 0xf),
          l = match_length + 240;
      while (l === 255) {
         l = input[i++];
         match_length += l;
      }

      // Copy the match
      let pos = j - offset; // position of the match copy in the current output
      const end = j + match_length + 4 // minmatch = 4;
      while (j < end) output[j++] = output[pos++]
   }

   return j;
}

/** @summary Reads header envelope, determines zipped size and unzip content
  * @returns {Promise} with unzipped content
  * @private */
function R__unzip(arr, tgtsize, noalert, src_shift) {

   const HDRSIZE = 9, totallen = arr.byteLength,
        getChar = o => String.fromCharCode(arr.getUint8(o)),
        getCode = o => arr.getUint8(o);

   let curr = src_shift || 0, fullres = 0, tgtbuf = null;

   const nextPortion = () => {

      while (fullres < tgtsize) {

         let fmt = "unknown", off = 0, CHKSUM = 0;

         if (curr + HDRSIZE >= totallen) {
            if (!noalert) console.error("Error R__unzip: header size exceeds buffer size");
            return Promise.resolve(null);
         }

         if (getChar(curr) == 'Z' && getChar(curr + 1) == 'L' && getCode(curr + 2) == 8) { fmt = "new"; off = 2; } else
         if (getChar(curr) == 'C' && getChar(curr + 1) == 'S' && getCode(curr + 2) == 8) { fmt = "old"; off = 0; } else
         if (getChar(curr) == 'X' && getChar(curr + 1) == 'Z' && getCode(curr + 2) == 0) fmt = "LZMA"; else
         if (getChar(curr) == 'Z' && getChar(curr + 1) == 'S' && getCode(curr + 2) == 1) fmt = "ZSTD"; else
         if (getChar(curr) == 'L' && getChar(curr + 1) == '4') { fmt = "LZ4"; off = 0; CHKSUM = 8; }

         /*   C H E C K   H E A D E R   */
         if ((fmt !== "new") && (fmt !== "old") && (fmt !== "LZ4") && (fmt !== "ZSTD")) {
            if (!noalert) console.error(`R__unzip: ${fmt} format is not supported!`);
            return Promise.resolve(null);
         }

         const srcsize = HDRSIZE + ((getCode(curr + 3) & 0xff) | ((getCode(curr + 4) & 0xff) << 8) | ((getCode(curr + 5) & 0xff) << 16));

         const uint8arr = new Uint8Array(arr.buffer, arr.byteOffset + curr + HDRSIZE + off + CHKSUM, Math.min(arr.byteLength - curr - HDRSIZE - off - CHKSUM, srcsize - HDRSIZE - CHKSUM));

         if (fmt === "ZSTD")  {
            const handleZsdt = ZstdCodec => {

               return new Promise((resolveFunc, rejectFunc) => {

                  ZstdCodec.run(zstd => {
                     // const simple = new zstd.Simple();
                     const streaming = new zstd.Streaming();

                     // const data2 = simple.decompress(uint8arr);
                     const data2 = streaming.decompress(uint8arr);

                     // console.log(`tgtsize ${tgtsize} zstd size ${data2.length} offset ${data2.byteOffset} rawlen ${data2.buffer.byteLength}`);

                     const reslen = data2.length;

                     if (data2.byteOffset !== 0)
                        return rejectFunc(Error("ZSTD result with byteOffset != 0"));

                     // shortcut when exactly required data unpacked
                     //if ((tgtsize == reslen) && data2.buffer)
                     //   resolveFunc(new DataView(data2.buffer));

                     // need to copy data while zstd does not provide simple way of doing it
                     if (!tgtbuf) tgtbuf = new ArrayBuffer(tgtsize);
                     let tgt8arr = new Uint8Array(tgtbuf, fullres);

                     for(let i = 0; i < reslen; ++i)
                        tgt8arr[i] = data2[i];

                     fullres += reslen;
                     curr += srcsize;
                     resolveFunc(true);
                  });
               });
            };

            let promise = isNodeJs() ? import('zstd-codec').then(handle => handleZsdt(handle.ZstdCodec))
                                        : loadScript('../../zstd/zstd-codec.min.js')
                                             .catch(() => loadScript('https://root.cern/js/zstd/zstd-codec.min.js'))
                                             .then(() => handleZsdt(ZstdCodec));
            return promise.then(() => nextPortion());
         }

         //  place for unpacking
         if (!tgtbuf) tgtbuf = new ArrayBuffer(tgtsize);

         let tgt8arr = new Uint8Array(tgtbuf, fullres);

         const reslen = (fmt === "LZ4") ? LZ4_uncompress(uint8arr, tgt8arr) : ZIP_inflate(uint8arr, tgt8arr);
         if (reslen <= 0) break;

         fullres += reslen;
         curr += srcsize;
      }

      if (fullres !== tgtsize) {
         if (!noalert) console.error(`R__unzip: fail to unzip data expects ${tgtsize}, got ${fullres}`);
         return Promise.resolve(null);
      }

      return Promise.resolve(new DataView(tgtbuf));
   };

   return nextPortion();
}


/**
  * @summary Buffer object to read data from TFile
  *
  * @private
  */

class TBuffer {
   constructor(arr, pos, file, length) {
      this._typename = "TBuffer";
      this.arr = arr;
      this.o = pos || 0;
      this.fFile = file;
      this.length = length || (arr ? arr.byteLength : 0); // use size of array view, blob buffer can be much bigger
      this.clearObjectMap();
      this.fTagOffset = 0;
      this.last_read_version = 0;
   }

   /** @summary locate position in the buffer  */
   locate(pos) { this.o = pos; }

   /** @summary shift position in the buffer  */
   shift(cnt) { this.o += cnt; }

   /** @summary Returns remaining place in the buffer */
   remain() { return this.length - this.o; }

   /** @summary Get mapped object with provided tag */
   getMappedObject(tag) { return this.fObjectMap[tag]; }

   /** @summary Map object */
   mapObject(tag, obj) { if (obj !== null) this.fObjectMap[tag] = obj; }

   /** @summary Map class */
   mapClass(tag, classname) { this.fClassMap[tag] = classname; }

   /** @summary Get mapped class with provided tag */
   getMappedClass(tag) { return (tag in this.fClassMap) ? this.fClassMap[tag] : -1; }

   /** @summary Clear objects map */
   clearObjectMap() {
      this.fObjectMap = {};
      this.fClassMap = {};
      this.fObjectMap[0] = null;
      this.fDisplacement = 0;
   }

   /** @summary  read class version from I/O buffer */
   readVersion() {
      let ver = {}, bytecnt = this.ntou4(); // byte count

      if (bytecnt & kByteCountMask)
         ver.bytecnt = bytecnt - kByteCountMask - 2; // one can check between Read version and end of streamer
      else
         this.o -= 4; // rollback read bytes, this is old buffer without byte count

      this.last_read_version = ver.val = this.ntoi2();
      this.last_read_checksum = 0;
      ver.off = this.o;

      if ((ver.val <= 0) && ver.bytecnt && (ver.bytecnt >= 4)) {
         ver.checksum = this.ntou4();
         if (!this.fFile.findStreamerInfo(undefined, undefined, ver.checksum)) {
            // console.error(`Fail to find streamer info with check sum ${ver.checksum} version ${ver.val}`);
            this.o -= 4; // not found checksum in the list
            delete ver.checksum; // remove checksum
         } else {
            this.last_read_checksum = ver.checksum;
         }
      }
      return ver;
   }

   /** @summary Check bytecount after object streaming */
   checkByteCount(ver, where) {
      if ((ver.bytecnt !== undefined) && (ver.off + ver.bytecnt !== this.o)) {
         if (where)
            console.log(`Missmatch in ${where} bytecount expected = ${ver.bytecnt}  got = ${this.o - ver.off}`);
         this.o = ver.off + ver.bytecnt;
         return false;
      }
      return true;
   }

   /** @summary Read TString object (or equivalent)
     * @desc std::string uses similar binary format */
   readTString() {
      let len = this.ntou1();
      // large strings
      if (len == 255) len = this.ntou4();
      if (len == 0) return "";

      const pos = this.o;
      this.o += len;

      return (this.codeAt(pos) == 0) ? '' : this.substring(pos, pos + len);
   }

    /** @summary read Char_t array as string
      * @desc string either contains all symbols or until 0 symbol */
   readFastString(n) {
      let res = "", code, closed = false;
      for (let i = 0; (n < 0) || (i < n); ++i) {
         code = this.ntou1();
         if (code == 0) { closed = true; if (n < 0) break; }
         if (!closed) res += String.fromCharCode(code);
      }

      return res;
   }

   /** @summary read uint8_t */
   ntou1() { return this.arr.getUint8(this.o++); }

   /** @summary read uint16_t */
   ntou2() {
      const o = this.o; this.o += 2;
      return this.arr.getUint16(o);
   }

   /** @summary read uint32_t */
   ntou4() {
      const o = this.o; this.o += 4;
      return this.arr.getUint32(o);
   }

   /** @summary read uint64_t */
   ntou8() {
      const high = this.arr.getUint32(this.o); this.o += 4;
      const low = this.arr.getUint32(this.o); this.o += 4;
      return (high < 0x200000) ? (high * 0x100000000 + low) : (BigInt(high) * BigInt(0x100000000) + BigInt(low));
   }

   /** @summary read int8_t */
   ntoi1() { return this.arr.getInt8(this.o++); }

   /** @summary read int16_t */
   ntoi2() {
      const o = this.o; this.o += 2;
      return this.arr.getInt16(o);
   }

   /** @summary read int32_t */
   ntoi4() {
      const o = this.o; this.o += 4;
      return this.arr.getInt32(o);
   }

   /** @summary read int64_t */
   ntoi8() {
      const high = this.arr.getUint32(this.o); this.o += 4;
      const low = this.arr.getUint32(this.o); this.o += 4;
      if (high < 0x80000000)
         return (high < 0x200000) ? (high * 0x100000000 + low) : (BigInt(high) * BigInt(0x100000000) + BigInt(low));
      return (~high < 0x200000) ? (-1 - ((~high) * 0x100000000 + ~low)) : (BigInt(-1) - (BigInt(~high) * BigInt(0x100000000) + BigInt(~low)));
   }

   /** @summary read float */
   ntof() {
      const o = this.o; this.o += 4;
      return this.arr.getFloat32(o);
   }

   /** @summary read double */
   ntod() {
      const o = this.o; this.o += 8;
      return this.arr.getFloat64(o);
   }

   /** @summary Reads array of n values from the I/O buffer */
   readFastArray(n, array_type) {
      let array, i = 0, o = this.o;
      const view = this.arr;
      switch (array_type) {
         case kDouble:
            array = new Float64Array(n);
            for (; i < n; ++i, o += 8)
               array[i] = view.getFloat64(o);
            break;
         case kFloat:
            array = new Float32Array(n);
            for (; i < n; ++i, o += 4)
               array[i] = view.getFloat32(o);
            break;
         case kLong:
         case kLong64:
            array = new Array(n);
            for (; i < n; ++i)
               array[i] = this.ntoi8();
            return array; // exit here to avoid conflicts
         case kULong:
         case kULong64:
            array = new Array(n);
            for (; i < n; ++i)
               array[i] = this.ntou8();
            return array; // exit here to avoid conflicts
         case kInt:
         case kCounter:
            array = new Int32Array(n);
            for (; i < n; ++i, o += 4)
               array[i] = view.getInt32(o);
            break;
         case kShort:
            array = new Int16Array(n);
            for (; i < n; ++i, o += 2)
               array[i] = view.getInt16(o);
            break;
         case kUShort:
            array = new Uint16Array(n);
            for (; i < n; ++i, o += 2)
               array[i] = view.getUint16(o);
            break;
         case kChar:
            array = new Int8Array(n);
            for (; i < n; ++i)
               array[i] = view.getInt8(o++);
            break;
         case kBool:
         case kUChar:
            array = new Uint8Array(n);
            for (; i < n; ++i)
               array[i] = view.getUint8(o++);
            break;
         case kTString:
            array = new Array(n);
            for (; i < n; ++i)
               array[i] = this.readTString();
            return array; // exit here to avoid conflicts
         case kDouble32:
            throw new Error('kDouble32 should not be used in readFastArray');
         case kFloat16:
            throw new Error('kFloat16 should not be used in readFastArray');
         // case kBits:
         // case kUInt:
         default:
            array = new Uint32Array(n);
            for (; i < n; ++i, o += 4)
               array[i] = view.getUint32(o);
            break;
      }

      this.o = o;

      return array;
   }

   /** @summary Check if provided regions can be extracted from the buffer */
   canExtract(place) {
      for (let n = 0; n < place.length; n += 2)
         if (place[n] + place[n + 1] > this.length) return false;
      return true;
   }

   /** @summary Extract area */
   extract(place) {
      if (!this.arr || !this.arr.buffer || !this.canExtract(place)) return null;
      if (place.length === 2) return new DataView(this.arr.buffer, this.arr.byteOffset + place[0], place[1]);

      let res = new Array(place.length / 2);

      for (let n = 0; n < place.length; n += 2)
         res[n / 2] = new DataView(this.arr.buffer, this.arr.byteOffset + place[n], place[n + 1]);

      return res; // return array of buffers
   }

   /** @summary Get code at buffer position */
   codeAt(pos) {
      return this.arr.getUint8(pos);
   }

   /** @summary Get part of buffer as string */
   substring(beg, end) {
      let res = "";
      for (let n = beg; n < end; ++n)
         res += String.fromCharCode(this.arr.getUint8(n));
      return res;
   }

   /** @summary Read buffer as N-dim array */
   readNdimArray(handle, func) {
      let ndim = handle.fArrayDim, maxindx = handle.fMaxIndex, res;
      if ((ndim < 1) && (handle.fArrayLength > 0)) { ndim = 1; maxindx = [handle.fArrayLength]; }
      if (handle.minus1) --ndim;

      if (ndim < 1) return func(this, handle);

      if (ndim === 1) {
         res = new Array(maxindx[0]);
         for (let n = 0; n < maxindx[0]; ++n)
            res[n] = func(this, handle);
      } else if (ndim === 2) {
         res = new Array(maxindx[0]);
         for (let n = 0; n < maxindx[0]; ++n) {
            let res2 = new Array(maxindx[1]);
            for (let k = 0; k < maxindx[1]; ++k)
               res2[k] = func(this, handle);
            res[n] = res2;
         }
      } else {
         let indx = [], arr = [], k;
         for (k = 0; k < ndim; ++k) { indx[k] = 0; arr[k] = []; }
         res = arr[0];
         while (indx[0] < maxindx[0]) {
            k = ndim - 1;
            arr[k].push(func(this, handle));
            ++indx[k];
            while ((indx[k] === maxindx[k]) && (k > 0)) {
               indx[k] = 0;
               arr[k - 1].push(arr[k]);
               arr[k] = [];
               ++indx[--k];
            }
         }
      }

      return res;
   }

   /** @summary read TKey data */
   readTKey(key) {
      if (!key) key = {};
      this.classStreamer(key, 'TKey');
      let name = key.fName.replace(/['"]/g, '');
      if (name !== key.fName) {
         key.fRealName = key.fName;
         key.fName = name;
      }
      return key;
   }

   /** @summary reading basket data
     * @desc this is remaining part of TBasket streamer to decode fEntryOffset
     * after unzipping of the TBasket data */
   readBasketEntryOffset(basket, offset) {

      this.locate(basket.fLast - offset);

      if (this.remain() <= 0) {
         if (!basket.fEntryOffset && (basket.fNevBuf <= 1)) basket.fEntryOffset = [basket.fKeylen];
         if (!basket.fEntryOffset) console.warn("No fEntryOffset when expected for basket with", basket.fNevBuf, "entries");
         return;
      }

      const nentries = this.ntoi4();
      // there is error in file=reco_103.root&item=Events;2/PCaloHits_g4SimHits_EcalHitsEE_Sim.&opt=dump;num:10;first:101
      // it is workaround, but normally I/O should fail here
      if ((nentries < 0) || (nentries > this.remain() * 4)) {
         console.error("Error when reading entries offset from basket fNevBuf", basket.fNevBuf, "remains", this.remain(), "want to read", nentries);
         if (basket.fNevBuf <= 1) basket.fEntryOffset = [basket.fKeylen];
         return;
      }

      basket.fEntryOffset = this.readFastArray(nentries, kInt);
      if (!basket.fEntryOffset) basket.fEntryOffset = [basket.fKeylen];

      if (this.remain() > 0)
         basket.fDisplacement = this.readFastArray(this.ntoi4(), kInt);
      else
         basket.fDisplacement = undefined;
   }

   /** @summary read class definition from I/O buffer */
   readClass() {
      const classInfo = { name: -1 }, bcnt = this.ntou4(), startpos = this.o;
      let tag;

      if (!(bcnt & kByteCountMask) || (bcnt == kNewClassTag)) {
         tag = bcnt;
         // bcnt = 0;
      } else {
         tag = this.ntou4();
      }
      if (!(tag & kClassMask)) {
         classInfo.objtag = tag + this.fDisplacement; // indicate that we have deal with objects tag
         return classInfo;
      }
      if (tag == kNewClassTag) {
         // got a new class description followed by a new object
         classInfo.name = this.readFastString(-1);

         if (this.getMappedClass(this.fTagOffset + startpos + kMapOffset) === -1)
            this.mapClass(this.fTagOffset + startpos + kMapOffset, classInfo.name);
      } else {
         // got a tag to an already seen class
         const clTag = (tag & ~kClassMask) + this.fDisplacement;
         classInfo.name = this.getMappedClass(clTag);

         if (classInfo.name === -1)
            console.error(`Did not found class with tag ${clTag}`);
      }

      return classInfo;
   }

   /** @summary Read any object from buffer data */
   readObjectAny() {
      const objtag = this.fTagOffset + this.o + kMapOffset,
            clRef = this.readClass();

      // class identified as object and should be handled so
      if ('objtag' in clRef)
         return this.getMappedObject(clRef.objtag);

      if (clRef.name === -1) return null;

      const arrkind = getArrayKind(clRef.name);
      let obj;

      if (arrkind === 0) {
         obj = this.readTString();
      } else if (arrkind > 0) {
         // reading array, can map array only afterwards
         obj = this.readFastArray(this.ntou4(), arrkind);
         this.mapObject(objtag, obj);
      } else {
         // reading normal object, should map before to
         obj = {};
         this.mapObject(objtag, obj);
         this.classStreamer(obj, clRef.name);
      }

      return obj;
   }

   /** @summary Invoke streamer for specified class  */
   classStreamer(obj, classname) {

      if (obj._typename === undefined) obj._typename = classname;

      const direct = DirectStreamers[classname];
      if (direct) {
         direct(this, obj);
         return obj;
      }

      const ver = this.readVersion(),
            streamer = this.fFile.getStreamer(classname, ver);

      if (streamer !== null) {

         const len = streamer.length;

         for (let n = 0; n < len; ++n)
            streamer[n].func(this, obj);

      } else {
         // just skip bytes belonging to not-recognized object
         // console.warn('skip object ', classname);

         addMethods(obj);
      }

      this.checkByteCount(ver, classname);

      return obj;
   }

} // class TBuffer

// ==============================================================================

/**
  * @summary A class that reads a TDirectory from a buffer.
  *
  * @private
  */

class TDirectory {
   /** @summary constructor */
   constructor(file, dirname, cycle) {
      this.fFile = file;
      this._typename = "TDirectory";
      this.dir_name = dirname;
      this.dir_cycle = cycle;
      this.fKeys = [];
   }

   /** @summary retrieve a key by its name and cycle in the list of keys */
   getKey(keyname, cycle, only_direct) {

      if (typeof cycle != 'number') cycle = -1;
      let bestkey = null;
      for (let i = 0; i < this.fKeys.length; ++i) {
         const key = this.fKeys[i];
         if (!key || (key.fName!==keyname)) continue;
         if (key.fCycle == cycle) { bestkey = key; break; }
         if ((cycle < 0) && (!bestkey || (key.fCycle > bestkey.fCycle))) bestkey = key;
      }
      if (bestkey)
         return only_direct ? bestkey : Promise.resolve(bestkey);

      let pos = keyname.lastIndexOf("/");
      // try to handle situation when object name contains slashed (bad practice anyway)
      while (pos > 0) {
         let dirname = keyname.slice(0, pos),
             subname = keyname.slice(pos+1),
             dirkey = this.getKey(dirname, undefined, true);

         if (dirkey && !only_direct && (dirkey.fClassName.indexOf("TDirectory")==0))
            return this.fFile.readObject(this.dir_name + "/" + dirname, 1)
                             .then(newdir => newdir.getKey(subname, cycle));

         pos = keyname.lastIndexOf("/", pos-1);
      }

      return only_direct ? null : Promise.reject(Error("Key not found " + keyname));
   }

   /** @summary Read object from the directory
     * @param {string} name - object name
     * @param {number} [cycle] - cycle number
     * @returns {Promise} with read object */
   readObject(obj_name, cycle) {
      return this.fFile.readObject(this.dir_name + "/" + obj_name, cycle);
   }

   /** @summary Read list of keys in directory  */
   readKeys(objbuf) {

      objbuf.classStreamer(this, 'TDirectory');

      if ((this.fSeekKeys <= 0) || (this.fNbytesKeys <= 0))
         return Promise.resolve(this);

      return this.fFile.readBuffer([this.fSeekKeys, this.fNbytesKeys]).then(blob => {
         // Read keys of the top directory

         const buf = new TBuffer(blob, 0, this.fFile);

         buf.readTKey();
         const nkeys = buf.ntoi4();

         for (let i = 0; i < nkeys; ++i)
            this.fKeys.push(buf.readTKey());

         this.fFile.fDirectories.push(this);

         return this;
      });
   }

} // class TDirectory

/**
  * @summary Interface to read objects from ROOT files
  *
  * @desc Use {@link openFile} to create instance of the class
  */

class TFile {

   constructor(url) {
      this._typename = "TFile";
      this.fEND = 0;
      this.fFullURL = url;
      this.fURL = url;
      // when disabled ('+' at the end of file name), complete file content read with single operation
      this.fAcceptRanges = true;
      // use additional time stamp parameter for file name to avoid browser caching problem
      this.fUseStampPar = settings.UseStamp ? "stamp=" + (new Date).getTime() : false;
      this.fFileContent = null; // this can be full or partial content of the file (if ranges are not supported or if 1K header read from file)
      // stored as TBuffer instance
      this.fMaxRanges = settings.MaxRanges || 200; // maximal number of file ranges requested at once
      this.fDirectories = [];
      this.fKeys = [];
      this.fSeekInfo = 0;
      this.fNbytesInfo = 0;
      this.fTagOffset = 0;
      this.fStreamers = 0;
      this.fStreamerInfos = null;
      this.fFileName = "";
      this.fStreamers = [];
      this.fBasicTypes = {}; // custom basic types, in most case enumerations

      if (typeof this.fURL != 'string') return this;

      if (this.fURL[this.fURL.length - 1] === "+") {
         this.fURL = this.fURL.slice(0, this.fURL.length - 1);
         this.fAcceptRanges = false;
      }

      if (this.fURL[this.fURL.length - 1] === "^") {
         this.fURL = this.fURL.slice(0, this.fURL.length - 1);
         this.fSkipHeadRequest = true;
      }

      if (this.fURL[this.fURL.length - 1] === "-") {
         this.fURL = this.fURL.slice(0, this.fURL.length - 1);
         this.fUseStampPar = false;
      }

      if (this.fURL.indexOf("file://") == 0) {
         this.fUseStampPar = false;
         this.fAcceptRanges = false;
      }

      const pos = Math.max(this.fURL.lastIndexOf("/"), this.fURL.lastIndexOf("\\"));
      this.fFileName = pos >= 0 ? this.fURL.slice(pos + 1) : this.fURL;
   }

   /** @summary Assign BufferArray with file contentOpen file
     * @private */
   assignFileContent(bufArray) {
      this.fFileContent = new TBuffer(new DataView(bufArray));
      this.fAcceptRanges = false;
      this.fUseStampPar = false;
      this.fEND = this.fFileContent.length;
   }

   /** @summary Open file
     * @returns {Promise} after file keys are read
     * @private */
   _open() {
      if (!this.fAcceptRanges || this.fSkipHeadRequest)
         return this.readKeys();

      return httpRequest(this.fURL, "head").then(res => {
         const accept_ranges = res.getResponseHeader("Accept-Ranges");
         if (!accept_ranges) this.fAcceptRanges = false;
         const len = res.getResponseHeader("Content-Length");
         if (len) this.fEND = parseInt(len);
             else this.fAcceptRanges = false;
         return this.readKeys();
      });
   }

   /** @summary read buffer(s) from the file
    * @returns {Promise} with read buffers
    * @private */
   readBuffer(place, filename, progress_callback) {

      if ((this.fFileContent !== null) && !filename && (!this.fAcceptRanges || this.fFileContent.canExtract(place)))
         return Promise.resolve(this.fFileContent.extract(place));

      let file = this, fileurl = file.fURL, resolveFunc, rejectFunc,
          promise = new Promise((resolve,reject) => { resolveFunc = resolve; rejectFunc = reject; }),
          first = 0, last = 0, blobs = [], read_callback; // array of requested segments

      if (filename && (typeof filename === 'string') && (filename.length > 0)) {
         const pos = fileurl.lastIndexOf("/");
         fileurl = (pos < 0) ? filename : fileurl.slice(0, pos + 1) + filename;
      }

      function send_new_request(increment) {

         if (increment) {
            first = last;
            last = Math.min(first + file.fMaxRanges * 2, place.length);
            if (first >= place.length) return resolveFunc(blobs);
         }

         let fullurl = fileurl, ranges = "bytes", totalsz = 0;
         // try to avoid browser caching by adding stamp parameter to URL
         if (file.fUseStampPar) fullurl += ((fullurl.indexOf('?') < 0) ? "?" : "&") + file.fUseStampPar;

         for (let n = first; n < last; n += 2) {
            ranges += (n > first ? "," : "=") + (place[n] + "-" + (place[n] + place[n + 1] - 1));
            totalsz += place[n + 1]; // accumulated total size
         }
         if (last - first > 2)
            totalsz += (last - first) * 60; // for multi-range ~100 bytes/per request

         return createHttpRequest(fullurl, "buf", read_callback, undefined, true).then(xhr => {

            if (file.fAcceptRanges) {
               xhr.setRequestHeader("Range", ranges);
               xhr.expected_size = Math.max(Math.round(1.1 * totalsz), totalsz + 200); // 200 if offset for the potential gzip
            }

            if (progress_callback && (typeof xhr.addEventListener === 'function')) {
               let sum1 = 0, sum2 = 0, sum_total = 0;
               for (let n = 1; n < place.length; n += 2) {
                  sum_total += place[n];
                  if (n < first) sum1 += place[n];
                  if (n < last) sum2 += place[n];
               }
               if (!sum_total) sum_total = 1;

               let progress_offest = sum1 / sum_total, progress_this = (sum2 - sum1) / sum_total;
               xhr.addEventListener("progress", function(oEvent) {
                  if (oEvent.lengthComputable)
                     progress_callback(progress_offest + progress_this * oEvent.loaded / oEvent.total);
               });
            }

            xhr.send(null);
         });
      }

      read_callback = function(res) {

         if (!res && file.fUseStampPar && (place[0] === 0) && (place.length === 2)) {
            // if fail to read file with stamp parameter, try once again without it
            file.fUseStampPar = false;
            return send_new_request();
         }

         if (res && (place[0] === 0) && (place.length === 2) && !file.fFileContent) {
            // special case - keep content of first request (could be complete file) in memory

            file.fFileContent = new TBuffer((typeof res == 'string') ? res : new DataView(res));

            if (!file.fAcceptRanges)
               file.fEND = file.fFileContent.length;

            return resolveFunc(file.fFileContent.extract(place));
         }

         if (!res) {
            if ((first === 0) && (last > 2) && (file.fMaxRanges > 1)) {
               // server return no response with multi request - try to decrease ranges count or fail

               if (last / 2 > 200)
                  file.fMaxRanges = 200;
               else if (last / 2 > 50)
                  file.fMaxRanges = 50;
               else if (last / 2 > 20)
                  file.fMaxRanges = 20;
               else if (last / 2 > 5)
                  file.fMaxRanges = 5;
               else
                  file.fMaxRanges = 1;
               last = Math.min(last, file.fMaxRanges * 2);
               // console.log('Change maxranges to ', file.fMaxRanges, 'last', last);
               return send_new_request();
            }

            return rejectFunc(Error("Fail to read with several ranges"));
         }

         // if only single segment requested, return result as is
         if (last - first === 2) {
            let b = new DataView(res);
            if (place.length === 2) return resolveFunc(b);
            blobs.push(b);
            return send_new_request(true);
         }

         // object to access response data
         let hdr = this.getResponseHeader('Content-Type'),
            ismulti = (typeof hdr === 'string') && (hdr.indexOf('multipart') >= 0),
            view = new DataView(res);

         if (!ismulti) {
            // server may returns simple buffer, which combines all segments together

            let hdr_range = this.getResponseHeader('Content-Range'), segm_start = 0, segm_last = -1;

            if (hdr_range && hdr_range.indexOf("bytes") >= 0) {
               let parts = hdr_range.slice(hdr_range.indexOf("bytes") + 6).split(/[\s-\/]+/);
               if (parts.length === 3) {
                  segm_start = parseInt(parts[0]);
                  segm_last = parseInt(parts[1]);
                  if (!Number.isInteger(segm_start) || !Number.isInteger(segm_last) || (segm_start > segm_last)) {
                     segm_start = 0; segm_last = -1;
                  }
               }
            }

            let canbe_single_segment = (segm_start <= segm_last);
            for (let n = first; n < last; n += 2)
               if ((place[n] < segm_start) || (place[n] + place[n + 1] - 1 > segm_last))
                  canbe_single_segment = false;

            if (canbe_single_segment) {
               for (let n = first; n < last; n += 2)
                  blobs.push(new DataView(res, place[n] - segm_start, place[n + 1]));
               return send_new_request(true);
            }

            if ((file.fMaxRanges === 1) || (first !== 0))
               return rejectFunc(Error('Server returns normal response when multipart was requested, disable multirange support'));

            file.fMaxRanges = 1;
            last = Math.min(last, file.fMaxRanges * 2);

            return send_new_request();
         }

         // multipart messages requires special handling

         let indx = hdr.indexOf("boundary="), boundary = "", n = first, o = 0;
         if (indx > 0) {
            boundary = hdr.slice(indx + 9);
            if ((boundary[0] == '"') && (boundary[boundary.length - 1] == '"'))
               boundary = boundary.slice(1, boundary.length - 1);
            boundary = "--" + boundary;
         } else console.error('Did not found boundary id in the response header');

         while (n < last) {

            let code1, code2 = view.getUint8(o), nline = 0, line = "",
               finish_header = false, segm_start = 0, segm_last = -1;

            while ((o < view.byteLength - 1) && !finish_header && (nline < 5)) {
               code1 = code2;
               code2 = view.getUint8(o + 1);

               if ((code1 == 13) && (code2 == 10)) {
                  if ((line.length > 2) && (line.slice(0, 2) == '--') && (line !== boundary))
                     return rejectFunc(Error('Decode multipart message, expect boundary' + boundary + ' got ' + line));

                  line = line.toLowerCase();

                  if ((line.indexOf("content-range") >= 0) && (line.indexOf("bytes") > 0)) {
                     let parts = line.slice(line.indexOf("bytes") + 6).split(/[\s-\/]+/);
                     if (parts.length === 3) {
                        segm_start = parseInt(parts[0]);
                        segm_last = parseInt(parts[1]);
                        if (!Number.isInteger(segm_start) || !Number.isInteger(segm_last) || (segm_start > segm_last)) {
                           segm_start = 0; segm_last = -1;
                        }
                     } else {
                        console.error('Fail to decode content-range', line, parts);
                     }
                  }

                  if ((nline > 1) && (line.length === 0)) finish_header = true;

                  o++; nline++; line = "";
                  code2 = view.getUint8(o + 1);
               } else {
                  line += String.fromCharCode(code1);
               }
               o++;
            }

            if (!finish_header)
               return rejectFunc(Error('Cannot decode header in multipart message'));

            if (segm_start > segm_last) {
               // fall-back solution, believe that segments same as requested
               blobs.push(new DataView(res, o, place[n + 1]));
               o += place[n + 1];
               n += 2;
            } else {
               while ((n < last) && (place[n] >= segm_start) && (place[n] + place[n + 1] - 1 <= segm_last)) {
                  blobs.push(new DataView(res, o + place[n] - segm_start, place[n + 1]));
                  n += 2;
               }

               o += (segm_last - segm_start + 1);
            }
         }

         send_new_request(true);
      }

      return send_new_request(true).then(() => promise);
   }

   /** @summary Returns file name */
   getFileName() { return this.fFileName; }

   /** @summary Get directory with given name and cycle
    * @desc Function only can be used for already read directories, which are preserved in the memory
    * @private */
   getDir(dirname, cycle) {

      if ((cycle === undefined) && (typeof dirname == 'string')) {
         const pos = dirname.lastIndexOf(';');
         if (pos > 0) { cycle = parseInt(dirname.slice(pos + 1)); dirname = dirname.slice(0, pos); }
      }

      for (let j = 0; j < this.fDirectories.length; ++j) {
         const dir = this.fDirectories[j];
         if (dir.dir_name != dirname) continue;
         if ((cycle !== undefined) && (dir.dir_cycle !== cycle)) continue;
         return dir;
      }
      return null;
   }

   /** @summary Retrieve a key by its name and cycle in the list of keys
    * @desc If only_direct not specified, returns Promise while key keys must be read first from the directory
    * @private */
   getKey(keyname, cycle, only_direct) {

      if (typeof cycle != 'number') cycle = -1;
      let bestkey = null;
      for (let i = 0; i < this.fKeys.length; ++i) {
         const key = this.fKeys[i];
         if (!key || (key.fName !== keyname)) continue;
         if (key.fCycle == cycle) { bestkey = key; break; }
         if ((cycle < 0) && (!bestkey || (key.fCycle > bestkey.fCycle))) bestkey = key;
      }
      if (bestkey)
         return only_direct ? bestkey : Promise.resolve(bestkey);

      let pos = keyname.lastIndexOf("/");
      // try to handle situation when object name contains slashes (bad practice anyway)
      while (pos > 0) {
         let dirname = keyname.slice(0, pos),
             subname = keyname.slice(pos + 1),
             dir = this.getDir(dirname);

         if (dir) return dir.getKey(subname, cycle, only_direct);

         let dirkey = this.getKey(dirname, undefined, true);
         if (dirkey && !only_direct && (dirkey.fClassName.indexOf("TDirectory") == 0))
            return this.readObject(dirname).then(newdir => newdir.getKey(subname, cycle));

         pos = keyname.lastIndexOf("/", pos - 1);
      }

      return only_direct ? null : Promise.reject(Error("Key not found " + keyname));
   }

   /** @summary Read and inflate object buffer described by its key
    * @private */
   readObjBuffer(key) {

      return this.readBuffer([key.fSeekKey + key.fKeylen, key.fNbytes - key.fKeylen]).then(blob1 => {

         if (key.fObjlen <= key.fNbytes - key.fKeylen) {
            let buf = new TBuffer(blob1, 0, this);
            buf.fTagOffset = key.fKeylen;
            return buf;
         }

         return R__unzip(blob1, key.fObjlen).then(objbuf => {
            if (!objbuf) return Promise.reject(Error("Fail to UNZIP buffer"));
            let buf = new TBuffer(objbuf, 0, this);
            buf.fTagOffset = key.fKeylen;
            return buf;

         });
      });
   }

   /** @summary Read any object from a root file
     * @desc One could specify cycle number in the object name or as separate argument
     * @param {string} obj_name - name of object, may include cycle number like "hpxpy;1"
     * @param {number} [cycle] - cycle number, also can be included in obj_name
     * @returns {Promise} promise with object read
     * @example
     * let f = await openFile("https://root.cern/js/files/hsimple.root");
     * let obj = await f.readObject("hpxpy;1");
     * console.log(`Read object of type ${obj._typename}`); */
   readObject(obj_name, cycle, only_dir) {

      let pos = obj_name.lastIndexOf(";");
      if (pos > 0) {
         cycle = parseInt(obj_name.slice(pos + 1));
         obj_name = obj_name.slice(0, pos);
      }

      if (typeof cycle != 'number') cycle = -1;
      // remove leading slashes
      while (obj_name.length && (obj_name[0] == "/")) obj_name = obj_name.slice(1);

      // one uses Promises while in some cases we need to
      // read sub-directory to get list of keys
      // in such situation calls are asynchrone
      return this.getKey(obj_name, cycle).then(key => {

         if ((obj_name == "StreamerInfo") && (key.fClassName == clTList))
            return this.fStreamerInfos;

         let isdir = false;

         if ((key.fClassName == 'TDirectory' || key.fClassName == 'TDirectoryFile')) {
            let dir = this.getDir(obj_name, cycle);
            if (dir) return dir;
            isdir = true;
         }

         if (!isdir && only_dir)
            return Promise.reject(Error(`Key ${obj_name} is not directory}`));

         return this.readObjBuffer(key).then(buf => {

            if (isdir) {
               let dir = new TDirectory(this, obj_name, cycle);
               dir.fTitle = key.fTitle;
               return dir.readKeys(buf);
            }

            let obj = {};
            buf.mapObject(1, obj); // tag object itself with id==1
            buf.classStreamer(obj, key.fClassName);

            if ((key.fClassName === 'TF1') || (key.fClassName === 'TF2'))
               return this._readFormulas(obj);

            return obj;
         });
      });
   }

   /** @summary read formulas from the file and add them to TF1/TF2 objects
     * @private */
   _readFormulas(tf1) {

      let arr = [];

      for (let indx = 0; indx < this.fKeys.length; ++indx)
         if (this.fKeys[indx].fClassName == 'TFormula')
            arr.push(this.readObject(this.fKeys[indx].fName, this.fKeys[indx].fCycle));

      return Promise.all(arr).then(formulas => {
         formulas.forEach(obj => tf1.addFormula(obj));
         return tf1;
      });
   }

   /** @summary extract streamer infos from the buffer
     * @private */
   extractStreamerInfos(buf) {
      if (!buf) return;

      let lst = {};
      buf.mapObject(1, lst);
      buf.classStreamer(lst, clTList);

      lst._typename = "TStreamerInfoList";

      this.fStreamerInfos = lst;

      if (internals.addStreamerInfosForPainter)
         internals.addStreamerInfosForPainter(lst);

      for (let k = 0; k < lst.arr.length; ++k) {
         let si = lst.arr[k];
         if (!si.fElements) continue;
         for (let l = 0; l < si.fElements.arr.length; ++l) {
            let elem = si.fElements.arr[l];

            if (!elem.fTypeName || !elem.fType) continue;

            let typ = elem.fType, typname = elem.fTypeName;

            if (typ >= 60) {
               if ((typ === kStreamer) && (elem._typename == "TStreamerSTL") && elem.fSTLtype && elem.fCtype && (elem.fCtype < 20)) {
                  let prefix = (StlNames[elem.fSTLtype] || "undef") + "<";
                  if ((typname.indexOf(prefix) === 0) && (typname[typname.length - 1] == ">")) {
                     typ = elem.fCtype;
                     typname = typname.slice(prefix.length, typname.length - 1).trim();

                     if ((elem.fSTLtype === kSTLmap) || (elem.fSTLtype === kSTLmultimap))
                        if (typname.indexOf(",") > 0) typname = typname.slice(0, typname.indexOf(",")).trim();
                        else continue;
                  }
               }
               if (typ >= 60) continue;
            } else {
               if ((typ > 20) && (typname[typname.length - 1] == "*")) typname = typname.slice(0, typname.length - 1);
               typ = typ % 20;
            }

            const kind = getTypeId(typname);
            if (kind === typ) continue;

            if ((typ === kBits) && (kind === kUInt)) continue;
            if ((typ === kCounter) && (kind === kInt)) continue;

            if (typname && typ && (this.fBasicTypes[typname] !== typ))
               this.fBasicTypes[typname] = typ;
         }
      }
   }

   /** @summary Read file keys
     * @private */
   readKeys() {

      // with the first readbuffer we read bigger amount to create header cache
      return this.readBuffer([0, 1024]).then(blob => {
         let buf = new TBuffer(blob, 0, this);

         if (buf.substring(0, 4) !== 'root')
            return Promise.reject(Error(`Not a ROOT file ${this.fURL}`));

         buf.shift(4);

         this.fVersion = buf.ntou4();
         this.fBEGIN = buf.ntou4();
         if (this.fVersion < 1000000) { //small file
            this.fEND = buf.ntou4();
            this.fSeekFree = buf.ntou4();
            this.fNbytesFree = buf.ntou4();
            buf.shift(4); // const nfree = buf.ntoi4();
            this.fNbytesName = buf.ntou4();
            this.fUnits = buf.ntou1();
            this.fCompress = buf.ntou4();
            this.fSeekInfo = buf.ntou4();
            this.fNbytesInfo = buf.ntou4();
         } else { // new format to support large files
            this.fEND = buf.ntou8();
            this.fSeekFree = buf.ntou8();
            this.fNbytesFree = buf.ntou4();
            buf.shift(4); // const nfree = buf.ntou4();
            this.fNbytesName = buf.ntou4();
            this.fUnits = buf.ntou1();
            this.fCompress = buf.ntou4();
            this.fSeekInfo = buf.ntou8();
            this.fNbytesInfo = buf.ntou4();
         }

         // empty file
         if (!this.fSeekInfo || !this.fNbytesInfo)
            return Promise.reject(Error(`File ${this.fURL} does not provide streamer infos`));

         // extra check to prevent reading of corrupted data
         if (!this.fNbytesName || this.fNbytesName > 100000)
            return Promise.reject(Error(`Cannot read directory info of the file ${this.fURL}`));

         //*-*-------------Read directory info
         let nbytes = this.fNbytesName + 22;
         nbytes += 4;  // fDatimeC.Sizeof();
         nbytes += 4;  // fDatimeM.Sizeof();
         nbytes += 18; // fUUID.Sizeof();
         // assume that the file may be above 2 Gbytes if file version is > 4
         if (this.fVersion >= 40000) nbytes += 12;

         // this part typically read from the header, no need to optimize
         return this.readBuffer([this.fBEGIN, Math.max(300, nbytes)]);
      }).then(blob3 => {

         let buf3 = new TBuffer(blob3, 0, this);

         // keep only title from TKey data
         this.fTitle = buf3.readTKey().fTitle;

         buf3.locate(this.fNbytesName);

         // we read TDirectory part of TFile
         buf3.classStreamer(this, 'TDirectory');

         if (!this.fSeekKeys)
            return Promise.reject(Error(`Empty keys list in ${this.fURL}`));

         // read with same request keys and streamer infos
         return this.readBuffer([this.fSeekKeys, this.fNbytesKeys, this.fSeekInfo, this.fNbytesInfo]);
      }).then(blobs => {

         const buf4 = new TBuffer(blobs[0], 0, this);

         buf4.readTKey(); //
         const nkeys = buf4.ntoi4();
         for (let i = 0; i < nkeys; ++i)
            this.fKeys.push(buf4.readTKey());

         const buf5 = new TBuffer(blobs[1], 0, this),
               si_key = buf5.readTKey();
         if (!si_key)
            return Promise.reject(Error(`Fail to read StreamerInfo data in ${this.fURL}`));

         this.fKeys.push(si_key);
         return this.readObjBuffer(si_key);
      }).then(blob6 => {
          this.extractStreamerInfos(blob6);
          return this;
      });
   }

   /** @summary Read the directory content from  a root file
     * @desc If directory was already read - return previously read object
     * Same functionality as {@link TFile#readObject}
     * @param {string} dir_name - directory name
     * @param {number} [cycle] - directory cycle
     * @returns {Promise} - promise with read directory */
   readDirectory(dir_name, cycle) {
      return this.readObject(dir_name, cycle, true);
   }

   /** @summary Search streamer info
     * @param {string} clanme - class name
     * @param {number} [clversion] - class version
     * @param {number} [checksum] - streamer info checksum, have to match when specified
     * @private */
   findStreamerInfo(clname, clversion, checksum) {
      if (!this.fStreamerInfos) return null;

      const arr = this.fStreamerInfos.arr, len = arr.length;

      if (checksum !== undefined) {
         let cache = this.fStreamerInfos.cache;
         if (!cache) cache = this.fStreamerInfos.cache = {};
         let si = cache[checksum];
         if (si !== undefined) return si;

         for (let i = 0; i < len; ++i) {
            si = arr[i];
            if (si.fCheckSum === checksum)
               return cache[checksum] = si;
         }
         cache[checksum] = null; // checksum didnot found, do not try again
      } else {
         for (let i = 0; i < len; ++i) {
            let si = arr[i];
            if ((si.fName === clname) && ((si.fClassVersion === clversion) || (clversion === undefined))) return si;
         }
      }

      return null;
   }

   /** @summary Returns streamer for the class 'clname',
     * @desc From the list of streamers or generate it from the streamer infos and add it to the list
     * @private */
   getStreamer(clname, ver, s_i) {

      // these are special cases, which are handled separately
      if (clname == 'TQObject' || clname == "TBasket") return null;

      let streamer, fullname = clname;

      if (ver) {
         fullname += (ver.checksum ? ("$chksum" + ver.checksum) : ("$ver" + ver.val));
         streamer = this.fStreamers[fullname];
         if (streamer !== undefined) return streamer;
      }

      let custom = CustomStreamers[clname];

      // one can define in the user streamers just aliases
      if (typeof custom === 'string')
         return this.getStreamer(custom, ver, s_i);

      // streamer is just separate function
      if (typeof custom === 'function') {
         streamer = [{ typename: clname, func: custom }];
         return addClassMethods(clname, streamer);
      }

      streamer = [];

      if (typeof custom === 'object') {
         if (!custom.name && !custom.func) return custom;
         streamer.push(custom); // special read entry, add in the beginning of streamer
      }

      // check element in streamer infos, one can have special cases
      if (!s_i) s_i = this.findStreamerInfo(clname, ver.val, ver.checksum);

      if (!s_i) {
         delete this.fStreamers[fullname];
         if (!ver.nowarning)
            console.warn(`Not found streamer for ${clname} ver ${ver.val} checksum ${ver.checksum} full ${fullname}`);
         return null;
      }

      // special handling for TStyle which has duplicated member name fLineStyle
      if ((s_i.fName == "TStyle") && s_i.fElements)
         s_i.fElements.arr.forEach(elem => {
            if (elem.fName == "fLineStyle") elem.fName = "fLineStyles"; // like in ROOT JSON now
         });

      // for each entry in streamer info produce member function
      if (s_i.fElements)
         for (let j = 0; j < s_i.fElements.arr.length; ++j)
            streamer.push(createMemberStreamer(s_i.fElements.arr[j], this));

      this.fStreamers[fullname] = streamer;

      return addClassMethods(clname, streamer);
   }

   /** @summary Here we produce list of members, resolving all base classes
     * @private */
   getSplittedStreamer(streamer, tgt) {
      if (!streamer) return tgt;

      if (!tgt) tgt = [];

      for (let n = 0; n < streamer.length; ++n) {
         let elem = streamer[n];

         if (elem.base === undefined) {
            tgt.push(elem);
            continue;
         }

         if (elem.basename == clTObject) {
            tgt.push({
               func(buf, obj) {
                  buf.ntoi2(); // read version, why it here??
                  obj.fUniqueID = buf.ntou4();
                  obj.fBits = buf.ntou4();
                  if (obj.fBits & kIsReferenced) buf.ntou2(); // skip pid
               }
            });
            continue;
         }

         let ver = { val: elem.base };

         if (ver.val === 4294967295) {
            // this is -1 and indicates foreign class, need more workarounds
            ver.val = 1; // need to search version 1 - that happens when several versions of foreign class exists ???
         }

         let parent = this.getStreamer(elem.basename, ver);
         if (parent) this.getSplittedStreamer(parent, tgt);
      }

      return tgt;
   }

   /** @summary Fully clenaup TFile data
     * @private */
   delete() {
      this.fDirectories = null;
      this.fKeys = null;
      this.fStreamers = null;
      this.fSeekInfo = 0;
      this.fNbytesInfo = 0;
      this.fTagOffset = 0;
   }

} // class TFile


/** @summary Reconstruct ROOT object from binary buffer
  * @desc Method can be used to reconstruct ROOT objects from binary buffer
  * which can be requested from running THttpServer, using **root.bin** request
  * To decode data, one has to request streamer infos data __after__ object data
  * as it shown in example.
  *
  * Method provided for convenience only to see how binary IO works.
  * It is strongly recommended to use **root.json** request to get data directly in
  * JSON format
  *
  * @param {string} class_name - Class name of the object
  * @param {binary} obj_rawdata - data of object root.bin request
  * @param {binary} sinfo_rawdata - data of streamer info root.bin request
  * @returns {object} - created JavaScript object
  * @example
  *
  * import { httpRequest } from 'http://localhost:8080/jsrootsys/modules/core.mjs';
  * import { reconstructObject } from 'http://localhost:8080/jsrootsys/modules/io.mjs';
  *
  * let obj_data = await httpRequest("http://localhost:8080/Files/job1.root/hpx/root.bin", "buf");
  * let si_data = await httpRequest("http://localhost:8080/StreamerInfo/root.bin", "buf");
  * let histo = await reconstructObject("TH1F", obj_data, si_data);
  * console.log(`Get histogram with title = ${histo.fTitle}`);
  *
  * // request same data via root.json request
  * httpRequest("http://localhost:8080/Files/job1.root/hpx/root.json", "object")
  *            .then(histo => console.log(`Get histogram with title = ${histo.fTitle}`)); */
function reconstructObject(class_name, obj_rawdata, sinfo_rawdata) {

   let file = new TFile;
   let buf = new TBuffer(sinfo_rawdata, 0, file);
   file.extractStreamerInfos(buf);

   let obj = {};

   buf = new TBuffer(obj_rawdata, 0, file);
   buf.mapObject(obj, 1);
   buf.classStreamer(obj, class_name);

   return obj;
}

/** @summary Function to read vector element in the streamer
  * @private */
function readVectorElement(buf) {

   if (this.member_wise) {

      const n = buf.ntou4();
      let streamer = null, ver = this.stl_version;

      if (n === 0) return []; // for empty vector no need to search split streamers

      if (n > 1000000) {
         throw new Error('member-wise streaming of ' + this.conttype + " num " + n + ' member ' + this.name);
         // return [];
      }

      if ((ver.val === this.member_ver) && (ver.checksum === this.member_checksum)) {
         streamer = this.member_streamer;
      } else {
         streamer = buf.fFile.getStreamer(this.conttype, ver);

         this.member_streamer = streamer = buf.fFile.getSplittedStreamer(streamer);
         this.member_ver = ver.val;
         this.member_checksum = ver.checksum;
      }

      let res = new Array(n), i, k, member;

      for (i = 0; i < n; ++i)
         res[i] = { _typename: this.conttype }; // create objects
      if (!streamer) {
         console.error('Fail to create split streamer for', this.conttype, 'need to read ', n, 'objects version', ver);
      } else {
         for (k = 0; k < streamer.length; ++k) {
            member = streamer[k];
            if (member.split_func) {
               member.split_func(buf, res, n);
            } else {
               for (i = 0; i < n; ++i)
                  member.func(buf, res[i]);
            }
         }
      }
      return res;
   }

   const n = buf.ntou4();
   let res = new Array(n), i = 0;

   if (n > 200000) { console.error('vector streaming for of', this.conttype, n); return res; }

   if (this.arrkind > 0) { while (i < n) res[i++] = buf.readFastArray(buf.ntou4(), this.arrkind); }
   else if (this.arrkind === 0) { while (i < n) res[i++] = buf.readTString(); }
   else if (this.isptr) { while (i < n) res[i++] = buf.readObjectAny(); }
   else if (this.submember) { while (i < n) res[i++] = this.submember.readelem(buf); }
   else { while (i < n) res[i++] = buf.classStreamer({}, this.conttype); }

   return res;
}


/** @summary Function used in streamer to read std::map object
  * @private */
function readMapElement(buf) {
   let streamer = this.streamer;

   if (this.member_wise) {
      // when member-wise streaming is used, version is written
      const ver = this.stl_version;

      if (this.si) {
         let si = buf.fFile.findStreamerInfo(this.pairtype, ver.val, ver.checksum);

         if (this.si !== si) {

            streamer = getPairStreamer(si, this.pairtype, buf.fFile);
            if (!streamer || streamer.length !== 2) {
               console.log('Fail to produce streamer for ', this.pairtype);
               return null;
            }
         }
      }
   }

   const n = buf.ntoi4();
   let i, res = new Array(n);
   if (this.member_wise && (buf.remain() >= 6)) {
      if (buf.ntoi2() == kStreamedMemberWise)
         buf.shift(4);
      else
         buf.shift(-2); // rewind
   }

   for (i = 0; i < n; ++i) {
      res[i] = { _typename: this.pairtype };
      streamer[0].func(buf, res[i]);
      if (!this.member_wise) streamer[1].func(buf, res[i]);
   }

   // due-to member-wise streaming second element read after first is completed
   if (this.member_wise)
      for (i = 0; i < n; ++i)
         streamer[1].func(buf, res[i]);

   return res;
}

// =============================================================

/**
  * @summary Interface to read local file in the browser
  *
  * @hideconstructor
  * @desc Use {@link openFile} to create instance of the class
  * @private
  */

class TLocalFile extends TFile {

   constructor(file) {
      super(null);
      this.fUseStampPar = false;
      this.fLocalFile = file;
      this.fEND = file.size;
      this.fFullURL = file.name;
      this.fURL = file.name;
      this.fFileName = file.name;
   }

   /** @summary Open local file
     * @returns {Promise} after file keys are read */
   _open() { return this.readKeys(); }

   /** @summary read buffer from local file */
   readBuffer(place, filename /*, progress_callback */) {
      let file = this.fLocalFile;

      return new Promise((resolve, reject) => {
         if (filename)
            return reject(Error(`Cannot access other local file ${filename}`));

         let reader = new FileReader(), cnt = 0, blobs = [];

         reader.onload = function(evnt) {
            let res = new DataView(evnt.target.result);
            if (place.length === 2) return resolve(res);

            blobs.push(res);
            cnt += 2;
            if (cnt >= place.length) return resolve(blobs);
            reader.readAsArrayBuffer(file.slice(place[cnt], place[cnt] + place[cnt + 1]));
         }

         reader.readAsArrayBuffer(file.slice(place[0], place[0] + place[1]));
      });
   }

} // class TLocalFile

/**
  * @summary Interface to read file in node.js
  *
  * @hideconstructor
  * @desc Use {@link openFile} to create instance of the class
  * @private
  */

class TNodejsFile extends TFile {
   constructor(filename) {
      super(null);
      this.fUseStampPar = false;
      this.fEND = 0;
      this.fFullURL = filename;
      this.fURL = filename;
      this.fFileName = filename;
   }

   /** @summary Open file in node.js
     * @returns {Promise} after file keys are read */
   _open() {
      return import('fs').then(fs => {

         this.fs = fs;

         return new Promise((resolve,reject) =>

            this.fs.open(this.fFileName, 'r', (status, fd) => {
               if (status) {
                  console.log(status.message);
                  return reject(Error(`Not possible to open ${this.fFileName} inside node.js`));
               }
               let stats = this.fs.fstatSync(fd);

               this.fEND = stats.size;

               this.fd = fd;

               this.readKeys().then(resolve).catch(reject);
            })
         );
      });
   }

   /** @summary Read buffer from node.js file
     * @returns {Promise} with requested blocks */
   readBuffer(place, filename /*, progress_callback */) {
      return new Promise((resolve, reject) => {
         if (filename)
            return reject(Error(`Cannot access other local file ${filename}`));

         if (!this.fs || !this.fd)
            return reject(Error(`File is not opened ${this.fFileName}`));

         let cnt = 0, blobs = [];

         let readfunc = (err, bytesRead, buf) => {

            let res = new DataView(buf.buffer, buf.byteOffset, place[cnt + 1]);
            if (place.length === 2) return resolve(res);

            blobs.push(res);
            cnt += 2;
            if (cnt >= place.length) return resolve(blobs);
            this.fs.read(this.fd, Buffer.alloc(place[cnt + 1]), 0, place[cnt + 1], place[cnt], readfunc);
         }

         this.fs.read(this.fd, Buffer.alloc(place[1]), 0, place[1], place[0], readfunc);
      });
   }

} // class TNodejsFile

/**
  * @summary Proxy to read file contenxt
  *
  * @desc Should implement followinf methods
  *        openFile() - return Promise with true when file can be open normally
  *        getFileName() - returns string with file name
  *        getFileSize() - returns size of file
  *        readBuffer(pos, len) - return promise with DataView for requested position and length
  * @private
  */

class FileProxy {

   openFile() { return Promise.resolve(false); }

   getFileName() { return ""; }

   getFileSize() { return 0; }

   readBuffer(/*pos, sz*/) { return Promise.resolve(null); }

} // class FileProxy

/**
  * @summary File to use file context via FileProxy
  *
  * @hideconstructor
  * @desc Use {@link openFile} to create instance of the class, providing FileProxy as argument
  * @private
  */

class TProxyFile extends TFile {

   constructor(proxy) {
      super(null);
      this.fUseStampPar = false;
      this.proxy = proxy;
   }

   /** @summary Open file
     * @returns {Promise} after file keys are read */
   _open() {
      return this.proxy.openFile().then(res => {
         if (!res) return false;
         this.fEND = this.proxy.getFileSize();
         this.fFullURL = this.fURL = this.fFileName = this.proxy.getFileName();
         if (typeof this.fFileName == 'string') {
            let p = this.fFileName.lastIndexOf("/");
            if ((p > 0) && (p < this.fFileName.length - 4))
               this.fFileName = this.fFileName.slice(p+1);
         }
         return this.readKeys();
      });
   }

   /** @summary Read buffer from FileProxy
     * @returns {Promise} with requested blocks */
   readBuffer(place, filename /*, progress_callback */) {
      if (filename)
         return Promise.reject(Error(`Cannot access other file ${filename}`));

      if (!this.proxy)
         return Promise.reject(Error(`File is not opened ${this.fFileName}`));

      if (place.length == 2)
         return this.proxy.readBuffer(place[0], place[1]);

      let arr = [];
      for (let k = 0; k < place.length; k+=2)
         arr.push(this.proxy.readBuffer(place[k], place[k+1]));
      return Promise.all(arr);
   }

} // class TProxyFile


/** @summary Open ROOT file for reading
  * @desc Generic method to open ROOT file for reading
  * Following kind of arguments can be provided:
  *  - string with file URL (see example). In node.js environment local file like "file://hsimple.root" can be specified
  *  - [File]{@link https://developer.mozilla.org/en-US/docs/Web/API/File} instance which let read local files from browser
  *  - [ArrayBuffer]{@link https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/ArrayBuffer} instance with complete file content
  *  - [FileProxy]{@link FileProxy} let access arbitrary files via tiny proxy API
  * @param {string|object} arg - argument for file open like url, see details
  * @returns {object} - Promise with {@link TFile} instance when file is opened
  * @example
  *
  * import { openFile } from '/path_to_jsroot/modules/io.mjs';
  * let f = await openFile("https://root.cern/js/files/hsimple.root");
  * console.log(`Open file ${f.getFileName()}`); */
function openFile(arg) {

   let file;

   if (isNodeJs() && (typeof arg == "string")) {
      if (arg.indexOf("file://") == 0)
         file = new TNodejsFile(arg.slice(7));
      else if (arg.indexOf("http") !== 0)
         file = new TNodejsFile(arg);
   }

   if (!file && (typeof arg === 'object') && (arg instanceof FileProxy))
      file = new TProxyFile(arg);

   if (!file && (typeof arg === 'object') && (arg instanceof ArrayBuffer)) {
      file = new TFile("localfile.root");
      file.assignFileContent(arg);
   }

   if (!file && (typeof arg === 'object') && arg.size && arg.name)
      file = new TLocalFile(arg);

   if (!file)
      file = new TFile(arg);

   return file._open();
}

// special way to assign methods when streaming objects
addClassMethods(clTNamed, CustomStreamers[clTNamed]);
addClassMethods(clTObjString, CustomStreamers[clTObjString]);

export {
   kChar, kShort, kInt, kLong, kFloat, kCounter,
   kCharStar, kDouble, kDouble32, kLegacyChar,
   kUChar, kUShort, kUInt, kULong, kBits,
   kLong64, kULong64, kBool, kFloat16,
   kBase, kOffsetL, kOffsetP, kObject, kAny, kObjectp, kObjectP, kTString,
   kAnyP, kStreamer, kStreamLoop, kSTLp, kSTL,
   R__unzip, addUserStreamer, createStreamerElement, createMemberStreamer,
   openFile, reconstructObject, FileProxy,
   TBuffer /*, TDirectory, TFile, TLocalFile, TNodejsFile */
};
