// JSROOTCore.js
//
// core methods for Javascript ROOT IO.
//


// Globals

BIT = function(n) { return 1 << (n); }

var EAxisBits = {
   kTickPlus      : BIT(9),
   kTickMinus     : BIT(10),
   kAxisRange     : BIT(11),
   kCenterTitle   : BIT(12),
   kCenterLabels  : BIT(14),
   kRotateTitle   : BIT(15),
   kPalette       : BIT(16),
   kNoExponent    : BIT(17),
   kLabelsHori    : BIT(18),
   kLabelsVert    : BIT(19),
   kLabelsDown    : BIT(20),
   kLabelsUp      : BIT(21),
   kIsInteger     : BIT(22),
   kMoreLogLabels : BIT(23),
   kDecimals      : BIT(11)
};

var EStatusBits = {
   kCanDelete     : BIT(0),   // if object in a list can be deleted
   kMustCleanup   : BIT(3),   // if object destructor must call RecursiveRemove()
   kObjInCanvas   : BIT(3),   // for backward compatibility only, use kMustCleanup
   kIsReferenced  : BIT(4),   // if object is referenced by a TRef or TRefArray
   kHasUUID       : BIT(5),   // if object has a TUUID (its fUniqueID=UUIDNumber)
   kCannotPick    : BIT(6),   // if object in a pad cannot be picked
   kNoContextMenu : BIT(8),   // if object does not want context menu
   kInvalidObject : BIT(13)   // if object ctor succeeded but object should not be used
};

   // TH1 status bits
var TH1StatusBits = {
   kNoStats       : BIT(9),  // don't draw stats box
   kUserContour   : BIT(10), // user specified contour levels
   kCanRebin      : BIT(11), // can rebin axis
   kLogX          : BIT(15), // X-axis in log scale
   kIsZoomed      : BIT(16), // bit set when zooming on Y axis
   kNoTitle       : BIT(17), // don't draw the histogram title
   kIsAverage     : BIT(18)  // Bin contents are average (used by Add)
};

var kTextNDC = BIT(14);

var kNotDraw = BIT(9);  // don't draw the function (TF1) when in a TH1

var kNstat = 13;

var EErrorType = {
   kERRORMEAN : 0,
   kERRORSPREAD : 1,
   kERRORSPREADI : 2,
   kERRORSPREADG : 3
};

var EBinErrorOpt = {
   kNormal : 0,    // errors with Normal (Wald) approximation: errorUp=errorLow= sqrt(N)
   kPoisson : 1,   // errors from Poisson interval at 68.3% (1 sigma)
   kPoisson2 : 2   // errors from Poisson interval at 95% CL (~ 2 sigma)
};

var kCARTESIAN   = 1;
var kPOLAR       = 2;
var kCYLINDRICAL = 3;
var kSPHERICAL   = 4;
var kRAPIDITY    = 5;

String.prototype.endsWith = function(str, ignoreCase) {
   return (ignoreCase ? this.toUpperCase() : this).slice(-str.length)
       == (ignoreCase ? str.toUpperCase() : str);
};

gaus = function(f, x, i) {
   return f['fParams'][i+0] * Math.exp(-0.5 * Math.pow((x-f['fParams'][i+1]) / f['fParams'][i+2], 2));
};

gausn = function(f, x, i) {
   return gaus(f, x, i)/(sqrt(2 * Math.PI) * f['fParams'][i+2]);
};

expo = function(f, x, i) {
   return Math.exp(f['fParams'][i+0] + f['fParams'][i+1] * x);
};

landau = function(f, x, i) {
   return JSROOTMath.Landau(x, f['fParams'][i+1],f['fParams'][i+2], false);
};

landaun = function(f, x, i) {
   return JSROOTMath.Landau(x, f['fParams'][i+1],f['fParams'][i+2], true);
};


(function(){

   if (typeof JSROOTCore == "object"){
      var e1 = new Error("JSROOTCore is already defined");
      e1.source = "JSROOTCore.js";
      throw e1;
   }

   JSROOTCore = {};

   JSROOTCore.version = "1.0 2012/08/08";

   JSROOTCore.clone = function(obj) {
      return jQuery.extend(true, {}, obj);
   };
   
   JSROOTCore.id_counter = 0;
   
   // This is part of the JSON-R code, found on 
   // https://github.com/graniteds/jsonr
   // Only unref part was used, arrays are not accounted as objects
   // Should be used to reintroduce objects references, produced by TBufferJSON 

	JSROOTCore.JSONR_unref = function(value, dy)
	{
	   var c, i, k, ks;
	   if (!dy) dy = [];
	
	   switch (typeof value) {
	   case 'string':
	       if ((value.length > 5) && (value.substr(0, 5) == "$ref:")) {
	          c = parseInt(value.substr(5));
	          if (!isNaN(c) && (c < dy.length)) {
	             value = dy[c];
	             // console.log("replace index " + c + "  name = " + value.fName);
	          }
	       }
	       break;
	
	   case 'object':
	      if (value !== null) {
	         
	         if (Object.prototype.toString.apply(value) === '[object Array]') {
	            for (i = 0; i < value.length; i++) {
	               value[i] = JSROOTCore.JSONR_unref(value[i], dy);
	            }
	         } else {
	
	            // account only objects in ref table
	            if (dy.indexOf(value) === -1) {
	               //if (dy.length<10) console.log("Add object " + value._typename + "  $ref:" + dy.length);
	               dy.push(value);
	            }
	
	            // add methods to all objects, where _typename is specified
	            if (('_typename' in value) && (typeof JSROOTCore == "object"))
	               JSROOTCore.addMethods(value);
	
	            ks = Object.keys(value);
	            for (i = 0; i < ks.length; i++) {
	               k = ks[i];
	               //if (dy.length<10) console.log("Check field " + k);
	               value[k] = JSROOTCore.JSONR_unref(value[k], dy);
	            }
	         }
	      }
	      break;
	   }
	
	   return value;
	}

   JSROOTCore.addFormula = function(obj) {
      var formula = obj['fTitle'];
      formula = formula.replace('abs(', 'Math.abs(');
      formula = formula.replace('sin(', 'Math.sin(');
      formula = formula.replace('cos(', 'Math.cos(');
      var code = obj['fName'] + " = function(x) { return " + formula + " };";
      eval(code);
      var sig = obj['fName']+'(x)';
      var pos = function_list.indexOf(sig);
      if (pos >= 0) {
         function_list.splice(pos, 1);
      }
      function_list.push(sig);
   };

   JSROOTCore.CreateTList = function() {
      var list = {};
      list['_typename'] = "JSROOTIO.TList";
      list['name'] = "TList";
      list['arr'] = new Array;
      list['opt'] = new Array;
      return list;
   }

   JSROOTCore.CreateTAxis = function() {
      var axis = {};

      axis['_typename'] = "JSROOTIO.TAxis";
      axis['fBits'] = 0x3000008;
      axis['fBits2'] = 0;
      axis['fXmin'] = 0;
      axis['fXmax'] = 0;
      axis['fNbins'] = 0;
      axis['fN'] = 0;
      axis['fXbins'] = new Array;
      axis['fFirst'] = 0;
      axis['fLast'] = 0;
      axis['fName'] = "";
      axis['fTitle'] = "";
      axis['fTimeDisplay'] = false;
      axis['fTimeFormat'] = "";
      axis['fNdivisions'] = 510;
      axis['fAxisColor'] = 1;
      axis['fLabelColor'] = 1;
      axis['fLabelFont'] = 42;
      axis['fLabelOffset']  = 0.05;
      axis['fLabelSize']  = 0.035;
      axis['fTickLength'] = 0.03;
      axis['fTitleOffset'] = 1;
      axis['fTitleSize']  = 0.035;
      axis['fTitleColor'] = 1;
      axis['fTitleFont'] = 42;
      JSROOTCore.addMethods(axis);
      return axis;
   }

   JSROOTCore.CreateTH1 = function(nbinsx) {
      var histo = {};
      histo['_typename'] = "JSROOTIO.TH1I";
      histo['fBits'] = 0x3000008;
      histo['fName'] = "dummy_histo_" + this.id_counter++;
      histo['fTitle'] = "dummytitle";
      histo['fMinimum'] = -1111;
      histo['fMaximum'] = -1111;
      histo['fOption'] = "";
      histo['fFillColor'] = 0;
      histo['fLineColor'] = 0;
      histo['fLineWidth'] = 1;
      histo['fBinStatErrOpt'] = 0;
      histo['fNcells'] = 0;
      histo['fN'] = 0;
      histo['fArray'] = new Array;
      histo['fSumw2'] = new Array;
      histo['fFunctions'] = JSROOTCore.CreateTList();

      histo['fXaxis'] = JSROOTCore.CreateTAxis();
      histo['fYaxis'] = JSROOTCore.CreateTAxis();

      if (nbinsx!=null) {
         histo['fNcells'] = nbinsx+2;
         for (var i=0;i<histo['fNcells'];i++) histo['fArray'].push(0);
         histo['fXaxis']['fNbins'] = nbinsx;
         histo['fXaxis']['fXmin'] = 0;
         histo['fXaxis']['fXmax'] = nbinsx;
      }

      JSROOTCore.addMethods(histo);

      return histo;
   }

   JSROOTCore.CreateTH2 = function(nbinsx, nbinsy) {
      var histo = {};
      histo['_typename'] = "JSROOTIO.TH2I";
      histo['fBits'] = 0x3000008;
      histo['fName'] = "dummy_histo_" + this.id_counter++;
      histo['fTitle'] = "dummytitle";
      histo['fMinimum'] = -1111;
      histo['fMaximum'] = -1111;
      histo['fOption'] = "";
      histo['fFillColor'] = 0;
      histo['fLineColor'] = 0;
      histo['fLineWidth'] = 1;
      histo['fBinStatErrOpt'] = 0;
      histo['fNcells'] = 0;
      histo['fN'] = 0;
      histo['fArray'] = new Array;
      histo['fSumw2'] = new Array;
      histo['fFunctions'] = JSROOTCore.CreateTList();
      histo['fContour'] = new Array;

      histo['fXaxis'] = JSROOTCore.CreateTAxis();
      histo['fYaxis'] = JSROOTCore.CreateTAxis();
      histo['fZaxis'] = JSROOTCore.CreateTAxis();

      if ((nbinsx!=null) && (nbinsy!=null)) {
         histo['fNcells'] = (nbinsx+2) * (nbinsy+2);
         for (var i=0;i<histo['fNcells'];i++) histo['fArray'].push(0);
         histo['fXaxis']['fNbins'] = nbinsx;
         histo['fYaxis']['fNbins'] = nbinsy;

         histo['fXaxis']['fXmin'] = 0;
         histo['fXaxis']['fXmax'] = nbinsx;
         histo['fYaxis']['fXmin'] = 0;
         histo['fYaxis']['fXmax'] = nbinsy;
      }

      JSROOTCore.addMethods(histo);

      return histo;
   }

   JSROOTCore.CreateTGraph = function(npoints) {
      var graph = {};
      graph['_typename'] = "JSROOTIO.TGraph";
      graph['fBits'] = 0x3000408;
      graph['fName'] = "dummy_graph_" + this.id_counter++;
      graph['fTitle'] = "dummytitle";
      graph['fMinimum'] = -1111;
      graph['fMaximum'] = -1111;
      graph['fOption'] = "";
      graph['fFillColor'] = 0;
      graph['fFillStyle'] = 1001;
      graph['fLineColor'] = 2;
      graph['fLineStyle'] = 1;
      graph['fLineWidth'] = 2;
      graph['fMarkerColor'] = 4;
      graph['fMarkerStyle'] = 21;
      graph['fMarkerSize'] = 1;
      graph['fMaxSize'] = 0;
      graph['fNpoints'] = 0;
      graph['fX'] = new Array;
      graph['fY'] = new Array;
      graph['fFunctions'] = JSROOTCore.CreateTList();
      graph['fHistogram'] = JSROOTCore.CreateTH1();

      if (npoints>0) {
         graph['fMaxSize'] = npoints;
         graph['fNpoints'] = npoints;
         for (var i=0;i<npoints;i++) {
            graph['fX'].push(i);
            graph['fY'].push(i);
         }
         JSROOTCore.AdjustTGraphRanges(graph);
      }

      JSROOTCore.addMethods(graph);
      return graph;
   }

   JSROOTCore.AdjustTGraphRanges = function(graph) {
      if (graph['fNpoints']==0) return;

      var minx = graph['fX'][0], maxx = minx;
      var miny = graph['fY'][0], maxy = miny;

      for (var i=1;i<graph['fNpoints'];i++) {
         if (graph['fX'][i] < minx) minx = graph['fX'][i];
         if (graph['fX'][i] > maxx) maxx = graph['fX'][i];
         if (graph['fY'][i] < miny) miny = graph['fY'][i];
         if (graph['fY'][i] > maxy) maxy = graph['fY'][i];
      }

      if (miny==maxy) maxy = miny + 1;

      // console.log("search minx = " + minx + " maxx = " + maxx);

      graph['fHistogram']['fXaxis']['fXmin'] = minx;
      graph['fHistogram']['fXaxis']['fXmax'] = maxx;

      graph['fHistogram']['fYaxis']['fXmin'] = miny;
      graph['fHistogram']['fYaxis']['fXmax'] = maxy;
   }


   JSROOTCore.addMethods = function(obj) {
      // check object type and add methods if needed
      if (('fBits' in obj) && !('TestBit' in obj)) {
         obj['TestBit'] = function (f) {
            return ((obj['fBits'] & f) != 0);
         };
      }
      if (!('_typename' in obj))
         return;
      if (obj['_typename'].indexOf("JSROOTIO.TAxis") == 0) {
         obj['getFirst'] = function() {
            if (!this.TestBit(EAxisBits.kAxisRange)) return 1;
            return this['fFirst'];
         };
         obj['getLast'] = function() {
            if (!this.TestBit(EAxisBits.kAxisRange)) return this['fNbins'];
            return this['fLast'];
         };
         obj['getBinCenter'] = function(bin) {
            // Return center of bin
            var binwidth;
            if (!this['fN'] || bin < 1 || bin > this['fNbins']) {
               binwidth = (this['fXmax'] - this['fXmin']) / this['fNbins'];
               return this['fXmin'] + (bin-1) * binwidth + 0.5*binwidth;
            } else {
               binwidth = this['fXbins'][bin] - this['fXbins'][bin-1];
               return this['fXbins'][bin-1] + 0.5*binwidth;
            }
         };
      }
      if ((obj['_typename'].indexOf("TFormula") != -1) ||
          (obj['_typename'].indexOf("JSROOTIO.TF1") == 0)) {
         obj['evalPar'] = function(x) {
            var i, _function = this['fTitle'];
            _function = _function.replace('TMath::Exp(', 'Math.exp(');
            _function = _function.replace('TMath::Abs(', 'Math.abs(');
            _function = _function.replace('gaus(', 'gaus(this, ' + x + ', ');
            _function = _function.replace('gausn(', 'gausn(this, ' + x + ', ');
            _function = _function.replace('expo(', 'expo(this, ' + x + ', ');
            _function = _function.replace('landau(', 'landau(this, ' + x + ', ');
            _function = _function.replace('landaun(', 'landaun(this, ' + x + ', ');
            _function = _function.replace('pi', 'Math.PI');
            for (i=0;i<this['fNpar'];++i) {
               while(_function.indexOf('['+i+']') != -1)
                  _function = _function.replace('['+i+']', this['fParams'][i])
            }
            for (i=0;i<function_list.length;++i) {
               var f = function_list[i].substring(0, function_list[i].indexOf('('));
               if (_function.indexOf(f) != -1) {
                  var fa = function_list[i].replace('(x)', '(' + x + ')');
                  _function = _function.replace(f, fa);
               }
            }
            // use regex to replace ONLY the x variable (i.e. not 'x' in Math.exp...)
            _function = _function.replace(/\b(x)\b/gi, x)
            _function = _function.replace(/\b(sin)\b/gi, 'Math.sin')
            _function = _function.replace(/\b(cos)\b/gi, 'Math.cos')
            _function = _function.replace(/\b(tan)\b/gi, 'Math.tan')
            var ret = eval(_function);
            return ret;
         };
      }
      if (obj['_typename'].indexOf("JSROOTIO.TGraph") == 0) {
         obj['computeRange'] = function() {
            // Compute the x/y range of the points in this graph
            var i, xmin = 0, xmax = 0, ymin = 0, ymax = 0;
            if (this['fNpoints'] > 0) {
               xmin = xmax = this['fX'][0];
               ymin = ymax = this['fY'][0];
               for (i=1; i<this['fNpoints']; i++) {
                  if (this['fX'][i] < xmin) xmin = this['fX'][i];
                  if (this['fX'][i] > xmax) xmax = this['fX'][i];
                  if (this['fY'][i] < ymin) ymin = this['fY'][i];
                  if (this['fY'][i] > ymax) ymax = this['fY'][i];
               }
            }
            return {
               xmin: xmin,
               xmax: xmax,
               ymin: ymin,
               ymax: ymax
            };
         };
      }
      if (obj['_typename'].indexOf("JSROOTIO.TH1") == 0) obj['fDimension'] = 1;
      if (obj['_typename'].indexOf("JSROOTIO.TH2") == 0) obj['fDimension'] = 2;
      if (obj['_typename'].indexOf("JSROOTIO.TH3") == 0) obj['fDimension'] = 3;
      if (obj['_typename'].indexOf("JSROOTIO.TH1") == 0 ||
          obj['_typename'].indexOf("JSROOTIO.TH2") == 0 ||
          obj['_typename'].indexOf("JSROOTIO.TH3") == 0) {
         obj['getBinError'] = function(bin) {
            //   -*-*-*-*-*Return value of error associated to bin number bin*-*-*-*-*
            //    if the sum of squares of weights has been defined (via Sumw2),
            //    this function returns the sqrt(sum of w2).
            //    otherwise it returns the sqrt(contents) for this bin.
            if (bin < 0) bin = 0;
            if (bin >= this['fNcells']) bin = this['fNcells'] - 1;
            if (this['fN'] && this['fSumw2'].length > 0) {
               var err2 = this['fSumw2'][bin];
               return Math.sqrt(err2);
            }
            var error2 = Math.abs(this['fArray'][bin]);
            return Math.sqrt(error2);
         };
         obj['getBinErrorLow'] = function(bin) {
            //   -*-*-*-*-*Return lower error associated to bin number bin*-*-*-*-*
            //    The error will depend on the statistic option used will return
            //     the binContent - lower interval value
            if (this['fBinStatErrOpt'] == EBinErrorOpt.kNormal || this['fN']) return this.getBinError(bin);
            if (bin < 0) bin = 0;
            if (bin >= this['fNcells']) bin = this['fNcells'] - 1;
            var alpha = 1.0 - 0.682689492;
            if (this['fBinStatErrOpt'] == EBinErrorOpt.kPoisson2) alpha = 0.05;
            var c = this['fArray'][bin];
            var n = Math.round(c);
            if (n < 0) {
               alert("GetBinErrorLow : Histogram has negative bin content-force usage to normal errors");
               this['fBinStatErrOpt'] = EBinErrorOpt.kNormal;
               return this.getBinError(bin);
            }
            if (n == 0) return 0;
            return c - JSROOTMath.gamma_quantile( alpha/2, n, 1.);
         };
         obj['getBinErrorUp'] = function(bin) {
            //   -*-*-*-*-*Return lower error associated to bin number bin*-*-*-*-*
            //    The error will depend on the statistic option used will return
            //     the binContent - lower interval value
            if (this['fBinStatErrOpt'] == EBinErrorOpt.kNormal || this['fN']) return this.getBinError(bin);
            if (bin < 0) bin = 0;
            if (bin >= this['fNcells']) bin = this['fNcells'] - 1;
            var alpha = 1.0 - 0.682689492;
            if (this['fBinStatErrOpt'] == EBinErrorOpt.kPoisson2) alpha = 0.05;
            var c = this['fArray'][bin];
            var n = Math.round(c);
            if (n < 0) {
               alert("GetBinErrorLow : Histogram has negative bin content-force usage to normal errors");
               this['fBinStatErrOpt'] = EBinErrorOpt.kNormal;
               return this.getBinError(bin);
            }
            // for N==0 return an upper limit at 0.68 or (1-alpha)/2 ?
            // decide to return always (1-alpha)/2 upper interval
            //if (n == 0) return ROOT::Math::gamma_quantile_c(alpha,n+1,1);
            return JSROOTMath.gamma_quantile_c( alpha/2, n+1, 1) - c;
         };
         obj['getBinLowEdge'] = function(bin) {
            // Return low edge of bin
            if (this['fXaxis']['fXbins']['fN'] && bin > 0 && bin <= this['fXaxis']['fNbins'])
               return this['fXaxis']['fXbins']['fArray'][bin-1];
            var binwidth = (this['fXaxis']['fXmax'] - this['fXaxis']['fXmin']) / this['fXaxis']['fNbins'];
            return this['fXaxis']['fXmin'] + (bin-1) * binwidth;
         };
         obj['getBinUpEdge'] = function(bin) {
            // Return up edge of bin
            var binwidth;
            if (!this['fXaxis']['fXbins']['fN'] || bin < 1 || bin > this['fXaxis']['fNbins']) {
               binwidth = (this['fXaxis']['fXmax'] - this['fXaxis']['fXmin']) / this['fXaxis']['fNbins'];
               return this['fXaxis']['fXmin'] + bin * binwidth;
            } else {
               binwidth = this['fArray'][bin] - this['fArray'][bin-1];
               return this['fArray'][bin-1] + binwidth;
            }
         };
         obj['getBinWidth'] = function(bin) {
            // Return bin width
            if (this['fXaxis']['fNbins'] <= 0) return 0;
            if (this['fXaxis']['fXbins']['fN'] <= 0)
               return (this['fXaxis']['fXmax'] - this['fXaxis']['fXmin']) / this['fXaxis']['fNbins'];
            if (bin > this['fXaxis']['fNbins']) bin = this['fXaxis']['fNbins'];
            if (bin < 1) bin = 1;
            return this['fArray'][bin] - this['fArray'][bin-1];
         };
         obj['add'] = function(h1, c1) {
            // Performs the operation: this = this + c1*h1
            // if errors are defined (see TH1::Sumw2), errors are also recalculated.
            // Note that if h1 has Sumw2 set, Sumw2 is automatically called for this
            // if not already set.
            if (!h1 || typeof(h1) == 'undefined') {
               alert("Add : Attempt to add a non-existing histogram");
               return false;
            }
            if (!c1 || typeof(c1) == 'undefined') c1 = 1;
            var nbinsx = this['fXaxis']['fNbins'],
                nbinsy = this['fYaxis']['fNbins'],
                nbinsz = this['fZaxis']['fNbins'];

            if (this['fDimension'] < 2) nbinsy = -1;
            if (this['fDimension'] < 3) nbinsz = -1;

            // Create Sumw2 if h1 has Sumw2 set
            if (this['fSumw2']['fN'] == 0 && h1['fSumw2']['fN'] != 0) this.sumw2();

            // - Add statistics
            if (this['fEntries'] == NaN) this['fEntries'] = 0;
            var entries = Math.abs( this['fEntries'] + c1 * h1['fEntries'] );

            // statistics can be preserved only in case of positive coefficients
            // otherwise with negative c1 (histogram subtraction) one risks to get negative variances
            var resetStats = (c1 < 0);
            var s1, s2;
            if (!resetStats) {
               // need to initialize to zero s1 and s2 since
               // GetStats fills only used elements depending on dimension and type
               s1 = this.getStats();
               s2 = h1.getStats();
            }
            this['fMinimum'] = -1111;
            this['fMaximum'] = -1111;

            // - Loop on bins (including underflows/overflows)
            var bin, binx, biny, binz;
            var cu, factor = 1;
            if (Math.abs(h1['fNormFactor']) > 2e-308) factor = h1['fNormFactor'] / h1.getSumOfWeights();
            for (binz=0;binz<=nbinsz+1;binz++) {
               for (biny=0;biny<=nbinsy+1;biny++) {
                  for (binx=0;binx<=nbinsx+1;binx++) {
                     bin = binx +(nbinsx+2)*(biny + (nbinsy+2)*binz);
                     //special case where histograms have the kIsAverage bit set
                     if (this.TestBit(TH1StatusBits.kIsAverage) && h1.TestBit(TH1StatusBits.kIsAverage)) {
                        var y1 = h1.getBinContent(bin),
                            y2 = this.getBinContent(bin),
                            e1 = h1.getBinError(bin),
                            e2 = this.getBinError(bin),
                            w1 = 1, w2 = 1;
                        // consider all special cases  when bin errors are zero
                        // see http://root.cern.ch/phpBB3//viewtopic.php?f=3&t=13299
                        if (e1 > 0)
                           w1 = 1.0 / (e1 * e1);
                        else if (h1['fSumw2']['fN']) {
                           w1 = 1.E200; // use an arbitrary huge value
                           if (y1 == 0) {
                              // use an estimated error from the global histogram scale
                              var sf = (s2[0] != 0) ? s2[1] / s2[0] : 1;
                              w1 = 1.0 / (sf * sf);
                           }
                        }
                        if (e2 > 0)
                           w2 = 1.0 / (e2 * e2);
                        else if (this['fSumw2']['fN']) {
                           w2 = 1.E200; // use an arbitrary huge value
                           if (y2 == 0) {
                              // use an estimated error from the global histogram scale
                              var sf = (s1[0] != 0) ? s1[1] / s1[0] : 1;
                              w2 = 1.0 / (sf * sf);
                           }
                        }
                        var y = (w1 * y1 + w2 * y2) / (w1 + w2);
                        this.setBinContent(bin, y);
                        if (this['fSumw2']['fN']) {
                           var err2 =  1.0 / (w1 + w2);
                           if (err2 < 1.E-200) err2 = 0;  // to remove arbitrary value when e1=0 AND e2=0
                           this['fSumw2']['fArray'][bin] = err2;
                        }
                     }
                     //normal case of addition between histograms
                     else {
                        cu  = c1 * factor * h1.getBinContent(bin);
                        this['fArray'][bin] += cu;
                        if (this['fSumw2']['fN']) {
                           var e1 = factor * h1.getBinError(bin);
                           this['fSumw2']['fArray'][bin] += c1 * c1 * e1 * e1;
                        }
                     }
                  }
               }
            }
            // update statistics (do here to avoid changes by SetBinContent)
            if (resetStats)  {
               // statistics need to be reset in case coefficient are negative
               this.resetStats();
            }
            else {
               for (var i=0;i<kNstat;i++) {
                  if (i == 1) s1[i] += c1 * c1 * s2[i];
                  else        s1[i] += c1 * s2[i];
               }
               //this.putStats(s1);
               this['fTsumw']   = s1[0];
               this['fTsumw2']  = s1[1];
               this['fTsumwx']  = s1[2];
               this['fTsumwx2'] = s1[3];
               this['fEntries'] = entries;
            }
            return true;
         };
         obj['getBin'] = function(binx, biny, binz) {
            //   -*-*-*-*Return Global bin number corresponding to binx,y,z*-*-*-*-*-*-*
            var nx, ny, nz;
            if (this['fDimension'] < 2) {
               nx  = this['fXaxis']['fNbins']+2;
               if (binx < 0)   binx = 0;
               if (binx >= nx) binx = nx-1;
               return binx;
            }
            if (this['fDimension'] < 3) {
               nx  = this['fXaxis']['fNbins']+2;
               if (binx < 0)   binx = 0;
               if (binx >= nx) binx = nx-1;
               ny  = this['fYaxis']['fNbins']+2;
               if (biny < 0)   biny = 0;
               if (biny >= ny) biny = ny-1;
               return  binx + nx*biny;
            }
            if (this['fDimension'] < 4) {
               nx  = this['fXaxis']['fNbins']+2;
               if (binx < 0)   binx = 0;
               if (binx >= nx) binx = nx-1;
               ny  = this['fYaxis']['fNbins']+2;
               if (biny < 0)   biny = 0;
               if (biny >= ny) biny = ny-1;
               nz  = this['fZaxis']['fNbins']+2;
               if (binz < 0)   binz = 0;
               if (binz >= nz) binz = nz-1;
               return  binx + nx*(biny +ny*binz);
            }
            return -1;
         };
         obj['getBinXYZ'] = function(binglobal) {
            // return binx, biny, binz corresponding to the global bin number globalbin
            // see TH1::GetBin function above
            var binx, biny, binz;
            var nx  = this['fXaxis']['fNbins']+2;
            var ny  = this['fYaxis']['fNbins']+2;
            if (this['fDimension'] < 2) {
               binx = binglobal%nx;
               biny = -1;
               binz = -1;
            }
            if (this['fDimension'] < 3) {
               binx = binglobal%nx;
               biny = ((binglobal-binx)/nx)%ny;
               binz = -1;
            }
            if (this['fDimension'] < 4) {
               binx = binglobal%nx;
               biny = ((binglobal-binx)/nx)%ny;
               binz = ((binglobal-binx)/nx -biny)/ny;
            }
            return { binsx: binx, biny: biny, binz: binz };
         };
         obj['getMaximum'] = function(maxval) {
            //  Return maximum value smaller than maxval of bins in the range,
            //  unless the value has been overridden by TH1::SetMaximum,
            //  in which case it returns that value. (This happens, for example,
            //  when the histogram is drawn and the y or z axis limits are changed
            //
            //  To get the maximum value of bins in the histogram regardless of
            //  whether the value has been overridden, use
            //      h->GetBinContent(h->GetMaximumBin())

            if (this['fMaximum'] != -1111) return this['fMaximum'];
            if (!maxval || typeof(maxval) == 'undefined') maxval = Number.MAX_VALUE;
            var bin, binx, biny, binz;
            var xfirst  = this['fXaxis'].getFirst();
                xlast   = this['fXaxis'].getLast(),
                yfirst  = this['fYaxis'].getFirst(),
                ylast   = this['fYaxis'].getLast(),
                zfirst  = this['fZaxis'].getFirst(),
                zlast   = this['fZaxis'].getLast();
            var maximum = -Number.MAX_VALUE, val;
            for (binz=zfirst;binz<=zlast;binz++) {
               for (biny=yfirst;biny<=ylast;biny++) {
                  for (binx=xfirst;binx<=xlast;binx++) {
                     bin = this.getBin(binx,biny,binz);
                     val = this.getBinContent(bin);
                     if (val > maximum && val < maxval) maximum = val;
                  }
               }
            }
            return maximum;
         };
         obj['getMinimum'] = function(minval) {
            //  Return minimum value smaller than maxval of bins in the range,
            //  unless the value has been overridden by TH1::SetMinimum,
            //  in which case it returns that value. (This happens, for example,
            //  when the histogram is drawn and the y or z axis limits are changed
            if (this['fMinimum'] != -1111) return this['fMinimum'];
            if (!minval || typeof(minval) == 'undefined') minval = -Number.MAX_VALUE;
            var bin, binx, biny, binz;
            var xfirst  = this['fXaxis'].getFirst();
                xlast   = this['fXaxis'].getLast(),
                yfirst  = this['fYaxis'].getFirst(),
                ylast   = this['fYaxis'].getLast(),
                zfirst  = this['fZaxis'].getFirst(),
                zlast   = this['fZaxis'].getLast();
            var minimum = Number.MAX_VALUE, val;
            for (binz=zfirst;binz<=zlast;binz++) {
               for (biny=yfirst;biny<=ylast;biny++) {
                  for (binx=xfirst;binx<=xlast;binx++) {
                     bin = this.getBin(binx,biny,binz);
                     val = this.getBinContent(bin);
                     if (val < minimum && val > minval) minimum = val;
                  }
               }
            }
            return minimum;
         };
         obj['getSumOfWeights'] = function() {
            //   -*-*-*-*-*-*Return the sum of weights excluding under/overflows*-*-*-*-*
            var bin, binx, biny, binz, sum = 0;
            for (binz=1; binz<=this['fZaxis']['fXbins']['fN']; binz++) {
               for (biny=1; biny<=this['fYaxis']['fXbins']['fN']; biny++) {
                  for (binx=1; binx<=this['fXaxis']['fXbins']['fN']; binx++) {
                     bin = this.getBin(binx,biny,binz);
                     sum += this.getBinContent(bin);
                  }
               }
            }
            return sum;
         };
         obj['labelsInflate'] = function(ax) {
            // Double the number of bins for axis.
            // Refill histogram

            var axis = null;
            var achoice = ax[0].toUpperCase();
            if (achoice == 'X') axis = this['fXaxis'];
            if (achoice == 'Y') axis = this['fYaxis'];
            if (achoice == 'Z') axis = this['fZaxis'];
            if (axis == null) return;

            var hold = JSROOTCore.clone(this);

            var timedisp = axis['fTimeDisplay'];
            var nbxold = this['fXaxis']['fNbins'];
            var nbyold = this['fYaxis']['fNbins'];
            var nbzold = this['fZaxis']['fNbins'];
            var nbins  = axis['fNbins'];
            var xmin = axis['fXmin'];
            var xmax = axis['fXmax'];
            xmax = xmin + 2 * (xmax - xmin);
            axis['fFirst'] = 1;
            axis['fLast'] = axis['fNbins'];
            this['fBits'] &= ~(EAxisBits.kAxisRange & 0x00ffffff); // SetBit(kAxisRange, 0);
            // double the bins and recompute ncells
            axis['fNbins'] = 2*nbins;
            axis['fXmin']  = xmin;
            axis['fXmax']  = xmax;
            this['fNcells'] = -1;
            this['fArray'].length = -1;
            var errors = this['fSumw2']['fN'];
            if (errors) ['fSumw2'].length = this['fNcells'];
            axis['fTimeDisplay'] = timedisp;

            Reset("ICE");  // reset content and error
            this['fSumw2'].splice(0, this['fSumw2'].length);
            this['fMinimum'] = -1111;
            this['fMaximum'] = -1111;

            //now loop on all bins and refill
            var oldEntries = this['fEntries'];
            var bin, ibin, bins;
            for (ibin = 0; ibin < this['fNcells']; ibin++) {
               bins = this.getBinXYZ(ibin);
               bin = hold.getBin(bins['binx'],bins['biny'],bins['binz']);
               // NOTE that overflow in hold will be not considered
               if (bins['binx'] > nbxold  || bins['biny'] > nbyold || bins['binz'] > nbzold) bin = -1;
               if (bin > 0)  {
                  var cu = hold.getBinContent(bin);
                  this['fArray'][bin] += cu;
                  if (errors) this['fSumw2']['fArray'][ibin] += hold['fSumw2']['fArray'][bin];
               }
            }
            this['fEntries'] = oldEntries;
            delete hold;
         };
         obj['resetStats'] = function() {
            // Reset the statistics including the number of entries
            // and replace with values calculates from bin content
            // The number of entries is set to the total bin content or (in case of weighted histogram)
            // to number of effective entries
            this['fTsumw'] = 0;
            this['fEntries'] = 1; // to force re-calculation of the statistics in TH1::GetStats
            var stats = this.getStats();
            this['fTsumw']   = stats[0];
            this['fTsumw2']  = stats[1];
            this['fTsumwx']  = stats[2];
            this['fTsumwx2'] = stats[3];
            this['fEntries'] = Math.abs(this['fTsumw']);
            // use effective entries for weighted histograms:  (sum_w) ^2 / sum_w2
            if (this['fSumw2']['fN'] > 0 && this['fTsumw'] > 0 && stats[1] > 0 )
               this['fEntries'] = stats[0] * stats[0] / stats[1];
         }
         obj['setBinContent'] = function(bin, content) {
            // Set bin content
            // see convention for numbering bins in TH1::GetBin
            // In case the bin number is greater than the number of bins and
            // the timedisplay option is set or the kCanRebin bit is set,
            // the number of bins is automatically doubled to accommodate the new bin

            this['fEntries']++;
            this['fTsumw'] = 0;
            if (bin < 0) return;
            if (bin >= this['fNcells']-1) {
               if (this['fXaxis']['fTimeDisplay'] || this.TestBit(TH1StatusBits.kCanRebin) ) {
                  while (bin >= this['fNcells']-1) this.labelsInflate();
               } else {
                  if (bin == this['fNcells']-1) this['fArray'][bin] = content;
                  return;
               }
            }
            this['fArray'][bin] = content;
         };
         obj['sumw2'] = function() {
            // Create structure to store sum of squares of weights*-*-*-*-*-*-*-*
            //
            //     if histogram is already filled, the sum of squares of weights
            //     is filled with the existing bin contents
            //
            //     The error per bin will be computed as sqrt(sum of squares of weight)
            //     for each bin.
            //
            //  This function is automatically called when the histogram is created
            //  if the static function TH1::SetDefaultSumw2 has been called before.

            if (this['fSumw2']['fN'] == this['fNcells']) {
               return;
            }
            this['fSumw2'].length = this['fNcells'];
            if ( this['fEntries'] > 0 ) {
               for (var bin=0; bin<this['fNcells']; bin++) {
                  this['fSumw2']['fArray'][bin] = Math.abs(this.getBinContent(bin));
               }
            }
         };
      }
      if (obj['_typename'].indexOf("JSROOTIO.TH1") == 0) {
         obj['getBinContent'] = function(bin) {
            if (bin < 0) bin = 0;
            if (bin >= this['fNcells']) bin = this['fNcells']-1;
            return this['fArray'][bin];
         };
         obj['getStats'] = function() {
            // fill the array stats from the contents of this histogram
            // The array stats must be correctly dimensioned in the calling program.
            // stats[0] = sumw
            // stats[1] = sumw2
            // stats[2] = sumwx
            // stats[3] = sumwx2
            // Loop on bins (possibly including underflows/overflows)
            var bin, binx, w, err, x, stats = new Array(0,0,0,0,0);
            // case of labels with rebin of axis set
            // statistics in x does not make any sense - set to zero
            if (this['fXaxis']['fLabels'] && this.TestBit(TH1StatusBits.kCanRebin) ) {
               stats[0] = this['fTsumw'];
               stats[1] = this['fTsumw2'];
               stats[2] = 0;
               stats[3] = 0;
            }
            else if ((this['fTsumw'] == 0 && this['fEntries'] > 0) ||
                     this['fXaxis'].TestBit(EAxisBits.kAxisRange)) {
               for (bin=0;bin<4;bin++) stats[bin] = 0;

               var firstBinX = this['fXaxis'].getFirst();
               var lastBinX  = this['fXaxis'].getLast();
               for (binx = firstBinX; binx <= lastBinX; binx++) {
                  x   = this['fXaxis'].getBinCenter(binx);
                  w   = this.getBinContent(binx);
                  err = Math.abs(this.getBinError(binx));
                  stats[0] += w;
                  stats[1] += err*err;
                  stats[2] += w*x;
                  stats[3] += w*x*x;
               }
            } else {
               stats[0] = this['fTsumw'];
               stats[1] = this['fTsumw2'];
               stats[2] = this['fTsumwx'];
               stats[3] = this['fTsumwx2'];
            }
            return stats;
         };
      }
      if (obj['_typename'].indexOf("JSROOTIO.TH2") == 0) {
         obj['getBin'] = function(x, y) {
            var nx = this['fXaxis']['fNbins']+2;
            return (x + nx * y);
         };
         obj['getBinContent'] = function(x, y) {
            return this['fArray'][this.getBin(x, y)];
         };
         obj['getStats'] = function() {
            var bin, binx, biny, stats = new Array(0,0,0,0,0,0,0,0,0,0,0,0,0);
            if ((this['fTsumw'] == 0 && this['fEntries'] > 0) || this['fXaxis'].TestBit(EAxisBits.kAxisRange) || this['fYaxis'].TestBit(EAxisBits.kAxisRange)) {
               var firstBinX = this['fXaxis'].getFirst();
               var lastBinX  = this['fXaxis'].getLast();
               var firstBinY = this['fYaxis'].getFirst();
               var lastBinY  = this['fYaxis'].getLast();
               // include underflow/overflow if TH1::StatOverflows(kTRUE) in case no range is set on the axis
               if (this['fgStatOverflows']) {
                 if ( !this['fXaxis'].TestBit(EAxisBits.kAxisRange) ) {
                     if (firstBinX == 1) firstBinX = 0;
                     if (lastBinX ==  this['fXaxis']['fNbins'] ) lastBinX += 1;
                  }
                  if ( !this['fYaxis'].TestBit(EAxisBits.kAxisRange) ) {
                     if (firstBinY == 1) firstBinY = 0;
                     if (lastBinY ==  this['fYaxis']['fNbins'] ) lastBinY += 1;
                  }
               }
               for (biny = firstBinY; biny <= lastBinY; biny++) {
                  y = this['fYaxis'].getBinCenter(biny);
                  for (binx = firstBinX; binx <= lastBinX; binx++) {
                     bin = this.getBin(binx,biny);
                     x   = this['fXaxis'].getBinCenter(binx);
                     w   = this.GetBinContent(bin);
                     err = Math.abs(this.getBinError(bin));
                     stats[0] += w;
                     stats[1] += err*err;
                     stats[2] += w*x;
                     stats[3] += w*x*x;
                     stats[4] += w*y;
                     stats[5] += w*y*y;
                     stats[6] += w*x*y;
                  }
               }
            } else {
               stats[0] = this['fTsumw'];
               stats[1] = this['fTsumw2'];
               stats[2] = this['fTsumwx'];
               stats[3] = this['fTsumwx2'];
               stats[4] = this['fTsumwy'];
               stats[5] = this['fTsumwy2'];
               stats[6] = this['fTsumwxy'];
            }
            return stats;
         };
      }
      if (obj['_typename'].indexOf("JSROOTIO.TH3") == 0) {
         obj['getBin'] = function(x, y, z) {
            var nx = this['fXaxis']['fNbins']+2;
            if (x < 0) x = 0;
            if (x >= nx) x = nx-1;
            var ny = this['fYaxis']['fNbins']+2;
            if (y < 0) y = 0;
            if (y >= ny) y = ny-1;
            return (x + nx * (y + ny * z));
         };
         obj['getBinContent'] = function(x, y, z) {
            return this['fArray'][this.getBin(x, y, z)];
         };
         obj['getStats'] = function() {
            var bin, binx, biny, binz, stats = new Array(0,0,0,0,0,0,0,0,0,0,0,0,0);
            if ((obj['fTsumw'] == 0 && obj['fEntries'] > 0) || obj['fXaxis'].TestBit(EAxisBits.kAxisRange) || obj['fYaxis'].TestBit(EAxisBits.kAxisRange) || obj['fZaxis'].TestBit(EAxisBits.kAxisRange)) {
               var firstBinX = obj['fXaxis'].getFirst();
               var lastBinX  = obj['fXaxis'].getLast();
               var firstBinY = obj['fYaxis'].getFirst();
               var lastBinY  = obj['fYaxis'].getLast();
               var firstBinZ = obj['fZaxis'].getFirst();
               var lastBinZ  = obj['fZaxis'].getLast();
               // include underflow/overflow if TH1::StatOverflows(kTRUE) in case no range is set on the axis
               if (obj['fgStatOverflows']) {
                 if ( !obj['fXaxis'].TestBit(EAxisBits.kAxisRange) ) {
                     if (firstBinX == 1) firstBinX = 0;
                     if (lastBinX ==  obj['fXaxis']['fNbins'] ) lastBinX += 1;
                  }
                  if ( !obj['fYaxis'].TestBit(EAxisBits.kAxisRange) ) {
                     if (firstBinY == 1) firstBinY = 0;
                     if (lastBinY ==  obj['fYaxis']['fNbins'] ) lastBinY += 1;
                  }
                  if ( !obj['fZaxis'].TestBit(EAxisBits.kAxisRange) ) {
                     if (firstBinZ == 1) firstBinZ = 0;
                     if (lastBinZ ==  obj['fZaxis']['fNbins'] ) lastBinZ += 1;
                  }
               }
               for (binz = firstBinZ; binz <= lastBinZ; binz++) {
                  z = obj['fZaxis'].getBinCenter(binz);
                  for (biny = firstBinY; biny <= lastBinY; biny++) {
                     y = obj['fYaxis'].getBinCenter(biny);
                     for (binx = firstBinX; binx <= lastBinX; binx++) {
                        bin = obj.getBin(binx,biny,binz);
                        x   = obj['fXaxis'].getBinCenter(binx);
                        w   = obj.GetBinContent(bin);
                        err = Math.abs(obj.getBinError(bin));
                        stats[0] += w;
                        stats[1] += err*err;
                        stats[2] += w*x;
                        stats[3] += w*x*x;
                        stats[4] += w*y;
                        stats[5] += w*y*y;
                        stats[6] += w*x*y;
                        stats[7] += w*z;
                        stats[8] += w*z*z;
                        stats[9] += w*x*z;
                        stats[10] += w*y*z;
                     }
                  }
               }
            } else {
               stats[0] = obj['fTsumw'];
               stats[1] = obj['fTsumw2'];
               stats[2] = obj['fTsumwx'];
               stats[3] = obj['fTsumwx2'];
               stats[4] = obj['fTsumwy'];
               stats[5] = obj['fTsumwy2'];
               stats[6] = obj['fTsumwxy'];
               stats[7] = obj['fTsumwz'];
               stats[8] = obj['fTsumwz2'];
               stats[9] = obj['fTsumwxz'];
               stats[10] =obj['fTsumwyz'];
            }
            return stats;
         };
      }
      if (obj['_typename'].indexOf("JSROOTIO.THStack") == 0) {
         obj['buildStack'] = function() {
            //  build sum of all histograms
            //  Build a separate list fStack containing the running sum of all histograms
            if ('fStack' in this) return;
            if (!'fHists' in this) return;
            var nhists = this['fHists'].arr.length;
            if (nhists <= 0) return;
            this['fStack'] = JSROOTCore.CreateTList();
            var h = JSROOTCore.clone(this['fHists'].arr[0]);
            this['fStack'].arr.push(h);
            for (var i=1;i<nhists;i++) {
               h = JSROOTCore.clone(this['fHists'].arr[i]);
               h.add(this['fStack'].arr[i-1]);
               this['fStack'].arr.splice(i, 1, h);
            }
         };
         obj['getMaximum'] = function(option) {
            // returns the maximum of all added histograms
            // returns the maximum of all histograms if option "nostack".
            var opt = option.toLowerCase();
            var lerr = false;
            if (opt.indexOf("e") != -1) lerr = true;
            var them = 0, themax = -1e300, c1, e1;
            if (!'fHists' in this) return 0;
            var nhists = this['fHists'].arr.length;
            var first, last;
            if (opt.indexOf("nostack") == -1) {
               this.buildStack();
               var h = this['fStack'].arr[nhists-1];
               themax = h.getMaximum();
            } else {
               for (var i=0;i<nhists;i++) {
                  h = this['fHists'].arr[i];
                  them = h.getMaximum();
                  if (them > themax) themax = them;
               }
            }
            if (lerr) {
               for (var i=0;i<nhists;i++) {
                  h = this['fHists'].arr[i];
                  first = h['fXaxis'].getFirst();
                  last  = h['fXaxis'].getLast();
                  for (var j=first; j<=last;j++) {
                     e1     = h.getBinError(j);
                     c1     = h.getBinContent(j);
                     themax = Math.max(themax, c1+e1);
                  }
               }
            }
            return themax;
         };
         obj['getMinimum'] = function(option, pad) {
            //  returns the minimum of all added histograms
            //  returns the minimum of all histograms if option "nostack".
            var opt = option.toLowerCase();
            var lerr = false;
            if (opt.indexOf("e") == -1) lerr = true;
            var them = 0, themin = 1e300, c1, e1;
            if (!'fHists' in this) return 0;
            var nhists = this['fHists'].arr.length;
            var first, last;
            if (opt.indexOf("nostack") == -1) {
               this.buildStack();
               var h = this['fStack'].arr[nhists-1];
               themin = h.getMinimum();
            } else {
               for (var i=0;i<nhists;i++) {
                  h = this['fHists'].arr[i];
                  them = h.getMinimum();
                  if (them <= 0 && pad && pad['fLogy']) them = h.getMinimum(0);
                  if (them < themin) themin = them;
               }
            }
            if (lerr) {
               for (var i=0;i<nhists;i++) {
                  h = this['fHists'].arr[i];
                  first = h['fXaxis'].getFirst();
                  last  = h['fXaxis'].getLast();
                  for (var j=first;j<=last;j++) {
                      e1     = h.getBinError(j);
                      c1     = h.getBinContent(j);
                      themin = Math.min(themin, c1 - e1);
                  }
               }
            }
            return themin;
         };
      }
      if ((obj['_typename'].indexOf("JSROOTIO.TH2") == 0) ||
          (obj['_typename'].indexOf("JSROOTIO.TH3") == 0) ||
          (obj['_typename'].indexOf("JSROOTIO.TProfile") == 0)) {
         obj['getMean'] = function(axis) {
            if (axis < 1 || (axis > 3 && axis < 11) || axis > 13) return 0;
            var stats = this.getStats();
            if (stats[0] == 0) return 0;
            var ax = new Array(2,4,7);
            return stats[ax[axis-1]]/stats[0];
         };
         obj['getRMS'] = function(axis) {
            if (axis < 1 || (axis > 3 && axis < 11) || axis > 13) return 0;
            var stats = this.getStats();
            if (stats[0] == 0) return 0;
            var ax = new Array(2,4,7);
            var axm = ax[axis%10 - 1];
            var x = stats[axm]/stats[0];
            var rms2 = Math.abs(stats[axm+1]/stats[0] -x*x);
            return Math.sqrt(rms2);
         };
      }
      if (obj['_typename'].indexOf("JSROOTIO.TProfile") == 0) {
         obj['getBinContent'] = function(bin) {
            if (bin < 0 || bin >= this['fNcells']) return 0;
            if (this['fBinEntries'][bin] < 1e-300) return 0;
            if (!this['fArray']) return 0;
            return this['fArray'][bin]/this['fBinEntries'][bin];
         };
         obj['getBinEffectiveEntries'] = function(bin) {
            if (bin < 0 || bin >= this['fNcells']) return 0;
            var sumOfWeights = this['fBinEntries'][bin];
            if ( this['fBinSumw2'].length == 0 || this['fBinSumw2'].length != this['fNcells']) {
               // this can happen  when reading an old file
               return sumOfWeights;
            }
            var sumOfWeightsSquare = this['fSumw2'][bin];
            return ( sumOfWeightsSquare > 0 ? sumOfWeights * sumOfWeights / sumOfWeightsSquare : 0 );
         };
         obj['getStats'] = function() {
            var bin, binx, stats = new Array(0,0,0,0,0,0,0,0,0,0,0,0,0);
            if (this['fTsumw'] < 1e-300 || this['fXaxis'].TestBit(EAxisBits.kAxisRange)) {
               var firstBinX = this['fXaxis'].getFirst();
               var lastBinX  = this['fXaxis'].getLast();
               for (binx = this['firstBinX']; binx <= lastBinX; binx++) {
                  var w   = onj['fBinEntries'][binx];
                  var w2  = (this['fN'] ? this['fBinSumw2'][binx] : w);
                  var x   = fXaxis.GetBinCenter(binx);
                  stats[0] += w;
                  stats[1] += w2;
                  stats[2] += w*x;
                  stats[3] += w*x*x;
                  stats[4] += this['fArray'][binx];
                  stats[5] += this['fSumw2'][binx];
               }
            } else {
               if (this['fTsumwy'] < 1e-300 && this['fTsumwy2'] < 1e-300) {
                  //this case may happen when processing TProfiles with version <=3
                  for (binx=this['fXaxis'].getFirst();binx<=this['fXaxis'].getLast();binx++) {
                     this['fTsumwy'] += this['fArray'][binx];
                     this['fTsumwy2'] += this['fSumw2'][binx];
                  }
               }
               stats[0] = this['fTsumw'];
               stats[1] = this['fTsumw2'];
               stats[2] = this['fTsumwx'];
               stats[3] = this['fTsumwx2'];
               stats[4] = this['fTsumwy'];
               stats[5] = this['fTsumwy2'];
            }
            return stats;
         };
         obj['getBinError'] = function(bin) {
            if (bin < 0 || bin >= this['fNcells']) return 0;
            var cont = this['fArray'][bin];               // sum of bin w *y
            var sum  = this['fBinEntries'][bin];          // sum of bin weights
            var err2 = this['fSumw2'][bin];               // sum of bin w * y^2
            var neff = this.getBinEffectiveEntries(bin);  // (sum of w)^2 / (sum of w^2)
            if (sum < 1e-300) return 0;                  // for empty bins
            // case the values y are gaussian distributed y +/- sigma and w = 1/sigma^2
            if (this['fErrorMode'] == EErrorType.kERRORSPREADG) {
               return (1.0/Math.sqrt(sum));
            }
            // compute variance in y (eprim2) and standard deviation in y (eprim)
            var contsum = cont/sum;
            var eprim2  = Math.abs(err2/sum - contsum*contsum);
            var eprim   = Math.sqrt(eprim2);
            if (this['fErrorMode'] == EErrorType.kERRORSPREADI) {
               if (eprim != 0) return eprim/Math.sqrt(neff);
               // in case content y is an integer (so each my has an error +/- 1/sqrt(12)
               // when the std(y) is zero
               return (1.0/Math.sqrt(12*neff));
            }
            // if approximate compute the sums (of w, wy and wy2) using all the bins
            //  when the variance in y is zero
            var testing = 1;
            if (err2 != 0 && neff < 5) testing = eprim2*sum/err2;
            if (this['fgApproximate'] && (testing < 1.e-4 || eprim2 < 1e-6)) { //3.04
               var stats = this.getStats();
               var ssum = stats[0];
               // for 1D profile
               var idx = 4;  // index in the stats array for 1D
               var scont = stats[idx];
               var serr2 = stats[idx+1];
               // compute mean and variance in y
               var scontsum = scont/ssum; // global mean
               var seprim2  = Math.abs(serr2/ssum - scontsum*scontsum); // global variance
               eprim = 2*Math.sqrt(seprim2); // global std (why factor of 2 ??)
               sum = ssum;
            }
            sum = Math.abs(sum);
            // case option "S" return standard deviation in y
            if (this['fErrorMode'] == EErrorType.kERRORSPREAD) return eprim;
            // default case : fErrorMode = kERRORMEAN
            // return standard error on the mean of y
            return (eprim/Math.sqrt(neff));
         };
      }
   };

})();

/// JSROOTCore.js ends

// JSROOTMath.js
//
// math methods for Javascript ROOT IO.
//

(function(){

   if (typeof JSROOTMath == "object"){
      var e1 = new Error("JSROOTMath is already defined");
      e1.source = "JSROOTMath.js";
      throw e1;
   }

   JSROOTMath = {};

   JSROOTMath.version = "1.0 2012/08/08";

   /* the machine roundoff error */
   var kMACHEP = 1.11022302462515654042363166809e-16;
   var s2pi = 2.50662827463100050242e0;

   var P0 = new Array(
      -5.99633501014107895267E1,
       9.80010754185999661536E1,
      -5.66762857469070293439E1,
       1.39312609387279679503E1,
      -1.23916583867381258016E0
   );

   var Q0 = new Array(
       1.95448858338141759834E0,
       4.67627912898881538453E0,
       8.63602421390890590575E1,
      -2.25462687854119370527E2,
       2.00260212380060660359E2,
      -8.20372256168333339912E1,
       1.59056225126211695515E1,
      -1.18331621121330003142E0
   );

   var P1 = new Array(
       4.05544892305962419923E0,
       3.15251094599893866154E1,
       5.71628192246421288162E1,
       4.40805073893200834700E1,
       1.46849561928858024014E1,
       2.18663306850790267539E0,
      -1.40256079171354495875E-1,
      -3.50424626827848203418E-2,
      -8.57456785154685413611E-4
   );

   var Q1 = new Array(
       1.57799883256466749731E1,
       4.53907635128879210584E1,
       4.13172038254672030440E1,
       1.50425385692907503408E1,
       2.50464946208309415979E0,
      -1.42182922854787788574E-1,
      -3.80806407691578277194E-2,
      -9.33259480895457427372E-4
   );

   var P2 = new Array(
       3.23774891776946035970E0,
       6.91522889068984211695E0,
       3.93881025292474443415E0,
       1.33303460815807542389E0,
       2.01485389549179081538E-1,
       1.23716634817820021358E-2,
       3.01581553508235416007E-4,
       2.65806974686737550832E-6,
       6.23974539184983293730E-9
   );

   var Q2 = new Array(
       6.02427039364742014255E0,
       3.67983563856160859403E0,
       1.37702099489081330271E0,
       2.16236993594496635890E-1,
       1.34204006088543189037E-2,
       3.28014464682127739104E-4,
       2.89247864745380683936E-6,
       6.79019408009981274425E-9
   );

   var p1 = new Array(0.4259894875,-0.1249762550, 0.03984243700, -0.006298287635,   0.001511162253);
   var q1 = new Array(1.0         ,-0.3388260629, 0.09594393323, -0.01608042283,    0.003778942063);
   var p2 = new Array(0.1788541609, 0.1173957403, 0.01488850518, -0.001394989411,   0.0001283617211);
   var q2 = new Array(1.0         , 0.7428795082, 0.3153932961,   0.06694219548,    0.008790609714);
   var p3 = new Array(0.1788544503, 0.09359161662,0.006325387654, 0.00006611667319,-0.000002031049101);
   var q3 = new Array(1.0         , 0.6097809921, 0.2560616665,   0.04746722384,    0.006957301675);
   var p4 = new Array(0.9874054407, 118.6723273,  849.2794360,   -743.7792444,      427.0262186);
   var q4 = new Array(1.0         , 106.8615961,  337.6496214,    2016.712389,      1597.063511);
   var p5 = new Array(1.003675074,  167.5702434,  4789.711289,    21217.86767,     -22324.94910);
   var q5 = new Array(1.0         , 156.9424537,  3745.310488,    9834.698876,      66924.28357);
   var p6 = new Array(1.000827619,  664.9143136,  62972.92665,    475554.6998,     -5743609.109);
   var q6 = new Array(1.0         , 651.4101098,  56974.73333,    165917.4725,     -2815759.939);
   var a1 = new Array(0.04166666667,-0.01996527778, 0.02709538966);
   var a2 = new Array(-1.845568670,-4.284640743);

   JSROOTMath.lgam = function( x ) {
      var p, q, u, w, z;
      var i;

      var sgngam = 1;

      if (x >= Number.POSITIVE_INFINITY)
         return(Number.POSITIVE_INFINITY);

      if ( x < -34.0 ) {
         q = -x;
         w = this.lgam(q);
         p = Math.floor(q);
         if ( p==q )//_unur_FP_same(p,q)
            return (Number.POSITIVE_INFINITY);
         i = Math.round(p);
         if ( (i & 1) == 0 )
            sgngam = -1;
         else
            sgngam = 1;
         z = q - p;
         if ( z > 0.5 ) {
            p += 1.0;
            z = p - q;
         }
         z = q * Math.sin( Math.PI * z );
         if ( z < 1e-300 )
            return (Number.POSITIVE_INFINITY);
         z = Math.log(Math.PI) - Math.log( z ) - w;
         return( z );
      }
      if ( x < 13.0 ) {
         z = 1.0;
         p = 0.0;
         u = x;
         while ( u >= 3.0 ) {
            p -= 1.0;
            u = x + p;
            z *= u;
         }
         while ( u < 2.0 ) {
            if ( u < 1e-300 )
               return (Number.POSITIVE_INFINITY);
            z /= u;
            p += 1.0;
            u = x + p;
         }
         if ( z < 0.0 ) {
            sgngam = -1;
            z = -z;
         }
         else
            sgngam = 1;
         if ( u == 2.0 )
            return( Math.log(z) );
         p -= 2.0;
         x = x + p;
         p = x * this.Polynomialeval(x, B, 5 ) / this.Polynomial1eval( x, C, 6);
         return( Math.log(z) + p );
      }
      if ( x > kMAXLGM )
         return( sgngam * Number.POSITIVE_INFINITY );

      q = ( x - 0.5 ) * Math.log(x) - x + LS2PI;
      if ( x > 1.0e8 )
         return( q );

      p = 1.0/(x*x);
      if ( x >= 1000.0 )
         q += ((7.9365079365079365079365e-4 * p
               - 2.7777777777777777777778e-3) *p
               + 0.0833333333333333333333) / x;
      else
         q += this.Polynomialeval( p, A, 4 ) / x;
      return( q );
   };

   /*
    * calculates a value of a polynomial of the form:
    * a[0]x^N+a[1]x^(N-1) + ... + a[N]
   */
   JSROOTMath.Polynomialeval = function(x, a, N) {
      if (N==0) return a[0];
      else {
         var pom = a[0];
         for (var i=1; i <= N; i++)
            pom = pom *x + a[i];
         return pom;
      }
   };

   /*
    * calculates a value of a polynomial of the form:
    * x^N+a[0]x^(N-1) + ... + a[N-1]
   */
   JSROOTMath.Polynomial1eval = function(x, a, N) {
      if (N==0) return a[0];
      else {
         var pom = x + a[0];
         for (var i=1; i < N; i++)
            pom = pom *x + a[i];
         return pom;
      }
   };

   JSROOTMath.ndtri = function( y0 ) {
      var x, y, z, y2, x0, x1;
      var code;
      if ( y0 <= 0.0 )
         return( Number.NEGATIVE_INFINITY );
      if ( y0 >= 1.0 )
         return( Number.POSITIVE_INFINITY );
      code = 1;
      y = y0;
      if ( y > (1.0 - 0.13533528323661269189) ) {
         y = 1.0 - y;
         code = 0;
      }
      if ( y > 0.13533528323661269189 ) {
         y = y - 0.5;
         y2 = y * y;
         x = y + y * (y2 * this.Polynomialeval( y2, P0, 4)/ this.Polynomial1eval( y2, Q0, 8 ));
         x = x * s2pi;
         return(x);
      }
      x = Math.sqrt( -2.0 * Math.log(y) );
      x0 = x - Math.log(x)/x;
      z = 1.0/x;
      if ( x < 8.0 )
         x1 = z * this.Polynomialeval( z, P1, 8 )/ this.Polynomial1eval ( z, Q1, 8 );
      else
         x1 = z * this.Polynomialeval( z, P2, 8 )/ this.Polynomial1eval( z, Q2, 8 );
      x = x0 - x1;
      if ( code != 0 )
         x = -x;
      return( x );
   };

   JSROOTMath.igami = function(a, y0) {
      var x0, x1, x, yl, yh, y, d, lgm, dithresh;
      var i, dir;

      // check the domain
      if (a <= 0) {
         alert("igami : Wrong domain for parameter a (must be > 0)");
         return 0;
      }
      if (y0 <= 0) {
         return Number.POSITIVE_INFINITY;
      }
      if (y0 >= 1) {
         return 0;
      }
      /* bound the solution */
      var kMAXNUM = Number.MAX_VALUE;
      x0 = kMAXNUM;
      yl = 0;
      x1 = 0;
      yh = 1.0;
      dithresh = 5.0 * kMACHEP;

      /* approximation to inverse function */
      d = 1.0/(9.0*a);
      y = ( 1.0 - d - this.ndtri(y0) * Math.sqrt(d) );
      x = a * y * y * y;

      lgm = this.lgam(a);

      for( i=0; i<10; i++ ) {
         if ( x > x0 || x < x1 )
            break;
         y = igamc(a,x);
         if ( y < yl || y > yh )
            break;
         if ( y < y0 ) {
            x0 = x;
            yl = y;
         }
         else {
            x1 = x;
            yh = y;
         }
         /* compute the derivative of the function at this point */
         d = (a - 1.0) * Math.log(x) - x - lgm;
         if ( d < -kMAXLOG )
            break;
         d = -Math.exp(d);
         /* compute the step to the next approximation of x */
         d = (y - y0)/d;
         if ( Math.abs(d/x) < kMACHEP )
            return( x );
         x = x - d;
      }
      /* Resort to interval halving if Newton iteration did not converge. */
      d = 0.0625;
      if ( x0 == kMAXNUM ) {
         if ( x <= 0.0 )
            x = 1.0;
         while ( x0 == kMAXNUM ) {
            x = (1.0 + d) * x;
            y = igamc( a, x );
            if ( y < y0 ) {
               x0 = x;
               yl = y;
               break;
            }
            d = d + d;
         }
      }
      d = 0.5;
      dir = 0;

      for( i=0; i<400; i++ ) {
         x = x1  +  d * (x0 - x1);
         y = igamc( a, x );
         lgm = (x0 - x1)/(x1 + x0);
         if ( Math.abs(lgm) < dithresh )
            break;
         lgm = (y - y0)/y0;
         if ( Math.abs(lgm) < dithresh )
            break;
         if ( x <= 0.0 )
            break;
         if ( y >= y0 ) {
            x1 = x;
            yh = y;
            if ( dir < 0 ) {
               dir = 0;
               d = 0.5;
            }
            else if ( dir > 1 )
               d = 0.5 * d + 0.5;
            else
               d = (y0 - yl)/(yh - yl);
            dir += 1;
         }
         else {
            x0 = x;
            yl = y;
            if ( dir > 0 ) {
               dir = 0;
               d = 0.5;
            }
            else if ( dir < -1 )
               d = 0.5 * d;
            else
               d = (y0 - yl)/(yh - yl);
            dir -= 1;
         }
      }
      return( x );
   };

   JSROOTMath.gamma_quantile_c = function(z, alpha, theta) {
      return theta * this.igami( alpha, z);
   };

   JSROOTMath.gamma_quantile = function(z, alpha, theta) {
      return theta * this.igami( alpha, 1.- z);
   };

   JSROOTMath.log10 = function(n) {
      return Math.log(n) / Math.log(10);
   };

   JSROOTMath.landau_pdf = function(x, xi, x0) {
      // LANDAU pdf : algorithm from CERNLIB G110 denlan
      // same algorithm is used in GSL
      if (xi <= 0) return 0;
      var v = (x - x0)/xi;
      var u, ue, us, denlan;
      if (v < -5.5) {
         u   = Math.exp(v+1.0);
         if (u < 1e-10) return 0.0;
         ue  = Math.exp(-1/u);
         us  = Math.sqrt(u);
         denlan = 0.3989422803*(ue/us)*(1+(a1[0]+(a1[1]+a1[2]*u)*u)*u);
      } else if(v < -1) {
         u   = Math.exp(-v-1);
         denlan = Math.exp(-u)*Math.sqrt(u)*
            (p1[0]+(p1[1]+(p1[2]+(p1[3]+p1[4]*v)*v)*v)*v)/
            (q1[0]+(q1[1]+(q1[2]+(q1[3]+q1[4]*v)*v)*v)*v);
      } else if(v < 1) {
         denlan = (p2[0]+(p2[1]+(p2[2]+(p2[3]+p2[4]*v)*v)*v)*v)/
            (q2[0]+(q2[1]+(q2[2]+(q2[3]+q2[4]*v)*v)*v)*v);
      } else if(v < 5) {
         denlan = (p3[0]+(p3[1]+(p3[2]+(p3[3]+p3[4]*v)*v)*v)*v)/
            (q3[0]+(q3[1]+(q3[2]+(q3[3]+q3[4]*v)*v)*v)*v);
      } else if(v < 12) {
         u   = 1/v;
         denlan = u*u*(p4[0]+(p4[1]+(p4[2]+(p4[3]+p4[4]*u)*u)*u)*u)/
            (q4[0]+(q4[1]+(q4[2]+(q4[3]+q4[4]*u)*u)*u)*u);
      } else if(v < 50) {
         u   = 1/v;
         denlan = u*u*(p5[0]+(p5[1]+(p5[2]+(p5[3]+p5[4]*u)*u)*u)*u)/
            (q5[0]+(q5[1]+(q5[2]+(q5[3]+q5[4]*u)*u)*u)*u);
      } else if(v < 300) {
         u   = 1/v;
         denlan = u*u*(p6[0]+(p6[1]+(p6[2]+(p6[3]+p6[4]*u)*u)*u)*u)/
            (q6[0]+(q6[1]+(q6[2]+(q6[3]+q6[4]*u)*u)*u)*u);
      } else {
         u   = 1/(v-v*Math.log(v)/(v+1));
         denlan = u*u*(1+(a2[0]+a2[1]*u)*u);
      }
      return denlan/xi;
   };

   JSROOTMath.Landau = function(x, mpv, sigma, norm) {
      if (sigma <= 0) return 0;
      var den = JSROOTMath.landau_pdf((x - mpv) / sigma, 1, 0);
      if (!norm) return den;
      return den/sigma;
   };

})();

/// JSROOTMath.js ends

