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

var kTextNDC = BIT(14);

var kNotDraw = BIT(9);  // don't draw the function (TF1) when in a TH1

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
      if (null == obj || 'object' != typeof obj) return obj;
      //var copy = obj.constructor();
      var copy = {};
      for (var attr in obj) {
         if (obj.hasOwnProperty(attr)) copy[attr] = obj[attr];
      }
      return copy;
   };

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

   JSROOTCore.addMethods = function(obj) {
      // check object type and add methods if needed
      if (!obj['_typename'] || typeof(obj['_typename']) == 'undefined')
         return;
      if (obj['_typename'].indexOf("JSROOTIO.TH1") == 0 ||
          obj['_typename'].indexOf("JSROOTIO.TH2") == 0 ||
          obj['_typename'].indexOf("JSROOTIO.TH3") == 0) {
         obj['getBinError'] = function(bin) {
            //   -*-*-*-*-*Return value of error associated to bin number bin*-*-*-*-*
            //    if the sum of squares of weights has been defined (via Sumw2),
            //    this function returns the sqrt(sum of w2).
            //    otherwise it returns the sqrt(contents) for this bin.
            if (bin < 0) bin = 0;
            if (bin >= obj['fNcells']) bin = obj['fNcells'] - 1;
            if (obj['fN'] && obj['fSumw2'].length > 0) {
               var err2 = obj['fSumw2'][bin];
               return Math.sqrt(err2);
            }
            var error2 = Math.abs(obj['fArray'][bin]);
            return Math.sqrt(error2);
         };
         obj['getBinErrorLow'] = function(bin) {
            //   -*-*-*-*-*Return lower error associated to bin number bin*-*-*-*-*
            //    The error will depend on the statistic option used will return
            //     the binContent - lower interval value
            if (obj['fBinStatErrOpt'] == EBinErrorOpt.kNormal || obj['fN']) return obj.getBinError(bin);
            if (bin < 0) bin = 0;
            if (bin >= obj['fNcells']) bin = obj['fNcells'] - 1;
            var alpha = 1.0 - 0.682689492;
            if (obj['fBinStatErrOpt'] == EBinErrorOpt.kPoisson2) alpha = 0.05;
            var c = obj['fArray'][bin];
            var n = Math.round(c);
            if (n < 0) {
               alert("GetBinErrorLow : Histogram has negative bin content-force usage to normal errors");
               obj['fBinStatErrOpt'] = EBinErrorOpt.kNormal;
               return obj.getBinError(bin);
            }
            if (n == 0) return 0;
            return c - JSROOTMath.gamma_quantile( alpha/2, n, 1.);
         };
         obj['getBinErrorUp'] = function(bin) {
            //   -*-*-*-*-*Return lower error associated to bin number bin*-*-*-*-*
            //    The error will depend on the statistic option used will return
            //     the binContent - lower interval value
            if (obj['fBinStatErrOpt'] == EBinErrorOpt.kNormal || obj['fN']) return obj.getBinError(bin);
            if (bin < 0) bin = 0;
            if (bin >= obj['fNcells']) bin = obj['fNcells'] - 1;
            var alpha = 1.0 - 0.682689492;
            if (obj['fBinStatErrOpt'] == EBinErrorOpt.kPoisson2) alpha = 0.05;
            var c = obj['fArray'][bin];
            var n = Math.round(c);
            if (n < 0) {
               alert("GetBinErrorLow : Histogram has negative bin content-force usage to normal errors");
               obj['fBinStatErrOpt'] = EBinErrorOpt.kNormal;
               return obj.getBinError(bin);
            }
            // for N==0 return an upper limit at 0.68 or (1-alpha)/2 ?
            // decide to return always (1-alpha)/2 upper interval
            //if (n == 0) return ROOT::Math::gamma_quantile_c(alpha,n+1,1);
            return JSROOTMath.gamma_quantile_c( alpha/2, n+1, 1) - c;
         };
         obj['getBinLowEdge'] = function(bin) {
            // Return low edge of bin
            if (obj['fXaxis']['fXbins']['fN'] && bin > 0 && bin <= obj['fXaxis']['fNbins'])
               return obj['fXaxis']['fXbins']['fArray'][bin-1];
            var binwidth = (obj['fXaxis']['fXmax'] - obj['fXaxis']['fXmin']) / obj['fXaxis']['fNbins'];
            return obj['fXaxis']['fXmin'] + (bin-1) * binwidth;
         };
         obj['getBinUpEdge'] = function(bin) {
            // Return up edge of bin
            var binwidth;
            if (!obj['fXaxis']['fXbins']['fN'] || bin < 1 || bin > obj['fXaxis']['fNbins']) {
               binwidth = (obj['fXaxis']['fXmax'] - obj['fXaxis']['fXmin']) / obj['fXaxis']['fNbins'];
               return obj['fXaxis']['fXmin'] + bin * binwidth;
            } else {
               binwidth = obj['fArray'][bin] - obj['fArray'][bin-1];
               return obj['fArray'][bin-1] + binwidth;
            }
         };
         obj['getBinWidth'] = function(bin) {
            // Return bin width
            if (obj['fXaxis']['fNbins'] <= 0) return 0;
            if (obj['fXaxis']['fXbins']['fN'] <= 0)
               return (obj['fXaxis']['fXmax'] - obj['fXaxis']['fXmin']) / obj['fXaxis']['fNbins'];
            if (bin > obj['fXaxis']['fNbins']) bin = obj['fXaxis']['fNbins'];
            if (bin < 1) bin = 1;
            return obj['fArray'][bin] - obj['fArray'][bin-1];
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
            var bin, binx, stats = new Array(0,0,0,0,0,0,0,0,0,0,0,0,0);
            if ((obj['fTsumw'] == 0 && obj['fEntries'] > 0) || obj['fXaxis'].TestBit(EAxisBits.kAxisRange) || obj['fYaxis'].TestBit(EAxisBits.kAxisRange)) {
               var firstBinX = obj['fXaxis'].getFirst();
               var lastBinX  = obj['fXaxis'].getLast();
               var firstBinY = obj['fYaxis'].getFirst();
               var lastBinY  = obj['fYaxis'].getLast();
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
               }
               for (biny = firstBinY; biny <= lastBinY; biny++) {
                  y = obj['fYaxis'].getBinCenter(biny);
                  for (binx = firstBinX; binx <= lastBinX; binx++) {
                     bin = obj.getBin(binx,biny);
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
            }
            return stats;
         };
      }
      if (obj['_typename'].indexOf("JSROOTIO.TAxis") == 0) {
         obj['getFirst'] = function() {
            if (!obj.TestBit(EAxisBits.kAxisRange)) return 1;
            return obj['fFirst'];
         };
         obj['getLast'] = function() {
            if (!obj.TestBit(EAxisBits.kAxisRange)) return obj['fNbins'];
            return obj['fLast'];
         };
         obj['getBinCenter'] = function(bin) {
            // Return center of bin
            var binwidth;
            if (!obj['fN'] || bin < 1 || bin > obj['fNbins']) {
               binwidth = (obj['fXmax'] - obj['fXmin']) / obj['fNbins'];
               return obj['fXmin'] + (bin-1) * binwidth + 0.5*binwidth;
            } else {
               binwidth = obj['fXbins'][bin] - obj['fXbins'][bin-1];
               return obj['fXbins'][bin-1] + 0.5*binwidth;
            }
         };
      }
      if ((obj['_typename'].indexOf("TFormula") != -1) ||
          (obj['_typename'].indexOf("JSROOTIO.TF1") == 0)) {
         obj['evalPar'] = function(x) {
            var i, _function = obj['fTitle'];
            _function = _function.replace('TMath::Exp(', 'Math.exp(');
            _function = _function.replace('gaus(', 'gaus(this, ' + x + ', ');
            _function = _function.replace('gausn(', 'gausn(this, ' + x + ', ');
            _function = _function.replace('expo(', 'expo(this, ' + x + ', ');
            _function = _function.replace('landau(', 'landau(this, ' + x + ', ');
            _function = _function.replace('landaun(', 'landaun(this, ' + x + ', ');
            _function = _function.replace('pi', 'Math.PI');
            for (i=0;i<obj['fNpar'];++i) {
               while(_function.indexOf('['+i+']') != -1)
                  _function = _function.replace('['+i+']', obj['fParams'][i])
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
            var ret = eval(_function);
            return ret;
         };
      }
      if ((obj['_typename'].indexOf("JSROOTIO.TProfile") == 0) ||
          (obj['_typename'].indexOf("JSROOTIO.TH2") == 0)) {
         obj['getMean'] = function(axis) {
            if (axis < 1 || (axis > 3 && axis < 11) || axis > 13) return 0;
            var stats = obj.getStats();
            if (stats[0] == 0) return 0;
            var ax = new Array(2,4,7);
            return stats[ax[axis-1]]/stats[0];
         };
         obj['getRMS'] = function(axis) {
            if (axis < 1 || (axis > 3 && axis < 11) || axis > 13) return 0;
            var stats = obj.getStats();
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
            if (bin < 0 || bin >= obj['fNcells']) return 0;
            if (obj['fBinEntries'][bin] < 1e-300) return 0;
            if (!obj['fArray']) return 0;
            return obj['fArray'][bin]/obj['fBinEntries'][bin];
         };
         obj['getBinEffectiveEntries'] = function(bin) {
            if (bin < 0 || bin >= obj['fNcells']) return 0;
            var sumOfWeights = obj['fBinEntries'][bin];
            if ( obj['fBinSumw2'].length == 0 || obj['fBinSumw2'].length != obj['fNcells']) {
               // this can happen  when reading an old file
               return sumOfWeights;
            }
            var sumOfWeightsSquare = obj['fSumw2'][bin];
            return ( sumOfWeightsSquare > 0 ? sumOfWeights * sumOfWeights / sumOfWeightsSquare : 0 );
         };
         obj['getStats'] = function() {
            var bin, binx, stats = new Array(0,0,0,0,0,0,0,0,0,0,0,0,0);
            if (obj['fTsumw'] < 1e-300 || obj['fXaxis'].TestBit(EAxisBits.kAxisRange)) {
               var firstBinX = obj['fXaxis'].getFirst();
               var lastBinX  = obj['fXaxis'].getLast();
               for (binx = obj['firstBinX']; binx <= lastBinX; binx++) {
                  var w   = onj['fBinEntries'][binx];
                  var w2  = (obj['fN'] ? obj['fBinSumw2'][binx] : w);
                  var x   = fXaxis.GetBinCenter(binx);
                  stats[0] += w;
                  stats[1] += w2;
                  stats[2] += w*x;
                  stats[3] += w*x*x;
                  stats[4] += obj['fArray'][binx];
                  stats[5] += obj['fSumw2'][binx];
               }
            } else {
               if (obj['fTsumwy'] < 1e-300 && obj['fTsumwy2'] < 1e-300) {
                  //this case may happen when processing TProfiles with version <=3
                  for (binx=obj['fXaxis'].getFirst();binx<=obj['fXaxis'].getLast();binx++) {
                     obj['fTsumwy'] += obj['fArray'][binx];
                     obj['fTsumwy2'] += obj['fSumw2'][binx];
                  }
               }
               stats[0] = obj['fTsumw'];
               stats[1] = obj['fTsumw2'];
               stats[2] = obj['fTsumwx'];
               stats[3] = obj['fTsumwx2'];
               stats[4] = obj['fTsumwy'];
               stats[5] = obj['fTsumwy2'];
            }
            return stats;
         };
         obj['getBinError'] = function(bin) {
            if (bin < 0 || bin >= obj['fNcells']) return 0;
            var cont = obj['fArray'][bin];               // sum of bin w *y
            var sum  = obj['fBinEntries'][bin];          // sum of bin weights
            var err2 = obj['fSumw2'][bin];               // sum of bin w * y^2
            var neff = obj.getBinEffectiveEntries(bin);  // (sum of w)^2 / (sum of w^2)
            if (sum < 1e-300) return 0;                  // for empty bins
            // case the values y are gaussian distributed y +/- sigma and w = 1/sigma^2
            if (obj['fErrorMode'] == EErrorType.kERRORSPREADG) {
               return (1.0/Math.sqrt(sum));
            }
            // compute variance in y (eprim2) and standard deviation in y (eprim)
            var contsum = cont/sum;
            var eprim2  = Math.abs(err2/sum - contsum*contsum);
            var eprim   = Math.sqrt(eprim2);
            if (obj['fErrorMode'] == EErrorType.kERRORSPREADI) {
               if (eprim != 0) return eprim/Math.sqrt(neff);
               // in case content y is an integer (so each my has an error +/- 1/sqrt(12)
               // when the std(y) is zero
               return (1.0/Math.sqrt(12*neff));
            }
            // if approximate compute the sums (of w, wy and wy2) using all the bins
            //  when the variance in y is zero
            var testing = 1;
            if (err2 != 0 && neff < 5) testing = eprim2*sum/err2;
            if (obj['fgApproximate'] && (testing < 1.e-4 || eprim2 < 1e-6)) { //3.04
               var stats = obj.getStats();
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
            if (obj['fErrorMode'] == EErrorType.kERRORSPREAD) return eprim;
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

