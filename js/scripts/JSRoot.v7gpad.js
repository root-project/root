/// @file JSRoot.v7gpad.js
/// JavaScript ROOT graphics for ROOT v7 classes

JSROOT.define(['d3', 'painter'], (d3, jsrp) => {

   "use strict";

   JSROOT.v7 = {}; // placeholder for v7-relevant code

   /** @summary Evaluate attributes using fAttr storage and configured RStyle */
   JSROOT.ObjectPainter.prototype.v7EvalAttr = function(name, dflt) {
      let obj = this.GetObject();
      if (!obj) return dflt;
      if (this.cssprefix) name = this.cssprefix + name;

      function type_check(res) {
         if (dflt === undefined) return res;
         let typ1 = typeof dflt;
         let typ2 = typeof res;
         if (typ1 == typ2) return res;
         if (typ1 == 'boolean') {
            if (typ2 == 'string') return (res != "") && (res != "0") && (res != "no") && (res != "off");
            return !!res;
         }
         if ((typ1 == 'number') && (typ2 == 'string'))
            return parseFloat(res);
         return res;
      }

      if (obj.fAttr && obj.fAttr.m) {
         let value = obj.fAttr.m[name];
         if (value) return type_check(value.v); // found value direct in attributes
      }

      if (this.rstyle && this.rstyle.fBlocks) {
         let blks = this.rstyle.fBlocks;
         for (let k = 0; k < blks.length; ++k) {
            let block = blks[k];

            let match = (this.csstype && (block.selector == this.csstype)) ||
                        (obj.fId && (block.selector == ("#" + obj.fId))) ||
                        (obj.fCssClass && (block.selector == ("." + obj.fCssClass)));

            if (match && block.map && block.map.m) {
               let value = block.map.m[name];
               if (value) return type_check(value.v);
            }
         }
      }

      return dflt;
   }

   /** @summary Set attributes value */
   JSROOT.ObjectPainter.prototype.v7SetAttr = function(name, value) {
      let obj = this.GetObject();
      if (this.cssprefix) name = this.cssprefix + name;

      if (obj && obj.fAttr && obj.fAttr.m)
         obj.fAttr.m[name] = { v: value };
   }

   /** @summary Decode pad length from string, return pixel value */
   JSROOT.ObjectPainter.prototype.v7EvalLength = function(name, sizepx, dflt) {
      if (sizepx <= 0) sizepx = 1;

      let value = this.v7EvalAttr(name);

      if (value === undefined)
         return Math.round(dflt*sizepx);

      if (typeof value == "number")
         return Math.round(value*sizepx);

      if (value === null)
         return 0;

      let norm = 0, px = 0, val = value, operand = 0, pos = 0;

      while (val.length > 0) {
         // skip empty spaces
         while ((pos < val.length) && ((val[pos] == ' ') || (val[pos] == '\t')))
            ++pos;

         if (pos >= val.length)
            break;

         if ((val[pos] == '-') || (val[pos] == '+')) {
            if (operand) {
               console.log("Fail to parse RPadLength " + value);
               return dflt;
            }
            operand = (val[pos] == '-') ? -1 : 1;
            pos++;
            continue;
         }

         if (pos > 0) { val = val.substr(pos); pos = 0; }

         while ((pos < val.length) && (((val[pos]>='0') && (val[pos]<='9')) || (val[pos]=='.'))) pos++;

         let v = parseFloat(val.substr(0, pos));
         if (isNaN(v)) {
            console.log("Fail to parse RPadLength " + value);
            return Math.round(dflt*sizepx);
         }

         val = val.substr(pos);
         pos = 0;
         if (!operand) operand = 1;
         if ((val.length > 0) && (val[0] == '%')) {
            val = val.substr(1);
            norm += operand*v*0.01;
         } else if ((val.length > 1) && (val[0] == 'p') && (val[1] == 'x')) {
            val = val.substr(2);
            px += operand*v;
         } else {
            norm += operand*v;
         }

         operand = 0;
      }

      return Math.round(norm*sizepx + px);
   }

   /** @summary Evaluate RColor using attribute storage and configured RStyle
     * @private */
   JSROOT.ObjectPainter.prototype.v7EvalColor = function(name, dflt) {
      let rgb = this.v7EvalAttr(name + "_rgb", "");

      if (rgb)
         return "#" + rgb + this.v7EvalAttr(name + "_a", "");

      return this.v7EvalAttr(name + "_name", "") || dflt;
   }

   /** @summary Evaluate RAttrText properties
     * @return {Object} FontHandler, can be used directly for the text drawing
     * @private */
   JSROOT.ObjectPainter.prototype.v7EvalFont = function(name, dflts, fontScale) {

      if (!dflts) dflts = {}; else
      if (typeof dflts == "number") dflts = { size: dflts };

      let text_size   = this.v7EvalAttr( name + "_size", dflts.size || 12),
          text_angle  = this.v7EvalAttr( name + "_angle", 0),
          text_align  = this.v7EvalAttr( name + "_align", dflts.align || "none"),
          text_color  = this.v7EvalColor( name + "_color", dflts.color || "none"),
          font_family = this.v7EvalAttr( name + "_font_family", "Arial"),
          font_style  = this.v7EvalAttr( name + "_font_style", ""),
          font_weight = this.v7EvalAttr( name + "_font_weight", "");

       if (typeof text_size == "string") text_size = parseFloat(text_size);
       if (isNaN(text_size) || (text_size <= 0)) text_size = 12;
       if (!fontScale) fontScale = this.pad_height() || 10;

       let handler = new JSROOT.FontHandler(null, text_size, fontScale, font_family, font_style, font_weight);

       if (text_angle) handler.setAngle(360 - text_angle);
       if (text_align !== "none") handler.setAlign(text_align);
       if (text_color !== "none") handler.setColor(text_color);

       return handler;
    }

   /** @summary Create this.fillatt object based on v7 fill attributes
     * @private */
   JSROOT.ObjectPainter.prototype.createv7AttFill = function(prefix) {
      if (!prefix || (typeof prefix != "string")) prefix = "fill_";

      let fill_color = this.v7EvalColor(prefix + "color", ""),
          fill_style = this.v7EvalAttr(prefix + "style", 1001);

      this.createAttFill({ pattern: fill_style, color: 0 });

      this.fillatt.SetSolidColor(fill_color || "none");
   }

   /** @summary Create this.lineatt object based on v7 line attributes
     * @private */
   JSROOT.ObjectPainter.prototype.createv7AttLine = function(prefix) {
      if (!prefix || (typeof prefix != "string")) prefix = "line_";

      let line_color = this.v7EvalColor(prefix + "color", "black"),
          line_width = this.v7EvalAttr(prefix + "width", 1),
          line_style = this.v7EvalAttr(prefix + "style", 1);

      this.createAttLine({ color: line_color, width: line_width, style: line_style });
   }

    /** @summary Create this.markeratt object based on v7 attributes */
   JSROOT.ObjectPainter.prototype.createv7AttMarker = function(prefix) {
      if (!prefix || (typeof prefix != "string")) prefix = "marker_";

      let marker_color = this.v7EvalColor(prefix + "color", "black"),
          marker_size = this.v7EvalAttr(prefix + "size", 1),
          marker_style = this.v7EvalAttr(prefix + "style", 1);

      this.createAttMarker({ color: marker_color, size: marker_size, style: marker_style });
   }

   /** @summary Create RChangeAttr, which can be applied on the server side */
   JSROOT.ObjectPainter.prototype.v7AttrChange = function(req, name, value, kind) {
      if (!this.snapid)
         return false;

      if (!req._typename) {
         req._typename = "ROOT::Experimental::RChangeAttrRequest";
         req.ids = [];
         req.names = [];
         req.values = [];
         req.update = true;
      }

      if (this.cssprefix) name = this.cssprefix + name;
      req.ids.push(this.snapid);
      req.names.push(name);
      let obj = null;

      if ((value === null) || (value === undefined)) {
        if (!kind) kind = 'none';
        if (kind !== 'none') console.error(`Trying to set ${kind} for none value`);
      }

      if (!kind)
         switch(typeof value) {
            case "number": kind = "double"; break;
            case "boolean": kind = "boolean"; break;
         }

      obj = { _typename: "ROOT::Experimental::RAttrMap::" };
      switch(kind) {
         case "none": obj._typename += "NoValue_t"; break;
         case "boolean": obj._typename += "BoolValue_t"; obj.v = value ? true : false; break;
         case "int": obj._typename += "IntValue_t"; obj.v = parseInt(value); break;
         case "double": obj._typename += "DoubleValue_t"; obj.v = parseFloat(value); break;
         default: obj._typename += "StringValue_t"; obj.v = (typeof value == "string") ? value : JSON.stringify(value); break;
      }

      req.values.push(obj);
      return true;
   }

   /** @summary Sends accumulated attribute changes to server
     * @private */
   JSROOT.ObjectPainter.prototype.v7SendAttrChanges = function(req, do_update) {
      let canp = this.canv_painter();
      if (canp && req && req._typename) {
         if (do_update !== undefined) req.update = do_update ? true : false;
         canp.v7SubmitRequest("", req);
      }
   }

   /** @summary Submit request to server-side drawable
    * @param kind defines request kind, only single request a time can be submitted
    * @param req is object derived from DrawableRequest, including correct _typename
    * @param method is method of painter object which will be called when getting reply
    * @private */
   JSROOT.ObjectPainter.prototype.v7SubmitRequest = function(kind, req, method) {
      let canp = this.canv_painter();
      if (!canp || !canp.SubmitDrawableRequest) return null;

      // special situation when snapid not yet assigned - just keep ref until snapid is there
      // maybe keep full list - for now not clear if really needed
      if (!this.snapid) {
         this._pending_request = { _kind: kind, _req: req, _method: method };
         return req;
      }

      return canp.SubmitDrawableRequest(kind, req, this, method);
   }

   /** @summary Assign snapid to the painter
     * @desc Overwrite default method
     * @private */
   JSROOT.ObjectPainter.prototype.AssignSnapId = function(id) {
      this.snapid = id;
      if (this.snapid && this._pending_request) {
         let req = this._pending_request;
         this.v7SubmitRequest(req._kind, req._req, req._method);
         delete this._pending_request;
      }
   }

   JSROOT.v7.CommMode = { kNormal: 1, kLessTraffic: 2, kOffline: 3 }

   /** Return communication mode with the server
    * kOffline means no server there,
    * kLessTraffic advise not to send commands if offline functionality available
    * kNormal is standard functionality with RCanvas on server side*/
   JSROOT.ObjectPainter.prototype.v7CommMode = function() {
      let canp = this.canv_painter();
      if (!canp || !canp.SubmitDrawableRequest || !canp._websocket)
         return JSROOT.v7.CommMode.kOffline;

      return JSROOT.v7.CommMode.kNormal;
   }

   // ================================================================================

   /** @summary assign methods for the RAxis objects
     * @private */
   JSROOT.v7.AssignRAxisMethods = function(axis) {
      if ((axis._typename == "ROOT::Experimental::RAxisEquidistant") || (axis._typename == "ROOT::Experimental::RAxisLabels")) {
         if (axis.fInvBinWidth === 0) {
            axis.$dummy = true;
            axis.fInvBinWidth = 1;
            axis.fNBinsNoOver = 0;
            axis.fLow = 0;
         }

         axis.min = axis.fLow;
         axis.max = axis.fLow + axis.fNBinsNoOver/axis.fInvBinWidth;
         axis.GetNumBins = function() { return this.fNBinsNoOver; }
         axis.GetBinCoord = function(bin) { return this.fLow + bin/this.fInvBinWidth; }
         axis.FindBin = function(x,add) { return Math.floor((x - this.fLow)*this.fInvBinWidth + add); }
      } else if (axis._typename == "ROOT::Experimental::RAxisIrregular") {
         axis.min = axis.fBinBorders[0];
         axis.max = axis.fBinBorders[axis.fBinBorders.length - 1];
         axis.GetNumBins = function() { return this.fBinBorders.length; }
         axis.GetBinCoord = function(bin) {
            let indx = Math.round(bin);
            if (indx <= 0) return this.fBinBorders[0];
            if (indx >= this.fBinBorders.length) return this.fBinBorders[this.fBinBorders.length - 1];
            if (indx==bin) return this.fBinBorders[indx];
            let indx2 = (bin < indx) ? indx - 1 : indx + 1;
            return this.fBinBorders[indx] * Math.abs(bin-indx2) + this.fBinBorders[indx2] * Math.abs(bin-indx);
         }
         axis.FindBin = function(x,add) {
            for (let k = 1; k < this.fBinBorders.length; ++k)
               if (x < this.fBinBorders[k]) return Math.floor(k-1+add);
            return this.fBinBorders.length - 1;
         }
      }

      // to support some code from ROOT6 drawing

      axis.GetBinCenter = function(bin) { return this.GetBinCoord(bin-0.5); }
      axis.GetBinLowEdge = function(bin) { return this.GetBinCoord(bin-1); }
   }


   function RAxisPainter(arg1, axis, cssprefix) {
      let drawable = cssprefix ? arg1.GetObject() : arg1;
      this.axis = axis;
      JSROOT.AxisBasePainter.call(this, drawable);
      if (cssprefix) { // drawing from the frame
         this.embedded = true; // indicate that painter embedded into the histo painter
         this.csstype = arg1.csstype; // for the moment only via frame one can set axis attributes
         this.cssprefix = cssprefix;
         this.rstyle = arg1.rstyle;
      } else {
         this.csstype = "axis";
         this.cssprefix = "axis_";
      }
   }

   RAxisPainter.prototype = Object.create(JSROOT.AxisBasePainter.prototype);

   /** @summary Use in GED to identify kind of axis */
   RAxisPainter.prototype.GetAxisType = function() { return "RAttrAxis"; }

   /** @summary Configure only base parameters, later same handle will be used for drawing  */
   RAxisPainter.prototype.ConfigureZAxis = function(name, fp) {
      this.name = name;
      this.kind = "normal";
      this.log = false;
      let _log = this.v7EvalAttr("log", 0);
      if (_log) {
         this.log = true;
         this.logbase = 10;
         if (Math.abs(_log - Math.exp(1))<0.1)
            this.logbase = Math.exp(1);
         else if (_log > 1.9)
            this.logbase = Math.round(_log);
      }
      fp.logz = this.log;
   }

   /** @summary Configure axis painter
     * @desc Axis can be drawn inside frame <g> group with offset to 0 point for the frame
     * Therefore one should distinguish when caclulated coordinates used for axis drawing itself or for calculation of frame coordinates
     * @private */
   RAxisPainter.prototype.ConfigureAxis = function(name, min, max, smin, smax, vertical, frame_range, axis_range, opts) {
      if (!opts) opts = {};
      this.name = name;
      this.full_min = min;
      this.full_max = max;
      this.kind = "normal";
      this.vertical = vertical;
      this.log = false;
      let _log = this.v7EvalAttr("log", 0);
      this.reverse = opts.reverse || false;

      if (this.v7EvalAttr("time")) {
         this.kind = 'time';
         this.timeoffset = 0;
         let toffset = this.v7EvalAttr("time_offset");
         if (toffset !== undefined) {
            toffset = parseFloat(toffset);
            if (!isNaN(toffset)) this.timeoffset = toffset*1000;
         }
      } else if (this.axis && this.axis.fLabelsIndex) {
         this.kind = 'labels';
         delete this.own_labels;
      } else if (opts.labels) {
         this.kind = 'labels';
      } else {
         this.kind = 'normal';
      }

      if (this.kind == 'time') {
         this.func = d3.scaleTime().domain([this.ConvertDate(smin), this.ConvertDate(smax)]);
      } else if (_log) {
         if (smax <= 0) smax = 1;
         if ((smin <= 0) || (smin >= smax))
            smin = smax * 0.0001;
         this.log = true;
         this.logbase = 10;
         if (Math.abs(_log - Math.exp(1))<0.1)
            this.logbase = Math.exp(1);
         else if (_log > 1.9)
            this.logbase = Math.round(_log);
         this.func = d3.scaleLog().base(this.logbase).domain([smin,smax]);
      } else {
         this.func = d3.scaleLinear().domain([smin,smax]);
      }

      this.scale_min = smin;
      this.scale_max = smax;

      this.gr_range = axis_range || 1000; // when not specified, one can ignore it

      let range = frame_range ? frame_range : [0, this.gr_range];

      this.axis_shift = range[1] - this.gr_range;

      if (this.reverse)
         this.func.range([range[1], range[0]]);
      else
         this.func.range(range);

      if (this.kind == 'time')
         this.gr = val => this.func(this.ConvertDate(val));
      else if (this.log)
         this.gr = val => (val < this.scale_xmin) ? (this.vertical ? this.func.range()[0]+5 : -5) : this.func(val);
      else
         this.gr = this.func;

      delete this.format;// remove formatting func

      let ndiv = this.v7EvalAttr("ndiv", 508);

      this.nticks = ndiv % 100;
      this.nticks2 = (ndiv % 10000 - this.nticks) / 100;
      this.nticks3 = Math.floor(ndiv/10000);

      if (this.nticks > 7) this.nticks = 7;

      let gr_range = Math.abs(this.gr_range) || 100;

      if (this.kind == 'time') {
         if (this.nticks > 8) this.nticks = 8;

         let scale_range = this.scale_max - this.scale_min,
             tf1 = this.v7EvalAttr("time_format", ""),
             tf2 = jsrp.chooseTimeFormat(scale_range / gr_range, false);

         if (!tf1 || (scale_range < 0.1 * (this.full_max - this.full_min)))
            tf1 = jsrp.chooseTimeFormat(scale_range / this.nticks, true);

         this.tfunc1 = this.tfunc2 = d3.timeFormat(tf1);
         if (tf2!==tf1)
            this.tfunc2 = d3.timeFormat(tf2);

         this.format = this.formatTime;

      } else if (this.log) {
         if (this.nticks2 > 1) {
            this.nticks *= this.nticks2; // all log ticks (major or minor) created centrally
            this.nticks2 = 1;
         }
         this.noexp = this.v7EvalAttr("noexp", false);
         if ((this.scale_max < 300) && (this.scale_min > 0.3) && (this.logbase == 10)) this.noexp = true;
         this.moreloglabels = this.v7EvalAttr("moreloglbls", false);

         this.format = this.formatLog;
      } else if (this.kind == 'labels') {
         this.nticks = 50; // for text output allow max 50 names
         let scale_range = this.scale_max - this.scale_min;
         if (this.nticks > scale_range)
            this.nticks = Math.round(scale_range);
         this.nticks2 = 1;

         this.format = this.formatLabels;
      } else {

         this.order = 0;
         this.ndig = 0;

         this.format = this.formatNormal;
      }
   }

   /** @summary Provide label for axis value */
   RAxisPainter.prototype.formatLabels = function(d) {
      let indx = Math.round(d);
      if (this.axis && this.axis.fLabelsIndex) {
         if ((indx < 0) || (indx >= this.axis.fNBinsNoOver)) return null;
         for (let i = 0; i < this.axis.fLabelsIndex.length; ++i) {
            let pair = this.axis.fLabelsIndex[i];
            if (pair.second === indx) return pair.first;
         }
      } else {
         let labels = this.GetObject().fLabels;
         if (labels && (indx>=0) && (indx < labels.length))
            return labels[indx];
      }
      return null;
   }

   /** @summary Creates array with minor/middle/major ticks */
   RAxisPainter.prototype.CreateTicks = function(only_major_as_array, optionNoexp, optionNoopt, optionInt) {

      if (optionNoopt && this.nticks && (this.kind == "normal")) this.noticksopt = true;

      let handle = { nminor: 0, nmiddle: 0, nmajor: 0, func: this.func };

      handle.minor = handle.middle = handle.major = this.ProduceTicks(this.nticks);

      if (only_major_as_array) {
         let res = handle.major, delta = (this.scale_max - this.scale_min)*1e-5;
         if (res[0] > this.scale_min + delta) res.unshift(this.scale_min);
         if (res[res.length-1] < this.scale_max - delta) res.push(this.scale_max);
         return res;
      }

      if ((this.nticks2 > 1) && (!this.log || (this.logbase === 10))) {
         handle.minor = handle.middle = this.ProduceTicks(handle.major.length, this.nticks2);

         let gr_range = Math.abs(this.func.range()[1] - this.func.range()[0]);

         // avoid black filling by middle-size
         if ((handle.middle.length <= handle.major.length) || (handle.middle.length > gr_range/3.5)) {
            handle.minor = handle.middle = handle.major;
         } else if ((this.nticks3 > 1) && !this.log)  {
            handle.minor = this.ProduceTicks(handle.middle.length, this.nticks3);
            if ((handle.minor.length <= handle.middle.length) || (handle.minor.length > gr_range/1.7)) handle.minor = handle.middle;
         }
      }

      handle.reset = function() {
         this.nminor = this.nmiddle = this.nmajor = 0;
      }

      handle.next = function(doround) {
         if (this.nminor >= this.minor.length) return false;

         this.tick = this.minor[this.nminor++];
         this.grpos = this.func(this.tick);
         if (doround) this.grpos = Math.round(this.grpos);
         this.kind = 3;

         if ((this.nmiddle < this.middle.length) && (Math.abs(this.grpos - this.func(this.middle[this.nmiddle])) < 1)) {
            this.nmiddle++;
            this.kind = 2;
         }

         if ((this.nmajor < this.major.length) && (Math.abs(this.grpos - this.func(this.major[this.nmajor])) < 1) ) {
            this.nmajor++;
            this.kind = 1;
         }
         return true;
      }

      handle.last_major = function() {
         return (this.kind !== 1) ? false : this.nmajor == this.major.length;
      }

      handle.next_major_grpos = function() {
         if (this.nmajor >= this.major.length) return null;
         return this.func(this.major[this.nmajor]);
      }

      this.order = 0;
      this.ndig = 0;

      // at the moment when drawing labels, we can try to find most optimal text representation for them

      if ((this.kind == "normal") && !this.log && (handle.major.length > 0)) {

         let maxorder = 0, minorder = 0, exclorder3 = false;

         if (!optionNoexp) {
            let maxtick = Math.max(Math.abs(handle.major[0]),Math.abs(handle.major[handle.major.length-1])),
                mintick = Math.min(Math.abs(handle.major[0]),Math.abs(handle.major[handle.major.length-1])),
                ord1 = (maxtick > 0) ? Math.round(Math.log10(maxtick)/3)*3 : 0,
                ord2 = (mintick > 0) ? Math.round(Math.log10(mintick)/3)*3 : 0;

             exclorder3 = (maxtick < 2e4); // do not show 10^3 for values below 20000

             if (maxtick || mintick) {
                maxorder = Math.max(ord1,ord2) + 3;
                minorder = Math.min(ord1,ord2) - 3;
             }
         }

         // now try to find best combination of order and ndig for labels

         let bestorder = 0, bestndig = this.ndig, bestlen = 1e10;

         for (let order = minorder; order <= maxorder; order+=3) {
            if (exclorder3 && (order===3)) continue;
            this.order = order;
            this.ndig = 0;
            let lbls = [], indx = 0, totallen = 0;
            while (indx<handle.major.length) {
               let lbl = this.format(handle.major[indx], true);
               if (lbls.indexOf(lbl)<0) {
                  lbls.push(lbl);
                  totallen += lbl.length;
                  indx++;
                  continue;
               }
               if (++this.ndig > 11) break; // not too many digits, anyway it will be exponential
               lbls = []; indx = 0; totallen = 0;
            }

            // for order==0 we should virually remove "0." and extra label on top
            if (!order && (this.ndig<4)) totallen-=(handle.major.length*2+3);

            if (totallen < bestlen) {
               bestlen = totallen;
               bestorder = this.order;
               bestndig = this.ndig;
            }
         }

         this.order = bestorder;
         this.ndig = bestndig;

         if (optionInt) {
            if (this.order) console.warn('Axis painter - integer labels are configured, but axis order ' + this.order + ' is preferable');
            if (this.ndig) console.warn('Axis painter - integer labels are configured, but ' + this.ndig + ' decimal digits are required');
            this.ndig = 0;
            this.order = 0;
         }
      }

      return handle;
   }

   /** @summary Is labels should be centered */
   RAxisPainter.prototype.IsCenterLabels = function() {
      if (this.kind === 'labels') return true;
      if (this.kind === 'log') return false;
      return this.v7EvalAttr("labels_center", false);
   }

   /** @summary Used to move axis labels instead of zooming
     * @private */
   RAxisPainter.prototype.processLabelsMove = function(arg, pos) {
      if (this.optionUnlab || !this.axis_g) return false;

      let label_g = this.axis_g.select(".axis_labels");
      if (!label_g || (label_g.size() != 1)) return false;

      if (arg == 'start') {
         // no moving without labels
         let box = label_g.node().getBBox();

         label_g.append("rect")
                 .classed("zoom", true)
                 .attr("x", box.x)
                 .attr("y", box.y)
                 .attr("width", box.width)
                 .attr("height", box.height)
                 .style("cursor", "move");
         if (this.vertical) {
            this.drag_pos0 = pos[0];
         } else {
            this.drag_pos0 = pos[1];
         }

         return true;
      }

      let offset = label_g.property('fix_offset');

      if (this.vertical) {
         offset += Math.round(pos[0] - this.drag_pos0);
         label_g.attr('transform', `translate(${offset},0)`);
      } else {
         offset += Math.round(pos[1] - this.drag_pos0);
         label_g.attr('transform', `translate(0,${offset})`);
      }
      if (!offset) label_g.attr('transform', null);

      if (arg == 'stop') {
         label_g.select("rect.zoom").remove();
         delete this.drag_pos0;
         if (offset != label_g.property('fix_offset')) {
            label_g.property('fix_offset', offset);
            let side = label_g.property('side') || 1;
            this.labelsOffset = offset / (this.vertical ? -side : side);
            this.ChangeAxisAttr(1, "labels_offset", this.labelsOffset/this.scaling_size);
         }
      }

      return true;
   }

   /** @summary Add interactive elements to draw axes title */
   RAxisPainter.prototype.addTitleDrag = function(title_g, side) {
      if (!JSROOT.settings.MoveResize || JSROOT.BatchMode) return;

      let drag_rect = null,
          acc_x, acc_y, new_x, new_y, alt_pos, curr_indx,
          drag_move = d3.drag().subject(Object);

      drag_move
         .on("start", evnt => {

            evnt.sourceEvent.preventDefault();
            evnt.sourceEvent.stopPropagation();

            let box = title_g.node().getBBox(), // check that elements visible, request precise value
                title_length = this.vertical ? box.height : box.width;

            new_x = acc_x = title_g.property('shift_x');
            new_y = acc_y = title_g.property('shift_y');

            if (this.titlePos == "center")
               curr_indx = 1;
            else
               curr_indx = (this.titlePos == "left") ? 0 : 2;

            // let d = ((this.gr_range > 0) && this.vertical) ? title_length : 0;
            alt_pos = [0, this.gr_range/2, this.gr_range]; // possible positions
            let off = this.vertical ? -title_length : title_length,
                swap = this.IsReverseAxis() ? 2 : 0;
            if (this.title_align == "middle") {
               alt_pos[swap] += off/2;
               alt_pos[2-swap] -= off/2;
            } else if ((this.title_align == "begin") ^ this.isTitleRotated()) {
               alt_pos[1] -= off/2;
               alt_pos[2-swap] -= off;
            } else { // end
               alt_pos[swap] += off;
               alt_pos[1] += off/2;
            }

            alt_pos[curr_indx] = this.vertical ? acc_y : acc_x;

            drag_rect = title_g.append("rect")
                 .classed("zoom", true)
                 .attr("x", box.x)
                 .attr("y", box.y)
                 .attr("width", box.width)
                 .attr("height", box.height)
                 .style("cursor", "move");
//                 .style("pointer-events","none"); // let forward double click to underlying elements
          }).on("drag", evnt => {
               if (!drag_rect) return;

               evnt.sourceEvent.preventDefault();
               evnt.sourceEvent.stopPropagation();

               acc_x += evnt.dx;
               acc_y += evnt.dy;

               let set_x = title_g.property('shift_x'),
                   set_y = title_g.property('shift_y'),
                   p = this.vertical ? acc_y : acc_x, besti = 0;

               for (let i=1; i<3; ++i)
                  if (Math.abs(p - alt_pos[i]) < Math.abs(p - alt_pos[besti])) besti = i;

               if (this.vertical) {
                  set_x = acc_x;
                  set_y = alt_pos[besti];
               } else {
                  set_x = alt_pos[besti];
                  set_y = acc_y;
               }

               new_x = set_x; new_y = set_y; curr_indx = besti;
               title_g.attr('transform', 'translate(' + Math.round(new_x) + ',' + Math.round(new_y) +  ')');

          }).on("end", evnt => {
               if (!drag_rect) return;

               evnt.sourceEvent.preventDefault();
               evnt.sourceEvent.stopPropagation();

               let basepos = title_g.property('basepos') || 0;

               title_g.property('shift_x', new_x)
                      .property('shift_y', new_y);

               this.titleOffset = (this.vertical ? basepos - new_x : new_y - basepos) * side;

               if (curr_indx == 1) {
                  this.titlePos = "center";
               } else if (curr_indx == 0) {
                  this.titlePos = "left";
               } else {
                  this.titlePos = "right";
               }

               this.ChangeAxisAttr(0, "title_position", this.titlePos, "title_offset", this.titleOffset/this.scaling_size);

               drag_rect.remove();
               drag_rect = null;
            });

      title_g.style("cursor", "move").call(drag_move);
   }

   /** @summary checks if value inside graphical range, taking into account delta */
   RAxisPainter.prototype.IsInsideGrRange = function(pos, delta1, delta2) {
      if (!delta1) delta1 = 0;
      if (delta2 === undefined) delta2 = delta1;
      if (this.gr_range < 0)
         return (pos >= this.gr_range - delta2) && (pos <= delta1);
      return (pos >= -delta1) && (pos <= this.gr_range + delta2);
   }

   /** @summary returns graphical range */
   RAxisPainter.prototype.GrRange = function(delta) {
      if (!delta) delta = 0;
      if (this.gr_range < 0)
         return this.gr_range - delta;
      return this.gr_range + delta;
   }

   /** @summary If axis direction is negative coordinates direction */
   RAxisPainter.prototype.IsReverseAxis = function() {
      return !this.vertical !== (this.GrRange() > 0);
   }

   /** @summary Draw axis ticks
     * @private */
   RAxisPainter.prototype.DrawMainLine = function(axis_g) {
      let ending = "";

      if (this.endingSize && this.endingStyle) {
         let sz = (this.gr_range > 0) ? -this.endingSize : this.endingSize,
             sz7 = Math.round(sz*0.7);
         sz = Math.round(sz);
         if (this.vertical)
            ending = "l" + sz7 + "," + sz +
                     "M0," + this.gr_range +
                     "l" + (-sz7) + "," + sz;
         else
            ending = "l" + sz + "," + sz7 +
                     "M" + this.gr_range + ",0" +
                     "l" + sz + "," + (-sz7);
      }

      axis_g.append("svg:path")
            .attr("d","M0,0" + (this.vertical ? "v" : "h") + this.gr_range + ending)
            .call(this.lineatt.func)
            .style('fill', ending ? "none" : null);
   }

   /** @summary Draw axis ticks
     * @returns {Promise} with gaps on left and right side
     * @private */
   RAxisPainter.prototype.drawTicks = function(axis_g, side, main_draw) {
      if (main_draw) this.ticks = [];

      this.handle.reset();

      let res = "", ticks_plusminus = 0, lastpos = 0, lasth = 0;
      if (this.ticksSide == "both") {
         side = 1;
         ticks_plusminus = 1;
      }

      while (this.handle.next(true)) {

         let h1 = Math.round(this.ticksSize/4), h2 = 0;

         if (this.handle.kind < 3)
            h1 = Math.round(this.ticksSize/2);

         let grpos = this.handle.grpos - this.axis_shift;

         if ((this.startingSize || this.endingSize) && !this.IsInsideGrRange(grpos, -Math.abs(this.startingSize), -Math.abs(this.endingSize))) continue;

         if (this.handle.kind == 1) {
            // if not showing labels, not show large tick
            if ((this.kind == "labels") || (this.format(this.handle.tick,true) !== null)) h1 = this.ticksSize;

            if (main_draw) this.ticks.push(grpos); // keep graphical positions of major ticks
         }

         if (ticks_plusminus > 0) h2 = -h1; else
         if (side < 0) { h2 = -h1; h1 = 0; } else { h2 = 0; }

         if (res.length == 0) {
            res = this.vertical ? "M"+h1+","+grpos : "M"+grpos+","+(-h1);
         } else {
            res += this.vertical ? "m"+(h1-lasth)+","+(grpos-lastpos) : "m"+(grpos-lastpos)+","+(lasth-h1);
         }

         res += this.vertical ? "h"+ (h2-h1) : "v"+ (h1-h2);

         lastpos = grpos;
         lasth = h2;
      }

      if (res)
         axis_g.append("svg:path")
               .attr("d", res)
               .style('stroke', this.ticksColor || this.lineatt.color);

       let gap0 = Math.round(0.25*this.ticksSize), gap = Math.round(1.25*this.ticksSize);
       return { "-1": (side > 0) || ticks_plusminus ? gap : gap0,
                "1": (side < 0) || ticks_plusminus ? gap : gap0 };
   }

   /** @summary Performs labels drawing
     * @returns {Promise} wwith gaps in both direction */
   RAxisPainter.prototype.drawLabels = function(axis_g, side, gaps) {
      let center_lbls = this.IsCenterLabels(),
          rotate_lbls = false,
          textscale = 1, maxtextlen = 0, lbls_tilt = false,
          label_g = axis_g.append("svg:g").attr("class","axis_labels").property('side', side),
          lbl_pos = this.handle.lbl_pos || this.handle.major,
          max_lbl_width = 0, max_lbl_height = 0;

      // function called when text is drawn to analyze width, required to correctly scale all labels
      function process_drawtext_ready(painter) {

         max_lbl_width = Math.max(max_lbl_width, this.result_width);
         max_lbl_height = Math.max(max_lbl_height, this.result_height);

         let textwidth = this.result_width;

         if (textwidth && ((!painter.vertical && !rotate_lbls) || (painter.vertical && rotate_lbls)) && !painter.log) {
            let maxwidth = this.gap_before*0.45 + this.gap_after*0.45;
            if (!this.gap_before) maxwidth = 0.9*this.gap_after; else
            if (!this.gap_after) maxwidth = 0.9*this.gap_before;
            textscale = Math.min(textscale, maxwidth / textwidth);
         }

         if ((textscale > 0.01) && (textscale < 0.8) && !painter.vertical && !rotate_lbls && (maxtextlen > 5) && (side > 0))
            lbls_tilt = true;

         let scale = textscale * (lbls_tilt ? 3 : 1);
         if ((scale > 0.01) && (scale < 1))
            painter.TextScaleFactor(1/scale, label_g);
      }

      this.labelsFont = this.v7EvalFont("labels", { size: 0.03 });
      this.labelsFont.roundAngle(180);
      if (this.labelsFont.angle) { this.labelsFont.angle = 270; rotate_lbls = true; }

      let lastpos = 0,
          fix_offset = Math.round((this.vertical ? -side : side)*this.labelsOffset),
          fix_coord = Math.round((this.vertical ? -side : side)*gaps[side]);

      if (fix_offset)
         label_g.attr('transform', this.vertical ? `translate(${fix_offset},0)` : `translate(0,${fix_offset})`);

      label_g.property('fix_offset', fix_offset);

      this.startTextDrawing(this.labelsFont, 'font', label_g);

      for (let nmajor=0;nmajor<lbl_pos.length;++nmajor) {

         let lbl = this.format(lbl_pos[nmajor], true);
         if (lbl === null) continue;

         let pos = Math.round(this.func(lbl_pos[nmajor]));

         let arg = { text: lbl, latex: 1, draw_g: label_g };

         arg.gap_before = (nmajor>0) ? Math.abs(Math.round(pos - this.func(lbl_pos[nmajor-1]))) : 0,
         arg.gap_after = (nmajor<lbl_pos.length-1) ? Math.abs(Math.round(this.func(lbl_pos[nmajor+1])-pos)) : 0;

         if (center_lbls) {
            let gap = arg.gap_after || arg.gap_before;
            pos = Math.round(pos - (this.vertical ? 0.5*gap : -0.5*gap));
            if (!this.IsInsideGrRange(pos, 5)) continue;
         }

         maxtextlen = Math.max(maxtextlen, lbl.length);

         pos -= this.axis_shift;

         if ((this.startingSize || this.endingSize) && !this.IsInsideGrRange(pos, -Math.abs(this.startingSize), -Math.abs(this.endingSize))) continue;

         if (this.vertical) {
            arg.x = fix_coord;
            arg.y = pos;
            arg.align = rotate_lbls ? ((side<0) ? 23 : 20) : ((side<0) ? 12 : 32);
         } else {
            arg.x = pos;
            arg.y = fix_coord;
            arg.align = rotate_lbls ? ((side<0) ? 12 : 32) : ((side<0) ? 20 : 23);
         }

         arg.post_process = process_drawtext_ready;

         this.drawText(arg);

         if (lastpos && (pos!=lastpos) && ((this.vertical && !rotate_lbls) || (!this.vertical && rotate_lbls))) {
            let axis_step = Math.abs(pos-lastpos);
            textscale = Math.min(textscale, 0.9*axis_step/this.labelsFont.size);
         }

         lastpos = pos;
      }

      if (this.order)
         this.drawText({ x: this.vertical ? side*5 : this.GrRange(5),
                         y: this.has_obstacle ? fix_coord : (this.vertical ? this.GrRange(3) : -3*side),
                         align: this.vertical ? ((side<0) ? 30 : 10) : ((this.has_obstacle ^ (side < 0)) ? 13 : 10),
                         latex: 1,
                         text: '#times' + this.formatExp(10, this.order),
                         draw_g: label_g
         });

      return this.finishTextDrawing(label_g).then(() => {

        if (lbls_tilt)
           label_g.selectAll("text").each(function () {
               let txt = d3.select(this), tr = txt.attr("transform");
               txt.attr("transform", tr + " rotate(25)").style("text-anchor", "start");
           });

         if (this.vertical) {
            gaps[side] += Math.round(rotate_lbls ? 1.2*max_lbl_height : max_lbl_width + 0.4*this.labelsFont.size) - side*fix_offset;
         } else {
            let tilt_height = lbls_tilt ? max_lbl_width * Math.sin(25/180*Math.PI) + max_lbl_height * (Math.cos(25/180*Math.PI) + 0.2) : 0;

            gaps[side] += Math.round(Math.max(rotate_lbls ? max_lbl_width + 0.4*this.labelsFont.size : 1.2*max_lbl_height, 1.2*this.labelsFont.size, tilt_height)) + fix_offset;
         }

         return gaps;
      });
   }

   /** @summary Add zomming rect to axis drawing */
   RAxisPainter.prototype.addZoomingRect = function(axis_g, side, lgaps) {
      if (JSROOT.settings.Zooming && !this.disable_zooming && !JSROOT.BatchMode) {
         let sz = Math.max(lgaps[side], 10);

         let d = this.vertical ? "v" + this.gr_range + "h"+(-side*sz) + "v" + (-this.gr_range)
                               : "h" + this.gr_range + "v"+(side*sz) + "h" + (-this.gr_range);
         axis_g.append("svg:path")
               .attr("d","M0,0" + d + "z")
               .attr("class", "axis_zoom")
               .style("opacity", "0")
               .style("cursor", "crosshair");
      }
   }

   /** @summary Returns true if axis title is rotated */
   RAxisPainter.prototype.isTitleRotated = function() {
      return this.titleFont && (this.titleFont.angle != (this.vertical ? 270 : 0));
   }

   /** @summary Draw axis title */
   RAxisPainter.prototype.drawTitle = function(axis_g, side, lgaps) {
      if (!this.fTitle) return Promise.resolve(true);

      let title_g = axis_g.append("svg:g").attr("class", "axis_title"),
          title_position = this.v7EvalAttr("title_position", "right"),
          center = (title_position == "center"),
          opposite = (title_position == "left"),
          title_shift_x = 0, title_shift_y = 0, title_basepos = 0;

      this.titleFont = this.v7EvalFont("title", { size: 0.03 }, this.pad_height());
      this.titleFont.roundAngle(180, this.vertical ? 270 : 0);

      this.titleOffset = this.v7EvalLength("title_offset", this.scaling_size, 0);
      this.titlePos = title_position;

      let rotated = this.isTitleRotated();

      this.startTextDrawing(this.titleFont, 'font', title_g);

      this.title_align = center ? "middle" : (opposite ^ (this.IsReverseAxis() || rotated) ? "begin" : "end");

      if (this.vertical) {
         title_basepos = Math.round(-side*(lgaps[side]));
         title_shift_x = title_basepos + Math.round(-side*this.titleOffset);
         title_shift_y = Math.round(center ? this.gr_range/2 : (opposite ? 0 : this.gr_range));
         this.drawText({ align: [this.title_align, ((side < 0) ^ rotated ? 'top' : 'bottom')],
                         text: this.fTitle, draw_g: title_g });
      } else {
         title_shift_x = Math.round(center ? this.gr_range/2 : (opposite ? 0 : this.gr_range));
         title_basepos = Math.round(side*lgaps[side]);
         title_shift_y = title_basepos + Math.round(side*this.titleOffset);
         this.drawText({ align: [this.title_align, ((side > 0) ^ rotated ? 'top' : 'bottom')],
                         text: this.fTitle, draw_g: title_g });
      }

      title_g.attr('transform', 'translate(' + title_shift_x + ',' + title_shift_y +  ')')
             .property('basepos', title_basepos)
             .property('shift_x', title_shift_x)
             .property('shift_y', title_shift_y);

      this.addTitleDrag(title_g, side);

      return this.finishTextDrawing(title_g);
   }

   /** @summary Extract major draw attributes, which are also used in interactive operations
     * @private  */
   RAxisPainter.prototype.extractDrawAttributes = function() {
       this.createv7AttLine("line_");

      this.endingStyle = this.v7EvalAttr("ending_style", "");
      this.endingSize = Math.round(this.v7EvalLength("ending_size", this.scaling_size, this.endingStyle ? 0.02 : 0));
      this.startingSize = Math.round(this.v7EvalLength("starting_size", this.scaling_size, 0));
      this.ticksSize = this.v7EvalLength("ticks_size", this.scaling_size, 0.02);
      this.ticksSide = this.v7EvalAttr("ticks_side", "normal");
      this.ticksColor = this.v7EvalColor("ticks_color", "");
      this.labelsOffset = this.v7EvalLength("labels_offset", this.scaling_size, 0);

      this.fTitle = this.v7EvalAttr("title", "");

      if (this.max_tick_size && (this.ticksSize > this.max_tick_size)) this.ticksSize = this.max_tick_size;
   }

   /** @summary Performs axis drawing
     * @returns {Promise} which resolved when drawing is completed */
   RAxisPainter.prototype.drawAxis = function(layer, transform, side) {
      let axis_g = layer,
          pad_w = this.pad_width() || 10,
          pad_h = this.pad_height() || 10;

      if (side === undefined) side = 1;

      if (!this.standalone) {
         axis_g = layer.select("." + this.name + "_container");
         if (axis_g.empty())
            axis_g = layer.append("svg:g").attr("class",this.name + "_container");
         else
            axis_g.selectAll("*").remove();
      }

      axis_g.attr("transform", transform || null);

      this.scaling_size = this.vertical ? pad_w : pad_h;
      this.extractDrawAttributes();
      this.axis_g = axis_g;
      this.side = side;

      if (this.ticksSide == "invert") side = -side;

      if (this.standalone)
         this.DrawMainLine(axis_g);

      let optionUnlab = false,  // no labels
          optionNoopt = false,  // no ticks position optimization
          optionInt = false,    // integer labels
          optionNoexp = false;  // do not create exp

      this.handle = this.CreateTicks(false, optionNoexp, optionNoopt, optionInt);

      // first draw ticks
      let tgaps = this.drawTicks(axis_g, side, true);

      this.optionUnlab = optionUnlab;

      // draw labels
      let labelsPromise = optionUnlab ? Promise.resolve(tgaps) : this.drawLabels(axis_g, side, tgaps);

      return labelsPromise.then(lgaps => {
         // when drawing axis on frame, zoom rect should be always outside
         this.addZoomingRect(axis_g, this.standalone ? side : this.side, lgaps);

         return this.drawTitle(axis_g, side, lgaps);
      });
   }

   /** @summary Assign handler, which is called when axis redraw by interactive changes
     * @desc Used by palette painter to reassign iteractive handlers
     * @private */
   RAxisPainter.prototype.setAfterDrawHandler = function(handler) {
      this._afterDrawAgain = handler;
   }

   /** @summary Draw axis with the same settings, used by interactive changes */
   RAxisPainter.prototype.drawAxisAgain = function() {
      if (!this.axis_g || !this.side) return;

      this.axis_g.selectAll("*").remove();

      this.extractDrawAttributes();

      let side = this.side;
      if (this.ticksSide == "invert") side = -side;

      if (this.standalone)
         this.DrawMainLine(this.axis_g);

      // first draw ticks
      let tgaps = this.drawTicks(this.axis_g, side, false);

      let labelsPromise = this.optionUnlab ? Promise.resolve(tgaps) : this.drawLabels(this.axis_g, side, tgaps);

      return labelsPromise.then(lgaps => {
         // when drawing axis on frame, zoom rect should be always outside
         this.addZoomingRect(this.axis_g, this.standalone ? side : this.side, lgaps);

         return this.drawTitle(this.axis_g, side, lgaps);
      }).then(() => {
         if (typeof this._afterDrawAgain == 'function')
            this._afterDrawAgain();
      });
   }

   /** @summary Draw axis again on opposite frame size */
   RAxisPainter.prototype.drawAxisOtherPlace = function(layer, transform, side, only_ticks) {
      let axis_g = layer.select("." + this.name + "_container2");
      if (axis_g.empty())
         axis_g = layer.append("svg:g").attr("class",this.name + "_container2");
      else
         axis_g.selectAll("*").remove();

      axis_g.attr("transform", transform || null);

      if (this.ticksSide == "invert") side = -side;

      // draw ticks again
      let tgaps = this.drawTicks(axis_g, side, false);

      // draw labels again
      let promise = this.optionUnlab || only_ticks ? Promise.resolve(tgaps) : this.drawLabels(axis_g, side, tgaps);

      return promise.then(lgaps => {
         this.addZoomingRect(axis_g, side, lgaps);
         return true;
      });
   }

   /** @summary Change zooming in standalone mode */
   RAxisPainter.prototype.ZoomStandalone = function(min,max) {
      this.ChangeAxisAttr(1, "zoommin", min, "zoommax", max);
   }

   /** @summary Redraw axis, used in standalone mode for RAxisDrawable */
   RAxisPainter.prototype.Redraw = function() {

      let drawable = this.GetObject(),
          pp   = this.pad_painter(),
          pos  = pp.GetCoordinate(drawable.fPos),
          len  = pp.GetPadLength(drawable.fVertical, drawable.fLength),
          reverse = this.v7EvalAttr("reverse", false),
          min = 0, max = 1;

      // in vertical direction axis drawn in negative direction
      if (drawable.fVertical) len = -len;

      if (drawable.fLabels) {
         min = 0;
         max = drawable.fLabels.length;
      } else {
         min = drawable.fMin;
         max = drawable.fMax;
      }

      let smin = this.v7EvalAttr("zoommin"),
          smax = this.v7EvalAttr("zoommax");
      if (smin === smax) {
         smin = min; smax = max;
      }

      this.ConfigureAxis("axis", min, max, smin, smax, drawable.fVertical, undefined, len, { reverse: reverse, labels: !!drawable.fLabels });

      this.CreateG();

      this.standalone = true;  // no need to clean axis container

      let promise = this.drawAxis(this.draw_g, "translate(" + pos.x + "," + pos.y +")");

      if (JSROOT.BatchMode) return promise;

      return promise.then(() => JSROOT.require('interactive')).then(inter => {
         if (JSROOT.settings.ContextMenu)
            this.draw_g.on("contextmenu", evnt => {
               evnt.stopPropagation(); // disable main context menu
               evnt.preventDefault();  // disable browser context menu
               jsrp.createMenu(this, evnt).then(menu => {
                 menu.add("header:RAxisDrawable");
                 menu.add("Unzoom", () => this.ZoomStandalone());
                 this.FillAxisContextMenu(menu, "");
                 menu.show();
               });
            });

         // attributes required only for moving, has no effect for drawing
         this.draw_g.attr("x", pos.x).attr("y", pos.y)
                    .attr("width", this.vertical ? 10 : len)
                    .attr("height", this.vertical ? len : 10);

         inter.DragMoveHandler.AddDrag(this, { only_move: true, redraw: this.PositionChanged.bind(this) });

         this.draw_g.on("dblclick", () => this.ZoomStandalone());

         if (JSROOT.settings.ZoomWheel)
            this.draw_g.on("wheel", evnt => {
               evnt.stopPropagation();
               evnt.preventDefault();

               let pos = d3.pointer(evnt, this.draw_g.node()),
                   coord = this.vertical ? (1 - pos[1] / len) : pos[0] / len,
                   item = this.analyzeWheelEvent(evnt, coord);

               if (item.changed) this.ZoomStandalone(item.min, item.max);
            });

      });
   }

   /** @summary Process interactive moving of the stats box */
   RAxisPainter.prototype.PositionChanged = function() {
      let axis_x = parseInt(this.draw_g.attr("x")),
          axis_y = parseInt(this.draw_g.attr("y")),
          drawable = this.GetObject(),
          xn = axis_x / this.pad_width(),
          yn = 1 - axis_y / this.pad_height();

      drawable.fPos.fHoriz.fArr = [ xn ];
      drawable.fPos.fVert.fArr = [ yn ];

      this.WebCanvasExec("SetPos({" + xn.toFixed(4) + "," + yn.toFixed(4) + "})");
   }

   /** @summary Change axis attribute, submit changes to server and redraw axis when specified
     * @desc Arguments as redraw_mode, name1, value1, name2, value2, ... */
   RAxisPainter.prototype.ChangeAxisAttr = function(redraw_mode) {
      let changes = {}, indx = 1;
      while (indx < arguments.length - 1) {
         this.v7AttrChange(changes, arguments[indx], arguments[indx+1]);
         this.v7SetAttr(arguments[indx], arguments[indx+1]);
         indx += 2;
      }
      this.v7SendAttrChanges(changes, false); // do not invoke canvas update on the server
      if (redraw_mode === 1) {
         if (this.standalone)
            this.Redraw();
         else
            this.drawAxisAgain();
      } else if (redraw_mode)
         this.RedrawPad();
   }

   /** @summary Change axis log scale kind */
   RAxisPainter.prototype.ChangeLog = function(arg) {
      if ((this.kind == "labels") || (this.kind == 'time')) return;
      if (arg === 'toggle') arg = this.log ? 0 : 10;

      arg = parseFloat(arg);
      if (!isNaN(arg)) this.ChangeAxisAttr(2, "log", arg);
   }

   /** @summary Provide context menu for axis */
   RAxisPainter.prototype.FillAxisContextMenu = function(menu, kind) {

      if (kind) menu.add("Unzoom", () => this.frame_painter().Unzoom(kind));

      menu.add("sub:Log scale", () => this.ChangeLog('toggle'));
      menu.addchk(!this.log, "linear", 0, arg => this.ChangeLog(arg));
      menu.addchk(this.log && (this.logbase==10), "log10", 10, arg => this.ChangeLog(arg));
      menu.addchk(this.log && (this.logbase==2), "log2", 2, arg => this.ChangeLog(arg));
      menu.addchk(this.log && Math.abs(this.logbase - Math.exp(1)) < 0.1, "ln", Math.exp(1), arg => this.ChangeLog(arg));
      menu.add("endsub:");

      menu.add("sub:Ticks");
      menu.RColorMenu("color", this.ticksColor, col => this.ChangeAxisAttr(1, "ticks_color_name", col));
      menu.SizeMenu("size", 0, 0.05, 0.01, this.ticksSize/this.scaling_size, sz => this.ChangeAxisAttr(1, "ticks_size", sz));
      menu.SelectMenu("side", ["normal", "invert", "both"], this.ticksSide, side => this.ChangeAxisAttr(1, "ticks_side", side));

      menu.add("endsub:");

      if (!this.optionUnlab && this.labelsFont) {
         menu.add("sub:Labels");
         menu.SizeMenu("offset", -0.05, 0.05, 0.01, this.labelsOffset/this.scaling_size, offset => {
            this.ChangeAxisAttr(1, "labels_offset", offset);
         });
         menu.RAttrTextItems(this.labelsFont, { noangle: 1, noalign: 1 }, change => {
            this.ChangeAxisAttr(1, "labels_" + change.name, change.value);
         });
         menu.addchk(this.labelsFont.angle, "rotate", res => {
            this.ChangeAxisAttr(1, "labels_angle", res ? 180 : 0);
         })
         menu.add("endsub:");
      }

      menu.add("sub:Title", () => {
         let t = prompt("Enter axis title", this.fTitle);
         if (t!==null) this.ChangeAxisAttr(1, "title", t);
      });

      if (this.fTitle) {
         menu.SizeMenu("offset", -0.05, 0.05, 0.01, this.titleOffset/this.scaling_size, offset => {
            this.ChangeAxisAttr(1, "title_offset", offset);
         });

         menu.SelectMenu("position", ["left", "center", "right"], this.titlePos, pos => {
            this.ChangeAxisAttr(1, "title_position", pos);
         });

         menu.addchk(this.isTitleRotated(), "rotate", flag => {
            this.ChangeAxisAttr(1, "title_angle", flag ? 180 : 0);
         });

         menu.RAttrTextItems(this.titleFont, { noangle: 1, noalign: 1 }, change => {
            this.ChangeAxisAttr(1, "title_" + change.name, change.value);
         });
      }

      menu.add("endsub:");
      return true;
   }

   let drawRAxis = (divid, obj /*, opt*/) => {
      let painter = new RAxisPainter(obj);
      painter.disable_zooming = true;
      return jsrp.ensureRCanvas(painter, divid, false)
                 .then(() => painter.Redraw())
                 .then(() => painter);
   }

   // ==========================================================================================

   let ProjectAitoff2xy = (l, b) => {
      const DegToRad = Math.PI/180,
            alpha2 = (l/2)*DegToRad,
            delta  = b*DegToRad,
            r2     = Math.sqrt(2),
            f      = 2*r2/Math.PI,
            cdec   = Math.cos(delta),
            denom  = Math.sqrt(1. + cdec*Math.cos(alpha2));
      return {
         x: cdec*Math.sin(alpha2)*2.*r2/denom/f/DegToRad,
         y: Math.sin(delta)*r2/denom/f/DegToRad
      };
   }

   let ProjectMercator2xy = (l, b) => {
      const aid = Math.tan((Math.PI/2 + b/180*Math.PI)/2);
      return { x: l, y: Math.log(aid) };
   }

   let ProjectSinusoidal2xy = (l, b) => {
      return { x: l*Math.cos(b/180*Math.PI), y: b };
   }

   let ProjectParabolic2xy = (l, b) => {
      return {
         x: l*(2.*Math.cos(2*b/180*Math.PI/3) - 1),
         y: 180*Math.sin(b/180*Math.PI/3)
      };
   }

   /**
    * @summary Painter class for RFrame, main handler for interactivity
    *
    * @class
    * @memberof JSROOT
    * @extends ObjectPainter
    * @param {object} tframe - RFrame object
    * @private
    */

   function RFramePainter(tframe) {
      JSROOT.ObjectPainter.call(this, tframe);
      this.csstype = "frame";
      this.mode3d = false;
      this.xmin = this.xmax = 0; // no scale specified, wait for objects drawing
      this.ymin = this.ymax = 0; // no scale specified, wait for objects drawing
      this.axes_drawn = false;
      this.keys_handler = null;
      this.projection = 0; // different projections
      this.v7_frame = true; // indicator of v7, used in interactive part
   }

   RFramePainter.prototype = Object.create(JSROOT.ObjectPainter.prototype);

   RFramePainter.prototype.frame_painter = function() {
      return this;
   }

   /** @summary Set active flag for frame - can block some events
    * @private */
   RFramePainter.prototype.SetActive = function(/*on*/) {
      // do nothing here - key handler is handled differently
   }

   RFramePainter.prototype.GetTipName = function(append) {
      let res = JSROOT.ObjectPainter.prototype.GetTipName.call(this) || "RFrame";
      if (append) res+=append;
      return res;
   }

   RFramePainter.prototype.Shrink = function(shrink_left, shrink_right) {
      this.fX1NDC += shrink_left;
      this.fX2NDC -= shrink_right;
   }

   RFramePainter.prototype.SetLastEventPos = function(pnt) {
      // set position of last context menu event, can be
      this.fLastEventPnt = pnt;
   }

   RFramePainter.prototype.GetLastEventPos = function() {
      // return position of last event
      return this.fLastEventPnt;
   }

   RFramePainter.prototype.UpdateAttributes = function(force) {
      if ((this.fX1NDC === undefined) || (force && !this.modified_NDC)) {

         let padw = this.pad_width(), padh = this.pad_height();

         this.fX1NDC = this.v7EvalLength("margin_left", padw, JSROOT.settings.FrameNDC.fX1NDC)/padw;
         this.fY1NDC = this.v7EvalLength("margin_bottom", padh, JSROOT.settings.FrameNDC.fY1NDC)/padh;
         this.fX2NDC = 1 - this.v7EvalLength("margin_right", padw, 1-JSROOT.settings.FrameNDC.fX2NDC)/padw;
         this.fY2NDC = 1 - this.v7EvalLength("margin_top", padh, 1-JSROOT.settings.FrameNDC.fY2NDC)/padh;
      }

      if (!this.fillatt)
         this.createv7AttFill("fill_");

      this.createv7AttLine("border_");
   }

   /** @summary Returns coordinates transformation func */
   RFramePainter.prototype.GetProjectionFunc = function() {
      switch (this.projection) {
         case 1: return ProjectAitoff2xy;
         case 2: return ProjectMercator2xy;
         case 3: return ProjectSinusoidal2xy;
         case 4: return ProjectParabolic2xy;
      }
   }

   /** @summary Rcalculate frame ranges using specified projection functions
     * @desc Not yet used in v7 */
   RFramePainter.prototype.RecalculateRange = function(Proj) {
      this.projection = Proj || 0;

      if ((this.projection == 2) && ((this.scale_ymin <= -90 || this.scale_ymax >=90))) {
         console.warn("Mercator Projection", "Latitude out of range", this.scale_ymin, this.scale_ymax);
         this.projection = 0;
      }

      let func = this.GetProjectionFunc();
      if (!func) return;

      let pnts = [ func(this.scale_xmin, this.scale_ymin),
                   func(this.scale_xmin, this.scale_ymax),
                   func(this.scale_xmax, this.scale_ymax),
                   func(this.scale_xmax, this.scale_ymin) ];
      if (this.scale_xmin<0 && this.scale_xmax>0) {
         pnts.push(func(0, this.scale_ymin));
         pnts.push(func(0, this.scale_ymax));
      }
      if (this.scale_ymin<0 && this.scale_ymax>0) {
         pnts.push(func(this.scale_xmin, 0));
         pnts.push(func(this.scale_xmax, 0));
      }

      this.original_xmin = this.scale_xmin;
      this.original_xmax = this.scale_xmax;
      this.original_ymin = this.scale_ymin;
      this.original_ymax = this.scale_ymax;

      this.scale_xmin = this.scale_xmax = pnts[0].x;
      this.scale_ymin = this.scale_ymax = pnts[0].y;

      for (let n = 1; n < pnts.length; ++n) {
         this.scale_xmin = Math.min(this.scale_xmin, pnts[n].x);
         this.scale_xmax = Math.max(this.scale_xmax, pnts[n].x);
         this.scale_ymin = Math.min(this.scale_ymin, pnts[n].y);
         this.scale_ymax = Math.max(this.scale_ymax, pnts[n].y);
      }
   }

   /** @summary Draw frame grids */
   RFramePainter.prototype.DrawGrids = function() {
      // grid can only be drawn by first painter

      let layer = this.svg_frame().select(".grid_layer");

      layer.selectAll(".xgrid").remove();
      layer.selectAll(".ygrid").remove();

      let h = this.frame_height(),
          w = this.frame_width(),
          gridx = this.v7EvalAttr("gridx", false),
          gridy = this.v7EvalAttr("gridy", false),
          grid_style = JSROOT.gStyle.fGridStyle,
          grid_color = (JSROOT.gStyle.fGridColor > 0) ? this.get_color(JSROOT.gStyle.fGridColor) : "black";

      if ((grid_style < 0) || (grid_style >= jsrp.root_line_styles.length)) grid_style = 11;

      // add a grid on x axis, if the option is set
      if (this.x_handle && gridx) {
         let grid = "";
         for (let n=0;n<this.x_handle.ticks.length;++n)
            if (this.swap_xy)
               grid += "M0,"+(h+this.x_handle.ticks[n])+"h"+w;
            else
               grid += "M"+this.x_handle.ticks[n]+",0v"+h;

         if (grid.length > 0)
          layer.append("svg:path")
               .attr("class", "xgrid")
               .attr("d", grid)
               .style('stroke',grid_color).style("stroke-width",JSROOT.gStyle.fGridWidth)
               .style("stroke-dasharray", jsrp.root_line_styles[grid_style]);
      }

      // add a grid on y axis, if the option is set
      if (this.y_handle && gridy) {
         let grid = "";
         for (let n=0;n<this.y_handle.ticks.length;++n)
            if (this.swap_xy)
               grid += "M"+this.y_handle.ticks[n]+",0v"+h;
            else
               grid += "M0,"+(h+this.y_handle.ticks[n])+"h"+w;

         if (grid.length > 0)
          layer.append("svg:path")
               .attr("class", "ygrid")
               .attr("d", grid)
               .style('stroke',grid_color).style("stroke-width",JSROOT.gStyle.fGridWidth)
               .style("stroke-dasharray", jsrp.root_line_styles[grid_style]);
      }
   }

   /** @summary Converts "raw" axis value into text */
   RFramePainter.prototype.AxisAsText = function(axis, value) {
      let handle = this[axis+"_handle"];

      if (handle)
         return handle.AxisAsText(value, JSROOT.settings[axis.toUpperCase() + "ValuesFormat"]);

      return value.toPrecision(4);
   }

   /** @summary Set axes ranges for drawing, check configured attributes if range already specified */
   RFramePainter.prototype.SetAxesRanges = function(xaxis, xmin, xmax, yaxis, ymin, ymax, zaxis, zmin, zmax) {
      if (this.axes_drawn) return;

      this.xaxis = xaxis;
      this.yaxis = yaxis;
      this.zaxis = zaxis;

      let min, max;

      if (this.xmin == this.xmax) {
         min = this.v7EvalAttr("x_min");
         max = this.v7EvalAttr("x_max");

         if (min !== undefined) xmin = min;
         if (max !== undefined) xmax = max;

         if (xmin < xmax) {
            this.xmin = xmin;
            this.xmax = xmax;
         }

         if ((this.zoom_xmin == this.zoom_xmax) && !this.zoomChangedInteractive("x")) {
            min = this.v7EvalAttr("x_zoommin");
            max = this.v7EvalAttr("x_zoommax");

            if ((min !== undefined) || (max !== undefined)) {
               this.zoom_xmin = (min === undefined) ? this.xmin : min;
               this.zoom_xmax = (max === undefined) ? this.xmax : max;
            }
         }
      }

      if (this.ymin == this.ymax) {
         min = this.v7EvalAttr("y_min");
         max = this.v7EvalAttr("y_max");

         if (min !== undefined) ymin = min;
         if (max !== undefined) ymax = max;

         if (ymin < ymax) {
            this.ymin = ymin;
            this.ymax = ymax;
         }

         if ((this.zoom_ymin == this.zoom_ymax) && !this.zoomChangedInteractive("y")) {
            min = this.v7EvalAttr("y_zoommin");
            max = this.v7EvalAttr("y_zoommax");

            if ((min !== undefined) || (max !== undefined)) {
               this.zoom_ymin = (min === undefined) ? this.ymin : min;
               this.zoom_ymax = (max === undefined) ? this.ymax : max;
            }
         }
      }

      if (this.zmin == this.zmax) {
         min = this.v7EvalAttr("z_min");
         max = this.v7EvalAttr("z_max");

         if (min !== undefined) zmin = min;
         if (max !== undefined) zmax = max;

         if (zmin < zmax) {
            this.zmin = zmin;
            this.zmax = zmax;
         }

         if ((this.zoom_zmin == this.zoom_zmax) && !this.zoomChangedInteractive("z")) {
            min = this.v7EvalAttr("z_zoommin");
            max = this.v7EvalAttr("z_zoommax");

            if ((min !== undefined) || (max !== undefined)) {
               this.zoom_zmin = (min === undefined) ? this.zmin : min;
               this.zoom_zmax = (max === undefined) ? this.zmax : max;
            }
         }
      }
   }

   /** @summary Draw configured axes on the frame
     * @desc axes can be drawn only for main histogram  */
   RFramePainter.prototype.DrawAxes = function() {

      if (this.axes_drawn || (this.xmin==this.xmax) || (this.ymin==this.ymax))
         return Promise.resolve(this.axes_drawn);

      this.CleanupAxes();

      this.swap_xy = false;
      let ticksx = this.v7EvalAttr("ticksx", 1),
          ticksy = this.v7EvalAttr("ticksy", 1),
          sidex = 1, sidey = 1;

      // ticksx = 2; ticksy = 2;

      if (this.v7EvalAttr("swapx", false)) sidex = -1;
      if (this.v7EvalAttr("swapy", false)) sidey = -1;

      let w = this.frame_width(), h = this.frame_height();

      this.scale_xmin = this.xmin;
      this.scale_xmax = this.xmax;

      this.scale_ymin = this.ymin;
      this.scale_ymax = this.ymax;

      if (this.zoom_xmin != this.zoom_xmax) {
         this.scale_xmin = this.zoom_xmin;
         this.scale_xmax = this.zoom_xmax;
      }

      if (this.zoom_ymin != this.zoom_ymax) {
         this.scale_ymin = this.zoom_ymin;
         this.scale_ymax = this.zoom_ymax;
      }

      this.RecalculateRange(0);

      this.x_handle = new RAxisPainter(this, this.xaxis, "x_");
      this.x_handle.SetDivId(this.divid, -1);
      this.x_handle.snapid = this.snapid;

      this.y_handle = new RAxisPainter(this, this.yaxis, "y_");
      this.y_handle.SetDivId(this.divid, -1);
      this.y_handle.snapid = this.snapid;

      this.z_handle = new RAxisPainter(this, this.zaxis, "z_");
      this.z_handle.SetDivId(this.divid, -1);
      this.z_handle.snapid = this.snapid;

      this.x_handle.ConfigureAxis("xaxis", this.xmin, this.xmax, this.scale_xmin, this.scale_xmax, false, [0,w], w, { reverse: false });
      this.x_handle.AssignFrameMembers(this,"x");

      this.y_handle.ConfigureAxis("yaxis", this.ymin, this.ymax, this.scale_ymin, this.scale_ymax, true, [h,0], -h, { reverse: false });
      this.y_handle.AssignFrameMembers(this,"y");

      // only get basic properties like log scale
      this.z_handle.ConfigureZAxis("zaxis", this);

      let layer = this.svg_frame().select(".axis_layer");

      this.x_handle.has_obstacle = false;

      let draw_horiz = this.swap_xy ? this.y_handle : this.x_handle,
          draw_vertical = this.swap_xy ? this.x_handle : this.y_handle,
          disable_axis_draw = false;

      if (!disable_axis_draw) {
         let pp = this.pad_painter();
         if (pp && pp._fast_drawing) disable_axis_draw = true;
      }

      if (!disable_axis_draw) {
         let promise1 = draw_horiz.drawAxis(layer, (sidex > 0) ? `translate(0,${h})` : "", sidex);

         let promise2 = draw_vertical.drawAxis(layer, (sidey > 0) ? `translate(0,${h})` : `translate(${w},${h})`, sidey);

         return Promise.all([promise1, promise2]).then(() => {

            let again = [];
            if (ticksx > 1)
               again.push(draw_horiz.drawAxisOtherPlace(layer, (sidex < 0) ? `translate(0,${h})` : "", -sidex, ticksx == 2));

            if (ticksy > 1)
               again.push(draw_vertical.drawAxisOtherPlace(layer, (sidey < 0) ? `translate(0,${h})` : `translate(${w},${h})`, -sidey, ticksy == 2));

             return Promise.all(again);
         }).then(() => {
             this.DrawGrids();
             this.axes_drawn = true;
             return true;
         });
      }

      this.axes_drawn = true;

      return Promise.resolve(true);
   }

   /** @summary function called at the end of resize of frame
     * @desc Used to update attributes on the server
     * @private */
   RFramePainter.prototype.SizeChanged = function() {

      let changes = {};
      this.v7AttrChange(changes, "margin_left", this.fX1NDC);
      this.v7AttrChange(changes, "margin_bottom", this.fY1NDC);
      this.v7AttrChange(changes, "margin_right", 1 - this.fX2NDC);
      this.v7AttrChange(changes, "margin_top", 1 - this.fY2NDC);
      this.v7SendAttrChanges(changes, false); // do not invoke canvas update on the server

      this.RedrawPad();
   }

   /** @summary Remove all axes drawings */
   RFramePainter.prototype.CleanupAxes = function() {
      // remove all axes drawings
      if (this.x_handle) {
         this.x_handle.Cleanup();
         delete this.x_handle;
      }

      if (this.y_handle) {
         this.y_handle.Cleanup();
         delete this.y_handle;
      }

      if (this.z_handle) {
         this.z_handle.Cleanup();
         delete this.z_handle;
      }

      if (this.draw_g) {
         this.draw_g.select(".grid_layer").selectAll("*").remove();
         this.draw_g.select(".axis_layer").selectAll("*").remove();
      }
      this.axes_drawn = false;

      delete this.grx;
      delete this.gry;
      delete this.grz;
   }

   /** @summary Removes all drawn elements of the frame
     * @private */
   RFramePainter.prototype.CleanFrameDrawings = function() {
      // cleanup all 3D drawings if any
      if (typeof this.create3DScene === 'function')
         this.create3DScene(-1);

      this.CleanupAxes();

      this.xmin = this.xmax = 0;
      this.ymin = this.ymax = 0;
      this.zmin = this.zmax = 0;

      this.zoom_xmin = this.zoom_xmax = 0;
      this.zoom_ymin = this.zoom_ymax = 0;
      this.zoom_zmin = this.zoom_zmax = 0;

      this.scale_xmin = this.scale_xmax = 0;
      this.scale_ymin = this.scale_ymax = 0;
      this.scale_zmin = this.scale_zmax = 0;

      if (this.draw_g) {
         this.draw_g.select(".main_layer").selectAll("*").remove();
         this.draw_g.select(".upper_layer").selectAll("*").remove();
      }
   }

   /** @summary Fully cleanup frame
     * @private */
   RFramePainter.prototype.Cleanup = function() {

      this.CleanFrameDrawings();

      if (this.draw_g) {
         this.draw_g.selectAll("*").remove();
         this.draw_g.on("mousedown", null)
                    .on("dblclick", null)
                    .on("wheel", null)
                    .on("contextmenu", null)
                    .property('interactive_set', null);
      }

      if (this.keys_handler) {
         window.removeEventListener('keydown', this.keys_handler, false);
         this.keys_handler = null;
      }

      delete this.xaxis;
      delete this.yaxis;
      delete this.zaxis;

      this.draw_g = null;
      delete this._click_handler;
      delete this._dblclick_handler;

      JSROOT.ObjectPainter.prototype.Cleanup.call(this);
   }

   /** @summary Redraw frame
     * @private */
   RFramePainter.prototype.Redraw = function() {

      let pp = this.pad_painter();
      if (pp) pp.frame_painter_ref = this;

      // first update all attributes from objects
      this.UpdateAttributes();

      let width = this.pad_width(),
          height = this.pad_height(),
          lm = Math.round(width * this.fX1NDC),
          w = Math.round(width * (this.fX2NDC - this.fX1NDC)),
          tm = Math.round(height * (1 - this.fY2NDC)),
          h = Math.round(height * (this.fY2NDC - this.fY1NDC)),
          rotate = false, fixpos = false,
          trans = "translate(" + lm + "," + tm + ")";

      if (pp && pp.options) {
         if (pp.options.RotateFrame) rotate = true;
         if (pp.options.FixFrame) fixpos = true;
      }

      if (rotate) {
         trans += " rotate(-90) " + "translate(" + -h + ",0)";
         let d = w; w = h; h = d;
      }

      // update values here to let access even when frame is not really updated
      this._frame_x = lm;
      this._frame_y = tm;
      this._frame_width = w;
      this._frame_height = h;
      this._frame_rotate = rotate;
      this._frame_fixpos = fixpos;

      if (this.mode3d) return; // no need for real draw in mode3d

      // this is svg:g object - container for every other items belonging to frame
      this.draw_g = this.svg_layer("primitives_layer").select(".root_frame");

      let top_rect, main_svg;

      if (this.draw_g.empty()) {

         let layer = this.svg_layer("primitives_layer");

         this.draw_g = layer.append("svg:g").attr("class", "root_frame");

         this.draw_g.append("svg:title").text("");

         top_rect = this.draw_g.append("svg:rect");

         // append for the moment three layers - for drawing and axis
         this.draw_g.append('svg:g').attr('class','grid_layer');

         main_svg = this.draw_g.append('svg:svg')
                           .attr('class','main_layer')
                           .attr("x", 0)
                           .attr("y", 0)
                           .attr('overflow', 'hidden');

         this.draw_g.append('svg:g').attr('class','axis_layer');
         this.draw_g.append('svg:g').attr('class','upper_layer');
      } else {
         top_rect = this.draw_g.select("rect");
         main_svg = this.draw_g.select(".main_layer");

         if (this.axes_drawn) {
            let xmin = this.v7EvalAttr("x_zoommin"),
                xmax = this.v7EvalAttr("x_zoommax"),
                ymin = this.v7EvalAttr("y_zoommin"),
                ymax = this.v7EvalAttr("y_zoommax");

            console.log('TODO: RFrame zooming update', xmin, xmax, ymin, ymax);
         }
      }

      this.axes_drawn = false;

      this.draw_g.attr("transform", trans);

      top_rect.attr("x", 0)
              .attr("y", 0)
              .attr("width", w)
              .attr("height", h)
              .call(this.fillatt.func)
              .call(this.lineatt.func);

      main_svg.attr("width", w)
              .attr("height", h)
              .attr("viewBox", "0 0 " + w + " " + h);

      if (JSROOT.BatchMode) return;

      JSROOT.require(['interactive']).then(inter => {
         top_rect.attr("pointer-events", "visibleFill");  // let process mouse events inside frame
         inter.FrameInteractive.assign(this);
         this.BasicInteractive();
      });
   }

   /** @summary Returns frame rectangle plus extra info for hint display */
   RFramePainter.prototype.GetFrameRect = function() {
      return {
         x: this.frame_x(),
         y: this.frame_y(),
         width: this.frame_width(),
         height: this.frame_height(),
         transform: this.draw_g ? this.draw_g.attr("transform") : "",
         hint_delta_x: 0,
         hint_delta_y: 0
      }
   }

   /** @summary Returns palette associated with frame. Either from existing palette painter or just default palette */
   RFramePainter.prototype.GetPalette = function() {
      let pp = this.FindPainterFor(undefined, undefined, "ROOT::Experimental::RPaletteDrawable");

      if (pp) return pp.GetPalette();

      if (!this.fDfltPalette) {
         this.fDfltPalette = {
            _typename : "ROOT::Experimental::RPalette",
            fColors : [{ fOrdinal : 0,     fColor : { fRGBA : [53, 42, 135] } },
                       { fOrdinal : 0.125, fColor : { fRGBA : [15, 92, 221] } },
                       { fOrdinal : 0.25,  fColor : { fRGBA : [20, 129, 214] } },
                       { fOrdinal : 0.375, fColor : { fRGBA : [6, 164, 202] } },
                       { fOrdinal : 0.5,   fColor : { fRGBA : [46, 183, 164] } },
                       { fOrdinal : 0.625, fColor : { fRGBA : [135, 191, 119] } },
                       { fOrdinal : 0.75,  fColor : { fRGBA : [209, 187, 89] } },
                       { fOrdinal : 0.875, fColor : { fRGBA : [254, 200, 50] } },
                       { fOrdinal : 1,     fColor : { fRGBA : [249, 251, 14] } }],
             fInterpolate : true,
             fNormalized : true
         };
         JSROOT.addMethods(this.fDfltPalette, "ROOT::Experimental::RPalette");
      }

      return this.fDfltPalette;
   }

   RFramePainter.prototype.configureUserClickHandler = function(handler) {
      this._click_handler = handler && (typeof handler == 'function') ? handler : null;
   }

   RFramePainter.prototype.configureUserDblclickHandler = function(handler) {
      this._dblclick_handler = handler && (typeof handler == 'function') ? handler : null;
   }

   /** @summary function can be used for zooming into specified range
     * @desc if both limits for each axis 0 (like xmin==xmax==0), axis will be unzoomed */
   RFramePainter.prototype.Zoom = function(xmin, xmax, ymin, ymax, zmin, zmax) {

      // disable zooming when axis conversion is enabled
      if (this.projection) return false;

      if (xmin==="x") { xmin = xmax; xmax = ymin; ymin = undefined; } else
      if (xmin==="y") { ymax = ymin; ymin = xmax; xmin = xmax = undefined; } else
      if (xmin==="z") { zmin = xmax; zmax = ymin; xmin = xmax = ymin = undefined; }

      let zoom_x = (xmin !== xmax), zoom_y = (ymin !== ymax), zoom_z = (zmin !== zmax),
          unzoom_x = false, unzoom_y = false, unzoom_z = false;

      if (zoom_x) {
         let cnt = 0;
         if (xmin <= this.xmin) { xmin = this.xmin; cnt++; }
         if (xmax >= this.xmax) { xmax = this.xmax; cnt++; }
         if (cnt === 2) { zoom_x = false; unzoom_x = true; }
      } else {
         unzoom_x = (xmin === xmax) && (xmin === 0);
      }

      if (zoom_y) {
         let cnt = 0;
         if (ymin <= this.ymin) { ymin = this.ymin; cnt++; }
         if (ymax >= this.ymax) { ymax = this.ymax; cnt++; }
         if (cnt === 2) { zoom_y = false; unzoom_y = true; }
      } else {
         unzoom_y = (ymin === ymax) && (ymin === 0);
      }

      if (zoom_z) {
         let cnt = 0;
         // if (this.logz && this.ymin_nz && this.Dimension()===2) main_zmin = 0.3*this.ymin_nz;
         if (zmin <= this.zmin) { zmin = this.zmin; cnt++; }
         if (zmax >= this.zmax) { zmax = this.zmax; cnt++; }
         if (cnt === 2) { zoom_z = false; unzoom_z = true; }
      } else {
         unzoom_z = (zmin === zmax) && (zmin === 0);
      }

      let changed = false, changes = {},
          r_x = "", r_y = "", r_z = "",
         req = {
         _typename: "ROOT::Experimental::RFrame::RUserRanges",
         values: [0, 0, 0, 0, 0, 0],
         flags: [false, false, false, false, false, false]
      };

      // first process zooming (if any)
      if (zoom_x || zoom_y || zoom_z)
         this.forEachPainter(obj => {
            if (typeof obj.canZoomInside != 'function') return;
            if (zoom_x && obj.canZoomInside("x", xmin, xmax)) {
               this.zoom_xmin = xmin;
               this.zoom_xmax = xmax;
               changed = true; r_x = "0";
               zoom_x = false;
               this.v7AttrChange(changes, "x_zoommin", xmin);
               this.v7AttrChange(changes, "x_zoommax", xmax);
               req.values[0] = xmin; req.values[1] = xmax;
               req.flags[0] = req.flags[1] = true;
            }
            if (zoom_y && obj.canZoomInside("y", ymin, ymax)) {
               this.zoom_ymin = ymin;
               this.zoom_ymax = ymax;
               changed = true; r_y = "1";
               zoom_y = false;
               this.v7AttrChange(changes, "y_zoommin", ymin);
               this.v7AttrChange(changes, "y_zoommax", ymax);
               req.values[2] = ymin; req.values[3] = ymax;
               req.flags[2] = req.flags[3] = true;
            }
            if (zoom_z && obj.canZoomInside("z", zmin, zmax)) {
               this.zoom_zmin = zmin;
               this.zoom_zmax = zmax;
               changed = true; r_z = "2";
               zoom_z = false;
               this.v7AttrChange(changes, "z_zoommin", zmin);
               this.v7AttrChange(changes, "z_zoommax", zmax);
               req.values[4] = zmin; req.values[5] = zmax;
               req.flags[4] = req.flags[5] = true;
            }
         });

      // and process unzoom, if any
      if (unzoom_x || unzoom_y || unzoom_z) {
         if (unzoom_x) {
            if (this.zoom_xmin !== this.zoom_xmax) { changed = true; r_x = "0"; }
            this.zoom_xmin = this.zoom_xmax = 0;
            this.v7AttrChange(changes, "x_zoommin", null);
            this.v7AttrChange(changes, "x_zoommax", null);
            req.values[0] = req.values[1] = -1;
         }
         if (unzoom_y) {
            if (this.zoom_ymin !== this.zoom_ymax) { changed = true; r_y = "1"; }
            this.zoom_ymin = this.zoom_ymax = 0;
            this.v7AttrChange(changes, "y_zoommin", null);
            this.v7AttrChange(changes, "y_zoommax", null);
            req.values[2] = req.values[3] = -1;
         }
         if (unzoom_z) {
            if (this.zoom_zmin !== this.zoom_zmax) { changed = true; r_z = "2"; }
            this.zoom_zmin = this.zoom_zmax = 0;
            this.v7AttrChange(changes, "z_zoommin", null);
            this.v7AttrChange(changes, "z_zoommax", null);
            req.values[4] = req.values[5] = -1;
         }
      }

      if (this.v7CommMode() == JSROOT.v7.CommMode.kNormal)
         this.v7SubmitRequest("zoom", { _typename: "ROOT::Experimental::RFrame::RZoomRequest", ranges: req });

      // this.v7SendAttrChanges(changes);

      if (changed)
         this.InteractiveRedraw("pad", "zoom" + r_x + r_y + r_z);

      return changed;
   }

   RFramePainter.prototype.IsAxisZoomed = function(axis) {
      return this['zoom_'+axis+'min'] !== this['zoom_'+axis+'max'];
   }

   /** @summary Unzoom specified axes */
   RFramePainter.prototype.Unzoom = function(dox, doy, doz) {
      if (typeof dox === 'undefined') { dox = true; doy = true; doz = true; } else
      if (typeof dox === 'string') { doz = dox.indexOf("z")>=0; doy = dox.indexOf("y")>=0; dox = dox.indexOf("x")>=0; }

      let changed = this.Zoom(dox ? 0 : undefined, dox ? 0 : undefined,
                              doy ? 0 : undefined, doy ? 0 : undefined,
                              doz ? 0 : undefined, doz ? 0 : undefined);

      if (changed && dox) this.zoomChangedInteractive("x", "unzoom");
      if (changed && doy) this.zoomChangedInteractive("y", "unzoom");
      if (changed && doz) this.zoomChangedInteractive("z", "unzoom");

      return changed;
   }

   /** @summary Mark/check if zoom for specific axis was changed interactively
     * @private */
   RFramePainter.prototype.zoomChangedInteractive = function(axis, value) {
      if (axis == 'reset') {
         this.zoom_changed_x = this.zoom_changed_y = this.zoom_changed_z = undefined;
         return;
      }
      if (!axis || axis == 'any')
         return this.zoom_changed_x || this.zoom_changed_y  || this.zoom_changed_z;

      if ((axis !== 'x') && (axis !== 'y') && (axis !== 'z')) return;

      let fld = "zoom_changed_" + axis;
      if (value === undefined) return this[fld];

      if (value === 'unzoom') {
         // special handling of unzoom
         if (this[fld])
            delete this[fld];
         else
            this[fld] = true;
         return;
      }

      if (value) this[fld] = true;
   }


   /** @summary Fill menu for frame when server is not there */
   RFramePainter.prototype.FillObjectOfflineMenu = function(menu, kind) {
      if ((kind!="x") && (kind!="y")) return;

      menu.add("Unzoom", this.Unzoom.bind(this, kind));

      //if (this[kind+"_kind"] == "normal")
      //   menu.addchk(this["log"+kind], "SetLog"+kind, this.ToggleLog.bind(this, kind));

      // here should be all axes attributes in offline
   }

   /** @summary Fill context menu */
   RFramePainter.prototype.FillContextMenu = function(menu, kind, /* obj */) {

      // when fill and show context menu, remove all zooming

      if ((kind=="x") || (kind=="y")) {
         let handle = this[kind+"_handle"];
         if (!handle) return false;
         menu.add("header: " + kind.toUpperCase() + " axis");
         return handle.FillAxisContextMenu(menu, kind);
      }

      let alone = menu.size()==0;

      if (alone)
         menu.add("header:Frame");
      else
         menu.add("separator");

      if (this.zoom_xmin !== this.zoom_xmax)
         menu.add("Unzoom X", this.Unzoom.bind(this,"x"));
      if (this.zoom_ymin !== this.zoom_ymax)
         menu.add("Unzoom Y", this.Unzoom.bind(this,"y"));
      if (this.zoom_zmin !== this.zoom_zmax)
         menu.add("Unzoom Z", this.Unzoom.bind(this,"z"));
      menu.add("Unzoom all", this.Unzoom.bind(this,"xyz"));

      // menu.addchk(this.logx, "SetLogx", this.ToggleLog.bind(this,"x"));
      // menu.addchk(this.logy, "SetLogy", this.ToggleLog.bind(this,"y"));
      // if (this.Dimension() == 2)
      //   menu.addchk(pad.fLogz, "SetLogz", this.ToggleLog.bind(main,"z"));
      menu.add("separator");


      menu.addchk(this.IsTooltipAllowed(), "Show tooltips", function() {
         this.SetTooltipAllowed("toggle");
      });
      menu.AddAttributesMenu(this, alone ? "" : "Frame ");
      menu.add("separator");
      menu.add("Save as frame.png", function() { this.pad_painter().SaveAs("png", 'frame', 'frame.png'); });
      menu.add("Save as frame.svg", function() { this.pad_painter().SaveAs("svg", 'frame', 'frame.svg'); });

      return true;
   }

   /** @summary Convert graphical coordinate into axis value */
   RFramePainter.prototype.RevertAxis = function(axis, pnt) {
      let handle = this[axis+"_handle"];
      return handle ? handle.RevertPoint(pnt) : 0;
   }

   /** @summary Show axis status message
   * @desc method called normally when mouse enter main object element */
   RFramePainter.prototype.ShowAxisStatus = function(axis_name, evnt) {
      // method called normally when mouse enter main object element

      let status_func = this.GetShowStatusFunc();

      if (typeof status_func != "function") return;

      let taxis = null, hint_name = axis_name, hint_title = "TAxis",
          m = d3.pointer(evnt, this.svg_frame().node()), id = (axis_name=="x") ? 0 : 1;

      if (taxis) { hint_name = taxis.fName; hint_title = taxis.fTitle || "histogram TAxis object"; }

      if (this.swap_xy) id = 1-id;

      let axis_value = this.RevertAxis(axis_name, m[id]);

      status_func(hint_name, hint_title, axis_name + " : " + this.AxisAsText(axis_name, axis_value), m[0]+","+m[1]);
   }

   /** @summary Add interactive keys handlers
    * @private */
   RFramePainter.prototype.AddKeysHandler = function() {
      if (JSROOT.BatchMode) return;
      JSROOT.require(['interactive']).then(inter => {
         inter.FrameInteractive.assign(this);
         this.AddKeysHandler();
      });
   }

   /** @summary Add interactive functionality to the frame
    * @private */
   RFramePainter.prototype.AddInteractive = function() {

      if (JSROOT.BatchMode || (!JSROOT.settings.Zooming && !JSROOT.settings.ContextMenu))
         return Promise.resolve(true);

      return JSROOT.require(['interactive']).then(inter => {
         inter.FrameInteractive.assign(this);
         return this.AddInteractive();
      });
   }

   /** @summary Set selected range back to pad object - to be implemented */
   RFramePainter.prototype.SetRootPadRange = function(/* pad, is3d */) {
      // TODO: change of pad range and send back to root application
   }

   /** @summary Toggle log scale on the specified axes */
   RFramePainter.prototype.ToggleLog = function(axis) {
      let handle = this[axis+"_handle"];
      if (handle) handle.ChangeLog('toggle');
   }

   function drawRFrame(divid, obj, opt) {
      let p = new RFramePainter(obj);
      if (opt == "3d") p.mode3d = true;
      return jsrp.ensureRCanvas(p, divid, false).then(() => {
         p.Redraw();
         return p;
      });
   }

   // ===========================================================================

   function RPadPainter(pad, iscan) {
      JSROOT.ObjectPainter.call(this, pad);
      this.csstype = "pad";
      this.pad = pad;
      this.iscan = iscan; // indicate if working with canvas
      this.this_pad_name = "";
      if (!this.iscan && (pad !== null)) {
         if (pad.fObjectID)
            this.this_pad_name = "pad" + pad.fObjectID; // use objectid as padname
         else
            this.this_pad_name = "ppp" + JSROOT._.id_counter++; // artificical name
      }
      this.painters = []; // complete list of all painters in the pad
      this.has_canvas = true;
   }

   RPadPainter.prototype = Object.create(JSROOT.ObjectPainter.prototype);

   /** @summary Returns SVG element for the specified pad (or itself)
    * @private */
   RPadPainter.prototype.svg_pad = function(pad_name) {
      if (pad_name === undefined)
         pad_name = this.this_pad_name;
      return JSROOT.ObjectPainter.prototype.svg_pad.call(this, pad_name);
   }

   /** @summary cleanup only pad itself, all child elements will be collected and cleanup separately */
   RPadPainter.prototype.Cleanup = function() {

      for (let k = 0; k < this.painters.length; ++k)
         this.painters[k].Cleanup();

      let svg_p = this.svg_pad();
      if (!svg_p.empty()) {
         svg_p.property('pad_painter', null);
         svg_p.property('mainpainter', null);
         if (!this.iscan) svg_p.remove();
      }

      delete this.frame_painter_ref;
      delete this.pads_cache;
      this.painters = [];
      this.pad = null;
      this.draw_object = null;
      this.pad_frame = null;
      this.this_pad_name = undefined;
      this.has_canvas = false;

      jsrp.selectActivePad({ pp: this, active: false });

      JSROOT.ObjectPainter.prototype.Cleanup.call(this);
   }

   /** @summary Returns frame painter inside the pad
    * @private */
   RPadPainter.prototype.frame_painter = function() {
      return this.frame_painter_ref;
   }

   /** @summary Cleanup primitives from pad - selector lets define which painters to remove
    * @private */
   RPadPainter.prototype.CleanPrimitives = function(selector) {
      if (!selector || (typeof selector !== 'function')) return;

      for (let k = this.painters.length-1; k >= 0; --k)
         if (selector(this.painters[k])) {
            this.painters[k].Cleanup();
            this.painters.splice(k, 1);
         }
   }

   /** @summary Call function for each painter in pad
     * @param {function} userfunc - function to call
     * @param {string} kind - "all" for all objects (default), "pads" only pads and subpads, "objects" only for object in current pad
     * @private */
   RPadPainter.prototype.forEachPainterInPad = function(userfunc, kind) {
      if (!kind) kind = "all";
      if (kind!="objects") userfunc(this);
      for (let k = 0; k < this.painters.length; ++k) {
         let sub = this.painters[k];
         if (typeof sub.forEachPainterInPad === 'function') {
            if (kind!="objects") sub.forEachPainterInPad(userfunc, kind);
         } else if (kind != "pads") userfunc(sub);
      }
   }

   /** @summary register for pad events receiver
     * @desc in pad painter, while pad may be drawn without canvas
     * @private */
   RPadPainter.prototype.RegisterForPadEvents = function(receiver) {
      this.pad_events_receiver = receiver;
   }

   /** @summary Generate pad events, normally handled by GED
     * @desc in pad painter, while pad may be drawn without canvas
     * @private */
   RPadPainter.prototype.PadEvent = function(_what, _padpainter, _painter, _position, _place) {
      if ((_what == "select") && (typeof this.selectActivePad == 'function'))
         this.selectActivePad(_padpainter, _painter, _position);

      if (this.pad_events_receiver)
         this.pad_events_receiver({ what: _what, padpainter:  _padpainter, painter: _painter, position: _position, place: _place });
   }

   /** @summary method redirect call to pad events receiver */
   RPadPainter.prototype.SelectObjectPainter = function(_painter, pos, _place) {

      let istoppad = (this.iscan || !this.has_canvas),
          canp = istoppad ? this : this.canv_painter();

      if (_painter === undefined) _painter = this;

      if (pos && !istoppad)
         pos = jsrp.getAbsPosInCanvas(this.svg_pad(), pos);

      jsrp.selectActivePad({ pp: this, active: true });

      canp.PadEvent("select", this, _painter, pos, _place);
   }

   /** @summary Called by framework when pad is supposed to be active and get focus
    * @private */
   RPadPainter.prototype.SetActive = function(on) {
      let fp = this.frame_painter();
      if (fp && (typeof fp.SetActive == 'function')) fp.SetActive(on);
   }

   /** @summary Create SVG element for the canvas */
   RPadPainter.prototype.CreateCanvasSvg = function(check_resize, new_size) {

      let factor = null, svg = null, lmt = 5, rect = null, btns;

      if (check_resize > 0) {

         if (this._fixed_size) return (check_resize > 1); // flag used to force re-drawing of all subpads

         svg = this.svg_canvas();

         if (svg.empty()) return false;

         factor = svg.property('height_factor');

         rect = this.testMainResize(check_resize, null, factor);

         if (!rect.changed) return false;

         if (!JSROOT.BatchMode)
            btns = this.svg_layer("btns_layer");

      } else {

         let render_to = this.select_main();

         if (render_to.style('position')=='static')
            render_to.style('position','relative');

         svg = render_to.append("svg")
             .attr("class", "jsroot root_canvas")
             .property('pad_painter', this) // this is custom property
             .property('mainpainter', null) // this is custom property
             .property('current_pad', "") // this is custom property
             .property('redraw_by_resize', false); // could be enabled to force redraw by each resize

         svg.append("svg:title").text("ROOT canvas");
         let frect = svg.append("svg:rect").attr("class","canvas_fillrect")
                               .attr("x",0).attr("y",0);
         if (!JSROOT.BatchMode)
            frect.style("pointer-events", "visibleFill")
                 .on("dblclick", this.EnlargePad.bind(this))
                 .on("click", this.SelectObjectPainter.bind(this, this, null))
                 .on("mouseenter", this.ShowObjectStatus.bind(this));

         svg.append("svg:g").attr("class","primitives_layer");
         svg.append("svg:g").attr("class","info_layer");
         if (!JSROOT.BatchMode)
            btns = svg.append("svg:g")
                      .attr("class","btns_layer")
                      .property('leftside', JSROOT.settings.ToolBarSide == 'left')
                      .property('vertical', JSROOT.settings.ToolBarVert);

         if (JSROOT.settings.ContextMenu && !JSROOT.BatchMode)
            svg.select(".canvas_fillrect").on("contextmenu", this.PadContextMenu.bind(this));

         factor = 0.66;
         if (this.pad && this.pad.fWinSize[0] && this.pad.fWinSize[1]) {
            factor = this.pad.fWinSize[1] / this.pad.fWinSize[0];
            if ((factor < 0.1) || (factor > 10)) factor = 0.66;
         }

         if (this._fixed_size) {
            render_to.style("overflow","auto");
            rect = { width: this.pad.fWinSize[0], height: this.pad.fWinSize[1] };
            if (!rect.width || !rect.height)
               rect = jsrp.getElementRect(render_to);
         } else {
            rect = this.testMainResize(2, new_size, factor);
         }
      }

      this.createAttFill({ pattern: 1001, color: 0 });

      if ((rect.width <= lmt) || (rect.height <= lmt)) {
         svg.style("display", "none");
         console.warn("Hide canvas while geometry too small w=",rect.width," h=",rect.height);
         rect.width = 200; rect.height = 100; // just to complete drawing
      } else {
         svg.style("display", null);
      }

      if (this._fixed_size) {
         svg.attr("x", 0)
            .attr("y", 0)
            .attr("width", rect.width)
            .attr("height", rect.height)
            .style("position", "absolute");
      } else {
        svg.attr("x", 0)
           .attr("y", 0)
           .style("width", "100%")
           .style("height", "100%")
           .style("position", "absolute")
           .style("left", 0)
           .style("top", 0)
           .style("right", 0)
           .style("bottom", 0);
      }

      svg.attr("viewBox", "0 0 " + rect.width + " " + rect.height)
         .attr("preserveAspectRatio", "none")  // we do not preserve relative ratio
         .property('height_factor', factor)
         .property('draw_x', 0)
         .property('draw_y', 0)
         .property('draw_width', rect.width)
         .property('draw_height', rect.height);

      svg.select(".canvas_fillrect")
         .attr("width", rect.width)
         .attr("height", rect.height)
         .call(this.fillatt.func);

      this._fast_drawing = JSROOT.settings.SmallPad && ((rect.width < JSROOT.settings.SmallPad.width) || (rect.height < JSROOT.settings.SmallPad.height));

      if (this.AlignBtns && btns)
         this.AlignBtns(btns, rect.width, rect.height);

      return true;
   }

   /** @summary Enlarge pad draw element when possible */
   RPadPainter.prototype.EnlargePad = function(evnt) {

      if (evnt) {
         evnt.preventDefault();
         evnt.stopPropagation();
      }

      let svg_can = this.svg_canvas(),
          pad_enlarged = svg_can.property("pad_enlarged");

      if (this.iscan || !this.has_canvas || (!pad_enlarged && !this.HasObjectsToDraw() && !this.painters)) {
         if (this._fixed_size) return; // canvas cannot be enlarged in such mode
         if (!this.enlargeMain('toggle')) return;
         if (this.enlargeMain('state')=='off') svg_can.property("pad_enlarged", null);
      } else if (!pad_enlarged) {
         this.enlargeMain(true, true);
         svg_can.property("pad_enlarged", this.pad);
      } else if (pad_enlarged === this.pad) {
         this.enlargeMain(false);
         svg_can.property("pad_enlarged", null);
      } else {
         console.error('missmatch with pad double click events');
      }

      let was_fast = this._fast_drawing;

      this.checkResize(true);

      if (this._fast_drawing != was_fast)
         this.ShowButtons();
   }

   /** @summary Create SVG element for the pad
     * @returns true when pad is displayed and all its items should be redrawn */
   RPadPainter.prototype.CreatePadSvg = function(only_resize) {

      if (!this.has_canvas) {
         this.CreateCanvasSvg(only_resize ? 2 : 0);
         return true;
      }

      let svg_parent = this.svg_pad(this.pad_name),
          svg_can = this.svg_canvas(),
          width = svg_parent.property("draw_width"),
          height = svg_parent.property("draw_height"),
          pad_enlarged = svg_can.property("pad_enlarged"),
          pad_visible = true,
          w = width, h = height, x = 0, y = 0,
          svg_pad = null, svg_rect = null, btns = null;

      if (this.pad && this.pad.fPos && this.pad.fSize) {
         x = Math.round(width * this.pad.fPos.fHoriz.fArr[0]);
         y = Math.round(height * this.pad.fPos.fVert.fArr[0]);
         w = Math.round(width * this.pad.fSize.fHoriz.fArr[0]);
         h = Math.round(height * this.pad.fSize.fVert.fArr[0]);
      }

      if (pad_enlarged) {
         pad_visible = false;
         if (pad_enlarged === this.pad)
            pad_visible = true;
         else
            this.forEachPainterInPad(pp => { if (pp.GetObject() == pad_enlarged) pad_visible = true; }, "pads");

         if (pad_visible) { w = width; h = height; x = y = 0; }
      }

      if (only_resize) {
         svg_pad = this.svg_pad();
         svg_rect = svg_pad.select(".root_pad_border");
         if (!JSROOT.BatchMode)
            btns = this.svg_layer("btns_layer");
      } else {
         svg_pad = svg_parent.select(".primitives_layer")
             .append("svg:svg") // here was g before, svg used to blend all drawin outside
             .classed("__root_pad_" + this.this_pad_name, true)
             .attr("pad", this.this_pad_name) // set extra attribute  to mark pad name
             .property('pad_painter', this) // this is custom property
             .property('mainpainter', null); // this is custom property
         svg_rect = svg_pad.append("svg:rect").attr("class", "root_pad_border");

         svg_pad.append("svg:g").attr("class","primitives_layer");
         if (!JSROOT.BatchMode)
            btns = svg_pad.append("svg:g")
                          .attr("class","btns_layer")
                          .property('leftside', JSROOT.settings.ToolBarSide != 'left')
                          .property('vertical', JSROOT.settings.ToolBarVert);

         if (JSROOT.settings.ContextMenu)
            svg_rect.on("contextmenu", this.PadContextMenu.bind(this));

         if (!JSROOT.BatchMode)
            svg_rect.attr("pointer-events", "visibleFill") // get events also for not visible rect
                    .on("dblclick", this.EnlargePad.bind(this))
                    .on("click", this.SelectObjectPainter.bind(this, this, null))
                    .on("mouseenter", this.ShowObjectStatus.bind(this));
      }

      this.createAttFill({ attr: this.pad });

      this.createAttLine({ attr: this.pad, color0: this.pad.fBorderMode == 0 ? 'none' : '' });

      svg_pad.attr("display", pad_visible ? null : "none")
             .attr("viewBox", "0 0 " + w + " " + h) // due to svg
             .attr("preserveAspectRatio", "none")   // due to svg, we do not preserve relative ratio
             .attr("x", x)    // due to svg
             .attr("y", y)   // due to svg
             .attr("width", w)    // due to svg
             .attr("height", h)   // due to svg
             .property('draw_x', x) // this is to make similar with canvas
             .property('draw_y', y)
             .property('draw_width', w)
             .property('draw_height', h);

      svg_rect.attr("x", 0)
              .attr("y", 0)
              .attr("width", w)
              .attr("height", h)
              .call(this.fillatt.func)
              .call(this.lineatt.func);

      this._fast_drawing = JSROOT.settings.SmallPad && ((w < JSROOT.settings.SmallPad.width) || (h < JSROOT.settings.SmallPad.height));

       // special case of 3D canvas overlay
      if (svg_pad.property('can3d') === JSROOT.constants.Embed3D.Overlay)
          this.select_main().select(".draw3d_" + this.this_pad_name)
              .style('display', pad_visible ? '' : 'none');

      if (this.AlignBtns && btns) this.AlignBtns(btns, w, h);

      return pad_visible;
   }

   RPadPainter.prototype.RemovePrimitive = function(obj) {
      if (!this.pad || !this.pad.fPrimitives) return;
      let indx = this.pad.fPrimitives.arr.indexOf(obj);
      if (indx>=0) this.pad.fPrimitives.RemoveAt(indx);
   }

   RPadPainter.prototype.FindPrimitive = function(exact_obj, classname, name) {
      if (!this.pad || !this.pad.fPrimitives) return null;

      for (let i=0; i < this.pad.fPrimitives.arr.length; i++) {
         let obj = this.pad.fPrimitives.arr[i];

         if ((exact_obj!==null) && (obj !== exact_obj)) continue;

         if ((classname !== undefined) && (classname !== null))
            if (obj._typename !== classname) continue;

         if ((name !== undefined) && (name !== null))
            if (obj.fName !== name) continue;

         return obj;
      }

      return null;
   }

   /** @summary returns true if any objects beside sub-pads exists in the pad */
   RPadPainter.prototype.HasObjectsToDraw = function() {

      let arr = this.pad ? this.pad.fPrimitives : null;

      if (arr)
         for (let n=0;n<arr.length;++n)
            if (arr[n] && arr[n]._typename != "ROOT::Experimental::RPadDisplayItem") return true;

      return false;
   }

   /** @summary Draw pad primitives */
   RPadPainter.prototype.DrawPrimitives = function(indx) {

      if (!indx) {
         indx = 0;
         // flag used to prevent immediate pad redraw during normal drawing sequence
         this._doing_pad_draw = true;

         if (this.iscan)
            this._start_tm = new Date().getTime();

         // set number of primitves
         this._num_primitives = this.pad && this.pad.fPrimitives ? this.pad.fPrimitives.length : 0;
      }

      if (!this.pad || (indx >= this._num_primitives)) {
         delete this._doing_pad_draw;

         if (this._start_tm) {
            let spenttm = new Date().getTime() - this._start_tm;
            if (spenttm > 3000) console.log("Canvas drawing took " + (spenttm*1e-3).toFixed(2) + "s");
            delete this._start_tm;
            delete this._lasttm_tm;
         }

         return Promise.resolve();
      }

      // handle used to invoke callback only when necessary
      return JSROOT.draw(this.divid, this.pad.fPrimitives[indx], "").then(ppainter => {
         // mark painter as belonging to primitives
         if (ppainter && (typeof ppainter == 'object'))
            ppainter._primitive = true;

         return this.DrawPrimitives(indx+1);
      });
   }

   RPadPainter.prototype.GetTooltips = function(pnt) {
      let painters = [], hints = [];

      // first count - how many processors are there
      if (this.painters !== null)
         this.painters.forEach(obj => {
            if ('ProcessTooltip' in obj) painters.push(obj);
         });

      if (pnt) pnt.nproc = painters.length;

      painters.forEach(obj => {
         let hint = obj.ProcessTooltip(pnt);
         if (!hint) hint = { user_info: null };
         hints.push(hint);
         if (hint && pnt && pnt.painters) hint.painter = obj;
      });

      return hints;
   }

   RPadPainter.prototype.FillContextMenu = function(menu) {

      if (this.pad)
         menu.add("header: " + this.pad._typename + "::" + this.pad.fName);
      else
         menu.add("header: Canvas");

      menu.addchk(this.IsTooltipAllowed(), "Show tooltips", this.SetTooltipAllowed.bind(this, "toggle"));

      if (!this._websocket) {

         function ToggleGridField(arg) {
            this.pad[arg] = this.pad[arg] ? 0 : 1;
            let main = this.main_painter();
            if (main && (typeof main.DrawGrids == 'function')) main.DrawGrids();
         }

         function SetTickField(arg) {
            this.pad[arg.substr(1)] = parseInt(arg[0]);

            let main = this.main_painter();
            if (main && (typeof main.DrawAxes == 'function')) main.DrawAxes();
         }

         menu.addchk(this.pad.fGridx, 'Grid x', 'fGridx', ToggleGridField);
         menu.addchk(this.pad.fGridy, 'Grid y', 'fGridy', ToggleGridField);
         menu.add("sub:Ticks x");
         menu.addchk(this.pad.fTickx == 0, "normal", "0fTickx", SetTickField);
         menu.addchk(this.pad.fTickx == 1, "ticks on both sides", "1fTickx", SetTickField);
         menu.addchk(this.pad.fTickx == 2, "labels up", "2fTickx", SetTickField);
         menu.add("endsub:");
         menu.add("sub:Ticks y");
         menu.addchk(this.pad.fTicky == 0, "normal", "0fTicky", SetTickField);
         menu.addchk(this.pad.fTicky == 1, "ticks on both side", "1fTicky", SetTickField);
         menu.addchk(this.pad.fTicky == 2, "labels right", "2fTicky", SetTickField);
         menu.add("endsub:");

         //menu.addchk(this.pad.fTickx, 'Tick x', 'fTickx', ToggleField);
         //menu.addchk(this.pad.fTicky, 'Tick y', 'fTicky', ToggleField);

         menu.AddAttributesMenu(this);
      }

      menu.add("separator");

      if (this.ToggleEventStatus)
         menu.addchk(this.HasEventStatus(), "Event status", this.ToggleEventStatus.bind(this));

      if (this.enlargeMain() || (this.has_canvas && this.HasObjectsToDraw()))
         menu.addchk((this.enlargeMain('state')=='on'), "Enlarge " + (this.iscan ? "canvas" : "pad"), this.EnlargePad.bind(this, null));

      let fname = this.this_pad_name;
      if (fname.length===0) fname = this.iscan ? "canvas" : "pad";
      menu.add("Save as "+fname+".png", fname+".png", this.SaveAs.bind(this, "png", false));
      menu.add("Save as "+fname+".svg", fname+".svg", this.SaveAs.bind(this, "svg", false));

      return true;
   }

   RPadPainter.prototype.PadContextMenu = function(evnt) {
      if (evnt.stopPropagation) {
         // this is normal event processing and not emulated jsroot event
         // for debug purposes keep original context menu for small region in top-left corner
         let pos = d3.pointer(evnt, this.svg_pad().node());

         if (pos && (pos.length==2) && (pos[0] >= 0) && (pos[0] < 10) && (pos[1] >= 0) && (pos[1] < 10)) return;

         evnt.stopPropagation(); // disable main context menu
         evnt.preventDefault();  // disable browser context menu

         let fp = this.frame_painter();
         if (fp) fp.SetLastEventPos();
      }

      jsrp.createMenu(this, evnt).then(menu => {
         this.FillContextMenu(menu);
         return this.fillObjectExecMenu(menu);
      }).then(menu => menu.show());
   }

   RPadPainter.prototype.Redraw = function(reason) {

      // prevent redrawing
      if (this._doing_pad_draw)
         return console.log('Prevent pad redrawing');

      let showsubitems = true;

      if (this.iscan) {
         this.CreateCanvasSvg(2);
      } else {
         showsubitems = this.CreatePadSvg(true);
      }

      // even sub-pad is not visible, we should redraw sub-sub-pads to hide them as well
      for (let i = 0; i < this.painters.length; ++i) {
         let sub = this.painters[i];
         if (showsubitems || sub.this_pad_name) sub.Redraw(reason);
      }

      if (jsrp.getActivePad() === this) {
         let canp = this.canv_painter();
         if (canp) canp.PadEvent("padredraw", this);
      }
   }

   RPadPainter.prototype.NumDrawnSubpads = function() {
      if (!this.painters) return 0;

      let num = 0;

      for (let i = 0; i < this.painters.length; ++i)
         if (this.painters[i] instanceof RPadPainter)
            num++;

      return num;
   }

   RPadPainter.prototype.RedrawByResize = function() {
      let elem = this.svg_pad();
      if (!elem.empty() && elem.property('can3d') === JSROOT.constants.Embed3D.Overlay) return true;

      for (let i = 0; i < this.painters.length; ++i)
         if (typeof this.painters[i].RedrawByResize === 'function')
            if (this.painters[i].RedrawByResize()) return true;

      return false;
   }

   RPadPainter.prototype.CheckCanvasResize = function(size, force) {

      if (!this.iscan && this.has_canvas) return false;

      if ((size === true) || (size === false)) { force = size; size = null; }

      if (size && (typeof size === 'object') && size.force) force = true;

      if (!force) force = this.RedrawByResize();

      let changed = this.CreateCanvasSvg(force ? 2 : 1, size);

      // if canvas changed, redraw all its subitems.
      // If redrawing was forced for canvas, same applied for sub-elements
      if (changed)
         for (let i = 0; i < this.painters.length; ++i)
            this.painters[i].Redraw(force ? "redraw" : "resize");

      return changed;
   }

   RPadPainter.prototype.UpdateObject = function(obj) {
      if (!obj) return false;

      this.pad.fStyle = obj.fStyle;
      this.pad.fAttr = obj.fAttr;

      if (this.iscan) {
         this.pad.fTitle = obj.fTitle;
         this.pad.fWinSize = obj.fWinSize;
      } else {
         this.pad.fPos = obj.fPos;
         this.pad.fSize = obj.fSize;
      }

      return true;
   }


   /** @summary Add object painter to list of primitives */
   RPadPainter.prototype.AddObjectPainter = function(objpainter, lst, indx) {
      if (objpainter && lst && lst[indx] && (objpainter.snapid === undefined)) {
         // keep snap id in painter, will be used for the
         if (this.painters.indexOf(objpainter) < 0)
            this.painters.push(objpainter);
         objpainter.AssignSnapId(lst[indx].fObjectID);
         if (!objpainter.rstyle) objpainter.rstyle = lst[indx].fStyle || this.rstyle;
      }
   }

   /** @summary Function called when drawing next snapshot from the list
     * @returns {Promise} with pad painter when ready
     * @private */
   RPadPainter.prototype.DrawNextSnap = function(lst, indx) {

      if (indx===undefined) {
         indx = -1;
         // flag used to prevent immediate pad redraw during first draw
         this._doing_pad_draw = true;
         this._snaps_map = {}; // to control how much snaps are drawn
         this._num_primitives = lst ? lst.length : 0;
      }

      delete this.next_rstyle;

      ++indx; // change to the next snap

      if (!lst || indx >= lst.length) {
         delete this._doing_pad_draw;
         delete this._snaps_map;
         return Promise.resolve(this);
      }

      let snap = lst[indx],
          snapid = snap.fObjectID,
          cnt = this._snaps_map[snapid],
          objpainter = null;

      if (cnt) cnt++; else cnt=1;
      this._snaps_map[snapid] = cnt; // check how many objects with same snapid drawn, use them again

      // empty object, no need to do something, take next
      if (snap.fDummy) return this.DrawNextSnap(lst, indx);

      // first appropriate painter for the object
      // if same object drawn twice, two painters will exists
      for (let k=0; k<this.painters.length; ++k) {
         if (this.painters[k].snapid === snapid)
            if (--cnt === 0) { objpainter = this.painters[k]; break;  }
      }

      if (objpainter) {

         if (snap._typename == "ROOT::Experimental::RPadDisplayItem")  // subpad
            return objpainter.RedrawPadSnap(snap).then(ppainter => {
               this.AddObjectPainter(ppainter, lst, indx);
               return this.DrawNextSnap(lst, indx);
            });

         if (objpainter.UpdateObject(snap.fDrawable || snap.fObject || snap, snap.fOption || ""))
            objpainter.Redraw();

         return this.DrawNextSnap(lst, indx); // call next
      }

      if (snap._typename == "ROOT::Experimental::RPadDisplayItem") { // subpad

         let subpad = snap; // not subpad, but just attributes

         let padpainter = new RPadPainter(subpad, false);
         padpainter.DecodeOptions("");
         padpainter.SetDivId(this.divid); // pad painter will be registered in parent painters list
         padpainter.AssignSnapId(snap.fObjectID);
         padpainter.rstyle = snap.fStyle;

         padpainter.CreatePadSvg();

         if (snap.fPrimitives && snap.fPrimitives.length > 0)
            padpainter.AddPadButtons();

         // we select current pad, where all drawing is performed
         let prev_name = padpainter.CurrentPadName(padpainter.this_pad_name);

         return padpainter.DrawNextSnap(snap.fPrimitives).then(() => {
            padpainter.CurrentPadName(prev_name);
            return this.DrawNextSnap(lst, indx);
         });
      }

      // will be used in SetDivId to assign style to painter
      this.next_rstyle = lst[indx].fStyle || this.rstyle;

      if (snap._typename === "ROOT::Experimental::TObjectDisplayItem") {

         // identifier used in RObjectDrawable
         let webSnapIds = { kNone: 0,  kObject: 1, kColors: 4, kStyle: 5, kPalette: 6 };

         if (snap.fKind == webSnapIds.kStyle) {
            JSROOT.extend(JSROOT.gStyle, snap.fObject);
            return this.DrawNextSnap(lst, indx);
         }

         if (snap.fKind == webSnapIds.kColors) {
            let ListOfColors = [], arr = snap.fObject.arr;
            for (let n = 0; n < arr.length; ++n) {
               let name = arr[n].fString, p = name.indexOf("=");
               if (p > 0)
                  ListOfColors[parseInt(name.substr(0,p))] =name.substr(p+1);
            }

            this.root_colors = ListOfColors;
            // set global list of colors
            // jsrp.adoptRootColors(ListOfColors);
            return this.DrawNextSnap(lst, indx);
         }

         if (snap.fKind == webSnapIds.kPalette) {
            let arr = snap.fObject.arr, palette = [];
            for (let n = 0; n < arr.length; ++n)
               palette[n] =  arr[n].fString;
            this.custom_palette = new JSROOT.ColorPalette(palette);
            return this.DrawNextSnap(lst, indx);
         }

         if (!this.frame_painter())
            return JSROOT.draw(this.divid, { _typename: "TFrame", $dummy: true }, "")
                         .then(() => this.DrawNextSnap(lst, indx-1)); // call same object again
      }

      // TODO - fDrawable is v7, fObject from v6, maybe use same data member?
      return JSROOT.draw(this.divid, snap.fDrawable || snap.fObject || snap, snap.fOption || "").then(objpainter => {
         this.AddObjectPainter(objpainter, lst, indx);
         return this.DrawNextSnap(lst, indx);
      });
   }

   /** @summary Search painter with specified snapid, also sub-pads are checked */
   RPadPainter.prototype.FindSnap = function(snapid, onlyid) {

      function check(checkid) {
         if (!checkid || (typeof checkid != 'string')) return false;
         if (checkid == snapid) return true;
         return onlyid && (checkid.length > snapid.length) &&
                (checkid.indexOf(snapid) == (checkid.length - snapid.length));
      }

      if (check(this.snapid)) return this;

      if (!this.painters) return null;

      for (let k=0;k<this.painters.length;++k) {
         let sub = this.painters[k];

         if (!onlyid && (typeof sub.FindSnap === 'function'))
            sub = sub.FindSnap(snapid);
         else if (!check(sub.snapid))
            sub = null;

         if (sub) return sub;
      }

      return null;
   }

   /** @summary Redraw pad snap
     * @desc Online version of drawing pad primitives
     * @return {Promise} with pad painter*/
   RPadPainter.prototype.RedrawPadSnap = function(snap) {
      // for the pad/canvas display item contains list of primitives plus pad attributes

      if (!snap || !snap.fPrimitives) return Promise.resolve(this);

      // for the moment only window size attributes are provided
      // let padattr = { fCw: snap.fWinSize[0], fCh: snap.fWinSize[1], fTitle: snap.fTitle };

      // if canvas size not specified in batch mode, temporary use 900x700 size
      // if (this.batch_mode && this.iscan && (!padattr.fCw || !padattr.fCh)) { padattr.fCw = 900; padattr.fCh = 700; }

      if (this.iscan && snap.fTitle && (typeof document !== "undefined"))
         document.title = snap.fTitle;

      if (this.snapid === undefined) {
         // first time getting snap, create all gui elements first

         this.AssignSnapId(snap.fObjectID);

         this.draw_object = snap;
         this.pad = snap;

         if (this.batch_mode && this.iscan)
             this._fixed_size = true;

         this.CreateCanvasSvg(0);
         this.SetDivId(this.divid);  // now add to painters list
         this.AddPadButtons(true);

         return this.DrawNextSnap(snap.fPrimitives);
      }

      // update only pad/canvas attributes
      this.UpdateObject(snap);

      // apply all changes in the object (pad or canvas)
      if (this.iscan) {
         this.CreateCanvasSvg(2);
      } else {
         this.CreatePadSvg(true);
      }

      let isanyfound = false, isanyremove = false;

      // find and remove painters which no longer exists in the list
      for (let k=0;k<this.painters.length;++k) {
         let sub = this.painters[k];
         if (sub.snapid===undefined) continue; // look only for painters with snapid

         for (let i=0;i<snap.fPrimitives.length;++i)
            if (snap.fPrimitives[i].fObjectID === sub.snapid) { sub = null; isanyfound = true; break; }

         if (sub) {
            // remove painter which does not found in the list of snaps
            this.painters.splice(k--,1);
            sub.Cleanup(); // cleanup such painter
            isanyremove = true;
         }
      }

      if (isanyremove) {
         delete this.pads_cache;
      }

      if (!isanyfound) {
         let svg_p = this.svg_pad(this.this_pad_name),
             fp = this.frame_painter();
         if (svg_p && !svg_p.empty())
            svg_p.property('mainpainter', null);
         for (let k=0;k<this.painters.length;++k)
            if (fp !== this.painters[k])
               this.painters[k].Cleanup();
         this.painters = [];
         if (fp) {
            this.painters.push(fp);
            fp.CleanFrameDrawings();
         }
         if (this.RemoveButtons) this.RemoveButtons();
         this.AddPadButtons(true);
      }

      let prev_name = this.CurrentPadName(this.this_pad_name);

      return this.DrawNextSnap(snap.fPrimitives).then(() => {
         this.CurrentPadName(prev_name);

         if (jsrp.getActivePad() === this) {
            let canp = this.canv_painter();

            if (canp) canp.PadEvent("padredraw", this);
         }

         return this;
      });
   }

   /** @summary Create image for the pad
     * @returns {Promise} with image data, coded with btoa() function */
   RPadPainter.prototype.CreateImage = function(format) {
      // use https://github.com/MrRio/jsPDF in the future here
      if (format=="pdf")
         return Promise.resolve(btoa("dummy PDF file"));

      if ((format=="png") || (format=="jpeg") || (format=="svg"))
         return this.ProduceImage(true, format).then(res => {
            if (!res || (format=="svg")) return res;
            let separ = res.indexOf("base64,");
            return (separ>0) ? res.substr(separ+7) : "";
         });

      return Promise.resolve("");
   }

   RPadPainter.prototype.ItemContextMenu = function(name) {
       let rrr = this.svg_pad().node().getBoundingClientRect();
       let evnt = { clientX: rrr.left+10, clientY: rrr.top + 10 };

       // use timeout to avoid conflict with mouse click and automatic menu close
       if (name=="pad")
          return setTimeout(this.PadContextMenu.bind(this, evnt), 50);

       let selp = null, selkind;

       switch(name) {
          case "xaxis":
          case "yaxis":
          case "zaxis":
             selp = this.main_painter();
             selkind = name[0];
             break;
          case "frame":
             selp = this.frame_painter();
             break;
          default: {
             let indx = parseInt(name);
             if (!isNaN(indx)) selp = this.painters[indx];
          }
       }

       console.log('Start fill here', selkind)

       if (!selp || (typeof selp.FillContextMenu !== 'function')) return;

       jsrp.createMenu(selp, evnt).then(menu => {
          if (selp.FillContextMenu(menu, selkind))
             setTimeout(menu.show.bind(menu), 50);
       });
   }

   RPadPainter.prototype.SaveAs = function(kind, full_canvas, filename) {
      if (!filename) {
         filename = this.this_pad_name;
         if (filename.length === 0) filename = this.iscan ? "canvas" : "pad";
         filename += "." + kind;
      }
      this.ProduceImage(full_canvas, kind).then(imgdata => {
         let a = document.createElement('a');
         a.download = filename;
         a.href = (kind != "svg") ? imgdata : "data:image/svg+xml;charset=utf-8,"+encodeURIComponent(imgdata);
         document.body.appendChild(a);
         a.addEventListener("click", () => a.parentNode.removeChild(a));
         a.click();
      });
   }

   /** @summary Prodce image for the pad
     * @returns {Promise} with created image */
   RPadPainter.prototype.ProduceImage = function(full_canvas, file_format) {

      let use_frame = (full_canvas === "frame");

      let elem = use_frame ? this.svg_frame() : (full_canvas ? this.svg_canvas() : this.svg_pad());

      if (elem.empty()) return Promise.resolve("");

      let painter = (full_canvas && !use_frame) ? this.canv_painter() : this;

      let items = []; // keep list of replaced elements, which should be moved back at the end

      if (!use_frame) // do not make transformations for the frame
      painter.forEachPainterInPad(pp => {

         let item = { prnt: pp.svg_pad() };
         items.push(item);

         // remove buttons from each subpad
         let btns = pp.svg_layer("btns_layer");
         item.btns_node = btns.node();
         if (item.btns_node) {
            item.btns_prnt = item.btns_node.parentNode;
            item.btns_next = item.btns_node.nextSibling;
            btns.remove();
         }

         let main = pp.frame_painter();
         if (!main || (typeof main.Render3D !== 'function')) return;

         let can3d = main.access_3d_kind();

         if ((can3d !== JSROOT.constants.Embed3D.Overlay) && (can3d !== JSROOT.constants.Embed3D.Embed)) return;

         let sz2 = main.size_for_3d(JSROOT.constants.Embed3D.Embed); // get size and position of DOM element as it will be embed

         let canvas = main.renderer.domElement;
         main.Render3D(0); // WebGL clears buffers, therefore we should render scene and convert immediately
         let dataUrl = canvas.toDataURL("image/png");

         // remove 3D drawings

         if (can3d === JSROOT.constants.Embed3D.Embed) {
            item.foreign = item.prnt.select("." + sz2.clname);
            item.foreign.remove();
         }

         let svg_frame = main.svg_frame();
         item.frame_node = svg_frame.node();
         if (item.frame_node) {
            item.frame_next = item.frame_node.nextSibling;
            svg_frame.remove();
         }

         // add svg image
         item.img = item.prnt.insert("image",".primitives_layer")     // create image object
                        .attr("x", sz2.x)
                        .attr("y", sz2.y)
                        .attr("width", canvas.width)
                        .attr("height", canvas.height)
                        .attr("href", dataUrl);

      }, "pads");

      function reEncode(data) {
         data = encodeURIComponent(data);
         data = data.replace(/%([0-9A-F]{2})/g, function(match, p1) {
           let c = String.fromCharCode('0x'+p1);
           return c === '%' ? '%25' : c;
         });
         return decodeURIComponent(data);
      }

      function reconstruct() {
         for (let k=0;k<items.length;++k) {
            let item = items[k];

            if (item.img)
               item.img.remove(); // delete embed image

            let prim = item.prnt.select(".primitives_layer");

            if (item.foreign) // reinsert foreign object
               item.prnt.node().insertBefore(item.foreign.node(), prim.node());

            if (item.frame_node) // reinsert frame as first in list of primitives
               prim.node().insertBefore(item.frame_node, item.frame_next);

            if (item.btns_node) // reinsert buttons
               item.btns_prnt.insertBefore(item.btns_node, item.btns_next);
         }
      }

      let width = elem.property('draw_width'), height = elem.property('draw_height');
      if (use_frame) { width = this.frame_width(); height = this.frame_height(); }

      let svg = '<svg width="' + width + '" height="' + height + '" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">' +
                 elem.node().innerHTML +
                 '</svg>';

      if (jsrp.processSvgWorkarounds)
         svg = jsrp.processSvgWorkarounds(svg);

      svg = jsrp.compressSVG(svg);

      if (file_format == "svg") {
         reconstruct();
         return Promise.resolve(svg); // return SVG file as is
      }

      let doctype = '<?xml version="1.0" standalone="no"?><!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">';

      let image = new Image();

      return new Promise(resolveFunc => {
         image.onload = function() {
            let canvas = document.createElement('canvas');
            canvas.width = image.width;
            canvas.height = image.height;
            let context = canvas.getContext('2d');
            context.drawImage(image, 0, 0);

            reconstruct();

            resolveFunc(canvas.toDataURL('image/' + file_format));
         }

         image.onerror = function(arg) {
            console.log('IMAGE ERROR', arg);
            reconstruct();
            resolveFunc(null);
         }

         image.src = 'data:image/svg+xml;base64,' + window.btoa(reEncode(doctype + svg));
      });
   }


   RPadPainter.prototype.PadButtonClick = function(funcname, evnt) {

      if (funcname == "CanvasSnapShot") return this.SaveAs("png", true);

      if (funcname == "EnlargePad") return this.EnlargePad(null);

      if (funcname == "PadSnapShot") return this.SaveAs("png", false);

      if (funcname == "PadContextMenus") {

         if (evnt) {
            evnt.preventDefault();
            evnt.stopPropagation();
         }

         if (jsrp.closeMenu && jsrp.closeMenu()) return;

         jsrp.createMenu(this, evnt).then(menu => {
            menu.add("header:Menus");

            if (this.iscan)
               menu.add("Canvas", "pad", this.ItemContextMenu);
            else
               menu.add("Pad", "pad", this.ItemContextMenu);

            if (this.frame_painter())
               menu.add("Frame", "frame", this.ItemContextMenu);

            let main = this.main_painter();

            if (main) {
               menu.add("X axis", "xaxis", this.ItemContextMenu);
               menu.add("Y axis", "yaxis", this.ItemContextMenu);
               if ((typeof main.Dimension === 'function') && (main.Dimension() > 1))
                  menu.add("Z axis", "zaxis", this.ItemContextMenu);
            }

            if (this.painters && (this.painters.length>0)) {
               menu.add("separator");
               let shown = [];
               for (let n=0;n<this.painters.length;++n) {
                  let pp = this.painters[n];
                  let obj = pp ? pp.GetObject() : null;
                  if (!obj || (shown.indexOf(obj)>=0)) continue;

                  let name = ('_typename' in obj) ? (obj._typename + "::") : "";
                  if ('fName' in obj) name += obj.fName;
                  if (name.length==0) name = "item" + n;
                  menu.add(name, n, this.ItemContextMenu);
               }
            }

            menu.show();
         });

         return;
      }

      // click automatically goes to all sub-pads
      // if any painter indicates that processing completed, it returns true
      let done = false;

      for (let i = 0; i < this.painters.length; ++i) {
         let pp = this.painters[i];

         if (typeof pp.PadButtonClick == 'function')
            pp.PadButtonClick(funcname);

         if (!done && (typeof pp.ButtonClick == 'function'))
            done = pp.ButtonClick(funcname);
      }
   }

   RPadPainter.prototype.AddButton = function(_btn, _tooltip, _funcname, _keyname) {
      if (!JSROOT.settings.ToolBar || JSROOT.BatchMode || this.batch_mode) return;

      if (!this._buttons) this._buttons = [];
      // check if there are duplications

      for (let k=0;k<this._buttons.length;++k)
         if (this._buttons[k].funcname == _funcname) return;

      this._buttons.push({ btn: _btn, tooltip: _tooltip, funcname: _funcname, keyname: _keyname });

      let iscan = this.iscan || !this.has_canvas;
      if (!iscan && (_funcname.indexOf("Pad")!=0) && (_funcname !== "EnlargePad")) {
         let cp = this.canv_painter();
         if (cp && (cp!==this)) cp.AddButton(_btn, _tooltip, _funcname);
      }
   }

   RPadPainter.prototype.AddPadButtons = function(/* is_online */) {

      this.AddButton("camera", "Create PNG", this.iscan ? "CanvasSnapShot" : "PadSnapShot", "Ctrl PrintScreen");

      if (JSROOT.settings.ContextMenu)
         this.AddButton("question", "Access context menus", "PadContextMenus");

      let add_enlarge = !this.iscan && this.has_canvas && this.HasObjectsToDraw()

      if (add_enlarge || this.enlargeMain('verify'))
         this.AddButton("circle", "Enlarge canvas", "EnlargePad");
   }


   RPadPainter.prototype.ShowButtons = function() {
      if (!this._buttons) return;

      JSROOT.require(['interactive']).then(inter => {
         inter.PadButtonsHandler.assign(this);
         this.ShowButtons();
      });
   }

   /** @summary Calculates RPadLength value */
   RPadPainter.prototype.GetPadLength = function(vertical, len, ignore_user) {
      if (!len) return 0;
      function GetV(indx, dflt) {
         return (len.fArr && (indx < len.fArr.length)) ? len.fArr[indx] : dflt;
      }

      let norm = GetV(0, 0),
          pixel = GetV(1, 0),
          user = ignore_user ? undefined : GetV(2);

      let res = pixel;
      if (norm) res += (vertical ? this.pad_height() : this.pad_width()) * norm;

      if (user !== undefined)
          console.log('Do implement user coordinates');
      return res;
   }


   /** @summary Calculates pad position for RPadPos values
    * @param {object} pos - instance of RPadPos */
   RPadPainter.prototype.GetCoordinate = function(pos) {
      let res = { x: 0, y: 0 };

      if (pos) {
         res.x = this.GetPadLength(false, pos.fHoriz);
         res.y = this.pad_height() - this.GetPadLength(true, pos.fVert);
      }

      return res;
   }

   RPadPainter.prototype.DecodeOptions = function(opt) {
      let pad = this.GetObject();
      if (!pad) return;

      let d = new JSROOT.DrawOptions(opt);

      if (d.check('WEBSOCKET')) this.OpenWebsocket();
      if (!this.options) this.options = {};

      JSROOT.extend(this.options, { GlobalColors: true, LocalColors: false, IgnorePalette: false, RotateFrame: false, FixFrame: false });

      if (d.check('NOCOLORS') || d.check('NOCOL')) this.options.GlobalColors = this.options.LocalColors = false;
      if (d.check('LCOLORS') || d.check('LCOL')) { this.options.GlobalColors = false; this.options.LocalColors = true; }
      if (d.check('NOPALETTE') || d.check('NOPAL')) this.options.IgnorePalette = true;
      if (d.check('ROTATE')) this.options.RotateFrame = true;
      if (d.check('FIXFRAME')) this.options.FixFrame = true;

      if (d.check('WHITE')) pad.fFillColor = 0;
      if (d.check('LOGX')) pad.fLogx = 1;
      if (d.check('LOGY')) pad.fLogy = 1;
      if (d.check('LOGZ')) pad.fLogz = 1;
      if (d.check('LOG')) pad.fLogx = pad.fLogy = pad.fLogz = 1;
      if (d.check('GRIDX')) pad.fGridx = 1;
      if (d.check('GRIDY')) pad.fGridy = 1;
      if (d.check('GRID')) pad.fGridx = pad.fGridy = 1;
      if (d.check('TICKX')) pad.fTickx = 1;
      if (d.check('TICKY')) pad.fTicky = 1;
      if (d.check('TICK')) pad.fTickx = pad.fTicky = 1;
   }

   let drawPad = (divid, pad, opt) => {
      let painter = new RPadPainter(pad, false);
      painter.DecodeOptions(opt);

      painter.SetDivId(divid); // pad painter will be registered in the canvas painters list

      if (painter.svg_canvas().empty()) {
         painter.has_canvas = false;
         painter.this_pad_name = "";
      }

      painter.CreatePadSvg();

      if (painter.MatchObjectType("TPad") && (!painter.has_canvas || painter.HasObjectsToDraw())) {
         painter.AddPadButtons();
      }

      // we select current pad, where all drawing is performed
      let prev_name = painter.has_canvas ? painter.CurrentPadName(painter.this_pad_name) : undefined;

      jsrp.selectActivePad({ pp: painter, active: false });

      // flag used to prevent immediate pad redraw during first draw
      return painter.DrawPrimitives().then(() => {
         painter.ShowButtons();
         // we restore previous pad name
         painter.CurrentPadName(prev_name);
         return painter;
      });
   }

   // ==========================================================================================

   function RCanvasPainter(canvas) {
      // used for online canvas painter
      RPadPainter.call(this, canvas, true);
      this._websocket = null;
      this.tooltip_allowed = JSROOT.settings.Tooltip;
   }

   RCanvasPainter.prototype = Object.create(RPadPainter.prototype);

   /** @summary Cleanup canvas painter */
   RCanvasPainter.prototype.Cleanup = function() {
      delete this._websocket;
      delete this._submreq;

      RPadPainter.prototype.Cleanup.call(this);
   }

   /** @summary Changes layout
     * @returns {Promise} indicating when finished */
   RCanvasPainter.prototype.ChangeLayout = function(layout_kind) {
      let current = this.getLayoutKind();
      if (current == layout_kind)
         return Promise.resolve(true);

      let origin = this.select_main('origin'),
          sidebar = origin.select('.side_panel'),
          main = this.select_main(), lst = [];

      while (main.node().firstChild)
         lst.push(main.node().removeChild(main.node().firstChild));

      if (!sidebar.empty()) JSROOT.cleanup(sidebar.node());

      this.setLayoutKind("simple"); // restore defaults
      origin.html(""); // cleanup origin

      if (layout_kind == 'simple') {
         main = origin;
         for (let k=0;k<lst.length;++k)
            main.node().appendChild(lst[k]);
         this.setLayoutKind(layout_kind);
         JSROOT.resize(main.node());
         return Promise.resolve(true);
      }

      return JSROOT.require("jq2d").then(() => {

         let grid = new JSROOT.GridDisplay(origin.node(), layout_kind);

         if (layout_kind.indexOf("vert")==0) {
            main = d3.select(grid.getGridFrame(0));
            sidebar = d3.select(grid.getGridFrame(1));
         } else {
            main = d3.select(grid.getGridFrame(1));
            sidebar = d3.select(grid.getGridFrame(0));
         }

         main.classed("central_panel", true).style('position','relative');
         sidebar.classed("side_panel", true).style('position','relative');

         // now append all childs to the new main
         for (let k=0;k<lst.length;++k)
            main.node().appendChild(lst[k]);

         this.setLayoutKind(layout_kind, ".central_panel");

         // remove reference to MDIDisplay, solves resize problem
         origin.property('mdi', null);

         // resize main drawing and let draw extras
         JSROOT.resize(main.node());

         return true;
      });
   }

   /** @summary Toggle projection
     * @return {Promise} indicating when ready */
   RCanvasPainter.prototype.ToggleProjection = function(kind) {
      delete this.proj_painter;

      if (kind) this.proj_painter = 1; // just indicator that drawing can be preformed

      if (this.ShowUI5ProjectionArea)
         return this.ShowUI5ProjectionArea(kind);

      let layout = 'simple';

      if (kind == "X") layout = 'vert2_31'; else
      if (kind == "Y") layout = 'horiz2_13';

      return this.ChangeLayout(layout);
   }

   RCanvasPainter.prototype.DrawProjection = function( /*kind,hist*/) {
      // dummy for the moment
   }

   RCanvasPainter.prototype.DrawInsidePanel = function(canv, opt) {
      let side = this.select_main('origin').select(".side_panel");
      if (side.empty()) return Promise.resolve(null);
      return JSROOT.draw(side.node(), canv, opt);
   }

   RCanvasPainter.prototype.ShowMessage = function(msg) {
      JSROOT.progress(msg, 7000);
   }

   /** @summary Function called when canvas menu item Save is called */
   RCanvasPainter.prototype.SaveCanvasAsFile = function(fname) {
      let pnt = fname.indexOf(".");
      this.CreateImage(fname.substr(pnt+1))
          .then(res => this.SendWebsocket("SAVE:" + fname + ":" + res));
   }

   RCanvasPainter.prototype.SendSaveCommand = function(fname) {
      this.SendWebsocket("PRODUCE:" + fname);
   }

   RCanvasPainter.prototype.SendWebsocket = function(msg, chid) {
      if (this._websocket)
         this._websocket.Send(msg, chid);
   }

   RCanvasPainter.prototype.CloseWebsocket = function(force) {
      if (this._websocket) {
         this._websocket.Close(force);
         this._websocket.Cleanup();
         delete this._websocket;
      }
   }

   RCanvasPainter.prototype.OpenWebsocket = function(socket_kind) {
      // create websocket for current object (canvas)
      // via websocket one recieved many extra information

      this.CloseWebsocket();

      this._websocket = new JSROOT.WebWindowHandle(socket_kind);
      this._websocket.SetReceiver(this);
      this._websocket.Connect();
   }

   RCanvasPainter.prototype.UseWebsocket = function(handle, href) {
      this.CloseWebsocket();

      this._websocket = handle;
      console.log('Use websocket', this._websocket.key);
      this._websocket.SetReceiver(this);
      this._websocket.Connect(href);
   }

   RCanvasPainter.prototype.WindowBeforeUnloadHanlder = function() {
      // when window closed, close socket
      this.CloseWebsocket(true);
   }

   RCanvasPainter.prototype.OnWebsocketOpened = function(/*handle*/) {
      // indicate that we are ready to recieve any following commands
   }

   RCanvasPainter.prototype.OnWebsocketClosed = function(/*handle*/) {
      JSROOT.CloseCurrentWindow();
   }

   RCanvasPainter.prototype.OnWebsocketMsg = function(handle, msg) {
      console.log("GET MSG " + msg.substr(0,30));

      if (msg == "CLOSE") {
         this.OnWebsocketClosed();
         this.CloseWebsocket(true);
      } else if (msg.substr(0,5)=='SNAP:') {
         msg = msg.substr(5);
         let p1 = msg.indexOf(":"),
             snapid = msg.substr(0,p1),
             snap = JSROOT.parse(msg.substr(p1+1));
         this.RedrawPadSnap(snap).then(() => {
            handle.Send("SNAPDONE:" + snapid); // send ready message back when drawing completed
         });
      } else if (msg.substr(0,4)=='JSON') {
         let obj = JSROOT.parse(msg.substr(4));
         // console.log("get JSON ", msg.length-4, obj._typename);
         this.RedrawObject(obj);
      } else if (msg.substr(0,9)=="REPL_REQ:") {
         this.ProcessDrawableReply(msg.substr(9));
      } else if (msg.substr(0,4)=='CMD:') {
         msg = msg.substr(4);
         let p1 = msg.indexOf(":"),
             cmdid = msg.substr(0,p1),
             cmd = msg.substr(p1+1),
             reply = "REPLY:" + cmdid + ":";
         if ((cmd == "SVG") || (cmd == "PNG") || (cmd == "JPEG")) {
            this.CreateImage(cmd.toLowerCase())
                .then(res => handle.Send(reply + res));
         } else if (cmd.indexOf("ADDPANEL:") == 0) {
            let relative_path = cmd.substr(9);
            if (!this.ShowUI5Panel) {
               handle.Send(reply + "false");
            } else {

               let conn = new JSROOT.WebWindowHandle(handle.kind);

               // set interim receiver until first message arrives
               conn.SetReceiver({
                  cpainter: this,

                  OnWebsocketOpened: function() {
                  },

                  OnWebsocketMsg: function(panel_handle, msg) {
                     let panel_name = (msg.indexOf("SHOWPANEL:")==0) ? msg.substr(10) : "";
                     this.cpainter.ShowUI5Panel(panel_name, panel_handle)
                                  .then(res => handle.Send(reply + (res ? "true" : "false")));
                  },

                  OnWebsocketClosed: function() {
                     // if connection failed,
                     handle.Send(reply + "false");
                  },

                  OnWebsocketError: function() {
                     // if connection failed,
                     handle.Send(reply + "false");
                  }

               });

               let addr = handle.href;
               if (relative_path.indexOf("../")==0) {
                  let ddd = addr.lastIndexOf("/",addr.length-2);
                  addr = addr.substr(0,ddd) + relative_path.substr(2);
               } else {
                  addr += relative_path;
               }
               // only when connection established, panel will be activated
               conn.Connect(addr);
            }
         } else {
            console.log('Unrecognized command ' + cmd);
            handle.Send(reply);
         }
      } else if ((msg.substr(0,7)=='DXPROJ:') || (msg.substr(0,7)=='DYPROJ:')) {
         let kind = msg[1],
             hist = JSROOT.parse(msg.substr(7));
         this.DrawProjection(kind, hist);
      } else if (msg.substr(0,5)=='SHOW:') {
         let that = msg.substr(5),
             on = that[that.length-1] == '1';
         this.ShowSection(that.substr(0,that.length-2), on);
      } else {
         console.log("unrecognized msg len:" + msg.length + " msg:" + msg.substr(0,20));
      }
   }

   /** Submit request to RDrawable object on server side */
   RCanvasPainter.prototype.SubmitDrawableRequest = function(kind, req, painter, method) {

      if (!this._websocket || !req || !req._typename ||
          !painter.snapid || (typeof painter.snapid != "string")) return null;

      if (kind && method) {
         // if kind specified - check if such request already was submitted
         if (!painter._requests) painter._requests = {};

         let prevreq = painter._requests[kind];

         if (prevreq) {
            let tm = new Date().getTime();
            if (!prevreq._tm || (tm - prevreq._tm < 5000)) {
               prevreq._nextreq = req; // submit when got reply
               return false;
            }
            delete painter._requests[kind]; // let submit new request after timeout
         }

         painter._requests[kind] = req; // keep reference on the request
      }

      req.id = painter.snapid;

      if (method) {
         if (!this._nextreqid) this._nextreqid = 1;
         req.reqid = this._nextreqid++;
      } else {
         req.reqid = 0; // request will not be replied
      }

      let msg = JSON.stringify(req);

      if (req.reqid) {
         req._kind = kind;
         req._painter = painter;
         req._method = method;
         req._tm = new Date().getTime();

         if (!this._submreq) this._submreq = {};
         this._submreq[req.reqid] = req; // fast access to submitted requests
      }

      // console.log('Sending request ', msg.substr(0,60));

      this.SendWebsocket("REQ:" + msg);
      return req;
   }

   RCanvasPainter.prototype.submitMenuRequest = function(painter, menukind, reqid) {
      return new Promise(resolveFunc => {
         this.SubmitDrawableRequest("", {
            _typename: "ROOT::Experimental::RDrawableMenuRequest",
            menukind: menukind || "",
            menureqid: reqid, // used to identify menu request
         }, painter, resolveFunc);
      });
   }

   /** @summary Submit executable command for given painter */
   RCanvasPainter.prototype.SubmitExec = function(painter, exec, subelem) {
      console.log('SubmitExec', exec, painter.snapid, subelem);

      // snapid is intentionally ignored - only painter.snapid has to be used
      if (!this._websocket) return;

      if (subelem) {
         if ((subelem == "x") || (subelem == "y") || (subelem == "z"))
            exec = subelem + "axis#" + exec;
         else
            return console.log(`not recoginzed subelem ${subelem} in SubmitExec`);
       }

      this.SubmitDrawableRequest("", {
         _typename: "ROOT::Experimental::RDrawableExecRequest",
         exec: exec
      }, painter);
   }

   /** @summary Process reply from request to RDrawable */
   RCanvasPainter.prototype.ProcessDrawableReply = function(msg) {
      let reply = JSROOT.parse(msg);
      if (!reply || !reply.reqid || !this._submreq) return false;

      let req = this._submreq[reply.reqid];
      if (!req) return false;

      // remove reference first
      delete this._submreq[reply.reqid];

      // remove blocking reference for that kind
      if (req._painter && req._kind && req._painter._requests)
         if (req._painter._requests[req._kind] === req)
            delete req._painter._requests[req._kind];

      if (req._method)
         req._method(reply, req);

      // resubmit last request of that kind
      if (req._nextreq && !req._painter._requests[req._kind])
         this.SubmitDrawableRequest(req._kind, req._nextreq, req._painter, req._method);
   }

   RCanvasPainter.prototype.ShowSection = function(that, on) {
      switch(that) {
         case "Menu": break;
         case "StatusBar": break;
         case "Editor": break;
         case "ToolBar": break;
         case "ToolTips": this.SetTooltipAllowed(on); break;
      }
      return Promise.resolve(true);
   }

   RCanvasPainter.prototype.CompleteCanvasSnapDrawing = function() {
      if (!this.pad) return;

      // FIXME: to be remove, has nothing to do with RCanvas
      let TCanvasStatusBits = {
         kShowEventStatus  : JSROOT.BIT(15),
         kAutoExec         : JSROOT.BIT(16),
         kMenuBar          : JSROOT.BIT(17),
         kShowToolBar      : JSROOT.BIT(18),
         kShowEditor       : JSROOT.BIT(19),
         kMoveOpaque       : JSROOT.BIT(20),
         kResizeOpaque     : JSROOT.BIT(21),
         kIsGrayscale      : JSROOT.BIT(22),
         kShowToolTips     : JSROOT.BIT(23)
      };

      if (document) document.title = this.pad.fTitle;

      if (this._all_sections_showed) return;
      this._all_sections_showed = true;
      this.ShowSection("Menu", this.pad.TestBit(TCanvasStatusBits.kMenuBar));
      this.ShowSection("StatusBar", this.pad.TestBit(TCanvasStatusBits.kShowEventStatus));
      this.ShowSection("ToolBar", this.pad.TestBit(TCanvasStatusBits.kShowToolBar));
      this.ShowSection("Editor", this.pad.TestBit(TCanvasStatusBits.kShowEditor));
      this.ShowSection("ToolTips", this.pad.TestBit(TCanvasStatusBits.kShowToolTips));
   }

   /** @summary Method informs that something was changed in the canvas
     * @desc used to update information on the server (when used with web6gui)
     * @private */
   RCanvasPainter.prototype.ProcessChanges = function(kind, painter, subelem) {
      // check if we could send at least one message more - for some meaningful actions
      if (!this._websocket || !this._websocket.CanSend(2) || (typeof kind !== "string")) return;

      let msg = "";
      if (!painter) painter = this;
      switch (kind) {
         case "sbits":
            console.log("Status bits in RCanvas are changed - that to do?");
            // msg = "STATUSBITS:" + this.GetStatusBits();
            break;
         case "frame": // when moving frame
         case "zoom":  // when changing zoom inside frame
            console.log("Frame moved or zoom is changed - that to do?");
            break;
         case "pave_moved":
            console.log('TPave is moved inside RCanvas - that to do?')
            break;
         default:
            if ((kind.substr(0,5) == "exec:") && painter && painter.snapid) {
               this.SubmitExec(painter, kind.substr(5), subelem);
            } else {
               console.log("UNPROCESSED CHANGES", kind);
            }
      }

      if (msg) {
         console.log("RCanvas::ProcessChanges want ro send  " + msg.length + "  " + msg.substr(0,40));
      }
   }

   /** @summary returns true when event status area exist for the canvas */
   RCanvasPainter.prototype.HasEventStatus = function() {
      return this.has_event_status;
   }

   function drawRCanvas(divid, can /*, opt */) {
      let nocanvas = !can;
      if (nocanvas)
         can = JSROOT.Create("ROOT::Experimental::TCanvas");

      let painter = new RCanvasPainter(can);
      painter.normal_canvas = !nocanvas;

      painter.SetDivId(divid, -1); // just assign id
      painter.CreateCanvasSvg(0);
      painter.SetDivId(divid);  // now add to painters list

      jsrp.selectActivePad({ pp: painter, active: false });

      return painter.DrawPrimitives().then(() => {
         painter.AddPadButtons();
         painter.ShowButtons();
         return painter;
      });
   }

   function drawPadSnapshot(divid, snap /*, opt*/) {
      let painter = new RCanvasPainter(null);
      painter.normal_canvas = false;
      painter.batch_mode = true;
      painter.SetDivId(divid, -1); // just assign id
      return painter.RedrawPadSnap(snap).then(() => {
         painter.ShowButtons();
         return painter;
      });
   }

     /** @summary Ensure TCanvas and TFrame for the painter object
    * @param {Object} painter  - painter object to process
    * @param {Object|string} divid - HTML element or element id
    * @param {string|boolean} frame_kind  - false for no frame or "3d" for special 3D mode
    * @desc Assign divid, creates TCanvas if necessary, add to list of pad painters and */
   let ensureRCanvas = function(painter, divid, frame_kind) {
      if (!painter) return Promise.reject('Painter not provided in ensureRCanvas');

      // assign divid and pad name as required
      painter.SetDivId(divid, -1);

      // simple check - if canvas there, can use painter
      let svg_c = painter.svg_canvas();
      let noframe = (frame_kind === false) || (frame_kind == "3d") ? "noframe" : "";

      let promise = !svg_c.empty() ? Promise.resolve(true) : drawRCanvas(divid, null, noframe);

      return promise.then(() => {
         if (frame_kind === false) return;
         if (painter.svg_frame().select(".main_layer").empty())
            return drawRFrame(divid, null, (typeof frame_kind === "string") ? frame_kind : "");
      }).then(() => {
         painter.addToPadPrimitives();
         return painter;
      });
   }


   // ======================================================================================

   /**
    * @summary Painter for RPave class
    *
    * @class
    * @memberof JSROOT
    * @extends ObjectPainter
    * @param {object} pave - object to draw
    * @param {string} [opt] - object draw options
    * @param {string} [csstype] - object css kind
    * @private
    */

   function RPavePainter(pave, opt, csstype) {
      JSROOT.ObjectPainter.call(this, pave, opt);
      this.csstype = csstype || "pave";
   }

   RPavePainter.prototype = Object.create(JSROOT.ObjectPainter.prototype);

   RPavePainter.prototype.DrawContent = function() {
      return Promise.resolve(this);
   }

   RPavePainter.prototype.DrawPave = function() {

      let pw = this.pad_width(),
          ph = this.pad_height(),
          fx, fy, fw;

      if (this.frame_painter()) {
         fx = this.frame_x();
         fy = this.frame_y();
         fw = this.frame_width();
         // fh = this.frame_height();
      } else {
         let st = JSROOT.gStyle;
         fx = Math.round(st.fPadLeftMargin*pw);
         fy = Math.round(st.fPadTopMargin*ph);
         fw = Math.round((1-st.fPadLeftMargin-st.fPadRightMargin)*pw);
         // fh = Math.round((1-st.fPadTopMargin-st.fPadBottomMargin)*ph);
      }

      let visible      = this.v7EvalAttr("visible", true),
          pave_cornerx = this.v7EvalLength("cornerx", pw, 0.02),
          pave_cornery = this.v7EvalLength("cornery", ph, -0.02),
          pave_width   = this.v7EvalLength("width", pw, 0.3),
          pave_height  = this.v7EvalLength("height", ph, 0.3),
          line_width   = this.v7EvalAttr("border_width", 1),
          line_style   = this.v7EvalAttr("border_style", 1),
          line_color   = this.v7EvalColor("border_color", "black"),
          fill_color   = this.v7EvalColor("fill_color", "white"),
          fill_style   = this.v7EvalAttr("fill_style", 1);

      this.CreateG(false);

      this.draw_g.classed("most_upper_primitives", true); // this primitive will remain on top of list

      if (!visible) return Promise.resolve(this);

      if (fill_style == 0) fill_color = "none";

      let pave_x = Math.round(fx + fw + pave_cornerx - pave_width),
          pave_y = Math.round(fy + pave_cornery);

      // x,y,width,height attributes used for drag functionality
      this.draw_g.attr("transform", "translate(" + pave_x + "," + pave_y + ")")
                 .attr("x", pave_x).attr("y", pave_y)
                 .attr("width", pave_width).attr("height", pave_height);

      this.draw_g.append("svg:rect")
                 .attr("x", 0)
                 .attr("width", pave_width)
                 .attr("y", 0)
                 .attr("height", pave_height)
                 .style("stroke", line_color)
                 .attr("stroke-width", line_width)
                 .style("stroke-dasharray", jsrp.root_line_styles[line_style])
                 .attr("fill", fill_color);

      this.pave_width = pave_width;
      this.pave_height = pave_height;

      // here should be fill and draw of text

      return this.DrawContent().then(() => {

         if (JSROOT.BatchMode) return this;

         return JSROOT.require(['interactive']).then(inter => {
            // TODO: provide pave context menu as in v6
            if (JSROOT.settings.ContextMenu && this.PaveContextMenu)
               this.draw_g.on("contextmenu", this.PaveContextMenu.bind(this));

            inter.DragMoveHandler.AddDrag(this, { minwidth: 20, minheight: 20, redraw: this.SizeChanged.bind(this) });

            return this;
         });
      });
   }

   /** @summary Process interactive moving of the stats box */
   RPavePainter.prototype.SizeChanged = function() {
      this.pave_width = parseInt(this.draw_g.attr("width"));
      this.pave_height = parseInt(this.draw_g.attr("height"));

      let pave_x = parseInt(this.draw_g.attr("x")),
          pave_y = parseInt(this.draw_g.attr("y")),
          pw     = this.pad_width(),
          ph     = this.pad_height(),
          fx, fy, fw;

      if (this.frame_painter()) {
         fx = this.frame_x();
         fy = this.frame_y();
         fw = this.frame_width();
         // fh = this.frame_height();
      } else {
         let st = JSROOT.gStyle;
         fx = Math.round(st.fPadLeftMargin*pw);
         fy = Math.round(st.fPadTopMargin*ph);
         fw = Math.round((1-st.fPadLeftMargin-st.fPadRightMargin)*pw);
         // fh = Math.round((1-st.fPadTopMargin-st.fPadBottomMargin)*ph);
      }

      let changes = {};
      this.v7AttrChange(changes, "cornerx", (pave_x + this.pave_width - fx - fw) / pw);
      this.v7AttrChange(changes, "cornery", (pave_y - fy) / ph);
      this.v7AttrChange(changes, "width", this.pave_width / pw);
      this.v7AttrChange(changes, "height", this.pave_height / ph);
      this.v7SendAttrChanges(changes, false); // do not invoke canvas update on the server

      this.draw_g.select("rect")
                 .attr("width", this.pave_width)
                 .attr("height", this.pave_height);

      this.DrawContent();
   }

   RPavePainter.prototype.Redraw = function(/*reason*/) {
      this.DrawPave();
   }

   let drawPave = (divid, pave, opt) => {
      let painter = new RPavePainter(pave, opt);

      return jsrp.ensureRCanvas(painter, divid, false).then(() => painter.DrawPave());
   }

   // =======================================================================================


   function drawRFrameTitle(reason) {
      let fp = this.frame_painter();
      if (!fp)
         return console.log('no frame painter - no title');

      let fx           = this.frame_x(),
          fy           = this.frame_y(),
          fw           = this.frame_width(),
          // fh           = this.frame_height(),
          ph           = this.pad_height(),
          title        = this.GetObject(),
          title_margin = this.v7EvalLength("margin", ph, 0.02),
          title_width  = fw,
          title_height = this.v7EvalLength("height", ph, 0.05),
          textFont     = this.v7EvalFont("text", { size: 24, color: "black", align: 22 });

      this.CreateG(false);

      if (reason == 'drag') {
         title_width = parseInt(this.draw_g.attr("width"));
         title_height = parseInt(this.draw_g.attr("height"));
         let changes = {};
         this.v7AttrChange(changes, "margin", (fy - parseInt(this.draw_g.attr("y")) - title_height) / ph );
         this.v7AttrChange(changes, "height", title_height / ph);
         this.v7SendAttrChanges(changes, false); // do not invoke canvas update on the server
      } else {
         this.draw_g.attr("transform","translate(" + fx + "," + Math.round(fy-title_margin-title_height) + ")")
                    .attr("x", fx).attr("y", Math.round(fy-title_margin-title_height))
                    .attr("width",title_width).attr("height",title_height);
      }

      let arg = { x: title_width/2, y: title_height/2, text: title.fText, latex: 1 };

      this.startTextDrawing(textFont, 'font');

      this.drawText(arg);

      this.finishTextDrawing();

      if (!JSROOT.BatchMode)
         JSROOT.require(['interactive'])
               .then(inter => inter.DragMoveHandler.AddDrag(this, { minwidth: 20, minheight: 20, no_change_x: true, redraw: this.Redraw.bind(this,'drag') }));
   }

   ////////////////////////////////////////////////////////////////////////////////////////////

   JSROOT.v7.ExtractRColor = function(rcolor) {
      if (rcolor.fName)
         return rcolor.fName;

      if (rcolor.fRGBA.length == 3)
         return "rgb(" + rcolor.fRGBA[0] + "," + rcolor.fRGBA[1] + "," + rcolor.fRGBA[2] + ")";

      if (rcolor.fRGBA.length == 4)
         return "rgba(" + rcolor.fRGBA[0] + "," + rcolor.fRGBA[1] + "," + rcolor.fRGBA[2] + "," + rcolor.fRGBA[3] + ")";

      return "black";
   }

   JSROOT.registerMethods("ROOT::Experimental::RPalette", {

      getColor: function(indx) {
         return this.palette[indx];
      },

      getContourIndex: function(zc) {
         let cntr = this.fContour, l = 0, r = cntr.length-1, mid;

         if (zc < cntr[0]) return -1;
         if (zc >= cntr[r]) return r-1;

         if (this.fCustomContour) {
            while (l < r-1) {
               mid = Math.round((l+r)/2);
               if (cntr[mid] > zc) r = mid; else l = mid;
            }
            return l;
         }

         // last color in palette starts from level cntr[r-1]
         return Math.floor((zc-cntr[0]) / (cntr[r-1] - cntr[0]) * (r-1));
      },

      getContourColor: function(zc) {
         let zindx = this.getContourIndex(zc);
         return (zindx < 0) ? "" : this.getColor(zindx);
      },

      GetContour: function() {
         return this.fContour && (this.fContour.length > 1) ? this.fContour : null;
      },

      DeleteContour: function() {
         delete this.fContour;
      },

      CreatePaletteColors: function(len) {
         let arr = [], indx = 0;

         while (arr.length < len) {
            let value = arr.length / (len-1);

            let entry = this.fColors[indx];

            if ((Math.abs(entry.fOrdinal - value)<0.0001) || (indx == this.fColors.length-1)) {
               arr.push(JSROOT.v7.ExtractRColor(entry.fColor));
               continue;
            }

            let next = this.fColors[indx+1];
            if (next.fOrdinal <= value) {
               indx++;
               continue;
            }

            let dist = next.fOrdinal - entry.fOrdinal,
                r1 = (next.fOrdinal - value) / dist,
                r2 = (value - entry.fOrdinal) / dist;

            // interpolate
            let col1 = d3.rgb(JSROOT.v7.ExtractRColor(entry.fColor));
            let col2 = d3.rgb(JSROOT.v7.ExtractRColor(next.fColor));

            let color = d3.rgb(Math.round(col1.r*r1 + col2.r*r2), Math.round(col1.g*r1 + col2.g*r2), Math.round(col1.b*r1 + col2.b*r2));

            arr.push(color.toString());
         }

         return arr;
      },

      CreateContour: function(logz, nlevels, zmin, zmax, zminpositive) {
         this.fContour = [];
         delete this.fCustomContour;
         this.colzmin = zmin;
         this.colzmax = zmax;

         if (logz) {
            if (this.colzmax <= 0) this.colzmax = 1.;
            if (this.colzmin <= 0)
               if ((zminpositive===undefined) || (zminpositive <= 0))
                  this.colzmin = 0.0001*this.colzmax;
               else
                  this.colzmin = ((zminpositive < 3) || (zminpositive>100)) ? 0.3*zminpositive : 1;
            if (this.colzmin >= this.colzmax) this.colzmin = 0.0001*this.colzmax;

            let logmin = Math.log(this.colzmin)/Math.log(10),
                logmax = Math.log(this.colzmax)/Math.log(10),
                dz = (logmax-logmin)/nlevels;
            this.fContour.push(this.colzmin);
            for (let level=1; level<nlevels; level++)
               this.fContour.push(Math.exp((logmin + dz*level)*Math.log(10)));
            this.fContour.push(this.colzmax);
            this.fCustomContour = true;
         } else {
            if ((this.colzmin === this.colzmax) && (this.colzmin !== 0)) {
               this.colzmax += 0.01*Math.abs(this.colzmax);
               this.colzmin -= 0.01*Math.abs(this.colzmin);
            }
            let dz = (this.colzmax-this.colzmin)/nlevels;
            for (let level=0; level<=nlevels; level++)
               this.fContour.push(this.colzmin + dz*level);
         }

         if (!this.palette || (this.palette.length != nlevels))
            this.palette = this.CreatePaletteColors(nlevels);
      }

   });

   // =============================================================

   /** @summary painter for RPalette
    *
    * @class
    * @memberof JSROOT
    * @extends JSROOT.ObjectPainter
    * @param {object} palette - RPalette object
    * @private
    */

   function RPalettePainter(palette) {
      JSROOT.ObjectPainter.call(this, palette);
      this.csstype = "palette";
   }

   RPalettePainter.prototype = Object.create(JSROOT.ObjectPainter.prototype);

   RPalettePainter.prototype.GetPalette = function() {
      let drawable = this.GetObject();
      let pal = drawable ? drawable.fPalette : null;

      if (pal && !pal.getColor)
         JSROOT.addMethods(pal, "ROOT::Experimental::RPalette");

      return pal;
   }

   /** @summary Draw palette */
   RPalettePainter.prototype.DrawPalette = function(after_resize) {

      let palette = this.GetPalette(),
          contour = palette.GetContour(),
          framep = this.frame_painter();

      if (!contour)
         return console.log('no contour - no palette');

      // frame painter must  be there
      if (!framep)
         return console.log('no frame painter - no palette');

      let zmin         = contour[0],
          zmax         = contour[contour.length-1],
          fx           = this.frame_x(),
          fy           = this.frame_y(),
          fw           = this.frame_width(),
          fh           = this.frame_height(),
          pw           = this.pad_width(),
          visible      = this.v7EvalAttr("visible", true),
          palette_width, palette_height;

      if (after_resize) {
         palette_width = parseInt(this.draw_g.attr("width"));
         palette_height = parseInt(this.draw_g.attr("height"));

         let changes = {};
         this.v7AttrChange(changes, "margin", (parseInt(this.draw_g.attr("x")) - fx - fw) / pw);
         this.v7AttrChange(changes, "size", palette_width / pw);
         this.v7SendAttrChanges(changes, false); // do not invoke canvas update on the server
      } else {
          let palette_margin = this.v7EvalLength("margin", pw, 0.02),
              palette_x = Math.round(fx + fw + palette_margin),
              palette_y = fy;

          palette_width = this.v7EvalLength("size", pw, 0.05);
          palette_height = fh;

          // x,y,width,height attributes used for drag functionality
          this.draw_g.attr("transform","translate(" + palette_x +  "," + palette_y + ")")
                     .attr("x", palette_x).attr("y", palette_y)
                     .attr("width", palette_width).attr("height", palette_height);
      }

      this.draw_g.selectAll("rect").remove();

      if (!visible) return;

      let g_btns = this.draw_g.select(".colbtns");
      if (g_btns.empty())
         g_btns = this.draw_g.append("svg:g").attr("class", "colbtns");
      else
         g_btns.selectAll().remove();

      g_btns.append("svg:rect")
          .attr("x", 0)
          .attr("width", palette_width)
          .attr("y", 0)
          .attr("height", palette_height)
          .style("stroke", "black")
          .attr("fill", "none");

      framep.z_handle.ConfigureAxis("zaxis", zmin, zmax, zmin, zmax, true, [palette_height, 0], -palette_height, { reverse: false });

      for (let i=0;i<contour.length-1;++i) {
         let z0 = framep.z_handle.gr(contour[i]),
             z1 = framep.z_handle.gr(contour[i+1]),
             col = palette.getContourColor((contour[i]+contour[i+1])/2);

         let r = g_btns.append("svg:rect")
                     .attr("x", 0)
                     .attr("y", Math.round(z1))
                     .attr("width", palette_width)
                     .attr("height", Math.round(z0) - Math.round(z1))
                     .style("fill", col)
                     .style("stroke", col)
                     .property("fill0", col)
                     .property("fill1", d3.rgb(col).darker(0.5).toString())

         if (this.IsTooltipAllowed())
            r.on('mouseover', function() {
               d3.select(this).transition().duration(100).style("fill", d3.select(this).property('fill1'));
            }).on('mouseout', function() {
               d3.select(this).transition().duration(100).style("fill", d3.select(this).property('fill0'));
            }).append("svg:title").text(contour[i].toFixed(2) + " - " + contour[i+1].toFixed(2));

         if (JSROOT.settings.Zooming)
            r.on("dblclick", () => framep.Unzoom("z"));
      }

      framep.z_handle.max_tick_size = Math.round(palette_width*0.3);

      let promise = framep.z_handle.drawAxis(this.draw_g, "translate(" + palette_width + "," + palette_height + ")", -1);

      if (JSROOT.BatchMode) return;

      promise.then(() => JSROOT.require(['interactive'])).then(inter => {

         if (JSROOT.settings.ContextMenu)
            this.draw_g.on("contextmenu", evnt => {
               evnt.stopPropagation(); // disable main context menu
               evnt.preventDefault();  // disable browser context menu
               jsrp.createMenu(this, evnt).then(menu => {
                 menu.add("header:Palette");
                 framep.z_handle.FillAxisContextMenu(menu, "z");
                 menu.show();
               });
            });

         if (!after_resize)
            inter.DragMoveHandler.AddDrag(this, { minwidth: 20, minheight: 20, no_change_y: true, redraw: this.DrawPalette.bind(this, true) });

         if (!JSROOT.settings.Zooming) return;

         let doing_zoom = false, sel1 = 0, sel2 = 0, zoom_rect, zoom_rect_visible, moving_labels, last_pos;

         let moveRectSel = evnt => {

            if (!doing_zoom) return;
            evnt.preventDefault();

            last_pos = d3.pointer(evnt, this.draw_g.node());

            if (moving_labels)
               return framep.z_handle.processLabelsMove('move', last_pos);

            sel2 = Math.min(Math.max(last_pos[1], 0), palette_height);

            let h = Math.abs(sel2-sel1);

            if (!zoom_rect_visible && (h > 1)) {
               zoom_rect.style('display', null);
               zoom_rect_visible = true;
            }

            zoom_rect.attr("y", Math.min(sel1, sel2))
                     .attr("height", h);
         }

         let endRectSel = evnt => {
            if (!doing_zoom) return;

            evnt.preventDefault();
            d3.select(window).on("mousemove.colzoomRect", null)
                             .on("mouseup.colzoomRect", null);
            zoom_rect.remove();
            zoom_rect = null;
            doing_zoom = false;

            if (moving_labels) {
               framep.z_handle.processLabelsMove('stop', last_pos);
            } else {
               let z = framep.z_handle.func, z1 = z.invert(sel1), z2 = z.invert(sel2);
               this.frame_painter().Zoom("z", Math.min(z1, z2), Math.max(z1, z2));
            }
         }

         let startRectSel = evnt => {
            // ignore when touch selection is activated
            if (doing_zoom) return;
            doing_zoom = true;

            evnt.preventDefault();
            evnt.stopPropagation();

            last_pos = d3.pointer(evnt, this.draw_g.node());

            sel1 = sel2 = last_pos[1];
            zoom_rect_visible = false;
            moving_labels = false;
            zoom_rect = g_btns
                 .append("svg:rect")
                 .attr("class", "zoom")
                 .attr("id", "colzoomRect")
                 .attr("x", "0")
                 .attr("width", palette_width)
                 .attr("y", sel1)
                 .attr("height", 1)
                 .style('display', 'none');

            d3.select(window).on("mousemove.colzoomRect", moveRectSel)
                             .on("mouseup.colzoomRect", endRectSel, true);

            setTimeout(() => {
               if (!zoom_rect_visible && doing_zoom)
                  moving_labels = framep.z_handle.processLabelsMove('start', last_pos);
            }, 500)
         }

         let assignHandlers = () => {
            this.draw_g.selectAll(".axis_zoom, .axis_labels")
                       .on("mousedown", startRectSel)
                       .on("dblclick", () => framep.Unzoom("z"));

            if (JSROOT.settings.ZoomWheel)
               this.draw_g.on("wheel", evnt => {
                  evnt.stopPropagation();
                  evnt.preventDefault();

                  let pos = d3.pointer(evnt, this.draw_g.node()),
                      coord = 1 - pos[1] / palette_height;

                  let item = framep.z_handle.analyzeWheelEvent(evnt, coord);
                  if (item.changed)
                     framep.Zoom("z", item.min, item.max);
               });
         }

         framep.z_handle.setAfterDrawHandler(assignHandlers);

         assignHandlers();
      });
   }

   let drawPalette = (divid, palette, opt) => {
      let painter = new RPalettePainter(palette, opt);

      return jsrp.ensureRCanvas(painter, divid, false).then(() => {
         painter.CreateG(false); // just create container, real drawing will be done by histogram
         return painter;
      });
   }

   // JSROOT.addDrawFunc({ name: "ROOT::Experimental::RPadDisplayItem", icon: "img_canvas", func: drawPad, opt: "" });

   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RHist1Drawable", icon: "img_histo1d", prereq: "v7hist", func: "JSROOT.v7.drawHist1", opt: "" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RHist2Drawable", icon: "img_histo2d", prereq: "v7hist", func: "JSROOT.v7.drawHist2", opt: "" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RHist3Drawable", icon: "img_histo3d", prereq: "v7hist3d", func: "JSROOT.v7.drawHist3", opt: "" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RHistDisplayItem", icon: "img_histo1d", prereq: "v7hist", func: "JSROOT.v7.drawHistDisplayItem", opt: "" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RText", icon: "img_text", prereq: "v7more", func: "JSROOT.v7.drawText", opt: "", direct: "v7", csstype: "text" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RFrameTitle", icon: "img_text", func: drawRFrameTitle, opt: "", direct: "v7", csstype: "title" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RPaletteDrawable", icon: "img_text", func: drawPalette, opt: "" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RDisplayHistStat", icon: "img_pavetext", prereq: "v7hist", func: "JSROOT.v7.drawHistStats", opt: "" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RLine", icon: "img_graph", prereq: "v7more", func: "JSROOT.v7.drawLine", opt: "", direct: "v7", csstype: "line" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RBox", icon: "img_graph", prereq: "v7more", func: "JSROOT.v7.drawBox", opt: "", direct: "v7", csstype: "box" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RMarker", icon: "img_graph", prereq: "v7more", func: "JSROOT.v7.drawMarker", opt: "", direct: "v7", csstype: "marker" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RPave", icon: "img_pavetext", func: drawPave, opt: "" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RLegend", icon: "img_graph", prereq: "v7more", func: "JSROOT.v7.drawLegend", opt: "" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RPaveText", icon: "img_pavetext", prereq: "v7more", func: "JSROOT.v7.drawPaveText", opt: "" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RFrame", icon: "img_frame", func: drawRFrame, opt: "" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RAxisDrawable", icon: "img_frame", func: drawRAxis, opt: "" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RAxisLabelsDrawable", icon: "img_frame", func: drawRAxis, opt: "" });

   JSROOT.v7.RAxisPainter = RAxisPainter;
   JSROOT.v7.RFramePainter = RFramePainter;
   JSROOT.v7.RPalettePainter = RPalettePainter;
   JSROOT.v7.RPadPainter = RPadPainter;
   JSROOT.v7.RCanvasPainter = RCanvasPainter;
   JSROOT.v7.RPavePainter = RPavePainter;
   JSROOT.v7.drawRAxis = drawRAxis;
   JSROOT.v7.drawRFrame = drawRFrame;
   JSROOT.v7.drawPad = drawPad;
   JSROOT.v7.drawRCanvas = drawRCanvas;
   JSROOT.v7.drawPadSnapshot = drawPadSnapshot;
   JSROOT.v7.drawPave = drawPave;
   JSROOT.v7.drawRFrameTitle = drawRFrameTitle;

   jsrp.ensureRCanvas = ensureRCanvas;

   return JSROOT;

});
