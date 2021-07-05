/// @file JSRoot.v7gpad.js
/// JavaScript ROOT graphics for ROOT v7 classes

JSROOT.define(['d3', 'painter'], (d3, jsrp) => {

   "use strict";

   JSROOT.v7 = {}; // placeholder for v7-relevant code

   /** @summary Evaluate v7 attributes using fAttr storage and configured RStyle
     * @private */
   JSROOT.ObjectPainter.prototype.v7EvalAttr = function(name, dflt) {
      let obj = this.getObject();
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
               let value = block.map.m[name.toLowerCase()];
               if (value) return type_check(value.v);
            }
         }
      }

      return dflt;
   }

   /** @summary Set v7 attributes value
     * @private */
   JSROOT.ObjectPainter.prototype.v7SetAttr = function(name, value) {
      let obj = this.getObject();
      if (this.cssprefix) name = this.cssprefix + name;

      if (obj && obj.fAttr && obj.fAttr.m)
         obj.fAttr.m[name] = { v: value };
   }

   /** @summary Decode pad length from string, return pixel value
     * @private */
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
         if (!Number.isFinite(v)) {
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
      let val = this.v7EvalAttr(name, "");
      if (!val || (typeof val != "string")) return dflt;

      if (val == "auto") {
         let pp = this.getPadPainter();
         if (pp && (pp._auto_color_cnt !== undefined)) {
            let pal = pp.getHistPalette();
            let cnt = pp._auto_color_cnt++, num = pp._num_primitives - 1;
            if (num < 2) num = 2;
            val = pal ? pal.getColorOrdinal((cnt % num) / num) : "blue";
            if (!this._auto_colors) this._auto_colors = {};
            this._auto_colors[name] = val;
         } else if (this._auto_colors && this._auto_colors[name]) {
            val = this._auto_colors[name];
         } else {
            console.error(`Autocolor ${name} not defined yet - please check code`);
            val = "";
         }
      } else if (val[0]=="[") {
         let ordinal = parseFloat(val.substr(1, val.length-2));
         val = "black";
         if (Number.isFinite(ordinal)) {
             let pp = this.getPadPainter(),
                 pal = pp ? pp.getHistPalette() : null;
             if (pal) val = pal.getColorOrdinal(ordinal);
         }
      }
      return val;
   }

   /** @summary Evaluate RAttrText properties
     * @returns {Object} FontHandler, can be used directly for the text drawing
     * @private */
   JSROOT.ObjectPainter.prototype.v7EvalFont = function(name, dflts, fontScale) {

      if (!dflts) dflts = {}; else
      if (typeof dflts == "number") dflts = { size: dflts };

      let text_size   = this.v7EvalAttr(name + "_size", dflts.size || 12),
          text_angle  = this.v7EvalAttr(name + "_angle", 0),
          text_align  = this.v7EvalAttr(name + "_align", dflts.align || "none"),
          text_color  = this.v7EvalColor(name + "_color", dflts.color || "none"),
          font_family = this.v7EvalAttr(name + "_font_family", "Arial"),
          font_style  = this.v7EvalAttr(name + "_font_style", ""),
          font_weight = this.v7EvalAttr(name + "_font_weight", "");

       if (typeof text_size == "string") text_size = parseFloat(text_size);
       if (!Number.isFinite(text_size) || (text_size <= 0)) text_size = 12;
       if (!fontScale) fontScale = this.getPadPainter().getPadHeight() || 10;

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

      this.fillatt.setSolidColor(fill_color || "none");
   }

   /** @summary Create this.lineatt object based on v7 line attributes
     * @private */
   JSROOT.ObjectPainter.prototype.createv7AttLine = function(prefix) {
      if (!prefix || (typeof prefix != "string")) prefix = "line_";

      let line_color = this.v7EvalColor(prefix + "color", "black"),
          line_width = this.v7EvalAttr(prefix + "width", 1),
          line_style = this.v7EvalAttr(prefix + "style", 1);

      this.createAttLine({ color: line_color, width: line_width, style: line_style });

      if (prefix == "border_") {
         this.lineatt.rx = this.v7EvalAttr(prefix + "rx", 0);
         this.lineatt.ry = this.v7EvalAttr(prefix + "ry", 0);
      }
   }

    /** @summary Create this.markeratt object based on v7 attributes
      * @private */
   JSROOT.ObjectPainter.prototype.createv7AttMarker = function(prefix) {
      if (!prefix || (typeof prefix != "string")) prefix = "marker_";

      let marker_color = this.v7EvalColor(prefix + "color", "black"),
          marker_size = this.v7EvalAttr(prefix + "size", 1),
          marker_style = this.v7EvalAttr(prefix + "style", 1);

      this.createAttMarker({ color: marker_color, size: marker_size, style: marker_style });
   }

   /** @summary Create RChangeAttr, which can be applied on the server side
     * @private */
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
      let canp = this.getCanvPainter();
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
      let canp = this.getCanvPainter();
      if (!canp || !canp.submitDrawableRequest) return null;

      // special situation when snapid not yet assigned - just keep ref until snapid is there
      // maybe keep full list - for now not clear if really needed
      if (!this.snapid) {
         this._pending_request = { _kind: kind, _req: req, _method: method };
         return req;
      }

      return canp.submitDrawableRequest(kind, req, this, method);
   }

   /** @summary Assign snapid to the painter
     * @desc Overwrite default method
     * @private */
   JSROOT.ObjectPainter.prototype.assignSnapId = function(id) {
      this.snapid = id;
      if (this.snapid && this._pending_request) {
         let req = this._pending_request;
         this.v7SubmitRequest(req._kind, req._req, req._method);
         delete this._pending_request;
      }
   }

   JSROOT.v7.CommMode = { kNormal: 1, kLessTraffic: 2, kOffline: 3 }

   /** @summary Return communication mode with the server
    * @desc Using constants from {@link JSROOT.v7.CommMode} object
    * kOffline means no server there,
    * kLessTraffic advise not to send commands if offline functionality available
    * kNormal is standard functionality with RCanvas on server side
    * @private */
   JSROOT.ObjectPainter.prototype.v7CommMode = function() {
      let canp = this.getCanvPainter();
      if (!canp || !canp.submitDrawableRequest || !canp._websocket)
         return JSROOT.v7.CommMode.kOffline;

      return JSROOT.v7.CommMode.kNormal;
   }

   // ================================================================================

   /**
    * @summary Axis painter for v7
    *
    * @class
    * @memberof JSROOT
    * @extends JSROOT.AxisBasePainter
    * @param {object|string} dom - identifier or dom element
    * @private
    */

   function RAxisPainter(dom, arg1, axis, cssprefix) {
      let drawable = cssprefix ? arg1.getObject() : arg1;
      this.axis = axis;
      JSROOT.AxisBasePainter.call(this, dom, drawable);
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
   RAxisPainter.prototype.getAxisType = function() { return "RAttrAxis"; }

   /** @summary Configure only base parameters, later same handle will be used for drawing  */
   RAxisPainter.prototype.configureZAxis = function(name, fp) {
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
   RAxisPainter.prototype.configureAxis = function(name, min, max, smin, smax, vertical, frame_range, axis_range, opts) {
      if (!opts) opts = {};
      this.name = name;
      this.full_min = min;
      this.full_max = max;
      this.kind = "normal";
      this.vertical = vertical;
      this.log = false;
      let _log = this.v7EvalAttr("log", 0),
          _symlog = this.v7EvalAttr("symlog", 0);
      this.reverse = opts.reverse || false;

      if (this.v7EvalAttr("time")) {
         this.kind = 'time';
         this.timeoffset = 0;
         let toffset = this.v7EvalAttr("timeOffset");
         if (toffset !== undefined) {
            toffset = parseFloat(toffset);
            if (Number.isFinite(toffset)) this.timeoffset = toffset*1000;
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
         this.func = d3.scaleTime().domain([this.convertDate(smin), this.convertDate(smax)]);
      } else if (_symlog && (_symlog > 0)) {
         this.symlog = _symlog;
         this.func = d3.scaleSymlog().constant(_symlog).domain([smin,smax]);
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
         this.gr = val => this.func(this.convertDate(val));
      else if (this.log)
         this.gr = val => (val < this.scale_min) ? (this.vertical ? this.func.range()[0]+5 : -5) : this.func(val);
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
             tf1 = this.v7EvalAttr("timeFormat", ""),
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
         let labels = this.getObject().fLabels;
         if (labels && (indx >= 0) && (indx < labels.length))
            return labels[indx];
      }
      return null;
   }

   /** @summary Creates array with minor/middle/major ticks */
   RAxisPainter.prototype.createTicks = function(only_major_as_array, optionNoexp, optionNoopt, optionInt) {

      if (optionNoopt && this.nticks && (this.kind == "normal")) this.noticksopt = true;

      let handle = { nminor: 0, nmiddle: 0, nmajor: 0, func: this.func };

      handle.minor = handle.middle = handle.major = this.produceTicks(this.nticks);

      if (only_major_as_array) {
         let res = handle.major, delta = (this.scale_max - this.scale_min)*1e-5;
         if (res[0] > this.scale_min + delta) res.unshift(this.scale_min);
         if (res[res.length-1] < this.scale_max - delta) res.push(this.scale_max);
         return res;
      }

      if ((this.nticks2 > 1) && (!this.log || (this.logbase === 10))) {
         handle.minor = handle.middle = this.produceTicks(handle.major.length, this.nticks2);

         let gr_range = Math.abs(this.func.range()[1] - this.func.range()[0]);

         // avoid black filling by middle-size
         if ((handle.middle.length <= handle.major.length) || (handle.middle.length > gr_range/3.5)) {
            handle.minor = handle.middle = handle.major;
         } else if ((this.nticks3 > 1) && !this.log)  {
            handle.minor = this.produceTicks(handle.middle.length, this.nticks3);
            if ((handle.minor.length <= handle.middle.length) || (handle.minor.length > gr_range/1.7)) handle.minor = handle.middle;
         }
      }

      handle.reset = function() {
         this.nminor = this.nmiddle = this.nmajor = 0;
      };

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
      };

      handle.last_major = function() {
         return (this.kind !== 1) ? false : this.nmajor == this.major.length;
      };

      handle.next_major_grpos = function() {
         if (this.nmajor >= this.major.length) return null;
         return this.func(this.major[this.nmajor]);
      };

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
   RAxisPainter.prototype.isCenteredLabels = function() {
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
            this.changeAxisAttr(1, "labels_offset", this.labelsOffset/this.scaling_size);
         }
      }

      return true;
   }

   /** @summary Add interactive elements to draw axes title */
   RAxisPainter.prototype.addTitleDrag = function(title_g, side) {
      if (!JSROOT.settings.MoveResize || JSROOT.batch_mode) return;

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
                swap = this.isReverseAxis() ? 2 : 0;
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

               let set_x, set_y,
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

               this.changeAxisAttr(0, "title_position", this.titlePos, "title_offset", this.titleOffset/this.scaling_size);

               drag_rect.remove();
               drag_rect = null;
            });

      title_g.style("cursor", "move").call(drag_move);
   }

   /** @summary checks if value inside graphical range, taking into account delta */
   RAxisPainter.prototype.isInsideGrRange = function(pos, delta1, delta2) {
      if (!delta1) delta1 = 0;
      if (delta2 === undefined) delta2 = delta1;
      if (this.gr_range < 0)
         return (pos >= this.gr_range - delta2) && (pos <= delta1);
      return (pos >= -delta1) && (pos <= this.gr_range + delta2);
   }

   /** @summary returns graphical range */
   RAxisPainter.prototype.getGrRange = function(delta) {
      if (!delta) delta = 0;
      if (this.gr_range < 0)
         return this.gr_range - delta;
      return this.gr_range + delta;
   }

   /** @summary If axis direction is negative coordinates direction */
   RAxisPainter.prototype.isReverseAxis = function() {
      return !this.vertical !== (this.getGrRange() > 0);
   }

   /** @summary Draw axis ticks
     * @private */
   RAxisPainter.prototype.drawMainLine = function(axis_g) {
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

         if ((this.startingSize || this.endingSize) && !this.isInsideGrRange(grpos, -Math.abs(this.startingSize), -Math.abs(this.endingSize))) continue;

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
               .style('stroke', this.ticksColor || this.lineatt.color)
               .style('stroke-width', !this.ticksWidth || (this.ticksWidth == 1) ? null : this.ticksWidth);

       let gap0 = Math.round(0.25*this.ticksSize), gap = Math.round(1.25*this.ticksSize);
       return { "-1": (side > 0) || ticks_plusminus ? gap : gap0,
                "1": (side < 0) || ticks_plusminus ? gap : gap0 };
   }

   /** @summary Performs labels drawing
     * @returns {Promise} wwith gaps in both direction */
   RAxisPainter.prototype.drawLabels = function(axis_g, side, gaps) {
      let center_lbls = this.isCenteredLabels(),
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
            painter.scaleTextDrawing(1/scale, label_g);
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
            if (!this.isInsideGrRange(pos, 5)) continue;
         }

         maxtextlen = Math.max(maxtextlen, lbl.length);

         pos -= this.axis_shift;

         if ((this.startingSize || this.endingSize) && !this.isInsideGrRange(pos, -Math.abs(this.startingSize), -Math.abs(this.endingSize))) continue;

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
         this.drawText({ x: this.vertical ? side*5 : this.getGrRange(5),
                         y: this.has_obstacle ? fix_coord : (this.vertical ? this.getGrRange(3) : -3*side),
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
      if (JSROOT.settings.Zooming && !this.disable_zooming && !JSROOT.batch_mode) {
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

      this.titleFont = this.v7EvalFont("title", { size: 0.03 }, this.getPadPainter().getPadHeight());
      this.titleFont.roundAngle(180, this.vertical ? 270 : 0);

      this.titleOffset = this.v7EvalLength("title_offset", this.scaling_size, 0);
      this.titlePos = title_position;

      let rotated = this.isTitleRotated();

      this.startTextDrawing(this.titleFont, 'font', title_g);

      this.title_align = center ? "middle" : (opposite ^ (this.isReverseAxis() || rotated) ? "begin" : "end");

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
      this.ticksWidth = this.v7EvalAttr("ticks_width", 1);
      this.labelsOffset = this.v7EvalLength("labels_offset", this.scaling_size, 0);
      this.optionUnlab = this.v7EvalAttr("labels_hide", false);

      this.fTitle = this.v7EvalAttr("title_value", "");

      if (this.max_tick_size && (this.ticksSize > this.max_tick_size)) this.ticksSize = this.max_tick_size;
   }

   /** @summary Performs axis drawing
     * @returns {Promise} which resolved when drawing is completed */
   RAxisPainter.prototype.drawAxis = function(layer, transform, side) {
      let axis_g = layer, rect = this.getPadPainter().getPadRect();

      if (side === undefined) side = 1;

      if (!this.standalone) {
         axis_g = layer.select("." + this.name + "_container");
         if (axis_g.empty())
            axis_g = layer.append("svg:g").attr("class",this.name + "_container");
         else
            axis_g.selectAll("*").remove();
      }

      axis_g.attr("transform", transform || null);

      this.scaling_size = this.vertical ? rect.width : rect.height;
      this.extractDrawAttributes();
      this.axis_g = axis_g;
      this.side = side;

      if (this.ticksSide == "invert") side = -side;

      if (this.standalone)
         this.drawMainLine(axis_g);

      let optionNoopt = false,  // no ticks position optimization
          optionInt = false,    // integer labels
          optionNoexp = false;  // do not create exp

      this.handle = this.createTicks(false, optionNoexp, optionNoopt, optionInt);

      // first draw ticks
      let tgaps = this.drawTicks(axis_g, side, true);

      // draw labels
      let labelsPromise = this.optionUnlab ? Promise.resolve(tgaps) : this.drawLabels(axis_g, side, tgaps);

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
         this.drawMainLine(this.axis_g);

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
   RAxisPainter.prototype.zoomStandalone = function(min,max) {
      this.changeAxisAttr(1, "zoomMin", min, "zoomMax", max);
   }

   /** @summary Redraw axis, used in standalone mode for RAxisDrawable */
   RAxisPainter.prototype.redraw = function() {

      let drawable = this.getObject(),
          pp   = this.getPadPainter(),
          pos  = pp.getCoordinate(drawable.fPos),
          len  = pp.getPadLength(drawable.fVertical, drawable.fLength),
          reverse = this.v7EvalAttr("reverse", false),
          labels_len = drawable.fLabels.length,
          min = (labels_len > 0) ? 0 : this.v7EvalAttr("min", 0),
          max = (labels_len > 0) ? labels_len : this.v7EvalAttr("max", 100);

      // in vertical direction axis drawn in negative direction
      if (drawable.fVertical) len -= pp.getPadHeight();

      let smin = this.v7EvalAttr("zoomMin"),
          smax = this.v7EvalAttr("zoomMax");
      if (smin === smax) {
         smin = min; smax = max;
      }

      this.configureAxis("axis", min, max, smin, smax, drawable.fVertical, undefined, len, { reverse: reverse, labels: labels_len > 0 });

      this.createG();

      this.standalone = true;  // no need to clean axis container

      let promise = this.drawAxis(this.draw_g, "translate(" + pos.x + "," + pos.y +")");

      if (JSROOT.batch_mode) return promise;

      return promise.then(() => JSROOT.require('interactive')).then(inter => {
         if (JSROOT.settings.ContextMenu)
            this.draw_g.on("contextmenu", evnt => {
               evnt.stopPropagation(); // disable main context menu
               evnt.preventDefault();  // disable browser context menu
               jsrp.createMenu(evnt, this).then(menu => {
                 menu.add("header:RAxisDrawable");
                 menu.add("Unzoom", () => this.zoomStandalone());
                 this.fillAxisContextMenu(menu, "");
                 menu.show();
               });
            });

         // attributes required only for moving, has no effect for drawing
         this.draw_g.attr("x", pos.x).attr("y", pos.y)
                    .attr("width", this.vertical ? 10 : len)
                    .attr("height", this.vertical ? len : 10);

         inter.addDragHandler(this, { only_move: true, redraw: this.positionChanged.bind(this) });

         this.draw_g.on("dblclick", () => this.zoomStandalone());

         if (JSROOT.settings.ZoomWheel)
            this.draw_g.on("wheel", evnt => {
               evnt.stopPropagation();
               evnt.preventDefault();

               let pos = d3.pointer(evnt, this.draw_g.node()),
                   coord = this.vertical ? (1 - pos[1] / len) : pos[0] / len,
                   item = this.analyzeWheelEvent(evnt, coord);

               if (item.changed) this.zoomStandalone(item.min, item.max);
            });

      });
   }

   /** @summary Process interactive moving of the axis drawing */
   RAxisPainter.prototype.positionChanged = function() {
      let axis_x = parseInt(this.draw_g.attr("x")),
          axis_y = parseInt(this.draw_g.attr("y")),
          drawable = this.getObject(),
          rect = this.getPadPainter().getPadRect(),
          xn = axis_x / rect.width,
          yn = 1 - axis_y / rect.height;

      drawable.fPos.fHoriz.fArr = [ xn ];
      drawable.fPos.fVert.fArr = [ yn ];

      this.submitCanvExec("SetPos({" + xn.toFixed(4) + "," + yn.toFixed(4) + "})");
   }

   /** @summary Change axis attribute, submit changes to server and redraw axis when specified
     * @desc Arguments as redraw_mode, name1, value1, name2, value2, ... */
   RAxisPainter.prototype.changeAxisAttr = function(redraw_mode) {
      let changes = {}, indx = 1;
      while (indx < arguments.length - 1) {
         this.v7AttrChange(changes, arguments[indx], arguments[indx+1]);
         this.v7SetAttr(arguments[indx], arguments[indx+1]);
         indx += 2;
      }
      this.v7SendAttrChanges(changes, false); // do not invoke canvas update on the server
      if (redraw_mode === 1) {
         if (this.standalone)
            this.redraw();
         else
            this.drawAxisAgain();
      } else if (redraw_mode)
         this.redrawPad();
   }

   /** @summary Change axis log scale kind */
   RAxisPainter.prototype.changeAxisLog = function(arg) {
      if ((this.kind == "labels") || (this.kind == 'time')) return;
      if (arg === 'toggle') arg = this.log ? 0 : 10;

      arg = parseFloat(arg);
      if (Number.isFinite(arg)) this.changeAxisAttr(2, "log", arg, "symlog", 0);
   }

   /** @summary Provide context menu for axis */
   RAxisPainter.prototype.fillAxisContextMenu = function(menu, kind) {

      if (kind) menu.add("Unzoom", () => this.getFramePainter().unzoom(kind));

      menu.add("sub:Log scale", () => this.changeAxisLog('toggle'));
      menu.addchk(!this.log && !this.symlog, "linear", 0, arg => this.changeAxisLog(arg));
      menu.addchk(this.log && !this.symlog && (this.logbase==10), "log10", () => this.changeAxisLog(10));
      menu.addchk(this.log && !this.symlog && (this.logbase==2), "log2", () => this.changeAxisLog(2));
      menu.addchk(this.log && !this.symlog && Math.abs(this.logbase - Math.exp(1)) < 0.1, "ln", () => this.changeAxisLog(Math.exp(1)));
      menu.addchk(!this.log && this.symlog, "symlog", 0, () => {
         menu.input("set symlog constant", this.symlog || 10, "float").then(v => this.changeAxisAttr(2,"symlog", v));
      });
      menu.add("endsub:");

      menu.add("sub:Ticks");
      menu.addRColorMenu("color", this.ticksColor, col => this.changeAxisAttr(1, "ticks_color", col));
      menu.addSizeMenu("size", 0, 0.05, 0.01, this.ticksSize/this.scaling_size, sz => this.changeAxisAttr(1, "ticks_size", sz));
      menu.addSelectMenu("side", ["normal", "invert", "both"], this.ticksSide, side => this.changeAxisAttr(1, "ticks_side", side));
      menu.add("endsub:");

      if (!this.optionUnlab && this.labelsFont) {
         menu.add("sub:Labels");
         menu.addSizeMenu("offset", -0.05, 0.05, 0.01, this.labelsOffset/this.scaling_size, offset => {
            this.changeAxisAttr(1, "labels_offset", offset);
         });
         menu.addRAttrTextItems(this.labelsFont, { noangle: 1, noalign: 1 }, change => {
            this.changeAxisAttr(1, "labels_" + change.name, change.value);
         });
         menu.addchk(this.labelsFont.angle, "rotate", res => {
            this.changeAxisAttr(1, "labels_angle", res ? 180 : 0);
         });
         menu.add("endsub:");
      }

      menu.add("sub:Title", () => {
         menu.input("Enter axis title", this.fTitle).then(t => this.changeAxisAttr(1, "title", t));
      });

      if (this.fTitle) {
         menu.addSizeMenu("offset", -0.05, 0.05, 0.01, this.titleOffset/this.scaling_size, offset => {
            this.changeAxisAttr(1, "title_offset", offset);
         });

         menu.addSelectMenu("position", ["left", "center", "right"], this.titlePos, pos => {
            this.changeAxisAttr(1, "title_position", pos);
         });

         menu.addchk(this.isTitleRotated(), "rotate", flag => {
            this.changeAxisAttr(1, "title_angle", flag ? 180 : 0);
         });

         menu.addRAttrTextItems(this.titleFont, { noangle: 1, noalign: 1 }, change => {
            this.changeAxisAttr(1, "title_" + change.name, change.value);
         });
      }

      menu.add("endsub:");
      return true;
   }

   let drawRAxis = (divid, obj /*, opt*/) => {
      let painter = new RAxisPainter(divid, obj);
      painter.disable_zooming = true;
      return jsrp.ensureRCanvas(painter, false)
                 .then(() => painter.redraw())
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
    * @extends JSROOT.ObjectPainter
    * @param {object|string} dom - DOM element for drawing or element id
    * @param {object} tframe - RFrame object
    * @private
    */

   function RFramePainter(dom, tframe) {
      JSROOT.ObjectPainter.call(this, dom, tframe);
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

   /** @summary Returns frame painter - object itself */
   RFramePainter.prototype.getFramePainter = function() { return this; }

   /** @summary Returns true if it is ROOT6 frame
    * @private */
   RFramePainter.prototype.is_root6 = function() { return false; }

   /** @summary Set active flag for frame - can block some events
    * @private */
   RFramePainter.prototype.setFrameActive = function(on) {
      this.enabledKeys = on && JSROOT.settings.HandleKeys ? true : false;
      // used only in 3D mode
      if (this.control)
         this.control.enableKeys = this.enabledKeys;
   }

   RFramePainter.prototype.setLastEventPos = function(pnt) {
      // set position of last context menu event, can be
      this.fLastEventPnt = pnt;
   }

   RFramePainter.prototype.getLastEventPos = function() {
      // return position of last event
      return this.fLastEventPnt;
   }

   /** @summary Update graphical attributes */
   RFramePainter.prototype.updateAttributes = function(force) {
      if ((this.fX1NDC === undefined) || (force && !this.modified_NDC)) {

         let rect = this.getPadPainter().getPadRect();
         this.fX1NDC = this.v7EvalLength("margins_left", rect.width, JSROOT.settings.FrameNDC.fX1NDC) / rect.width;
         this.fY1NDC = this.v7EvalLength("margins_bottom", rect.height, JSROOT.settings.FrameNDC.fY1NDC) / rect.height;
         this.fX2NDC = 1 - this.v7EvalLength("margins_right", rect.width, 1-JSROOT.settings.FrameNDC.fX2NDC) / rect.width;
         this.fY2NDC = 1 - this.v7EvalLength("margins_top", rect.height, 1-JSROOT.settings.FrameNDC.fY2NDC) / rect.height;
      }

      if (!this.fillatt)
         this.createv7AttFill("fill_");

      this.createv7AttLine("border_");
   }

   /** @summary Returns coordinates transformation func */
   RFramePainter.prototype.getProjectionFunc = function() {
      switch (this.projection) {
         case 1: return ProjectAitoff2xy;
         case 2: return ProjectMercator2xy;
         case 3: return ProjectSinusoidal2xy;
         case 4: return ProjectParabolic2xy;
      }
   }

   /** @summary Rcalculate frame ranges using specified projection functions
     * @desc Not yet used in v7 */
   RFramePainter.prototype.recalculateRange = function(Proj) {
      this.projection = Proj || 0;

      if ((this.projection == 2) && ((this.scale_ymin <= -90 || this.scale_ymax >=90))) {
         console.warn("Mercator Projection", "Latitude out of range", this.scale_ymin, this.scale_ymax);
         this.projection = 0;
      }

      let func = this.getProjectionFunc();
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

   /** @summary Draw axes grids
     * @desc Called immediately after axes drawing */
   RFramePainter.prototype.drawGrids = function() {
      let layer = this.getFrameSvg().select(".grid_layer");

      layer.selectAll(".xgrid").remove();
      layer.selectAll(".ygrid").remove();

      let h = this.getFrameHeight(),
          w = this.getFrameWidth(),
          gridx = this.v7EvalAttr("gridX", false),
          gridy = this.v7EvalAttr("gridY", false),
          grid_style = JSROOT.gStyle.fGridStyle,
          grid_color = (JSROOT.gStyle.fGridColor > 0) ? this.getColor(JSROOT.gStyle.fGridColor) : "black";

      if ((grid_style < 0) || (grid_style >= jsrp.root_line_styles.length)) grid_style = 11;

      if (this.x_handle)
         this.x_handle.draw_grid = gridx;

      // add a grid on x axis, if the option is set
      if (this.x_handle && this.x_handle.draw_grid) {
         let grid = "";
         for (let n = 0; n < this.x_handle.ticks.length; ++n)
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

      if (this.y_handle)
         this.y_handle.draw_grid = gridy;

      // add a grid on y axis, if the option is set
      if (this.y_handle && this.y_handle.draw_grid) {
         let grid = "";
         for (let n = 0; n < this.y_handle.ticks.length; ++n)
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
   RFramePainter.prototype.axisAsText = function(axis, value) {
      let handle = this[axis+"_handle"];

      if (handle)
         return handle.axisAsText(value, JSROOT.settings[axis.toUpperCase() + "ValuesFormat"]);

      return value.toPrecision(4);
   }

   /** @summary Set axix range */
   RFramePainter.prototype._setAxisRange = function(prefix, vmin, vmax) {
      let nmin = prefix + "min", nmax = prefix + "max";
      if (this[nmin] != this[nmax]) return;
      let min = this.v7EvalAttr(prefix + "_min"),
          max = this.v7EvalAttr(prefix + "_max");

      if (min !== undefined) vmin = min;
      if (max !== undefined) vmax = max;

      if (vmin < vmax) {
         this[nmin] = vmin;
         this[nmax] = vmax;
      }

      let nzmin = "zoom_" + prefix + "min", nzmax = "zoom_" + prefix + "max";

      if ((this[nzmin] == this[nzmax]) && !this.zoomChangedInteractive(prefix)) {
         min = this.v7EvalAttr(prefix + "_zoomMin");
         max = this.v7EvalAttr(prefix + "_zoomMax");

         if ((min !== undefined) || (max !== undefined)) {
            this[nzmin] = (min === undefined) ? this[nmin] : min;
            this[nzmax] = (max === undefined) ? this[nmax] : max;
         }
      }
   }

   /** @summary Set axes ranges for drawing, check configured attributes if range already specified */
   RFramePainter.prototype.setAxesRanges = function(xaxis, xmin, xmax, yaxis, ymin, ymax, zaxis, zmin, zmax) {
      if (this.axes_drawn) return;
      this.xaxis = xaxis;
      this._setAxisRange("x", xmin, xmax);
      this.yaxis = yaxis;
      this._setAxisRange("y", ymin, ymax);
      this.zaxis = zaxis;
      this._setAxisRange("z", zmin, zmax);
   }

   /** @summary Set secondary axes ranges */
   RFramePainter.prototype.setAxes2Ranges = function(second_x, xaxis, xmin, xmax, second_y, yaxis, ymin, ymax) {
      if (second_x) {
         this.x2axis = xaxis;
         this._setAxisRange("x2", xmin, xmax);
      }
      if (second_y) {
         this.y2axis = yaxis;
         this._setAxisRange("y2", ymin, ymax);
      }
   }

   /** @summary Create x,y objects which maps user coordinates into pixels
     * @desc Must be used only for v6 objects, see TFramePainter for more details
     * @private */
   RFramePainter.prototype.createXY = function(opts) {

      if (this.self_drawaxes) return;

      this.cleanXY(); // remove all previous configurations

      if (!opts) opts = {};

      this.v6axes = true;
      this.swap_xy = opts.swap_xy || false;
      this.reverse_x = opts.reverse_x || false;
      this.reverse_y = opts.reverse_y || false;

      this.logx = this.v7EvalAttr("x_log", 0);
      this.logy = this.v7EvalAttr("y_log", 0);

      let w = this.getFrameWidth(), h = this.getFrameHeight();

      this.scale_xmin = this.xmin;
      this.scale_xmax = this.xmax;

      this.scale_ymin = this.ymin;
      this.scale_ymax = this.ymax;

      if (opts.extra_y_space) {
         let log_scale = this.swap_xy ? this.logx : this.logy;
         if (log_scale && (this.scale_ymax > 0))
            this.scale_ymax = Math.exp(Math.log(this.scale_ymax)*1.1);
         else
            this.scale_ymax += (this.scale_ymax - this.scale_ymin)*0.1;
      }

      // if (opts.check_pad_range) {
         // take zooming out of pad or axis attributes - skip!
      // }

      if ((this.zoom_ymin == this.zoom_ymax) && (opts.zoom_ymin != opts.zoom_ymax) && !this.zoomChangedInteractive("y")) {
         this.zoom_ymin = opts.zoom_ymin;
         this.zoom_ymax = opts.zoom_ymax;
      }

      if (this.zoom_xmin != this.zoom_xmax) {
         this.scale_xmin = this.zoom_xmin;
         this.scale_xmax = this.zoom_xmax;
      }

      if (this.zoom_ymin != this.zoom_ymax) {
         this.scale_ymin = this.zoom_ymin;
         this.scale_ymax = this.zoom_ymax;
      }

      let xaxis = this.xaxis, yaxis = this.yaxis;
      if (!xaxis || xaxis._typename != "TAxis") xaxis = JSROOT.create("TAxis");
      if (!yaxis || yaxis._typename != "TAxis") yaxis = JSROOT.create("TAxis");

      this.x_handle = new JSROOT.TAxisPainter(this.getDom(), xaxis, true);
      this.x_handle.setPadName(this.getPadName());
      this.x_handle.optionUnlab = this.v7EvalAttr("x_labels_hide", false);

      this.x_handle.configureAxis("xaxis", this.xmin, this.xmax, this.scale_xmin, this.scale_xmax, this.swap_xy, this.swap_xy ? [0,h] : [0,w],
                                      { reverse: this.reverse_x,
                                        log: this.swap_xy ? this.logy : this.logx,
                                        symlog: this.swap_xy ? opts.symlog_y : opts.symlog_x,
                                        logcheckmin: this.swap_xy,
                                        logminfactor: 0.0001 });

      this.x_handle.assignFrameMembers(this,"x");

      this.y_handle = new JSROOT.TAxisPainter(this.getDom(), yaxis, true);
      this.y_handle.setPadName(this.getPadName());
      this.y_handle.optionUnlab = this.v7EvalAttr("y_labels_hide", false);

      this.y_handle.configureAxis("yaxis", this.ymin, this.ymax, this.scale_ymin, this.scale_ymax, !this.swap_xy, this.swap_xy ? [0,w] : [0,h],
                                      { reverse: this.reverse_y,
                                        log: this.swap_xy ? this.logx : this.logy,
                                        symlog: this.swap_xy ? opts.symlog_x : opts.symlog_y,
                                        logcheckmin: (opts.ndim < 2) || this.swap_xy,
                                        log_min_nz: opts.ymin_nz && (opts.ymin_nz < 0.01*this.ymax) ? 0.3 * opts.ymin_nz : 0,
                                        logminfactor: 3e-4 });

      this.y_handle.assignFrameMembers(this,"y");
   }

   /** @summary Identify if requested axes are drawn
     * @desc Checks if x/y axes are drawn. Also if second side is already there */
   RFramePainter.prototype.hasDrawnAxes = function(second_x, second_y) {
      return !second_x && !second_y ? this.axes_drawn : false;
   }

   /** @summary Draw configured axes on the frame
     * @desc axes can be drawn only for main histogram  */
   RFramePainter.prototype.drawAxes = function() {

      if (this.axes_drawn || (this.xmin == this.xmax) || (this.ymin == this.ymax))
         return Promise.resolve(this.axes_drawn);

      let ticksx = this.v7EvalAttr("ticksX", 1),
          ticksy = this.v7EvalAttr("ticksY", 1),
          sidex = 1, sidey = 1;

      if (this.v7EvalAttr("swapX", false)) sidex = -1;
      if (this.v7EvalAttr("swapY", false)) sidey = -1;

      let w = this.getFrameWidth(), h = this.getFrameHeight();

      if (!this.v6axes) {
         // this is partially same as v6 createXY method

         this.cleanupAxes();

         this.swap_xy = false;

         if (this.zoom_xmin != this.zoom_xmax) {
            this.scale_xmin = this.zoom_xmin;
            this.scale_xmax = this.zoom_xmax;
         } else {
            this.scale_xmin = this.xmin;
            this.scale_xmax = this.xmax;
         }

         if (this.zoom_ymin != this.zoom_ymax) {
            this.scale_ymin = this.zoom_ymin;
            this.scale_ymax = this.zoom_ymax;
         } else {
            this.scale_ymin = this.ymin;
            this.scale_ymax = this.ymax;
         }

         this.recalculateRange(0);

         this.x_handle = new RAxisPainter(this.getDom(), this, this.xaxis, "x_");
         this.x_handle.setPadName(this.getPadName());
         this.x_handle.snapid = this.snapid;
         this.x_handle.draw_swapside = (sidex < 0);
         this.x_handle.draw_ticks = ticksx;

         this.y_handle = new RAxisPainter(this.getDom(), this, this.yaxis, "y_");
         this.y_handle.setPadName(this.getPadName());
         this.y_handle.snapid = this.snapid;
         this.y_handle.draw_swapside = (sidey < 0);
         this.y_handle.draw_ticks = ticksy;

         this.z_handle = new RAxisPainter(this.getDom(), this, this.zaxis, "z_");
         this.z_handle.setPadName(this.getPadName());
         this.z_handle.snapid = this.snapid;

         this.x_handle.configureAxis("xaxis", this.xmin, this.xmax, this.scale_xmin, this.scale_xmax, false, [0,w], w, { reverse: false });
         this.x_handle.assignFrameMembers(this,"x");

         this.y_handle.configureAxis("yaxis", this.ymin, this.ymax, this.scale_ymin, this.scale_ymax, true, [h,0], -h, { reverse: false });
         this.y_handle.assignFrameMembers(this,"y");

         // only get basic properties like log scale
         this.z_handle.configureZAxis("zaxis", this);
      }

      let layer = this.getFrameSvg().select(".axis_layer");

      this.x_handle.has_obstacle = false;

      let draw_horiz = this.swap_xy ? this.y_handle : this.x_handle,
          draw_vertical = this.swap_xy ? this.x_handle : this.y_handle,
          pp = this.getPadPainter(), draw_promise;

      if (pp && pp._fast_drawing) {
         draw_promise = Promise.resolve(true)
      } else if (this.v6axes) {

         // in v7 ticksx/y values shifted by 1 relative to v6
         // In v7 ticksx==0 means no ticks, ticksx==1 equivalent to ==0 in v6

         let can_adjust_frame = false, disable_x_draw = false, disable_y_draw = false;

         draw_horiz.disable_ticks = (ticksx <= 0);
         draw_vertical.disable_ticks = (ticksy <= 0);

         let promise1 = draw_horiz.drawAxis(layer, w, h,
                                            draw_horiz.invert_side ? undefined : `translate(0,${h})`,
                                            (ticksx > 1) ? -h : 0, disable_x_draw,
                                            undefined, false);

         let promise2 = draw_vertical.drawAxis(layer, w, h,
                                               draw_vertical.invert_side ? `translate(${w},0)` : undefined,
                                               (ticksy > 1) ? w : 0, disable_y_draw,
                                               draw_vertical.invert_side ? 0 : this._frame_x, can_adjust_frame);
         draw_promise = Promise.all([promise1, promise2]).then(() => this.drawGrids());

      } else {
         let promise1 = (ticksx > 0) ? draw_horiz.drawAxis(layer, (sidex > 0) ? `translate(0,${h})` : "", sidex) : true;

         let promise2 = (ticksy > 0) ? draw_vertical.drawAxis(layer, (sidey > 0) ? `translate(0,${h})` : `translate(${w},${h})`, sidey) : true;

         draw_promise = Promise.all([promise1, promise2]).then(() => {

            let again = [];
            if (ticksx > 1)
               again.push(draw_horiz.drawAxisOtherPlace(layer, (sidex < 0) ? `translate(0,${h})` : "", -sidex, ticksx == 2));

            if (ticksy > 1)
               again.push(draw_vertical.drawAxisOtherPlace(layer, (sidey < 0) ? `translate(0,${h})` : `translate(${w},${h})`, -sidey, ticksy == 2));

             return Promise.all(again);
         }).then(() => this.drawGrids());
      }

      return draw_promise.then(() => {
         this.axes_drawn = true;
         return true;
      });
   }

   /** @summary Draw secondary configuread axes */
   RFramePainter.prototype.drawAxes2 = function(second_x, second_y) {
      let w = this.getFrameWidth(), h = this.getFrameHeight(),
          layer = this.getFrameSvg().select(".axis_layer"),
          promise1 = true, promise2 = true;

      if (second_x) {
         if (this.zoom_x2min != this.zoom_x2max) {
            this.scale_x2min = this.zoom_x2min;
            this.scale_x2max = this.zoom_x2max;
         } else {
           this.scale_x2min = this.x2min;
           this.scale_x2max = this.x2max;
         }
         this.x2_handle = new RAxisPainter(this.getDom(), this, this.x2axis, "x2_");
         this.x2_handle.setPadName(this.getPadName());
         this.x2_handle.snapid = this.snapid;

         this.x2_handle.configureAxis("x2axis", this.x2min, this.x2max, this.scale_x2min, this.scale_x2max, false, [0,w], w, { reverse: false });
         this.x2_handle.assignFrameMembers(this,"x2");

         promise1 = this.x2_handle.drawAxis(layer, "", -1);
      }

      if (second_y) {
         if (this.zoom_y2min != this.zoom_y2max) {
            this.scale_y2min = this.zoom_y2min;
            this.scale_y2max = this.zoom_y2max;
         } else {
            this.scale_y2min = this.y2min;
            this.scale_y2max = this.y2max;
         }

         this.y2_handle = new RAxisPainter(this.getDom(), this, this.y2axis, "y2_");
         this.y2_handle.setPadName(this.getPadName());
         this.y2_handle.snapid = this.snapid;

         this.y2_handle.configureAxis("y2axis", this.y2min, this.y2max, this.scale_y2min, this.scale_y2max, true, [h,0], -h, { reverse: false });
         this.y2_handle.assignFrameMembers(this,"y2");

         promise2 = this.y2_handle.drawAxis(layer, `translate(${w},${h})`, -1);
      }

      return Promise.all([promise1, promise2]);
   }

   /** @summary Return functions to create x/y points based on coordinates
     * @desc In default case returns frame painter itself
     * @private */
   RFramePainter.prototype.getGrFuncs = function(second_x, second_y) {
      let use_x2 = second_x && this.grx2,
          use_y2 = second_y && this.gry2;
      if (!use_x2 && !use_y2) return this;

      return {
         use_x2: use_x2,
         grx: use_x2 ? this.grx2 : this.grx,
         x_handle: use_x2 ? this.x2_handle : this.x_handle,
         logx: use_x2 ? this.x2_handle.log : this.x_handle.log,
         scale_xmin: use_x2 ? this.scale_x2min : this.scale_xmin,
         scale_xmax: use_x2 ? this.scale_x2max : this.scale_xmax,
         use_y2: use_y2,
         gry: use_y2 ? this.gry2 : this.gry,
         y_handle: use_y2 ? this.y2_handle : this.y_handle,
         logy: use_y2 ? this.y2_handle.log : this.y_handle.log,
         scale_ymin: use_y2 ? this.scale_y2min : this.scale_ymin,
         scale_ymax: use_y2 ? this.scale_y2max : this.scale_ymax,
         swap_xy: this.swap_xy,
         fp: this,
         revertAxis: function(name, v) {
            if ((name == "x") && this.use_x2) name = "x2";
            if ((name == "y") && this.use_y2) name = "y2";
            return this.fp.revertAxis(name, v);
         },
         axisAsText: function(name, v) {
            if ((name == "x") && this.use_x2) name = "x2";
            if ((name == "y") && this.use_y2) name = "y2";
            return this.fp.axisAsText(name, v);
         }
      };
   }

   /** @summary function called at the end of resize of frame
     * @desc Used to update attributes on the server
     * @private */
   RFramePainter.prototype.sizeChanged = function() {

      let changes = {};
      this.v7AttrChange(changes, "margins_left", this.fX1NDC);
      this.v7AttrChange(changes, "margins_bottom", this.fY1NDC);
      this.v7AttrChange(changes, "margins_right", 1 - this.fX2NDC);
      this.v7AttrChange(changes, "margins_top", 1 - this.fY2NDC);
      this.v7SendAttrChanges(changes, false); // do not invoke canvas update on the server

      this.redrawPad();
   }

   /** @summary Remove all x/y functions
     * @private */
   RFramePainter.prototype.cleanXY = function() {
      // remove all axes drawings
      let clean = (name,grname) => {
         if (this[name]) {
            this[name].cleanup();
            delete this[name];
         }
         delete this[grname];
      };

      clean("x_handle", "grx");
      clean("y_handle", "gry");
      clean("z_handle", "grz");
      clean("x2_handle", "grx2");
      clean("y2_handle", "gry2");

      delete this.v6axes; // marker that v6 axes are used
   }

   /** @summary Remove all axes drawings
     * @private */
   RFramePainter.prototype.cleanupAxes = function() {
      this.cleanXY();

      if (this.draw_g) {
         this.draw_g.select(".grid_layer").selectAll("*").remove();
         this.draw_g.select(".axis_layer").selectAll("*").remove();
      }
      this.axes_drawn = false;
   }

   /** @summary Removes all drawn elements of the frame
     * @private */
   RFramePainter.prototype.cleanFrameDrawings = function() {
      // cleanup all 3D drawings if any
      if (typeof this.create3DScene === 'function')
         this.create3DScene(-1);

      this.cleanupAxes();

      let clean = (name) => {
         this[name+"min"] = this[name+"max"] = 0;
         this["zoom_"+name+"min"] = this["zoom_"+name+"max"] = 0;
         this["scale_"+name+"min"] = this["scale_"+name+"max"] = 0;
      };

      clean("x");
      clean("y");
      clean("z");
      clean("x2");
      clean("y2");

      if (this.draw_g) {
         this.draw_g.select(".main_layer").selectAll("*").remove();
         this.draw_g.select(".upper_layer").selectAll("*").remove();
      }
   }

   /** @summary Fully cleanup frame
     * @private */
   RFramePainter.prototype.cleanup = function() {

      this.cleanFrameDrawings();

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
      delete this.enabledKeys;
      delete this.self_drawaxes;

      delete this.xaxis;
      delete this.yaxis;
      delete this.zaxis;
      delete this.x2axis;
      delete this.y2axis;

      delete this.draw_g; // frame <g> element managet by the pad

      delete this._click_handler;
      delete this._dblclick_handler;

      let pp = this.getPadPainter();
      if (pp && (pp.frame_painter_ref === this))
         delete pp.frame_painter_ref;

      JSROOT.ObjectPainter.prototype.cleanup.call(this);
   }

   /** @summary Redraw frame
     * @private */
   RFramePainter.prototype.redraw = function() {

      let pp = this.getPadPainter();
      if (pp) pp.frame_painter_ref = this;

      // first update all attributes from objects
      this.updateAttributes();

      let rect = pp ? pp.getPadRect() : { width: 10, height: 10 },
          lm = Math.round(rect.width * this.fX1NDC),
          w = Math.round(rect.width * (this.fX2NDC - this.fX1NDC)),
          tm = Math.round(rect.height * (1 - this.fY2NDC)),
          h = Math.round(rect.height * (this.fY2NDC - this.fY1NDC)),
          rotate = false, fixpos = false, trans = `translate(${lm},${tm})`;

      if (pp && pp.options) {
         if (pp.options.RotateFrame) rotate = true;
         if (pp.options.FixFrame) fixpos = true;
      }

      if (rotate) {
         trans += ` rotate(-90) translate(${-h},0)`;
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
      this.draw_g = this.getLayerSvg("primitives_layer").select(".root_frame");

      let top_rect, main_svg;

      if (this.draw_g.empty()) {

         let layer = this.getLayerSvg("primitives_layer");

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
      }

      this.axes_drawn = false;

      this.draw_g.attr("transform", trans);

      top_rect.attr("x", 0)
              .attr("y", 0)
              .attr("width", w)
              .attr("height", h)
              .attr("rx", this.lineatt.rx || null)
              .attr("ry", this.lineatt.ry || null)
              .call(this.fillatt.func)
              .call(this.lineatt.func);

      main_svg.attr("width", w)
              .attr("height", h)
              .attr("viewBox", "0 0 " + w + " " + h);

      let promise = Promise.resolve(true);

      if (this.v7EvalAttr("drawAxes")) {
         this.self_drawaxes = true;
         this.setAxesRanges();
         promise = this.drawAxes().then(() => this.addInteractivity());
      }

      if (JSROOT.batch_mode) return promise;

      return promise.then(() => JSROOT.require(['interactive'])).then(inter => {
         top_rect.attr("pointer-events", "visibleFill");  // let process mouse events inside frame
         inter.FrameInteractive.assign(this);
         this.addBasicInteractivity();
      });
   }

   /** @summary Returns frame width */
   RFramePainter.prototype.getFrameWidth = function() { return this._frame_width || 0; }

   /** @summary Returns frame height */
   RFramePainter.prototype.getFrameHeight = function() { return this._frame_height || 0; }

   /** @summary Returns frame rectangle plus extra info for hint display */
   RFramePainter.prototype.getFrameRect = function() {
      return {
         x: this._frame_x || 0,
         y: this._frame_y || 0,
         width: this.getFrameWidth(),
         height: this.getFrameHeight(),
         transform: this.draw_g ? this.draw_g.attr("transform") : "",
         hint_delta_x: 0,
         hint_delta_y: 0
      }
   }

   /** @summary Returns palette associated with frame */
   RFramePainter.prototype.getHistPalette = function() {
      return this.getPadPainter().getHistPalette();
   }

   RFramePainter.prototype.configureUserClickHandler = function(handler) {
      this._click_handler = handler && (typeof handler == 'function') ? handler : null;
   }

   RFramePainter.prototype.configureUserDblclickHandler = function(handler) {
      this._dblclick_handler = handler && (typeof handler == 'function') ? handler : null;
   }

   /** @summary function can be used for zooming into specified range
     * @desc if both limits for each axis 0 (like xmin==xmax==0), axis will be unzoomed
     * @returns {Promise} with boolean flag if zoom operation was performed */
   RFramePainter.prototype.zoom = function(xmin, xmax, ymin, ymax, zmin, zmax) {

      // disable zooming when axis conversion is enabled
      if (this.projection) return Promise.resolve(false);

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
         // if (this.logz && this.ymin_nz && this.getDimension()===2) main_zmin = 0.3*this.ymin_nz;
         if (zmin <= this.zmin) { zmin = this.zmin; cnt++; }
         if (zmax >= this.zmax) { zmax = this.zmax; cnt++; }
         if (cnt === 2) { zoom_z = false; unzoom_z = true; }
      } else {
         unzoom_z = (zmin === zmax) && (zmin === 0);
      }

      let changed = false,
          r_x = "", r_y = "", r_z = "", is_any_check = false,
         req = {
            _typename: "ROOT::Experimental::RFrame::RUserRanges",
            values: [0, 0, 0, 0, 0, 0],
            flags: [false, false, false, false, false, false]
         };

      let checkZooming = (painter, force) => {
         if (!force && (typeof painter.canZoomInside != 'function')) return;

         is_any_check = true;

         if (zoom_x && (force || painter.canZoomInside("x", xmin, xmax))) {
            this.zoom_xmin = xmin;
            this.zoom_xmax = xmax;
            changed = true; r_x = "0";
            zoom_x = false;
            req.values[0] = xmin; req.values[1] = xmax;
            req.flags[0] = req.flags[1] = true;
         }
         if (zoom_y && (force || painter.canZoomInside("y", ymin, ymax))) {
            this.zoom_ymin = ymin;
            this.zoom_ymax = ymax;
            changed = true; r_y = "1";
            zoom_y = false;
            req.values[2] = ymin; req.values[3] = ymax;
            req.flags[2] = req.flags[3] = true;
         }
         if (zoom_z && (force || painter.canZoomInside("z", zmin, zmax))) {
            this.zoom_zmin = zmin;
            this.zoom_zmax = zmax;
            changed = true; r_z = "2";
            zoom_z = false;
            req.values[4] = zmin; req.values[5] = zmax;
            req.flags[4] = req.flags[5] = true;
         }
      };

      // first process zooming (if any)
      if (zoom_x || zoom_y || zoom_z)
         this.forEachPainter(painter => checkZooming(painter));

      // force zooming when no any other painter can verify zoom range
      if (!is_any_check && this.self_drawaxes)
         checkZooming(null, true);

      // and process unzoom, if any
      if (unzoom_x || unzoom_y || unzoom_z) {
         if (unzoom_x) {
            if (this.zoom_xmin !== this.zoom_xmax) { changed = true; r_x = "0"; }
            this.zoom_xmin = this.zoom_xmax = 0;
            req.values[0] = req.values[1] = -1;
         }
         if (unzoom_y) {
            if (this.zoom_ymin !== this.zoom_ymax) { changed = true; r_y = "1"; }
            this.zoom_ymin = this.zoom_ymax = 0;
            req.values[2] = req.values[3] = -1;
         }
         if (unzoom_z) {
            if (this.zoom_zmin !== this.zoom_zmax) { changed = true; r_z = "2"; }
            this.zoom_zmin = this.zoom_zmax = 0;
            req.values[4] = req.values[5] = -1;
         }
      }

      if (!changed) return Promise.resolve(false);

      if (this.v7CommMode() == JSROOT.v7.CommMode.kNormal)
         this.v7SubmitRequest("zoom", { _typename: "ROOT::Experimental::RFrame::RZoomRequest", ranges: req });

      return this.interactiveRedraw("pad", "zoom" + r_x + r_y + r_z).then(() => true);
   }

   /** @summary Provide zooming of single axis
     * @desc One can specify names like x/y/z but also second axis x2 or y2 */
   RFramePainter.prototype.zoomSingle = function(name, vmin, vmax) {

      let names = ["x","y","z","x2","y2"], indx = names.indexOf(name);

      // disable zooming when axis conversion is enabled
      if (this.projection || !this[name+"_handle"] || (indx < 0))
         return Promise.resolve(false);

      let zoom_v = (vmin !== vmax), unzoom_v = false;

      if (zoom_v) {
         let cnt = 0;
         if (vmin <= this[name+"min"]) { vmin = this[name+"min"]; cnt++; }
         if (vmax >= this[name+"max"]) { vmax = this[name+"max"]; cnt++; }
         if (cnt === 2) { zoom_v = false; unzoom_v = true; }
      } else {
         unzoom_v = (vmin === vmax) && (vmin === 0);
      }

      let changed = false, is_any_check = false,
          req = {
             _typename: "ROOT::Experimental::RFrame::RUserRanges",
             values: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             flags: [false, false, false, false, false, false, false, false, false, false]
          };

      let checkZooming = (painter, force) => {
         if (!force && (typeof painter.canZoomInside != 'function')) return;

         is_any_check = true;

         if (zoom_v && (force || painter.canZoomInside(name[0], vmin, vmax))) {
            this["zoom_" + name + "min"] = vmin;
            this["zoom_" + name + "max"] = vmax;
            changed = true;
            zoom_v = false;
            req.values[indx*2] = vmin; req.values[indx*2+1] = vmax;
            req.flags[indx*2] = req.flags[indx*2+1] = true;
         }
      }

      // first process zooming (if any)
      if (zoom_v)
         this.forEachPainter(painter => checkZooming(painter));

      // force zooming when no any other painter can verify zoom range
      if (!is_any_check && this.self_drawaxes)
         checkZooming(null, true);

      if (unzoom_v) {
         if (this["zoom_" + name + "min"] !== this["zoom_" + name + "max"]) changed = true;
         this["zoom_" + name + "min"] = this["zoom_" + name + "max"] = 0;
         req.values[indx*2] = req.values[indx*2+1] = -1;
      }

      if (!changed) return Promise.resolve(false);

      if (this.v7CommMode() == JSROOT.v7.CommMode.kNormal)
         this.v7SubmitRequest("zoom", { _typename: "ROOT::Experimental::RFrame::RZoomRequest", ranges: req });

      return this.interactiveRedraw("pad", "zoom" + indx).then(() => true);
   }

   /** @summary Checks if specified axis zoomed */
   RFramePainter.prototype.isAxisZoomed = function(axis) {
      return this['zoom_'+axis+'min'] !== this['zoom_'+axis+'max'];
   }

   /** @summary Unzoom specified axes
     * @returns {Promise} with boolean flag if zoom is changed */
   RFramePainter.prototype.unzoom = function(dox, doy, doz) {
      if (dox == "all")
         return this.unzoom("x2").then(() => this.unzoom("y2")).then(() => this.unzoom("xyz"));

      if ((dox == "x2") || (dox == "y2"))
         return this.zoomSingle(dox, 0, 0).then(changed => {
            if (changed) this.zoomChangedInteractive(dox, "unzoom");
            return changed;
         });

      if (typeof dox === 'undefined') { dox = doy = doz = true; } else
      if (typeof dox === 'string') { doz = dox.indexOf("z") >= 0; doy = dox.indexOf("y") >= 0; dox = dox.indexOf("x") >= 0; }

      return this.zoom(dox ? 0 : undefined, dox ? 0 : undefined,
                       doy ? 0 : undefined, doy ? 0 : undefined,
                       doz ? 0 : undefined, doz ? 0 : undefined).then(changed => {

         if (changed && dox) this.zoomChangedInteractive("x", "unzoom");
         if (changed && doy) this.zoomChangedInteractive("y", "unzoom");
         if (changed && doz) this.zoomChangedInteractive("z", "unzoom");

         return changed;
      });
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
   RFramePainter.prototype.fillObjectOfflineMenu = function(menu, kind) {
      if ((kind!="x") && (kind!="y")) return;

      menu.add("Unzoom", () => this.unzoom(kind));

      //if (this[kind+"_kind"] == "normal")
      //   menu.addchk(this["log"+kind], "SetLog"+kind, this.toggleAxisLog.bind(this, kind));

      // here should be all axes attributes in offline
   }

   /** @summary Set grid drawing for specified axis */
   RFramePainter.prototype.changeFrameAttr = function(attr, value) {
      let changes = {};
      this.v7AttrChange(changes, attr, value);
      this.v7SetAttr(attr, value);
      this.v7SendAttrChanges(changes, false); // do not invoke canvas update on the server
      this.redrawPad();
   }

   /** @summary Fill context menu */
   RFramePainter.prototype.fillContextMenu = function(menu, kind, /* obj */) {

      // when fill and show context menu, remove all zooming

      if ((kind=="x") || (kind=="y") || (kind=="x2") || (kind=="y2")) {
         let handle = this[kind+"_handle"];
         if (!handle) return false;
         menu.add("header: " + kind.toUpperCase() + " axis");
         return handle.fillAxisContextMenu(menu, kind);
      }

      let alone = menu.size()==0;

      if (alone)
         menu.add("header:Frame");
      else
         menu.add("separator");

      if (this.zoom_xmin !== this.zoom_xmax)
         menu.add("Unzoom X", () => this.unzoom("x"));
      if (this.zoom_ymin !== this.zoom_ymax)
         menu.add("Unzoom Y", () => this.unzoom("y"));
      if (this.zoom_zmin !== this.zoom_zmax)
         menu.add("Unzoom Z", () => this.unzoom("z"));
      if (this.zoom_x2min !== this.zoom_x2max)
         menu.add("Unzoom X2", () => this.unzoom("x2"));
      if (this.zoom_y2min !== this.zoom_y2max)
         menu.add("Unzoom Y2", () => this.unzoom("y2"));
      menu.add("Unzoom all", () => this.unzoom("all"));

      menu.add("separator");

      menu.addchk(this.isTooltipAllowed(), "Show tooltips", () => this.setTooltipAllowed("toggle"));

      if (this.x_handle)
         menu.addchk(this.x_handle.draw_grid, "Grid x", flag => this.changeFrameAttr("gridX", flag));
      if (this.y_handle)
         menu.addchk(this.y_handle.draw_grid, "Grid y", flag => this.changeFrameAttr("gridY", flag));
      if (this.x_handle && !this.x2_handle)
         menu.addchk(this.x_handle.draw_swapside, "Swap x", flag => this.changeFrameAttr("swapX", flag));
      if (this.y_handle && !this.y2_handle)
         menu.addchk(this.y_handle.draw_swapside, "Swap y", flag => this.changeFrameAttr("swapY", flag));
      if (this.x_handle && !this.x2_handle) {
         menu.add("sub:Ticks x");
         menu.addchk(this.x_handle.draw_ticks == 0, "off", () => this.changeFrameAttr("ticksX", 0));
         menu.addchk(this.x_handle.draw_ticks == 1, "normal", () => this.changeFrameAttr("ticksX", 1));
         menu.addchk(this.x_handle.draw_ticks == 2, "ticks on both sides", () => this.changeFrameAttr("ticksX", 2));
         menu.addchk(this.x_handle.draw_ticks == 3, "labels on both sides", () => this.changeFrameAttr("ticksX", 3));
         menu.add("endsub:");
       }
      if (this.y_handle && !this.y2_handle) {
         menu.add("sub:Ticks y");
         menu.addchk(this.y_handle.draw_ticks == 0, "off", () => this.changeFrameAttr("ticksY", 0));
         menu.addchk(this.y_handle.draw_ticks == 1, "normal", () => this.changeFrameAttr("ticksY", 1));
         menu.addchk(this.y_handle.draw_ticks == 2, "ticks on both sides", () => this.changeFrameAttr("ticksY", 2));
         menu.addchk(this.y_handle.draw_ticks == 3, "labels on both sides", () => this.changeFrameAttr("ticksY", 3));
         menu.add("endsub:");
       }

      menu.addAttributesMenu(this, alone ? "" : "Frame ");
      menu.add("separator");
      menu.add("Save as frame.png", () => this.getPadPainter().saveAs("png", 'frame', 'frame.png'));
      menu.add("Save as frame.svg", () => this.getPadPainter().saveAs("svg", 'frame', 'frame.svg'));

      return true;
   }

   /** @summary Convert graphical coordinate into axis value */
   RFramePainter.prototype.revertAxis = function(axis, pnt) {
      let handle = this[axis+"_handle"];
      return handle ? handle.revertPoint(pnt) : 0;
   }

   /** @summary Show axis status message
     * @desc method called normally when mouse enter main object element
     * @private */
   RFramePainter.prototype.showAxisStatus = function(axis_name, evnt) {

      let taxis = null, hint_name = axis_name, hint_title = "axis",
          m = d3.pointer(evnt, this.getFrameSvg().node()), id = (axis_name=="x") ? 0 : 1;

      if (taxis) { hint_name = taxis.fName; hint_title = taxis.fTitle || "axis object"; }

      if (this.swap_xy) id = 1-id;

      let axis_value = this.revertAxis(axis_name, m[id]);

      this.showObjectStatus(hint_name, hint_title, axis_name + " : " + this.axisAsText(axis_name, axis_value), Math.round(m[0])+","+Math.round(m[1]));
   }

   /** @summary Add interactive keys handlers
    * @private */
   RFramePainter.prototype.addKeysHandler = function() {
      if (JSROOT.batch_mode) return;
      JSROOT.require(['interactive']).then(inter => {
         inter.FrameInteractive.assign(this);
         this.addKeysHandler();
      });
   }

   /** @summary Add interactive functionality to the frame
    * @private */
   RFramePainter.prototype.addInteractivity = function(for_second_axes) {

      if (JSROOT.batch_mode || (!JSROOT.settings.Zooming && !JSROOT.settings.ContextMenu))
         return Promise.resolve(true);

      return JSROOT.require(['interactive']).then(inter => {
         inter.FrameInteractive.assign(this);
         return this.addInteractivity(for_second_axes);
      });
   }

   /** @summary Set selected range back to pad object - to be implemented
     * @private */
   RFramePainter.prototype.setRootPadRange = function(/* pad, is3d */) {
      // TODO: change of pad range and send back to root application
   }

   /** @summary Toggle log scale on the specified axes */
   RFramePainter.prototype.toggleAxisLog = function(axis) {
      let handle = this[axis+"_handle"];
      if (handle) handle.changeAxisLog('toggle');
   }

   function drawRFrame(divid, obj, opt) {
      let p = new RFramePainter(divid, obj);
      if (opt == "3d") p.mode3d = true;
      return jsrp.ensureRCanvas(p, false).then(() => {
         p.redraw();
         return p;
      });
   }

   // ===========================================================================

   /**
    * @summary Painter class for RPad
    *
    * @class
    * @memberof JSROOT
    * @extends JSROOT.ObjectPainter
    * @param {object|string} dom - DOM element for drawing or element id
    * @param {object} pad - RPad object
    * @param {boolean} [iscan] - true when used for RCanvas
    * @private
    */

   function RPadPainter(divid, pad, iscan) {
      JSROOT.ObjectPainter.call(this, divid, pad);
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

   /** @summary Indicates that is not Root6 pad painter
    * @private */
   RPadPainter.prototype.isRoot6 = function() { return false; }

  /** @summary Returns SVG element for the pad itself
    * @private */
   RPadPainter.prototype.svg_this_pad = function() {
      return this.getPadSvg(this.this_pad_name);
   }

   /** @summary Returns main painter on the pad
     * @desc Typically main painter is TH1/TH2 object which is drawing axes
    * @private */
   RPadPainter.prototype.getMainPainter = function() {
      return this.main_painter_ref || null;
   }

   /** @summary Assign main painter on the pad
    * @private */
   RPadPainter.prototype.setMainPainter = function(painter, force) {
      if (!this.main_painter_ref || force)
         this.main_painter_ref = painter;
   }

   /** @summary cleanup pad and all primitives inside */
   RPadPainter.prototype.cleanup = function() {
      if (this._doing_draw)
         console.error('pad drawing is not completed when cleanup is called');

      this.painters.forEach(p => p.cleanup());

      let svg_p = this.svg_this_pad();
      if (!svg_p.empty()) {
         svg_p.property('pad_painter', null);
         if (!this.iscan) svg_p.remove();
      }

      delete this.main_painter_ref;
      delete this.frame_painter_ref;
      delete this.pads_cache;
      delete this._pad_x;
      delete this._pad_y;
      delete this._pad_width;
      delete this._pad_height;
      delete this._doing_draw;

      this.painters = [];
      this.pad = null;
      this.draw_object = null;
      this.pad_frame = null;
      this.this_pad_name = undefined;
      this.has_canvas = false;

      jsrp.selectActivePad({ pp: this, active: false });

      JSROOT.ObjectPainter.prototype.cleanup.call(this);
   }

   /** @summary Returns frame painter inside the pad
    * @private */
   RPadPainter.prototype.getFramePainter = function() { return this.frame_painter_ref; }

   /** @summary get pad width */
   RPadPainter.prototype.getPadWidth = function() { return this._pad_width || 0; }

   /** @summary get pad height */
   RPadPainter.prototype.getPadHeight = function() { return this._pad_height || 0; }

   /** @summary get pad rect */
   RPadPainter.prototype.getPadRect = function() {
      return {
         x: this._pad_x || 0,
         y: this._pad_y || 0,
         width: this.getPadWidth(),
         height: this.getPadHeight()
      }
   }

   /** @summary return RPad object */
   RPadPainter.prototype.getRootPad = function(is_root6) {
      return (is_root6 === undefined) || !is_root6 ? this.pad : null;
   }

   /** @summary Cleanup primitives from pad - selector lets define which painters to remove
    * @private */
   RPadPainter.prototype.cleanPrimitives = function(selector) {
      if (!selector || (typeof selector !== 'function')) return;

      for (let k = this.painters.length-1; k >= 0; --k)
         if (selector(this.painters[k])) {
            this.painters[k].cleanup();
            this.painters.splice(k, 1);
         }
   }

   /** @summary Try to find painter for specified object
     * @desc can be used to find painter for some special objects, registered as
     * histogram functions
     * @private */
   RPadPainter.prototype.findPainterFor = function(selobj, selname, seltype) {
      return this.painters.find(p => {
         let pobj = p.getObject();
         if (!pobj) return;

         if (selobj && (pobj === selobj)) return true;
         if (!selname && !seltype) return;
         if (selname && (pobj.fName !== selname)) return;
         if (seltype && (pobj._typename !== seltype)) return;
         return true;
      });
   }

   /** @summary Returns palette associated with pad.
     * @desc Either from existing palette painter or just default palette */
   RPadPainter.prototype.getHistPalette = function() {
      let pp = this.findPainterFor(undefined, undefined, "ROOT::Experimental::RPaletteDrawable");

      if (pp) return pp.getHistPalette();

      if (!this.fDfltPalette) {
         this.fDfltPalette = {
            _typename: "ROOT::Experimental::RPalette",
            fColors: [{ fOrdinal : 0,     fColor : { fColor : "rgb(53, 42, 135)" } },
                      { fOrdinal : 0.125, fColor : { fColor : "rgb(15, 92, 221)" } },
                      { fOrdinal : 0.25,  fColor : { fColor : "rgb(20, 129, 214)" } },
                      { fOrdinal : 0.375, fColor : { fColor : "rgb(6, 164, 202)" } },
                      { fOrdinal : 0.5,   fColor : { fColor : "rgb(46, 183, 164)" } },
                      { fOrdinal : 0.625, fColor : { fColor : "rgb(135, 191, 119)" } },
                      { fOrdinal : 0.75,  fColor : { fColor : "rgb(209, 187, 89)" } },
                      { fOrdinal : 0.875, fColor : { fColor : "rgb(254, 200, 50)" } },
                      { fOrdinal : 1,     fColor : { fColor : "rgb(249, 251, 14)" } }],
             fInterpolate: true,
             fNormalized: true
         };
         JSROOT.addMethods(this.fDfltPalette, "ROOT::Experimental::RPalette");
      }

      return this.fDfltPalette;
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

   /** @summary For pad painter equivalent to forEachPainterInPad */
   RPadPainter.prototype.forEachPainter = RPadPainter.prototype.forEachPainterInPad;

   /** @summary register for pad events receiver
     * @desc in pad painter, while pad may be drawn without canvas
     * @private */
   RPadPainter.prototype.registerForPadEvents = function(receiver) {
      this.pad_events_receiver = receiver;
   }

   /** @summary Generate pad events, normally handled by GED
     * @desc in pad painter, while pad may be drawn without canvas
     * @private */
   RPadPainter.prototype.producePadEvent = function(_what, _padpainter, _painter, _position, _place) {
      if ((_what == "select") && (typeof this.selectActivePad == 'function'))
         this.selectActivePad(_padpainter, _painter, _position);

      if (this.pad_events_receiver)
         this.pad_events_receiver({ what: _what, padpainter:  _padpainter, painter: _painter, position: _position, place: _place });
   }

   /** @summary method redirect call to pad events receiver */
   RPadPainter.prototype.selectObjectPainter = function(_painter, pos, _place) {

      let istoppad = (this.iscan || !this.has_canvas),
          canp = istoppad ? this : this.getCanvPainter();

      if (_painter === undefined) _painter = this;

      if (pos && !istoppad)
         pos = jsrp.getAbsPosInCanvas(this.svg_this_pad(), pos);

      jsrp.selectActivePad({ pp: this, active: true });

      canp.producePadEvent("select", this, _painter, pos, _place);
   }

   /** @summary Create SVG element for the canvas */
   RPadPainter.prototype.createCanvasSvg = function(check_resize, new_size) {

      let factor = null, svg = null, lmt = 5, rect = null, btns;

      if (check_resize > 0) {

         if (this._fixed_size) return (check_resize > 1); // flag used to force re-drawing of all subpads

         svg = this.getCanvSvg();

         if (svg.empty()) return false;

         factor = svg.property('height_factor');

         rect = this.testMainResize(check_resize, null, factor);

         if (!rect.changed) return false;

         if (!JSROOT.batch_mode)
            btns = this.getLayerSvg("btns_layer", this.this_pad_name);

      } else {

         let render_to = this.selectDom();

         if (render_to.style('position')=='static')
            render_to.style('position','relative');

         svg = render_to.append("svg")
             .attr("class", "jsroot root_canvas")
             .property('pad_painter', this) // this is custom property
             .property('current_pad', "") // this is custom property
             .property('redraw_by_resize', false); // could be enabled to force redraw by each resize

         this.setTopPainter(); //assign canvas as top painter of that element

         svg.append("svg:title").text("ROOT canvas");
         let frect = svg.append("svg:rect").attr("class","canvas_fillrect")
                               .attr("x",0).attr("y",0);
         if (!JSROOT.batch_mode)
            frect.style("pointer-events", "visibleFill")
                 .on("dblclick", evnt => this.enlargePad(evnt))
                 .on("click", () => this.selectObjectPainter(this, null))
                 .on("mouseenter", () => this.showObjectStatus());

         svg.append("svg:g").attr("class","primitives_layer");
         svg.append("svg:g").attr("class","info_layer");
         if (!JSROOT.batch_mode)
            btns = svg.append("svg:g")
                      .attr("class","btns_layer")
                      .property('leftside', JSROOT.settings.ToolBarSide == 'left')
                      .property('vertical', JSROOT.settings.ToolBarVert);

         if (JSROOT.settings.ContextMenu && !JSROOT.batch_mode)
            svg.select(".canvas_fillrect").on("contextmenu", evnt => this.padContextMenu(evnt));

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

      this._pad_x = 0;
      this._pad_y = 0;
      this._pad_width = rect.width;
      this._pad_height = rect.height;

      svg.select(".canvas_fillrect")
         .attr("width", rect.width)
         .attr("height", rect.height)
         .call(this.fillatt.func);

      this._fast_drawing = JSROOT.settings.SmallPad && ((rect.width < JSROOT.settings.SmallPad.width) || (rect.height < JSROOT.settings.SmallPad.height));

      if (this.alignButtons && btns)
         this.alignButtons(btns, rect.width, rect.height);

      return true;
   }

   /** @summary Enlarge pad draw element when possible */
   RPadPainter.prototype.enlargePad = function(evnt) {

      if (evnt) {
         evnt.preventDefault();
         evnt.stopPropagation();
      }

      let svg_can = this.getCanvSvg(),
          pad_enlarged = svg_can.property("pad_enlarged");

      if (this.iscan || !this.has_canvas || (!pad_enlarged && !this.hasObjectsToDraw() && !this.painters)) {
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
         this.showPadButtons();
   }

   /** @summary Create SVG element for the pad
     * @returns true when pad is displayed and all its items should be redrawn */
   RPadPainter.prototype.createPadSvg = function(only_resize) {

      if (!this.has_canvas) {
         this.createCanvasSvg(only_resize ? 2 : 0);
         return true;
      }

      let svg_parent = this.getPadSvg(this.pad_name), // this.pad_name MUST be here to select parent pad
          svg_can = this.getCanvSvg(),
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
            this.forEachPainterInPad(pp => { if (pp.getObject() == pad_enlarged) pad_visible = true; }, "pads");

         if (pad_visible) { w = width; h = height; x = y = 0; }
      }

      if (only_resize) {
         svg_pad = this.svg_this_pad();
         svg_rect = svg_pad.select(".root_pad_border");
         if (!JSROOT.batch_mode)
            btns = this.getLayerSvg("btns_layer", this.this_pad_name);
      } else {
         svg_pad = svg_parent.select(".primitives_layer")
             .append("svg:svg") // here was g before, svg used to blend all drawin outside
             .classed("__root_pad_" + this.this_pad_name, true)
             .attr("pad", this.this_pad_name) // set extra attribute  to mark pad name
             .property('pad_painter', this); // this is custom property
         svg_rect = svg_pad.append("svg:rect").attr("class", "root_pad_border");

         svg_pad.append("svg:g").attr("class","primitives_layer");
         if (!JSROOT.batch_mode)
            btns = svg_pad.append("svg:g")
                          .attr("class","btns_layer")
                          .property('leftside', JSROOT.settings.ToolBarSide != 'left')
                          .property('vertical', JSROOT.settings.ToolBarVert);

         if (JSROOT.settings.ContextMenu)
            svg_rect.on("contextmenu", evnt => this.padContextMenu(evnt));

         if (!JSROOT.batch_mode)
            svg_rect.attr("pointer-events", "visibleFill") // get events also for not visible rect
                    .on("dblclick", evnt => this.enlargePad(evnt))
                    .on("click", () => this.selectObjectPainter(this, null))
                    .on("mouseenter", () => this.showObjectStatus());
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

      this._pad_x = x;
      this._pad_y = y;
      this._pad_width = w;
      this._pad_height = h;

      svg_rect.attr("x", 0)
              .attr("y", 0)
              .attr("width", w)
              .attr("height", h)
              .call(this.fillatt.func)
              .call(this.lineatt.func);

      this._fast_drawing = JSROOT.settings.SmallPad && ((w < JSROOT.settings.SmallPad.width) || (h < JSROOT.settings.SmallPad.height));

       // special case of 3D canvas overlay
      if (svg_pad.property('can3d') === JSROOT.constants.Embed3D.Overlay)
          this.selectDom().select(".draw3d_" + this.this_pad_name)
              .style('display', pad_visible ? '' : 'none');

      if (this.alignButtons && btns) this.alignButtons(btns, w, h);

      return pad_visible;
   }

   /** @summary returns true if any objects beside sub-pads exists in the pad */
   RPadPainter.prototype.hasObjectsToDraw = function() {
      let arr = this.pad ? this.pad.fPrimitives : null;
      return arr && arr.find(obj => obj._typename != "ROOT::Experimental::RPadDisplayItem") ? true : false;
   }

   /** @summary sync drawing/redrawing/resize of the pad
     * @param {string} kind - kind of draw operation, if true - always queued
     * @returns {Promise} when pad is ready for draw operation or false if operation already queued
     * @private */
   RPadPainter.prototype.syncDraw = function(kind) {
      let entry = { kind : kind || "redraw" };
      if (this._doing_draw === undefined) {
         this._doing_draw = [ entry ];
         return Promise.resolve(true);
      }
      // if queued operation registered, ignore next calls, indx == 0 is running operation
      if ((entry.kind !== true) && (this._doing_draw.findIndex((e,i) => (i > 0) && (e.kind == entry.kind)) > 0))
         return false;
      this._doing_draw.push(entry);
      return new Promise(resolveFunc => {
         entry.func = resolveFunc;
      });
   }

   /** @summary confirms that drawing is completed, may trigger next drawing immediately
     * @private */
   RPadPainter.prototype.confirmDraw = function() {
      if (this._doing_draw === undefined)
         return console.warn("failure, should not happen");
      this._doing_draw.shift();
      if (this._doing_draw.length == 0) {
         delete this._doing_draw;
      } else {
         let entry = this._doing_draw[0];
         if(entry.func) { entry.func(); delete entry.func; }
      }
   }

   /** @summary Draw pad primitives
     * @private */
   RPadPainter.prototype.drawPrimitives = function(indx) {

      if (indx === undefined) {
         if (this.iscan)
            this._start_tm = new Date().getTime();

         // set number of primitves
         this._num_primitives = this.pad && this.pad.fPrimitives ? this.pad.fPrimitives.length : 0;

         return this.syncDraw(true).then(() => this.drawPrimitives(0));
      }

      if (!this.pad || (indx >= this._num_primitives)) {

         this.confirmDraw();

         if (this._start_tm) {
            let spenttm = new Date().getTime() - this._start_tm;
            if (spenttm > 3000) console.log("Canvas drawing took " + (spenttm*1e-3).toFixed(2) + "s");
            delete this._start_tm;
         }

         return Promise.resolve();
      }

      // handle used to invoke callback only when necessary
      return JSROOT.draw(this.getDom(), this.pad.fPrimitives[indx], "").then(ppainter => {
         // mark painter as belonging to primitives
         if (ppainter && (typeof ppainter == 'object'))
            ppainter._primitive = true;

         return this.drawPrimitives(indx+1);
      });
   }

   /** @summary Process tooltip event in the pad
     * @private */
   RPadPainter.prototype.processPadTooltipEvent = function(pnt) {
      let painters = [], hints = [];

      // first count - how many processors are there
      if (this.painters !== null)
         this.painters.forEach(obj => {
            if (typeof obj.processTooltipEvent == 'function') painters.push(obj);
         });

      if (pnt) pnt.nproc = painters.length;

      painters.forEach(obj => {
         let hint = obj.processTooltipEvent(pnt);
         if (!hint) hint = { user_info: null };
         hints.push(hint);
         if (pnt && pnt.painters) hint.painter = obj;
      });

      return hints;
   }

   /** @summary Fill pad context menu
     * @private */
   RPadPainter.prototype.fillContextMenu = function(menu) {

      if (this.iscan)
         menu.add("header: RCanvas");
      else
         menu.add("header: RPad");

      menu.addchk(this.isTooltipAllowed(), "Show tooltips", () => this.setTooltipAllowed("toggle"));

      if (!this._websocket)
         menu.addAttributesMenu(this);

      menu.add("separator");

      if (typeof this.hasMenuBar == 'function' && typeof this.actiavteMenuBar == 'function')
         menu.addchk(this.hasMenuBar(), "Menu bar", flag => this.actiavteMenuBar(flag));

      if (typeof this.hasEventStatus == 'function' && typeof this.activateStatusBar == 'function')
         menu.addchk(this.hasEventStatus(), "Event status", () => this.activateStatusBar('toggle'));

      if (this.enlargeMain() || (this.has_canvas && this.hasObjectsToDraw()))
         menu.addchk((this.enlargeMain('state')=='on'), "Enlarge " + (this.iscan ? "canvas" : "pad"), () => this.enlargePad());

      let fname = this.this_pad_name;
      if (!fname) fname = this.iscan ? "canvas" : "pad";
      menu.add("Save as "+fname+".png", fname+".png", () => this.saveAs("png", false));
      menu.add("Save as "+fname+".svg", fname+".svg", () => this.saveAs("svg", false));

      return true;
   }

   /** @summary Show pad context menu
     * @private */
   RPadPainter.prototype.padContextMenu = function(evnt) {
      if (evnt.stopPropagation) {
         // this is normal event processing and not emulated jsroot event
         // for debug purposes keep original context menu for small region in top-left corner
         let pos = d3.pointer(evnt, this.svg_this_pad().node());

         if (pos && (pos.length==2) && (pos[0] >= 0) && (pos[0] < 10) && (pos[1] >= 0) && (pos[1] < 10)) return;

         evnt.stopPropagation(); // disable main context menu
         evnt.preventDefault();  // disable browser context menu

         let fp = this.getFramePainter();
         if (fp) fp.setLastEventPos();
      }

      jsrp.createMenu(evnt, this).then(menu => {
         this.fillContextMenu(menu);
         return this.fillObjectExecMenu(menu);
      }).then(menu => menu.show());
   }

   /** @summary Redraw pad means redraw ourself
     * @returns {Promise} when redrawing ready */
   RPadPainter.prototype.redrawPad = function(reason) {

      let sync_promise = this.syncDraw(reason);
      if (sync_promise === false) {
         console.log('Prevent RPad redrawing');
         return Promise.resolve(false);
      }

      let showsubitems = true;
      let redrawNext = indx => {
         while (indx < this.painters.length) {
            let sub = this.painters[indx++], res = 0;
            if (showsubitems || sub.this_pad_name)
               res = sub.redraw(reason);

            if (jsrp.isPromise(res))
               return res.then(() => redrawNext(indx));
         }
         return Promise.resolve(true);
      };

      return sync_promise.then(() => {
         if (this.iscan) {
            this.createCanvasSvg(2);
         } else {
            showsubitems = this.createPadSvg(true);
         }
         return redrawNext(0);
      }).then(() => {
         if (jsrp.getActivePad() === this) {
            let canp = this.getCanvPainter();
            if (canp) canp.producePadEvent("padredraw", this);
         }
         this.confirmDraw();
         return true;
      });
   }

   /** @summary redraw pad */
   RPadPainter.prototype.redraw = function(reason) {
      return this.redrawPad(reason);
   }


   /** @summary Checks if pad should be redrawn by resize
     * @private */
   RPadPainter.prototype.needRedrawByResize = function() {
      let elem = this.svg_this_pad();
      if (!elem.empty() && elem.property('can3d') === JSROOT.constants.Embed3D.Overlay) return true;

      for (let i = 0; i < this.painters.length; ++i)
         if (typeof this.painters[i].needRedrawByResize === 'function')
            if (this.painters[i].needRedrawByResize()) return true;

      return false;
   }

   /** @summary Check resize of canvas */
   RPadPainter.prototype.checkCanvasResize = function(size, force) {

      if (!this.iscan && this.has_canvas) return false;

      let sync_promise = this.syncDraw("canvas_resize");
      if (sync_promise === false) return false;

      if ((size === true) || (size === false)) { force = size; size = null; }

      if (size && (typeof size === 'object') && size.force) force = true;

      if (!force) force = this.needRedrawByResize();

      let changed = false,
          redrawNext = indx => {
             if (!changed || (indx >= this.painters.length)) {
                this.confirmDraw();
                return changed;
             }

             let res = this.painters[indx].redraw(force ? "redraw" : "resize");
             if (!jsrp.isPromise(res)) res = Promise.resolve();
              return res.then(() => redrawNext(indx+1));
          };

      return sync_promise.then(() => {

         changed = this.createCanvasSvg(force ? 2 : 1, size);

         // if canvas changed, redraw all its subitems.
         // If redrawing was forced for canvas, same applied for sub-elements
         return redrawNext(0);
      });
   }

   /** @summary update RPad object
     * @private */
   RPadPainter.prototype.updateObject = function(obj) {
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


   /** @summary Add object painter to list of primitives
     * @private */
   RPadPainter.prototype.addObjectPainter = function(objpainter, lst, indx) {
      if (objpainter && lst && lst[indx] && (objpainter.snapid === undefined)) {
         // keep snap id in painter, will be used for the
         if (this.painters.indexOf(objpainter) < 0)
            this.painters.push(objpainter);
         objpainter.assignSnapId(lst[indx].fObjectID);
         if (!objpainter.rstyle) objpainter.rstyle = lst[indx].fStyle || this.rstyle;
      }
   }

   /** @summary Extract properties from TObjectDisplayItem */
   RPadPainter.prototype.extractTObjectProp = function(snap) {
      if (snap.fColIndex && snap.fColValue) {
         let colors = this.root_colors || jsrp.root_colors;
         for (let k = 0; k < snap.fColIndex.length; ++k)
            colors[snap.fColIndex[k]] = snap.fColValue[k];
       }

      // painter used only for evaluation of attributes
      let pattr = new JSROOT.ObjectPainter(), obj = snap.fObject;
      pattr.assignObject(snap);
      pattr.csstype = snap.fCssType;
      pattr.rstyle = snap.fStyle;

      snap.fOption = pattr.v7EvalAttr("options", "");

      let extract_color = (member_name, attr_name) => {
         let col = pattr.v7EvalColor(attr_name, "");
         if (col) obj[member_name] = jsrp.addColor(col, this.root_colors);
      }

      // handle TAttLine
      if ((obj.fLineColor !== undefined) && (obj.fLineWidth !== undefined) && (obj.fLineStyle !== undefined)) {
         extract_color("fLineColor", "line_color");
         obj.fLineWidth = pattr.v7EvalAttr("line_width", obj.fLineWidth);
         obj.fLineStyle = pattr.v7EvalAttr("line_style", obj.fLineStyle);
      }

      // handle TAttFill
      if ((obj.fFillColor !== undefined) && (obj.fFillStyle !== undefined)) {
         extract_color("fFillColor", "fill_color");
         obj.fFillStyle = pattr.v7EvalAttr("fill_style", obj.fFillStyle);
      }

      // handle TAttMarker
      if ((obj.fMarkerColor !== undefined) && (obj.fMarkerStyle !== undefined) && (obj.fMarkerSize !== undefined)) {
         extract_color("fMarkerColor", "marker_color");
         obj.fMarkerStyle = pattr.v7EvalAttr("marker_style", obj.fMarkerStyle);
         obj.fMarkerSize = pattr.v7EvalAttr("marker_size", obj.fMarkerSize);
      }

      // handle TAttText
      if ((obj.fTextColor !== undefined) && (obj.fTextAlign !== undefined) && (obj.fTextAngle !== undefined) && (obj.fTextSize !== undefined)) {
         extract_color("fTextColor", "text_color");
         obj.fTextAlign = pattr.v7EvalAttr("text_align", obj.fTextAlign);
         obj.fTextAngle = pattr.v7EvalAttr("text_angle", obj.fTextAngle);
         obj.fTextSize = pattr.v7EvalAttr("text_size", obj.fTextSize);
         // TODO: v7 font handling differs much from v6, ignore for the moment
      }
   }

   /** @summary Function called when drawing next snapshot from the list
     * @returns {Promise} with pad painter when ready
     * @private */
   RPadPainter.prototype.drawNextSnap = function(lst, indx) {

      if (indx===undefined) {
         indx = -1;
         // flag used to prevent immediate pad redraw during first draw
         this._snaps_map = {}; // to control how much snaps are drawn
         this._num_primitives = lst ? lst.length : 0;
         this._auto_color_cnt = 0;
      }

      delete this.next_rstyle;

      ++indx; // change to the next snap

      if (!lst || indx >= lst.length) {
         delete this._snaps_map;
         delete this._auto_color_cnt;
         return Promise.resolve(this);
      }

      let snap = lst[indx],
          snapid = snap.fObjectID,
          cnt = this._snaps_map[snapid],
          objpainter = null;

      if (cnt) cnt++; else cnt=1;
      this._snaps_map[snapid] = cnt; // check how many objects with same snapid drawn, use them again

      // empty object, no need to do something, take next
      if (snap.fDummy) return this.drawNextSnap(lst, indx);

      // first appropriate painter for the object
      // if same object drawn twice, two painters will exists
      for (let k=0; k<this.painters.length; ++k) {
         if (this.painters[k].snapid === snapid)
            if (--cnt === 0) { objpainter = this.painters[k]; break;  }
      }

      if (objpainter) {

         if (snap._typename == "ROOT::Experimental::RPadDisplayItem")  // subpad
            return objpainter.redrawPadSnap(snap).then(ppainter => {
               this.addObjectPainter(ppainter, lst, indx);
               return this.drawNextSnap(lst, indx);
            });

         if (snap._typename === "ROOT::Experimental::TObjectDisplayItem")
            this.extractTObjectProp(snap);

         let promise;

         if (objpainter.updateObject(snap.fDrawable || snap.fObject || snap, snap.fOption || ""))
            promise = objpainter.redraw();

         if (!jsrp.isPromise(promise)) promise = Promise.resolve(true);

         return promise.then(() => this.drawNextSnap(lst, indx)); // call next
      }

      if (snap._typename == "ROOT::Experimental::RPadDisplayItem") { // subpad

         let subpad = snap; // not subpad, but just attributes

         let padpainter = new RPadPainter(this.getDom(), subpad, false);
         padpainter.decodeOptions("");
         padpainter.addToPadPrimitives(this.this_pad_name); // only set parent pad name
         padpainter.assignSnapId(snap.fObjectID);
         padpainter.rstyle = snap.fStyle;

         padpainter.createPadSvg();

         if (snap.fPrimitives && snap.fPrimitives.length > 0)
            padpainter.addPadButtons();

         // we select current pad, where all drawing is performed
         let prev_name = padpainter.selectCurrentPad(padpainter.this_pad_name);

         return padpainter.drawNextSnap(snap.fPrimitives).then(() => {
            padpainter.selectCurrentPad(prev_name);
            return this.drawNextSnap(lst, indx);
         });
      }

      // will be used in addToPadPrimitives to assign style to sub-painters
      this.next_rstyle = lst[indx].fStyle || this.rstyle;

      if (snap._typename === "ROOT::Experimental::TObjectDisplayItem") {

         // identifier used in RObjectDrawable
         const webSnapIds = { kNone: 0,  kObject: 1, kColors: 4, kStyle: 5, kPalette: 6 };

         if (snap.fKind == webSnapIds.kStyle) {
            JSROOT.extend(JSROOT.gStyle, snap.fObject);
            return this.drawNextSnap(lst, indx);
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
            return this.drawNextSnap(lst, indx);
         }

         if (snap.fKind == webSnapIds.kPalette) {
            let arr = snap.fObject.arr, palette = [];
            for (let n = 0; n < arr.length; ++n)
               palette[n] =  arr[n].fString;
            this.custom_palette = new JSROOT.ColorPalette(palette);
            return this.drawNextSnap(lst, indx);
         }

         if (!this.getFramePainter())
            return JSROOT.draw(this.getDom(), { _typename: "TFrame", $dummy: true }, "")
                         .then(() => this.drawNextSnap(lst, indx-1)); // call same object again

         this.extractTObjectProp(snap);
      }

      // TODO - fDrawable is v7, fObject from v6, maybe use same data member?
      return JSROOT.draw(this.getDom(), snap.fDrawable || snap.fObject || snap, snap.fOption || "").then(objpainter => {
         this.addObjectPainter(objpainter, lst, indx);
         return this.drawNextSnap(lst, indx);
      });
   }

   /** @summary Search painter with specified snapid, also sub-pads are checked
     * @private */
   RPadPainter.prototype.findSnap = function(snapid, onlyid) {

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

         if (!onlyid && (typeof sub.findSnap === 'function'))
            sub = sub.findSnap(snapid);
         else if (!check(sub.snapid))
            sub = null;

         if (sub) return sub;
      }

      return null;
   }

   /** @summary Redraw pad snap
     * @desc Online version of drawing pad primitives
     * @returns {Promise} with pad painter*/
   RPadPainter.prototype.redrawPadSnap = function(snap) {
      // for the pad/canvas display item contains list of primitives plus pad attributes

      if (!snap || !snap.fPrimitives) return Promise.resolve(this);

      // for the moment only window size attributes are provided
      // let padattr = { fCw: snap.fWinSize[0], fCh: snap.fWinSize[1], fTitle: snap.fTitle };

      // if canvas size not specified in batch mode, temporary use 900x700 size
      // if (this.batch_mode && this.iscan && (!padattr.fCw || !padattr.fCh)) { padattr.fCw = 900; padattr.fCh = 700; }

      if (this.iscan && this._websocket && snap.fTitle && !this.embed_canvas && (typeof document !== "undefined"))
         document.title = snap.fTitle;

      if (this.snapid === undefined) {
         // first time getting snap, create all gui elements first

         this.assignSnapId(snap.fObjectID);

         this.draw_object = snap;
         this.pad = snap;

         if (this.batch_mode && this.iscan)
             this._fixed_size = true;

         if (JSROOT.BrowserLayout && !this.batch_mode && !this.use_openui && !this.brlayout) {
            let mainid = this.selectDom().attr("id");
            if (mainid && (typeof mainid == "string")) {
               this.brlayout = new JSROOT.BrowserLayout(mainid, null, this);
               this.brlayout.create(mainid, true);
               this.setDom(this.brlayout.drawing_divid()); // need to create canvas
               jsrp.registerForResize(this.brlayout);
            }
         }

         this.createCanvasSvg(0);
         this.addPadButtons(true);

         return this.drawNextSnap(snap.fPrimitives);
      }

      // update only pad/canvas attributes
      this.updateObject(snap);

      // apply all changes in the object (pad or canvas)
      if (this.iscan) {
         this.createCanvasSvg(2);
      } else {
         this.createPadSvg(true);
      }

      let isanyfound = false, isanyremove = false;

      // find and remove painters which no longer exists in the list
      for (let k=0;k<this.painters.length;++k) {
         let sub = this.painters[k];
         if (sub.snapid===undefined) continue; // look only for painters with snapid

         snap.fPrimitives.forEach(prim => {
            if (sub && (prim.fObjectID === sub.snapid)) {
               sub = null; isanyfound = true;
            }
         });

         if (sub) {
            // remove painter which does not found in the list of snaps
            this.painters.splice(k--,1);
            sub.cleanup(); // cleanup such painter
            isanyremove = true;
         }
      }

      if (isanyremove) {
         delete this.pads_cache;
      }

      if (!isanyfound) {
         let fp = this.getFramePainter();
         // cannot preserve ROOT6 frame - it must be recreated
         if (fp && fp.is_root6()) fp = null;
         for (let k = 0; k < this.painters.length; ++k)
             if (fp !== this.painters[k])
               this.painters[k].cleanup();
         this.painters = [];
         delete this.main_painter_ref;
         if (fp) {
            this.painters.push(fp);
            fp.cleanFrameDrawings();
            fp.redraw(); // need to create all layers again
         }
         if (this.removePadButtons) this.removePadButtons();
         this.addPadButtons(true);
      }

      let prev_name = this.selectCurrentPad(this.this_pad_name);

      return this.drawNextSnap(snap.fPrimitives).then(() => {
         this.selectCurrentPad(prev_name);

         if (jsrp.getActivePad() === this) {
            let canp = this.getCanvPainter();
            if (canp) canp.producePadEvent("padredraw", this);
         }

         return this;
      });
   }

   /** @summary Create image for the pad
     * @desc Used with web-based canvas to create images for server side
     * @returns {Promise} with image data, coded with btoa() function
     * @private */
   RPadPainter.prototype.createImage = function(format) {
      // use https://github.com/MrRio/jsPDF in the future here
      if (format == "pdf")
         return Promise.resolve(btoa("dummy PDF file"));

      if ((format == "png") || (format == "jpeg") || (format == "svg"))
         return this.produceImage(true, format).then(res => {
            if (!res || (format == "svg")) return res;
            let separ = res.indexOf("base64,");
            return (separ>0) ? res.substr(separ+7) : "";
         });

      return Promise.resolve("");
   }

   /** @summary Show context menu for specified item
     * @private */
   RPadPainter.prototype.itemContextMenu = function(name) {
       let rrr = this.svg_this_pad().node().getBoundingClientRect(),
           evnt = { clientX: rrr.left+10, clientY: rrr.top + 10 };

       // use timeout to avoid conflict with mouse click and automatic menu close
       if (name == "pad")
          return setTimeout(() => this.padContextMenu(evnt), 50);

       let selp = null, selkind;

       switch(name) {
          case "xaxis":
          case "yaxis":
          case "zaxis":
             selp = this.getMainPainter();
             selkind = name[0];
             break;
          case "frame":
             selp = this.getFramePainter();
             break;
          default: {
             let indx = parseInt(name);
             if (Number.isInteger(indx)) selp = this.painters[indx];
          }
       }

       if (!selp || (typeof selp.fillContextMenu !== 'function')) return;

       jsrp.createMenu(evnt, selp).then(menu => {
          if (selp.fillContextMenu(menu, selkind))
             setTimeout(() => menu.show(), 50);
       });
   }

   /** @summary Save pad in specified format
     * @desc Used from context menu */
   RPadPainter.prototype.saveAs = function(kind, full_canvas, filename) {
      if (!filename) {
         filename = this.this_pad_name;
         if (filename.length === 0) filename = this.iscan ? "canvas" : "pad";
         filename += "." + kind;
      }
      this.produceImage(full_canvas, kind).then(imgdata => {
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
   RPadPainter.prototype.produceImage = function(full_canvas, file_format) {

      let use_frame = (full_canvas === "frame");

      let elem = use_frame ? this.getFrameSvg() : (full_canvas ? this.getCanvSvg() : this.svg_this_pad());

      if (elem.empty()) return Promise.resolve("");

      let painter = (full_canvas && !use_frame) ? this.getCanvPainter() : this;

      let items = []; // keep list of replaced elements, which should be moved back at the end

      if (!use_frame) // do not make transformations for the frame
      painter.forEachPainterInPad(pp => {

         let item = { prnt: pp.svg_this_pad() };
         items.push(item);

         // remove buttons from each subpad
         let btns = pp.getLayerSvg("btns_layer", this.this_pad_name);
         item.btns_node = btns.node();
         if (item.btns_node) {
            item.btns_prnt = item.btns_node.parentNode;
            item.btns_next = item.btns_node.nextSibling;
            btns.remove();
         }

         let main = pp.getFramePainter();
         if (!main || (typeof main.render3D !== 'function') || (typeof main.access3dKind != 'function')) return;

         let can3d = main.access3dKind();

         if ((can3d !== JSROOT.constants.Embed3D.Overlay) && (can3d !== JSROOT.constants.Embed3D.Embed)) return;

         let sz2 = main.getSizeFor3d(JSROOT.constants.Embed3D.Embed); // get size and position of DOM element as it will be embed

         let canvas = main.renderer.domElement;
         main.render3D(0); // WebGL clears buffers, therefore we should render scene and convert immediately
         let dataUrl = canvas.toDataURL("image/png");

         // remove 3D drawings

         if (can3d === JSROOT.constants.Embed3D.Embed) {
            item.foreign = item.prnt.select("." + sz2.clname);
            item.foreign.remove();
         }

         let svg_frame = main.getFrameSvg();
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
      if (use_frame) {
         let fp = this.getFramePainter();
         width = fp.getFrameWidth();
         height = fp.getFrameHeight();
      }

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

   /** @summary Process pad button click */
   RPadPainter.prototype.clickPadButton = function(funcname, evnt) {

      if (funcname == "CanvasSnapShot") return this.saveAs("png", true);

      if (funcname == "enlargePad") return this.enlargePad();

      if (funcname == "PadSnapShot") return this.saveAs("png", false);

      if (funcname == "PadContextMenus") {

         if (evnt) {
            evnt.preventDefault();
            evnt.stopPropagation();
         }

         if (jsrp.closeMenu && jsrp.closeMenu()) return;

         jsrp.createMenu(evnt, this).then(menu => {
            menu.add("header:Menus");

            if (this.iscan)
               menu.add("Canvas", "pad", this.itemContextMenu);
            else
               menu.add("Pad", "pad", this.itemContextMenu);

            if (this.getFramePainter())
               menu.add("Frame", "frame", this.itemContextMenu);

            let main = this.getMainPainter(); // here pad painter method

            if (main) {
               menu.add("X axis", "xaxis", this.itemContextMenu);
               menu.add("Y axis", "yaxis", this.itemContextMenu);
               if ((typeof main.getDimension === 'function') && (main.getDimension() > 1))
                  menu.add("Z axis", "zaxis", this.itemContextMenu);
            }

            if (this.painters && (this.painters.length>0)) {
               menu.add("separator");
               let shown = [];
               for (let n=0;n<this.painters.length;++n) {
                  let pp = this.painters[n];
                  let obj = pp ? pp.getObject() : null;
                  if (!obj || (shown.indexOf(obj)>=0)) continue;

                  let name = ('_typename' in obj) ? (obj._typename + "::") : "";
                  if ('fName' in obj) name += obj.fName;
                  if (name.length==0) name = "item" + n;
                  menu.add(name, n, this.itemContextMenu);
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

         if (typeof pp.clickPadButton == 'function')
            pp.clickPadButton(funcname);

         if (!done && (typeof pp.clickButton == 'function'))
            done = pp.clickButton(funcname);
      }
   }

   /** @summary Add button to the pad
     * @private */
   RPadPainter.prototype.addPadButton = function(_btn, _tooltip, _funcname, _keyname) {
      if (!JSROOT.settings.ToolBar || JSROOT.batch_mode || this.batch_mode) return;

      if (!this._buttons) this._buttons = [];
      // check if there are duplications

      for (let k=0;k<this._buttons.length;++k)
         if (this._buttons[k].funcname == _funcname) return;

      this._buttons.push({ btn: _btn, tooltip: _tooltip, funcname: _funcname, keyname: _keyname });

      let iscan = this.iscan || !this.has_canvas;
      if (!iscan && (_funcname.indexOf("Pad")!=0) && (_funcname !== "enlargePad")) {
         let cp = this.getCanvPainter();
         if (cp && (cp!==this)) cp.addPadButton(_btn, _tooltip, _funcname);
      }
   }

   /** @summary Add buttons for pad or canvas
     * @private */
   RPadPainter.prototype.addPadButtons = function(is_online) {

      this.addPadButton("camera", "Create PNG", this.iscan ? "CanvasSnapShot" : "PadSnapShot", "Ctrl PrintScreen");

      if (JSROOT.settings.ContextMenu)
         this.addPadButton("question", "Access context menus", "PadContextMenus");

      let add_enlarge = !this.iscan && this.has_canvas && this.hasObjectsToDraw()

      if (add_enlarge || this.enlargeMain('verify'))
         this.addPadButton("circle", "Enlarge canvas", "enlargePad");

      if (is_online && this.brlayout) {
         this.addPadButton("diamand", "Toggle Ged", "ToggleGed");
         this.addPadButton("three_circles", "Toggle Status", "ToggleStatus");
      }

   }

   /** @summary Show pad buttons
     * @private */
   RPadPainter.prototype.showPadButtons = function() {
      if (!this._buttons) return;

      JSROOT.require(['interactive']).then(inter => {
         inter.PadButtonsHandler.assign(this);
         this.showPadButtons();
      });
   }

   /** @summary Calculates RPadLength value */
   RPadPainter.prototype.getPadLength = function(vertical, len, frame_painter) {
      let sign = vertical ? -1 : 1,
          rect, res,
          getV = (indx, dflt) => (indx < len.fArr.length) ? len.fArr[indx] : dflt,
          getRect = () => {
             if (!rect)
                rect = frame_painter ? frame_painter.getFrameRect() : this.getPadRect();
             return rect;
          };

      if (frame_painter) {
         let user = getV(2), func = vertical ? "gry" : "grx";
         if ((user !== undefined) && frame_painter[func])
            res = frame_painter[func](user);
      }

      if (res === undefined)
         res = vertical ? getRect().height : 0;

      let norm = getV(0, 0), pixel = getV(1, 0);

      res += sign*pixel;

      if (norm)
         res += sign * (vertical ? getRect().height : getRect().width) * norm;

      return Math.round(res);
   }


   /** @summary Calculates pad position for RPadPos values
     * @param {object} pos - instance of RPadPos
     * @param {object} frame_painter - if drawing will be performed inside frame, frame painter */
   RPadPainter.prototype.getCoordinate = function(pos, frame_painter) {
      return {
         x: this.getPadLength(false, pos.fHoriz, frame_painter),
         y: this.getPadLength(true, pos.fVert, frame_painter)
      }
   }

   /** @summary Decode pad draw options
     * @private */
   RPadPainter.prototype.decodeOptions = function(opt) {
      let pad = this.getObject();
      if (!pad) return;

      let d = new JSROOT.DrawOptions(opt);

      if (d.check('WEBSOCKET') && this.openWebsocket) this.openWebsocket();
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
      let painter = new RPadPainter(divid, pad, false);
      painter.decodeOptions(opt);

      if (painter.getCanvSvg().empty()) {
         painter.has_canvas = false;
         painter.this_pad_name = "";
         painter.setTopPainter();
      } else {
         painter.addToPadPrimitives(painter.pad_name); // must be here due to pad painter
      }

      painter.createPadSvg();

      if (painter.matchObjectType("TPad") && (!painter.has_canvas || painter.hasObjectsToDraw())) {
         painter.addPadButtons();
      }

      // we select current pad, where all drawing is performed
      let prev_name = painter.has_canvas ? painter.selectCurrentPad(painter.this_pad_name) : undefined;

      jsrp.selectActivePad({ pp: painter, active: false });

      // flag used to prevent immediate pad redraw during first draw
      return painter.drawPrimitives().then(() => {
         painter.showPadButtons();
         // we restore previous pad name
         painter.selectCurrentPad(prev_name);
         return painter;
      });
   }

   // ==========================================================================================

   function RCanvasPainter(divid, canvas) {
      // used for online canvas painter
      RPadPainter.call(this, divid, canvas, true);
      this._websocket = null;
      this.tooltip_allowed = JSROOT.settings.Tooltip;
      this.v7canvas = true;
   }

   RCanvasPainter.prototype = Object.create(RPadPainter.prototype);

   /** @summary Cleanup canvas painter */
   RCanvasPainter.prototype.cleanup = function() {
      delete this._websocket;
      delete this._submreq;

     if (this._changed_layout)
         this.setLayoutKind('simple');
      delete this._changed_layout;

      RPadPainter.prototype.cleanup.call(this);
   }

   /** @summary Returns layout kind */
   RCanvasPainter.prototype.getLayoutKind = function() {
      let origin = this.selectDom('origin'),
         layout = origin.empty() ? "" : origin.property('layout');
      return layout || 'simple';
   }

   /** @summary Set canvas layout kind */
   RCanvasPainter.prototype.setLayoutKind = function(kind, main_selector) {
      let origin = this.selectDom('origin');
      if (!origin.empty()) {
         if (!kind) kind = 'simple';
         origin.property('layout', kind);
         origin.property('layout_selector', (kind != 'simple') && main_selector ? main_selector : null);
         this._changed_layout = (kind !== 'simple'); // use in cleanup
      }
   }

   /** @summary Changes layout
     * @returns {Promise} indicating when finished */
   RCanvasPainter.prototype.changeLayout = function(layout_kind) {
      let current = this.getLayoutKind();
      if (current == layout_kind)
         return Promise.resolve(true);

      let origin = this.selectDom('origin'),
          sidebar = origin.select('.side_panel'),
          main = this.selectDom(), lst = [];

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
     * @returns {Promise} indicating when ready
     * @private */
   RCanvasPainter.prototype.toggleProjection = function(kind) {
      delete this.proj_painter;

      if (kind) this.proj_painter = 1; // just indicator that drawing can be preformed

      if (this.showUI5ProjectionArea)
         return this.showUI5ProjectionArea(kind);

      let layout = 'simple';

      if (kind == "X") layout = 'vert2_31'; else
      if (kind == "Y") layout = 'horiz2_13';

      return this.changeLayout(layout);
   }

   /** @summary Draw projection for specified histogram
     * @private */
   RCanvasPainter.prototype.drawProjection = function( /*kind,hist*/) {
      // dummy for the moment
   }

   /** @summary Draw in side panel
     * @private */
   RCanvasPainter.prototype.drawInSidePanel = function(canv, opt) {
      let side = this.selectDom('origin').select(".side_panel");
      if (side.empty()) return Promise.resolve(null);
      return JSROOT.draw(side.node(), canv, opt);
   }

   /** @summary Checks if canvas shown inside ui5 widget
     * @desc Function should be used only from the func which supposed to be replaced by ui5
     * @private */
   RCanvasPainter.prototype.testUI5 = function() {
      if (!this.use_openui) return false;
      console.warn("full ui5 should be used - not loaded yet? Please check!!");
      return true;
   }

   /** @summary Show message
     * @desc Used normally with web-based canvas and handled in ui5
     * @private */
   RCanvasPainter.prototype.showMessage = function(msg) {
      if (!this.testUI5())
         jsrp.showProgress(msg, 7000);
   }

   /** @summary Function called when canvas menu item Save is called */
   RCanvasPainter.prototype.saveCanvasAsFile = function(fname) {
      let pnt = fname.indexOf(".");
      this.createImage(fname.substr(pnt+1))
          .then(res => { console.log('save', fname, res.length); this.sendWebsocket("SAVE:" + fname + ":" + res); });
   }

   /** @summary Send command to server to save canvas with specified name
     * @desc Should be only used in web-based canvas
     * @private */
   RCanvasPainter.prototype.sendSaveCommand = function(fname) {
      this.sendWebsocket("PRODUCE:" + fname);
   }

   /** @summary Send message via web socket
     * @private */
   RCanvasPainter.prototype.sendWebsocket = function(msg, chid) {
      if (this._websocket)
         this._websocket.send(msg, chid);
   }

   /** @summary Close websocket connection to canvas
     * @private */
   RCanvasPainter.prototype.closeWebsocket = function(force) {
      if (this._websocket) {
         this._websocket.close(force);
         this._websocket.cleanup();
         delete this._websocket;
      }
   }

   /** @summary Create websocket for the canvas
     * @private */
   RCanvasPainter.prototype.openWebsocket = function(socket_kind) {
      this.closeWebsocket();

      this._websocket = new JSROOT.WebWindowHandle(socket_kind);
      this._websocket.setReceiver(this);
      this._websocket.connect();
   }

   /** @summary Use provided connection for the web canvas
     * @private */
   RCanvasPainter.prototype.useWebsocket = function(handle) {
      this.closeWebsocket();

      this._websocket = handle;
      this._websocket.setReceiver(this);
      this._websocket.connect();
   }

   /** @summary Hanler for websocket open event
     * @private */
   RCanvasPainter.prototype.onWebsocketOpened = function(/*handle*/) {
   }

   /** @summary Hanler for websocket close event
     * @private */
   RCanvasPainter.prototype.onWebsocketClosed = function(/*handle*/) {
      if (!this.embed_canvas)
         jsrp.closeCurrentWindow();
   }

   /** @summary Hanler for websocket message
     * @private */
   RCanvasPainter.prototype.onWebsocketMsg = function(handle, msg) {
      console.log("GET_MSG " + msg.substr(0,30));

      if (msg == "CLOSE") {
         this.onWebsocketClosed();
         this.closeWebsocket(true);
      } else if (msg.substr(0,5)=='SNAP:') {
         msg = msg.substr(5);
         let p1 = msg.indexOf(":"),
             snapid = msg.substr(0,p1),
             snap = JSROOT.parse(msg.substr(p1+1));
         this.syncDraw(true)
             .then(() => this.redrawPadSnap(snap))
             .then(() => {
                 handle.send("SNAPDONE:" + snapid); // send ready message back when drawing completed
                 this.confirmDraw();
              });
      } else if (msg.substr(0,4)=='JSON') {
         let obj = JSROOT.parse(msg.substr(4));
         // console.log("get JSON ", msg.length-4, obj._typename);
         this.redrawObject(obj);
      } else if (msg.substr(0,9)=="REPL_REQ:") {
         this.processDrawableReply(msg.substr(9));
      } else if (msg.substr(0,4)=='CMD:') {
         msg = msg.substr(4);
         let p1 = msg.indexOf(":"),
             cmdid = msg.substr(0,p1),
             cmd = msg.substr(p1+1),
             reply = "REPLY:" + cmdid + ":";
         if ((cmd == "SVG") || (cmd == "PNG") || (cmd == "JPEG")) {
            this.createImage(cmd.toLowerCase())
                .then(res => handle.send(reply + res));
         } else if (cmd.indexOf("ADDPANEL:") == 0) {
            let relative_path = cmd.substr(9);
            if (!this.showUI5Panel) {
               handle.send(reply + "false");
            } else {

               let conn = new JSROOT.WebWindowHandle(handle.kind);

               // set interim receiver until first message arrives
               conn.setReceiver({
                  cpainter: this,

                  onWebsocketOpened: function() {
                  },

                  onWebsocketMsg: function(panel_handle, msg) {
                     let panel_name = (msg.indexOf("SHOWPANEL:")==0) ? msg.substr(10) : "";
                     this.cpainter.showUI5Panel(panel_name, panel_handle)
                                  .then(res => handle.send(reply + (res ? "true" : "false")));
                  },

                  onWebsocketClosed: function() {
                     // if connection failed,
                     handle.send(reply + "false");
                  },

                  onWebsocketError: function() {
                     // if connection failed,
                     handle.send(reply + "false");
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
               conn.connect(addr);
            }
         } else {
            console.log('Unrecognized command ' + cmd);
            handle.send(reply);
         }
      } else if ((msg.substr(0,7)=='DXPROJ:') || (msg.substr(0,7)=='DYPROJ:')) {
         let kind = msg[1],
             hist = JSROOT.parse(msg.substr(7));
         this.drawProjection(kind, hist);
      } else if (msg.substr(0,5)=='SHOW:') {
         let that = msg.substr(5),
             on = that[that.length-1] == '1';
         this.showSection(that.substr(0,that.length-2), on);
      } else {
         console.log("unrecognized msg len:" + msg.length + " msg:" + msg.substr(0,20));
      }
   }

   /** @summary Submit request to RDrawable object on server side */
   RCanvasPainter.prototype.submitDrawableRequest = function(kind, req, painter, method) {

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

      this.sendWebsocket("REQ:" + msg);
      return req;
   }

   /** @summary Submit menu request
     * @private */
   RCanvasPainter.prototype.submitMenuRequest = function(painter, menukind, reqid) {
      return new Promise(resolveFunc => {
         this.submitDrawableRequest("", {
            _typename: "ROOT::Experimental::RDrawableMenuRequest",
            menukind: menukind || "",
            menureqid: reqid, // used to identify menu request
         }, painter, resolveFunc);
      });
   }

   /** @summary Submit executable command for given painter */
   RCanvasPainter.prototype.submitExec = function(painter, exec, subelem) {
      console.log('SubmitExec', exec, painter.snapid, subelem);

      // snapid is intentionally ignored - only painter.snapid has to be used
      if (!this._websocket) return;

      if (subelem) {
         if ((subelem == "x") || (subelem == "y") || (subelem == "z"))
            exec = subelem + "axis#" + exec;
         else
            return console.log(`not recoginzed subelem ${subelem} in SubmitExec`);
       }

      this.submitDrawableRequest("", {
         _typename: "ROOT::Experimental::RDrawableExecRequest",
         exec: exec
      }, painter);
   }

   /** @summary Process reply from request to RDrawable */
   RCanvasPainter.prototype.processDrawableReply = function(msg) {
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
         this.submitDrawableRequest(req._kind, req._nextreq, req._painter, req._method);
   }

   RCanvasPainter.prototype.showSection = function(that, on) {
      switch(that) {
         case "Menu": break;
         case "StatusBar": break;
         case "Editor": break;
         case "ToolBar": break;
         case "ToolTips": this.setTooltipAllowed(on); break;
      }
      return Promise.resolve(true);
   }

   /** @summary Method informs that something was changed in the canvas
     * @desc used to update information on the server (when used with web6gui)
     * @private */
   RCanvasPainter.prototype.processChanges = function(kind, painter, subelem) {
      // check if we could send at least one message more - for some meaningful actions
      if (!this._websocket || !this._websocket.canSend(2) || (typeof kind !== "string")) return;

      let msg = "";
      if (!painter) painter = this;
      switch (kind) {
         case "sbits":
            console.log("Status bits in RCanvas are changed - that to do?");
            break;
         case "frame": // when moving frame
         case "zoom":  // when changing zoom inside frame
            console.log("Frame moved or zoom is changed - that to do?");
            break;
         case "pave_moved":
            console.log('TPave is moved inside RCanvas - that to do?');
            break;
         default:
            if ((kind.substr(0,5) == "exec:") && painter && painter.snapid) {
               this.submitExec(painter, kind.substr(5), subelem);
            } else {
               console.log("UNPROCESSED CHANGES", kind);
            }
      }

      if (msg) {
         console.log("RCanvas::processChanges want to send  " + msg.length + "  " + msg.substr(0,40));
      }
   }

   /** @summary Handle pad button click event
     * @private */
   RCanvasPainter.prototype.clickPadButton = function(funcname, evnt) {
      if (funcname == "ToggleGed") return this.activateGed(this, null, "toggle");
      if (funcname == "ToggleStatus") return this.activateStatusBar("toggle");
      RPadPainter.prototype.clickPadButton.call(this, funcname, evnt);
   }

   /** @summary returns true when event status area exist for the canvas */
   RCanvasPainter.prototype.hasEventStatus = function() {
      if (this.testUI5()) return false;
      return this.brlayout ? this.brlayout.hasStatus() : false;
   }

   /** @summary Show/toggle event status bar
     * @private */
   RCanvasPainter.prototype.activateStatusBar = function(state) {
      if (this.testUI5()) return;
      if (this.brlayout)
         this.brlayout.createStatusLine(23, state);
      this.processChanges("sbits", this);
   }

   /** @summary Returns true if GED is present on the canvas */
   RCanvasPainter.prototype.hasGed = function() {
      if (this.testUI5()) return false;
      return this.brlayout ? this.brlayout.hasContent() : false;
   }

   /** @summary Function used to de-activate GED
     * @private */
   RCanvasPainter.prototype.removeGed = function() {
      if (this.testUI5()) return;

      this.registerForPadEvents(null);

      if (this.ged_view) {
         this.ged_view.getController().cleanupGed();
         this.ged_view.destroy();
         delete this.ged_view;
      }
      if (this.brlayout)
         this.brlayout.deleteContent();

      this.processChanges("sbits", this);
   }

   /** @summary Function used to activate GED
     * @returns {Promise} when GED is there
     * @private */
   RCanvasPainter.prototype.activateGed = function(objpainter, kind, mode) {
      if (this.testUI5() || !this.brlayout)
         return Promise.resolve(false);

      if (this.brlayout.hasContent()) {
         if ((mode === "toggle") || (mode === false)) {
            this.removeGed();
         } else {
            let pp = objpainter ? objpainter.getPadPainter() : null;
            if (pp) pp.selectObjectPainter(objpainter);
         }

         return Promise.resolve(true);
      }

      if (mode === false)
         return Promise.resolve(false);

      let btns = this.brlayout.createBrowserBtns();

      JSROOT.require('interactive').then(inter => {

         inter.ToolbarIcons.createSVG(btns, inter.ToolbarIcons.diamand, 15, "toggle fix-pos mode")
                            .style("margin","3px").on("click", () => this.brlayout.toggleKind('fix'));

         inter.ToolbarIcons.createSVG(btns, inter.ToolbarIcons.circle, 15, "toggle float mode")
                            .style("margin","3px").on("click", () => this.brlayout.toggleKind('float'));

         inter.ToolbarIcons.createSVG(btns, inter.ToolbarIcons.cross, 15, "delete GED")
                            .style("margin","3px").on("click", () => this.removeGed());
      });

      // be aware, that jsroot_browser_hierarchy required for flexible layout that element use full browser area
      this.brlayout.setBrowserContent("<div class='jsroot_browser_hierarchy' id='ged_placeholder'>Loading GED ...</div>");
      this.brlayout.setBrowserTitle("GED");
      this.brlayout.toggleBrowserKind(kind || "float");

      return new Promise(resolveFunc => {

         JSROOT.require('openui5').then(() => {

            d3.select("#ged_placeholder").text("");

            sap.ui.define(["sap/ui/model/json/JSONModel", "sap/ui/core/mvc/XMLView"], (JSONModel,XMLView) => {

               let oModel = new JSONModel({ handle: null });

               XMLView.create({
                  viewName: "rootui5.canv.view.Ged"
               }).then(oGed => {

                  oGed.setModel(oModel);

                  oGed.placeAt("ged_placeholder");

                  this.ged_view = oGed;

                  // TODO: should be moved into Ged controller - it must be able to detect canvas painter itself
                  this.registerForPadEvents(oGed.getController().padEventsReceiver.bind(oGed.getController()));

                  let pp = objpainter ? objpainter.getPadPainter() : null;
                  if (pp) pp.selectObjectPainter(objpainter);

                  this.processChanges("sbits", this);

                  resolveFunc(true);
               });
            });
         });
      });
   }

   /** @summary produce JSON for RCanvas, which can be used to display canvas once again
     * @private */
   RCanvasPainter.prototype.produceJSON = function() {
      console.error('RCanvasPainter.produceJSON not yet implemented');
      return "";
   }

   function drawRCanvas(divid, can /*, opt */) {
      let nocanvas = !can;
      if (nocanvas)
         can = JSROOT.create("ROOT::Experimental::TCanvas");

      let painter = new RCanvasPainter(divid, can);
      painter.normal_canvas = !nocanvas;
      painter.createCanvasSvg(0);

      jsrp.selectActivePad({ pp: painter, active: false });

      return painter.drawPrimitives().then(() => {
         painter.addPadButtons();
         painter.showPadButtons();
         return painter;
      });
   }

   function drawPadSnapshot(divid, snap /*, opt*/) {
      let painter = new RCanvasPainter(divid, null);
      painter.normal_canvas = false;
      painter.batch_mode = JSROOT.batch_mode;
      return painter.syncDraw(true).then(() => painter.redrawPadSnap(snap)).then(() => {
         painter.confirmDraw();
         painter.showPadButtons();
         return painter;
      });
   }

   /** @summary Ensure RCanvas and RFrame for the painter object
     * @param {Object} painter  - painter object to process
     * @param {string|boolean} frame_kind  - false for no frame or "3d" for special 3D mode
     * @desc Assign divid, creates and draw RCanvas and RFrame if necessary, add painter to pad list of painters
     * @returns {Promise} for ready */
   let ensureRCanvas = (painter, frame_kind) => {
      if (!painter) return Promise.reject('Painter not provided in ensureRCanvas');

      // simple check - if canvas there, can use painter
      let svg_c = painter.getCanvSvg();
      // let noframe = (frame_kind === false) || (frame_kind == "3d") ? "noframe" : "";

      let promise = !svg_c.empty() ? Promise.resolve(true) : drawRCanvas(painter.getDom(), null /* , noframe */);

      return promise.then(() => {
         if (frame_kind === false) return;
         if (painter.getFrameSvg().select(".main_layer").empty())
            return drawRFrame(painter.getDom(), null, (typeof frame_kind === "string") ? frame_kind : "");
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
    * @extends JSROOT.ObjectPainter
    * @param {object|string} dom - DOM element for drawing or element id
    * @param {object} pave - object to draw
    * @param {string} [opt] - object draw options
    * @param {string} [csstype] - object css kind
    * @private
    */

   function RPavePainter(divid, pave, opt, csstype) {
      JSROOT.ObjectPainter.call(this, divid, pave, opt);
      this.csstype = csstype || "pave";
   }

   RPavePainter.prototype = Object.create(JSROOT.ObjectPainter.prototype);

   /** @summary Draw pave content
     * @desc assigned depending on pave class */
   RPavePainter.prototype.drawContent = function() {
      return Promise.resolve(this);
   }

   RPavePainter.prototype.drawPave = function() {

      let rect = this.getPadPainter().getPadRect(),
          fp = this.getFramePainter(),
          fx, fy, fw;

      if (fp) {
         let frame_rect = fp.getFrameRect();
         fx = frame_rect.x;
         fy = frame_rect.y;
         fw = frame_rect.width;
         // fh = frame_rect.height;
      } else {
         let st = JSROOT.gStyle;
         fx = Math.round(st.fPadLeftMargin * rect.width);
         fy = Math.round(st.fPadTopMargin * rect.height);
         fw = Math.round((1-st.fPadLeftMargin-st.fPadRightMargin) * rect.width);
         // fh = Math.round((1-st.fPadTopMargin-st.fPadBottomMargin) * rect.height);
      }

      let visible      = this.v7EvalAttr("visible", true),
          pave_cornerx = this.v7EvalLength("cornerX", rect.width, 0.02),
          pave_cornery = this.v7EvalLength("cornerY", rect.height, -0.02),
          pave_width   = this.v7EvalLength("width", rect.width, 0.3),
          pave_height  = this.v7EvalLength("height", rect.height, 0.3),
          line_width   = this.v7EvalAttr("border_width", 1),
          line_style   = this.v7EvalAttr("border_style", 1),
          line_color   = this.v7EvalColor("border_color", "black"),
          border_rx    = this.v7EvalAttr("border_rx", 0),
          border_ry    = this.v7EvalAttr("border_ry", 0),
          fill_color   = this.v7EvalColor("fill_color", "white"),
          fill_style   = this.v7EvalAttr("fill_style", 1);

      this.createG();

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
                 .attr("rx", border_rx || null)
                 .attr("ry", border_ry || null)
                 .style("stroke", line_color)
                 .attr("stroke-width", line_width)
                 .style("stroke-dasharray", jsrp.root_line_styles[line_style])
                 .attr("fill", fill_color);

      this.pave_width = pave_width;
      this.pave_height = pave_height;

      // here should be fill and draw of text

      return this.drawContent().then(() => {

         if (JSROOT.batch_mode) return this;

         return JSROOT.require(['interactive']).then(inter => {
            // TODO: provide pave context menu as in v6
            if (JSROOT.settings.ContextMenu && this.paveContextMenu)
               this.draw_g.on("contextmenu", evnt => this.paveContextMenu(evnt));

            inter.addDragHandler(this, { minwidth: 20, minheight: 20, redraw: this.sizeChanged.bind(this) });

            return this;
         });
      });
   }

   /** @summary Process interactive moving of the stats box */
   RPavePainter.prototype.sizeChanged = function() {
      this.pave_width = parseInt(this.draw_g.attr("width"));
      this.pave_height = parseInt(this.draw_g.attr("height"));

      let pave_x = parseInt(this.draw_g.attr("x")),
          pave_y = parseInt(this.draw_g.attr("y")),
          rect = this.getPadPainter().getPadRect(),
          fp = this.getFramePainter(),
          fx, fy, fw;

      if (fp) {
         let frame_rect = fp.getFrameRect();
         fx = frame_rect.x;
         fy = frame_rect.y;
         fw = frame_rect.width;
         // fh = frame_rect.height;
      } else {
         let st = JSROOT.gStyle;
         fx = Math.round(st.fPadLeftMargin * rect.width);
         fy = Math.round(st.fPadTopMargin * rect.height);
         fw = Math.round((1-st.fPadLeftMargin-st.fPadRightMargin) * rect.width);
         // fh = Math.round((1-st.fPadTopMargin-st.fPadBottomMargin) * rect.height);
      }

      let changes = {};
      this.v7AttrChange(changes, "cornerX", (pave_x + this.pave_width - fx - fw) / rect.width);
      this.v7AttrChange(changes, "cornerY", (pave_y - fy) / rect.height);
      this.v7AttrChange(changes, "width", this.pave_width / rect.width);
      this.v7AttrChange(changes, "height", this.pave_height / rect.height);
      this.v7SendAttrChanges(changes, false); // do not invoke canvas update on the server

      this.draw_g.select("rect")
                 .attr("width", this.pave_width)
                 .attr("height", this.pave_height);

      this.drawContent();
   }

   RPavePainter.prototype.redraw = function(/*reason*/) {
      this.drawPave();
   }

   let drawPave = (divid, pave, opt) => {
      let painter = new RPavePainter(divid, pave, opt);

      return jsrp.ensureRCanvas(painter, false).then(() => painter.drawPave());
   }

   // =======================================================================================


   /** @summary Function used for direct draw of RFrameTitle
     * @memberof JSROOT.Painter
     * @private */
   function drawRFrameTitle(reason) {
      let fp = this.getFramePainter();
      if (!fp)
         return console.log('no frame painter - no title');

      let rect         = fp.getFrameRect(),
          fx           = rect.x,
          fy           = rect.y,
          fw           = rect.width,
          // fh           = rect.height,
          ph           = this.getPadPainter().getPadHeight(),
          title        = this.getObject(),
          title_margin = this.v7EvalLength("margin", ph, 0.02),
          title_width  = fw,
          title_height = this.v7EvalLength("height", ph, 0.05),
          textFont     = this.v7EvalFont("text", { size: 24, color: "black", align: 22 });

      this.createG();

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

      if (!JSROOT.batch_mode)
         JSROOT.require(['interactive'])
               .then(inter => inter.addDragHandler(this, { minwidth: 20, minheight: 20, no_change_x: true, redraw: this.redraw.bind(this,'drag') }));
   }

   ////////////////////////////////////////////////////////////////////////////////////////////

   JSROOT.v7.extractRColor = function(rcolor) {
      return rcolor.fColor || "black";
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

      getContour: function() {
         return this.fContour && (this.fContour.length > 1) ? this.fContour : null;
      },

      deleteContour: function() {
         delete this.fContour;
      },

      calcColor: function(value, entry1, entry2) {
         let dist = entry2.fOrdinal - entry1.fOrdinal,
             r1 = entry2.fOrdinal - value,
             r2 = value - entry1.fOrdinal;

         if (!this.fInterpolate || (dist <= 0))
            return (r1 < r2) ? entry2.fColor : entry1.fColor;

         // interpolate
         let col1 = d3.rgb(JSROOT.v7.extractRColor(entry1.fColor)),
             col2 = d3.rgb(JSROOT.v7.extractRColor(entry2.fColor)),
             color = d3.rgb(Math.round((col1.r*r1 + col2.r*r2)/dist),
                            Math.round((col1.g*r1 + col2.g*r2)/dist),
                            Math.round((col1.b*r1 + col2.b*r2)/dist));

         return color.toString();
      },

      createPaletteColors: function(len) {
         let arr = [], indx = 0;

         while (arr.length < len) {
            let value = arr.length / (len-1);

            let entry = this.fColors[indx];

            if ((Math.abs(entry.fOrdinal - value)<0.0001) || (indx == this.fColors.length - 1)) {
               arr.push(JSROOT.v7.extractRColor(entry.fColor));
               continue;
            }

            let next = this.fColors[indx+1];
            if (next.fOrdinal <= value)
               indx++;
            else
               arr.push(this.calcColor(value, entry, next));
         }

         return arr;
      },

      /** @summary extract color with ordinal value between 0 and 1 */
      getColorOrdinal : function(value) {
         if (!this.fColors)
            return "black";
         if ((typeof value != "number") || (value < 0))
            value = 0;
         else if (value > 1)
            value = 1;

         // TODO: implement better way to find index

         let entry, next = this.fColors[0];
         for (let indx = 0; indx < this.fColors.length-1; ++indx) {
            entry = next;

            if (Math.abs(entry.fOrdinal - value) < 0.0001)
               return JSROOT.v7.extractRColor(entry.fColor);

            next = this.fColors[indx+1];
            if (next.fOrdinal > value)
               return this.calcColor(value, entry, next);
         }

         return JSROOT.v7.extractRColor(next.fColor);
      },

      /** @summary set full z scale range, used in zooming */
      setFullRange: function(min, max) {
          this.full_min = min;
          this.full_max = max;
      },

      createContour: function(logz, nlevels, zmin, zmax, zminpositive) {
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
            this.palette = this.createPaletteColors(nlevels);
      }

   });

   // =============================================================

   /** @summary painter for RPalette
    *
    * @class
    * @memberof JSROOT
    * @extends JSROOT.ObjectPainter
    * @param {object|string} dom - DOM element for drawing or element id
    * @param {object} palette - RPalette object
    * @private
    */

   function RPalettePainter(divid, palette) {
      JSROOT.ObjectPainter.call(this, divid, palette);
      this.csstype = "palette";
   }

   RPalettePainter.prototype = Object.create(JSROOT.ObjectPainter.prototype);

   RPalettePainter.prototype.getHistPalette = function() {
      let drawable = this.getObject();
      let pal = drawable ? drawable.fPalette : null;

      if (pal && !pal.getColor)
         JSROOT.addMethods(pal, "ROOT::Experimental::RPalette");

      return pal;
   }

   /** @summary Draw palette */
   RPalettePainter.prototype.drawPalette = function(after_resize) {

      let palette = this.getHistPalette(),
          contour = palette.getContour(),
          framep = this.getFramePainter();

      if (!contour)
         return console.log('no contour - no palette');

      // frame painter must  be there
      if (!framep)
         return console.log('no frame painter - no palette');

      let gmin         = palette.full_min,
          gmax         = palette.full_max,
          zmin         = contour[0],
          zmax         = contour[contour.length-1],
          rect         = framep.getFrameRect(),
          fx           = rect.x,
          fy           = rect.y,
          fw           = rect.width,
          fh           = rect.height,
          pw           = this.getPadPainter().getPadWidth(),
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

          palette_width = this.v7EvalLength("width", pw, 0.05);
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

      if ((gmin === undefined) || (gmax === undefined)) { gmin = zmin; gmax = zmax; }

      framep.z_handle.configureAxis("zaxis", gmin, gmax, zmin, zmax, true, [palette_height, 0], -palette_height, { reverse: false });

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
                     .property("fill1", d3.rgb(col).darker(0.5).toString());

         if (this.isTooltipAllowed())
            r.on('mouseover', function() {
               d3.select(this).transition().duration(100).style("fill", d3.select(this).property('fill1'));
            }).on('mouseout', function() {
               d3.select(this).transition().duration(100).style("fill", d3.select(this).property('fill0'));
            }).append("svg:title").text(contour[i].toFixed(2) + " - " + contour[i+1].toFixed(2));

         if (JSROOT.settings.Zooming)
            r.on("dblclick", () => framep.unzoom("z"));
      }

      framep.z_handle.max_tick_size = Math.round(palette_width*0.3);

      let promise = framep.z_handle.drawAxis(this.draw_g, "translate(" + palette_width + "," + palette_height + ")", -1);

      if (JSROOT.batch_mode) return;

      promise.then(() => JSROOT.require(['interactive'])).then(inter => {

         if (JSROOT.settings.ContextMenu)
            this.draw_g.on("contextmenu", evnt => {
               evnt.stopPropagation(); // disable main context menu
               evnt.preventDefault();  // disable browser context menu
               jsrp.createMenu(evnt, this).then(menu => {
                 menu.add("header:Palette");
                 framep.z_handle.fillAxisContextMenu(menu, "z");
                 menu.show();
               });
            });

         if (!after_resize)
            inter.addDragHandler(this, { minwidth: 20, minheight: 20, no_change_y: true, redraw: this.drawPalette.bind(this, true) });

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
               this.getFramePainter().zoom("z", Math.min(z1, z2), Math.max(z1, z2));
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
            }, 500);
         }

         let assignHandlers = () => {
            this.draw_g.selectAll(".axis_zoom, .axis_labels")
                       .on("mousedown", startRectSel)
                       .on("dblclick", () => framep.unzoom("z"));

            if (JSROOT.settings.ZoomWheel)
               this.draw_g.on("wheel", evnt => {
                  evnt.stopPropagation();
                  evnt.preventDefault();

                  let pos = d3.pointer(evnt, this.draw_g.node()),
                      coord = 1 - pos[1] / palette_height;

                  let item = framep.z_handle.analyzeWheelEvent(evnt, coord);
                  if (item.changed)
                     framep.zoom("z", item.min, item.max);
               });
         }

         framep.z_handle.setAfterDrawHandler(assignHandlers);

         assignHandlers();
      });
   }

   let drawPalette = (divid, palette /*, opt */) => {
      let painter = new RPalettePainter(divid, palette);

      return jsrp.ensureRCanvas(painter, false).then(() => {
         painter.createG(); // just create container, real drawing will be done by histogram
         return painter;
      });
   }

   function drawRFont() {
      let font      = this.getObject(),
          svg       = this.getCanvSvg(),
          defs      = svg.select('.canvas_defs'),
          clname = "custom_font_" + font.fFamily+font.fWeight+font.fStyle;

      if (defs.empty())
         defs = svg.insert("svg:defs", ":first-child").attr("class", "canvas_defs");

      let entry = defs.select("." + clname);
      if (entry.empty())
         entry = defs.append("style").attr("type", "text/css").attr("class", clname);

      entry.text(`@font-face { font-family: "${font.fFamily}"; font-weight: ${font.fWeight ? font.fWeight : "normal"}; font-style: ${font.fStyle ? font.fStyle : "normal"}; src: ${font.fSrc}; }`);

      return true;
   }


   // jsrp.addDrawFunc({ name: "ROOT::Experimental::RPadDisplayItem", icon: "img_canvas", func: drawPad, opt: "" });

   jsrp.addDrawFunc({ name: "ROOT::Experimental::RHist1Drawable", icon: "img_histo1d", prereq: "v7hist", func: "JSROOT.v7.drawHist1", opt: "" });
   jsrp.addDrawFunc({ name: "ROOT::Experimental::RHist2Drawable", icon: "img_histo2d", prereq: "v7hist", func: "JSROOT.v7.drawHist2", opt: "" });
   jsrp.addDrawFunc({ name: "ROOT::Experimental::RHist3Drawable", icon: "img_histo3d", prereq: "v7hist3d", func: "JSROOT.v7.drawHist3", opt: "" });
   jsrp.addDrawFunc({ name: "ROOT::Experimental::RHistDisplayItem", icon: "img_histo1d", prereq: "v7hist", func: "JSROOT.v7.drawHistDisplayItem", opt: "" });
   jsrp.addDrawFunc({ name: "ROOT::Experimental::RText", icon: "img_text", prereq: "v7more", func: "JSROOT.v7.drawText", opt: "", direct: "v7", csstype: "text" });
   jsrp.addDrawFunc({ name: "ROOT::Experimental::RFrameTitle", icon: "img_text", func: drawRFrameTitle, opt: "", direct: "v7", csstype: "title" });
   jsrp.addDrawFunc({ name: "ROOT::Experimental::RPaletteDrawable", icon: "img_text", func: drawPalette, opt: "" });
   jsrp.addDrawFunc({ name: "ROOT::Experimental::RDisplayHistStat", icon: "img_pavetext", prereq: "v7hist", func: "JSROOT.v7.drawHistStats", opt: "" });
   jsrp.addDrawFunc({ name: "ROOT::Experimental::RLine", icon: "img_graph", prereq: "v7more", func: "JSROOT.v7.drawLine", opt: "", direct: "v7", csstype: "line" });
   jsrp.addDrawFunc({ name: "ROOT::Experimental::RBox", icon: "img_graph", prereq: "v7more", func: "JSROOT.v7.drawBox", opt: "", direct: "v7", csstype: "box" });
   jsrp.addDrawFunc({ name: "ROOT::Experimental::RMarker", icon: "img_graph", prereq: "v7more", func: "JSROOT.v7.drawMarker", opt: "", direct: "v7", csstype: "marker" });
   jsrp.addDrawFunc({ name: "ROOT::Experimental::RPave", icon: "img_pavetext", func: drawPave, opt: "" });
   jsrp.addDrawFunc({ name: "ROOT::Experimental::RLegend", icon: "img_graph", prereq: "v7more", func: "JSROOT.v7.drawLegend", opt: "" });
   jsrp.addDrawFunc({ name: "ROOT::Experimental::RPaveText", icon: "img_pavetext", prereq: "v7more", func: "JSROOT.v7.drawPaveText", opt: "" });
   jsrp.addDrawFunc({ name: "ROOT::Experimental::RFrame", icon: "img_frame", func: drawRFrame, opt: "" });
   jsrp.addDrawFunc({ name: "ROOT::Experimental::RFont", icon: "img_text", func: drawRFont, opt: "", direct: "v7", csstype: "font" });
   jsrp.addDrawFunc({ name: "ROOT::Experimental::RAxisDrawable", icon: "img_frame", func: drawRAxis, opt: "" });

   JSROOT.v7.RAxisPainter = RAxisPainter;
   JSROOT.v7.RFramePainter = RFramePainter;
   JSROOT.v7.RPalettePainter = RPalettePainter;
   JSROOT.v7.RPadPainter = RPadPainter;
   JSROOT.v7.RCanvasPainter = RCanvasPainter;
   JSROOT.v7.RPavePainter = RPavePainter;
   JSROOT.v7.drawRAxis = drawRAxis;
   JSROOT.v7.drawRFrame = drawRFrame;
   JSROOT.v7.drawRFont = drawRFont;
   JSROOT.v7.drawPad = drawPad;
   JSROOT.v7.drawRCanvas = drawRCanvas;
   JSROOT.v7.drawPadSnapshot = drawPadSnapshot;
   JSROOT.v7.drawPave = drawPave;
   JSROOT.v7.drawRFrameTitle = drawRFrameTitle;

   jsrp.ensureRCanvas = ensureRCanvas;

   return JSROOT;

});
