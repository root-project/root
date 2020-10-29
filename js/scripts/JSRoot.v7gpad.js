/// @file JSRoot.v7gpad.js
/// JavaScript ROOT graphics for ROOT v7 classes

JSROOT.define(['d3', 'painter'], (d3, jsrp) => {

   "use strict";

   JSROOT.v7 = {}; // placeholder for v7-relevant code

   /** Evaluate attributes using fAttr storage and configured RStyle */
   JSROOT.ObjectPainter.prototype.v7EvalAttr = function(name, dflt) {
      let obj = this.GetObject();
      if (!obj) return dflt;

      if (obj.fAttr && obj.fAttr.m) {
         let value = obj.fAttr.m[name];
         if (value) return value.v; // found value direct in attributes
      }

      if (this.rstyle && this.rstyle.fBlocks) {
         let blks = this.rstyle.fBlocks;
         for (let k=0;k<blks.length;++k) {
            let block = blks[k];

            let match = (this.csstype && (block.selector == this.csstype)) ||
                        (obj.fId && (block.selector == ("#" + obj.fId))) ||
                        (obj.fCssClass && (block.selector == ("." + obj.fCssClass)));

            if (match && block.map && block.map.m) {
               let value = block.map.m[name];
               if (value) return value.v;
            }
         }
      }

      return dflt;
   }

   JSROOT.ObjectPainter.prototype.v7SetAttr = function(name, value) {
      let obj = this.GetObject();

      if (obj && obj.fAttr && obj.fAttr.m)
         obj.fAttr.m[name] = { v: value };
   }

   /** Decode pad length from string, return pixel value */
   JSROOT.ObjectPainter.prototype.v7EvalLength = function(name, sizepx, dflt) {
      if (sizepx <= 0) sizepx = 1;

      let value = this.v7EvalAttr(name);

      if (value === undefined)
         return Math.round(dflt*sizepx);

      if (typeof value == "number")
         return Math.round(value*sizepx);

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

   JSROOT.ObjectPainter.prototype.createv7AttMarker = function(prefix) {
      if (!prefix || (typeof prefix != "string")) prefix = "marker_";

      let marker_color = this.v7EvalColor(prefix + "color", "black"),
          marker_size = this.v7EvalAttr(prefix + "size", 1),
          marker_style = this.v7EvalAttr(prefix + "style", 1);

      this.createAttMarker({ color: marker_color, size: marker_size, style: marker_style });
   }

   /** @summary Create RChangeAttr, which can be applied on the server side */
   JSROOT.ObjectPainter.prototype.v7AttrChange = function(req,name,value,kind) {
      if (!this.snapid)
         return false;

      if (!req._typename) {
         req._typename = "ROOT::Experimental::RChangeAttrRequest";
         req.ids = [];
         req.names = [];
         req.values = [];
         req.update = true;
      }

      req.ids.push(this.snapid);
      req.names.push(name);
      let obj = null;

      if ((value !== null) && (value !== undefined)) {
         if (!kind) {
            if (typeof value == "string") kind = "string"; else
            if (typeof value == "number") kind = "double";
         }
         obj = { _typename: "ROOT::Experimental::RAttrMap::" };
         switch(kind) {
            case "none": obj._typename += "NoValue_t"; break;
            case "bool": obj._typename += "BoolValue_t"; obj.v = value ? true : false; break;
            case "int": obj._typename += "IntValue_t"; obj.v = parseInt(value); break;
            case "double": obj._typename += "DoubleValue_t"; obj.v = parseFloat(value); break;
            default: obj._typename += "StringValue_t"; obj.v = (typeof value == "string") ? value : JSON.stringify(value); break;
         }
      }

      req.values.push(obj);
      return true;
   }

   /** @summary Sends accumulated attribute changes to server */
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

   function RAxisPainter(embedded, cssprefix) {
      let dummy = JSROOT.Create("TAxis"); // just dummy before all attributes are implemented

      JSROOT.ObjectPainter.call(this, dummy);

      this.embedded = embedded; // indicate that painter embedded into the histo painter
      this.csstype = "frame"; // for the moment only via frame one can set axis attributes
      this.cssprefix = cssprefix;

      this.name = "yaxis";
      this.kind = "normal";
      this.func = null;
      this.order = 0; // scaling order for axis labels

      this.full_min = 0;
      this.full_max = 1;
      this.scale_min = 0;
      this.scale_max = 1;
      this.ticks = []; // list of major ticks
      this.invert_side = false;
      this.lbls_both_sides = false; // draw labels on both sides
   }

   RAxisPainter.prototype = Object.create(JSROOT.ObjectPainter.prototype);

   RAxisPainter.prototype.Cleanup = function() {

      this.ticks = [];
      this.func = null;
      delete this.format;

      JSROOT.ObjectPainter.prototype.Cleanup.call(this);
   }

   RAxisPainter.prototype.SetAxisConfig = function(name, kind, func, min, max, smin, smax) {
      this.name = name;
      this.kind = kind;
      this.func = func;

      this.full_min = min;
      this.full_max = max;
      this.scale_min = smin;
      this.scale_max = smax;
   }

   RAxisPainter.prototype.format10Exp = function(order, value) {
      let res = "";
      if (value) {
         value = Math.round(value/Math.pow(10,order));
         if ((value!=0) && (value!=1)) res = value.toString() + (JSROOT.settings.Latex ? "#times" : "x");
      }
      res += "10";
      if (JSROOT.settings.Latex > JSROOT.constants.Latex.Symbols)
         return res + "^{" + order + "}";
      const superscript_symbols = {
            '0': '\u2070', '1': '\xB9', '2': '\xB2', '3': '\xB3', '4': '\u2074', '5': '\u2075',
            '6': '\u2076', '7': '\u2077', '8': '\u2078', '9': '\u2079', '-': '\u207B'
         };
      let str = order.toString();
      for (let n = 0; n < str.length; ++n)
         res += superscript_symbols[str[n]];
      return res;
   }

   RAxisPainter.prototype.CreateFormatFuncs = function() {

      let axis = this.GetObject(),
          is_gaxis = (axis && axis._typename === 'TGaxis');

      delete this.format;// remove formatting func

      let ndiv = 508;
      if (is_gaxis) ndiv = axis.fNdiv; else
      if (axis) ndiv = Math.max(axis.fNdivisions, 4);

      this.nticks = ndiv % 100;
      this.nticks2 = (ndiv % 10000 - this.nticks) / 100;
      this.nticks3 = Math.floor(ndiv/10000);

      if (axis && !is_gaxis && (this.nticks > 7)) this.nticks = 7;

      let gr_range = Math.abs(this.func.range()[1] - this.func.range()[0]);
      if (gr_range<=0) gr_range = 100;

      if (this.kind == 'time') {
         if (this.nticks > 8) this.nticks = 8;

         let scale_range = this.scale_max - this.scale_min,
             tf1 = jsrp.getTimeFormat(axis),
             tf2 = jsrp.chooseTimeFormat(scale_range / gr_range, false);

         if ((tf1.length == 0) || (scale_range < 0.1 * (this.full_max - this.full_min)))
            tf1 = jsrp.chooseTimeFormat(scale_range / this.nticks, true);

         this.tfunc1 = this.tfunc2 = d3.timeFormat(tf1);
         if (tf2!==tf1)
            this.tfunc2 = d3.timeFormat(tf2);

         this.format = function(d, asticks) {
            return asticks ? this.tfunc1(d) : this.tfunc2(d);
         }

      } else if (this.kind == 'log') {
         if (this.nticks2 > 1) {
            this.nticks *= this.nticks2; // all log ticks (major or minor) created centrally
            this.nticks2 = 1;
         }
         this.noexp = axis ? axis.TestBit(JSROOT.EAxisBits.kNoExponent) : false;
         if ((this.scale_max < 300) && (this.scale_min > 0.3)) this.noexp = true;
         this.moreloglabels = axis ? axis.TestBit(JSROOT.EAxisBits.kMoreLogLabels) : false;

         this.format = function(d, asticks, notickexp_fmt) {
            let val = parseFloat(d), rnd = Math.round(val);
            if (!asticks)
               return ((rnd === val) && (Math.abs(rnd)<1e9)) ? rnd.toString() : JSROOT.FFormat(val, notickexp_fmt || JSROOT.gStyle.fStatFormat);

            if (val <= 0) return null;
            let vlog = Math.log10(val);
            if (this.moreloglabels || (Math.abs(vlog - Math.round(vlog))<0.001)) {
               if (!this.noexp && !notickexp_fmt)
                  return this.format10Exp(Math.floor(vlog+0.01), val);

               return (vlog<0) ? val.toFixed(Math.round(-vlog+0.5)) : val.toFixed(0);
            }
            return null;
         }
      } else if (this.kind == 'labels') {
         this.nticks = 50; // for text output allow max 50 names
         let scale_range = this.scale_max - this.scale_min;
         if (this.nticks > scale_range)
            this.nticks = Math.round(scale_range);
         this.nticks2 = 1;

         this.regular_labels = true;

         if (axis && axis.fNbins && axis.fLabels) {
            if ((axis.fNbins != Math.round(axis.fXmax - axis.fXmin)) ||
                (axis.fXmin != 0) || (axis.fXmax != axis.fNbins)) {
               this.regular_labels = false;
            }
         }

         this.axis = axis;

         this.format = function(d) {
            let indx = parseFloat(d);
            if (!this.regular_labels)
               indx = (indx - this.axis.fXmin)/(this.axis.fXmax - this.axis.fXmin) * this.axis.fNbins;
            indx = Math.round(indx);
            if ((indx<0) || (indx>=this.axis.fNbins)) return null;
            for (let i = 0; i < this.axis.fLabels.arr.length; ++i) {
               let tstr = this.axis.fLabels.arr[i];
               if (tstr.fUniqueID === indx+1) return tstr.fString;
            }
            return null;
         }
      } else {

         this.order = 0;
         this.ndig = 0;

         this.format = function(d, asticks, fmt) {
            let val = parseFloat(d);
            if (asticks && this.order) val = val / Math.pow(10, this.order);

            if (val === Math.round(val))
               return (Math.abs(val)<1e9) ? val.toFixed(0) : val.toExponential(4);

            if (asticks) return (this.ndig>10) ? val.toExponential(this.ndig-11) : val.toFixed(this.ndig);

            return JSROOT.FFormat(val, fmt || JSROOT.gStyle.fStatFormat);
         }
      }
   }

   RAxisPainter.prototype.ProduceTicks = function(ndiv, ndiv2) {
      if (!this.noticksopt) {
         let arr = this.func.ticks(ndiv * (ndiv2 || 1));
         // FIXME: workaround - prvent creation too much log ticks when min >= 1, but this should be checked differently
         if ((this.kind == "log") && (arr.length > 30) && this.scale_min > 0.8)
             arr = this.func.ticks(10);
         return arr;
      }

      if (ndiv2) ndiv = (ndiv-1) * ndiv2;
      let dom = this.func.domain(), ticks = [];
      for (let n=0;n<=ndiv;++n)
         ticks.push((dom[0]*(ndiv-n) + dom[1]*n)/ndiv);
      return ticks;
   }

   RAxisPainter.prototype.CreateTicks = function(only_major_as_array, optionNoexp, optionNoopt, optionInt) {
      // function used to create array with minor/middle/major ticks

      if (optionNoopt && this.nticks && (this.kind == "normal")) this.noticksopt = true;

      let handle = { nminor: 0, nmiddle: 0, nmajor: 0, func: this.func };

      handle.minor = handle.middle = handle.major = this.ProduceTicks(this.nticks);

      if (only_major_as_array) {
         let res = handle.major, delta = (this.scale_max - this.scale_min)*1e-5;
         if (res[0] > this.scale_min + delta) res.unshift(this.scale_min);
         if (res[res.length-1] < this.scale_max - delta) res.push(this.scale_max);
         return res;
      }

      if ((this.kind == 'labels') && !this.regular_labels) {
         handle.lbl_pos = [];
         for (let n=0;n<this.axis.fNbins;++n) {
            let x = this.axis.fXmin + n / this.axis.fNbins * (this.axis.fXmax - this.axis.fXmin);
            if ((x >= this.scale_min) && (x < this.scale_max)) handle.lbl_pos.push(x);
         }
      }

      if (this.nticks2 > 1) {
         handle.minor = handle.middle = this.ProduceTicks(handle.major.length, this.nticks2);

         let gr_range = Math.abs(this.func.range()[1] - this.func.range()[0]);

         // avoid black filling by middle-size
         if ((handle.middle.length <= handle.major.length) || (handle.middle.length > gr_range/3.5)) {
            handle.minor = handle.middle = handle.major;
         } else
         if ((this.nticks3 > 1) && (this.kind !== 'log'))  {
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

      if ((this.kind == "normal") && (handle.major.length > 0)) {

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

   RAxisPainter.prototype.IsCenterLabels = function() {
      if (this.kind === 'labels') return true;
      if (this.kind === 'log') return false;
      let axis = this.GetObject();
      return axis && axis.TestBit(JSROOT.EAxisBits.kCenterLabels);
   }

   RAxisPainter.prototype.AddTitleDrag = function(title_g, vertical, offset_k, reverse, axis_length) {
      if (!JSROOT.settings.MoveResize) return;

      let drag_rect = null,
          acc_x, acc_y, new_x, new_y, sign_0, alt_pos,
          drag_move = d3.drag().subject(Object);

      drag_move
         .on("start", evnt => {

            evnt.sourceEvent.preventDefault();
            evnt.sourceEvent.stopPropagation();

            let box = title_g.node().getBBox(), // check that elements visible, request precise value
                axis = this.GetObject();

            new_x = acc_x = title_g.property('shift_x');
            new_y = acc_y = title_g.property('shift_y');

            sign_0 = vertical ? (acc_x>0) : (acc_y>0); // sign should remain

            if (axis.TestBit(JSROOT.EAxisBits.kCenterTitle))
               alt_pos = (reverse === vertical) ? axis_length : 0;
            else
               alt_pos = Math.round(axis_length/2);

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
                   set_y = title_g.property('shift_y');

               if (vertical) {
                  set_x = acc_x;
                  if (Math.abs(acc_y - set_y) > Math.abs(acc_y - alt_pos)) set_y = alt_pos;
               } else {
                  set_y = acc_y;
                  if (Math.abs(acc_x - set_x) > Math.abs(acc_x - alt_pos)) set_x = alt_pos;
               }

               if (sign_0 === (vertical ? (set_x>0) : (set_y>0))) {
                  new_x = set_x; new_y = set_y;
                  title_g.attr('transform', 'translate(' + new_x + ',' + new_y +  ')');
               }

          }).on("end", evnt => {
               if (!drag_rect) return;

               evnt.sourceEvent.preventDefault();
               evnt.sourceEvent.stopPropagation();

               title_g.property('shift_x', new_x)
                      .property('shift_y', new_y);

               let axis = this.GetObject();

               axis.fTitleOffset = (vertical ? new_x : new_y) / offset_k;
               if ((vertical ? new_y : new_x) === alt_pos) axis.InvertBit(JSROOT.EAxisBits.kCenterTitle);

               drag_rect.remove();
               drag_rect = null;
            });

      title_g.style("cursor", "move").call(drag_move);
   }

   RAxisPainter.prototype.DrawAxis = function(vertical, layer, w, h, transform, reverse, second_shift, disable_axis_drawing, max_text_width) {
      let axis = this.GetObject(), chOpt = "",
          is_gaxis = false,
          axis_g = layer, tickSize = 0.03,
          scaling_size = 100, draw_lines = true,
          pad_w = this.pad_width() || 10,
          pad_h = this.pad_height() || 10,
          resolveFunc, totalTextCallbacks = 0, totalDone = false,
          promise = new Promise(resolve => { resolveFunc = resolve; });

      let checkTextCallBack = (is_callback) => {
          if (is_callback) totalTextCallbacks--; else totalDone = true;
          if (!totalTextCallbacks && totalDone && resolveFunc) {
            resolveFunc(true);
            resolveFunc = null;
         }
      };

      this.vertical = vertical;

      function myXor(a,b) { return ( a && !b ) || (!a && b); }

      // shift for second ticks set (if any)
      if (!second_shift) second_shift = 0; else
      if (this.invert_side) second_shift = -second_shift;

      this.createv7AttLine(this.cssprefix + "line_");

      chOpt = myXor(vertical, this.invert_side) ? "-S" : "+S";
      tickSize = axis.fTickLength;
      scaling_size = (vertical ? pad_w : pad_h);

      if (!is_gaxis || (this.name === "zaxis")) {
         axis_g = layer.select("." + this.name + "_container");
         if (axis_g.empty())
            axis_g = layer.append("svg:g").attr("class",this.name + "_container");
         else
            axis_g.selectAll("*").remove();
      } else {
         if (!disable_axis_drawing && draw_lines)
            axis_g.append("svg:line")
                  .attr("x1",0).attr("y1",0)
                  .attr("x1",vertical ? 0 : w)
                  .attr("y1", vertical ? h : 0)
                  .call(this.lineatt.func);
      }

      axis_g.attr("transform", transform || null);

      let side = 1, ticks_plusminus = 0,
          text_scaling_size = Math.min(pad_w, pad_h),
          optionPlus = (chOpt.indexOf("+")>=0),
          optionMinus = (chOpt.indexOf("-")>=0),
          optionSize = (chOpt.indexOf("S")>=0),
          // optionY = (chOpt.indexOf("Y")>=0),
          // optionUp = (chOpt.indexOf("0")>=0),
          // optionDown = (chOpt.indexOf("O")>=0),
          optionUnlab = (chOpt.indexOf("U")>=0),  // no labels
          optionNoopt = (chOpt.indexOf("N")>=0),  // no ticks position optimization
          optionInt = (chOpt.indexOf("I")>=0),    // integer labels
          optionNoexp = axis.TestBit(JSROOT.EAxisBits.kNoExponent);

      if (is_gaxis && axis.TestBit(JSROOT.EAxisBits.kTickPlus)) optionPlus = true;
      if (is_gaxis && axis.TestBit(JSROOT.EAxisBits.kTickMinus)) optionMinus = true;

      if (optionPlus && optionMinus) { side = 1; ticks_plusminus = 1; } else
      if (optionMinus) { side = myXor(reverse,vertical) ? 1 : -1; } else
      if (optionPlus) { side = myXor(reverse,vertical) ? -1 : 1; }

      tickSize = Math.round((optionSize ? tickSize : 0.03) * scaling_size);

      if (this.max_tick_size && (tickSize > this.max_tick_size)) tickSize = this.max_tick_size;

      this.CreateFormatFuncs();

      let res = "", res2 = "", lastpos = 0, lasth = 0;

      // first draw ticks

      this.ticks = [];

      let handle = this.CreateTicks(false, optionNoexp, optionNoopt, optionInt);

      while (handle.next(true)) {

         let h1 = Math.round(tickSize/4), h2 = 0;

         if (handle.kind < 3)
            h1 = Math.round(tickSize/2);

         if (handle.kind == 1) {
            // if not showing labels, not show large tick
            if (!('format' in this) || (this.format(handle.tick,true)!==null)) h1 = tickSize;
            this.ticks.push(handle.grpos); // keep graphical positions of major ticks
         }

         if (ticks_plusminus > 0) h2 = -h1; else
         if (side < 0) { h2 = -h1; h1 = 0; } else { h2 = 0; }

         if (res.length == 0) {
            res = vertical ? ("M"+h1+","+handle.grpos) : ("M"+handle.grpos+","+(-h1));
            res2 = vertical ? ("M"+(second_shift-h1)+","+handle.grpos) : ("M"+handle.grpos+","+(second_shift+h1));
         } else {
            res += vertical ? ("m"+(h1-lasth)+","+(handle.grpos-lastpos)) : ("m"+(handle.grpos-lastpos)+","+(lasth-h1));
            res2 += vertical ? ("m"+(lasth-h1)+","+(handle.grpos-lastpos)) : ("m"+(handle.grpos-lastpos)+","+(h1-lasth));
         }

         res += vertical ? ("h"+ (h2-h1)) : ("v"+ (h1-h2));
         res2 += vertical ? ("h"+ (h1-h2)) : ("v"+ (h2-h1));

         lastpos = handle.grpos;
         lasth = h2;
      }

      if ((res.length > 0) && !disable_axis_drawing && draw_lines)
         axis_g.append("svg:path").attr("d", res).call(this.lineatt.func);

      if ((second_shift!==0) && (res2.length>0) && !disable_axis_drawing  && draw_lines)
         axis_g.append("svg:path").attr("d", res2).call(this.lineatt.func);

      let labelsize = Math.round( (axis.fLabelSize < 1) ? axis.fLabelSize * text_scaling_size : axis.fLabelSize);
      if ((labelsize <= 0) || (Math.abs(axis.fLabelOffset) > 1.1)) optionUnlab = true; // disable labels when size not specified

      // draw labels (on both sides, when needed)
      if (!disable_axis_drawing && !optionUnlab) {

         let label_color = this.get_color(axis.fLabelColor),
             labeloffset = Math.round(axis.fLabelOffset*text_scaling_size /*+ 0.5*labelsize*/),
             center_lbls = this.IsCenterLabels(),
             rotate_lbls = axis.TestBit(JSROOT.EAxisBits.kLabelsVert),
             textscale = 1, maxtextlen = 0, lbls_tilt = false, labelfont = null,
             label_g = [ axis_g.append("svg:g").attr("class","axis_labels") ],
             lbl_pos = handle.lbl_pos || handle.major,
             total_draw_cnt = 0, all_done = 0;

         if (this.lbls_both_sides)
            label_g.push(axis_g.append("svg:g").attr("class","axis_labels").attr("transform", vertical ? "translate(" + w + ",0)" : "translate(0," + (-h) + ")"));

         // function called when text text is drawn to analyze width, required to correctly scale all labels
         function process_drawtext_ready(painter) {
            let textwidth = this.result_width;

            if (textwidth && ((!vertical && !rotate_lbls) || (vertical && rotate_lbls)) && (painter.kind != 'log')) {
               let maxwidth = this.gap_before*0.45 + this.gap_after*0.45;
               if (!this.gap_before) maxwidth = 0.9*this.gap_after; else
               if (!this.gap_after) maxwidth = 0.9*this.gap_before;
               textscale = Math.min(textscale, maxwidth / textwidth);
            } else if (vertical && max_text_width && this.normal_side && (max_text_width - labeloffset > 20) && (textwidth > max_text_width - labeloffset)) {
               textscale = Math.min(textscale, (max_text_width - labeloffset) / textwidth);
            }

            total_draw_cnt--;
            if ((all_done != 1) || (total_draw_cnt > 0)) return;

            all_done = 2; // mark that function is finished

            if ((textscale > 0.01) && (textscale < 0.7) && !vertical && !rotate_lbls && (maxtextlen > 5) && !painter.lbls_both_sides) {
               lbls_tilt = true;
               textscale *= 3;
            }

            for (let cnt = 0; cnt < this.lgs.length; ++cnt) {
               if ((textscale > 0.01) && (textscale < 1))
                  painter.TextScaleFactor(1/textscale, this.lgs[cnt]);
            }
         }

         for (let lcnt = 0; lcnt < label_g.length; ++lcnt) {

            if (lcnt > 0) side = -side;

            let lastpos = 0,
                fix_coord = vertical ? -labeloffset*side : (labeloffset+2)*side + ticks_plusminus*tickSize;

            labelfont = new JSROOT.FontHandler(axis.fLabelFont, labelsize);

            this.StartTextDrawing(labelfont, 'font', label_g[lcnt]);

            for (let nmajor=0;nmajor<lbl_pos.length;++nmajor) {

               let lbl = this.format(lbl_pos[nmajor], true);
               if (lbl === null) continue;

               let pos = Math.round(this.func(lbl_pos[nmajor]));

               let arg = { text: lbl, color: label_color, latex: 1, draw_g: label_g[lcnt], normal_side: (lcnt == 0), lgs: label_g };

               arg.gap_before = (nmajor>0) ? Math.abs(Math.round(pos - this.func(lbl_pos[nmajor-1]))) : 0,
               arg.gap_after = (nmajor<lbl_pos.length-1) ? Math.abs(Math.round(this.func(lbl_pos[nmajor+1])-pos)) : 0;

               if (center_lbls) {
                  let gap = arg.gap_after || arg.gap_before;
                  pos = Math.round(pos - (vertical ? 0.5*gap : -0.5*gap));
                  if ((pos < -5) || (pos > (vertical ? h : w) + 5)) continue;
               }

               maxtextlen = Math.max(maxtextlen, lbl.length);

               if (vertical) {
                  arg.x = fix_coord;
                  arg.y = pos;
                  arg.align = rotate_lbls ? ((side<0) ? 23 : 20) : ((side<0) ? 12 : 32);
               } else {
                  arg.x = pos;
                  arg.y = fix_coord;
                  arg.align = rotate_lbls ? ((side<0) ? 12 : 32) : ((side<0) ? 20 : 23);
               }

               if (rotate_lbls) arg.rotate = 270;

               arg.post_process = process_drawtext_ready;

               total_draw_cnt++;
               this.DrawText(arg);

               if (lastpos && (pos!=lastpos) && ((vertical && !rotate_lbls) || (!vertical && rotate_lbls))) {
                  let axis_step = Math.abs(pos-lastpos);
                  textscale = Math.min(textscale, 0.9*axis_step/labelsize);
               }

               lastpos = pos;
            }

            if (this.order)
               this.DrawText({ color: label_color,
                               x: vertical ? side*5 : w+5,
                               y: this.has_obstacle ? fix_coord : (vertical ? -3 : -3*side),
                               align: vertical ? ((side<0) ? 30 : 10) : ( myXor(this.has_obstacle, (side<0)) ? 13 : 10 ),
                               latex: 1,
                               text: '#times' + this.format10Exp(this.order),
                               draw_g: label_g[lcnt]
               });

         }

         totalTextCallbacks += label_g.length;
         for (let lcnt = 0; lcnt < label_g.length; ++lcnt)
            this.FinishTextDrawing(label_g[lcnt], () => {
              if (lbls_tilt)
                 label_g[lcnt].selectAll("text").each(() => {
                     let txt = d3.select(this), tr = txt.attr("transform");
                     txt.attr("transform", tr + " rotate(25)").style("text-anchor", "start");
                 });
               checkTextCallBack(true);
            });

         if (label_g.length > 1) side = -side;

         if (labelfont) labelsize = labelfont.size; // use real font size
      }

      if (JSROOT.settings.Zooming && !this.disable_zooming && !JSROOT.BatchMode) {
         let r =  axis_g.append("svg:rect")
                        .attr("class", "axis_zoom")
                        .style("opacity", "0")
                        .style("cursor", "crosshair");

         if (vertical)
            r.attr("x", (side>0) ? (-2*labelsize - 3) : 3)
             .attr("y", 0)
             .attr("width", 2*labelsize + 3)
             .attr("height", h)
         else
            r.attr("x", 0).attr("y", (side>0) ? 0 : -labelsize-3)
             .attr("width", w).attr("height", labelsize + 3);
      }

      if ((axis.fTitle.length > 0) && !disable_axis_drawing) {
         let title_g = axis_g.append("svg:g").attr("class", "axis_title"),
             title_fontsize = (axis.fTitleSize >= 1) ? axis.fTitleSize : Math.round(axis.fTitleSize * text_scaling_size),
             title_offest_k = 1.6*(axis.fTitleSize<1 ? axis.fTitleSize : axis.fTitleSize/(this.pad_height("") || 10)),
             center = axis.TestBit(JSROOT.EAxisBits.kCenterTitle),
             rotate = axis.TestBit(JSROOT.EAxisBits.kRotateTitle) ? -1 : 1,
             title_color = this.get_color(axis.fTitleColor),
             shift_x = 0, shift_y = 0;

         this.StartTextDrawing(axis.fTitleFont, title_fontsize, title_g);

         let myxor = ((rotate<0) && !reverse) || ((rotate>=0) && reverse);

         if (vertical) {
            title_offest_k *= -side*pad_w;

            shift_x = Math.round(title_offest_k*axis.fTitleOffset);

            if ((this.name == "zaxis") && is_gaxis && ('getBoundingClientRect' in axis_g.node())) {
               // special handling for color palette labels - draw them always on right side
               let rect = axis_g.node().getBoundingClientRect();
               if (shift_x < rect.width - tickSize) shift_x = Math.round(rect.width - tickSize);
            }

            shift_y = Math.round(center ? h/2 : (reverse ? h : 0));

            this.DrawText({ align: (center ? "middle" : (myxor ? "begin" : "end" )) + ";middle",
                            rotate: (rotate<0) ? 90 : 270,
                            text: axis.fTitle, color: title_color, draw_g: title_g });
         } else {
            title_offest_k *= side*pad_h;

            shift_x = Math.round(center ? w/2 : (reverse ? 0 : w));
            shift_y = Math.round(title_offest_k*axis.fTitleOffset);
            this.DrawText({ align: (center ? 'middle' : (myxor ? 'begin' : 'end')) + ";middle",
                            rotate: (rotate<0) ? 180 : 0,
                            text: axis.fTitle, color: title_color, draw_g: title_g });
         }

         let axis_rect = null;
         if (vertical && (axis.fTitleOffset == 0) && ('getBoundingClientRect' in axis_g.node()))
            axis_rect = axis_g.node().getBoundingClientRect();

         totalTextCallbacks++;
         this.FinishTextDrawing(title_g, () => {
            if (axis_rect) {
               let title_rect = title_g.node().getBoundingClientRect();
               shift_x = (side>0) ? Math.round(axis_rect.left - title_rect.right - title_fontsize*0.3) :
                                    Math.round(axis_rect.right - title_rect.left + title_fontsize*0.3);
            }

            title_g.attr('transform', 'translate(' + shift_x + ',' + shift_y +  ')')
                   .property('shift_x', shift_x)
                   .property('shift_y', shift_y);

            checkTextCallBack(true);
         });


         this.AddTitleDrag(title_g, vertical, title_offest_k, reverse, vertical ? h : w);
      }

      this.position = 0;

      if ('getBoundingClientRect' in axis_g.node()) {
         let rect1 = axis_g.node().getBoundingClientRect(),
             rect2 = this.svg_pad().node().getBoundingClientRect();

         this.position = rect1.left - rect2.left; // use to control left position of Y scale
      }

      checkTextCallBack(false);

      return promise;
   }

   RAxisPainter.prototype.Redraw = function() {

      let gaxis = this.GetObject(),
          x1 = this.AxisToSvg("x", gaxis.fX1),
          y1 = this.AxisToSvg("y", gaxis.fY1),
          x2 = this.AxisToSvg("x", gaxis.fX2),
          y2 = this.AxisToSvg("y", gaxis.fY2),
          w = x2 - x1, h = y1 - y2,
          vertical = Math.abs(w) < Math.abs(h),
          func = null, reverse = false, kind = "normal",
          min = gaxis.fWmin, max = gaxis.fWmax,
          domain_min = min, domain_max = max;

      if (gaxis.fChopt.indexOf("t")>=0) {
         func = d3.scaleTime();
         kind = "time";
         this.toffset = jsrp.getTimeOffset(gaxis);
         domain_min = new Date(this.toffset + min*1000);
         domain_max = new Date(this.toffset + max*1000);
      } else if (gaxis.fChopt.indexOf("G")>=0) {
         func = d3.scaleLog();
         kind = "log";
      } else {
         func = d3.scaleLinear();
         kind = "normal";
      }

      func.domain([domain_min, domain_max]);

      if (vertical) {
         if (h > 0) {
            func.range([h,0]);
         } else {
            let d = y1; y1 = y2; y2 = d;
            h = -h; reverse = true;
            func.range([0,h]);
         }
      } else {
         if (w > 0) {
            func.range([0,w]);
         } else {
            let d = x1; x1 = x2; x2 = d;
            w = -w; reverse = true;
            func.range([w,0]);
         }
      }

      this.SetAxisConfig(vertical ? "yaxis" : "xaxis", kind, func, min, max, min, max);

      this.CreateG();

      this.DrawAxis(vertical, this.draw_g, w, h, "translate(" + x1 + "," + y2 +")", reverse);
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
      this.x_kind = 'normal'; // 'normal', 'log', 'time', 'labels'
      this.y_kind = 'normal'; // 'normal', 'log', 'time', 'labels'
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
               grid += "M0,"+this.x_handle.ticks[n]+"h"+w;
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
               grid += "M0,"+this.y_handle.ticks[n]+"h"+w;

         if (grid.length > 0)
          layer.append("svg:path")
               .attr("class", "ygrid")
               .attr("d", grid)
               .style('stroke',grid_color).style("stroke-width",JSROOT.gStyle.fGridWidth)
               .style("stroke-dasharray", jsrp.root_line_styles[grid_style]);
      }
   }

   RFramePainter.prototype.AxisAsText = function(axis, value) {
      if (axis == "x") {
         if (this.x_kind == 'time')
            value = this.ConvertX(value);
         if (this.x_handle && ('format' in this.x_handle))
            return this.x_handle.format(value, false, JSROOT.settings.XValuesFormat);
      } else if (axis == "y") {
         if (this.y_kind == 'time')
            value = this.ConvertY(value);
         if (this.y_handle && ('format' in this.y_handle))
            return this.y_handle.format(value, false, JSROOT.settings.YValuesFormat);
      } else {
         if (this.z_handle && ('format' in this.z_handle))
            return this.z_handle.format(value, false, JSROOT.settings.ZValuesFormat);
      }

      return value.toPrecision(4);
   }


   /** @summary Set axes ranges for drawing, check configured attributes if range already specified */
   RFramePainter.prototype.SetAxesRanges = function(xmin, xmax, ymin, ymax, zmin, zmax) {
      if (this.axes_drawn) return;

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

         if ((this.zoom_xmin == this.zoom_xmax) && !this.zoom_changed_interactive) {
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

         if ((this.zoom_ymin == this.zoom_ymax) && !this.zoom_changed_interactive) {
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

         if ((this.zoom_zmin == this.zoom_zmax) && !this.zoom_changed_interactive) {
            min = this.v7EvalAttr("z_zoommin");
            max = this.v7EvalAttr("z_zoommax");

            if ((min !== undefined) || (max !== undefined)) {
               this.zoom_zmin = (min === undefined) ? this.zmin : min;
               this.zoom_zmax = (max === undefined) ? this.zmax : max;
            }
         }
      }

   }

   /** @summary axes can be drawn only for main histogram  */
   RFramePainter.prototype.DrawAxes = function() {

      if (this.axes_drawn) return Promise.resolve(true);

      if ((this.xmin==this.xmax) || (this.ymin==this.ymax)) return Promise.resolve(false);

      this.CleanupAxes();
      this.CleanXY();

      this.CreateXY();

      let layer = this.svg_frame().select(".axis_layer"),
          w = this.frame_width(), h = this.frame_height();

      this.x_handle = new RAxisPainter(true, "x_");
      this.x_handle.SetDivId(this.divid, -1);
      this.x_handle.pad_name = this.pad_name;
      this.x_handle.rstyle = this.rstyle;

      this.x_handle.SetAxisConfig("xaxis",
                                  (this.logx && (this.x_kind !== "time")) ? "log" : this.x_kind,
                                  this.x, this.xmin, this.xmax, this.scale_xmin, this.scale_xmax);
      this.x_handle.invert_side = false;
      this.x_handle.lbls_both_sides = false;
      this.x_handle.has_obstacle = false;

      this.y_handle = new RAxisPainter(true, "y_");
      this.y_handle.SetDivId(this.divid, -1);
      this.y_handle.pad_name = this.pad_name;
      this.y_handle.rstyle = this.rstyle;

      this.y_handle.SetAxisConfig("yaxis",
                                  (this.logy && this.y_kind !== "time") ? "log" : this.y_kind,
                                  this.y, this.ymin, this.ymax, this.scale_ymin, this.scale_ymax);
      this.y_handle.invert_side = false; // ((this.options.AxisPos % 10) === 1) || (pad.fTicky > 1);
      this.y_handle.lbls_both_sides = false;

      let draw_horiz = this.swap_xy ? this.y_handle : this.x_handle,
          draw_vertical = this.swap_xy ? this.x_handle : this.y_handle,
          disable_axis_draw = false, show_second_ticks = false;

      if (!disable_axis_draw) {
         let pp = this.pad_painter();
         if (pp && pp._fast_drawing) disable_axis_draw = true;
      }

      if (!disable_axis_draw) {
         let promise1 = draw_horiz.DrawAxis(false, layer, w, h,
                                            draw_horiz.invert_side ? undefined : "translate(0," + h + ")",
                                            false, show_second_ticks ? -h : 0, disable_axis_draw);

         let promise2 = draw_vertical.DrawAxis(true, layer, w, h,
                                               draw_vertical.invert_side ? "translate(" + w + ",0)" : undefined,
                                               false, show_second_ticks ? w : 0, disable_axis_draw,
                                               draw_vertical.invert_side ? 0 : this.frame_x());

         return Promise.all([promise1, promise2]).then(() => {
             this.DrawGrids();
             this.axes_drawn = true;
             return true;
         });
      }

      this.axes_drawn = true;

      return Promise.resolve(true);
   }

   RFramePainter.prototype.SizeChanged = function() {
      // function called at the end of resize of frame
      // One should apply changes to the pad

    /*  let pad = this.root_pad();

      if (pad) {
         pad.fLeftMargin = this.fX1NDC;
         pad.fRightMargin = 1 - this.fX2NDC;
         pad.fBottomMargin = this.fY1NDC;
         pad.fTopMargin = 1 - this.fY2NDC;
         this.SetRootPadRange(pad);
      }
      */

      let changes = {};
      this.v7AttrChange(changes, "margin_left", this.fX1NDC);
      this.v7AttrChange(changes, "margin_bottom", this.fY1NDC);
      this.v7AttrChange(changes, "margin_right", 1 - this.fX2NDC);
      this.v7AttrChange(changes, "margin_top", 1 - this.fY2NDC);
      this.v7SendAttrChanges(changes, false); // do not invoke canvas update on the server

      this.RedrawPad();
   }

   RFramePainter.prototype.CleanXY = function() {
      // remove all kinds of X/Y function for axes transformation
      delete this.x; delete this.grx;
      delete this.ConvertX; delete this.RevertX;
      delete this.y; delete this.gry;
      delete this.ConvertY; delete this.RevertY;
      delete this.z; delete this.grz;
   }

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
   }

   /** @summary Removes all drawn elements of the frame
     * @private */
   RFramePainter.prototype.CleanFrameDrawings = function() {
      // cleanup all 3D drawings if any
      if (typeof this.Create3DScene === 'function')
         this.Create3DScene(-1);

      this.CleanupAxes();
      this.CleanXY();

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

      this.draw_g = null;
      delete this._click_handler;
      delete this._dblclick_handler;

      JSROOT.ObjectPainter.prototype.Cleanup.call(this);
   }

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

   RFramePainter.prototype.ConfigureUserClickHandler = function(handler) {
      this._click_handler = handler && (typeof handler == 'function') ? handler : null;
   }

   RFramePainter.prototype.ConfigureUserDblclickHandler = function(handler) {
      this._dblclick_handler = handler && (typeof handler == 'function') ? handler : null;
   }

   RFramePainter.prototype.Zoom = function(xmin, xmax, ymin, ymax, zmin, zmax) {
      // function can be used for zooming into specified range
      // if both limits for each axis 0 (like xmin==xmax==0), axis will be unzoomed

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
         this.ForEachPainter(obj => {
            if (zoom_x && obj.CanZoomIn("x", xmin, xmax)) {
               this.zoom_xmin = xmin;
               this.zoom_xmax = xmax;
               changed = true; r_x = "0";
               zoom_x = false;
               this.v7AttrChange(changes, "x_zoommin", xmin);
               this.v7AttrChange(changes, "x_zoommax", xmax);
               req.values[0] = xmin; req.values[1] = xmax;
               req.flags[0] = req.flags[1] = true;
            }
            if (zoom_y && obj.CanZoomIn("y", ymin, ymax)) {
               this.zoom_ymin = ymin;
               this.zoom_ymax = ymax;
               changed = true; r_y = "1";
               zoom_y = false;
               this.v7AttrChange(changes, "y_zoommin", ymin);
               this.v7AttrChange(changes, "y_zoommax", ymax);
               req.values[2] = ymin; req.values[3] = ymax;
               req.flags[2] = req.flags[3] = true;
            }
            if (zoom_z && obj.CanZoomIn("z", zmin, zmax)) {
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

   RFramePainter.prototype.Unzoom = function(dox, doy, doz) {
      if (typeof dox === 'undefined') { dox = true; doy = true; doz = true; } else
      if (typeof dox === 'string') { doz = dox.indexOf("z")>=0; doy = dox.indexOf("y")>=0; dox = dox.indexOf("x")>=0; }

      let last = this.zoom_changed_interactive;

      if (dox || doy || doz) this.zoom_changed_interactive = 2;

      let changed = this.Zoom(dox ? 0 : undefined, dox ? 0 : undefined,
                              doy ? 0 : undefined, doy ? 0 : undefined,
                              doz ? 0 : undefined, doz ? 0 : undefined);

      // if unzooming has no effect, decrease counter
      if ((dox || doy || doz) && !changed)
         this.zoom_changed_interactive = (!isNaN(last) && (last>0)) ? last - 1 : 0;

      return changed;
   }

   /** @summary Fill menu for frame when server is not there */
   RFramePainter.prototype.FillObjectOfflineMenu = function(menu, kind) {
      if ((kind!="x") && (kind!="y")) return;

      menu.add("Unzoom", this.Unzoom.bind(this, kind));

      if (this[kind+"_kind"] == "normal")
         menu.addchk(this["log"+kind], "SetLog"+kind, this.ToggleLog.bind(this, kind));

      // here should be all axes attributes in offline
   }

   RFramePainter.prototype.FillContextMenu = function(menu, kind /*, obj*/) {

      // when fill and show context menu, remove all zooming

      if ((kind=="x") || (kind=="y")) {
         menu.add("header: " + kind.toUpperCase() + " axis");
         return true;
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

      menu.addchk(this.logx, "SetLogx", this.ToggleLog.bind(this,"x"));
      menu.addchk(this.logy, "SetLogy", this.ToggleLog.bind(this,"y"));
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

      let axis_value = (axis_name=="x") ? this.RevertX(m[id]) : this.RevertY(m[id]);

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
         return Promise.resolve(false);

      return JSROOT.require(['interactive']).then(inter => {
         inter.FrameInteractive.assign(this);
         return this.AddInteractive();
      });
   }

   /** @summary Create x,y objects which maps user coordinates into pixels
     * @desc While only first painter really need such object, all others just reuse it
     * following functions are introduced
     *     this.GetBin[X/Y]  return bin coordinate
     *     this.Convert[X/Y]  converts root value in JS date when date scale is used
     *     this.[x,y]  these are d3.scale objects
     *    this.gr[x,y]  converts root scale into graphical value
     *    this.Revert[X/Y]  converts graphical coordinates to user coordinates
     * @private */
   RFramePainter.prototype.CreateXY = function() {

      this.swap_xy = false;
      this.reverse_x = false;
      this.reverse_y = false;

      // if (this.options.BarStyle>=20) this.swap_xy = true;
      this.logx = this.logy = this.logz = false;

      this.logx = !!this.v7EvalAttr("x_log");
      this.logy = !!this.v7EvalAttr("y_log");
      this.logz = !!this.v7EvalAttr("z_log");

      let w = this.frame_width(), h = this.frame_height();

      this.scale_xmin = this.xmin;
      this.scale_xmax = this.xmax;

      this.scale_ymin = this.ymin;
      this.scale_ymax = this.ymax;

      // if (opts.extra_y_space) {
      //    let log_scale = this.swap_xy ? pad.fLogx : pad.fLogy;
      //    if (log_scale && (this.scale_ymax > 0))
      //       this.scale_ymax = Math.exp(Math.log(this.scale_ymax)*1.1);
      //    else
      //       this.scale_ymax += (this.scale_ymax - this.scale_ymin) * 0.1;
      // }

      this.RecalculateRange(0);

      if (this._xaxis_timedisplay) {
         this.x_kind = 'time';
         this.timeoffsetx = jsrp.getTimeOffset(/*this.histo.fXaxis*/);
         this.ConvertX = function(x) { return new Date(this.timeoffsetx + x*1000); };
         this.RevertX = function(grx) { return (this.x.invert(grx) - this.timeoffsetx) / 1000; };
      } else {
         this.x_kind = 'normal'; // (this.histo.fXaxis.fLabels==null) ? 'normal' : 'labels';
         this.ConvertX = function(x) { return x; };
         this.RevertX = function(grx) { return this.x.invert(grx); };
      }

      if (this.zoom_xmin != this.zoom_xmax) {
         this.scale_xmin = this.zoom_xmin;
         this.scale_xmax = this.zoom_xmax;
      }

      if (this.x_kind == 'time') {
         this.x = d3.scaleTime();
      } else if (this.logx) {
         if (this.scale_xmax <= 0) this.scale_xmax = 1;
         if ((this.scale_xmin <= 0) || (this.scale_xmin >= this.scale_xmax))
            this.scale_xmin = this.scale_xmax * 0.0001;

         this.x = d3.scaleLog();
      } else {
         this.x = d3.scaleLinear();
      }

      let gr_range_x = this.reverse_x ? [ w, 0 ] : [ 0, w ],
          gr_range_y = this.reverse_y ? [ 0, h ] : [ h, 0 ];

      this.x.domain([this.ConvertX(this.scale_xmin), this.ConvertX(this.scale_xmax)])
            .range(this.swap_xy ? gr_range_y : gr_range_x);

      if (this.x_kind == 'time') {
         // we emulate scale functionality
         this.grx = function(val) { return this.x(this.ConvertX(val)); }
      } else if (this.logx) {
         this.grx = function(val) { return (val < this.scale_xmin) ? (this.swap_xy ? this.x.range()[0]+5 : -5) : this.x(val); }
      } else {
         this.grx = this.x;
      }

      if (this.zoom_ymin != this.zoom_ymax) {
         this.scale_ymin = this.zoom_ymin;
         this.scale_ymax = this.zoom_ymax;
      }

      if (this._yaxis_timedisplay) {
         this.y_kind = 'time';
         this.timeoffsety = jsrp.getTimeOffset(/*this.histo.fYaxis*/);
         this.ConvertY = function(y) { return new Date(this.timeoffsety + y*1000); };
         this.RevertY = function(gry) { return (this.y.invert(gry) - this.timeoffsety) / 1000; };
      } else {
         this.y_kind = 'normal'; // !this.histo.fYaxis.fLabels ? 'normal' : 'labels';
         this.ConvertY = function(y) { return y; };
         this.RevertY = function(gry) { return this.y.invert(gry); };
      }

      if (this.logy) {
         if (this.scale_ymax <= 0) this.scale_ymax = 1;
         if ((this.scale_ymin <= 0) || (this.scale_ymin >= this.scale_ymax))
            this.scale_ymin = 3e-4 * this.scale_ymax;

         this.y = d3.scaleLog();
      } else if (this.y_kind == 'time') {
         this.y = d3.scaleTime();
      } else {
         this.y = d3.scaleLinear()
      }

      this.y.domain([ this.ConvertY(this.scale_ymin), this.ConvertY(this.scale_ymax) ])
            .range(this.swap_xy ? gr_range_x : gr_range_y);

      if (this.y_kind=='time') {
         // we emulate scale functionality
         this.gry = function(val) { return this.y(this.ConvertY(val)); }
      } else if (this.logy) {
         // make protection for log
         this.gry = function(val) { return (val < this.scale_ymin) ? (this.swap_xy ? -5 : this.y.range()[0]+5) : this.y(val); }
      } else {
         this.gry = this.y;
      }

      // this.SetRootPadRange();
   }

   /** Set selected range back to TPad object */
   RFramePainter.prototype.SetRootPadRange = function(/* pad, is3d */) {
      // TODO: change of pad range and send back to root application
/*
      if (!pad || this.options.Same) return;

      if (is3d) {
         // this is fake values, algorithm should be copied from TView3D class of ROOT
         pad.fLogx = pad.fLogy = 0;
         pad.fUxmin = pad.fUymin = -0.9;
         pad.fUxmax = pad.fUymax = 0.9;
      } else {
         pad.fLogx = (this.swap_xy ? this.logy : this.logx) ? 1 : 0;
         pad.fUxmin = this.scale_xmin;
         pad.fUxmax = this.scale_xmax;
         pad.fLogy = (this.swap_xy ? this.logx : this.logy) ? 1 : 0;
         pad.fUymin = this.scale_ymin;
         pad.fUymax = this.scale_ymax;
      }

      if (pad.fLogx) {
         pad.fUxmin = Math.log10(pad.fUxmin);
         pad.fUxmax = Math.log10(pad.fUxmax);
      }
      if (pad.fLogy) {
         pad.fUymin = Math.log10(pad.fUymin);
         pad.fUymax = Math.log10(pad.fUymax);
      }

      let rx = pad.fUxmax - pad.fUxmin,
          mx = 1 - pad.fLeftMargin - pad.fRightMargin,
          ry = pad.fUymax - pad.fUymin,
          my = 1 - pad.fBottomMargin - pad.fTopMargin;

      if (mx <= 0) mx = 0.01; // to prevent overflow
      if (my <= 0) my = 0.01;

      pad.fX1 = pad.fUxmin - rx/mx*pad.fLeftMargin;
      pad.fX2 = pad.fUxmax + rx/mx*pad.fRightMargin;
      pad.fY1 = pad.fUymin - ry/my*pad.fBottomMargin;
      pad.fY2 = pad.fUymax + ry/my*pad.fTopMargin;
     */
   }

   RFramePainter.prototype.ToggleLog = function(axis) {
      let curr  = this["log" + axis];

      // do not allow log scale for labels
      if (!curr) {
         let kind = this[axis+"_kind"];
         if (this.swap_xy && axis==="x") kind = this["y_kind"]; else
         if (this.swap_xy && axis==="y") kind = this["x_kind"];
         if (kind === "labels") return;
      }

      if (this.v7CommMode() == JSROOT.v7.CommMode.kOffline) {
         this.v7SetAttr(axis + "_log", !curr);
         this.RedrawPad();
      } else {
         // should we use here attributes change?
         this.WebCanvasExec("Attr" + axis.toUpperCase() + "().SetLog" + (curr ? "(false)" : "(true)"));
      }
   }

   function drawFrame(divid, obj, opt) {
      let p = new RFramePainter(obj);
      if (opt == "3d") p.mode3d = true;
      p.SetDivId(divid, 2);
      p.Redraw();
      return p.DrawingReady();
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

   RPadPainter.prototype.Cleanup = function() {
      // cleanup only pad itself, all child elements will be collected and cleanup separately

      for (let k=0;k<this.painters.length;++k)
         this.painters[k].Cleanup();

      let svg_p = this.svg_pad(this.this_pad_name);
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
      this.this_pad_name = "";
      this.has_canvas = false;

      jsrp.SelectActivePad({ pp: this, active: false });

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

   /// call function for each painter
   /// kind == "all" for all objects (default)
   /// kind == "pads" only pads and subpads
   /// kind == "objects" only for object in current pad
   RPadPainter.prototype.ForEachPainterInPad = function(userfunc, kind) {
      if (!kind) kind = "all";
      if (kind!="objects") userfunc(this);
      for (let k = 0; k < this.painters.length; ++k) {
         let sub = this.painters[k];
         if (typeof sub.ForEachPainterInPad === 'function') {
            if (kind!="objects") sub.ForEachPainterInPad(userfunc, kind);
         } else if (kind != "pads") userfunc(sub);
      }
   }

   RPadPainter.prototype.RegisterForPadEvents = function(receiver) {
      this.pad_events_receiver = receiver;
   }

   RPadPainter.prototype.SelectObjectPainter = function(_painter, pos) {
      // dummy function, redefined in the RCanvasPainter

      let istoppad = (this.iscan || !this.has_canvas),
          canp = istoppad ? this : this.canv_painter(),
          pp = _painter instanceof RPadPainter ? _painter : _painter.pad_painter();

      if (pos && !istoppad)
          this.CalcAbsolutePosition(this.svg_pad(this.this_pad_name), pos);

      jsrp.SelectActivePad({ pp: pp, active: true });

      if (typeof canp.SelectActivePad == "function")
          canp.SelectActivePad(pp, _painter, pos);

      if (canp.pad_events_receiver)
         canp.pad_events_receiver({ what: "select", padpainter: pp, painter: _painter, position: pos });
   }

   /** @summary Called by framework when pad is supposed to be active and get focus
    * @private */
   RPadPainter.prototype.SetActive = function(on) {
      let fp = this.frame_painter();
      if (fp && (typeof fp.SetActive == 'function')) fp.SetActive(on);
   }

   RPadPainter.prototype.CreateCanvasSvg = function(check_resize, new_size) {

      let factor = null, svg = null, lmt = 5, rect = null, btns;

      if (check_resize > 0) {

         if (this._fixed_size) return (check_resize > 1); // flag used to force re-drawing of all subpads

         svg = this.svg_canvas();

         if (svg.empty()) return false;

         factor = svg.property('height_factor');

         rect = this.check_main_resize(check_resize, null, factor);

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
               rect = this.get_visible_rect(render_to);
         } else {
            rect = this.check_main_resize(2, new_size, factor);
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

   RPadPainter.prototype.EnlargePad = function(evnt) {

      if (evnt) {
         evnt.preventDefault();
         evnt.stopPropagation();
      }

      let svg_can = this.svg_canvas(),
          pad_enlarged = svg_can.property("pad_enlarged");

      if (this.iscan || !this.has_canvas || (!pad_enlarged && !this.HasObjectsToDraw() && !this.painters)) {
         if (this._fixed_size) return; // canvas cannot be enlarged in such mode
         if (!this.enlarge_main('toggle')) return;
         if (this.enlarge_main('state')=='off') svg_can.property("pad_enlarged", null);
      } else if (!pad_enlarged) {
         this.enlarge_main(true, true);
         svg_can.property("pad_enlarged", this.pad);
      } else if (pad_enlarged === this.pad) {
         this.enlarge_main(false);
         svg_can.property("pad_enlarged", null);
      } else {
         console.error('missmatch with pad double click events');
      }

      let was_fast = this._fast_drawing;

      this.CheckResize({ force: true });

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
            this.ForEachPainterInPad(pp => { if (pp.GetObject() == pad_enlarged) pad_visible = true; }, "pads");

         if (pad_visible) { w = width; h = height; x = y = 0; }
      }

      if (only_resize) {
         svg_pad = this.svg_pad(this.this_pad_name);
         svg_rect = svg_pad.select(".root_pad_border");
         if (!JSROOT.BatchMode)
            btns = this.svg_layer("btns_layer", this.this_pad_name);
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

      if (svg_pad.property('can3d') === 1)
         // special case of 3D canvas overlay
          this.select_main()
              .select(".draw3d_" + this.this_pad_name)
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

   RPadPainter.prototype.HasObjectsToDraw = function() {
      // return true if any objects beside sub-pads exists in the pad

      let arr = this.pad ? this.pad.fPrimitives : null;

      if (arr)
         for (let n=0;n<arr.length;++n)
            if (arr[n] && arr[n]._typename != "ROOT::Experimental::RPadDisplayItem") return true;

      return false;
   }

   RPadPainter.prototype.DrawPrimitives = function(indx, callback, ppainter) {

      if (indx===0) {
         // flag used to prevent immediate pad redraw during normal drawing sequence
         this._doing_pad_draw = true;

         if (this.iscan)
            this._start_tm = new Date().getTime();

         // set number of primitves
         this._num_primitives = this.pad && this.pad.fPrimitives ? this.pad.fPrimitives.length : 0;
      }

      if (ppainter && (typeof ppainter == 'object')) ppainter._primitive = true; // mark painter as belonging to primitives

      if (!this.pad || (indx >= this._num_primitives)) {
         delete this._doing_pad_draw;

         if (this._start_tm) {
            let spenttm = new Date().getTime() - this._start_tm;
            if (spenttm > 3000) console.log("Canvas drawing took " + (spenttm*1e-3).toFixed(2) + "s");
            delete this._start_tm;
            delete this._lasttm_tm;
         }

         return JSROOT.CallBack(callback);
      }

      // handle used to invoke callback only when necessary
      let handle_func = this.DrawPrimitives.bind(this, indx+1, callback);

      JSROOT.draw(this.divid, this.pad.fPrimitives[indx], "").then(handle_func);
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
            let main = this.svg_pad(this.this_pad_name).property('mainpainter');
            if (main && (typeof main.DrawGrids == 'function')) main.DrawGrids();
         }

         function SetTickField(arg) {
            this.pad[arg.substr(1)] = parseInt(arg[0]);

            let main = this.svg_pad(this.this_pad_name).property('mainpainter');
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

      if (this.enlarge_main() || (this.has_canvas && this.HasObjectsToDraw()))
         menu.addchk((this.enlarge_main('state')=='on'), "Enlarge " + (this.iscan ? "canvas" : "pad"), this.EnlargePad.bind(this, null));

      let fname = this.this_pad_name;
      if (fname.length===0) fname = this.iscan ? "canvas" : "pad";
      menu.add("Save as "+fname+".png", fname+".png", this.SaveAs.bind(this, "png", false));
      menu.add("Save as "+fname+".svg", fname+".svg", this.SaveAs.bind(this, "svg", false));

      return true;
   }

   RPadPainter.prototype.PadContextMenu = function(evnt) {
      if (evnt.stopPropagation) { // this is normal event processing and not emulated jsroot event
         // for debug purposes keep original context menu for small region in top-left corner
         let pos = d3.pointer(evnt, this.svg_pad(this.this_pad_name).node());

         if (pos && (pos.length==2) && (pos[0]>0) && (pos[0]<10) && (pos[1]>0) && pos[1]<10) return;

         evnt.stopPropagation(); // disable main context menu
         evnt.preventDefault();  // disable browser context menu

         let fp = this.frame_painter();
         if (fp) fp.SetLastEventPos();
      }

      JSROOT.require('interactive')
            .then(() => jsrp.createMenu(this, evnt))
            .then(menu =>
         {
            this.FillContextMenu(menu);
            this.FillObjectExecMenu(menu, "", () => menu.show());
         });
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
   }

   RPadPainter.prototype.NumDrawnSubpads = function() {
      if (this.painters === undefined) return 0;

      let num = 0;

      for (let i = 0; i < this.painters.length; ++i) {
         let obj = this.painters[i].GetObject();
         if (obj && (obj._typename === "TPad")) num++;
      }

      return num;
   }

   RPadPainter.prototype.RedrawByResize = function() {
      if (this.access_3d_kind() === JSROOT.constants.Embed3D.Overlay) return true;

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

   /** @symmary function called when drawing next snapshot from the list
     * @desc it is also used as callback for drawing of previous snap
     * @private */
   RPadPainter.prototype.DrawNextSnap = function(lst, indx, call_back, objpainter) {

      if (indx===-1) {
         // flag used to prevent immediate pad redraw during first draw
         this._doing_pad_draw = true;
         this._snaps_map = {}; // to control how much snaps are drawn
         this._num_primitives = lst ? lst.length : 0;
      }

      // workaround to insert v6 frame in list of primitives
      if (objpainter === "workaround") { --indx; objpainter = null; }

      while (true) {

         if (objpainter && lst && lst[indx] && (objpainter.snapid === undefined)) {
            // keep snap id in painter, will be used for the
            if (this.painters.indexOf(objpainter)<0) this.painters.push(objpainter);
            objpainter.AssignSnapId(lst[indx].fObjectID);
            if (!objpainter.rstyle) objpainter.rstyle = lst[indx].fStyle || this.rstyle;
         }

         delete this.next_rstyle;

         objpainter = null;

         ++indx; // change to the next snap

         if (!lst || indx >= lst.length) {
            delete this._doing_pad_draw;
            delete this._snaps_map;
            return JSROOT.CallBack(call_back, this);
         }

         let snap = lst[indx],
             snapid = snap.fObjectID,
             cnt = this._snaps_map[snapid];

         if (cnt) cnt++; else cnt=1;
         this._snaps_map[snapid] = cnt; // check how many objects with same snapid drawn, use them again

         // empty object, no need to do something, take next
         if (snap.fDummy) continue;

         // first appropriate painter for the object
         // if same object drawn twice, two painters will exists
         for (let k=0; k<this.painters.length; ++k) {
            if (this.painters[k].snapid === snapid)
               if (--cnt === 0) { objpainter = this.painters[k]; break;  }
         }

         // function which should be called when drawing of next item finished
         let draw_callback = this.DrawNextSnap.bind(this, lst, indx, call_back);

         if (objpainter) {

            if (snap._typename == "ROOT::Experimental::RPadDisplayItem")  // subpad
               return objpainter.RedrawPadSnap(snap, draw_callback);

            if (objpainter.UpdateObject(snap.fDrawable || snap.fObject || snap, snap.fOption || ""))
               objpainter.Redraw();

            continue; // call next
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

            padpainter.DrawNextSnap(snap.fPrimitives, -1, () => {
               padpainter.CurrentPadName(prev_name);
               draw_callback(padpainter);
            });
            return;
         }

         // will be used in SetDivId to assign style to painter
         this.next_rstyle = lst[indx].fStyle || this.rstyle;

         if (snap._typename === "ROOT::Experimental::TObjectDisplayItem") {

            // identifier used in RObjectDrawable
            let webSnapIds = { kNone: 0,  kObject: 1, kColors: 4, kStyle: 5, kPalette: 6 };

            if (snap.fKind == webSnapIds.kStyle) {
               JSROOT.extend(JSROOT.gStyle, snap.fObject);
               continue;
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
               continue;
            }

            if (snap.fKind == webSnapIds.kPalette) {
               let arr = snap.fObject.arr, palette = [];
               for (let n = 0; n < arr.length; ++n)
                  palette[n] =  arr[n].fString;
               this.custom_palette = new JSROOT.ColorPalette(palette);
               continue;
            }

            if (!this.frame_painter())
               return JSROOT.draw(this.divid, { _typename: "TFrame", $dummy: true }, "")
                            .then(() => draw_callback("workaround")); // call function with "workaround" as argument
         }

         // TODO - fDrawable is v7, fObject from v6, maybe use same data member?
         JSROOT.draw(this.divid, snap.fDrawable || snap.fObject || snap, snap.fOption || "").then(draw_callback);

         return; // should be handled by Promise
      }
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

   RPadPainter.prototype.RedrawPadSnap = function(snap, call_back) {
      // for the pad/canvas display item contains list of primitives plus pad attributes

      if (!snap || !snap.fPrimitives) return;

      // for the moment only window size attributes are provided
      // let padattr = { fCw: snap.fWinSize[0], fCh: snap.fWinSize[1], fTitle: snap.fTitle };

      // if canvas size not specified in batch mode, temporary use 900x700 size
      // if (this.batch_mode && this.iscan && (!padattr.fCw || !padattr.fCh)) { padattr.fCw = 900; padattr.fCh = 700; }

      if (this.iscan && snap.fTitle && document)
         document.title = snap.fTitle;

      if (this.iscan && snap.fTitle && document)
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

         this.DrawNextSnap(snap.fPrimitives, -1, call_back);

         return;
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

      this.DrawNextSnap(snap.fPrimitives, -1, () => {
         this.CurrentPadName(prev_name);
         call_back(this);
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
       let rrr = this.svg_pad(this.this_pad_name).node().getBoundingClientRect();
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

       if (!selp || (typeof selp.FillContextMenu !== 'function')) return;

       jsrp.createMenu(selp, evnt).then(menu => {
          if (selp.FillContextMenu(menu,selkind))
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

      let elem = use_frame ? this.svg_frame() : (full_canvas ? this.svg_canvas() : this.svg_pad(this.this_pad_name));

      if (elem.empty()) return Promise.resolve("");

      let painter = (full_canvas && !use_frame) ? this.canv_painter() : this;

      let items = []; // keep list of replaced elements, which should be moved back at the end

      if (!use_frame) // do not make transformations for the frame
      painter.ForEachPainterInPad(pp => {

         let item = { prnt: pp.svg_pad(pp.this_pad_name) };
         items.push(item);

         // remove buttons from each subpad
         let btns = pp.svg_layer("btns_layer", pp.this_pad_name);
         item.btns_node = btns.node();
         if (item.btns_node) {
            item.btns_prnt = item.btns_node.parentNode;
            item.btns_next = item.btns_node.nextSibling;
            btns.remove();
         }

         let main = pp.frame_painter();
         if (!main || (typeof main.Render3D !== 'function')) return;

         let can3d = pp.access_3d_kind();

         if ((can3d !== JSROOT.constants.Embed3D.Overlay) && (can3d !== JSROOT.constants.Embed3D.Embed)) return;

         let sz2 = pp.size_for_3d(JSROOT.constants.Embed3D.Embed); // get size and position of DOM element as it will be embed

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

      if (jsrp.ProcessSVGWorkarounds)
         svg = jsrp.ProcessSVGWorkarounds(svg);

      svg = jsrp.CompressSVG(svg);

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

         if (jsrp.closeMenu()) return;

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

      if (add_enlarge || this.enlarge_main('verify'))
         this.AddButton("circle", "Enlarge canvas", "EnlargePad");
   }


   RPadPainter.prototype.ShowButtons = function() {
      if (!this._buttons) return;

      JSROOT.require(['interactive']).then(inter => {
         inter.PadButtonsHandler.assign(this);
         this.ShowButtons();
      });
   }

   RPadPainter.prototype.GetCoordinate = function(pos) {
      let res = { x: 0, y: 0 };

      if (!pos) return res;

      function GetV(len, indx, dflt) {
         return (len.fArr && (len.fArr.length>indx)) ? len.fArr[indx] : dflt;
      }

      let w = this.pad_width(this.this_pad_name),
          h = this.pad_height(this.this_pad_name),
          h_norm = GetV(pos.fHoriz, 0, 0),
          h_pixel = GetV(pos.fHoriz, 1, 0),
          h_user = GetV(pos.fHoriz, 2),
          v_norm = GetV(pos.fVert, 0, 0),
          v_pixel = GetV(pos.fVert, 1, 0),
          v_user = GetV(pos.fVert, 2);

      if (!this.pad_frame || (h_user === undefined)) {
         res.x = h_norm * w + h_pixel;
      } else {
         // TO DO - user coordiantes
      }

      if (!this.pad_frame || (v_user === undefined)) {
         res.y = h - v_norm * h - v_pixel;
      } else {
         //  TO DO - user coordiantes
      }

      return res;
   }


//   RPadPainter.prototype.DrawingReady = function(res_painter) {
//      let main = this.main_painter();
//      if (main && main.mode3d && typeof main.Render3D == 'function') main.Render3D(-2222);
//      BasePainter.prototype.DrawingReady.call(this, res_painter);
//   }

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

      jsrp.SelectActivePad({ pp: painter, active: false });

      // flag used to prevent immediate pad redraw during first draw
      painter.DrawPrimitives(0, () => {
         painter.ShowButtons();
         // we restore previous pad name
         painter.CurrentPadName(prev_name);
         painter.DrawingReady();
      });

      return painter;
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
      let current = this.get_layout_kind();
      if (current == layout_kind)
         return Promise.resolve(true);

      let origin = this.select_main('origin'),
          sidebar = origin.select('.side_panel'),
          main = this.select_main(), lst = [];

      while (main.node().firstChild)
         lst.push(main.node().removeChild(main.node().firstChild));

      if (!sidebar.empty()) JSROOT.cleanup(sidebar.node());

      this.set_layout_kind("simple"); // restore defaults
      origin.html(""); // cleanup origin

      if (layout_kind == 'simple') {
         main = origin;
         for (let k=0;k<lst.length;++k)
            main.node().appendChild(lst[k]);
         this.set_layout_kind(layout_kind);
         JSROOT.resize(main.node());
         return Promise.resolve(true);
      }

      return JSROOT.require("jq2d").then(() => {

         let grid = new JSROOT.GridDisplay(origin.node(), layout_kind);

         if (layout_kind.indexOf("vert")==0) {
            main = d3.select(grid.GetFrame(0));
            sidebar = d3.select(grid.GetFrame(1));
         } else {
            main = d3.select(grid.GetFrame(1));
            sidebar = d3.select(grid.GetFrame(0));
         }

         main.classed("central_panel", true).style('position','relative');
         sidebar.classed("side_panel", true).style('position','relative');

         // now append all childs to the new main
         for (let k=0;k<lst.length;++k)
            main.node().appendChild(lst[k]);

         this.set_layout_kind(layout_kind, ".central_panel");

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

   RCanvasPainter.prototype.DrawProjection = function(kind,hist) {
      if (!this.proj_painter) return; // ignore drawing if projection not configured

      if (this.proj_painter === 1) {

         let canv = JSROOT.Create("TCanvas"), pad = this.root_pad(), main = this.main_painter(), drawopt;

         if (kind == "X") {
            canv.fLeftMargin = pad.fLeftMargin;
            canv.fRightMargin = pad.fRightMargin;
            canv.fLogx = main.logx ? 1 : 0;
            canv.fUxmin = main.logx ? Math.log10(main.scale_xmin) : main.scale_xmin;
            canv.fUxmax = main.logx ? Math.log10(main.scale_xmax) : main.scale_xmax;
            drawopt = "fixframe";
         } else {
            canv.fBottomMargin = pad.fBottomMargin;
            canv.fTopMargin = pad.fTopMargin;
            canv.fLogx = main.logy ? 1 : 0;
            canv.fUxmin = main.logy ? Math.log10(main.scale_ymin) : main.scale_ymin;
            canv.fUxmax = main.logy ? Math.log10(main.scale_ymax) : main.scale_ymax;
            drawopt = "rotate";
         }

         canv.fPrimitives.Add(hist, "hist");

         if (this.DrawInUI5ProjectionArea) {
            // copy frame attributes
            this.DrawInUI5ProjectionArea(canv, drawopt, painter => { this.proj_painter = painter; })
         } else {
            this.DrawInSidePanel(canv, drawopt, painter => { this.proj_painter = painter; })
         }
      } else {
         let hp = this.proj_painter.main_painter();
         if (hp) hp.UpdateObject(hist, "hist");
         this.proj_painter.RedrawPad();
      }
   }

   RCanvasPainter.prototype.DrawInSidePanel = function(canv, opt, call_back) {
      let side = this.select_main('origin').select(".side_panel");
      if (side.empty()) return JSROOT.CallBack(call_back, null);
      JSROOT.draw(side.node(), canv, opt).then(call_back);
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
         this.RedrawPadSnap(snap, function() {
            handle.Send("SNAPDONE:" + snapid); // send ready message back when drawing completed
         });
      } else if (msg.substr(0,4)=='JSON') {
         let obj = JSROOT.parse(msg.substr(4));
         // console.log("get JSON ", msg.length-4, obj._typename);
         this.RedrawObject(obj);
      } else if (msg.substr(0,9)=="REPL_REQ:") {
         this.ProcessDrawableReply(msg.substr(9));
      } else if (msg.substr(0,5)=='MENU:') {
         // this is container with object id and list of menu items
         let lst = JSROOT.parse(msg.substr(5));
         // console.log("get MENUS ", typeof lst, 'nitems', lst.length, msg.length-4);
         if (typeof this._getmenu_callback == 'function')
            this._getmenu_callback(lst);
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

   RCanvasPainter.prototype.SubmitMenuRequest = function(painter, menukind, reqid, call_back) {
      this.SubmitDrawableRequest("", {
         _typename: "ROOT::Experimental::RDrawableMenuRequest",
         menukind: menukind || "",
         menureqid: reqid, // used to identify menu request
      }, painter, call_back);
   }

   /** @summary Submit executable command for given painter */
   RCanvasPainter.prototype.SubmitExec = function(painter, exec, subelem) {
      console.log('SubmitExec', exec, painter.snapid, subelem);

      // snapid is intentionally ignored - only painter.snapid has to be used
      if (!this._websocket) return;

      if (subelem) {
         if ((subelem == "xaxis") || (subelem == "yaxis") || (subelem == "zaxis"))
            exec = subelem + "#" + exec;
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

   RCanvasPainter.prototype.CompeteCanvasSnapDrawing = function() {
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

   function drawCanvas(divid, can /*, opt */) {
      let nocanvas = !can;
      if (nocanvas) {
         console.log("No canvas specified");
         return null;
         // can = JSROOT.Create("ROOT::Experimental::TCanvas");
      }

      let painter = new RCanvasPainter(can);
      painter.normal_canvas = !nocanvas;

      painter.SetDivId(divid, -1); // just assign id
      painter.CreateCanvasSvg(0);
      painter.SetDivId(divid);  // now add to painters list

      jsrp.SelectActivePad({ pp: painter, active: false });

      painter.DrawPrimitives(0, function() {
         painter.AddPadButtons();
         painter.ShowButtons();
         painter.DrawingReady();
      });

      return painter;
   }

   function drawPadSnapshot(divid, snap /*, opt*/) {
      let painter = new RCanvasPainter(null);
      painter.normal_canvas = false;
      painter.batch_mode = true;
      painter.SetDivId(divid, -1); // just assign id
      painter.RedrawPadSnap(snap, function() { painter.ShowButtons(); painter.DrawingReady(); });
      return painter;
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
      // do nothing, will be reimplemented in derived classes
   }

   RPavePainter.prototype.DrawPave = function() {

      //var framep = this.frame_painter();

      // frame painter must  be there
      //if (!framep)
      //   return console.log('no frame painter - no RPave drawing');

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

      if (!visible) return;

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

      this.DrawContent();

      if (JSROOT.BatchMode) return;

      JSROOT.require(['interactive']).then(inter => {
         // TODO: provide pave context menu as in v6
         if (JSROOT.settings.ContextMenu && this.PaveContextMenu)
            this.draw_g.on("contextmenu", this.PaveContextMenu.bind(this));

         inter.DragMoveHandler.AddDrag(this, { minwidth: 20, minheight: 20, redraw: this.SizeChanged.bind(this) });
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

      painter.SetDivId(divid);

      painter.DrawPave();

      return painter.DrawingReady();
   }

   // =======================================================================================


   function drawFrameTitle(reason) {
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
          text_size    = this.v7EvalAttr("text_size", 20),
          text_angle   = -1 * this.v7EvalAttr("text_angle", 0),
          // text_align   = this.v7EvalAttr("text_align", 22),
          text_color   = this.v7EvalColor("text_color", "black"),
          text_font    = this.v7EvalAttr("text_font", 41);

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

      let arg = { align: 22, x: title_width/2, y: title_height/2, text: title.fText, rotate: text_angle, color: text_color, latex: 1 };

      this.StartTextDrawing(text_font, text_size);

      this.DrawText(arg);

      this.FinishTextDrawing();

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

      let z = null, z_kind = "normal";

      if (framep && framep.logz) {
         z = d3.scaleLog();
         z_kind = "log";
      } else {
         z = d3.scaleLinear();
      }
      z.domain([zmin, zmax]).range([palette_height,0]);

      for (let i=0;i<contour.length-1;++i) {
         let z0 = z(contour[i]),
             z1 = z(contour[i+1]),
             col = palette.getContourColor((contour[i]+contour[i+1])/2);

         let r = g_btns.append("svg:rect")
                     .attr("x", 0)
                     .attr("y",  Math.round(z1))
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
            r.on("dblclick", function() { framep.Unzoom("z"); });
      }

      this.z_handle.SetAxisConfig("zaxis", z_kind, z, zmin, zmax, zmin, zmax);

      this.z_handle.max_tick_size = Math.round(palette_width*0.3);

      this.z_handle.DrawAxis(true, this.draw_g, palette_width, palette_height, "translate(" + palette_width + ", 0)");

      if (JSROOT.BatchMode) return;

      JSROOT.require(['interactive']).then(inter => {

         if (!after_resize)
            inter.DragMoveHandler.AddDrag(this, { minwidth: 20, minheight: 20, no_change_y: true, redraw: this.DrawPalette.bind(this, true) });

         if (!JSROOT.settings.Zooming) return;

         let doing_zoom = false, sel1 = 0, sel2 = 0, zoom_rect = null;

         let moveRectSel = evnt => {

            if (!doing_zoom) return;

            let m = d3.pointer(evnt);

            if (m[1] < sel1) sel1 = m[1]; else sel2 = m[1];

            zoom_rect.attr("y", sel1)
                    .attr("height", Math.abs(sel2-sel1));
         }

         let endRectSel = evnt => {
            if (!doing_zoom) return;

            evnt.preventDefault();
            this.draw_g.on("mousemove.colzoomRect", null)
                       .on("mouseup.colzoomRect", null);
            zoom_rect.remove();
            zoom_rect = null;
            doing_zoom = false;

            let zmin = Math.min(z.invert(sel1), z.invert(sel2)),
                zmax = Math.max(z.invert(sel1), z.invert(sel2));

            this.frame_painter().Zoom("z", zmin, zmax);
         }

         let startRectSel = evnt => {
            // ignore when touch selection is activated
            if (doing_zoom) return;
            doing_zoom = true;

            evnt.preventDefault();

            let origin = d3.pointer(evnt);

            sel1 = sel2 = origin[1];

            zoom_rect = g_btns
                 .append("svg:rect")
                 .attr("class", "zoom")
                 .attr("id", "colzoomRect")
                 .attr("x", "0")
                 .attr("width", palette_width)
                 .attr("y", sel1)
                 .attr("height", 5);

            this.draw_g.on("mousemove.colzoomRect", moveRectSel)
                       .on("mouseup.colzoomRect", endRectSel, true);

            evnt.stopPropagation();
         }

         this.draw_g.select(".axis_zoom")
                    .on("mousedown", startRectSel)
                    .on("dblclick", function() { framep.Unzoom("z"); });
      });
   }

   let drawPalette = (divid, palette, opt) => {
      let painter = new RPalettePainter(palette, opt);

      painter.SetDivId(divid);

      painter.CreateG(false);

      painter.z_handle = new JSROOT.v7.RAxisPainter(true, "z_");
      painter.z_handle.SetDivId(divid, -1);
      painter.z_handle.pad_name = painter.pad_name;
      painter.z_handle.invert_side = true;
      painter.z_handle.rstyle = painter.rstyle;

      return painter.DrawingReady();
   }

   // JSROOT.addDrawFunc({ name: "ROOT::Experimental::RPadDisplayItem", icon: "img_canvas", func: drawPad, opt: "" });

   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RHist1Drawable", icon: "img_histo1d", prereq: "v7hist", func: "JSROOT.v7.drawHist1", opt: "" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RHist2Drawable", icon: "img_histo2d", prereq: "v7hist", func: "JSROOT.v7.drawHist2", opt: "" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RHist3Drawable", icon: "img_histo3d", prereq: "v7hist3d", func: "JSROOT.v7.drawHist3", opt: "" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RHistDisplayItem", icon: "img_histo1d", prereq: "v7hist", func: "JSROOT.v7.drawHistDisplayItem", opt: "" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RText", icon: "img_text", prereq: "v7more", func: "JSROOT.v7.drawText", opt: "", direct: true, csstype: "text" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RFrameTitle", icon: "img_text", func: drawFrameTitle, opt: "", direct: true, csstype: "title" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RPaletteDrawable", icon: "img_text", func: drawPalette, opt: "" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RDisplayHistStat", icon: "img_pavetext", prereq: "v7hist", func: "JSROOT.v7.drawHistStats", opt: "" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RLine", icon: "img_graph", prereq: "v7more", func: "JSROOT.v7.drawLine", opt: "", direct: true, csstype: "line" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RBox", icon: "img_graph", prereq: "v7more", func: "JSROOT.v7.drawBox", opt: "", direct: true, csstype: "box" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RMarker", icon: "img_graph", prereq: "v7more", func: "JSROOT.v7.drawMarker", opt: "", direct: true, csstype: "marker" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RPave", icon: "img_pavetext", func: drawPave, opt: "" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RLegend", icon: "img_graph", prereq: "v7more", func: "JSROOT.v7.drawLegend", opt: "" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RPaveText", icon: "img_pavetext", prereq: "v7more", func: "JSROOT.v7.drawPaveText", opt: "" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RFrame", icon: "img_frame", func: "JSROOT.v7.drawFrame", opt: "" });

   JSROOT.v7.RAxisPainter = RAxisPainter;
   JSROOT.v7.RFramePainter = RFramePainter;
   JSROOT.v7.RPalettePainter = RPalettePainter;
   JSROOT.v7.RPadPainter = RPadPainter;
   JSROOT.v7.RCanvasPainter = RCanvasPainter;
   JSROOT.v7.RPavePainter = RPavePainter;
   JSROOT.v7.drawFrame = drawFrame;
   JSROOT.v7.drawPad = drawPad;
   JSROOT.v7.drawCanvas = drawCanvas;
   JSROOT.v7.drawPadSnapshot = drawPadSnapshot;
   JSROOT.v7.drawPave = drawPave;
   JSROOT.v7.drawFrameTitle = drawFrameTitle;

   return JSROOT;

});