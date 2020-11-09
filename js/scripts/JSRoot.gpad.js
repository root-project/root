/// @file JSRoot.gpad.js
/// JSROOT TPad/TCanvas/TFrame support

JSROOT.define(['d3', 'painter'], (d3, jsrp) => {

   "use strict";

   // identifier used in TWebCanvas painter
   let webSnapIds = { kNone: 0,  kObject: 1, kSVG: 2, kSubPad: 3, kColors: 4, kStyle: 5 };

   // =======================================================================


   /**
    * @summary Painter for TAxis/TGaxis objects.
    *
    * @class
    * @memberof JSROOT
    * @extends JSROOT.ObjectPainter
    * @param {object} axis - object to draw
    * @param {boolean} embedded - if true, painter used in other objects painters
    * @private
    */

   function TAxisPainter(axis, embedded) {
      JSROOT.ObjectPainter.call(this, axis);

      this.embedded = embedded; // indicate that painter embedded into the histo painter

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

   TAxisPainter.prototype = Object.create(JSROOT.ObjectPainter.prototype);

   TAxisPainter.prototype.Cleanup = function() {

      this.ticks = [];
      delete this.func;
      delete this.format;
      delete this.gr;

      JSROOT.ObjectPainter.prototype.Cleanup.call(this);
   }

   TAxisPainter.prototype.ConvertDate = function(v) {
      return new Date(this.timeoffset + v*1000);
   }

   /** @summary Convert graphical point back into axis value */
   TAxisPainter.prototype.RevertPoint = function(pnt) {
      let value = this.func.invert(pnt);
      return (this.kind == "time") ?  (value - this.timeoffset) / 1000 : value;
   }

   TAxisPainter.prototype.ConfigureAxis = function(name, min, max, smin, smax, vertical, range, opts) {
      this.name = name;
      this.full_min = min;
      this.full_max = max;
      this.kind = "normal";
      this.vertical = vertical;
      this.log = opts.log || 0;
      this.reverse = opts.reverse || false;
      this.swap_side = opts.swap_side || false;

      let axis = this.GetObject();

      if (opts.time_scale || axis.fTimeDisplay) {
         this.kind = 'time';
         this.timeoffset = jsrp.getTimeOffset(axis);
      } else {
         this.kind = !axis.fLabels ? 'normal' : 'labels';
      }

      if (this.kind == 'time') {
         this.func = d3.scaleTime().domain([this.ConvertDate(smin), this.ConvertDate(smax)]);
      } else if (this.log) {

         if (smax <= 0) smax = 1;

         if ((smin <= 0) && axis && !opts.logcheckmin)
            for (let i = 0; i < axis.fNbins; ++i) {
               smin = Math.max(smin, axis.GetBinLowEdge(i+1));
               if (smin > 0) break;
            }

         if ((smin <= 0) && opts.log_min_nz)
            smin = opts.log_min_nz;

         if ((smin <= 0) || (smin >= smax))
            smin = smax * (opts.logminfactor || 1e-4);

         this.func = d3.scaleLog().base((this.log == 2) ? 2 : 10).domain([smin,smax]);
      } else {
         this.func = d3.scaleLinear().domain([smin,smax]);
      }

      if (this.vertical ^ this.reverse) {
         let d = range[0]; range[0] = range[1]; range[1] = d;
      }

      this.func.range(range);

      this.scale_min = smin;
      this.scale_max = smax;

      if (this.kind == 'time')
         this.gr = val => this.func(this.ConvertDate(val));
      else if (this.log)
         this.gr = val => (val < this.scale_min) ? (this.vertical ? this.func.range()[0]+5 : -5) : this.func(val);
      else
         this.gr = this.func;


      let is_gaxis = (axis && axis._typename === 'TGaxis');

      delete this.format;// remove formatting func

      let ndiv = 508;
      if (is_gaxis)
         ndiv = axis.fNdiv;
       else if (axis)
          ndiv = Math.max(axis.fNdivisions, 4);

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

         this.format = this.formatTime;

      } else if (this.log) {
         if (this.nticks2 > 1) {
            this.nticks *= this.nticks2; // all log ticks (major or minor) created centrally
            this.nticks2 = 1;
         }
         this.noexp = axis ? axis.TestBit(JSROOT.EAxisBits.kNoExponent) : false;
         if ((this.scale_max < 300) && (this.scale_min > 0.3)) this.noexp = true;
         this.moreloglabels = axis ? axis.TestBit(JSROOT.EAxisBits.kMoreLogLabels) : false;

         this.format = this.formatLog;

      } else if (this.kind == 'labels') {
         this.nticks = 50; // for text output allow max 50 names
         let scale_range = this.scale_max - this.scale_min;
         if (this.nticks > scale_range)
            this.nticks = Math.round(scale_range);

         this.regular_labels = true;

         if (axis && axis.fNbins && axis.fLabels) {
            if ((axis.fNbins != Math.round(axis.fXmax - axis.fXmin)) ||
                (axis.fXmin != 0) || (axis.fXmax != axis.fNbins)) {
               this.regular_labels = false;
            }
         }

         this.nticks2 = 1;

         this.format = this.formatLabels;
      } else {
         this.order = 0;
         this.ndig = 0;
         this.format = this.formatNormal;
      }
   }

   TAxisPainter.prototype.formatTime = function(d, asticks) {
       return asticks ? this.tfunc1(d) : this.tfunc2(d);
   }

   TAxisPainter.prototype.formatLabels = function(d) {
      let indx = parseFloat(d), a = this.GetObject();
      if (!this.regular_labels)
         indx = (indx - a.fXmin)/(a.fXmax - a.fXmin) * a.fNbins;
      indx = Math.floor(indx);
      if ((indx < 0) || (indx >= a.fNbins)) return null;
      for (let i = 0; i < a.fLabels.arr.length; ++i) {
         let tstr = a.fLabels.arr[i];
         if (tstr.fUniqueID === indx+1) return tstr.fString;
      }
      return null;
   }

   TAxisPainter.prototype.formatLog = function(d, asticks, fmt) {
      let val = parseFloat(d), rnd = Math.round(val);
      if (!asticks)
         return ((rnd === val) && (Math.abs(rnd)<1e9)) ? rnd.toString() : JSROOT.FFormat(val, fmt || JSROOT.gStyle.fStatFormat);
      if (val <= 0) return null;
      let vlog = Math.log10(val), base = 10;
      if (this.log == 2) { base = 2; vlog = vlog / Math.log10(2); }
      if (this.moreloglabels || (Math.abs(vlog - Math.round(vlog)) < 0.001)) {
         if (!this.noexp && (asticks != 2))
            return this.formatExp(base, Math.floor(vlog+0.01), val);

         return (vlog < 0) ? val.toFixed(Math.round(-vlog+0.5)) : val.toFixed(0);
      }
      return null;
   }

   TAxisPainter.prototype.formatNormal = function(d, asticks, fmt) {
      let val = parseFloat(d);
      if (asticks && this.order) val = val / Math.pow(10, this.order);

      if (val === Math.round(val))
         return (Math.abs(val)<1e9) ? val.toFixed(0) : val.toExponential(4);

      if (asticks) return (this.ndig>10) ? val.toExponential(this.ndig-11) : val.toFixed(this.ndig);

      return JSROOT.FFormat(val, fmt || JSROOT.gStyle.fStatFormat);
   }

   /** @summary Assign often used members of frame painter
     * @private */
   TAxisPainter.prototype.AssignFrameMembers = function(fp, axis) {
      fp["gr"+axis] = this.gr;                    // fp.grx
      fp["log"+axis] = this.log;                  // fp.logx
      fp["scale_"+axis+"min"] = this.scale_min;   // fp.scale_xmin
      fp["scale_"+axis+"max"] = this.scale_max;   // fp.scale_xmax
   }

   TAxisPainter.prototype.formatExp = function(base, order, value) {
      let res = "";
      if (value) {
         value = Math.round(value/Math.pow(base,order));
         if ((value!=0) && (value!=1)) res = value.toString() + (JSROOT.settings.Latex ? "#times" : "x");
      }
      res += base.toString();
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


   /** @summary Convert "raw" axis value into text */
   TAxisPainter.prototype.AxisAsText = function(value, fmt) {
      if (this.kind == 'time')
         value = this.ConvertDate(value);
      if (this.format)
         return this.format(value, false, fmt);
      return value.toPrecision(4);
   }

   TAxisPainter.prototype.ProduceTicks = function(ndiv, ndiv2) {
      if (!this.noticksopt) {
         let total = ndiv * (ndiv2 || 1);
         return this.log ? jsrp.PoduceLogTicks(this.func, total) : this.func.ticks(total);
      }

      let dom = this.func.domain(), ticks = [];
      if (ndiv2) ndiv = (ndiv-1) * ndiv2;
      for (let n=0;n<=ndiv;++n)
         ticks.push((dom[0]*(ndiv-n) + dom[1]*n)/ndiv);
      return ticks;
   }

   /** @summary Creates array with minor/middle/major ticks */
   TAxisPainter.prototype.CreateTicks = function(only_major_as_array, optionNoexp, optionNoopt, optionInt) {

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
         let axis = this.GetObject();
         for (let n = 0; n < axis.fNbins; ++n) {
            let x = axis.fXmin + n / axis.fNbins * (axis.fXmax - axis.fXmin);
            if ((x >= this.scale_min) && (x < this.scale_max)) handle.lbl_pos.push(x);
         }
      }

      if ((this.nticks2 > 1) && (this.log != 2)) {
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
            while (indx < handle.major.length) {
               let lbl = this.format(handle.major[indx], true);
               if (lbls.indexOf(lbl) < 0) {
                  lbls.push(lbl);
                  totallen += lbl.length;
                  indx++;
                  continue;
               }
               if (++this.ndig > 15) break; // not too many digits, anyway it will be exponential
               lbls = []; indx = 0; totallen = 0;
            }

            // for order==0 we should virtually remove "0." and extra label on top
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

   TAxisPainter.prototype.IsCenterLabels = function() {
      if (this.kind === 'labels') return true;
      if (this.log) return false;
      let axis = this.GetObject();
      return axis && axis.TestBit(JSROOT.EAxisBits.kCenterLabels);
   }

   /** @summary Add interactive elements to draw axes title */
   TAxisPainter.prototype.AddTitleDrag = function(title_g, vertical, offset_k, reverse, axis_length) {
      if (!JSROOT.settings.MoveResize || JSROOT.BatchMode) return;

      let drag_rect = null,
          acc_x, acc_y, new_x, new_y, sign_0, alt_pos, curr_indx,
          drag_move = d3.drag().subject(Object);

      drag_move
         .on("start", evnt => {

            evnt.sourceEvent.preventDefault();
            evnt.sourceEvent.stopPropagation();

            let box = title_g.node().getBBox(), // check that elements visible, request precise value
                axis = this.GetObject(),
                title_length = vertical ? box.height : box.width,
                opposite = axis.TestBit(JSROOT.EAxisBits.kOppositeTitle);

            new_x = acc_x = title_g.property('shift_x');
            new_y = acc_y = title_g.property('shift_y');

            sign_0 = vertical ? (acc_x > 0) : (acc_y > 0); // sign should remain

            alt_pos = vertical ? [axis_length, axis_length/2, 0] : [0, axis_length/2, axis_length]; // possible positions
            let off = vertical ? -title_length/2 : title_length/2;
            if (this.title_align == "middle") {
               alt_pos[0] +=  off;
               alt_pos[2] -=  off;
            } else if (this.title_align == "begin") {
               alt_pos[1] -= off;
               alt_pos[2] -= 2*off;
            } else { // end
               alt_pos[0] += 2*off;
               alt_pos[1] += off;
            }

            if (axis.TestBit(JSROOT.EAxisBits.kCenterTitle))
               curr_indx = 1;
            else if (reverse ^ opposite)
               curr_indx = 0;
            else
               curr_indx = 2;

            alt_pos[curr_indx] = vertical ? acc_y : acc_x;

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
                   p = vertical ? acc_y : acc_x, besti = 0;

               for (let i=1; i<3; ++i)
                  if (Math.abs(p - alt_pos[i]) < Math.abs(p - alt_pos[besti])) besti = i;

               if (vertical) {
                  set_x = acc_x;
                  set_y = alt_pos[besti];
               } else {
                  set_y = acc_y;
                  set_x = alt_pos[besti];
               }

               if (sign_0 === (vertical ? (set_x > 0) : (set_y > 0))) {
                  new_x = set_x; new_y = set_y; curr_indx = besti;
                  title_g.attr('transform', 'translate(' + new_x + ',' + new_y +  ')');
               }

          }).on("end", evnt => {
               if (!drag_rect) return;

               evnt.sourceEvent.preventDefault();
               evnt.sourceEvent.stopPropagation();

               title_g.property('shift_x', new_x)
                      .property('shift_y', new_y);

               let axis = this.GetObject(), abits = JSROOT.EAxisBits;

               function set_bit(bit, on) {
                  if (axis.TestBit(bit) != on) axis.InvertBit(bit);
               }

               axis.fTitleOffset = (vertical ? new_x : new_y) / offset_k;
               if (curr_indx == 1) {
                  set_bit(abits.kCenterTitle, true);
                  set_bit(abits.kOppositeTitle, false);
               } else if (curr_indx == 0) {
                  set_bit(abits.kCenterTitle, false);
                  set_bit(abits.kOppositeTitle, true);
               } else {
                  set_bit(abits.kCenterTitle, false);
                  set_bit(abits.kOppositeTitle, false);
               }

               drag_rect.remove();
               drag_rect = null;
            });

      title_g.style("cursor", "move").call(drag_move);
   }

   /** @summary function draws  TAxis or TGaxis object  */
   TAxisPainter.prototype.DrawAxis = function(layer, w, h, transform, second_shift, disable_axis_drawing, max_text_width) {

      let axis = this.GetObject(), chOpt = "",
          is_gaxis = (axis && axis._typename === 'TGaxis'),
          axis_g = layer, tickSize = 0.03,
          scaling_size = 100, draw_lines = true,
          pad_w = this.pad_width() || 10,
          pad_h = this.pad_height() || 10,
          resolveFunc, totalTextCallbacks = 0, totalDone = false,
          promise = new Promise(resolve => { resolveFunc = resolve; }),
          vertical = this.vertical,
          swap_side = this.swap_side || false;

      let checkTextCallBack = (is_callback) => {
          if (is_callback) totalTextCallbacks--; else totalDone = true;
          if (!totalTextCallbacks && totalDone && resolveFunc) {
            resolveFunc(true);
            resolveFunc = null;
         }
      };

      // shift for second ticks set (if any)
      if (!second_shift) second_shift = 0; else
      if (this.invert_side) second_shift = -second_shift;

      if (is_gaxis) {
         this.createAttLine({ attr: axis });
         draw_lines = axis.fLineColor != 0;
         chOpt = axis.fChopt;
         tickSize = axis.fTickSize;
         scaling_size = (vertical ? 1.7*h : 0.6*w);
      } else {
         this.createAttLine({ color: axis.fAxisColor, width: 1, style: 1 });
         chOpt = (vertical ^ this.invert_side) ? "-S" : "+S";
         tickSize = axis.fTickLength;
         scaling_size = (vertical ? pad_w : pad_h);
      }

      // indicate that attributes created not for TAttLine, therefore cannot be updated as TAttLine in GED
      this.lineatt.not_standard = true;

      if (!is_gaxis || (this.name === "zaxis")) {
         axis_g = layer.select("." + this.name + "_container");
         if (axis_g.empty())
            axis_g = layer.append("svg:g").attr("class",this.name + "_container");
         else
            axis_g.selectAll("*").remove();
      }

      if (!disable_axis_drawing && draw_lines)
         axis_g.append("svg:line")
               .attr("x1",0).attr("y1",0)
               .attr("x2", vertical ? 0 : w)
               .attr("y2", vertical ? h : 0)
               .call(this.lineatt.func);

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
      if (optionMinus) { side = (swap_side ^ vertical) ? 1 : -1; } else
      if (optionPlus) { side = (swap_side ^ vertical) ? -1 : 1; }

      tickSize = Math.round((optionSize ? tickSize : 0.03) * scaling_size);

      if (this.max_tick_size && (tickSize > this.max_tick_size)) tickSize = this.max_tick_size;

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
            // FIXME: for labels last tick is smaller,
            if (/*(this.kind == "labels") || */ (this.format(handle.tick,true) !== null)) h1 = tickSize;
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

      // draw labels (sometime on both sides)
      if (!disable_axis_drawing && !optionUnlab) {

         let label_color = this.get_color(axis.fLabelColor),
             labeloffset = Math.round(Math.abs(axis.fLabelOffset)*text_scaling_size),
             center_lbls = this.IsCenterLabels(),
             rotate_lbls = axis.TestBit(JSROOT.EAxisBits.kLabelsVert),
             textscale = 1, maxtextlen = 0, lbls_tilt = false, labelfont = null,
             label_g = [ axis_g.append("svg:g").attr("class","axis_labels") ],
             lbl_pos = handle.lbl_pos || handle.major,
             total_draw_cnt = 0, all_done = 0;

         if (this.lbls_both_sides)
            label_g.push(axis_g.append("svg:g").attr("class","axis_labels").attr("transform", vertical ? "translate(" + w + ",0)" : "translate(0," + (-h) + ")"));

         // function called when text is drawn to analyze width, required to correctly scale all labels
         function process_drawtext_ready(painter) {
            let textwidth = this.result_width;

            if (textwidth && ((!vertical && !rotate_lbls) || (vertical && rotate_lbls)) && !painter.log) {
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

               let arg = { text: lbl, color: label_color, latex: 1, draw_g: label_g[lcnt], normal_side: (lcnt == 0), lgs: label_g };

               let pos = Math.round(this.func(lbl_pos[nmajor]));

               arg.gap_before = (nmajor>0) ? Math.abs(Math.round(pos - this.func(lbl_pos[nmajor-1]))) : 0;

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
                               align: vertical ? ((side < 0) ? 30 : 10) : ( (this.has_obstacle ^ (side < 0)) ? 13 : 10 ),
                               latex: 1,
                               text: '#times' + this.formatExp(10, this.order),
                               draw_g: label_g[lcnt]
               });
         }

         all_done = 1;

         totalTextCallbacks += label_g.length;
         for (let lcnt = 0; lcnt < label_g.length; ++lcnt)
            this.FinishTextDrawing(label_g[lcnt], () => {
               if (lbls_tilt)
                  label_g[lcnt].selectAll("text").each(function() {
                     let txt = d3.select(this), tr = txt.attr("transform");
                     txt.attr("transform", tr + " rotate(25)").style("text-anchor", "start");
                  });

               checkTextCallBack(true);
            });

         if (label_g.length > 1) side = -side;

         if (labelfont) labelsize = labelfont.size; // use real font size
      }

      if (JSROOT.settings.Zooming && !this.disable_zooming && !JSROOT.BatchMode) {
         let r = axis_g.append("svg:rect")
                       .attr("class", "axis_zoom")
                       .style("opacity", "0")
                       .style("cursor", "crosshair");

         if (vertical)
            r.attr("x", (side>0) ? (-2*labelsize - 3) : 3)
             .attr("y", 0)
             .attr("width", 2*labelsize + 3)
             .attr("height", h)
         else
            r.attr("x", 0).attr("y", (side>0) ? 0 : -labelsize - 3)
             .attr("width", w).attr("height", labelsize + 3);
      }

      if ((axis.fTitle.length > 0) && !disable_axis_drawing) {
         let title_g = axis_g.append("svg:g").attr("class", "axis_title"),
             title_fontsize = (axis.fTitleSize >= 1) ? axis.fTitleSize : Math.round(axis.fTitleSize * text_scaling_size),
             title_offest_k = 1.6*(axis.fTitleSize<1 ? axis.fTitleSize : axis.fTitleSize/(this.canv_painter().pad_height() || 10)),
             center = axis.TestBit(JSROOT.EAxisBits.kCenterTitle),
             opposite = axis.TestBit(JSROOT.EAxisBits.kOppositeTitle),
             rotate = axis.TestBit(JSROOT.EAxisBits.kRotateTitle) ? -1 : 1,
             title_color = this.get_color(is_gaxis ? axis.fTextColor : axis.fTitleColor),
             shift_x = 0, shift_y = 0;

         this.StartTextDrawing(axis.fTitleFont, title_fontsize, title_g);

         let xor_reverse = swap_side ^ opposite, myxor = (rotate < 0) ^ xor_reverse;

         this.title_align = center ? "middle" : (myxor ? "begin" : "end");

         if (vertical) {
            title_offest_k *= -side*pad_w;

            shift_x = Math.round(title_offest_k*axis.fTitleOffset);

            if ((this.name == "zaxis") && is_gaxis && ('getBoundingClientRect' in axis_g.node())) {
               // special handling for color palette labels - draw them always on right side
               let rect = axis_g.node().getBoundingClientRect();
               if (shift_x < rect.width - tickSize) shift_x = Math.round(rect.width - tickSize);
            }

            shift_y = Math.round(center ? h/2 : (xor_reverse ? h : 0));

            this.DrawText({ align: this.title_align+";middle",
                            rotate: (rotate<0) ? 90 : 270,
                            text: axis.fTitle, color: title_color, draw_g: title_g });
         } else {
            title_offest_k *= side*pad_h;

            shift_x = Math.round(center ? w/2 : (xor_reverse ? 0 : w));
            shift_y = Math.round(title_offest_k*axis.fTitleOffset);
            this.DrawText({ align: this.title_align+";middle",
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
               shift_x = (side > 0) ? Math.round(axis_rect.left - title_rect.right - title_fontsize*0.3) :
                                      Math.round(axis_rect.right - title_rect.left + title_fontsize*0.3);
            }

            title_g.attr('transform', 'translate(' + shift_x + ',' + shift_y + ')')
                   .property('shift_x', shift_x)
                   .property('shift_y', shift_y);

            checkTextCallBack(true);
         });

         this.AddTitleDrag(title_g, vertical, title_offest_k, swap_side, vertical ? h : w);
      }

      this.position = 0;

      if (!disable_axis_drawing && ('getBoundingClientRect' in axis_g.node())) {
         let rect1 = axis_g.node().getBoundingClientRect(),
             rect2 = this.svg_pad().node().getBoundingClientRect();

         this.position = rect1.left - rect2.left; // use to control left position of Y scale
      }

      checkTextCallBack(false);

      return promise;
   }

   TAxisPainter.prototype.Redraw = function() {

      let gaxis = this.GetObject(),
          x1 = this.AxisToSvg("x", gaxis.fX1),
          y1 = this.AxisToSvg("y", gaxis.fY1),
          x2 = this.AxisToSvg("x", gaxis.fX2),
          y2 = this.AxisToSvg("y", gaxis.fY2),
          w = x2 - x1, h = y1 - y2,
          vertical = Math.abs(w) < Math.abs(h),
          sz = vertical ? h : w,
          reverse = false,
          min = gaxis.fWmin, max = gaxis.fWmax;

      if (sz < 0) {
         reverse = true;
         sz = -sz;
         if (vertical) y2 = y1; else x1 = x2;
      }

      this.ConfigureAxis(vertical ? "yaxis" : "xaxis", min, max, min, max, vertical, [0, sz], {
         time_scale: gaxis.fChopt.indexOf("t") >= 0,
         log: (gaxis.fChopt.indexOf("G") >= 0) ? 1 : 0,
         reverse: reverse,
         swap_side: reverse
      });

      this.CreateG();

      return this.DrawAxis(this.draw_g, Math.abs(w), Math.abs(h), "translate(" + x1 + "," + y2 +")");
   }

   let drawGaxis = (divid, obj /*, opt*/) => {
      let painter = new TAxisPainter(obj, false);

      painter.SetDivId(divid);

      painter.disable_zooming = true;

      return painter.Redraw().then(() => painter);
   }

   // ===============================================

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
    * @summary Painter class for TFrame, main handler for interactivity
    *
    * @class
    * @memberof JSROOT
    * @extends ObjectPainter
    * @param {object} tframe - TFrame object
    * @private
    */

   function TFramePainter(tframe) {
      JSROOT.ObjectPainter.call(this, (tframe && tframe.$dummy) ? null : tframe);
      this.zoom_kind = 0;
      this.mode3d = false;
      this.shrink_frame_left = 0.;
      this.xmin = this.xmax = 0; // no scale specified, wait for objects drawing
      this.ymin = this.ymax = 0; // no scale specified, wait for objects drawing
      this.ranges_set = false;
      this.axes_drawn = false;
      this.keys_handler = null;
      this.projection = 0; // different projections
   }

   TFramePainter.prototype = Object.create(JSROOT.ObjectPainter.prototype);

   /** @summary Returns frame painter - object itself */
   TFramePainter.prototype.frame_painter = function() { return this; }

   /** @summary Returns frame or sub-objects, used in GED editor */
   TFramePainter.prototype.GetObject = function(place) {
      if (place === "xaxis") return this.xaxis;
      if (place === "yaxis") return this.yaxis;
      return JSROOT.ObjectPainter.prototype.GetObject.call(this);
   }

   /** @summary Set active flag for frame - can block some events
    * @private */
   TFramePainter.prototype.SetActive = function(/* on */) {
      // do nothing here - key handler is handled differently
   }

   TFramePainter.prototype.GetTipName = function(append) {
      let res = JSROOT.ObjectPainter.prototype.GetTipName.call(this) || "TFrame";
      if (append) res+=append;
      return res;
   }

   TFramePainter.prototype.Shrink = function(shrink_left, shrink_right) {
      this.fX1NDC += shrink_left;
      this.fX2NDC -= shrink_right;
   }

   /** @summary Set position of last context menu event */
   TFramePainter.prototype.SetLastEventPos = function(pnt) {
      this.fLastEventPnt = pnt;
   }

   /** @summary Return position of last event */
   TFramePainter.prototype.GetLastEventPos = function() { return this.fLastEventPnt; }

   /** @summary  Returns coordinates transformation func */
   TFramePainter.prototype.GetProjectionFunc = function() {
      switch (this.projection) {
         case 1: return ProjectAitoff2xy;
         case 2: return ProjectMercator2xy;
         case 3: return ProjectSinusoidal2xy;
         case 4: return ProjectParabolic2xy;
      }
   }

   /** @summary Rcalculate frame ranges using specified projection functions */
   TFramePainter.prototype.RecalculateRange = function(Proj) {
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

   /** @summary Configure axes ranges */
   TFramePainter.prototype.SetAxesRanges = function(xaxis, xmin, xmax, yaxis, ymin, ymax, zaxis, zmin, zmax) {
      this.ranges_set = true;

      this.xaxis = xaxis;
      this.xmin = xmin;
      this.xmax = xmax;

      this.yaxis = yaxis;
      this.ymin = ymin;
      this.ymax = ymax;

      this.zaxis = zaxis;
      this.zmin = zmin;
      this.zmax = zmax;
   }

   /** @summary Retuns axis object */
   TFramePainter.prototype.GetAxis = function(name) {
      switch(name) {
         case "x": return this.xaxis;
         case "y": return this.yaxis;
         case "z": return this.zaxis;
      }
      return null;
   }

   TFramePainter.prototype.CheckAxisZoom = function(name) {
      let axis = this.GetAxis(name);
      if (axis && axis.TestBit(JSROOT.EAxisBits.kAxisRange)) {
         if ((axis.fFirst !== axis.fLast) && ((axis.fFirst > 1) || (axis.fLast < axis.fNbins))) {
            this['zoom_' + name + 'min'] = axis.fFirst > 1 ? axis.GetBinLowEdge(axis.fFirst) : axis.fXmin;
            this['zoom_' + name + 'max'] = axis.fLast < axis.fNbins ? axis.GetBinLowEdge(axis.fLast+1) : axis.fXmax;
            // reset user range for main painter
            axis.InvertBit(JSROOT.EAxisBits.kAxisRange);
            axis.fFirst = 1; axis.fLast = axis.fNbins
         }
      }
   }

   TFramePainter.prototype.CheckPadUserRange = function(pad, name) {
      if (!pad) return;

      // seems to be, not allways user range calculated
      let umin = pad['fU' + name + 'min'],
          umax = pad['fU' + name + 'max'],
          eps = 1e-7;

      if (name == "x") {
         if ((Math.abs(pad.fX1) > eps) || (Math.abs(pad.fX2-1) > eps)) {
            let dx = pad.fX2 - pad.fX1;
            umin = pad.fX1 + dx*pad.fLeftMargin;
            umax = pad.fX2 - dx*pad.fRightMargin;
         }
      } else {
         if ((Math.abs(pad.fY1) > eps) || (Math.abs(pad.fY2-1) > eps)) {
            let dy = pad.fY2 - pad.fY1;
            umin = pad.fY1 + dy*pad.fBottomMargin;
            umax = pad.fY2 - dy*pad.fTopMargin;
         }
      }

      if ((umin>=umax) || (Math.abs(umin)<eps && Math.abs(umax-1)<eps)) return;

      if (pad['fLog' + name] > 0) {
         umin = Math.exp(umin * Math.log(10));
         umax = Math.exp(umax * Math.log(10));
      }

      let aname = name;
      if (this.swap_xy) aname = (name=="x") ? "y" : "x";
      let smin = 'scale_' + aname + 'min',
          smax = 'scale_' + aname + 'max';

      eps = (this[smax] - this[smin]) * 1e-7;

      if ((Math.abs(umin - this[smin]) > eps) || (Math.abs(umax - this[smax]) > eps)) {
         this["zoom_" + aname + "min"] = umin;
         this["zoom_" + aname + "max"] = umax;
      }
   }

   /** @summary Create x,y objects which maps user coordinates into pixels
     * @desc While only first painter really need such object, all others just reuse it
     * following functions are introduced
     *    this.GetBin[X/Y]  return bin coordinate
     *    this.[x,y]  these are d3.scale objects
     *    this.gr[x,y]  converts root scale into graphical value
     * @private */
   TFramePainter.prototype.CreateXY = function(opts) {

      this.CleanXY(); // remove all previous configurations

      if (!opts) opts = {};

      this.swap_xy = opts.swap_xy || false;
      this.reverse_x = opts.reverse_x || false;
      this.reverse_y = opts.reverse_y || false;

      this.logx = this.logy = 0;

      let w = this.frame_width(), h = this.frame_height(), pad = this.root_pad();

      this.scale_xmin = this.xmin;
      this.scale_xmax = this.xmax;

      this.scale_ymin = this.ymin;
      this.scale_ymax = this.ymax;

      if (opts.extra_y_space) {
         let log_scale = this.swap_xy ? pad.fLogx : pad.fLogy;
         if (log_scale && (this.scale_ymax > 0))
            this.scale_ymax = Math.exp(Math.log(this.scale_ymax)*1.1);
         else
            this.scale_ymax += (this.scale_ymax - this.scale_ymin)*0.1;
      }

      if (opts.check_pad_range) {
         // take zooming out of pad or axis attributes

         this.zoom_xmin = this.zoom_xmax = 0;
         this.zoom_ymin = this.zoom_ymax = 0;
         this.zoom_zmin = this.zoom_zmax = 0;

         this.CheckAxisZoom('x');
         if (opts.ndim && (opts.ndim > 1)) this.CheckAxisZoom('y');
         if (opts.ndim && (opts.ndim > 2)) this.CheckAxisZoom('z');

         if (opts.check_pad_range === "pad_range") {
            let canp = this.canv_painter();
            // ignore range set in the online canvas
            if (!canp || !canp.online_canvas) {
               this.CheckPadUserRange(pad, 'x');
               this.CheckPadUserRange(pad, 'y');
            }
         }
      }

      if ((this.zoom_ymin == this.zoom_ymax) && (opts.zoom_ymin != opts.zoom_ymax) && !this.zoom_changed_interactive) {
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

      // projection should be assigned
      this.RecalculateRange(opts.Proj);

      this.x_handle = new TAxisPainter(this.xaxis, true);
      this.x_handle.SetDivId(this.divid, -1);
      this.x_handle.pad_name = this.pad_name;

      this.x_handle.ConfigureAxis("xaxis", this.xmin, this.xmax, this.scale_xmin, this.scale_xmax, this.swap_xy, this.swap_xy ? [0,h] : [0,w],
                                      { reverse: this.reverse_x,
                                        log: this.swap_xy ? pad.fLogy : pad.fLogx,
                                        logcheckmin: this.swap_xy,
                                        logminfactor: 0.0001 });

      this.x_handle.AssignFrameMembers(this,"x");

      this.y_handle = new TAxisPainter(this.yaxis, true);
      this.y_handle.SetDivId(this.divid, -1);
      this.y_handle.pad_name = this.pad_name;

      this.y_handle.ConfigureAxis("yaxis", this.ymin, this.ymax, this.scale_ymin, this.scale_ymax, !this.swap_xy, this.swap_xy ? [0,w] : [0,h],
                                      { reverse: this.reverse_y,
                                        log: this.swap_xy ? pad.fLogx : pad.fLogy,
                                        logcheckmin: (opts.ndim < 2) || this.swap_xy,
                                        log_min_nz: opts.ymin_nz && (opts.ymin_nz < 0.01*this.ymax) ? 0.3 * opts.ymin_nz : 0,
                                        logminfactor: 3e-4 });

      this.y_handle.AssignFrameMembers(this,"y");

      this.SetRootPadRange(pad);
   }

   /** @summary Set selected range back to TPad object */
   TFramePainter.prototype.SetRootPadRange = function(pad, is3d) {
      if (!pad || !this.ranges_set) return;

      if (is3d) {
         // this is fake values, algorithm should be copied from TView3D class of ROOT
         // pad.fLogx = pad.fLogy = 0;
         pad.fUxmin = pad.fUymin = -0.9;
         pad.fUxmax = pad.fUymax = 0.9;
      } else {
         pad.fLogx = this.swap_xy ? this.logy : this.logx;
         pad.fUxmin = pad.fLogx ? Math.log10(this.scale_xmin) : this.scale_xmin;
         pad.fUxmax = pad.fLogx ? Math.log10(this.scale_xmax) : this.scale_xmax;
         pad.fLogy = this.swap_xy ? this.logx : this.logy;
         pad.fUymin = pad.fLogy ? Math.log10(this.scale_ymin) : this.scale_ymin;
         pad.fUymax = pad.fLogy ? Math.log10(this.scale_ymax) : this.scale_ymax;
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
   }


   /** @summary grid can only be drawn by first painter */
   TFramePainter.prototype.DrawGrids = function() {

      let layer = this.svg_frame().select(".grid_layer");

      layer.selectAll(".xgrid").remove();
      layer.selectAll(".ygrid").remove();

      let pad = this.root_pad(),
          h = this.frame_height(),
          w = this.frame_width(),
          grid_style = JSROOT.gStyle.fGridStyle;

      if ((grid_style < 0) || (grid_style >= jsrp.root_line_styles.length)) grid_style = 11;

      // add a grid on x axis, if the option is set
      if (pad && pad.fGridx && this.x_handle) {
         let gridx = "";
         for (let n=0;n<this.x_handle.ticks.length;++n)
            if (this.swap_xy)
               gridx += "M0,"+this.x_handle.ticks[n]+"h"+w;
            else
               gridx += "M"+this.x_handle.ticks[n]+",0v"+h;

         let colid = (JSROOT.gStyle.fGridColor > 0) ? JSROOT.gStyle.fGridColor : (this.GetAxis("x") ? this.GetAxis("x").fAxisColor : 1),
             grid_color = this.get_color(colid) || "black";

         if (gridx.length > 0)
           layer.append("svg:path")
                .attr("class", "xgrid")
                .attr("d", gridx)
                .style('stroke', grid_color)
                .style("stroke-width", JSROOT.gStyle.fGridWidth)
                .style("stroke-dasharray", jsrp.root_line_styles[grid_style]);
      }

      // add a grid on y axis, if the option is set
      if (pad && pad.fGridy && this.y_handle) {
         let gridy = "";
         for (let n=0;n<this.y_handle.ticks.length;++n)
            if (this.swap_xy)
               gridy += "M"+this.y_handle.ticks[n]+",0v"+h;
            else
               gridy += "M0,"+this.y_handle.ticks[n]+"h"+w;

         let colid = (JSROOT.gStyle.fGridColor > 0) ? JSROOT.gStyle.fGridColor : (this.GetAxis("y") ? this.GetAxis("y").fAxisColor : 1),
             grid_color = this.get_color(colid) || "black";

         if (gridy.length > 0)
           layer.append("svg:path")
                .attr("class", "ygrid")
                .attr("d", gridy)
                .style('stroke',grid_color)
                .style("stroke-width",JSROOT.gStyle.fGridWidth)
                .style("stroke-dasharray", jsrp.root_line_styles[grid_style]);
      }
   }

   /** @summary Converts "raw" axis value into text */
   TFramePainter.prototype.AxisAsText = function(axis, value) {
      let handle = this[axis+"_handle"];

      if (handle)
         return handle.AxisAsText(value, JSROOT.settings[axis.toUpperCase() + "ValuesFormat"]);

      return value.toPrecision(4);
   }

   /** @summary draw axes, return Promise which ready when drawing is completed  */
   TFramePainter.prototype.DrawAxes = function(shrink_forbidden, disable_axis_draw, AxisPos, has_x_obstacle) {

      this.CleanAxesDrawings();

      if ((this.xmin==this.xmax) || (this.ymin==this.ymax))
         return Promise.resolve(false);

      if (AxisPos === undefined) AxisPos = 0;

      let layer = this.svg_frame().select(".axis_layer"),
          w = this.frame_width(),
          h = this.frame_height(),
          pad = this.root_pad();

      this.x_handle.invert_side = (AxisPos >= 10);
      this.x_handle.lbls_both_sides = !this.x_handle.invert_side && (pad.fTickx > 1); // labels on both sides
      this.x_handle.has_obstacle = has_x_obstacle;

      this.y_handle.invert_side = ((AxisPos % 10) === 1);
      this.y_handle.lbls_both_sides = !this.y_handle.invert_side && (pad.fTicky > 1); // labels on both sides

      let draw_horiz = this.swap_xy ? this.y_handle : this.x_handle,
          draw_vertical = this.swap_xy ? this.x_handle : this.y_handle;

      if (!disable_axis_draw) {
         let pp = this.pad_painter();
         if (pp && pp._fast_drawing) disable_axis_draw = true;
      }

      if (!disable_axis_draw) {
         let promise1 = draw_horiz.DrawAxis(layer, w, h,
                                            draw_horiz.invert_side ? undefined : "translate(0," + h + ")",
                                            pad.fTickx ? -h : 0, disable_axis_draw);

         let promise2 = draw_vertical.DrawAxis(layer, w, h,
                                               draw_vertical.invert_side ? "translate(" + w + ",0)" : undefined,
                                               pad.fTicky ? w : 0, disable_axis_draw,
                                               draw_vertical.invert_side ? 0 : this.frame_x());

         return Promise.all([promise1, promise2]).then(() => {
            this.DrawGrids();

            if (!shrink_forbidden && JSROOT.settings.CanAdjustFrame) {

               let shrink = 0., ypos = draw_vertical.position;

               if ((-0.2*w < ypos) && (ypos < 0)) {
                  shrink = -ypos/w + 0.001;
                  this.shrink_frame_left += shrink;
               } else if ((ypos>0) && (ypos<0.3*w) && (this.shrink_frame_left > 0) && (ypos/w > this.shrink_frame_left)) {
                  shrink = -this.shrink_frame_left;
                  this.shrink_frame_left = 0.;
               }

               if (shrink != 0) {
                  this.Shrink(shrink, 0);
                  this.Redraw();
                  return this.DrawAxes(true);
               }

               this.axes_drawn = true;
               return true; // finished
            }
         });
      }

      this.axes_drawn = true;
      return Promise.resolve(true);
   }

   /** @summary Update frame attributes
     * @private */
   TFramePainter.prototype.UpdateAttributes = function(force) {
      let pad = this.root_pad(),
          tframe = this.GetObject();

      if ((this.fX1NDC === undefined) || (force && !this.modified_NDC)) {
         if (!pad || (pad.fLeftMargin===undefined)) {
            JSROOT.extend(this, JSROOT.settings.FrameNDC);
         } else {
            JSROOT.extend(this, {
               fX1NDC: pad.fLeftMargin,
               fX2NDC: 1 - pad.fRightMargin,
               fY1NDC: pad.fBottomMargin,
               fY2NDC: 1 - pad.fTopMargin
            });
         }
      }

      if (this.fillatt === undefined) {
         if (tframe)
            this.createAttFill({ attr: tframe });
         else if (pad && pad.fFrameFillColor)
            this.createAttFill({ pattern: pad.fFrameFillStyle, color: pad.fFrameFillColor });
         else if (pad)
            this.createAttFill({ attr: pad });
         else
            this.createAttFill({ pattern: 1001, color: 0 });

         // force white color for the canvas frame
         if (!tframe && this.fillatt.empty() && this.pad_painter() && this.pad_painter().iscan)
            this.fillatt.SetSolidColor('white');
      }

      if (!tframe && pad && (pad.fFrameLineColor!==undefined))
         this.createAttLine({ color: pad.fFrameLineColor, width: pad.fFrameLineWidth, style: pad.fFrameLineStyle });
      else
         this.createAttLine({ attr: tframe, color: 'black' });
   }

   /** @summary Function called at the end of resize of frame
     * @desc One should apply changes to the pad
     * @private */
   TFramePainter.prototype.SizeChanged = function() {

      let pad = this.root_pad();

      if (pad) {
         pad.fLeftMargin = this.fX1NDC;
         pad.fRightMargin = 1 - this.fX2NDC;
         pad.fBottomMargin = this.fY1NDC;
         pad.fTopMargin = 1 - this.fY2NDC;
         this.SetRootPadRange(pad);
      }

      this.InteractiveRedraw("pad", "frame");
   }

    /** @summary Remove all kinds of X/Y function for axes transformation */
   TFramePainter.prototype.CleanXY = function() {
      delete this.grx;
      delete this.gry;
      delete this.grz;

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
   }

   TFramePainter.prototype.CleanAxesDrawings = function() {
      // remove all axes drawings
      if (this.x_handle) this.x_handle.RemoveDrawG();
      if (this.y_handle) this.y_handle.RemoveDrawG();
      if (this.z_handle) this.z_handle.RemoveDrawG();

      if (this.draw_g) {
         this.draw_g.select(".grid_layer").selectAll("*").remove();
         this.draw_g.select(".axis_layer").selectAll("*").remove();
      }
      this.axes_drawn = false;
   }

   /** @summary Returns frame rectangle plus extra info for hint display */
   TFramePainter.prototype.CleanFrameDrawings = function() {

      // cleanup all 3D drawings if any
      if (typeof this.Create3DScene === 'function')
         this.Create3DScene(-1);

      this.CleanAxesDrawings();
      this.CleanXY();

      this.ranges_set = false;

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

      this.xaxis = null;
      this.yaxis = null;
      this.zaxis = null;

      if (this.draw_g) {
         this.draw_g.selectAll("*").remove();
         this.draw_g.on("mousedown", null)
                    .on("dblclick", null)
                    .on("wheel", null)
                    .on("contextmenu", null)
                    .property('interactive_set', null);
         this.draw_g.remove();
      }

      this.draw_g = null;

      if (this.keys_handler) {
         window.removeEventListener('keydown', this.keys_handler, false);
         this.keys_handler = null;
      }
   }

   /** @summary Cleanup frame */
   TFramePainter.prototype.Cleanup = function() {
      this.CleanFrameDrawings();
      delete this._click_handler;
      delete this._dblclick_handler;

      JSROOT.ObjectPainter.prototype.Cleanup.call(this);
   }

   TFramePainter.prototype.Redraw = function(/* reason */) {
      let pp = this.pad_painter();
      if (pp) pp.frame_painter_ref = this; // keep direct reference to the frame painter

      // first update all attributes from objects
      this.UpdateAttributes();

      let width = this.pad_width(),
          height = this.pad_height(),
          lm = Math.round(width * this.fX1NDC),
          w = Math.round(width * (this.fX2NDC - this.fX1NDC)),
          tm = Math.round(height * (1 - this.fY2NDC)),
          h = Math.round(height * (this.fY2NDC - this.fY1NDC)),
          rotate = false, fixpos = false, trans = "translate(" + lm + "," + tm + ")";

      if (pp && pp.options) {
         if (pp.options.RotateFrame) rotate = true;
         if (pp.options.FixFrame) fixpos = true;
      }

      if (rotate) {
         trans += " rotate(-90) " + "translate(" + -h + ",0)";
         let d = w; w = h; h = d;
      }

      this._frame_x = lm;
      this._frame_y = tm;
      this._frame_width = w;
      this._frame_height = h;
      this._frame_rotate = rotate;
      this._frame_fixpos = fixpos;

      if (this.mode3d) return; // no need to create any elements in 3d mode

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
         top_rect.attr("pointer-events", "visibleFill"); // let process mouse events inside frame
         inter.FrameInteractive.assign(this);
         this.BasicInteractive();
      });
   }

   /** @summary Change log state of specified axis
     * @param {number} value - 0 (linear), 1 (log) or 2 (log2) */
   TFramePainter.prototype.ChangeLog = function(axis, value) {
      let pad = this.root_pad();
      if (!pad) return;

      let name = "fLog" + axis;

      // do not allow log scale for labels
      if (!pad[name]) {
         if (this.swap_xy && axis==="x") axis = "y"; else
         if (this.swap_xy && axis==="y") axis = "x";
         let handle = this[axis + "_handle"];
         if (handle && (handle.kind === "labels")) return;
      }

      if ((value == "toggle") || (value === undefined))
         value = pad[name] ? 0 : 1;

      // directly change attribute in the pad
      pad[name] = value;

      this.InteractiveRedraw("pad", "log"+axis);
   }

   /** @summary Toggle log state on the specified axis */
   TFramePainter.prototype.ToggleLog = function(axis) {
      this.ChangeLog(axis, "toggle");
   }

   /** @summary Fill context menu for the frame
     * @desc It could be appended to the histogram menus */
   TFramePainter.prototype.FillContextMenu = function(menu, kind, obj) {

      let main = this.main_painter(), pad = this.root_pad();

      if ((kind=="x") || (kind=="y") || (kind=="z")) {
         let faxis = obj || this[kind+'axis'];
         menu.add("header: " + kind.toUpperCase() + " axis");
         menu.add("Unzoom", this.Unzoom.bind(this, kind));
         menu.add("sub:SetLog "+kind);
         menu.addchk(pad["fLog" + kind] == 0, "linear", "0", arg => this.ChangeLog(kind, parseInt(arg)));
         menu.addchk(pad["fLog" + kind] == 1, "log", "1", arg => this.ChangeLog(kind, parseInt(arg)));
         menu.addchk(pad["fLog" + kind] == 2, "log2", "2", arg => this.ChangeLog(kind, parseInt(arg)));
         menu.add("endsub:");
         menu.addchk(faxis.TestBit(JSROOT.EAxisBits.kMoreLogLabels), "More log",
               function() { faxis.InvertBit(JSROOT.EAxisBits.kMoreLogLabels); this.RedrawPad(); });
         menu.addchk(faxis.TestBit(JSROOT.EAxisBits.kNoExponent), "No exponent",
               function() { faxis.InvertBit(JSROOT.EAxisBits.kNoExponent); this.RedrawPad(); });

         if ((kind === "z") && main && main.options && main.options.Zscale)
            if (typeof main.FillPaletteMenu == 'function') main.FillPaletteMenu(menu);

         if (faxis)
            menu.AddTAxisMenu(main || this, faxis, kind);
         return true;
      }

      let alone = menu.size() == 0;

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

      if (pad) {
         menu.addchk(pad.fLogx, "SetLogx", this.ToggleLog.bind(this,"x"));
         menu.addchk(pad.fLogy, "SetLogy", this.ToggleLog.bind(this,"y"));

         if (main && (typeof main.Dimension === 'function') && (main.Dimension() > 1))
            menu.addchk(pad.fLogz, "SetLogz", this.ToggleLog.bind(this,"z"));
         menu.add("separator");
      }

      menu.addchk(this.IsTooltipAllowed(), "Show tooltips", function() {
         this.SetTooltipAllowed("toggle");
      });
      menu.AddAttributesMenu(this, alone ? "" : "Frame ");
      menu.add("separator");
      menu.add("Save as frame.png", function() { this.pad_painter().SaveAs("png", 'frame', 'frame.png'); });
      menu.add("Save as frame.svg", function() { this.pad_painter().SaveAs("svg", 'frame', 'frame.svg'); });

      return true;
   }

   TFramePainter.prototype.FillWebObjectOptions = function(res) {
      res.fcust = "frame";
      res.fopt = [this.scale_xmin || 0, this.scale_ymin || 0, this.scale_xmax || 0, this.scale_ymax || 0];
      return res;
   }

   TFramePainter.prototype.GetFrameRect = function() {
      // returns frame rectangle plus extra info for hint display

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

   TFramePainter.prototype.ConfigureUserClickHandler = function(handler) {
      this._click_handler = handler && (typeof handler == 'function') ? handler : null;
   }

   TFramePainter.prototype.ConfigureUserDblclickHandler = function(handler) {
      this._dblclick_handler = handler && (typeof handler == 'function') ? handler : null;
   }

    /** @summary Function can be used for zooming into specified range
      * @desc if both limits for each axis 0 (like xmin==xmax==0), axis will be unzoomed */
   TFramePainter.prototype.Zoom = function(xmin, xmax, ymin, ymax, zmin, zmax) {

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
         if (zmin <= this.zmin) { zmin = this.zmin; cnt++; }
         if (zmax >= this.zmax) { zmax = this.zmax; cnt++; }
         if (cnt === 2) { zoom_z = false; unzoom_z = true; }
      } else {
         unzoom_z = (zmin === zmax) && (zmin === 0);
      }

      let changed = false;

      // first process zooming (if any)
      if (zoom_x || zoom_y || zoom_z)
         this.ForEachPainter(obj => {
            if (zoom_x && obj.CanZoomIn("x", xmin, xmax)) {
               this.zoom_xmin = xmin;
               this.zoom_xmax = xmax;
               changed = true;
               zoom_x = false;
            }
            if (zoom_y && obj.CanZoomIn("y", ymin, ymax)) {
               this.zoom_ymin = ymin;
               this.zoom_ymax = ymax;
               changed = true;
               zoom_y = false;
            }
            if (zoom_z && obj.CanZoomIn("z", zmin, zmax)) {
               this.zoom_zmin = zmin;
               this.zoom_zmax = zmax;
               changed = true;
               zoom_z = false;
            }
         });

      // and process unzoom, if any
      if (unzoom_x || unzoom_y || unzoom_z) {
         if (unzoom_x) {
            if (this.zoom_xmin !== this.zoom_xmax) changed = true;
            this.zoom_xmin = this.zoom_xmax = 0;
         }
         if (unzoom_y) {
            if (this.zoom_ymin !== this.zoom_ymax) changed = true;
            this.zoom_ymin = this.zoom_ymax = 0;
         }
         if (unzoom_z) {
            if (this.zoom_zmin !== this.zoom_zmax) changed = true;
            this.zoom_zmin = this.zoom_zmax = 0;
         }

      // than try to unzoom all overlapped objects
         if (!changed) {
            let pp = this.pad_painter();
            if (pp && pp.painters)
               pp.painters.forEach(painter => {
                  if (painter && (typeof painter.UnzoomUserRange == 'function'))
                     if (painter.UnzoomUserRange(unzoom_x, unzoom_y, unzoom_z)) changed = true;
            });
         }
      }

      if (changed)
         this.InteractiveRedraw("pad", "zoom");

      return changed;
   }

   /** @summary Checks if specified axes zoom */
   TFramePainter.prototype.IsAxisZoomed = function(axis) { return this['zoom_'+axis+'min'] !== this['zoom_'+axis+'max']; }

   TFramePainter.prototype.Unzoom = function(dox, doy, doz) {
      if (typeof dox === 'undefined') { dox = true; doy = true; doz = true; } else
      if (typeof dox === 'string') { doz = dox.indexOf("z")>=0; doy = dox.indexOf("y")>=0; dox = dox.indexOf("x")>=0; }

      let last = this.zoom_changed_interactive;

      if (dox || doy || doz) this.zoom_changed_interactive = 1;

      let changed = this.Zoom(dox ? 0 : undefined, dox ? 0 : undefined,
                              doy ? 0 : undefined, doy ? 0 : undefined,
                              doz ? 0 : undefined, doz ? 0 : undefined);

      // if unzooming has no effect, decrease counter
      if ((dox || doy || doz) && !changed)
         this.zoom_changed_interactive = (!isNaN(last) && (last>0)) ? last - 1 : 0;

      return changed;
   }

   /** @summary Convert graphical coordinate into axis value */
   TFramePainter.prototype.RevertAxis = function(axis, pnt) {
      let handle = this[axis+"_handle"];
      return handle ? handle.RevertPoint(pnt) : 0;
   }

   /** @summary Show axis status message
   * @desc method called normally when mouse enter main object element
   * @private */
   TFramePainter.prototype.ShowAxisStatus = function(axis_name, evnt) {

      let status_func = this.GetShowStatusFunc();

      if (typeof status_func != "function") return;

      let taxis = this.GetAxis(axis_name), hint_name = axis_name, hint_title = "TAxis",
          m = d3.pointer(evnt, this.svg_frame().node()), id = (axis_name=="x") ? 0 : 1;

      if (taxis) { hint_name = taxis.fName; hint_title = taxis.fTitle || ("TAxis object for " + axis_name); }
      if (this.swap_xy) id = 1-id;

      let axis_value = this.RevertAxis(axis_name, m[id]);

      status_func(hint_name, hint_title, axis_name + " : " + this.AxisAsText(axis_name, axis_value), m[0]+","+m[1]);
   }

   /** @summary Add interactive keys handlers
    * @private */
   TFramePainter.prototype.AddKeysHandler = function() {
      if (JSROOT.BatchMode) return;
      JSROOT.require(['interactive']).then(inter => {
         inter.FrameInteractive.assign(this);
         this.AddKeysHandler();
      });
   }

   /** @summary Add interactive functionality to the frame
    * @private */
   TFramePainter.prototype.AddInteractive = function() {
      if (JSROOT.BatchMode || (!JSROOT.settings.Zooming && !JSROOT.settings.ContextMenu))
         return Promise.resolve(false);

      return JSROOT.require(['interactive']).then(inter => {
         inter.FrameInteractive.assign(this);
         return this.AddInteractive();
      });
   }

   let drawFrame = (divid, obj, opt) => {
      let p = new TFramePainter(obj);
      if (opt == "3d") p.mode3d = true;
      p.SetDivId(divid, 2);
      p.Redraw();
      return p.DrawingReady();
   }

   // ===========================================================================

   /**
     * @summary Painter for TPad object
     *
     * @class
     * @memberof JSROOT
     * @param {object} pad - TPad object to draw
     * @param {boolean} iscan - if TCanvas object
     * @private
     */

   function TPadPainter(pad, iscan) {
      JSROOT.ObjectPainter.call(this, pad);
      this.pad = pad;
      this.iscan = iscan; // indicate if working with canvas
      this.this_pad_name = "";
      if (!this.iscan && (pad !== null) && ('fName' in pad)) {
         this.this_pad_name = pad.fName.replace(" ", "_"); // avoid empty symbol in pad name
         let regexp = new RegExp("^[A-Za-z][A-Za-z0-9_]*$");
         if (!regexp.test(this.this_pad_name)) this.this_pad_name = 'jsroot_pad_' + JSROOT.id_counter++;
      }
      this.painters = []; // complete list of all painters in the pad
      this.has_canvas = true;
   }

   TPadPainter.prototype = Object.create(JSROOT.ObjectPainter.prototype);

   /** @summary Returns SVG element for the specified pad (or itself)
    * @private */
   TPadPainter.prototype.svg_pad = function(pad_name) {
      if (pad_name === undefined)
         pad_name = this.this_pad_name;
      return JSROOT.ObjectPainter.prototype.svg_pad.call(this, pad_name);
   }

   /** @summary cleanup only pad itself, all child elements will be collected and cleanup separately */
   TPadPainter.prototype.Cleanup = function() {

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
      delete this.custom_palette;

      this.painters = [];
      this.pad = null;
      this.this_pad_name = undefined;
      this.has_canvas = false;

      jsrp.SelectActivePad({ pp: this, active: false });

      JSROOT.ObjectPainter.prototype.Cleanup.call(this);
   }

   /** @summary Returns frame painter inside the pad
     * @private */
   TPadPainter.prototype.frame_painter = function() {
      return this.frame_painter_ref;
   }

   /** @summary Cleanup primitives from pad - selector lets define which painters to remove */
   TPadPainter.prototype.CleanPrimitives = function(selector) {
      if (!selector || (typeof selector !== 'function')) return;

      for (let k = this.painters.length-1; k >= 0; --k)
         if (selector(this.painters[k])) {
            this.painters[k].Cleanup();
            this.painters.splice(k, 1);
         }
   }

   /** @summary Generates automatic color for some objects painters */
   TPadPainter.prototype.CreateAutoColor = function() {
      let pad = this.root_pad(),
          numprimitives = pad && pad.fPrimitves ? pad.fPrimitves.arr.length : 5;

      let pal = this.get_palette(true);

      let indx = this._auto_color || 0;
      this._auto_color = indx+1;

      if (pal) {
         if (numprimitives<2) numprimitives = 2;
         if (indx >= numprimitives) indx = numprimitives - 1;
         let palindx = Math.round(indx * (pal.getLength()-3) / (numprimitives-1));
         let colvalue = pal.getColor(palindx);
         let colindx = this.add_color(colvalue);
         return colindx;
      }

      this._auto_color = this._auto_color % 8;
      return indx+2;
   }

   /** @summary Call function for each painter in pad
     * @param {function} userfunc - function to call
     * @param {string} kind - "all" for all objects (default), "pads" only pads and subpads, "objects" only for object in current pad
     * @private */
   TPadPainter.prototype.ForEachPainterInPad = function(userfunc, kind) {
      if (!kind) kind = "all";
      if (kind!="objects") userfunc(this);
      for (let k = 0; k < this.painters.length; ++k) {
         let sub = this.painters[k];
         if (typeof sub.ForEachPainterInPad === 'function') {
            if (kind!="objects") sub.ForEachPainterInPad(userfunc, kind);
         } else if (kind != "pads") userfunc(sub);
      }
   }

   /** @summary register for pad events receiver
     * @desc in pad painter, while pad may be drawn without canvas
     * @private */
   TPadPainter.prototype.RegisterForPadEvents = function(receiver) {
      this.pad_events_receiver = receiver;
   }

   /** @summary Generate pad events, normally handled by GED
    * @desc in pad painter, while pad may be drawn without canvas
     * @private */
   TPadPainter.prototype.PadEvent = function(_what, _padpainter, _painter, _position, _place) {

      if ((_what == "select") && (typeof this.SelectActivePad == 'function'))
         this.SelectActivePad(_padpainter, _painter, _position);

      if (this.pad_events_receiver)
         this.pad_events_receiver({ what: _what, padpainter:  _padpainter, painter: _painter, position: _position, place: _place });
   }

   /** @summary method redirect call to pad events receiver */
   TPadPainter.prototype.SelectObjectPainter = function(_painter, pos, _place) {
      let istoppad = (this.iscan || !this.has_canvas),
          canp = istoppad ? this : this.canv_painter();

      if (_painter === undefined) _painter = this;

      if (pos && !istoppad)
         this.CalcAbsolutePosition(this.svg_pad(), pos);

      jsrp.SelectActivePad({ pp: this, active: true });

      if (canp) canp.PadEvent("select", this, _painter, pos, _place);
   }

   /** @summary Called by framework when pad is supposed to be active and get focus
    * @private */
   TPadPainter.prototype.SetActive = function(on) {
      let fp = this.frame_painter();
      if (fp && (typeof fp.SetActive == 'function')) fp.SetActive(on);
   }

   /** @summary Draw pad active border
    * @private */
   TPadPainter.prototype.DrawActiveBorder = function(svg_rect, is_active) {
      if (is_active !== undefined) {
         if (this.is_active_pad === is_active) return;
         this.is_active_pad = is_active;
      }

      if (this.is_active_pad === undefined) return;

      if (!svg_rect)
         svg_rect = this.iscan ? this.svg_canvas().select(".canvas_fillrect") :
                                 this.svg_pad().select(".root_pad_border");

      let lineatt = this.is_active_pad ? new JSROOT.TAttLineHandler({ style: 1, width: 1, color: "red" }) : this.lineatt;

      if (!lineatt) lineatt = new JSROOT.TAttLineHandler({ color: "none" });

      svg_rect.call(lineatt.func);
   }

   TPadPainter.prototype.CreateCanvasSvg = function(check_resize, new_size) {

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

         if (JSROOT.BatchMode) {
            svg.attr("xmlns", "http://www.w3.org/2000/svg");
            svg.attr("xmlns:xlink", "http://www.w3.org/1999/xlink");
         }

         svg.append("svg:title").text("ROOT canvas");
         let frect = svg.append("svg:rect").attr("class","canvas_fillrect")
                               .attr("x",0).attr("y",0);
         if (!JSROOT.BatchMode)
            frect.style("pointer-events", "visibleFill")
                 .on("dblclick", this.EnlargePad.bind(this))
                 .on("click", () => this.SelectObjectPainter())
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
         if (this.pad && this.pad.fCw && this.pad.fCh && (this.pad.fCw > 0)) {
            factor = this.pad.fCh / this.pad.fCw;
            if ((factor < 0.1) || (factor > 10)) factor = 0.66;
         }

         if (this._fixed_size) {
            render_to.style("overflow","auto");
            rect = { width: this.pad.fCw, height: this.pad.fCh };
            if (!rect.width || !rect.height)
               rect = this.get_visible_rect(render_to);
         } else {
            rect = this.check_main_resize(2, new_size, factor);
         }
      }

      this.createAttFill({ attr: this.pad });

      if ((rect.width <= lmt) || (rect.height <= lmt)) {
         svg.style("display", "none");
         console.warn("Hide canvas while geometry too small w=" + rect.width + " h=" + rect.height);
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

      // console.log('CANVAS SVG width = ' + rect.width + " height = " + rect.height);

      svg.attr("viewBox", "0 0 " + rect.width + " " + rect.height)
         .attr("preserveAspectRatio", "none")  // we do not preserve relative ratio
         .property('height_factor', factor)
         .property('draw_x', 0)
         .property('draw_y', 0)
         .property('draw_width', rect.width)
         .property('draw_height', rect.height);

      let fill_rect = svg.select(".canvas_fillrect")
         .attr("width", rect.width)
         .attr("height", rect.height)
         .call(this.fillatt.func);

      this._fast_drawing = JSROOT.settings.SmallPad && ((rect.width < JSROOT.settings.SmallPad.width) || (rect.height < JSROOT.settings.SmallPad.height));

      this.DrawActiveBorder(fill_rect);

      if (this.AlignBtns && btns)
         this.AlignBtns(btns, rect.width, rect.height);

      return true;
   }

   TPadPainter.prototype.EnlargePad = function(evnt) {

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

   /** @summary Create main SVG element for pad
     * @returns true when pad is displayed and all its items should be redrawn */
   TPadPainter.prototype.CreatePadSvg = function(only_resize) {

      if (!this.has_canvas) {
         this.CreateCanvasSvg(only_resize ? 2 : 0);
         return true;
      }

      let svg_can = this.svg_canvas(),
          width = svg_can.property("draw_width"),
          height = svg_can.property("draw_height"),
          pad_enlarged = svg_can.property("pad_enlarged"),
          pad_visible = !pad_enlarged || (pad_enlarged === this.pad),
          w = Math.round(this.pad.fAbsWNDC * width),
          h = Math.round(this.pad.fAbsHNDC * height),
          x = Math.round(this.pad.fAbsXlowNDC * width),
          y = Math.round(height * (1 - this.pad.fAbsYlowNDC)) - h,
          svg_pad = null, svg_rect = null, btns = null;

      if (pad_enlarged === this.pad) { w = width; h = height; x = y = 0; }

      if (only_resize) {
         svg_pad = this.svg_pad();
         svg_rect = svg_pad.select(".root_pad_border");
         if (!JSROOT.BatchMode)
            btns = this.svg_layer("btns_layer");
      } else {
         svg_pad = svg_can.select(".primitives_layer")
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
                    .on("click", () => this.SelectObjectPainter())
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

      this.DrawActiveBorder(svg_rect);

      this._fast_drawing = JSROOT.settings.SmallPad && ((w < JSROOT.settings.SmallPad.width) || (h < JSROOT.settings.SmallPad.height));

       // special case of 3D canvas overlay
      if (svg_pad.property('can3d') === JSROOT.constants.Embed3D.Overlay)
          this.select_main().select(".draw3d_" + this.this_pad_name)
              .style('display', pad_visible ? '' : 'none');

      if (this.AlignBtns && btns)
         this.AlignBtns(btns, w, h);

      return pad_visible;
   }

   TPadPainter.prototype.CheckSpecial = function(obj) {

      if (!obj) return false;

      if (obj._typename == "TStyle") {
         JSROOT.extend(JSROOT.gStyle, obj);
         return true;
      }

      if ((obj._typename == "TObjArray") && (obj.name == "ListOfColors")) {

         if (this.options && this.options.CreatePalette) {
            let arr = [];
            for (let n = obj.arr.length - this.options.CreatePalette; n<obj.arr.length; ++n) {
               let col = jsrp.getRGBfromTColor(obj.arr[n]);
               if (!col) { console.log('Fail to create color for palette'); arr = null; break; }
               arr.push(col);
            }
            if (arr) this.custom_palette = new JSROOT.ColorPalette(arr);
         }

         if (!this.options || this.options.GlobalColors) // set global list of colors
            jsrp.adoptRootColors(obj);

         // copy existing colors and extend with new values
         if (this.options && this.options.LocalColors)
            this.root_colors = jsrp.extendRootColors(null, obj);
         return true;
      }

      if ((obj._typename == "TObjArray") && (obj.name == "CurrentColorPalette")) {
         let arr = [], missing = false;
         for (let n = 0; n < obj.arr.length; ++n) {
            let col = obj.arr[n];
            if (col && (col._typename == 'TColor')) {
               arr[n] = jsrp.getRGBfromTColor(col);
            } else {
               console.log('Missing color with index ' + n); missing = true;
            }
         }
         if (!this.options || (!missing && !this.options.IgnorePalette))
            this.custom_palette = new JSROOT.ColorPalette(arr);
         return true;
      }

      return false;
   }

   TPadPainter.prototype.CheckSpecialsInPrimitives = function(can) {
      let lst = can ? can.fPrimitives : null;
      if (!lst) return;
      for (let i = 0; i < lst.arr.length; ++i) {
         if (this.CheckSpecial(lst.arr[i])) {
            lst.arr.splice(i,1);
            lst.opt.splice(i,1);
            i--;
         }
      }
   }

   TPadPainter.prototype.RemovePrimitive = function(obj) {
      if (!this.pad || !this.pad.fPrimitives) return;
      let indx = this.pad.fPrimitives.arr.indexOf(obj);
      if (indx>=0) this.pad.fPrimitives.RemoveAt(indx);
   }

   TPadPainter.prototype.FindPrimitive = function(exact_obj, classname, name) {
      if (!this.pad || !this.pad.fPrimitives) return null;

      for (let i=0; i < this.pad.fPrimitives.arr.length; i++) {
         let obj = this.pad.fPrimitives.arr[i];

         if ((exact_obj !== null) && (obj !== exact_obj)) continue;

         if ((classname !== undefined) && (classname !== null))
            if (obj._typename !== classname) continue;

         if ((name !== undefined) && (name !== null))
            if (obj.fName !== name) continue;

         return obj;
      }

      return null;
   }

   /** Return true if any objects beside sub-pads exists in the pad */
   TPadPainter.prototype.HasObjectsToDraw = function() {

      if (!this.pad || !this.pad.fPrimitives) return false;

      for (let n=0;n<this.pad.fPrimitives.arr.length;++n)
         if (this.pad.fPrimitives.arr[n] && this.pad.fPrimitives.arr[n]._typename != "TPad") return true;

      return false;
   }

   TPadPainter.prototype.DrawPrimitives = function(indx, callback, ppainter) {

      if (indx===0) {
         // flag used to prevent immediate pad redraw during normal drawing sequence
         this._doing_pad_draw = true;

         if (this.iscan)
            this._start_tm = this._lasttm_tm = new Date().getTime();

         // set number of primitves
         this._num_primitives = this.pad && this.pad.fPrimitives ? this.pad.fPrimitives.arr.length : 0;
      }

      if (ppainter && (typeof ppainter == 'object'))
         ppainter._primitive = true; // mark painter as belonging to primitives

      if (indx >= this._num_primitives) {
         delete this._doing_pad_draw;
         if (this._start_tm) {
            let spenttm = new Date().getTime() - this._start_tm;
            if (spenttm > 1000) console.log("Canvas drawing took " + (spenttm*1e-3).toFixed(2) + "s");
            delete this._start_tm;
            delete this._lasttm_tm;
         }

         return JSROOT.callBack(callback);
      }

      // use of Promise should avoid large call-stack depth when many primitives are drawn
      let handle_func = this.DrawPrimitives.bind(this, indx+1, callback);

      JSROOT.draw(this.divid, this.pad.fPrimitives.arr[indx], this.pad.fPrimitives.opt[indx]).then(handle_func);
   }

   TPadPainter.prototype.GetTooltips = function(pnt) {
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

   TPadPainter.prototype.FillContextMenu = function(menu) {

      if (this.pad)
         menu.add("header: " + this.pad._typename + "::" + this.pad.fName);
      else
         menu.add("header: Canvas");

      menu.addchk(this.IsTooltipAllowed(), "Show tooltips", this.SetTooltipAllowed.bind(this, "toggle"));

      if (!this._websocket) {

         function SetPadField(arg) {
            this.pad[arg.substr(1)] = parseInt(arg[0]);
            this.InteractiveRedraw("axes", arg.substr(1));
         }

         menu.addchk(this.pad.fGridx, 'Grid x', (this.pad.fGridx ? '0' : '1') + 'fGridx', SetPadField);
         menu.addchk(this.pad.fGridy, 'Grid y', (this.pad.fGridy ? '0' : '1') + 'fGridy', SetPadField);
         menu.add("sub:Ticks x");
         menu.addchk(this.pad.fTickx == 0, "normal", "0fTickx", SetPadField);
         menu.addchk(this.pad.fTickx == 1, "ticks on both sides", "1fTickx", SetPadField);
         menu.addchk(this.pad.fTickx == 2, "labels on both sides", "2fTickx", SetPadField);
         menu.add("endsub:");
         menu.add("sub:Ticks y");
         menu.addchk(this.pad.fTicky == 0, "normal", "0fTicky", SetPadField);
         menu.addchk(this.pad.fTicky == 1, "ticks on both sides", "1fTicky", SetPadField);
         menu.addchk(this.pad.fTicky == 2, "labels on both sides", "2fTicky", SetPadField);
         menu.add("endsub:");

         menu.AddAttributesMenu(this);
      }

      menu.add("separator");

      if (this.ActivateStatusBar)
         menu.addchk(this.HasEventStatus(), "Event status", this.ActivateStatusBar.bind(this, 'toggle'));

      if (this.enlarge_main() || (this.has_canvas && this.HasObjectsToDraw()))
         menu.addchk((this.enlarge_main('state')=='on'), "Enlarge " + (this.iscan ? "canvas" : "pad"), this.EnlargePad.bind(this, null));

      let fname = this.this_pad_name;
      if (fname.length===0) fname = this.iscan ? "canvas" : "pad";

      menu.add("Save as "+ fname+".png", fname+".png", this.SaveAs.bind(this, "png", false));
      menu.add("Save as "+ fname+".svg", fname+".svg", this.SaveAs.bind(this, "svg", false));

      return true;
   }

   TPadPainter.prototype.PadContextMenu = function(evnt) {

      if (evnt.stopPropagation) { // this is normal event processing and not emulated jsroot event

         // for debug purposes keep original context menu for small region in top-left corner
         let pos = d3.pointer(evnt, this.svg_pad().node());

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

   /** @summary redraw pad */
   TPadPainter.prototype.Redraw = function(reason) {

      // prevent redrawing
      if (this._doing_pad_draw)
         return console.log('Prevent redrawing', this.pad.fName);

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

      if (jsrp.GetActivePad() === this) {
         let canp = this.canv_painter();
         if (canp) canp.PadEvent("padredraw", this );
      }
   }

   TPadPainter.prototype.NumDrawnSubpads = function() {
      if (!this.painters) return 0;

      let num = 0;

      for (let i = 0; i < this.painters.length; ++i)
         if (this.painters[i] instanceof TPadPainter)
            num++;

      return num;
   }

   TPadPainter.prototype.RedrawByResize = function() {
      let elem = this.svg_pad();
      if (!elem.empty() && elem.property('can3d') === JSROOT.constants.Embed3D.Overlay) return true;

      for (let i = 0; i < this.painters.length; ++i)
         if (typeof this.painters[i].RedrawByResize === 'function')
            if (this.painters[i].RedrawByResize()) return true;

      return false;
   }

   TPadPainter.prototype.CheckCanvasResize = function(size, force) {

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

   TPadPainter.prototype.UpdateObject = function(obj) {
      if (!obj) return false;

      this.pad.fBits = obj.fBits;
      this.pad.fTitle = obj.fTitle;

      this.pad.fGridx = obj.fGridx;
      this.pad.fGridy = obj.fGridy;
      this.pad.fTickx = obj.fTickx;
      this.pad.fTicky = obj.fTicky;
      this.pad.fLogx  = obj.fLogx;
      this.pad.fLogy  = obj.fLogy;
      this.pad.fLogz  = obj.fLogz;

      this.pad.fUxmin = obj.fUxmin;
      this.pad.fUxmax = obj.fUxmax;
      this.pad.fUymin = obj.fUymin;
      this.pad.fUymax = obj.fUymax;

      this.pad.fX1 = obj.fX1;
      this.pad.fX2 = obj.fX2;
      this.pad.fY1 = obj.fY1;
      this.pad.fY2 = obj.fY2;

      this.pad.fLeftMargin   = obj.fLeftMargin;
      this.pad.fRightMargin  = obj.fRightMargin;
      this.pad.fBottomMargin = obj.fBottomMargin
      this.pad.fTopMargin    = obj.fTopMargin;

      this.pad.fFillColor = obj.fFillColor;
      this.pad.fFillStyle = obj.fFillStyle;
      this.pad.fLineColor = obj.fLineColor;
      this.pad.fLineStyle = obj.fLineStyle;
      this.pad.fLineWidth = obj.fLineWidth;

      this.pad.fPhi = obj.fPhi;
      this.pad.fTheta = obj.fTheta;

      if (this.iscan) this.CheckSpecialsInPrimitives(obj);

      let fp = this.frame_painter();
      if (fp) fp.UpdateAttributes(!fp.modified_NDC);

      if (!obj.fPrimitives) return false;

      let isany = false, p = 0;
      for (let n = 0; n < obj.fPrimitives.arr.length; ++n) {
         while (p < this.painters.length) {
            let pp = this.painters[p++];
            if (!pp._primitive) continue;
            if (pp.UpdateObject(obj.fPrimitives.arr[n])) isany = true;
            break;
         }
      }

      return isany;
   }

   /** Function called when drawing next snapshot from the list
     * it is also used as callback for drawing of previous snap */
   TPadPainter.prototype.DrawNextSnap = function(lst, indx, call_back, objpainter) {

      if (indx === -1) {
         // flag used to prevent immediate pad redraw during first draw
         this._doing_pad_draw = true;
         this._snaps_map = {}; // to control how much snaps are drawn
         this._num_primitives = lst ? lst.length : 0;
      }

      while (true) {

         if (objpainter && lst && lst[indx] && (objpainter.snapid === undefined)) {
            // keep snap id in painter, will be used for the
            let pi = this.painters.indexOf(objpainter);
            if (pi<0) this.painters.push(objpainter);
            objpainter.snapid = lst[indx].fObjectID;
            if (objpainter.$primary && (pi > 0) && this.painters[pi-1].$secondary) {
               this.painters[pi-1].snapid = objpainter.snapid + "#hist";
               console.log('ASSIGN SECONDARY HIST ID', this.painters[pi-1].snapid);
            }
         }

         objpainter = null;

         ++indx; // change to the next snap

         if (!lst || (indx >= lst.length)) {
            delete this._doing_pad_draw;
            delete this._snaps_map;
            return JSROOT.callBack(call_back, this);
         }

         let snap = lst[indx],
             snapid = snap.fObjectID,
             cnt = this._snaps_map[snapid];

         if (cnt) cnt++; else cnt = 1;
         this._snaps_map[snapid] = cnt; // check how many objects with same snapid drawn, use them again

         // first appropriate painter for the object
         // if same object drawn twice, two painters will exists
         for (let k=0; k<this.painters.length; ++k) {
            if (this.painters[k].snapid === snapid)
               if (--cnt === 0) { objpainter = this.painters[k]; break; }
         }

         // function which should be called when drawing of next item finished
         let draw_callback = this.DrawNextSnap.bind(this, lst, indx, call_back);

         if (objpainter) {

            if (snap.fKind === webSnapIds.kObject) { // object itself
               if (objpainter.UpdateObject(snap.fSnapshot, snap.fOption)) objpainter.Redraw();
               continue; // call next
            }

            if (snap.fKind === webSnapIds.kSVG) { // update SVG
               if (objpainter.UpdateObject(snap.fSnapshot)) objpainter.Redraw();
               continue; // call next
            }

            if (snap.fKind === webSnapIds.kSubPad) { // subpad
               return objpainter.RedrawPadSnap(snap, draw_callback);
            }

            continue; // call next
         }

         // gStyle object
         if (snap.fKind === webSnapIds.kStyle) {
            JSROOT.extend(JSROOT.gStyle, snap.fSnapshot);
            continue;
         }

         // list of colors
         if (snap.fKind === webSnapIds.kColors) {

            let ListOfColors = [], arr = snap.fSnapshot.fOper.split(";");
            for (let n = 0; n < arr.length; ++n) {
               let name = arr[n], p = name.indexOf(":");
               if (p > 0) {
                  ListOfColors[parseInt(name.substr(0,p))] = "rgb(" + name.substr(p+1) + ")";
               } else {
                  p = name.indexOf("=");
                  ListOfColors[parseInt(name.substr(0,p))] = "rgba(" + name.substr(p+1) + ")";
               }
            }

            // set global list of colors
            if (!this.options || this.options.GlobalColors)
               jsrp.adoptRootColors(ListOfColors);

            // copy existing colors and extend with new values
            if (this.options && this.options.LocalColors)
               this.root_colors = jsrp.extendRootColors(null, ListOfColors);

            // set palette
            if (snap.fSnapshot.fBuf && (!this.options || !this.options.IgnorePalette)) {
               let palette = [];
               for (let n=0;n<snap.fSnapshot.fBuf.length;++n)
                  palette[n] = ListOfColors[Math.round(snap.fSnapshot.fBuf[n])];

               this.custom_palette = new JSROOT.ColorPalette(palette);
            }

            continue;
         }

         if (snap.fKind === webSnapIds.kSubPad) { // subpad

            let subpad = snap.fSnapshot;

            subpad.fPrimitives = null; // clear primitives, they just because of I/O

            let padpainter = new TPadPainter(subpad, false);
            padpainter.DecodeOptions(snap.fOption);
            padpainter.SetDivId(this.divid); // pad painter will be registered in the canvas painters list
            padpainter.snapid = snap.fObjectID;

            padpainter.CreatePadSvg();

            if (padpainter.MatchObjectType("TPad") && snap.fPrimitives.length > 0)
               padpainter.AddPadButtons();

            // we select current pad, where all drawing is performed
            let prev_name = padpainter.CurrentPadName(padpainter.this_pad_name);
            padpainter.DrawNextSnap(snap.fPrimitives, -1, () => {
               padpainter.CurrentPadName(prev_name);
               draw_callback(padpainter);
            });
            return;
         }

         // here the case of normal drawing, will be handled in promise
         if ((snap.fKind === webSnapIds.kObject) || (snap.fKind === webSnapIds.kSVG))
            return JSROOT.draw(this.divid, snap.fSnapshot, snap.fOption).then(draw_callback);
      }
   }

   TPadPainter.prototype.FindSnap = function(snapid) {

      if (this.snapid === snapid) return this;

      if (!this.painters) return null;

      for (let k=0;k<this.painters.length;++k) {
         let sub = this.painters[k];

         if (typeof sub.FindSnap === 'function') sub = sub.FindSnap(snapid);
         else if (sub.snapid !== snapid) sub = null;

         if (sub) return sub;
      }

      return null;
   }

   TPadPainter.prototype.RedrawPadSnap = function(snap, call_back) {
      // for the canvas snapshot contains list of objects
      // as first entry, graphical properties of canvas itself is provided
      // in ROOT6 it also includes primitives, but we ignore them

      if (!snap || !snap.fPrimitives) return;

      this.is_active_pad = !!snap.fActive; // enforce boolean flag
      this._readonly = (snap.fReadOnly === undefined) ? true : snap.fReadOnly; // readonly flag

      let first = snap.fSnapshot;
      first.fPrimitives = null; // primitives are not interesting, they are disabled in IO

      if (this.snapid === undefined) {
         // first time getting snap, create all gui elements first

         this.snapid = snap.fObjectID;

         this.draw_object = first;
         this.pad = first;
         // this._fixed_size = true;

         // if canvas size not specified in batch mode, temporary use 900x700 size
         if (this.batch_mode && (!first.fCw || !first.fCh)) { first.fCw = 900; first.fCh = 700; }

         // case of ROOT7 with always dummy TPad as first entry
         if (!first.fCw || !first.fCh) this._fixed_size = false;

         if (JSROOT.BrowserLayout && !this.batch_mode && !this.use_openui && !this.brlayout) {
            let mainid = this.divid;
            if (mainid && (typeof mainid == 'object'))
               mainid = d3.select(mainid).attr("id");
            if (mainid && (typeof mainid == "string")) {
               this.brlayout = new JSROOT.BrowserLayout(mainid, null, this);
               this.brlayout.Create(mainid, true);
               // this.brlayout.ToggleBrowserKind("float");
               this.SetDivId(this.brlayout.drawing_divid(), -1);  // assign id for drawing
               JSROOT.RegisterForResize(this.brlayout);
            }
         }

         this.CreateCanvasSvg(0);
         this.SetDivId(this.divid);  // now add to painters list
         if (!this.batch_mode)
            this.AddPadButtons(true);

         if (snap.fScripts && (typeof snap.fScripts == "string")) {
            let arg = "";

            if (snap.fScripts.indexOf("load:") == 0) arg = snap.fScripts; else
            if (snap.fScripts.indexOf("assert:") == 0) arg = snap.fScripts.substr(7);
            if (arg) {
               JSROOT.require(arg).then(this.DrawNextSnap.bind(this, snap.fPrimitives, -1, call_back));
            } else {
               console.log('Calling eval ' + snap.fScripts.length);
               eval(snap.fScripts);
               console.log('Calling eval done');
               this.DrawNextSnap(snap.fPrimitives, -1, call_back);
            }
         } else {
            this.DrawNextSnap(snap.fPrimitives, -1, call_back);
         }

         return;
      }

      this.UpdateObject(first); // update only object attributes

      // apply all changes in the object (pad or canvas)
      if (this.iscan) {
         this.CreateCanvasSvg(2);
      } else {
         this.CreatePadSvg(true);
      }

      let isanyfound = false, isanyremove = false;

      // check if frame or title was recreated, we could reassign handlers for them directly

      function MatchPrimitive(painters, primitives, class_name, obj_name) {
         let painter, primitive;
         for (let k=0;k<painters.length;++k) {
            if (painters[k].snapid === undefined) continue;
            if (!painters[k].MatchObjectType(class_name)) continue;
            if (obj_name && (!painters[k].GetObject() || (painters[k].GetObject().fName !== obj_name))) continue;
            painter = painters[k];
            break;
         }
         if (!painter) return;
         for (let k=0;k<primitives.length;++k) {
            if ((primitives[k].fKind !== 1) || !primitives[k].fSnapshot || (primitives[k].fSnapshot._typename !== class_name)) continue;
            if (obj_name && (primitives[k].fSnapshot.fName !== obj_name)) continue;
            primitive = primitives[k];
            break;
         }
         if (!primitive) return;

         // force painter to use new object id
         if (painter.snapid !== primitive.fObjectID)
            painter.snapid = primitive.fObjectID;
      }

      // while this is temporary objects, which can be recreated very often, try to catch such situation ourselfs
      MatchPrimitive(this.painters, snap.fPrimitives, "TFrame");
      MatchPrimitive(this.painters, snap.fPrimitives, "TPaveText", "title");

      // find and remove painters which no longer exists in the list
      for (let k=0;k<this.painters.length;++k) {
         let sub = this.painters[k];
         if ((sub.snapid===undefined) || sub.$secondary) continue; // look only for painters with snapid

         for (let i=0;i<snap.fPrimitives.length;++i)
            if (snap.fPrimitives[i].fObjectID === sub.snapid) { sub = null; isanyfound = true; break; }

         if (sub) {
            console.log('Remove painter' + k + ' from ' + this.painters.length + ' ' + sub.GetObject()._typename);
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
         let svg_p = this.svg_pad(),
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
         if (jsrp.GetActivePad() === this) {
            let canp = this.canv_painter();

            if (canp) canp.PadEvent("padredraw", this);
         }

         JSROOT.callBack(call_back, this);
      });
   }

   /** @summary Create image for the pad
     * @returns {Promise} with image data, coded with btoa() function */
   TPadPainter.prototype.CreateImage = function(format) {
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

   /** Collects pad information for TWebCanvas, need to update different states */
   TPadPainter.prototype.GetWebPadOptions = function(arg) {
      let is_top = (arg === undefined), elem = null, scan_subpads = true;
      // no any options need to be collected in readonly mode
      if (is_top && this._readonly) return "";
      if (arg === "only_this") { is_top = true; scan_subpads = false; }
      if (is_top) arg = [];

      if (this.snapid) {
         elem = { _typename: "TWebPadOptions", snapid: this.snapid.toString(),
                  active: !!this.is_active_pad,
                  bits: 0, primitives: [],
                  logx: this.pad.fLogx, logy: this.pad.fLogy, logz: this.pad.fLogz,
                  gridx: this.pad.fGridx, gridy: this.pad.fGridy,
                  tickx: this.pad.fTickx, ticky: this.pad.fTicky,
                  mleft: this.pad.fLeftMargin, mright: this.pad.fRightMargin,
                  mtop: this.pad.fTopMargin, mbottom: this.pad.fBottomMargin,
                  zx1:0, zx2:0, zy1:0, zy2:0, zz1:0, zz2:0 };

         if (this.iscan) elem.bits = this.GetStatusBits();

         if (this.GetPadRanges(elem))
            arg.push(elem);
         else
            console.log('fail to get ranges for pad ' +  this.pad.fName);
      }

      for (let k=0; k<this.painters.length; ++k) {
         let sub = this.painters[k];
         if (typeof sub.GetWebPadOptions == "function") {
            if (scan_subpads) sub.GetWebPadOptions(arg);
         } else if (sub.snapid) {
            let opt = { _typename: "TWebObjectOptions", snapid: sub.snapid.toString(), opt: sub.OptionsAsString(), fcust: "", fopt: [] };
            if (typeof sub.FillWebObjectOptions == "function")
               opt = sub.FillWebObjectOptions(opt);
            elem.primitives.push(opt);
         }
      }

      if (is_top) return JSROOT.toJSON(arg);
   }

   /** @summary returns actual ranges in the pad, which can be applied to the server
     * @private */
   TPadPainter.prototype.GetPadRanges = function(r) {

      if (!r) return false;

      let main = this.frame_painter(),
          p = this.svg_pad();

      r.ranges = main && main.ranges_set ? true : false; // indicate that ranges are assigned

      r.ux1 = r.px1 = r.ranges ? main.scale_xmin : 0; // need to initialize for JSON reader
      r.uy1 = r.py1 = r.ranges ? main.scale_ymin : 0;
      r.ux2 = r.px2 = r.ranges ? main.scale_xmax : 0;
      r.uy2 = r.py2 = r.ranges ? main.scale_ymax : 0;

      if (main) {
         if (main.zoom_xmin !== main.zoom_xmax) {
            r.zx1 = main.zoom_xmin; r.zx2 = main.zoom_xmax;
         }

         if (main.zoom_ymin !== main.zoom_ymax) {
            r.zy1 = main.zoom_ymin; r.zy2 = main.zoom_ymax;
         }

         if (main.zoom_zmin !== main.zoom_zmax) {
            r.zz1 = main.zoom_zmin; r.zz2 = main.zoom_zmax;
         }
      }

      if (!r.ranges || p.empty()) return true;

      // calculate user range for full pad
      let same = x => x,
          direct_funcs = [same, Math.log10, x => Math.log10(x)/Math.log10(2) ],
          revert_funcs = [same, x => Math.pow(10, x), x => Math.pow(2, x)],
          func = direct_funcs[main.logx],
          func2 = revert_funcs[main.logx],
          k = (func(main.scale_xmax) - func(main.scale_xmin))/p.property("draw_width"),
          x1 = func(main.scale_xmin) - k*p.property("draw_x"),
          x2 = x1 + k*p.property("draw_width");

       // method checks if new value v1 close to the old value v0
       function match(v1, v0, range) {
          return (Math.abs(v0-v1)<Math.abs(range)*1e-10) ? v0 : v1;
       }

      r.ux1 = match( func2(x1), r.ux1, r.px2-r.px1);
      r.ux2 = match( func2(x2), r.ux2, r.px2-r.px1);

      func = direct_funcs[main.logy];
      func2 = revert_funcs[main.logy];

      k = (func(main.scale_ymax) - func(main.scale_ymin))/p.property("draw_height");
      let y2 = func(main.scale_ymax) + k*p.property("draw_y"),
          y1 = y2 - k*p.property("draw_height");

      r.uy1 = match( func2(y1), r.uy1, r.py2-r.py1);
      r.uy2 = match( func2(y2), r.uy2, r.py2-r.py1);

      return true;
   }

   TPadPainter.prototype.ItemContextMenu = function(name) {
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
             selp = this.frame_painter();
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
          if (selp.FillContextMenu(menu, selkind))
             setTimeout(menu.show.bind(menu), 50);
       });
   }

   TPadPainter.prototype.SaveAs = function(kind, full_canvas, filename) {
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
   TPadPainter.prototype.ProduceImage = function(full_canvas, file_format) {

      let use_frame = (full_canvas === "frame");

      let elem = use_frame ? this.svg_frame() : (full_canvas ? this.svg_canvas() : this.svg_pad());

      if (elem.empty()) return Promise.resolve("");

      let painter = (full_canvas && !use_frame) ? this.canv_painter() : this;

      let items = []; // keep list of replaced elements, which should be moved back at the end

      if (!use_frame) // do not make transformations for the frame
      painter.ForEachPainterInPad(pp => {

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
         data = data.replace(/%([0-9A-F]{2})/g, (match, p1) => {
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

   TPadPainter.prototype.PadButtonClick = function(funcname, evnt) {

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

            if (this.painters && (this.painters.length > 0)) {
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

   TPadPainter.prototype.AddButton = function(_btn, _tooltip, _funcname, _keyname) {
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

   TPadPainter.prototype.ShowButtons = function() {
      if (!this._buttons) return;

      JSROOT.require(['interactive']).then(inter => {
         inter.PadButtonsHandler.assign(this);
         this.ShowButtons();
      });
   }

   /** Add buttons for pad or canvas
     * @private */
   TPadPainter.prototype.AddPadButtons = function(is_online) {

      this.AddButton("camera", "Create PNG", this.iscan ? "CanvasSnapShot" : "PadSnapShot", "Ctrl PrintScreen");

      if (JSROOT.settings.ContextMenu)
         this.AddButton("question", "Access context menus", "PadContextMenus");

      let add_enlarge = !this.iscan && this.has_canvas && this.HasObjectsToDraw()

      if (add_enlarge || this.enlarge_main('verify'))
         this.AddButton("circle", "Enlarge canvas", "EnlargePad");

      if (is_online && this.brlayout) {
         this.AddButton("diamand", "Toggle Ged", "ToggleGed");
         this.AddButton("three_circles", "Toggle Status", "ToggleStatus");
      }
   }

   TPadPainter.prototype.DrawingReady = function(res_painter) {

      let main = this.frame_painter();

      if (main && main.mode3d && typeof main.Render3D == 'function') main.Render3D(-2222);

      JSROOT.ObjectPainter.prototype.DrawingReady.call(this, res_painter);
   }

   TPadPainter.prototype.DecodeOptions = function(opt) {
      let pad = this.GetObject();
      if (!pad) return;

      let d = new JSROOT.DrawOptions(opt);

      if (d.check('WEBSOCKET')) this.OpenWebsocket();
      if (!this.options) this.options = {};

      JSROOT.extend(this.options, { GlobalColors: true, LocalColors: false, CreatePalette: 0, IgnorePalette: false, RotateFrame: false, FixFrame: false });

      if (d.check('NOCOLORS') || d.check('NOCOL')) this.options.GlobalColors = this.options.LocalColors = false;
      if (d.check('LCOLORS') || d.check('LCOL')) { this.options.GlobalColors = false; this.options.LocalColors = true; }
      if (d.check('NOPALETTE') || d.check('NOPAL')) this.options.IgnorePalette = true;
      if (d.check('ROTATE')) this.options.RotateFrame = true;
      if (d.check('FIXFRAME')) this.options.FixFrame = true;

      if (d.check("CP",true)) this.options.CreatePalette = d.partAsInt(0,0);

      if (d.check('WHITE')) pad.fFillColor = 0;
      if (d.check('LOG2X')) { pad.fLogx = 2; pad.fUxmin = 0; pad.fUxmax = 1; pad.fX1 = 0; pad.fX2 = 1; }
      if (d.check('LOGX')) { pad.fLogx = 1; pad.fUxmin = 0; pad.fUxmax = 1; pad.fX1 = 0; pad.fX2 = 1; }
      if (d.check('LOG2Y')) { pad.fLogy = 2; pad.fUymin = 0; pad.fUymax = 1; pad.fY1 = 0; pad.fY2 = 1; }
      if (d.check('LOGY')) { pad.fLogy = 1; pad.fUymin = 0; pad.fUymax = 1; pad.fY1 = 0; pad.fY2 = 1; }
      if (d.check('LOG2Z')) pad.fLogz = 2;
      if (d.check('LOGZ')) pad.fLogz = 1;
      if (d.check('LOG2')) pad.fLogx = pad.fLogy = pad.fLogz = 2;
      if (d.check('LOG')) pad.fLogx = pad.fLogy = pad.fLogz = 1;
      if (d.check('GRIDX')) pad.fGridx = 1;
      if (d.check('GRIDY')) pad.fGridy = 1;
      if (d.check('GRID')) pad.fGridx = pad.fGridy = 1;
      if (d.check('TICKX')) pad.fTickx = 1;
      if (d.check('TICKY')) pad.fTicky = 1;
      if (d.check('TICK')) pad.fTickx = pad.fTicky = 1;
      if (d.check('OTX')) pad.$OTX = true;
      if (d.check('OTY')) pad.$OTY = true;
      if (d.check('CTX')) pad.$CTX = true;
      if (d.check('CTY')) pad.$CTY = true;
      if (d.check('RX')) pad.$RX = true;
      if (d.check('RY')) pad.$RY = true;

      this.OptionsStore(opt);
   }

   let drawPad = (divid, pad, opt) => {
      let painter = new TPadPainter(pad, false);
      painter.DecodeOptions(opt);

      painter.SetDivId(divid); // pad painter will be registered in the canvas painters list

      if (painter.svg_canvas().empty()) {
         painter.has_canvas = false;
         painter.this_pad_name = "";
      }

      painter.CreatePadSvg();

      if (painter.MatchObjectType("TPad") && (!painter.has_canvas || painter.HasObjectsToDraw()))
         painter.AddPadButtons();

      // we select current pad, where all drawing is performed
      let prev_name = painter.has_canvas ? painter.CurrentPadName(painter.this_pad_name) : undefined;

      // set active pad
      jsrp.SelectActivePad({ pp: painter, active: true });

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

   let TCanvasStatusBits = {
      kShowEventStatus: JSROOT.BIT(15),
      kAutoExec: JSROOT.BIT(16),
      kMenuBar: JSROOT.BIT(17),
      kShowToolBar: JSROOT.BIT(18),
      kShowEditor: JSROOT.BIT(19),
      kMoveOpaque: JSROOT.BIT(20),
      kResizeOpaque: JSROOT.BIT(21),
      kIsGrayscale: JSROOT.BIT(22),
      kShowToolTips: JSROOT.BIT(23)
   }

   /**
     * @summary Painter for TCanvas object
     *
     * @class
     * @memberof JSROOT
     * @private
     */

   function TCanvasPainter(canvas) {
      TPadPainter.call(this, canvas, true);
      this._websocket = null;
      this.tooltip_allowed = JSROOT.settings.Tooltip;
   }

   TCanvasPainter.prototype = Object.create(TPadPainter.prototype);

   /** @summary Changes layout
     * @returns {Promise} indicating when finished */
   TCanvasPainter.prototype.ChangeLayout = function(layout_kind) {
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
   TCanvasPainter.prototype.ToggleProjection = function(kind) {
      delete this.proj_painter;

      if (kind) this.proj_painter = 1; // just indicator that drawing can be preformed

      if (this.ShowUI5ProjectionArea)
         return this.ShowUI5ProjectionArea(kind);

      let layout = 'simple';

      if (kind == "X") layout = 'vert2_31'; else
      if (kind == "Y") layout = 'horiz2_13';

      return this.ChangeLayout(layout);
   }

   TCanvasPainter.prototype.DrawProjection = function(kind,hist) {

      if (!this.proj_painter) return; // ignore drawing if projection not configured

      if (this.proj_painter === 1) {

         let canv = JSROOT.Create("TCanvas"),
             pad = this.root_pad(),
             main = this.frame_painter(), drawopt;

         if (kind == "X") {
            canv.fLeftMargin = pad.fLeftMargin;
            canv.fRightMargin = pad.fRightMargin;
            canv.fLogx = main.logx;
            canv.fUxmin = main.logx ? Math.log10(main.scale_xmin) : main.scale_xmin;
            canv.fUxmax = main.logx ? Math.log10(main.scale_xmax) : main.scale_xmax;
            drawopt = "fixframe";
         } else {
            canv.fBottomMargin = pad.fBottomMargin;
            canv.fTopMargin = pad.fTopMargin;
            canv.fLogx = main.logy;
            canv.fUxmin = main.logy ? Math.log10(main.scale_ymin) : main.scale_ymin;
            canv.fUxmax = main.logy ? Math.log10(main.scale_ymax) : main.scale_ymax;
            drawopt = "rotate";
         }

         canv.fPrimitives.Add(hist, "hist");

         let promise = this.DrawInUI5ProjectionArea
                       ? this.DrawInUI5ProjectionArea(canv, drawopt)
                       : this.DrawInsidePanel(canv, drawopt);

         promise.then(painter => { this.proj_painter = painter; })
      } else {
         let hp = this.proj_painter.main_painter();
         if (hp) hp.UpdateObject(hist, "hist");
         this.proj_painter.RedrawPad();
      }
   }

   TCanvasPainter.prototype.DrawInsidePanel = function(canv, opt) {
      let side = this.select_main('origin').select(".side_panel");
      if (side.empty()) return Promise.resolve(null);
      return JSROOT.draw(side.node(), canv, opt);
   }

   TCanvasPainter.prototype.ShowMessage = function(msg) {
      if (this.testUI5()) return;
      JSROOT.progress(msg, 7000);
   }

   /** @summary Function called when canvas menu item Save is called */
   TCanvasPainter.prototype.SaveCanvasAsFile = function(fname) {
      let pnt = fname.indexOf(".");
      this.CreateImage(fname.substr(pnt+1))
          .then(res => this.SendWebsocket("SAVE:" + fname + ":" + res));
   }

   TCanvasPainter.prototype.SendSaveCommand = function(fname) {
      this.SendWebsocket("PRODUCE:" + fname);
   }

   TCanvasPainter.prototype.SubmitMenuRequest = function(painter, kind, reqid, call_back) {
      // only single request can be handled
      this._getmenu_callback = call_back;

      this.SendWebsocket('GETMENU:' + reqid); // request menu items for given painter
   }

   TCanvasPainter.prototype.SubmitExec = function(painter, exec, snapid) {
      if (this._readonly || !painter) return;

      if (!snapid) snapid = painter.snapid;
      if (!snapid || (typeof snapid != 'string')) return;

      this.SendWebsocket("OBJEXEC:" + snapid + ":" + exec);
   }

   TCanvasPainter.prototype.WindowBeforeUnloadHanlder = function() {
      // when window closed, close socket
      this.CloseWebsocket(true);
   }

   TCanvasPainter.prototype.SendWebsocket = function(msg) {
      if (!this._websocket) return;
      if (this._websocket.CanSend())
         this._websocket.Send(msg);
      else
         console.warn("DROP SEND: " + msg);
   }

   TCanvasPainter.prototype.CloseWebsocket = function(force) {
      if (this._websocket) {
         this._websocket.Close(force);
         this._websocket.Cleanup();
         delete this._websocket;
      }
   }

   TCanvasPainter.prototype.OpenWebsocket = function(socket_kind) {
      // create websocket for current object (canvas)
      // via websocket one recieved many extra information

      this.CloseWebsocket();

      this._websocket = new JSROOT.WebWindowHandle(socket_kind);
      this._websocket.SetReceiver(this);
      this._websocket.Connect();
   }

   TCanvasPainter.prototype.UseWebsocket = function(handle, href) {
      this.CloseWebsocket();

      this._websocket = handle;
      this._websocket.SetReceiver(this);
      this._websocket.Connect(href);
   }

   TCanvasPainter.prototype.OnWebsocketOpened = function(/*handle*/) {
      // indicate that we are ready to recieve any following commands
   }

   TCanvasPainter.prototype.OnWebsocketClosed = function(/*handle*/) {
      JSROOT.CloseCurrentWindow();
   }

   TCanvasPainter.prototype.OnWebsocketMsg = function(handle, msg) {
      console.log("GET MSG len:" + msg.length + " " + msg.substr(0,60));

      if (msg == "CLOSE") {
         this.OnWebsocketClosed();
         this.CloseWebsocket(true);
      } else if (msg.substr(0,6)=='SNAP6:') {
         // This is snapshot, produced with ROOT6

         let snap = JSROOT.parse(msg.substr(6));

         this.RedrawPadSnap(snap, () => {
            this.CompeteCanvasSnapDrawing();
            let ranges = this.GetWebPadOptions(); // all data, including subpads
            if (ranges) ranges = ":" + ranges;
            handle.Send("READY6:" + snap.fVersion + ranges); // send ready message back when drawing completed
         });
      } else if (msg.substr(0,5)=='MENU:') {
         // this is menu with exact identifier for object
         let lst = JSROOT.parse(msg.substr(5));
         if (typeof this._getmenu_callback == 'function') {
            this._getmenu_callback(lst);
            delete this._getmenu_callback;
         }
      } else if (msg.substr(0,4)=='CMD:') {
         msg = msg.substr(4);
         let p1 = msg.indexOf(":"),
             cmdid = msg.substr(0,p1),
             cmd = msg.substr(p1+1),
             reply = "REPLY:" + cmdid + ":";
         if ((cmd == "SVG") || (cmd == "PNG") || (cmd == "JPEG")) {
            this.CreateImage(cmd.toLowerCase())
                .then(res => handle.Send(reply + res));
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
             on = (that[that.length-1] == '1');
         this.ShowSection(that.substr(0,that.length-2), on);
      } else if (msg.substr(0,5) == "EDIT:") {
         let obj_painter = this.FindSnap(msg.substr(5));
         console.log('GET EDIT ' + msg.substr(5) +  ' found ' + !!obj_painter);
         if (obj_painter)
            this.ShowSection("Editor", true)
                .then(() => this.PadEvent("select", obj_painter.pad_painter(), obj_painter));

      } else {
         console.log("unrecognized msg " + msg);
      }
   }

   TCanvasPainter.prototype.PadButtonClick = function(funcname, evnt) {
      if (funcname == "ToggleGed") return this.ActivateGed(this, null, "toggle");
      if (funcname == "ToggleStatus") return this.ActivateStatusBar("toggle");
      TPadPainter.prototype.PadButtonClick.call(this, funcname, evnt);
   }

   TCanvasPainter.prototype.testUI5 = function() {
      if (!this.use_openui) return false;
      console.warn("full ui5 should be used - not loaded yet? Please check!!");
      return true;
   }

   TCanvasPainter.prototype.HasEventStatus = function() {
      if (this.testUI5()) return false;
      return this.brlayout ? this.brlayout.HasStatus() : false;
   }

   TCanvasPainter.prototype.ActivateStatusBar = function(state) {
      if (this.testUI5()) return;
      if (this.brlayout)
         this.brlayout.CreateStatusLine(23, state);
      this.ProcessChanges("sbits", this);
   }

   /** Returns true if GED is present on the canvas */
   TCanvasPainter.prototype.HasGed = function() {
      if (this.testUI5()) return false;
      return this.brlayout ? this.brlayout.HasContent() : false;
   }

   /** Function used to de-activate GED */
   TCanvasPainter.prototype.RemoveGed = function() {
      if (this.testUI5()) return;

      this.RegisterForPadEvents(null);

      if (this.ged_view) {
         this.ged_view.getController().cleanupGed();
         this.ged_view.destroy();
         delete this.ged_view;
      }
      if (this.brlayout)
         this.brlayout.DeleteContent();

      this.ProcessChanges("sbits", this);
   }

   /** @summary Function used to activate GED
     * @returns {Promise} when GED is there */
   TCanvasPainter.prototype.ActivateGed = function(objpainter, kind, mode) {
      if (this.testUI5() || !this.brlayout)
         return Promise.resolve(false);

      if (this.brlayout.HasContent()) {
         if ((mode === "toggle") || (mode === false)) {
            this.RemoveGed();
         } else {
            let pp = objpainter ? objpainter.pad_painter() : null;
            if (pp) pp.SelectObjectPainter(objpainter);
         }

         return Promise.resolve(true);
      }

      if (mode === false)
         return Promise.resolve(false);

      let btns = this.brlayout.CreateBrowserBtns();

      JSROOT.require('interactive').then(inter => {

         inter.ToolbarIcons.CreateSVG(btns, inter.ToolbarIcons.diamand, 15, "toggle fix-pos mode")
                            .style("margin","3px").on("click", () => this.brlayout.Toggle('fix'));

         inter.ToolbarIcons.CreateSVG(btns, inter.ToolbarIcons.circle, 15, "toggle float mode")
                            .style("margin","3px").on("click", () => this.brlayout.Toggle('float'));

         inter.ToolbarIcons.CreateSVG(btns, inter.ToolbarIcons.cross, 15, "delete GED")
                            .style("margin","3px").on("click", () => this.RemoveGed());
      });

      // be aware, that jsroot_browser_hierarchy required for flexible layout that element use full browser area
      this.brlayout.SetBrowserContent("<div class='jsroot_browser_hierarchy' id='ged_placeholder'>Loading GED ...</div>");
      this.brlayout.SetBrowserTitle("GED");
      this.brlayout.ToggleBrowserKind(kind || "float");

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
                  this.RegisterForPadEvents(oGed.getController().padEventsReceiver.bind(oGed.getController()));

                  let pp = objpainter ? objpainter.pad_painter() : null;
                  if (pp) pp.SelectObjectPainter(objpainter);

                  this.ProcessChanges("sbits", this);

                  resolveFunc(true);
               });
            });
         });
      });
   }


   TCanvasPainter.prototype.ShowSection = function(that, on) {
      if (this.testUI5())
         return Promise.resolve(false);

      console.log('Show section ' + that + ' flag = ' + on);

      switch(that) {
         case "Menu": break;
         case "StatusBar": this.ActivateStatusBar(on); break;
         case "Editor": return this.ActivateGed(this, null, !!on);
         case "ToolBar": break;
         case "ToolTips": this.SetTooltipAllowed(on); break;

      }
      return Promise.resolve(true);
   }

   TCanvasPainter.prototype.CompeteCanvasSnapDrawing = function() {
      if (!this.pad) return;

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
   TCanvasPainter.prototype.ProcessChanges = function(kind, painter, subelem) {
      // check if we could send at least one message more - for some meaningful actions
      if (!this._websocket || this._readonly || !this._websocket.CanSend(2) || (typeof kind !== "string")) return;

      let msg = "";
      if (!painter) painter = this;
      switch (kind) {
         case "sbits":
            msg = "STATUSBITS:" + this.GetStatusBits();
            break;
         case "frame": // when moving frame
         case "zoom":  // when changing zoom inside frame
            if (!painter.GetWebPadOptions)
               painter = painter.pad_painter();
            if (typeof painter.GetWebPadOptions == "function")
               msg = "OPTIONS6:" + painter.GetWebPadOptions("only_this");
            break;
         case "pave_moved":
            if (painter.FillWebObjectOptions) {
               let info = painter.FillWebObjectOptions();
               if (info) msg = "PRIMIT6:" + JSROOT.toJSON(info);
            }
            break;
         default:
            if ((kind.substr(0,5) == "exec:") && painter && painter.snapid) {
               msg = "PRIMIT6:" + JSROOT.toJSON({
                  _typename: "TWebObjectOptions",
                  snapid: painter.snapid.toString() + (subelem ? "#"+subelem : ""),
                  opt: kind.substr(5),
                  fcust: "exec",
                  fopt: []
               });
            } else {
               console.log("UNPROCESSED CHANGES", kind);
            }
      }

      if (msg) {
         console.log("Sending " + msg.length + "  " + msg.substr(0,40));
         this._websocket.Send(msg);
      }
   }

   /** @summary Select active pad on the canvas */
   TCanvasPainter.prototype.SelectActivePad = function(pad_painter, obj_painter, click_pos) {
      if ((this.snapid === undefined) || !pad_painter) return; // only interactive canvas

      let arg = null, ischanged = false;

      if ((pad_painter.snapid !== undefined) && this._websocket)
         arg = { _typename: "TWebPadClick", padid: pad_painter.snapid.toString(), objid: "", x: -1, y: -1, dbl: false };

      if (!pad_painter.is_active_pad) {
         ischanged = true;
         this.ForEachPainterInPad(pp => pp.DrawActiveBorder(null, pp === pad_painter), "pads");
      }

      if (obj_painter && (obj_painter.snapid!==undefined) && arg) {
         ischanged = true;
         arg.objid = obj_painter.snapid.toString();
      }

      if (click_pos && arg) {
         ischanged = true;
         arg.x = Math.round(click_pos.x || 0);
         arg.y = Math.round(click_pos.y || 0);
         if (click_pos.dbl) arg.dbl = true;
      }

      if (arg && ischanged)
         this.SendWebsocket("PADCLICKED:" + JSROOT.toJSON(arg));
   }

   TCanvasPainter.prototype.GetStatusBits = function() {
      let bits = 0;
      if (this.HasEventStatus()) bits |= TCanvasStatusBits.kShowEventStatus;
      if (this.HasGed()) bits |= TCanvasStatusBits.kShowEditor;
      if (this.IsTooltipAllowed()) bits |= TCanvasStatusBits.kShowToolTips;
      if (this.use_openui) bits |= TCanvasStatusBits.kMenuBar;
      return bits;
   }

   /**  @summary produce JSON for TCanvas, which can be used to display canvas once again */
   TCanvasPainter.prototype.ProduceJSON = function() {

      let canv = this.GetObject(),
          fill0 = (canv.fFillStyle == 0);

      if (fill0) canv.fFillStyle = 1001;

      if (!this.normal_canvas) {

         // fill list of primitives from painters
         this.ForEachPainterInPad(p => {
            if (p.$secondary) return; // ignore all secoandry painters

            let subobj = p.GetObject();
            if (subobj && subobj._typename)
               canv.fPrimitives.Add(subobj, p.OptionsAsString());
         }, "objects");
      }

      let res = JSROOT.toJSON(canv);

      if (fill0) canv.fFillStyle = 0;

      if (!this.normal_canvas)
         canv.fPrimitives.Clear();

      return res;
   }

   /** Check if TGeo objects in the canvas - draw them directly */
   TCanvasPainter.prototype.DirectGeoDraw = function() {
      let lst = this.pad ? this.pad.fPrimitives : null;
      if (!lst || (lst.arr.length != 1)) return;

      let obj = lst.arr[0];
      if (obj && obj._typename && (obj._typename.indexOf("TGeo")==0))
         return JSROOT.draw(this.divid, obj, lst.opt[0]); // return promise
   }

   let drawCanvas = (divid, can, opt) => {
      let nocanvas = !can;
      if (nocanvas) can = JSROOT.Create("TCanvas");

      let painter = new TCanvasPainter(can);
      painter.SetDivId(divid, -1); // just assign id

      if (!nocanvas && can.fCw && can.fCh && !JSROOT.BatchMode) {
         let rect0 = painter.select_main().node().getBoundingClientRect();
         if (!rect0.height && (rect0.width > 0.1*can.fCw)) {
            painter.select_main().style("width", can.fCw+"px").style("height", can.fCh+"px");
            painter._fixed_size = true;
         }
      }

      let direct = painter.DirectGeoDraw();
      if (direct) return direct;

      painter.DecodeOptions(opt);
      painter.normal_canvas = !nocanvas;
      painter.CheckSpecialsInPrimitives(can);
      painter.CreateCanvasSvg(0);
      painter.SetDivId(divid);  // now add to painters list

      painter.AddPadButtons();

      if (nocanvas && opt.indexOf("noframe") < 0)
         drawFrame(divid, null);

      // select global reference - required for keys handling
      jsrp.SelectActivePad({ pp: painter, active: true });

      painter.DrawPrimitives(0, () => { painter.ShowButtons(); painter.DrawingReady(); });
      return painter;
   }

   let drawPadSnapshot = (divid, snap /*, opt*/) => {
      // just for debugging without running web canvas

      let can = JSROOT.Create("TCanvas");

      let painter = new TCanvasPainter(can);
      painter.normal_canvas = false;

      painter.SetDivId(divid, -1); // just assign id

      painter.AddPadButtons();

      painter.RedrawPadSnap(snap, () => { painter.ShowButtons(); painter.DrawingReady(); });

      return painter;
   }

   JSROOT.TAxisPainter = TAxisPainter;
   JSROOT.TFramePainter = TFramePainter;
   JSROOT.TPadPainter = TPadPainter;
   JSROOT.TCanvasPainter = TCanvasPainter;

   jsrp.drawGaxis = drawGaxis;
   jsrp.drawFrame = drawFrame;
   jsrp.drawPad = drawPad;
   jsrp.drawCanvas = drawCanvas;
   jsrp.drawPadSnapshot = drawPadSnapshot;

   return JSROOT;
});
