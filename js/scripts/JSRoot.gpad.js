// @file JSRoot.gpad.js
/// JSROOT TPad/TCanvas/TFrame support

JSROOT.define(['d3', 'painter'], (d3, jsrp) => {

   "use strict";

   // identifier used in TWebCanvas painter
   const webSnapIds = { kNone: 0,  kObject: 1, kSVG: 2, kSubPad: 3, kColors: 4, kStyle: 5 };

   // =======================================================================

   JSROOT.EAxisBits = {
      kDecimals: JSROOT.BIT(7),
      kTickPlus: JSROOT.BIT(9),
      kTickMinus: JSROOT.BIT(10),
      kAxisRange: JSROOT.BIT(11),
      kCenterTitle: JSROOT.BIT(12),
      kCenterLabels: JSROOT.BIT(14),
      kRotateTitle: JSROOT.BIT(15),
      kPalette: JSROOT.BIT(16),
      kNoExponent: JSROOT.BIT(17),
      kLabelsHori: JSROOT.BIT(18),
      kLabelsVert: JSROOT.BIT(19),
      kLabelsDown: JSROOT.BIT(20),
      kLabelsUp: JSROOT.BIT(21),
      kIsInteger: JSROOT.BIT(22),
      kMoreLogLabels: JSROOT.BIT(23),
      kOppositeTitle: JSROOT.BIT(32) // atrificial bit, not possible to set in ROOT
   };

   /** @summary Return time offset value for given TAxis object
     * @private */
   function getTimeOffset(axis) {
      let dflt_time_offset = 788918400000;
      if (!axis) return dflt_time_offset;
      let idF = axis.fTimeFormat.indexOf('%F');
      if (idF < 0) return JSROOT.gStyle.fTimeOffset * 1000;
      let sof = axis.fTimeFormat.substr(idF + 2);
      // default string in axis offset
      if (sof.indexOf('1995-01-01 00:00:00s0') == 0) return dflt_time_offset;
      // special case, used from DABC painters
      if ((sof == "0") || (sof == "")) return 0;

      // decode time from ROOT string
      const next = (separ, min, max) => {
         let pos = sof.indexOf(separ);
         if (pos < 0) return min;
         let val = parseInt(sof.substr(0, pos));
         sof = sof.substr(pos + 1);
         if (!Number.isInteger(val) || (val < min) || (val > max)) return min;
         return val;
      }, year = next("-", 1970, 2300),
         month = next("-", 1, 12) - 1,
         day = next(" ", 1, 31),
         hour = next(":", 0, 23),
         min = next(":", 0, 59),
         sec = next("s", 0, 59),
         msec = next(" ", 0, 999),
         dt = new Date(Date.UTC(year, month, day, hour, min, sec, msec));

      let offset = dt.getTime();

      // now also handle suffix like GMT or GMT -0600
      sof = sof.toUpperCase();

      if (sof.indexOf('GMT') == 0) {
         offset += dt.getTimezoneOffset() * 60000;
         sof = sof.substr(4).trim();
         if (sof.length > 3) {
            let p = 0, sign = 1000;
            if (sof[0] == '-') { p = 1; sign = -1000; }
            offset -= sign * (parseInt(sof.substr(p, 2)) * 3600 + parseInt(sof.substr(p + 2, 2)) * 60);
         }
      }

      return offset;
   }


   /**
    * @summary Painter for TAxis/TGaxis objects
    *
    * @memberof JSROOT
    * @private
    */

   class TAxisPainter extends JSROOT.ObjectPainter {

      /** @summary constructor
        * @param {object|string} dom - identifier or dom element
        * @param {object} axis - object to draw
        * @param {boolean} embedded - if true, painter used in other objects painters */
      constructor(dom, axis, embedded) {
         super(dom, axis);

         Object.assign(this, JSROOT.AxisPainterMethods);
         this.initAxisPainter();

         this.embedded = embedded; // indicate that painter embedded into the histo painter
         this.invert_side = false;
         this.lbls_both_sides = false; // draw labels on both sides
      }

      /** @summary cleanup painter */
      cleanup() {
         this.cleanupAxisPainter();
         super.cleanup();
      }

      /** @summary Use in GED to identify kind of axis */
      getAxisType() { return "TAxis"; }


      /** @summary Configure axis painter
        * @desc Axis can be drawn inside frame <g> group with offset to 0 point for the frame
        * Therefore one should distinguish when caclulated coordinates used for axis drawing itself or for calculation of frame coordinates
        * @private */
      configureAxis(name, min, max, smin, smax, vertical, range, opts) {
         this.name = name;
         this.full_min = min;
         this.full_max = max;
         this.kind = "normal";
         this.vertical = vertical;
         this.log = opts.log || 0;
         this.symlog = opts.symlog || false;
         this.reverse = opts.reverse || false;
         this.swap_side = opts.swap_side || false;
         this.fixed_ticks = opts.fixed_ticks || null;
         this.max_tick_size = opts.max_tick_size || 0;

         let axis = this.getObject();

         if (opts.time_scale || axis.fTimeDisplay) {
            this.kind = 'time';
            this.timeoffset = getTimeOffset(axis);
         } else {
            this.kind = !axis.fLabels ? 'normal' : 'labels';
         }

         if (this.kind == 'time') {
            this.func = d3.scaleTime().domain([this.convertDate(smin), this.convertDate(smax)]);
         } else if (this.log) {
            this.logbase = this.log === 2 ? 2 : 10;
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
         } else if (this.symlog) {
            let v = Math.max(Math.abs(smin), Math.abs(smax));
            if (Number.isInteger(this.symlog) && (this.symlog > 0))
               v *= Math.pow(10,-1*this.symlog);
            else
               v *= 0.01;
            this.func = d3.scaleSymlog().constant(v).domain([smin,smax]);
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
            this.gr = val => this.func(this.convertDate(val));
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

         if (axis && !is_gaxis && (this.nticks > 20)) this.nticks = 20;

         let gr_range = Math.abs(this.func.range()[1] - this.func.range()[0]);
         if (gr_range<=0) gr_range = 100;

         if (this.kind == 'time') {
            if (this.nticks > 8) this.nticks = 8;

            let scale_range = this.scale_max - this.scale_min,
                idF = axis.fTimeFormat.indexOf('%F'),
                tf1 = (idF >= 0) ? axis.fTimeFormat.substr(0, idF) : axis.fTimeFormat,
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

      /** @summary Return scale min */
      getScaleMin() {
         return this.func ? this.func.domain()[0] : 0;
      }

      /** @summary Return scale max */
      getScaleMax() {
         return this.func ? this.func.domain()[1] : 0;
      }

      /** @summary Provide label for axis value */
      formatLabels(d) {
         let indx = parseFloat(d), a = this.getObject();
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

      /** @summary Creates array with minor/middle/major ticks */
      createTicks(only_major_as_array, optionNoexp, optionNoopt, optionInt) {

         if (optionNoopt && this.nticks && (this.kind == "normal")) this.noticksopt = true;

         let handle = { nminor: 0, nmiddle: 0, nmajor: 0, func: this.func }, ticks;

         if (this.fixed_ticks) {
            ticks = [];
            this.fixed_ticks.forEach(v => {
               if ((v >= this.scale_min) && (v <= this.scale_max)) ticks.push(v);
            });
         } else if ((this.kind == 'labels') && !this.regular_labels) {
            ticks = [];
            handle.lbl_pos = [];
            let axis = this.getObject();
            for (let n = 0; n < axis.fNbins; ++n) {
               let x = axis.fXmin + n / axis.fNbins * (axis.fXmax - axis.fXmin);
               if ((x >= this.scale_min) && (x < this.scale_max)) {
                  handle.lbl_pos.push(x);
                  if (x > this.scale_min) ticks.push(x);
               }
            }
         } else {
            ticks = this.produceTicks(this.nticks);
         }

         handle.minor = handle.middle = handle.major = ticks;

         if (only_major_as_array) {
            let res = handle.major, delta = (this.scale_max - this.scale_min)*1e-5;
            if (res[0] > this.scale_min + delta) res.unshift(this.scale_min);
            if (res[res.length-1] < this.scale_max - delta) res.push(this.scale_max);
            return res;
         }

         if ((this.nticks2 > 1) && (!this.log || (this.logbase === 10)) && !this.fixed_ticks) {
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

      /** @summary Is labels should be centered */
      isCenteredLabels() {
         if (this.kind === 'labels') return true;
         if (this.log) return false;
         let axis = this.getObject();
         return axis && axis.TestBit(JSROOT.EAxisBits.kCenterLabels);
      }

      /** @summary Add interactive elements to draw axes title */
      addTitleDrag(title_g, vertical, offset_k, reverse, axis_length) {
         if (!JSROOT.settings.MoveResize || JSROOT.batch_mode) return;

         let drag_rect = null,
             acc_x, acc_y, new_x, new_y, sign_0, alt_pos, curr_indx,
             drag_move = d3.drag().subject(Object);

         drag_move
            .on("start", evnt => {

               evnt.sourceEvent.preventDefault();
               evnt.sourceEvent.stopPropagation();

               let box = title_g.node().getBBox(), // check that elements visible, request precise value
                   axis = this.getObject(),
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

                  let set_x, set_y,
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

                  let axis = this.getObject(), abits = JSROOT.EAxisBits;

                  const set_bit = (bit, on) => { if (axis.TestBit(bit) != on) axis.InvertBit(bit); };

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

      /** @summary Produce svg path for axis ticks */
      produceTicksPath(handle, side, tickSize, ticksPlusMinus, secondShift, real_draw) {
         let res = "", res2 = "", lastpos = 0, lasth = 0;
         this.ticks = [];

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

            if (ticksPlusMinus > 0) {
               h2 = -h1;
            } else if (side < 0) {
               h2 = -h1; h1 = 0;
            }

            if (res.length == 0) {
               res = this.vertical ? `M${h1},${handle.grpos}` : `M${handle.grpos},${-h1}`;
               res2 = this.vertical ? `M${secondShift-h1},${handle.grpos}` : `M${handle.grpos},${secondShift+h1}`;
            } else {
               res += this.vertical ? `m${h1-lasth},${handle.grpos-lastpos}` : `m${handle.grpos-lastpos},${lasth-h1}`;
               res2 += this.vertical ? `m${lasth-h1},${handle.grpos-lastpos}` : `m${handle.grpos-lastpos},${h1-lasth}`;
            }

            res += this.vertical ? `h${h2-h1}` : `v${h1-h2}`;
            res2 += this.vertical ? `h${h1-h2}` : `v${h2-h1}`;

            lastpos = handle.grpos;
            lasth = h2;
         }

         if (secondShift !== 0) res += res2;

         return real_draw ? res  : "";
      }

      /** @summary Returns modifier for axis label */
      findLabelModifier(axis, nlabel, num_labels) {
         if (!axis.fModLabs) return null;
         for (let n = 0; n < axis.fModLabs.arr.length; ++n) {
            let mod = axis.fModLabs.arr[n];
            if (mod.fLabNum === nlabel + 1) return mod;
            if ((mod.fLabNum < 0) && (nlabel === num_labels + mod.fLabNum)) return mod;
         }
         return null;
      }

      /** @summary Draw axis labels
        * @returns {Promise} with array label size and max width */
      drawLabels(axis_g, axis, w, h, handle, side, labelSize, labeloffset, tickSize, ticksPlusMinus, max_text_width) {
         let label_color = this.getColor(axis.fLabelColor),
             center_lbls = this.isCenteredLabels(),
             rotate_lbls = axis.TestBit(JSROOT.EAxisBits.kLabelsVert),
             textscale = 1, maxtextlen = 0, applied_scale = 0,
             label_g = [ axis_g.append("svg:g").attr("class","axis_labels") ],
             lbl_pos = handle.lbl_pos || handle.major, lbl_tilt = false, max_textwidth = 0;

         if (this.lbls_both_sides)
            label_g.push(axis_g.append("svg:g").attr("class","axis_labels").attr("transform", this.vertical ? `translate(${w})` : `translate(0,${-h})`));

         // function called when text is drawn to analyze width, required to correctly scale all labels
         // must be function to correctly handle 'this' argument
         function process_drawtext_ready(painter) {
            let textwidth = this.result_width;
            max_textwidth = Math.max(max_textwidth, textwidth);

            if (textwidth && ((!painter.vertical && !rotate_lbls) || (painter.vertical && rotate_lbls)) && !painter.log) {
               let maxwidth = this.gap_before*0.45 + this.gap_after*0.45;
               if (!this.gap_before) maxwidth = 0.9*this.gap_after; else
               if (!this.gap_after) maxwidth = 0.9*this.gap_before;
               textscale = Math.min(textscale, maxwidth / textwidth);
            } else if (painter.vertical && max_text_width && this.normal_side && (max_text_width - labeloffset > 20) && (textwidth > max_text_width - labeloffset)) {
               textscale = Math.min(textscale, (max_text_width - labeloffset) / textwidth);
            }

            if ((textscale > 0.01) && (textscale < 0.7) && !painter.vertical && !rotate_lbls && (maxtextlen > 5) && (label_g.length == 1))
               lbl_tilt = true;

            let scale = textscale * (lbl_tilt ? 3 : 1);

            if ((scale > 0.01) && (scale < 1)) {
               applied_scale = 1/scale;
               painter.scaleTextDrawing(applied_scale, label_g[0]);
            }
         }

         const labelfont = new JSROOT.FontHandler(axis.fLabelFont, labelSize);

         for (let lcnt = 0; lcnt < label_g.length; ++lcnt) {

            if (lcnt > 0) side = -side;

            let lastpos = 0, fix_coord = this.vertical ? -labeloffset*side : (labeloffset+2)*side + ticksPlusMinus*tickSize;

            this.startTextDrawing(labelfont, 'font', label_g[lcnt]);

            for (let nmajor = 0; nmajor < lbl_pos.length; ++nmajor) {

               let lbl = this.format(lbl_pos[nmajor], true);
               if (lbl === null) continue;

               let mod = this.findLabelModifier(axis, nmajor, lbl_pos.length);
               if (mod && (mod.fTextSize == 0)) continue;

               if (mod && mod.fLabText) lbl = mod.fLabText;

               let arg = { text: lbl, color: label_color, latex: 1, draw_g: label_g[lcnt], normal_side: (lcnt == 0) };

               let pos = Math.round(this.func(lbl_pos[nmajor]));

               if (mod && mod.fTextColor > 0) arg.color = this.getColor(mod.fTextColor);

               arg.gap_before = (nmajor > 0) ? Math.abs(Math.round(pos - this.func(lbl_pos[nmajor-1]))) : 0;

               arg.gap_after = (nmajor < lbl_pos.length-1) ? Math.abs(Math.round(this.func(lbl_pos[nmajor+1])-pos)) : 0;

               if (center_lbls) {
                  let gap = arg.gap_after || arg.gap_before;
                  pos = Math.round(pos - (this.vertical ? 0.5*gap : -0.5*gap));
                  if ((pos < -5) || (pos > (this.vertical ? h : w) + 5)) continue;
               }

               maxtextlen = Math.max(maxtextlen, lbl.length);

               if (this.vertical) {
                  arg.x = fix_coord;
                  arg.y = pos;
                  arg.align = rotate_lbls ? ((side<0) ? 23 : 20) : ((side<0) ? 12 : 32);
               } else {
                  arg.x = pos;
                  arg.y = fix_coord;
                  arg.align = rotate_lbls ? ((side<0) ? 12 : 32) : ((side<0) ? 20 : 23);
               }

               if (rotate_lbls)
                  arg.rotate = 270;

               // only for major text drawing scale factor need to be checked
               if (lcnt == 0) arg.post_process = process_drawtext_ready;

               this.drawText(arg);

               if (lastpos && (pos!=lastpos) && ((this.vertical && !rotate_lbls) || (!this.vertical && rotate_lbls))) {
                  let axis_step = Math.abs(pos-lastpos);
                  textscale = Math.min(textscale, 0.9*axis_step/labelSize);
               }

               lastpos = pos;
            }

            if (this.order)
               this.drawText({ color: label_color,
                               x: this.vertical ? side*5 : w+5,
                               y: this.has_obstacle ? fix_coord : (this.vertical ? -3 : -3*side),
                               align: this.vertical ? ((side < 0) ? 30 : 10) : ( (this.has_obstacle ^ (side < 0)) ? 13 : 10 ),
                               latex: 1,
                               text: '#times' + this.formatExp(10, this.order),
                               draw_g: label_g[lcnt]
               });
         }

         // first complete major labels drawing
         return this.finishTextDrawing(label_g[0], true).then(() => {
            if (label_g.length > 1) {
               // now complete drawing of second half with scaling if necessary
               if (applied_scale)
                  this.scaleTextDrawing(applied_scale, label_g[1]);
               return this.finishTextDrawing(label_g[1], true);
             }
             return true;
         }).then(() => {
            if (lbl_tilt)
               label_g[0].selectAll("text").each(function() {
                  let txt = d3.select(this), tr = txt.attr("transform");
                  txt.attr("transform", tr + " rotate(25)").style("text-anchor", "start");
               });

            if (labelfont) labelSize = labelfont.size; // use real font size

            return [ labelSize, max_textwidth ];
         });
      }

      /** @summary function draws TAxis or TGaxis object
        * @returns {Promise} for drawing ready */
      drawAxis(layer, w, h, transform, secondShift, disable_axis_drawing, max_text_width, calculate_position) {

         let axis = this.getObject(), chOpt = "",
             is_gaxis = (axis && axis._typename === 'TGaxis'),
             axis_g = layer, tickSize = 0.03,
             scaling_size, draw_lines = true,
             pp = this.getPadPainter(),
             pad_w = pp ? pp.getPadWidth() : 10,
             pad_h = pp ? pp.getPadHeight() : 10,
             vertical = this.vertical,
             swap_side = this.swap_side || false;

         // shift for second ticks set (if any)
         if (!secondShift) secondShift = 0; else
         if (this.invert_side) secondShift = -secondShift;

         if (is_gaxis) {
            this.createAttLine({ attr: axis });
            draw_lines = (axis.fLineColor != 0);
            chOpt = axis.fChopt;
            tickSize = axis.fTickSize;
            scaling_size = vertical ? 1.7*h : 0.6*w;
         } else {
            this.createAttLine({ color: axis.fAxisColor, width: 1, style: 1 });
            chOpt = (vertical ^ this.invert_side) ? "-S" : "+S";
            tickSize = axis.fTickLength;
            scaling_size = vertical ? pad_w : pad_h;
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

         let axis_lines = "";
         if (draw_lines) {
            axis_lines = "M0,0" + (vertical ? `v${h}` : `h${w}`);
            if (secondShift !== 0)
               axis_lines += vertical ? `M${secondShift},0v${h}` : `M0,${secondShift}h${w}`;
         }

         axis_g.attr("transform", transform || null);

         let side = 1, ticksPlusMinus = 0,
             text_scaling_size = Math.min(pad_w, pad_h),
             optionPlus = (chOpt.indexOf("+")>=0),
             optionMinus = (chOpt.indexOf("-")>=0),
             optionSize = (chOpt.indexOf("S")>=0),
             // optionY = (chOpt.indexOf("Y")>=0),
             // optionUp = (chOpt.indexOf("0")>=0),
             // optionDown = (chOpt.indexOf("O")>=0),
             optionUnlab = (chOpt.indexOf("U")>=0) || this.optionUnlab,  // no labels
             optionNoopt = (chOpt.indexOf("N")>=0),  // no ticks position optimization
             optionInt = (chOpt.indexOf("I")>=0),    // integer labels
             optionNoexp = axis.TestBit(JSROOT.EAxisBits.kNoExponent);

         if (text_scaling_size <= 0) text_scaling_size = 0.0001;

         if (is_gaxis && axis.TestBit(JSROOT.EAxisBits.kTickPlus)) optionPlus = true;
         if (is_gaxis && axis.TestBit(JSROOT.EAxisBits.kTickMinus)) optionMinus = true;

         if (optionPlus && optionMinus) { side = 1; ticksPlusMinus = 1; } else
         if (optionMinus) { side = (swap_side ^ vertical) ? 1 : -1; } else
         if (optionPlus) { side = (swap_side ^ vertical) ? -1 : 1; }

         tickSize = Math.round((optionSize ? tickSize : 0.03) * scaling_size);
         if (this.max_tick_size && (tickSize > this.max_tick_size)) tickSize = this.max_tick_size;

         // first draw ticks

         const handle = this.createTicks(false, optionNoexp, optionNoopt, optionInt);

         axis_lines += this.produceTicksPath(handle, side, tickSize, ticksPlusMinus, secondShift, draw_lines && !disable_axis_drawing && !this.disable_ticks);

         if (!disable_axis_drawing && axis_lines && !this.lineatt.empty())
            axis_g.append("svg:path").attr("d", axis_lines)
                  .call(this.lineatt.func);

         let labelSize0 = Math.round( (axis.fLabelSize < 1) ? axis.fLabelSize * text_scaling_size : axis.fLabelSize),
             labeloffset = Math.round(Math.abs(axis.fLabelOffset)*text_scaling_size);

         if ((labelSize0 <= 0) || (Math.abs(axis.fLabelOffset) > 1.1)) optionUnlab = true; // disable labels when size not specified

         let labelsPromise, title_shift_x = 0, title_shift_y = 0, title_g = null, axis_rect = null,
             title_fontsize = 0, labelMaxWidth = 0;

         // draw labels (sometime on both sides)
         if (!disable_axis_drawing && !optionUnlab)
            labelsPromise = this.drawLabels(axis_g, axis, w, h, handle, side, labelSize0, labeloffset, tickSize, ticksPlusMinus, max_text_width);
         else
            labelsPromise = Promise.resolve([labelSize0, 0]);

         return labelsPromise.then(arr => {
            labelMaxWidth = arr[1];
            if (JSROOT.settings.Zooming && !this.disable_zooming && !JSROOT.batch_mode) {
               let labelSize = arr[0],
                   r = axis_g.append("svg:rect")
                             .attr("class", "axis_zoom")
                             .style("opacity", "0")
                             .style("cursor", "crosshair");

               if (vertical) {
                  let rw = (labelMaxWidth || 2*labelSize) + 3;
                  r.attr("x", (side > 0) ? -rw : 0)
                   .attr("y", 0)
                   .attr("width", rw)
                   .attr("height", h);
               } else {
                  r.attr("x", 0).attr("y", (side > 0) ? 0 : -labelSize - 3)
                   .attr("width", w).attr("height", labelSize + 3);
               }
            }

            this.position = 0;

            if (calculate_position) {
               let node1 = axis_g.node(), node2 = this.getPadSvg().node();
               if (node1 && node2 && node1.getBoundingClientRect && node2.getBoundingClientRect) {
                  let rect1 = node1.getBoundingClientRect(),
                      rect2 = node2.getBoundingClientRect();

                  this.position = rect1.left - rect2.left; // use to control left position of Y scale
               }
               if (node1 && !node2)
                  console.warn("Why PAD element missing when search for position");
            }

            if (!axis.fTitle || disable_axis_drawing) return true;

            title_g = axis_g.append("svg:g").attr("class", "axis_title");
            title_fontsize = (axis.fTitleSize >= 1) ? axis.fTitleSize : Math.round(axis.fTitleSize * text_scaling_size);

            let title_offest_k = 1.6*((axis.fTitleSize < 1) ? axis.fTitleSize : axis.fTitleSize/(text_scaling_size || 10)),
                center = axis.TestBit(JSROOT.EAxisBits.kCenterTitle),
                opposite = axis.TestBit(JSROOT.EAxisBits.kOppositeTitle),
                rotate = axis.TestBit(JSROOT.EAxisBits.kRotateTitle) ? -1 : 1,
                title_color = this.getColor(is_gaxis ? axis.fTextColor : axis.fTitleColor);

            this.startTextDrawing(axis.fTitleFont, title_fontsize, title_g);

            let xor_reverse = swap_side ^ opposite, myxor = (rotate < 0) ^ xor_reverse;

            this.title_align = center ? "middle" : (myxor ? "begin" : "end");

            if (vertical) {
               title_offest_k *= -side*pad_w;

               title_shift_x = Math.round(title_offest_k*axis.fTitleOffset);

               if ((this.name == "zaxis") && is_gaxis && ('getBoundingClientRect' in axis_g.node())) {
                  // special handling for color palette labels - draw them always on right side
                  let rect = axis_g.node().getBoundingClientRect();
                  if (title_shift_x < rect.width - tickSize) title_shift_x = Math.round(rect.width - tickSize);
               }

               title_shift_y = Math.round(center ? h/2 : (xor_reverse ? h : 0));

               this.drawText({ align: this.title_align+";middle",
                               rotate: (rotate<0) ? 90 : 270,
                               text: axis.fTitle, color: title_color, draw_g: title_g });
            } else {
               title_offest_k *= side*pad_h;

               title_shift_x = Math.round(center ? w/2 : (xor_reverse ? 0 : w));
               title_shift_y = Math.round(title_offest_k*axis.fTitleOffset);
               this.drawText({ align: this.title_align+";middle",
                               rotate: (rotate<0) ? 180 : 0,
                               text: axis.fTitle, color: title_color, draw_g: title_g });
            }

            if (vertical && (axis.fTitleOffset == 0) && ('getBoundingClientRect' in axis_g.node()))
               axis_rect = axis_g.node().getBoundingClientRect();

            this.addTitleDrag(title_g, vertical, title_offest_k, swap_side, vertical ? h : w);

            return this.finishTextDrawing(title_g);

         }).then(() => {

            if (title_g) {
               // fine-tuning of title position when possible
               if (axis_rect) {
                  let title_rect = title_g.node().getBoundingClientRect();
                  if ((axis_rect.left != axis_rect.right) && (title_rect.left != title_rect.right))
                     title_shift_x = (side > 0) ? Math.round(axis_rect.left - title_rect.right - title_fontsize*0.3) :
                                                  Math.round(axis_rect.right - title_rect.left + title_fontsize*0.3);
                  else
                     title_shift_x = -1 * Math.round(((side > 0) ? (labeloffset + labelMaxWidth) : 0) + title_fontsize*0.7);
               }

               title_g.attr('transform', 'translate(' + title_shift_x + ',' + title_shift_y + ')')
                      .property('shift_x', title_shift_x)
                      .property('shift_y', title_shift_y);
            }

            return this;
         });
      }

      /** @summary Convert TGaxis position into NDC to fix it when frame zoomed */
      convertTo(opt) {
         let gaxis = this.getObject(),
             x1 = this.axisToSvg("x", gaxis.fX1),
             y1 = this.axisToSvg("y", gaxis.fY1),
             x2 = this.axisToSvg("x", gaxis.fX2),
             y2 = this.axisToSvg("y", gaxis.fY2);

         if (opt == "ndc") {
             let pw = this.getPadPainter().getPadWidth(),
                 ph = this.getPadPainter().getPadHeight();

             gaxis.fX1 = x1 / pw;
             gaxis.fX2 = x2 / pw;
             gaxis.fY1 = (ph - y1) / ph;
             gaxis.fY2 = (ph - y2)/ ph;
             this.use_ndc = true;
         } else if (opt == "frame") {
            let rect = this.getFramePainter().getFrameRect();
            gaxis.fX1 = (x1 - rect.x) / rect.width;
            gaxis.fX2 = (x2 - rect.x) / rect.width;
            gaxis.fY1 = (y1 - rect.y) / rect.height;
            gaxis.fY2 = (y2 - rect.y) / rect.height;
            this.bind_frame = true;
         }
      }

      /** @summary Redraw axis, used in standalone mode for TGaxis */
      redraw() {

         let gaxis = this.getObject(), x1, y1, x2, y2;

         if (this.bind_frame) {
            let rect = this.getFramePainter().getFrameRect();
            x1 = Math.round(rect.x + gaxis.fX1 * rect.width);
            x2 = Math.round(rect.x + gaxis.fX2 * rect.width);
            y1 = Math.round(rect.y + gaxis.fY1 * rect.height);
            y2 = Math.round(rect.y + gaxis.fY2 * rect.height);
         } else {
             x1 = this.axisToSvg("x", gaxis.fX1, this.use_ndc);
             y1 = this.axisToSvg("y", gaxis.fY1, this.use_ndc);
             x2 = this.axisToSvg("x", gaxis.fX2, this.use_ndc);
             y2 = this.axisToSvg("y", gaxis.fY2, this.use_ndc);
         }
         let w = x2 - x1, h = y1 - y2,
             vertical = Math.abs(w) < Math.abs(h),
             sz = vertical ? h : w,
             reverse = false,
             min = gaxis.fWmin, max = gaxis.fWmax;

         if (sz < 0) {
            reverse = true;
            sz = -sz;
            if (vertical) y2 = y1; else x1 = x2;
         }

         this.configureAxis(vertical ? "yaxis" : "xaxis", min, max, min, max, vertical, [0, sz], {
            time_scale: gaxis.fChopt.indexOf("t") >= 0,
            log: (gaxis.fChopt.indexOf("G") >= 0) ? 1 : 0,
            reverse: reverse,
            swap_side: reverse
         });

         this.createG();

         return this.drawAxis(this.getG(), Math.abs(w), Math.abs(h), `translate(${x1},${y2})`);
      }

      /** @summary draw TGaxis object */
      static draw(dom, obj, opt) {
         let painter = new TAxisPainter(dom, obj, false);
         painter.disable_zooming = true;

         return jsrp.ensureTCanvas(painter, false)
                .then(() => { if (opt) painter.convertTo(opt); return painter.redraw(); });
      }

   } // TAxisPainter

   // ===============================================

   /**
    * @summary Painter class for TFrame, main handler for interactivity
    *
    * @memberof JSROOT
    * @private
    */

   class TFramePainter extends JSROOT.ObjectPainter {

      /** @summary constructor
        * @param {object|string} dom - DOM element for drawing or element id
        * @param {object} tframe - TFrame object */

      constructor(dom, tframe) {
         super(dom, (tframe && tframe.$dummy) ? null : tframe);
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

      /** @summary Returns frame painter - object itself */
      getFramePainter() { return this; }

      /** @summary Returns true if it is ROOT6 frame
        * @private */
      is_root6() { return true; }

      /** @summary Returns frame or sub-objects, used in GED editor */
      getObject(place) {
         if (place === "xaxis") return this.xaxis;
         if (place === "yaxis") return this.yaxis;
         return super.getObject();
      }

      /** @summary Set active flag for frame - can block some events
        * @private */
      setFrameActive(on) {
         this.enabledKeys = on && JSROOT.settings.HandleKeys ? true : false;
         // used only in 3D mode where control is used
         if (this.control)
            this.control.enableKeys = this.enabledKeys;
      }

      /** @summary Shrink frame size
        * @private */
      shrinkFrame(shrink_left, shrink_right) {
         this.fX1NDC += shrink_left;
         this.fX2NDC -= shrink_right;
      }

      /** @summary Set position of last context menu event */
      setLastEventPos(pnt) {
         this.fLastEventPnt = pnt;
      }

      /** @summary Return position of last event
        * @private */
      getLastEventPos() { return this.fLastEventPnt; }

      /** @summary Returns coordinates transformation func */
      getProjectionFunc() {
         switch (this.projection) {
            // Aitoff2xy
            case 1: return (l, b) => {
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
            };
            // mercator
            case 2: return (l, b) => { return { x: l, y: Math.log(Math.tan((Math.PI/2 + b/180*Math.PI)/2)) }; };
            // sinusoidal
            case 3: return (l, b) => { return { x: l*Math.cos(b/180*Math.PI), y: b } };
            // parabolic
            case 4: return (l, b) => { return { x: l*(2.*Math.cos(2*b/180*Math.PI/3) - 1), y: 180*Math.sin(b/180*Math.PI/3) }; };
         }
      }

      /** @summary Rcalculate frame ranges using specified projection functions */
      recalculateRange(Proj) {
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

      /** @summary Configure frame axes ranges */
      setAxesRanges(xaxis, xmin, xmax, yaxis, ymin, ymax, zaxis, zmin, zmax) {
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

      /** @summary Configure secondary frame axes ranges */
      setAxes2Ranges(second_x, xaxis, xmin, xmax, second_y, yaxis, ymin, ymax) {
         if (second_x) {
            this.x2axis = xaxis;
            this.x2min = xmin;
            this.x2max = xmax;
         }
         if (second_y) {
            this.y2axis = yaxis;
            this.y2min = ymin;
            this.y2max = ymax;
         }
      }

      /** @summary Retuns associated axis object */
      getAxis(name) {
         switch(name) {
            case "x": return this.xaxis;
            case "y": return this.yaxis;
            case "z": return this.zaxis;
            case "x2": return this.x2axis;
            case "y2": return this.y2axis;
         }
         return null;
      }

      /** @summary Apply axis zooming from pad user range
        * @private */
      applyPadUserRange(pad, name) {
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

         if ((umin >= umax) || (Math.abs(umin) < eps && Math.abs(umax-1) < eps)) return;

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
      createXY(opts) {

         this.cleanXY(); // remove all previous configurations

         if (!opts) opts = { ndim: 1 };

         this.swap_xy = opts.swap_xy || false;
         this.reverse_x = opts.reverse_x || false;
         this.reverse_y = opts.reverse_y || false;

         this.logx = this.logy = 0;

         let w = this.getFrameWidth(), h = this.getFrameHeight(),
             pp = this.getPadPainter(),
             pad = pp.getRootPad();

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

            const applyAxisZoom = name => {
               if (this.zoomChangedInteractive(name)) return;
               this[`zoom_${name}min`] = this[`zoom_${name}max`] = 0;

               const axis = this.getAxis(name);

               if (axis && axis.TestBit(JSROOT.EAxisBits.kAxisRange)) {
                  if ((axis.fFirst !== axis.fLast) && ((axis.fFirst > 1) || (axis.fLast < axis.fNbins))) {
                     this[`zoom_${name}min`] = axis.fFirst > 1 ? axis.GetBinLowEdge(axis.fFirst) : axis.fXmin;
                     this[`zoom_${name}max`] = axis.fLast < axis.fNbins ? axis.GetBinLowEdge(axis.fLast + 1) : axis.fXmax;
                     // reset user range for main painter
                     axis.InvertBit(JSROOT.EAxisBits.kAxisRange);
                     axis.fFirst = 1; axis.fLast = axis.fNbins;
                  }
               }
            };

            applyAxisZoom('x');
            if (opts.ndim > 1) applyAxisZoom('y');
            if (opts.ndim > 2) applyAxisZoom('z');

            if (opts.check_pad_range === "pad_range") {
               let canp = this.getCanvPainter();
               // ignore range set in the online canvas
               if (!canp || !canp.online_canvas) {
                  this.applyPadUserRange(pad, 'x');
                  this.applyPadUserRange(pad, 'y');
               }
            }
         }

         if ((opts.zoom_ymin != opts.zoom_ymax) && (this.zoom_ymin == this.zoom_ymax) && !this.zoomChangedInteractive("y")) {
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
         this.recalculateRange(opts.Proj);

         this.x_handle = new TAxisPainter(this.getDom(), this.xaxis, true);
         this.x_handle.setPadName(this.getPadName());

         this.x_handle.configureAxis("xaxis", this.xmin, this.xmax, this.scale_xmin, this.scale_xmax, this.swap_xy, this.swap_xy ? [0,h] : [0,w],
                                         { reverse: this.reverse_x,
                                           log: this.swap_xy ? pad.fLogy : pad.fLogx,
                                           symlog: this.swap_xy ? opts.symlog_y : opts.symlog_x,
                                           logcheckmin: this.swap_xy,
                                           logminfactor: 0.0001 });

         this.x_handle.assignFrameMembers(this, "x");

         this.y_handle = new TAxisPainter(this.getDom(), this.yaxis, true);
         this.y_handle.setPadName(this.getPadName());

         this.y_handle.configureAxis("yaxis", this.ymin, this.ymax, this.scale_ymin, this.scale_ymax, !this.swap_xy, this.swap_xy ? [0,w] : [0,h],
                                         { reverse: this.reverse_y,
                                           log: this.swap_xy ? pad.fLogx : pad.fLogy,
                                           symlog: this.swap_xy ? opts.symlog_x : opts.symlog_y,
                                           logcheckmin: (opts.ndim < 2) || this.swap_xy,
                                           log_min_nz: opts.ymin_nz && (opts.ymin_nz < 0.01*this.ymax) ? 0.3 * opts.ymin_nz : 0,
                                           logminfactor: 3e-4 });

         this.y_handle.assignFrameMembers(this, "y");

         this.setRootPadRange(pad);
      }

      /** @summary Create x,y objects for drawing of second axes
        * @private */
      createXY2(opts) {

         if (!opts) opts = {};

         this.reverse_x2 = opts.reverse_x || false;
         this.reverse_y2 = opts.reverse_y || false;

         this.logx2 = this.logy2 = 0;

         let w = this.getFrameWidth(), h = this.getFrameHeight(),
             pp = this.getPadPainter(),
             pad = pp.getRootPad();

         if (opts.second_x) {
            this.scale_x2min = this.x2min;
            this.scale_x2max = this.x2max;
         }

         if (opts.second_y) {
            this.scale_y2min = this.y2min;
            this.scale_y2max = this.y2max;
         }

         if (opts.extra_y_space && opts.second_y) {
            let log_scale = this.swap_xy ? pad.fLogx : pad.fLogy;
            if (log_scale && (this.scale_y2max > 0))
               this.scale_y2max = Math.exp(Math.log(this.scale_y2max)*1.1);
            else
               this.scale_y2max += (this.scale_y2max - this.scale_y2min)*0.1;
         }

         if ((this.zoom_x2min != this.zoom_x2max) && opts.second_x) {
            this.scale_x2min = this.zoom_x2min;
            this.scale_x2max = this.zoom_x2max;
         }

         if ((this.zoom_y2min != this.zoom_y2max) && opts.second_y) {
            this.scale_y2min = this.zoom_y2min;
            this.scale_y2max = this.zoom_y2max;
         }

         if (opts.second_x) {
            this.x2_handle = new TAxisPainter(this.getDom(), this.x2axis, true);
            this.x2_handle.setPadName(this.getPadName());

            this.x2_handle.configureAxis("x2axis", this.x2min, this.x2max, this.scale_x2min, this.scale_x2max, this.swap_xy, this.swap_xy ? [0,h] : [0,w],
                                            { reverse: this.reverse_x2,
                                              log: this.swap_xy ? pad.fLogy : pad.fLogx,
                                              logcheckmin: this.swap_xy,
                                              logminfactor: 0.0001 });
            this.x2_handle.assignFrameMembers(this,"x2");
         }

         if (opts.second_y) {
            this.y2_handle = new TAxisPainter(this.getDom(), this.y2axis, true);
            this.y2_handle.setPadName(this.getPadName());

            this.y2_handle.configureAxis("y2axis", this.y2min, this.y2max, this.scale_y2min, this.scale_y2max, !this.swap_xy, this.swap_xy ? [0,w] : [0,h],
                                            { reverse: this.reverse_y2,
                                              log: this.swap_xy ? pad.fLogx : pad.fLogy,
                                              logcheckmin: (opts.ndim < 2) || this.swap_xy,
                                              log_min_nz: opts.ymin_nz && (opts.ymin_nz < 0.01*this.y2max) ? 0.3 * opts.ymin_nz : 0,
                                              logminfactor: 3e-4 });

            this.y2_handle.assignFrameMembers(this,"y2");
         }
      }

      /** @summary Return functions to create x/y points based on coordinates
        * @desc In default case returns frame painter itself
        * @private */
      getGrFuncs(second_x, second_y) {
         let use_x2 = second_x && this.grx2,
             use_y2 = second_y && this.gry2;
         if (!use_x2 && !use_y2) return this;

         return {
            use_x2: use_x2,
            grx: use_x2 ? this.grx2 : this.grx,
            logx: this.logx,
            x_handle: use_x2 ? this.x2_handle : this.x_handle,
            scale_xmin: use_x2 ? this.scale_x2min : this.scale_xmin,
            scale_xmax: use_x2 ? this.scale_x2max : this.scale_xmax,
            use_y2: use_y2,
            gry: use_y2 ? this.gry2 : this.gry,
            logy: this.logy,
            y_handle: use_y2 ? this.y2_handle : this.y_handle,
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

      /** @summary Set selected range back to TPad object
        * @private */
      setRootPadRange(pad, is3d) {
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


      /** @summary Draw axes grids
        * @desc Called immediately after axes drawing */
      drawGrids() {

         let layer = this.getFrameSvg().select(".grid_layer");

         layer.selectAll(".xgrid").remove();
         layer.selectAll(".ygrid").remove();

         let pp = this.getPadPainter(),
             pad = pp ? pp.getRootPad(true) : null,
             h = this.getFrameHeight(),
             w = this.getFrameWidth(),
             grid_style = JSROOT.gStyle.fGridStyle;

         if ((grid_style < 0) || (grid_style >= jsrp.root_line_styles.length)) grid_style = 11;

         // add a grid on x axis, if the option is set
         if (pad && pad.fGridx && this.x_handle) {
            let gridx = "";
            for (let n = 0; n < this.x_handle.ticks.length; ++n)
               if (this.swap_xy)
                  gridx += `M0,${this.x_handle.ticks[n]}h${w}`;
               else
                  gridx += `M${this.x_handle.ticks[n]},0v${h}`;

            let colid = (JSROOT.gStyle.fGridColor > 0) ? JSROOT.gStyle.fGridColor : (this.getAxis("x") ? this.getAxis("x").fAxisColor : 1),
                grid_color = this.getColor(colid) || "black";

            if (gridx.length > 0)
              layer.append("svg:path")
                   .attr("class", "xgrid")
                   .attr("d", gridx)
                   .style("stroke", grid_color)
                   .style("stroke-width", JSROOT.gStyle.fGridWidth)
                   .style("stroke-dasharray", jsrp.root_line_styles[grid_style]);
         }

         // add a grid on y axis, if the option is set
         if (pad && pad.fGridy && this.y_handle) {
            let gridy = "";
            for (let n = 0; n < this.y_handle.ticks.length; ++n)
               if (this.swap_xy)
                  gridy += `M${this.y_handle.ticks[n]},0v${h}`;
               else
                  gridy += `M0,${this.y_handle.ticks[n]}h${w}`;

            let colid = (JSROOT.gStyle.fGridColor > 0) ? JSROOT.gStyle.fGridColor : (this.getAxis("y") ? this.getAxis("y").fAxisColor : 1),
                grid_color = this.getColor(colid) || "black";

            if (gridy.length > 0)
              layer.append("svg:path")
                   .attr("class", "ygrid")
                   .attr("d", gridy)
                   .style("stroke", grid_color)
                   .style("stroke-width",JSROOT.gStyle.fGridWidth)
                   .style("stroke-dasharray", jsrp.root_line_styles[grid_style]);
         }
      }

      /** @summary Converts "raw" axis value into text */
      axisAsText(axis, value) {
         let handle = this[axis+"_handle"];

         if (handle)
            return handle.axisAsText(value, JSROOT.settings[axis.toUpperCase() + "ValuesFormat"]);

         return value.toPrecision(4);
      }

      /** @summary Identify if requested axes are drawn
        * @desc Checks if x/y axes are drawn. Also if second side is already there */
      hasDrawnAxes(second_x, second_y) {
         return !second_x && !second_y ? this.axes_drawn : false;
      }

      /** @summary draw axes, return Promise which ready when drawing is completed  */
      drawAxes(shrink_forbidden,
                                                  disable_x_draw, disable_y_draw,
                                                  AxisPos, has_x_obstacle, has_y_obstacle) {

         this.cleanAxesDrawings();

         if ((this.xmin == this.xmax) || (this.ymin == this.ymax))
            return Promise.resolve(false);

         if (AxisPos === undefined) AxisPos = 0;

         let layer = this.getFrameSvg().select(".axis_layer"),
             w = this.getFrameWidth(),
             h = this.getFrameHeight(),
             pp = this.getPadPainter(),
             pad = pp.getRootPad(true);

         this.x_handle.invert_side = (AxisPos >= 10);
         this.x_handle.lbls_both_sides = !this.x_handle.invert_side && pad && (pad.fTickx > 1); // labels on both sides
         this.x_handle.has_obstacle = has_x_obstacle;

         this.y_handle.invert_side = ((AxisPos % 10) === 1);
         this.y_handle.lbls_both_sides = !this.y_handle.invert_side && pad && (pad.fTicky > 1); // labels on both sides
         this.y_handle.has_obstacle = has_y_obstacle;

         let draw_horiz = this.swap_xy ? this.y_handle : this.x_handle,
             draw_vertical = this.swap_xy ? this.x_handle : this.y_handle,
             draw_promise;

         if (!disable_x_draw || !disable_y_draw) {
            let pp = this.getPadPainter();
            if (pp && pp._fast_drawing) disable_x_draw = disable_y_draw = true;
         }

         if (disable_x_draw && disable_y_draw) {
            draw_promise = Promise.resolve(true);
         } else {

            let can_adjust_frame = !shrink_forbidden && JSROOT.settings.CanAdjustFrame;

            let promise1 = draw_horiz.drawAxis(layer, w, h,
                                               draw_horiz.invert_side ? undefined : `translate(0,${h})`,
                                               pad && pad.fTickx ? -h : 0, disable_x_draw,
                                               undefined, false);

            let promise2 = draw_vertical.drawAxis(layer, w, h,
                                                  draw_vertical.invert_side ? `translate(${w})` : undefined,
                                                  pad && pad.fTicky ? w : 0, disable_y_draw,
                                                  draw_vertical.invert_side ? 0 : this._frame_x, can_adjust_frame);

            draw_promise = Promise.all([promise1, promise2]).then(() => {

               this.drawGrids();

               if (can_adjust_frame) {
                  let shrink = 0., ypos = draw_vertical.position;

                  if ((-0.2*w < ypos) && (ypos < 0)) {
                     shrink = -ypos/w + 0.001;
                     this.shrink_frame_left += shrink;
                  } else if ((ypos>0) && (ypos<0.3*w) && (this.shrink_frame_left > 0) && (ypos/w > this.shrink_frame_left)) {
                     shrink = -this.shrink_frame_left;
                     this.shrink_frame_left = 0.;
                  }

                  if (shrink != 0) {
                     this.shrinkFrame(shrink, 0);
                     this.redraw();
                     return this.drawAxes(true);
                  }
               }
            });
         }

         return draw_promise.then(() => {
            if (!shrink_forbidden)
               this.axes_drawn = true;
            return true;
         });
      }

      /** @summary draw second axes (if any)  */
      drawAxes2(second_x, second_y) {

         let layer = this.getFrameSvg().select(".axis_layer"),
             w = this.getFrameWidth(),
             h = this.getFrameHeight(),
             pp = this.getPadPainter(),
             pad = pp.getRootPad(true);

         if (second_x) {
            this.x2_handle.invert_side = true;
            this.x2_handle.lbls_both_sides = false;
            this.x2_handle.has_obstacle = false;
         }

         if (second_y) {
            this.y2_handle.invert_side = true;
            this.y2_handle.lbls_both_sides = false;
         }

         let draw_horiz = this.swap_xy ? this.y2_handle : this.x2_handle,
             draw_vertical = this.swap_xy ? this.x2_handle : this.y2_handle;

         if (draw_horiz || draw_vertical) {
            let pp = this.getPadPainter();
            if (pp && pp._fast_drawing) draw_horiz = draw_vertical = null;
         }

         let promise1 = true, promise2 = true;

         if (draw_horiz)
            promise1 = draw_horiz.drawAxis(layer, w, h,
                                           draw_horiz.invert_side ? undefined : `translate(0,${h})`,
                                           pad && pad.fTickx ? -h : 0, false,
                                           undefined, false);

         if (draw_vertical)
            promise2 = draw_vertical.drawAxis(layer, w, h,
                                               draw_vertical.invert_side ? `translate(${w})` : undefined,
                                               pad && pad.fTicky ? w : 0, false,
                                               draw_vertical.invert_side ? 0 : this._frame_x, false);

         return Promise.all([promise1, promise2]);
      }


      /** @summary Update frame attributes
        * @private */
      updateAttributes(force) {
         let pp = this.getPadPainter(),
             pad = pp ? pp.getRootPad(true) : null,
             tframe = this.getObject();

         if ((this.fX1NDC === undefined) || (force && !this.modified_NDC)) {
            if (!pad) {
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
            if (!tframe && this.fillatt.empty() && pp && pp.iscan)
               this.fillatt.setSolidColor('white');
         }

         if (!tframe && pad && (pad.fFrameLineColor !== undefined))
            this.createAttLine({ color: pad.fFrameLineColor, width: pad.fFrameLineWidth, style: pad.fFrameLineStyle });
         else
            this.createAttLine({ attr: tframe, color: 'black' });
      }

      /** @summary Function called at the end of resize of frame
        * @desc One should apply changes to the pad
        * @private */
      sizeChanged() {

         let pp = this.getPadPainter(),
             pad = pp ? pp.getRootPad(true) : null;

         if (pad) {
            pad.fLeftMargin = this.fX1NDC;
            pad.fRightMargin = 1 - this.fX2NDC;
            pad.fBottomMargin = this.fY1NDC;
            pad.fTopMargin = 1 - this.fY2NDC;
            this.setRootPadRange(pad);
         }

         this.interactiveRedraw("pad", "frame");
      }

       /** @summary Remove all kinds of X/Y function for axes transformation */
      cleanXY() {
         delete this.grx;
         delete this.gry;
         delete this.grz;

         if (this.x_handle) {
            this.x_handle.cleanup();
            delete this.x_handle;
         }

         if (this.y_handle) {
            this.y_handle.cleanup();
            delete this.y_handle;
         }

         if (this.z_handle) {
            this.z_handle.cleanup();
            delete this.z_handle;
         }

         // these are drawing of second axes
         delete this.grx2;
         delete this.gry2;

         if (this.x2_handle) {
            this.x2_handle.cleanup();
            delete this.x2_handle;
         }

         if (this.y2_handle) {
            this.y2_handle.cleanup();
            delete this.y2_handle;
         }

      }

      /** @summary remove all axes drawings */
      cleanAxesDrawings() {
         if (this.x_handle) this.x_handle.removeG();
         if (this.y_handle) this.y_handle.removeG();
         if (this.z_handle) this.z_handle.removeG();
         if (this.x2_handle) this.x2_handle.removeG();
         if (this.y2_handle) this.y2_handle.removeG();

         let g = this.getG();
         if (g) {
            g.select(".grid_layer").selectAll("*").remove();
            g.select(".axis_layer").selectAll("*").remove();
         }
         this.axes_drawn = false;
      }

      /** @summary Returns frame rectangle plus extra info for hint display */
      cleanFrameDrawings() {

         // cleanup all 3D drawings if any
         if (typeof this.create3DScene === 'function')
            this.create3DScene(-1);

         this.cleanAxesDrawings();
         this.cleanXY();

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

         delete this.draw_g; // frame <g> element managet by the pad

         if (this.keys_handler) {
            window.removeEventListener('keydown', this.keys_handler, false);
            this.keys_handler = null;
         }
      }

      /** @summary Cleanup frame */
      cleanup() {
         this.cleanFrameDrawings();
         delete this._click_handler;
         delete this._dblclick_handler;
         delete this.enabledKeys;

         let pp = this.getPadPainter();
         if (pp && (pp.frame_painter_ref === this))
            delete pp.frame_painter_ref;

         super.cleanup();
      }

      /** @summary Redraw TFrame */
      redraw(/* reason */) {
         let pp = this.getPadPainter();
         if (pp) pp.frame_painter_ref = this; // keep direct reference to the frame painter

         // first update all attributes from objects
         this.updateAttributes();

         let rect = pp ? pp.getPadRect() : { width: 10, height: 10},
             lm = Math.round(rect.width * this.fX1NDC),
             w = Math.round(rect.width * (this.fX2NDC - this.fX1NDC)),
             tm = Math.round(rect.height * (1 - this.fY2NDC)),
             h = Math.round(rect.height * (this.fY2NDC - this.fY1NDC)),
             rotate = false, fixpos = false, trans;

         if (pp && pp.options) {
            if (pp.options.RotateFrame) rotate = true;
            if (pp.options.FixFrame) fixpos = true;
         }

         if (rotate) {
            trans = `rotate(-90,${lm},${tm}) translate(${lm-h},${tm})`;
            let d = w; w = h; h = d;
         } else {
            trans = `translate(${lm},${tm})`;
         }

         this._frame_x = lm;
         this._frame_y = tm;
         this._frame_width = w;
         this._frame_height = h;
         this._frame_rotate = rotate;
         this._frame_fixpos = fixpos;

         if (this.mode3d) return; // no need to create any elements in 3d mode

         // this is svg:g object - container for every other items belonging to frame
         this.draw_g = this.getFrameSvg();

         let top_rect, main_svg;

         if (this.draw_g.empty()) {

            this.draw_g = this.getLayerSvg("primitives_layer").append("svg:g").attr("class", "root_frame");

            // empty title on the frame required to suppress title of the canvas
            if (!JSROOT.batch_mode)
               this.draw_g.append("svg:title").text("");

            top_rect = this.draw_g.append("svg:path");

            // append for the moment three layers - for drawing and axis
            this.draw_g.append('svg:g').attr('class','grid_layer');

            main_svg = this.draw_g.append('svg:svg')
                              .attr('class','main_layer')
                              .attr("x", 0)
                              .attr("y", 0)
                              .attr('overflow', 'hidden');

            this.draw_g.append('svg:g').attr('class', 'axis_layer');
            this.draw_g.append('svg:g').attr('class', 'upper_layer');
         } else {
            top_rect = this.draw_g.select("path");
            main_svg = this.draw_g.select(".main_layer");
         }

         this.axes_drawn = false;

         this.draw_g.attr("transform", trans);

         top_rect.attr("d", `M0,0H${w}V${h}H0Z`)
                 .call(this.fillatt.func)
                 .call(this.lineatt.func);

         main_svg.attr("width", w)
                 .attr("height", h)
                 .attr("viewBox", `0 0 ${w} ${h}`);

         if (JSROOT.batch_mode) return;

         top_rect.style("pointer-events", "visibleFill"); // let process mouse events inside frame

         JSROOT.require(['interactive']).then(inter => {
            inter.FrameInteractive.assign(this);
            this.addBasicInteractivity();
         });
      }

      /** @summary Change log state of specified axis
        * @param {number} value - 0 (linear), 1 (log) or 2 (log2) */
      changeAxisLog(axis, value) {
         let pp = this.getPadPainter(),
             pad = pp ? pp.getRootPad(true) : null;
         if (!pad) return;

         pp._interactively_changed = true;

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

         this.interactiveRedraw("pad", "log"+axis);
      }

      /** @summary Toggle log state on the specified axis */
      toggleAxisLog(axis) {
         this.changeAxisLog(axis, "toggle");
      }

      /** @summary Fill context menu for the frame
        * @desc It could be appended to the histogram menus */
      fillContextMenu(menu, kind, obj) {
         let main = this.getMainPainter(),
             pp = this.getPadPainter(),
             pad = pp ? pp.getRootPad(true) : null;

         if ((kind=="x") || (kind=="y") || (kind=="z") || (kind == "x2") || (kind == "y2")) {
            let faxis = obj || this[kind+'axis'];
            menu.add("header: " + kind.toUpperCase() + " axis");
            menu.add("Unzoom", () => this.unzoom(kind));
            if (pad) {
               menu.add("sub:SetLog "+kind[0]);
               menu.addchk(pad["fLog" + kind[0]] == 0, "linear", () => this.changeAxisLog(kind[0], 0));
               menu.addchk(pad["fLog" + kind[0]] == 1, "log", () => this.changeAxisLog(kind[0], 1));
               menu.addchk(pad["fLog" + kind[0]] == 2, "log2", () => this.changeAxisLog(kind[0], 2));
               menu.add("endsub:");
            }
            menu.addchk(faxis.TestBit(JSROOT.EAxisBits.kMoreLogLabels), "More log",
                  () => { faxis.InvertBit(JSROOT.EAxisBits.kMoreLogLabels); this.redrawPad(); });
            menu.addchk(faxis.TestBit(JSROOT.EAxisBits.kNoExponent), "No exponent",
                  () => { faxis.InvertBit(JSROOT.EAxisBits.kNoExponent); this.redrawPad(); });

            if ((kind === "z") && main && main.options && main.options.Zscale)
               if (typeof main.fillPaletteMenu == 'function')
                  main.fillPaletteMenu(menu);

            if (faxis) {
               let handle = this[kind+"_handle"];

               if (handle && (handle.kind == "labels") && (faxis.fNbins > 20))
                  menu.add("Find label", () => menu.input("Label id").then(id => {
                     if (!id) return;
                     for (let bin = 0; bin < faxis.fNbins; ++bin) {
                        let lbl = handle.formatLabels(bin);
                        if (lbl == id)
                           return this.zoom(kind, Math.max(0, bin - 4), Math.min(faxis.fNbins, bin+5));
                      }
                  }));

               menu.addTAxisMenu(main || this, faxis, kind);
            }
            return true;
         }

         const alone = menu.size() == 0;

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

         if (pad) {
            menu.addchk(pad.fLogx, "SetLogx", () => this.toggleAxisLog("x"));
            menu.addchk(pad.fLogy, "SetLogy", () => this.toggleAxisLog("y"));

            if (main && (typeof main.getDimension === 'function') && (main.getDimension() > 1))
               menu.addchk(pad.fLogz, "SetLogz", () => this.toggleAxisLog("z"));
            menu.add("separator");
         }

         menu.addchk(this.isTooltipAllowed(), "Show tooltips", () => this.setTooltipAllowed("toggle"));
         menu.addAttributesMenu(this, alone ? "" : "Frame ");
         menu.add("separator");
         menu.add("Save as frame.png", () => pp.saveAs("png", 'frame', 'frame.png'));
         menu.add("Save as frame.svg", () => pp.saveAs("svg", 'frame', 'frame.svg'));

         return true;
      }

      /** @summary Fill option object used in TWebCanvas
        * @private */
      fillWebObjectOptions(res) {
         if (!res) {
            if (!this.snapid) return null;
            res = { _typename: "TWebObjectOptions", snapid: this.snapid.toString(), opt: this.getDrawOpt(), fcust: "", fopt: [] };
          }

         res.fcust = "frame";
         res.fopt = [this.scale_xmin || 0, this.scale_ymin || 0, this.scale_xmax || 0, this.scale_ymax || 0];
         return res;
      }

      /** @summary Returns frame width */
      getFrameWidth() { return this._frame_width || 0; }

      /** @summary Returns frame height */
      getFrameHeight() { return this._frame_height || 0; }

      /** @summary Returns frame rectangle plus extra info for hint display */
      getFrameRect() {
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

      /** @summary Configure user-defined click handler
        * @desc Function will be called every time when frame click was perfromed
        * As argument, tooltip object with selected bins will be provided
        * If handler function returns true, default handling of click will be disabled */
      configureUserClickHandler(handler) {
         this._click_handler = handler && (typeof handler == 'function') ? handler : null;
      }

      /** @summary Configure user-defined dblclick handler
        * @desc Function will be called every time when double click was called
        * As argument, tooltip object with selected bins will be provided
        * If handler function returns true, default handling of dblclick (unzoom) will be disabled */
      configureUserDblclickHandler(handler) {
         this._dblclick_handler = handler && (typeof handler == 'function') ? handler : null;
      }

       /** @summary Function can be used for zooming into specified range
         * @desc if both limits for each axis 0 (like xmin==xmax==0), axis will be unzoomed
         * @param {number} xmin
         * @param {number} xmax
         * @param {number} [ymin]
         * @param {number} [ymax]
         * @param {number} [zmin]
         * @param {number} [zmax]
         * @returns {Promise} with boolean flag if zoom operation was performed */
      zoom(xmin, xmax, ymin, ymax, zmin, zmax) {

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
            if (zmin <= this.zmin) { zmin = this.zmin; cnt++; }
            if (zmax >= this.zmax) { zmax = this.zmax; cnt++; }
            if (cnt === 2) { zoom_z = false; unzoom_z = true; }
         } else {
            unzoom_z = (zmin === zmax) && (zmin === 0);
         }

         let changed = false;

         // first process zooming (if any)
         if (zoom_x || zoom_y || zoom_z)
            this.forEachPainter(obj => {
               if (typeof obj.canZoomInside != 'function') return;
               if (zoom_x && obj.canZoomInside("x", xmin, xmax)) {
                  this.zoom_xmin = xmin;
                  this.zoom_xmax = xmax;
                  changed = true;
                  zoom_x = false;
               }
               if (zoom_y && obj.canZoomInside("y", ymin, ymax)) {
                  this.zoom_ymin = ymin;
                  this.zoom_ymax = ymax;
                  changed = true;
                  zoom_y = false;
               }
               if (zoom_z && obj.canZoomInside("z", zmin, zmax)) {
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
               let pp = this.getPadPainter();
               if (pp && pp.painters)
                  pp.painters.forEach(painter => {
                     if (painter && (typeof painter.unzoomUserRange == 'function'))
                        if (painter.unzoomUserRange(unzoom_x, unzoom_y, unzoom_z)) changed = true;
               });
            }
         }

         if (!changed) return Promise.resolve(false);

         return this.interactiveRedraw("pad", "zoom").then(() => true);
      }

      /** @summary Provide zooming of single axis
        * @desc One can specify names like x/y/z but also second axis x2 or y2 */
      zoomSingle(name, vmin, vmax) {
         // disable zooming when axis conversion is enabled
         if (this.projection || !this[name+"_handle"]) return Promise.resolve(false);

         let zoom_v = (vmin !== vmax), unzoom_v = false;

         if (zoom_v) {
            let cnt = 0;
            if (vmin <= this[name+"min"]) { vmin = this[name+"min"]; cnt++; }
            if (vmax >= this[name+"max"]) { vmax = this[name+"max"]; cnt++; }
            if (cnt === 2) { zoom_v = false; unzoom_v = true; }
         } else {
            unzoom_v = (vmin === vmax) && (vmin === 0);
         }

         let changed = false;

         // first process zooming
         if (zoom_v)
            this.forEachPainter(obj => {
               if (typeof obj.canZoomInside != 'function') return;
               if (zoom_v && obj.canZoomInside(name[0], vmin, vmax)) {
                  this["zoom_" + name + "min"] = vmin;
                  this["zoom_" + name + "max"] = vmax;
                  changed = true;
                  zoom_v = false;
               }
            });

         // and process unzoom, if any
         if (unzoom_v) {
            if (this["zoom_" + name + "min"] !== this["zoom_" + name + "max"]) changed = true;
            this["zoom_" + name + "min"] = this["zoom_" + name + "max"] = 0;
         }

         if (!changed) return Promise.resolve(false);

         return this.interactiveRedraw("pad", "zoom").then(() => true);
      }

      /** @summary Checks if specified axis zoomed */
      isAxisZoomed(axis) {
         return this['zoom_'+axis+'min'] !== this['zoom_'+axis+'max'];
      }

      /** @summary Unzoom speicified axes
        * @returns {Promise} with boolean flag if zooming changed */
      unzoom(dox, doy, doz) {
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
      zoomChangedInteractive(axis, value) {
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

      /** @summary Convert graphical coordinate into axis value */
      revertAxis(axis, pnt) {
         let handle = this[axis+"_handle"];
         return handle ? handle.revertPoint(pnt) : 0;
      }

      /** @summary Show axis status message
       * @desc method called normally when mouse enter main object element
       * @private */
      showAxisStatus(axis_name, evnt) {
         let taxis = this.getAxis(axis_name), hint_name = axis_name, hint_title = "TAxis",
             m = d3.pointer(evnt, this.getFrameSvg().node()), id = (axis_name=="x") ? 0 : 1;

         if (taxis) { hint_name = taxis.fName; hint_title = taxis.fTitle || ("TAxis object for " + axis_name); }
         if (this.swap_xy) id = 1-id;

         let axis_value = this.revertAxis(axis_name, m[id]);

         this.showObjectStatus(hint_name, hint_title, axis_name + " : " + this.axisAsText(axis_name, axis_value), m[0]+","+m[1]);
      }

      /** @summary Add interactive keys handlers
       * @private */
      addKeysHandler() {
         if (JSROOT.batch_mode) return;
         JSROOT.require(['interactive']).then(inter => {
            inter.FrameInteractive.assign(this);
            this.addKeysHandler();
         });
      }

      /** @summary Add interactive functionality to the frame
        * @private */
      addInteractivity(for_second_axes) {
         if (JSROOT.batch_mode || (!JSROOT.settings.Zooming && !JSROOT.settings.ContextMenu))
            return Promise.resolve(false);

         return JSROOT.require(['interactive']).then(inter => {
            inter.FrameInteractive.assign(this);
            return this.addInteractivity(for_second_axes);
         });
      }

      /** @summary draw TFrame object */
      static draw(dom, obj, opt) {
         let fp = new TFramePainter(dom, obj);
         return jsrp.ensureTCanvas(fp, false).then(() => {
            if (opt == "3d") fp.mode3d = true;
            fp.redraw();
            return fp;
         })
      }

   } // TFramePainter

   // ===========================================================================

   /**
     * @summary Painter for TPad object
     *
     * @memberof JSROOT
     * @private
     */

   class TPadPainter extends JSROOT.ObjectPainter {

      /** @summary constructor
        * @param {object|string} dom - DOM element for drawing or element id
        * @param {object} pad - TPad object to draw
        * @param {boolean} [iscan] - if TCanvas object */
      constructor(dom, pad, iscan) {
         super(dom, pad);
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
         this.forEachPainter = this.forEachPainterInPad;
      }

      /** @summary Indicates that is is Root6 pad painter
       * @private */
      isRoot6() { return true; }

      /** @summary Returns SVG element for the pad itself
       * @private */
      svg_this_pad() {
         return this.getPadSvg(this.this_pad_name);
      }

      /** @summary Returns main painter on the pad
        * @desc Typically main painter is TH1/TH2 object which is drawing axes
       * @private */
      getMainPainter() {
         return this.main_painter_ref || null;
      }

      /** @summary Assign main painter on the pad
        * @desc Typically main painter is TH1/TH2 object which is drawing axes
       * @private */
      setMainPainter(painter, force) {
         if (!this.main_painter_ref || force)
            this.main_painter_ref = painter;
      }

      /** @summary cleanup pad and all primitives inside */
      cleanup() {
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
         delete this.custom_palette;
         delete this._pad_x;
         delete this._pad_y;
         delete this._pad_width;
         delete this._pad_height;
         delete this._doing_draw;
         delete this._interactively_changed;

         this.painters = [];
         this.pad = null;
         this.this_pad_name = undefined;
         this.has_canvas = false;

         jsrp.selectActivePad({ pp: this, active: false });

         super.cleanup();
      }

      /** @summary Returns frame painter inside the pad
        * @private */
      getFramePainter() { return this.frame_painter_ref; }

      /** @summary get pad width */
      getPadWidth() { return this._pad_width || 0; }

      /** @summary get pad height */
      getPadHeight() { return this._pad_height || 0; }

      /** @summary get pad rect */
      getPadRect() {
         return {
            x: this._pad_x || 0,
            y: this._pad_y || 0,
            width: this.getPadWidth(),
            height: this.getPadHeight()
         }
      }

      /** @summary Returns frame coordiantes - also when frame is not drawn */
      getFrameRect() {
         let fp = this.getFramePainter();
         if (fp) return fp.getFrameRect();

         let w = this.getPadWidth(),
             h = this.getPadHeight(),
             rect = {};

         if (this.pad) {
            rect.szx = Math.round(Math.max(0.0, 0.5 - Math.max(this.pad.fLeftMargin, this.pad.fRightMargin))*w);
            rect.szy = Math.round(Math.max(0.0, 0.5 - Math.max(this.pad.fBottomMargin, this.pad.fTopMargin))*h);
         } else {
            rect.szx = Math.round(0.5*w);
            rect.szy = Math.round(0.5*h);
         }

         rect.width = 2*rect.szx;
         rect.height = 2*rect.szy;
         rect.x = Math.round(w/2 - rect.szx);
         rect.y = Math.round(h/2 - rect.szy);
         rect.hint_delta_x = rect.szx;
         rect.hint_delta_y = rect.szy;
         rect.transform = `translate(${rect.x},${rect.y})`;

         return rect;
      }

      /** @summary return RPad object */
      getRootPad(is_root6) {
         return (is_root6 === undefined) || is_root6 ? this.pad : null;
      }

      /** @summary Cleanup primitives from pad - selector lets define which painters to remove */
      cleanPrimitives(selector) {
         if (!selector || (typeof selector !== 'function')) return;

         for (let k = this.painters.length-1; k >= 0; --k)
            if (selector(this.painters[k])) {
               this.painters[k].cleanup();
               this.painters.splice(k, 1);
            }
      }

     /** @summary returns custom palette associated with pad or top canvas
       * @private */
      getCustomPalette() {
         if (this.custom_palette)
            return this.custom_palette;

         let cp = this.getCanvPainter();
         return cp ? cp.custom_palette : null;
      }

      /** @summary Returns number of painters
        * @private */
      getNumPainters() { return this.painters.length; }

      /** @summary Call function for each painter in pad
        * @param {function} userfunc - function to call
        * @param {string} kind - "all" for all objects (default), "pads" only pads and subpads, "objects" only for object in current pad
        * @private */
      forEachPainterInPad(userfunc, kind) {
         if (!kind) kind = "all";
         if (kind != "objects") userfunc(this);
         for (let k = 0; k < this.painters.length; ++k) {
            let sub = this.painters[k];
            if (typeof sub.forEachPainterInPad === 'function') {
               if (kind != "objects") sub.forEachPainterInPad(userfunc, kind);
            } else if (kind != "pads") {
               userfunc(sub);
            }
         }
      }

      /** @summary register for pad events receiver
        * @desc in pad painter, while pad may be drawn without canvas */
      registerForPadEvents(receiver) {
         this.pad_events_receiver = receiver;
      }

      /** @summary Generate pad events, normally handled by GED
       * @desc in pad painter, while pad may be drawn without canvas
        * @private */
      producePadEvent(_what, _padpainter, _painter, _position, _place) {

         if ((_what == "select") && (typeof this.selectActivePad == 'function'))
            this.selectActivePad(_padpainter, _painter, _position);

         if (this.pad_events_receiver)
            this.pad_events_receiver({ what: _what, padpainter:  _padpainter, painter: _painter, position: _position, place: _place });
      }

      /** @summary method redirect call to pad events receiver */
      selectObjectPainter(_painter, pos, _place) {
         let istoppad = (this.iscan || !this.has_canvas),
             canp = istoppad ? this : this.getCanvPainter();

         if (_painter === undefined) _painter = this;

         if (pos && !istoppad)
            pos = jsrp.getAbsPosInCanvas(this.svg_this_pad(), pos);

         jsrp.selectActivePad({ pp: this, active: true });

         if (canp) canp.producePadEvent("select", this, _painter, pos, _place);
      }

      /** @summary Draw pad active border
       * @private */
      drawActiveBorder(svg_rect, is_active) {
         if (is_active !== undefined) {
            if (this.is_active_pad === is_active) return;
            this.is_active_pad = is_active;
         }

         if (this.is_active_pad === undefined) return;

         if (!svg_rect)
            svg_rect = this.iscan ? this.getCanvSvg().select(".canvas_fillrect") :
                                    this.svg_this_pad().select(".root_pad_border");

         let lineatt = this.is_active_pad ? new JSROOT.TAttLineHandler({ style: 1, width: 1, color: "red" }) : this.lineatt;

         if (!lineatt) lineatt = new JSROOT.TAttLineHandler({ color: "none" });

         svg_rect.call(lineatt.func);
      }

      /** @summary Create SVG element for canvas */
      createCanvasSvg(check_resize, new_size) {

         let factor = null, svg = null, lmt = 5, rect = null, btns, frect;

         if (check_resize > 0) {

            if (this._fixed_size) return (check_resize > 1); // flag used to force re-drawing of all subpads

            svg = this.getCanvSvg();

            if (svg.empty()) return false;

            factor = svg.property('height_factor');

            rect = this.testMainResize(check_resize, null, factor);

            if (!rect.changed) return false;

            if (!JSROOT.batch_mode)
               btns = this.getLayerSvg("btns_layer", this.this_pad_name);

            frect = svg.select(".canvas_fillrect");

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

            if (JSROOT.batch_mode) {
               svg.attr("xmlns", "http://www.w3.org/2000/svg");
               svg.attr("xmlns:xlink", "http://www.w3.org/1999/xlink");
            } else {
               svg.append("svg:title").text("ROOT canvas");
            }

            if (!JSROOT.batch_mode || (this.pad.fFillStyle > 0))
               frect = svg.append("svg:path").attr("class","canvas_fillrect");

            if (!JSROOT.batch_mode)
               frect.style("pointer-events", "visibleFill")
                    .on("dblclick", evnt => this.enlargePad(evnt))
                    .on("click", () => this.selectObjectPainter())
                    .on("mouseenter", () => this.showObjectStatus())
                    .on("contextmenu", JSROOT.settings.ContextMenu ? evnt => this.padContextMenu(evnt) : null);

            svg.append("svg:g").attr("class","primitives_layer");
            svg.append("svg:g").attr("class","info_layer");
            if (!JSROOT.batch_mode)
               btns = svg.append("svg:g")
                         .attr("class","btns_layer")
                         .property('leftside', JSROOT.settings.ToolBarSide == 'left')
                         .property('vertical', JSROOT.settings.ToolBarVert);

            factor = 0.66;
            if (this.pad && this.pad.fCw && this.pad.fCh && (this.pad.fCw > 0)) {
               factor = this.pad.fCh / this.pad.fCw;
               if ((factor < 0.1) || (factor > 10)) factor = 0.66;
            }

            if (this._fixed_size) {
               render_to.style("overflow","auto");
               rect = { width: this.pad.fCw, height: this.pad.fCh };
               if (!rect.width || !rect.height)
                  rect = jsrp.getElementRect(render_to);
            } else {
               rect = this.testMainResize(2, new_size, factor);
            }
         }

         this.createAttFill({ attr: this.pad });

         if ((rect.width <= lmt) || (rect.height <= lmt)) {
            svg.style("display", "none");
            console.warn(`Hide canvas while geometry too small w=${rect.width} h=${rect.height}`);
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

         svg.attr("viewBox", `0 0 ${rect.width} ${rect.height}`)
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

         if (frect) {
            frect.attr("d", `M0,0H${rect.width}V${rect.height}H0Z`)
                 .call(this.fillatt.func);
            this.drawActiveBorder(frect);
         }

         this._fast_drawing = JSROOT.settings.SmallPad && ((rect.width < JSROOT.settings.SmallPad.width) || (rect.height < JSROOT.settings.SmallPad.height));

         if (this.alignButtons && btns)
            this.alignButtons(btns, rect.width, rect.height);

         return true;
      }

      /** @summary Enlarge pad draw element when possible */
      enlargePad(evnt) {

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

      /** @summary Create main SVG element for pad
        * @returns true when pad is displayed and all its items should be redrawn */
      createPadSvg(only_resize) {

         if (!this.has_canvas) {
            this.createCanvasSvg(only_resize ? 2 : 0);
            return true;
         }

         let svg_can = this.getCanvSvg(),
             width = svg_can.property("draw_width"),
             height = svg_can.property("draw_height"),
             pad_enlarged = svg_can.property("pad_enlarged"),
             pad_visible = !this.pad_draw_disabled && (!pad_enlarged || (pad_enlarged === this.pad)),
             w = Math.round(this.pad.fAbsWNDC * width),
             h = Math.round(this.pad.fAbsHNDC * height),
             x = Math.round(this.pad.fAbsXlowNDC * width),
             y = Math.round(height * (1 - this.pad.fAbsYlowNDC)) - h,
             svg_pad = null, svg_rect = null, btns = null;

         if (pad_enlarged === this.pad) { w = width; h = height; x = y = 0; }

         if (only_resize) {
            svg_pad = this.svg_this_pad();
            svg_rect = svg_pad.select(".root_pad_border");
            if (!JSROOT.batch_mode)
               btns = this.getLayerSvg("btns_layer", this.this_pad_name);
         } else {
            svg_pad = svg_can.select(".primitives_layer")
                .append("svg:svg") // here was g before, svg used to blend all drawin outside
                .classed("__root_pad_" + this.this_pad_name, true)
                .attr("pad", this.this_pad_name) // set extra attribute  to mark pad name
                .property('pad_painter', this); // this is custom property

            if (!JSROOT.batch_mode)
               svg_pad.append("svg:title").text("subpad " + this.this_pad_name);

            // need to check attributes directly while attributes objects will be created later
            if (!JSROOT.batch_mode || (this.pad.fFillStyle > 0) || ((this.pad.fLineStyle > 0) && (this.pad.fLineColor > 0)))
               svg_rect = svg_pad.append("svg:path").attr("class", "root_pad_border");

            if (!JSROOT.batch_mode)
               svg_rect.style("pointer-events", "visibleFill") // get events also for not visible rect
                       .on("dblclick", evnt => this.enlargePad(evnt))
                       .on("click", () => this.selectObjectPainter())
                       .on("mouseenter", () => this.showObjectStatus())
                       .on("contextmenu", JSROOT.settings.ContextMenu ? evnt => this.padContextMenu(evnt) : null);

            svg_pad.append("svg:g").attr("class","primitives_layer");
            if (!JSROOT.batch_mode)
               btns = svg_pad.append("svg:g")
                             .attr("class","btns_layer")
                             .property('leftside', JSROOT.settings.ToolBarSide != 'left')
                             .property('vertical', JSROOT.settings.ToolBarVert);
         }

         this.createAttFill({ attr: this.pad });
         this.createAttLine({ attr: this.pad, color0: this.pad.fBorderMode == 0 ? 'none' : '' });

         svg_pad.style("display", pad_visible ? null : "none")
                .attr("viewBox", `0 0 ${w} ${h}`) // due to svg
                .attr("preserveAspectRatio", "none")   // due to svg, we do not preserve relative ratio
                .attr("x", x)        // due to svg
                .attr("y", y)        // due to svg
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

         if (svg_rect) {
            svg_rect.attr("d", `M0,0H${w}V${h}H0Z`)
                    .call(this.fillatt.func)
                    .call(this.lineatt.func);

            this.drawActiveBorder(svg_rect);
         }

         this._fast_drawing = JSROOT.settings.SmallPad && ((w < JSROOT.settings.SmallPad.width) || (h < JSROOT.settings.SmallPad.height));

         // special case of 3D canvas overlay
         if (svg_pad.property('can3d') === JSROOT.constants.Embed3D.Overlay)
             this.selectDom().select(".draw3d_" + this.this_pad_name)
                 .style('display', pad_visible ? '' : 'none');

         if (this.alignButtons && btns)
            this.alignButtons(btns, w, h);

         return pad_visible;
      }

      /** @summary Disable pad drawing
        * @desc Complete SVG element will be hidden */
      disablePadDrawing() {
         if (!this.pad_draw_disabled && this.has_canvas && !this.iscan) {
            this.pad_draw_disabled = true;
            this.createPadSvg(true);
         }
      }

      /** @summary Check if it is special object, which should be handled separately
        * @desc It can be TStyle or list of colors or palette object
        * @returns {boolean} tru if any */
      checkSpecial(obj) {

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

      /** @summary Check if special objects appears in primitives
        * @desc it could be list of colors or palette */
      checkSpecialsInPrimitives(can) {
         let lst = can ? can.fPrimitives : null;
         if (!lst) return;
         for (let i = 0; i < lst.arr.length; ++i) {
            if (this.checkSpecial(lst.arr[i])) {
               lst.arr.splice(i,1);
               lst.opt.splice(i,1);
               i--;
            }
         }
      }

      /** @summary try to find object by name in list of pad primitives
        * @desc used to find title drawing
        * @private */
      findInPrimitives(objname, objtype) {
         let arr = this.pad && this.pad.fPrimitives ? this.pad.fPrimitives.arr : null;

         return arr ? arr.find(obj => (obj.fName == objname) && (objtype ? (obj.typename == objtype) : true)) : null;
      }

      /** @summary Try to find painter for specified object
        * @desc can be used to find painter for some special objects, registered as
        * histogram functions
        * @param {object} selobj - object to which painter should be search, set null to ignore parameter
        * @param {string} [selname] - object name, set to null to ignore
        * @param {string} [seltype] - object type, set to null to ignore
        * @returns {object} - painter for specified object (if any)
        * @private */
      findPainterFor(selobj, selname, seltype) {
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

      /** @summary Return true if any objects beside sub-pads exists in the pad */
      hasObjectsToDraw() {
         let arr = this.pad && this.pad.fPrimitives ? this.pad.fPrimitives.arr : null;
         return arr && arr.find(obj => obj._typename != "TPad") ? true : false;
      }

      /** @summary sync drawing/redrawing/resize of the pad
        * @param {string} kind - kind of draw operation, if true - always queued
        * @returns {Promise} when pad is ready for draw operation or false if operation already queued
        * @private */
      syncDraw(kind) {
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

      /** @summary indicates if painter performing objects draw
        * @private */
      doingDraw() {
         return this._doing_draw !== undefined;
      }

      /** @summary confirms that drawing is completed, may trigger next drawing immediately
        * @private */
      confirmDraw() {
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
        * @returns {Promise} when drawing completed
        * @private */
      drawPrimitives(indx) {

         if (indx === undefined) {
            if (this.iscan)
               this._start_tm = new Date().getTime();

            // set number of primitves
            this._num_primitives = this.pad && this.pad.fPrimitives ? this.pad.fPrimitives.arr.length : 0;

            // sync to prevent immediate pad redraw during normal drawing sequence
            return this.syncDraw(true).then(() => this.drawPrimitives(0));
         }

         if (indx >= this._num_primitives) {
            if (this._start_tm) {
               let spenttm = new Date().getTime() - this._start_tm;
               if (spenttm > 1000) console.log(`Canvas ${this.pad ? this.pad.fName : "---"} drawing took ${(spenttm*1e-3).toFixed(2)}s`);
               delete this._start_tm;
            }

            this.confirmDraw();
            return Promise.resolve();
         }

         // use of Promise should avoid large call-stack depth when many primitives are drawn
         return JSROOT.draw(this.getDom(), this.pad.fPrimitives.arr[indx], this.pad.fPrimitives.opt[indx]).then(op => {
            if (op && (typeof op == 'object'))
               op._primitive = true; // mark painter as belonging to primitives

            return this.drawPrimitives(indx+1);
         });
      }

      /** @summary Divide pad on subpads
        * @returns {Promise} when finished
        * @private */
      divide(nx, ny) {
         if (!ny) {
            let ndiv = nx;
            if (ndiv < 2) return Promise.resolve(this);
            nx = ny = Math.round(Math.sqrt(ndiv));
            if (nx*ny < ndiv) nx += 1;
         }

         if (nx*ny < 2) return Promise.resolve(this);

         let xmargin = 0.01, ymargin = 0.01,
             dy = 1/ny, dx = 1/nx, n = 0, subpads = [];
         for (let iy = 0; iy < ny; iy++) {
            let y2 = 1 - iy*dy - ymargin,
                y1 = y2 - dy + 2*ymargin;
            if (y1 < 0) y1 = 0;
            if (y1 > y2) continue;
            for (let ix = 0; ix < nx; ix++) {
               let x1 = ix*dx + xmargin,
                   x2 = x1 +dx -2*xmargin;
               if (x1 > x2) continue;
               n++;
               let pad = JSROOT.create("TPad");
               pad.fName = pad.fTitle = this.pad.fName + "_" + n;
               pad.fNumber = n;
               if (!this.iscan) {
                  pad.fAbsWNDC = (x2-x1) * this.pad.fAbsWNDC;
                  pad.fAbsHNDC = (y2-y1) * this.pad.fAbsHNDC;
                  pad.fAbsXlowNDC = this.pad.fAbsXlowNDC + x1 * this.pad.fAbsWNDC;
                  pad.fAbsYlowNDC = this.pad.fAbsYlowNDC + y1 * this.pad.fAbsWNDC;
               } else {
                  pad.fAbsWNDC = x2 - x1;
                  pad.fAbsHNDC = y2 - y1;
                  pad.fAbsXlowNDC = x1;
                  pad.fAbsYlowNDC = y1;
               }

               subpads.push(pad);
            }
         }

         const drawNext = () => {
            if (subpads.length == 0)
               return Promise.resolve(this);
            return JSROOT.draw(this.getDom(), subpads.shift()).then(drawNext);
         };

         return drawNext();
      }

      /** @summary Return sub-pads painter, only direct childs are checked
        * @private */
      getSubPadPainter(n) {
         for (let k = 0; k < this.painters.length; ++k) {
            let sub = this.painters[k];
            if (sub.pad && (typeof sub.forEachPainterInPad === 'function') && (sub.pad.fNumber === n)) return sub;
         }
         return null;
      }


      /** @summary Process tooltip event in the pad
        * @private */
      processPadTooltipEvent(pnt) {
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
      fillContextMenu(menu) {

         if (this.pad)
            menu.add("header: " + this.pad._typename + "::" + this.pad.fName);
         else
            menu.add("header: Canvas");

         menu.addchk(this.isTooltipAllowed(), "Show tooltips", () => this.setTooltipAllowed("toggle"));

         if (!this._websocket) {

            function SetPadField(arg) {
               this.pad[arg.substr(1)] = parseInt(arg[0]);
               this.interactiveRedraw("pad", arg.substr(1));
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

            menu.addAttributesMenu(this);
         }

         menu.add("separator");

         if (typeof this.hasMenuBar == 'function' && typeof this.actiavteMenuBar == 'function')
            menu.addchk(this.hasMenuBar(), "Menu bar", flag => this.actiavteMenuBar(flag));

         if (typeof this.hasEventStatus == 'function' && typeof this.activateStatusBar == 'function')
            menu.addchk(this.hasEventStatus(), "Event status", () => this.activateStatusBar('toggle'));

         if (this.enlargeMain() || (this.has_canvas && this.hasObjectsToDraw()))
            menu.addchk((this.enlargeMain('state')=='on'), "Enlarge " + (this.iscan ? "canvas" : "pad"), () => this.enlargePad());

         let fname = this.this_pad_name;
         if (fname.length===0) fname = this.iscan ? "canvas" : "pad";

         menu.add("Save as "+ fname+".png", fname+".png", () => this.saveAs("png", false));
         menu.add("Save as "+ fname+".svg", fname+".svg", () => this.saveAs("svg", false));

         return true;
      }

      /** @summary Show pad context menu
        * @private */
      padContextMenu(evnt) {

         if (evnt.stopPropagation) { // this is normal event processing and not emulated jsroot event

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
            return this.fillObjectExecMenu(menu, "");
         }).then(menu => menu.show());
      }

      /** @summary Redraw pad means redraw ourself
        * @returns {Promise} when redrawing ready */
      redrawPad(reason) {

         let sync_promise = this.syncDraw(reason);
         if (sync_promise === false) {
            console.log('Prevent redrawing', this.pad.fName);
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
            this.confirmDraw();
            if (jsrp.getActivePad() === this) {
               let canp = this.getCanvPainter();
               if (canp) canp.producePadEvent("padredraw", this);
            }
            return true;
         });
      }

      /** @summary redraw pad */
      redraw(reason) {
         // intentially do not return Promise to let re-draw sub-pads in parallel
         this.redrawPad(reason);
      }

      /** @summary Checks if pad should be redrawn by resize
        * @private */
      needRedrawByResize() {
         let elem = this.svg_this_pad();
         if (!elem.empty() && elem.property('can3d') === JSROOT.constants.Embed3D.Overlay) return true;

         return this.painters.findIndex(objp => {
            if (typeof objp.needRedrawByResize === 'function')
               return objp.needRedrawByResize();
         }) >= 0;
      }

      /** @summary Check resize of canvas
        * @returns {Promise} with result */
      checkCanvasResize(size, force) {

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

      /** @summary Update TPad object */
      updateObject(obj) {
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
         this.pad.fBottomMargin = obj.fBottomMargin;
         this.pad.fTopMargin    = obj.fTopMargin;

         this.pad.fFillColor = obj.fFillColor;
         this.pad.fFillStyle = obj.fFillStyle;
         this.pad.fLineColor = obj.fLineColor;
         this.pad.fLineStyle = obj.fLineStyle;
         this.pad.fLineWidth = obj.fLineWidth;

         this.pad.fPhi = obj.fPhi;
         this.pad.fTheta = obj.fTheta;

         if (this.iscan) this.checkSpecialsInPrimitives(obj);

         let fp = this.getFramePainter();
         if (fp) fp.updateAttributes(!fp.modified_NDC);

         if (!obj.fPrimitives) return false;

         let isany = false, p = 0;
         for (let n = 0; n < obj.fPrimitives.arr.length; ++n) {
            while (p < this.painters.length) {
               let pp = this.painters[p++];
               if (!pp._primitive) continue;
               if (pp.updateObject(obj.fPrimitives.arr[n])) isany = true;
               break;
            }
         }

         return isany;
      }

      /** @summary Add object painter to list of primitives
        * @private */
      addObjectPainter(objpainter, lst, indx) {
         if (objpainter && lst && lst[indx] && (objpainter.snapid === undefined)) {
            // keep snap id in painter, will be used for the
            let pi = this.painters.indexOf(objpainter);
            if (pi < 0) this.painters.push(objpainter);

            if (typeof objpainter.setSnapId == 'function')
               objpainter.setSnapId(lst[indx].fObjectID);
            else
               objpainter.snapid = lst[indx].fObjectID;

            if (objpainter.$primary && (pi > 0) && this.painters[pi-1].$secondary) {
               this.painters[pi-1].snapid = objpainter.snapid + "#hist";
               console.log('ASSIGN SECONDARY HIST ID', this.painters[pi-1].snapid);
            }
         }
      }

      /** @summary Function called when drawing next snapshot from the list
        * @returns {Promise} for drawing of the snap
        * @private */
      drawNextSnap(lst, indx) {

         if (indx === undefined) {
            indx = -1;
            this._snaps_map = {}; // to control how much snaps are drawn
            this._num_primitives = lst ? lst.length : 0;
         }

         ++indx; // change to the next snap

         if (!lst || (indx >= lst.length)) {
            delete this._snaps_map;
            return Promise.resolve(this);
         }

         let snap = lst[indx],
             snapid = snap.fObjectID,
             cnt = this._snaps_map[snapid],
             objpainter = null;

         if (cnt) cnt++; else cnt = 1;
         this._snaps_map[snapid] = cnt; // check how many objects with same snapid drawn, use them again

         // first appropriate painter for the object
         // if same object drawn twice, two painters will exists
         for (let k = 0;  k < this.painters.length; ++k) {
            if (this.painters[k].snapid === snapid)
               if (--cnt === 0) { objpainter = this.painters[k]; break; }
         }

         if (objpainter) {

            if (snap.fKind === webSnapIds.kSubPad) // subpad
               return objpainter.redrawPadSnap(snap).then(() => this.drawNextSnap(lst, indx));

            let promise;

            if (snap.fKind === webSnapIds.kObject) { // object itself
               if (objpainter.updateObject(snap.fSnapshot, snap.fOption))
                  promise = objpainter.redraw();
            } else if (snap.fKind === webSnapIds.kSVG) { // update SVG
               if (objpainter.updateObject(snap.fSnapshot))
                  promise = objpainter.redraw();
            }

            if (!jsrp.isPromise(promise)) promise = Promise.resolve(true);

            return promise.then(() => this.drawNextSnap(lst, indx)); // call next
         }

         // gStyle object
         if (snap.fKind === webSnapIds.kStyle) {
            JSROOT.extend(JSROOT.gStyle, snap.fSnapshot);
            return this.drawNextSnap(lst, indx); // call next
         }

         // list of colors
         if (snap.fKind === webSnapIds.kColors) {

            let ListOfColors = [], arr = snap.fSnapshot.fOper.split(";");
            for (let n = 0; n < arr.length; ++n) {
               let name = arr[n], p = name.indexOf(":");
               if (p > 0) {
                  ListOfColors[parseInt(name.substr(0,p))] = d3.color("rgb(" + name.substr(p+1) + ")").formatHex();
               } else {
                  p = name.indexOf("=");
                  ListOfColors[parseInt(name.substr(0,p))] = d3.color("rgba(" + name.substr(p+1) + ")").formatHex();
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
               for (let n = 0; n < snap.fSnapshot.fBuf.length; ++n)
                  palette[n] = ListOfColors[Math.round(snap.fSnapshot.fBuf[n])];

               this.custom_palette = new JSROOT.ColorPalette(palette);
            }

            return this.drawNextSnap(lst, indx); // call next
         }

         if (snap.fKind === webSnapIds.kSubPad) { // subpad

            let subpad = snap.fSnapshot;

            subpad.fPrimitives = null; // clear primitives, they just because of I/O

            let padpainter = new TPadPainter(this.getDom(), subpad, false);
            padpainter.decodeOptions(snap.fOption);
            padpainter.addToPadPrimitives(this.this_pad_name);
            padpainter.snapid = snap.fObjectID;

            padpainter.createPadSvg();

            if (padpainter.matchObjectType("TPad") && (snap.fPrimitives.length > 0))
               padpainter.addPadButtons(true);

            // we select current pad, where all drawing is performed
            let prev_name = padpainter.selectCurrentPad(padpainter.this_pad_name);
            return padpainter.drawNextSnap(snap.fPrimitives).then(() => {
               padpainter.selectCurrentPad(prev_name);
               return this.drawNextSnap(lst, indx); // call next
            });
         }

         // here the case of normal drawing, will be handled in promise
         if ((snap.fKind === webSnapIds.kObject) || (snap.fKind === webSnapIds.kSVG))
            return JSROOT.draw(this.getDom(), snap.fSnapshot, snap.fOption).then(objpainter => {
               this.addObjectPainter(objpainter, lst, indx);
               return this.drawNextSnap(lst, indx);
            });

         return this.drawNextSnap(lst, indx);
      }

      /** @summary Return painter with specified id
        * @private */
      findSnap(snapid) {

         if (this.snapid === snapid) return this;

         if (!this.painters) return null;

         for (let k = 0; k < this.painters.length; ++k) {
            let sub = this.painters[k];

            if (typeof sub.findSnap === 'function')
               sub = sub.findSnap(snapid);
            else if (sub.snapid !== snapid)
               sub = null;

            if (sub) return sub;
         }

         return null;
      }

      /** @summary Redraw pad snap
        * @desc Online version of drawing pad primitives
        * for the canvas snapshot contains list of objects
        * as first entry, graphical properties of canvas itself is provided
        * in ROOT6 it also includes primitives, but we ignore them
        * @returns {Promise} with pad painter when drawing completed
        * @private */
      redrawPadSnap(snap) {
         if (!snap || !snap.fPrimitives)
            return Promise.resolve(this);

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

            let layout_promise = Promise.resolve(true),
                mainid = this.selectDom().attr("id");

            if (!this.batch_mode && !this.use_openui && !this.brlayout && mainid && (typeof mainid == "string"))
               layout_promise = JSROOT.require('hierarchy').then(() => {
                  this.brlayout = new JSROOT.BrowserLayout(mainid, null, this);
                  this.brlayout.create(mainid, true);
                  // this.brlayout.toggleBrowserKind("float");
                  this.setDom(this.brlayout.drawing_divid()); // need to create canvas
                  jsrp.registerForResize(this.brlayout);
              });

            return layout_promise.then(() => {

               this.createCanvasSvg(0);

               if (!this.batch_mode)
                  this.addPadButtons(true);

               if (typeof snap.fHighlightConnect !== 'undefined')
                  this._highlight_connect = snap.fHighlightConnect;

               if ((typeof snap.fScripts == "string") && snap.fScripts) {
                  let arg = "";

                  if (snap.fScripts.indexOf("load:") == 0)
                     arg = snap.fScripts;
                  else if (snap.fScripts.indexOf("assert:") == 0)
                     arg = snap.fScripts.substr(7);

                  if (arg)
                     return JSROOT.require(arg).then(() => this.drawNextSnap(snap.fPrimitives));

                  console.log('Calling eval ' + snap.fScripts.length);
                  eval(snap.fScripts);
                  console.log('Calling eval done');
               }

               return this.drawNextSnap(snap.fPrimitives);
            });
         }

         this.updateObject(first); // update only object attributes

         // apply all changes in the object (pad or canvas)
         if (this.iscan) {
            this.createCanvasSvg(2);
         } else {
            this.createPadSvg(true);
         }

         const MatchPrimitive = (painters, primitives, class_name, obj_name) => {
            let painter = painters.find(p => {
               if (p.snapid === undefined) return;
               if (!p.matchObjectType(class_name)) return;
               if (obj_name && (!p.getObject() || (p.getObject().fName !== obj_name))) return;
               return true;
            });
            if (!painter) return;
            let primitive = primitives.find(pr => {
               if ((pr.fKind !== 1) || !pr.fSnapshot || (pr.fSnapshot._typename !== class_name)) return;
               if (obj_name && (pr.fSnapshot.fName !== obj_name)) return;
               return true;
            });
            if (!primitive) return;

            // force painter to use new object id
            if (painter.snapid !== primitive.fObjectID)
               painter.snapid = primitive.fObjectID;
         };

         // check if frame or title was recreated, we could reassign handlers for them directly
         // while this is temporary objects, which can be recreated very often, try to catch such situation ourselfs
         MatchPrimitive(this.painters, snap.fPrimitives, "TFrame");
         MatchPrimitive(this.painters, snap.fPrimitives, "TPaveText", "title");

         let isanyfound = false, isanyremove = false;

         // find and remove painters which no longer exists in the list
         for (let k = 0; k < this.painters.length; ++k) {
            let sub = this.painters[k];
            if ((sub.snapid===undefined) || sub.$secondary) continue; // look only for painters with snapid

            for (let i = 0; i < snap.fPrimitives.length; ++i)
               if (snap.fPrimitives[i].fObjectID === sub.snapid) { sub = null; isanyfound = true; break; }

            if (sub) {
               // console.log(`Remove painter ${k} from ${this.painters.length} class ${sub.getClassName()}`);
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
            // TODO: maybe just remove frame painter?
            let fp = this.getFramePainter();
            this.painters.forEach(objp => {
               if (fp !== objp) objp.cleanup();
            });
            delete this.main_painter_ref;
            this.painters = [];
            if (fp) {
               this.painters.push(fp);
               fp.cleanFrameDrawings();
               fp.redraw();
            }
            if (this.removePadButtons) this.removePadButtons();
            this.addPadButtons(true);
         }

         let prev_name = this.selectCurrentPad(this.this_pad_name);

         return this.drawNextSnap(snap.fPrimitives).then(() => {
            // redraw secondaries like stat box
            let promises = [];
            this.painters.forEach(sub => {
               if ((sub.snapid===undefined) || sub.$secondary) {
                  let res = sub.redraw();
                  if (jsrp.isPromise(res)) promises.push(res);
               }
            });
            return Promise.all(promises);
         }).then(() => {
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
      createImage(format) {
         // use https://github.com/MrRio/jsPDF in the future here
         if (format == "pdf")
            return Promise.resolve(btoa("dummy PDF file"));

         if ((format == "png") || (format == "jpeg") || (format == "svg"))
            return this.produceImage(true, format).then(res => {
               if (!res || (format=="svg")) return res;
               let separ = res.indexOf("base64,");
               return (separ>0) ? res.substr(separ+7) : "";
            });

         return Promise.resolve("");
      }

      /** @summary Collects pad information for TWebCanvas
        * @desc need to update different states
        * @private */
      getWebPadOptions(arg) {
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

            if (this.iscan) elem.bits = this.getStatusBits();

            if (this.getPadRanges(elem))
               arg.push(elem);
            else
               console.log('fail to get ranges for pad ' +  this.pad.fName);
         }

         this.painters.forEach(sub => {
            if (typeof sub.getWebPadOptions == "function") {
               if (scan_subpads) sub.getWebPadOptions(arg);
            } else if (sub.snapid) {
               let opt = { _typename: "TWebObjectOptions", snapid: sub.snapid.toString(), opt: sub.getDrawOpt(), fcust: "", fopt: [] };
               if (typeof sub.fillWebObjectOptions == "function")
                  opt = sub.fillWebObjectOptions(opt);
               elem.primitives.push(opt);
            }
         });

         if (is_top) return JSROOT.toJSON(arg);
      }

      /** @summary returns actual ranges in the pad, which can be applied to the server
        * @private */
      getPadRanges(r) {

         if (!r) return false;

         let main = this.getFramePainter(),
             p = this.svg_this_pad();

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
         const same = x => x,
               direct_funcs = [same, Math.log10, x => Math.log10(x)/Math.log10(2)],
               revert_funcs = [same, x => Math.pow(10, x), x => Math.pow(2, x)],
               match = (v1, v0, range) => (Math.abs(v0-v1) < Math.abs(range)*1e-10) ? v0 : v1;

         let func = direct_funcs[main.logx],
             func2 = revert_funcs[main.logx],
             k = (func(main.scale_xmax) - func(main.scale_xmin))/p.property("draw_width"),
             x1 = func(main.scale_xmin) - k*p.property("draw_x"),
             x2 = x1 + k*p.property("draw_width");

         r.ux1 = match(func2(x1), r.ux1, r.px2-r.px1);
         r.ux2 = match(func2(x2), r.ux2, r.px2-r.px1);

         func = direct_funcs[main.logy];
         func2 = revert_funcs[main.logy];

         k = (func(main.scale_ymax) - func(main.scale_ymin))/p.property("draw_height");
         let y2 = func(main.scale_ymax) + k*p.property("draw_y"),
             y1 = y2 - k*p.property("draw_height");

         r.uy1 = match(func2(y1), r.uy1, r.py2-r.py1);
         r.uy2 = match(func2(y2), r.uy2, r.py2-r.py1);

         return true;
      }

      /** @summary Show context menu for specified item
        * @private */
      itemContextMenu(name) {
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
                selp = this.getFramePainter();
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
      saveAs(kind, full_canvas, filename) {
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
      produceImage(full_canvas, file_format) {

         let use_frame = (full_canvas === "frame"),
             elem = use_frame ? this.getFrameSvg() : (full_canvas ? this.getCanvSvg() : this.svg_this_pad());

         if (elem.empty()) return Promise.resolve("");

         let painter = (full_canvas && !use_frame) ? this.getCanvPainter() : this,
             items = [], // keep list of replaced elements, which should be moved back at the end
             active_pp = null;

         painter.forEachPainterInPad(pp => {

             if (pp.is_active_pad && !active_pp) {
                active_pp = pp;
                active_pp.drawActiveBorder(null, false);
             }

            if (use_frame) return; // do not make transformations for the frame

            let item = { prnt: pp.svg_this_pad() };
            items.push(item);

            // remove buttons from each subpad
            let btns = pp.getLayerSvg("btns_layer", pp.this_pad_name);
            item.btns_node = btns.node();
            if (item.btns_node) {
               item.btns_prnt = item.btns_node.parentNode;
               item.btns_next = item.btns_node.nextSibling;
               btns.remove();
            }

            let main = pp.getFramePainter();
            if (!main || (typeof main.render3D !== 'function') || (typeof main.access3dKind !== 'function')) return;

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

         let reEncode = data => {
            data = encodeURIComponent(data);
            data = data.replace(/%([0-9A-F]{2})/g, (match, p1) => {
              let c = String.fromCharCode('0x'+p1);
              return c === '%' ? '%25' : c;
            });
            return decodeURIComponent(data);
         };

         let reconstruct = () => {
            // reactivate border
            if (active_pp)
               active_pp.drawActiveBorder(null, true);

            for (let k = 0; k < items.length; ++k) {
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
         };

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
      clickPadButton(funcname, evnt) {

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

               if (this.painters && (this.painters.length > 0)) {
                  menu.add("separator");
                  let shown = [];
                  this.painters.forEach((pp,indx) => {
                     let obj = pp ? pp.getObject() : null;
                     if (!obj || (shown.indexOf(obj) >= 0)) return;
                     if (pp.$secondary) return;
                     let name = ('_typename' in obj) ? (obj._typename + "::") : "";
                     if ('fName' in obj) name += obj.fName;
                     if (!name.length) name = "item" + indx;
                     menu.add(name, indx, this.itemContextMenu);
                  });
               }

               menu.show();
            });

            return;
         }

         // click automatically goes to all sub-pads
         // if any painter indicates that processing completed, it returns true
         let done = false;

         this.painters.forEach(pp => {
            if (typeof pp.clickPadButton == 'function')
               pp.clickPadButton(funcname);

            if (!done && (typeof pp.clickButton == 'function'))
               done = pp.clickButton(funcname);
         });
      }

      /** @summary Add button to the pad
        * @private */
      addPadButton(_btn, _tooltip, _funcname, _keyname) {
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

      /** @summary Show pad buttons
        * @private */
      showPadButtons() {
         if (!this._buttons) return;

         JSROOT.require(['interactive']).then(inter => {
            inter.PadButtonsHandler.assign(this);
            this.showPadButtons();
         });
      }

      /** @summary Add buttons for pad or canvas
        * @private */
      addPadButtons(is_online) {

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

      /** @summary Decode pad draw options
        * @private */
      decodeOptions(opt) {
         let pad = this.getObject();
         if (!pad) return;

         let d = new JSROOT.DrawOptions(opt);

         if (d.check('WEBSOCKET') && this.openWebsocket) this.openWebsocket();
         if (!this.options) this.options = {};

         JSROOT.extend(this.options, { GlobalColors: true, LocalColors: false, CreatePalette: 0, IgnorePalette: false, RotateFrame: false, FixFrame: false });

         if (d.check('NOCOLORS') || d.check('NOCOL')) this.options.GlobalColors = this.options.LocalColors = false;
         if (d.check('LCOLORS') || d.check('LCOL')) { this.options.GlobalColors = false; this.options.LocalColors = true; }
         if (d.check('NOPALETTE') || d.check('NOPAL')) this.options.IgnorePalette = true;
         if (d.check('ROTATE')) this.options.RotateFrame = true;
         if (d.check('FIXFRAME')) this.options.FixFrame = true;

         if (d.check("CP",true)) this.options.CreatePalette = d.partAsInt(0,0);

         if (d.check("NOZOOMX")) this.options.NoZoomX = true;
         if (d.check("NOZOOMY")) this.options.NoZoomY = true;

         if (d.check("NOMARGINS")) pad.fLeftMargin = pad.fRightMargin = pad.fBottomMargin = pad.fTopMargin = 0;
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

         this.storeDrawOpt(opt);
      }

      /** @summary draw TPad object */
      static draw(dom, pad, opt) {
         let painter = new TPadPainter(dom, pad, false);
         painter.decodeOptions(opt);

         if (painter.getCanvSvg().empty()) {
            // one can draw pad without canvas
            painter.has_canvas = false;
            painter.this_pad_name = "";
            painter.setTopPainter();
         } else {
            // pad painter will be registered in the canvas painters list
            painter.addToPadPrimitives(painter.pad_name);
         }

         painter.createPadSvg();

         if (painter.matchObjectType("TPad") && (!painter.has_canvas || painter.hasObjectsToDraw()))
            painter.addPadButtons();

         // we select current pad, where all drawing is performed
         let prev_name = painter.has_canvas ? painter.selectCurrentPad(painter.this_pad_name) : undefined;

         // set active pad
         jsrp.selectActivePad({ pp: painter, active: true });

         // flag used to prevent immediate pad redraw during first draw
         return painter.drawPrimitives().then(() => {
            painter.showPadButtons();
            // we restore previous pad name
            painter.selectCurrentPad(prev_name);
            return painter;
         });
      };

   } // TPadPainter

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
   };

   /**
     * @summary Painter for TCanvas object
     *
     * @memberof JSROOT
     * @private
     */

   class TCanvasPainter extends TPadPainter {

      /** @summary Constructor */
      constructor(dom, canvas) {
         super(dom, canvas, true);
         this._websocket = null;
         this.tooltip_allowed = JSROOT.settings.Tooltip;
      }

      /** @summary Cleanup canvas painter */
      cleanup() {
         if (this._changed_layout)
            this.setLayoutKind('simple');
         delete this._changed_layout;
         super.cleanup();
      }

      /** @summary Returns layout kind */
      getLayoutKind() {
         let origin = this.selectDom('origin'),
            layout = origin.empty() ? "" : origin.property('layout');

         return layout || 'simple';
      }

      /** @summary Set canvas layout kind */
      setLayoutKind(kind, main_selector) {
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
      changeLayout(layout_kind, mainid) {
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
            for (let k = 0; k < lst.length; ++k)
               main.node().appendChild(lst[k]);
            this.setLayoutKind(layout_kind);
            JSROOT.resize(main.node());
            return Promise.resolve(true);
         }

         return JSROOT.require("hierarchy").then(() => {

            let grid = new JSROOT.GridDisplay(origin.node(), layout_kind);

            if (mainid == undefined)
               mainid = (layout_kind.indexOf("vert") == 0) ? 0 : 1;

            main = d3.select(grid.getGridFrame(mainid));
            sidebar = d3.select(grid.getGridFrame(1 - mainid));

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
      toggleProjection(kind) {
         delete this.proj_painter;

         if (kind) this.proj_painter = 1; // just indicator that drawing can be preformed

         if (this.showUI5ProjectionArea)
            return this.showUI5ProjectionArea(kind);

         let layout = 'simple', mainid;

         switch(kind) {
            case "X":
            case "bottom": layout = 'vert2_31'; mainid = 0; break;
            case "Y":
            case "left": layout = 'horiz2_13'; mainid = 1; break;
            case "top": layout = 'vert2_13'; mainid = 1; break;
            case "right": layout = 'horiz2_31'; mainid = 0; break;
         }

         return this.changeLayout(layout, mainid);
      }

      /** @summary Draw projection for specified histogram
        * @private */
      drawProjection(kind, hist, hopt) {

         if (!this.proj_painter) return; // ignore drawing if projection not configured

         if (hopt === undefined) hopt = "hist";

         if (this.proj_painter === 1) {

            let canv = JSROOT.create("TCanvas"),
                pad = this.pad,
                main = this.getFramePainter(), drawopt;

            if (kind == "X") {
               canv.fLeftMargin = pad.fLeftMargin;
               canv.fRightMargin = pad.fRightMargin;
               canv.fLogx = main.logx;
               canv.fUxmin = main.logx ? Math.log10(main.scale_xmin) : main.scale_xmin;
               canv.fUxmax = main.logx ? Math.log10(main.scale_xmax) : main.scale_xmax;
               drawopt = "fixframe";
            } else if (kind == "Y") {
               canv.fBottomMargin = pad.fBottomMargin;
               canv.fTopMargin = pad.fTopMargin;
               canv.fLogx = main.logy;
               canv.fUxmin = main.logy ? Math.log10(main.scale_ymin) : main.scale_ymin;
               canv.fUxmax = main.logy ? Math.log10(main.scale_ymax) : main.scale_ymax;
               drawopt = "rotate";
            }

            canv.fPrimitives.Add(hist, hopt);

            let promise = this.drawInUI5ProjectionArea
                          ? this.drawInUI5ProjectionArea(canv, drawopt)
                          : this.drawInSidePanel(canv, drawopt);

            promise.then(painter => { this.proj_painter = painter; });
         } else {
            let hp = this.proj_painter.getMainPainter();
            if (hp) hp.updateObject(hist, hopt);
            this.proj_painter.redrawPad();
         }
      }

      /** @summary Checks if canvas shown inside ui5 widget
        * @desc Function should be used only from the func which supposed to be replaced by ui5
        * @private */
      testUI5() {
         if (!this.use_openui) return false;
         console.warn("full ui5 should be used - not loaded yet? Please check!!");
         return true;
      }

      /** @summary Draw in side panel
        * @private */
      drawInSidePanel(canv, opt) {
         let side = this.selectDom('origin').select(".side_panel");
         if (side.empty()) return Promise.resolve(null);
         return JSROOT.draw(side.node(), canv, opt);
      }

      /** @summary Show message
        * @desc Used normally with web-based canvas and handled in ui5
        * @private */
      showMessage(msg) {
         if (!this.testUI5())
            jsrp.showProgress(msg, 7000);
      }

      /** @summary Function called when canvas menu item Save is called */
      saveCanvasAsFile(fname) {
         let pnt = fname.indexOf(".");
         this.createImage(fname.substr(pnt+1))
             .then(res => this.sendWebsocket("SAVE:" + fname + ":" + res));
      }

      /** @summary Send command to server to save canvas with specified name
        * @desc Should be only used in web-based canvas
        * @private */
      sendSaveCommand(fname) {
         this.sendWebsocket("PRODUCE:" + fname);
      }

      /** @summary Submit menu request
        * @private */
      submitMenuRequest(painter, kind, reqid) {
         // only single request can be handled, no limit better in RCanvas
         return new Promise(resolveFunc => {
            this._getmenu_callback = resolveFunc;
            this.sendWebsocket('GETMENU:' + reqid); // request menu items for given painter
         });
      }

      /** @summary Submit object exec request
        * @private */
      submitExec(painter, exec, snapid) {
         if (this._readonly || !painter) return;

         if (!snapid) snapid = painter.snapid;
         if (!snapid || (typeof snapid != 'string')) return;

         this.sendWebsocket("OBJEXEC:" + snapid + ":" + exec);
      }

      /** @summary Send text message with web socket
        * @desc used for communication with server-side of web canvas
        * @private */
      sendWebsocket(msg) {
         if (!this._websocket) return;
         if (this._websocket.canSend())
            this._websocket.send(msg);
         else
            console.warn("DROP SEND: " + msg);
      }

      /** @summary Close websocket connection to canvas
        * @private */
      closeWebsocket(force) {
         if (this._websocket) {
            this._websocket.close(force);
            this._websocket.cleanup();
            delete this._websocket;
         }
      }

      /** @summary Create websocket for the canvas
        * @private */
      openWebsocket(socket_kind) {
         this.closeWebsocket();

         this._websocket = new JSROOT.WebWindowHandle(socket_kind);
         this._websocket.setReceiver(this);
         this._websocket.connect();
      }

      /** @summary Use provided connection for the web canvas
        * @private */
      useWebsocket(handle) {
         this.closeWebsocket();

         this._websocket = handle;
         this._websocket.setReceiver(this);
         this._websocket.connect();
      }

      /** @summary Hanler for websocket open event
        * @private */
      onWebsocketOpened(/*handle*/) {
         // indicate that we are ready to recieve any following commands
      }

      /** @summary Hanler for websocket close event
        * @private */
      onWebsocketClosed(/*handle*/) {
         if (!this.embed_canvas)
            jsrp.closeCurrentWindow();
      }

      /** @summary Handle websocket messages
        * @private */
      onWebsocketMsg(handle, msg) {
         console.log("GET MSG len:" + msg.length + " " + msg.substr(0,60));

         if (msg == "CLOSE") {
            this.onWebsocketClosed();
            this.closeWebsocket(true);
         } else if (msg.substr(0,6)=='SNAP6:') {
            // This is snapshot, produced with ROOT6

            let snap = JSROOT.parse(msg.substr(6));

            this.syncDraw(true).then(() => this.redrawPadSnap(snap)).then(() => {
               this.completeCanvasSnapDrawing();
               let ranges = this.getWebPadOptions(); // all data, including subpads
               if (ranges) ranges = ":" + ranges;
               handle.send("READY6:" + snap.fVersion + ranges); // send ready message back when drawing completed
               this.confirmDraw();
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
               this.createImage(cmd.toLowerCase())
                   .then(res => handle.send(reply + res));
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
                on = (that[that.length-1] == '1');
            this.showSection(that.substr(0,that.length-2), on);
         } else if (msg.substr(0,5) == "EDIT:") {
            let obj_painter = this.findSnap(msg.substr(5));
            console.log('GET EDIT ' + msg.substr(5) +  ' found ' + !!obj_painter);
            if (obj_painter)
               this.showSection("Editor", true)
                   .then(() => this.producePadEvent("select", obj_painter.getPadPainter(), obj_painter));

         } else {
            console.log("unrecognized msg " + msg);
         }
      }

      /** @summary Handle pad button click event */
      clickPadButton(funcname, evnt) {
         if (funcname == "ToggleGed") return this.activateGed(this, null, "toggle");
         if (funcname == "ToggleStatus") return this.activateStatusBar("toggle");
         super.clickPadButton(funcname, evnt);
      }

      /** @summary Returns true if event status shown in the canvas */
      hasEventStatus() {
         if (this.testUI5()) return false;
         return this.brlayout ? this.brlayout.hasStatus() : false;
      }

      /** @summary Show/toggle event status bar
        * @private */
      activateStatusBar(state) {
         if (this.testUI5()) return;
         if (this.brlayout)
            this.brlayout.createStatusLine(23, state);
         this.processChanges("sbits", this);
      }

      /** @summary Returns true if GED is present on the canvas */
      hasGed() {
         if (this.testUI5()) return false;
         return this.brlayout ? this.brlayout.hasContent() : false;
      }

      /** @summary Function used to de-activate GED
        * @private */
      removeGed() {
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
      activateGed(objpainter, kind, mode) {
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

      /** @summary Show section of canvas  like menu or editor */
      showSection(that, on) {
         if (this.testUI5())
            return Promise.resolve(false);

         console.log('Show section ' + that + ' flag = ' + on);

         switch(that) {
            case "Menu": break;
            case "StatusBar": this.activateStatusBar(on); break;
            case "Editor": return this.activateGed(this, null, !!on);
            case "ToolBar": break;
            case "ToolTips": this.setTooltipAllowed(on); break;

         }
         return Promise.resolve(true);
      }

      /** @summary Complete handling of online canvas drawing
        * @private */
      completeCanvasSnapDrawing() {
         if (!this.pad) return;

         if (document && !this.embed_canvas && this._websocket)
            document.title = this.pad.fTitle;

         if (this._all_sections_showed) return;
         this._all_sections_showed = true;
         this.showSection("Menu", this.pad.TestBit(TCanvasStatusBits.kMenuBar));
         this.showSection("StatusBar", this.pad.TestBit(TCanvasStatusBits.kShowEventStatus));
         this.showSection("ToolBar", this.pad.TestBit(TCanvasStatusBits.kShowToolBar));
         this.showSection("Editor", this.pad.TestBit(TCanvasStatusBits.kShowEditor));
         this.showSection("ToolTips", this.pad.TestBit(TCanvasStatusBits.kShowToolTips) || this._highlight_connect);
      }

      /** @summary Handle highlight in canvas - delver information to server
        * @private */
      processHighlightConnect(hints) {
         if (!hints || hints.length == 0 || !this._highlight_connect ||
              !this._websocket || this.doingDraw() || !this._websocket.canSend(2)) return;

         let hint = hints[0] || hints[1];
         if (!hint || !hint.painter || !hint.painter.snapid || !hint.user_info) return;
         let pp = hint.painter.getPadPainter() || this;
         if (!pp.snapid) return;

         let arr = [pp.snapid, hint.painter.snapid, "0", "0"];

         if ((hint.user_info.binx !== undefined) && (hint.user_info.biny !== undefined)) {
            arr[2] = hint.user_info.binx.toString();
            arr[3] = hint.user_info.biny.toString();
         }  else if (hint.user_info.bin !== undefined) {
            arr[2] = hint.user_info.bin.toString();
         }

         let msg = JSON.stringify(arr);

         if (this._last_highlight_msg != msg) {
            this._last_highlight_msg = msg;
            this.sendWebsocket("HIGHLIGHT:" + msg);
         }
      }

      /** @summary Method informs that something was changed in the canvas
        * @desc used to update information on the server (when used with web6gui)
        * @private */
      processChanges(kind, painter, subelem) {
         // check if we could send at least one message more - for some meaningful actions
         if (!this._websocket || this._readonly || !this._websocket.canSend(2) || (typeof kind !== "string")) return;

         let msg = "";
         if (!painter) painter = this;
         switch (kind) {
            case "sbits":
               msg = "STATUSBITS:" + this.getStatusBits();
               break;
            case "frame": // when moving frame
            case "zoom":  // when changing zoom inside frame
               if (!painter.getWebPadOptions)
                  painter = painter.getPadPainter();
               if (typeof painter.getWebPadOptions == "function")
                  msg = "OPTIONS6:" + painter.getWebPadOptions("only_this");
               break;
            case "pave_moved":
               if (painter.fillWebObjectOptions) {
                  let info = painter.fillWebObjectOptions();
                  if (info) msg = "PRIMIT6:" + JSROOT.toJSON(info);
               }
               break;
            default:
               if ((kind.substr(0,5) == "exec:") && painter && painter.snapid) {
                  console.log('Call exec', painter.snapid);

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
            this._websocket.send(msg);
         }
      }

      /** @summary Select active pad on the canvas */
      selectActivePad(pad_painter, obj_painter, click_pos) {
         if ((this.snapid === undefined) || !pad_painter) return; // only interactive canvas

         let arg = null, ischanged = false;

         if ((pad_painter.snapid !== undefined) && this._websocket)
            arg = { _typename: "TWebPadClick", padid: pad_painter.snapid.toString(), objid: "", x: -1, y: -1, dbl: false };

         if (!pad_painter.is_active_pad) {
            ischanged = true;
            this.forEachPainterInPad(pp => pp.drawActiveBorder(null, pp === pad_painter), "pads");
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
            this.sendWebsocket("PADCLICKED:" + JSROOT.toJSON(arg));
      }

      /** @summary Return actual TCanvas status bits  */
      getStatusBits() {
         let bits = 0;
         if (this.hasEventStatus()) bits |= TCanvasStatusBits.kShowEventStatus;
         if (this.hasGed()) bits |= TCanvasStatusBits.kShowEditor;
         if (this.isTooltipAllowed()) bits |= TCanvasStatusBits.kShowToolTips;
         if (this.use_openui) bits |= TCanvasStatusBits.kMenuBar;
         return bits;
      }

      /** @summary produce JSON for TCanvas, which can be used to display canvas once again */
      produceJSON() {

         let canv = this.getObject(),
             fill0 = (canv.fFillStyle == 0);

         if (fill0) canv.fFillStyle = 1001;

         if (!this.normal_canvas) {

            // fill list of primitives from painters
            this.forEachPainterInPad(p => {
               if (p.$secondary) return; // ignore all secoandry painters

               let subobj = p.getObject();
               if (subobj && subobj._typename)
                  canv.fPrimitives.Add(subobj, p.getDrawOpt());
            }, "objects");
         }

         let res = JSROOT.toJSON(canv);

         if (fill0) canv.fFillStyle = 0;

         if (!this.normal_canvas)
            canv.fPrimitives.Clear();

         return res;
      }

      /** @summary draw TCanvas */
      static draw(dom, can, opt) {
         let nocanvas = !can;
         if (nocanvas) can = JSROOT.create("TCanvas");

         let painter = new TCanvasPainter(dom, can);
         painter.checkSpecialsInPrimitives(can);

         if (!nocanvas && can.fCw && can.fCh && !JSROOT.batch_mode) {
            let rect0 = painter.selectDom().node().getBoundingClientRect();
            if (!rect0.height && (rect0.width > 0.1*can.fCw)) {
               painter.selectDom().style("width", can.fCw+"px").style("height", can.fCh+"px");
               painter._fixed_size = true;
            }
         }

         painter.decodeOptions(opt);
         painter.normal_canvas = !nocanvas;
         painter.createCanvasSvg(0);

         painter.addPadButtons();

         let promise = (nocanvas && opt.indexOf("noframe") < 0) ? TFramePainter.draw(dom, null) : Promise.resolve(true);
         return promise.then(() => {
            // select global reference - required for keys handling
            jsrp.selectActivePad({ pp: painter, active: true });

            return painter.drawPrimitives();
         }).then(() => {
            painter.showPadButtons();
            return painter;
         });
      }

   } // TCanvasPainter

  /** @summary Ensure TCanvas and TFrame for the painter object
    * @param {Object} painter  - painter object to process
    * @param {string|boolean} frame_kind  - false for no frame or "3d" for special 3D mode
    * @desc Assign dom, creates TCanvas if necessary, add to list of pad painters
    * @memberof JSROOT.Painter */
   function ensureTCanvas(painter, frame_kind) {
      if (!painter) return Promise.reject('Painter not provided in ensureTCanvas');

      // simple check - if canvas there, can use painter
      let svg_c = painter.getCanvSvg(),
          noframe = (frame_kind === false) || (frame_kind == "3d") ? "noframe" : "",
          promise = !svg_c.empty() ? Promise.resolve(true) : TCanvasPainter.draw(painter.getDom(), null, noframe);

      return promise.then(() => {
         if (frame_kind === false) return;

         if (painter.getFrameSvg().select(".main_layer").empty() && !painter.getFramePainter())
            return TFramePainter.draw(painter.getDom(), null, (typeof frame_kind === "string") ? frame_kind : "");
      }).then(() => {
         painter.addToPadPrimitives();
         return painter;
      });
   }

   /** @summary draw TPad snapshot from TWebCanvas
     * @memberof JSROOT.Painter
     * @private */
   function drawTPadSnapshot(dom, snap /*, opt*/) {
      let can = JSROOT.create("TCanvas"),
          painter = new TCanvasPainter(dom, can);
      painter.normal_canvas = false;
      painter.addPadButtons();

      return painter.syncDraw(true)
                    .then(() => painter.redrawPadSnap(snap))
                    .then(() => { painter.confirmDraw(); painter.showPadButtons(); return painter; });
   }

   JSROOT.TAxisPainter = TAxisPainter;
   JSROOT.TFramePainter = TFramePainter;
   JSROOT.TPadPainter = TPadPainter;
   JSROOT.TCanvasPainter = TCanvasPainter;

   jsrp.ensureTCanvas = ensureTCanvas;
   jsrp.drawTPadSnapshot = drawTPadSnapshot;

   if (JSROOT.nodejs) module.exports = JSROOT;
   return JSROOT;
});
