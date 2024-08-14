import { gStyle, settings, constants, clTAxis, clTGaxis, isFunc } from '../core.mjs';
import { select as d3_select, drag as d3_drag, timeFormat as d3_timeFormat, utcFormat as d3_utcFormat,
         scaleTime as d3_scaleTime, scaleSymlog as d3_scaleSymlog,
         scaleLog as d3_scaleLog, scaleLinear as d3_scaleLinear } from '../d3.mjs';
import { floatToString, makeTranslate, addHighlightStyle } from '../base/BasePainter.mjs';
import { ObjectPainter, EAxisBits, kAxisLabels, kAxisNormal, kAxisFunc, kAxisTime } from '../base/ObjectPainter.mjs';
import { FontHandler } from '../base/FontHandler.mjs';


/** @summary Return time offset value for given TAxis object
  * @private */
function getTimeOffset(axis) {
   const dflt_time_offset = 788918400000;
   if (!axis) return dflt_time_offset;
   const idF = axis.fTimeFormat.indexOf('%F');
   if (idF < 0) return gStyle.fTimeOffset * 1000;
   let sof = axis.fTimeFormat.slice(idF + 2);
   // default string in axis offset
   if (sof.indexOf('1995-01-01 00:00:00s0') === 0) return dflt_time_offset;
   // special case, used from DABC painters
   if ((sof === '0') || (sof === '')) return 0;

   // decode time from ROOT string
   const next = (separ, min, max) => {
      const pos = sof.indexOf(separ);
      if (pos < 0) return min;
      const val = parseInt(sof.slice(0, pos));
      sof = sof.slice(pos + 1);
      if (!Number.isInteger(val) || (val < min) || (val > max)) return min;
      return val;
   }, year = next('-', 1900, 2900),
      month = next('-', 1, 12) - 1,
      day = next(' ', 1, 31),
      hour = next(':', 0, 23),
      min = next(':', 0, 59),
      sec = next('s', 0, 59),
      msec = next(' ', 0, 999),
      dt = new Date(Date.UTC(year, month, day, hour, min, sec, msec));

   let offset = dt.getTime();

   // now also handle suffix like GMT or GMT -0600
   sof = sof.toUpperCase();

   if (sof.indexOf('GMT') === 0) {
      sof = sof.slice(4).trim();
      if (sof.length > 3) {
         let p = 0, sign = 1000;
         if (sof[0] === '-') { p = 1; sign = -1000; }
         offset -= sign * (parseInt(sof.slice(p, p + 2)) * 3600 + parseInt(sof.slice(p + 2, p + 4)) * 60);
      }
   }

   return offset;
}

/** @summary Return true when GMT option configured in time format
  * @private */
function getTimeGMT(axis) {
   const fmt = axis?.fTimeFormat ?? '';
   return (fmt.indexOf('gmt') > 0) || (fmt.indexOf('GMT') > 0);
}

/** @summary Tries to choose time format for provided time interval
  * @private */
function chooseTimeFormat(awidth, ticks) {
   if (awidth < 0.5) return ticks ? '%S.%L' : '%H:%M:%S.%L';
   if (awidth < 30) return ticks ? '%Mm%S' : '%H:%M:%S';
   awidth /= 60; if (awidth < 30) return ticks ? '%Hh%M' : '%d/%m %H:%M';
   awidth /= 60; if (awidth < 12) return ticks ? '%d-%Hh' : '%d/%m/%y %Hh';
   awidth /= 24; if (awidth < 15.218425) return ticks ? '%d/%m' : '%d/%m/%y';
   awidth /= 30.43685; if (awidth < 6) return '%d/%m/%y';
   awidth /= 12; if (awidth < 2) return ticks ? '%m/%y' : '%d/%m/%y';
   return '%Y';
}

/**
  * @summary Base axis painter methods
  *
  * @private
  */

const AxisPainterMethods = {

   initAxisPainter() {
      this.name = 'yaxis';
      this.kind = kAxisNormal;
      this.func = null;
      this.order = 0; // scaling order for axis labels

      this.full_min = 0;
      this.full_max = 1;
      this.scale_min = 0;
      this.scale_max = 1;
      this.ticks = []; // list of major ticks
   },

   /** @summary Cleanup axis painter */
   cleanupAxisPainter() {
      this.ticks = [];
      delete this.format;
      delete this.func;
      delete this.tfunc1;
      delete this.tfunc2;
      delete this.gr;
   },

   /** @summary Assign often used members of frame painter */
   assignFrameMembers(fp, axis) {
      fp[`gr${axis}`] = this.gr;                 // fp.grx
      fp[`log${axis}`] = this.log;               // fp.logx
      fp[`scale_${axis}min`] = this.scale_min;   // fp.scale_xmin
      fp[`scale_${axis}max`] = this.scale_max;   // fp.scale_xmax
   },

   /** @summary Convert axis value into the Date object */
   convertDate(v) {
      return new Date(this.timeoffset + v*1000);
   },

   /** @summary Convert graphical point back into axis value */
   revertPoint(pnt) {
      const value = this.func.invert(pnt);
      return this.kind === kAxisTime ? (value - this.timeoffset) / 1000 : value;
   },

   /** @summary Provide label for time axis */
   formatTime(d, asticks) {
      return asticks ? this.tfunc1(d) : this.tfunc2(d);
   },

   /** @summary Provide label for log axis */
   formatLog(d, asticks, fmt) {
      const val = parseFloat(d), rnd = Math.round(val);
      if (!asticks)
         return ((rnd === val) && (Math.abs(rnd) < 1e9)) ? rnd.toString() : floatToString(val, fmt || gStyle.fStatFormat);
      if (val <= 0) return null;
      let vlog = Math.log10(val);
      const base = this.logbase;
      if (base !== 10) vlog = vlog / Math.log10(base);
      if (this.moreloglabels || (Math.abs(vlog - Math.round(vlog)) < 0.001)) {
         if (!this.noexp && (asticks !== 2))
            return this.formatExp(base, Math.floor(vlog + 0.01), val);
         if (Math.abs(base - Math.E) < 0.001)
            return floatToString(val, fmt || gStyle.fStatFormat);
         return (vlog < 0) ? val.toFixed(Math.round(-vlog + 0.5)) : val.toFixed(0);
      }
      return null;
   },

   /** @summary Provide label for normal axis */
   formatNormal(d, asticks, fmt) {
      let val = parseFloat(d);
      if (asticks && this.order)
         val = val / Math.pow(10, this.order);

      if (gStyle.fStripDecimals && (val === Math.round(val)))
         return Math.abs(val) < 1e9 ? val.toFixed(0) : val.toExponential(4);

      if (asticks) {
         if (this.ndig > 10)
            return val.toExponential(this.ndig - 11);
         let res = val.toFixed(this.ndig);
         const p = res.indexOf('.');
         if ((p > 0) && settings.StripAxisLabels) {
            while ((res.length >= p) && ((res[res.length-1] === '0') || (res[res.length-1] === '.')))
               res = res.slice(0, res.length - 1);
         }
         return res;
      }

      return floatToString(val, fmt || gStyle.fStatFormat);
   },

   /** @summary Provide label for exponential form */
   formatExp(base, order, value) {
      let res = '';
      const sbase = Math.abs(base - Math.E) < 0.001 ? 'e' : base.toString();
      if (value) {
         value = Math.round(value/Math.pow(base, order));
         if (settings.StripAxisLabels) {
            if (order === 0)
               return value.toString();
            else if ((order === 1) && (value === 1))
               return sbase;
         }
         if (value !== 1)
            res = value.toString() + (settings.Latex ? '#times' : 'x');
      }
      res += sbase;
      if (settings.Latex > constants.Latex.Symbols)
         return res + `^{${order}}`;
      const superscript_symbols = {
            0: '\u2070', 1: '\xB9', 2: '\xB2', 3: '\xB3', 4: '\u2074', 5: '\u2075',
            6: '\u2076', 7: '\u2077', 8: '\u2078', 9: '\u2079', '-': '\u207B'
      }, str = order.toString();
      for (let n = 0; n < str.length; ++n)
         res += superscript_symbols[str[n]];
      return res;
   },

   /** @summary Convert 'raw' axis value into text */
   axisAsText(value, fmt) {
      if (this.kind === kAxisTime)
         value = this.convertDate(value);
      if (this.format)
         return this.format(value, false, fmt);
      return value.toPrecision(4);
   },

   /** @summary Produce ticks for d3.scaleLog
     * @desc Fixing following problem, described [here]{@link https://stackoverflow.com/questions/64649793} */
   poduceLogTicks(func, number) {
      const linearArray = arr => {
         let sum1 = 0, sum2 = 0;
         for (let k = 1; k < arr.length; ++k) {
            const diff = (arr[k] - arr[k-1]);
            sum1 += diff;
            sum2 += diff**2;
         }
         const mean = sum1/(arr.length-1),
             dev = sum2/(arr.length-1) - mean**2;

         if (dev <= 0) return true;
         if (Math.abs(mean) < 1e-100) return false;
         return Math.sqrt(dev)/mean < 1e-6;
      };

      let arr = func.ticks(number);

      while ((number > 4) && linearArray(arr)) {
         number = Math.round(number*0.8);
         arr = func.ticks(number);
      }

      // if still linear array, try to sort out 'bad' ticks
      if ((number < 5) && linearArray(arr) && this.logbase && (this.logbase !== 10)) {
         const arr2 = [];
         arr.forEach(val => {
            const pow = Math.log10(val) / Math.log10(this.logbase);
            if (Math.abs(Math.round(pow) - pow) < 0.01) arr2.push(val);
         });
         if (arr2.length > 0) arr = arr2;
      }

      return arr;
   },

   /** @summary Produce axis ticks */
   produceTicks(ndiv, ndiv2) {
      if (!this.noticksopt) {
         const total = ndiv * (ndiv2 || 1);

         if (this.log) return this.poduceLogTicks(this.func, total);

         const dom = this.func.domain(),
            check = ticks => {
               if (ticks.length <= total) return true;
               if (ticks.length > total + 1) return false;
               return (ticks[0] === dom[0]) || (ticks[total] === dom[1]); // special case of N+1 ticks, but match any range
            }, res1 = this.func.ticks(total);
         if (ndiv2 || check(res1)) return res1;

         const res2 = this.func.ticks(Math.round(total * 0.7));
         return (res2.length > 2) && check(res2) ? res2 : res1;
      }

      const dom = this.func.domain(), ticks = [];
      if (ndiv2) ndiv = (ndiv-1) * ndiv2;
      for (let n = 0; n <= ndiv; ++n)
         ticks.push((dom[0]*(ndiv-n) + dom[1]*n)/ndiv);
      return ticks;
   },

   /** @summary Method analyze mouse wheel event and returns item with suggested zooming range */
   analyzeWheelEvent(evnt, dmin, item, test_ignore) {
      if (!item) item = {};

      let delta = 0, delta_left = 1, delta_right = 1;

      if ('dleft' in item) { delta_left = item.dleft; delta = 1; }
      if ('dright' in item) { delta_right = item.dright; delta = 1; }

      if (item.delta)
         delta = item.delta;
       else if (evnt)
         delta = evnt.wheelDelta ? -evnt.wheelDelta : (evnt.deltaY || evnt.detail);

      if (!delta || (test_ignore && item.ignore)) return;

      delta = (delta < 0) ? -0.2 : 0.2;
      delta_left *= delta;
      delta_right *= delta;

      const lmin = item.min = this.scale_min,
            lmax = item.max = this.scale_max,
            gmin = this.full_min,
            gmax = this.full_max;

      if ((item.min === item.max) && (delta < 0)) {
         item.min = gmin;
         item.max = gmax;
      }

      if (item.min >= item.max) return;

      if (item.reverse) dmin = 1 - dmin;

      if ((dmin > 0) && (dmin < 1)) {
         if (this.log) {
            let factor = (item.min > 0) ? Math.log10(item.max/item.min) : 2;
            if (factor > 10) factor = 10; else if (factor < 0.01) factor = 0.01;
            item.min = item.min / Math.pow(10, factor*delta_left*dmin);
            item.max = item.max * Math.pow(10, factor*delta_right*(1-dmin));
         } else if ((delta_left === -delta_right) && !item.reverse) {
            // shift left/right, try to keep range constant
            let delta = (item.max - item.min) * delta_right * dmin;

            if ((Math.round(item.max) === item.max) && (Math.round(item.min) === item.min) && (Math.abs(delta) > 1)) delta = Math.round(delta);

            if (item.min + delta < gmin)
               delta = gmin - item.min;
            else if (item.max + delta > gmax)
               delta = gmax - item.max;

            if (delta !== 0) {
               item.min += delta;
               item.max += delta;
             } else {
               delete item.min;
               delete item.max;
            }
         } else {
            let rx_left = (item.max - item.min), rx_right = rx_left;
            if (delta_left > 0) rx_left = 1.001 * rx_left / (1-delta_left);
            item.min += -delta_left*dmin*rx_left;
            if (delta_right > 0) rx_right = 1.001 * rx_right / (1-delta_right);
            item.max -= -delta_right*(1-dmin)*rx_right;
         }
         if (item.min >= item.max)
            item.min = item.max = undefined;
          else if (delta_left !== delta_right) {
            // extra check case when moving left or right
            if (((item.min < gmin) && (lmin === gmin)) ||
                ((item.max > gmax) && (lmax === gmax)))
                   item.min = item.max = undefined;
         } else {
            if (item.min < gmin) item.min = gmin;
            if (item.max > gmax) item.max = gmax;
         }
      } else
         item.min = item.max = undefined;


      item.changed = ((item.min !== undefined) && (item.max !== undefined));

      return item;
   }

}; // AxisPainterMethods


/**
 * @summary Painter for TAxis object
 *
 * @private
 */

class TAxisPainter extends ObjectPainter {

   /** @summary constructor
     * @param {object|string} dom - identifier or dom element
     * @param {object} axis - object to draw
     * @param {boolean} embedded - if true, painter used in other objects painters */
   constructor(dom, axis, embedded) {
      super(dom, axis);

      this.is_gaxis = axis?._typename === clTGaxis;

      Object.assign(this, AxisPainterMethods);
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
   getAxisType() { return clTAxis; }

   /** @summary Configure axis painter
     * @desc Axis can be drawn inside frame <g> group with offset to 0 point for the frame
     * Therefore one should distinguish when caclulated coordinates used for axis drawing itself or for calculation of frame coordinates
     * @private */
   configureAxis(name, min, max, smin, smax, vertical, range, opts) {
      this.name = name;
      this.full_min = min;
      this.full_max = max;
      this.kind = kAxisNormal;
      this.vertical = vertical;
      this.log = opts.log || 0;
      this.noexp_changed = opts.noexp_changed;
      this.symlog = opts.symlog || false;
      this.reverse = opts.reverse || false;
      this.swap_side = opts.swap_side || false;
      this.fixed_ticks = opts.fixed_ticks || null;
      this.maxTickSize = opts.maxTickSize || 0;

      const axis = this.getObject();

      if (opts.time_scale || axis.fTimeDisplay) {
         this.kind = kAxisTime;
         this.timeoffset = getTimeOffset(axis);
         this.timegmt = getTimeGMT(axis);
      } else if (opts.axis_func)
         this.kind = kAxisFunc;
       else
         this.kind = !axis.fLabels ? kAxisNormal : kAxisLabels;


      if (this.kind === kAxisTime)
         this.func = d3_scaleTime().domain([this.convertDate(smin), this.convertDate(smax)]);
       else if (this.log) {
         if ((this.log === 1) || (this.log === 10))
            this.logbase = 10;
         else if (this.log === 3)
            this.logbase = Math.E;
         else
            this.logbase = Math.round(this.log);

         if (smax <= 0) smax = 1;

         if ((smin <= 0) && axis && !opts.logcheckmin) {
            for (let i = 0; i < axis.fNbins; ++i) {
               smin = Math.max(smin, axis.GetBinLowEdge(i+1));
               if (smin > 0) break;
            }
         }

         if ((smin <= 0) && opts.log_min_nz)
            smin = this.log_min_nz = opts.log_min_nz;

         if ((smin <= 0) || (smin >= smax))
            smin = smax * (opts.logminfactor || 1e-4);

         if (this.kind === kAxisFunc)
            this.func = this.createFuncHandle(opts.axis_func, this.logbase, smin, smax);
         else
            this.func = d3_scaleLog().base(this.logbase).domain([smin, smax]);
      } else if (this.symlog) {
         let v = Math.max(Math.abs(smin), Math.abs(smax));
         if (Number.isInteger(this.symlog) && (this.symlog > 0))
            v *= Math.pow(10, -1*this.symlog);
         else
            v *= 0.01;
         this.func = d3_scaleSymlog().constant(v).domain([smin, smax]);
      } else if (this.kind === kAxisFunc)
         this.func = this.createFuncHandle(opts.axis_func, 0, smin, smax);
       else
         this.func = d3_scaleLinear().domain([smin, smax]);


      if (this.vertical ^ this.reverse) {
         const d = range[0]; range[0] = range[1]; range[1] = d;
      }

      this.func.range(range);

      this.scale_min = smin;
      this.scale_max = smax;

      if (this.kind === kAxisTime)
         this.gr = val => this.func(this.convertDate(val));
      else if (this.log)
         this.gr = val => (val < this.scale_min) ? (this.vertical ? this.func.range()[0]+5 : -5) : this.func(val);
      else
         this.gr = this.func;

      delete this.format;// remove formatting func

      let ndiv = 508;
      if (this.is_gaxis)
         ndiv = axis.fNdiv;
      else if (axis) {
          if (!axis.fNdivisions)
             ndiv = 0;
          else
             ndiv = Math.max(axis.fNdivisions, 4);
      }

      this.nticks = ndiv % 100;
      this.nticks2 = (ndiv % 10000 - this.nticks) / 100;
      this.nticks3 = Math.floor(ndiv/10000);

      if (axis && !this.is_gaxis && (this.nticks > 20)) this.nticks = 20;

      let gr_range = Math.abs(this.func.range()[1] - this.func.range()[0]);
      if (gr_range <= 0) gr_range = 100;

      if (this.kind === kAxisTime) {
         if (this.nticks > 8) this.nticks = 8;

         const scale_range = this.scale_max - this.scale_min,
               idF = axis.fTimeFormat.indexOf('%F'),
               tf2 = chooseTimeFormat(scale_range / gr_range, false);
         let tf1 = (idF >= 0) ? axis.fTimeFormat.slice(0, idF) : axis.fTimeFormat;

         if (!tf1 || (scale_range < 0.1 * (this.full_max - this.full_min)))
            tf1 = chooseTimeFormat(scale_range / this.nticks, true);

         this.tfunc1 = this.tfunc2 = this.timegmt ? d3_utcFormat(tf1) : d3_timeFormat(tf1);
         if (tf2 !== tf1)
            this.tfunc2 = this.timegmt ? d3_utcFormat(tf2) : d3_timeFormat(tf2);

         this.format = this.formatTime;
      } else if (this.log) {
         if (this.nticks2 > 1) {
            this.nticks *= this.nticks2; // all log ticks (major or minor) created centrally
            this.nticks2 = 1;
         }
         this.noexp = axis?.TestBit(EAxisBits.kNoExponent);
         if ((this.scale_max < 300) && (this.scale_min > 0.3) && !this.noexp_changed) this.noexp = true;
         this.moreloglabels = axis?.TestBit(EAxisBits.kMoreLogLabels);
         this.format = this.formatLog;
      } else if (this.kind === kAxisLabels) {
         this.nticks = 50; // for text output allow max 50 names
         const scale_range = this.scale_max - this.scale_min;
         if (this.nticks > scale_range)
            this.nticks = Math.round(scale_range);

         this.regular_labels = true;

         if (axis && axis.fNbins && axis.fLabels) {
            if ((axis.fNbins !== Math.round(axis.fXmax - axis.fXmin)) ||
                (axis.fXmin !== 0) || (axis.fXmax !== axis.fNbins))
               this.regular_labels = false;
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
      return this.func?.domain()[0] ?? 0;
   }

   /** @summary Return scale max */
   getScaleMax() {
      return this.func?.domain()[1] ?? 0;
   }

   /** @summary Provide label for axis value */
   formatLabels(d) {
      const a = this.getObject();
      let indx = parseFloat(d);
      if (!this.regular_labels)
         indx = Math.round((indx - a.fXmin)/(a.fXmax - a.fXmin) * a.fNbins);
      else
         indx = Math.floor(indx);
      if ((indx < 0) || (indx >= a.fNbins)) return null;
      for (let i = 0; i < a.fLabels.arr.length; ++i) {
         const tstr = a.fLabels.arr[i];
         if (tstr.fUniqueID === indx+1) return tstr.fString;
      }
      return null;
   }

   /** @summary Creates array with minor/middle/major ticks */
   createTicks(only_major_as_array, optionNoexp, optionNoopt, optionInt) {
      if (optionNoopt && this.nticks && (this.kind === kAxisNormal))
         this.noticksopt = true;

      const handle = { painter: this, nminor: 0, nmiddle: 0, nmajor: 0, func: this.func, minor: [], middle: [], major: [] };
      let ticks = [];

      if (this.fixed_ticks) {
         this.fixed_ticks.forEach(v => {
            if ((v >= this.scale_min) && (v <= this.scale_max)) ticks.push(v);
         });
      } else if (this.kind === kAxisLabels) {
         handle.lbl_pos = [];
         const axis = this.getObject();
         for (let n = 0; n <= axis.fNbins; ++n) {
            const x = this.regular_labels ? n : axis.fXmin + n / axis.fNbins * (axis.fXmax - axis.fXmin);
            if ((x >= this.scale_min) && (x <= this.scale_max)) {
               handle.lbl_pos.push(x);
               ticks.push(x);
            }
         }
      } else
         ticks = this.produceTicks(this.nticks);

      handle.minor = handle.middle = handle.major = ticks;

      if (only_major_as_array) {
         const res = handle.major, delta = (this.scale_max - this.scale_min)*1e-5;
         if (res[0] > this.scale_min + delta) res.unshift(this.scale_min);
         if (res[res.length-1] < this.scale_max - delta) res.push(this.scale_max);
         return res;
      }

      if ((this.nticks2 > 1) && (!this.log || (this.logbase === 10)) && !this.fixed_ticks) {
         handle.minor = handle.middle = this.produceTicks(handle.major.length, this.nticks2);

         const gr_range = Math.abs(this.func.range()[1] - this.func.range()[0]);

         // avoid black filling by middle-size
         if ((handle.middle.length <= handle.major.length) || (handle.middle.length > gr_range/3.5))
            handle.minor = handle.middle = handle.major;
          else if ((this.nticks3 > 1) && !this.log) {
            handle.minor = this.produceTicks(handle.middle.length, this.nticks3);
            if ((handle.minor.length <= handle.middle.length) || (handle.minor.length > gr_range/1.7))
               handle.minor = handle.middle;
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

         if ((this.nmajor < this.major.length) && (Math.abs(this.grpos - this.func(this.major[this.nmajor])) < 1)) {
            this.nmajor++;
            this.kind = 1;
         }
         return true;
      };

      handle.last_major = function() {
         return (this.kind !== 1) ? false : this.nmajor === this.major.length;
      };

      handle.next_major_grpos = function() {
         if (this.nmajor >= this.major.length) return null;
         return this.func(this.major[this.nmajor]);
      };

      handle.get_modifier = function() {
         return this.painter.findLabelModifier(this.painter.getObject(), this.nmajor-1, this.major);
      };

      this.order = 0;
      this.ndig = 0;

      // at the moment when drawing labels, we can try to find most optimal text representation for them

      if (((this.kind === kAxisNormal) || (this.kind === kAxisFunc)) && !this.log && (handle.major.length > 0)) {
         let maxorder = 0, minorder = 0, exclorder3 = false;

         if (!optionNoexp) {
            const maxtick = Math.max(Math.abs(handle.major[0]), Math.abs(handle.major[handle.major.length-1])),
                mintick = Math.min(Math.abs(handle.major[0]), Math.abs(handle.major[handle.major.length-1])),
                ord1 = (maxtick > 0) ? Math.round(Math.log10(maxtick)/3)*3 : 0,
                ord2 = (mintick > 0) ? Math.round(Math.log10(mintick)/3)*3 : 0;

             exclorder3 = (maxtick < 2e4); // do not show 10^3 for values below 20000

             if (maxtick || mintick) {
                maxorder = Math.max(ord1, ord2) + 3;
                minorder = Math.min(ord1, ord2) - 3;
             }
         }

         // now try to find best combination of order and ndig for labels

         let bestorder = 0, bestndig = this.ndig, bestlen = 1e10;

         for (let order = minorder; order <= maxorder; order+=3) {
            if (exclorder3 && (order === 3)) continue;
            this.order = order;
            this.ndig = 0;
            let lbls = [], indx = 0, totallen = 0;
            while (indx < handle.major.length) {
               const lbl = this.format(handle.major[indx], true);
               if (lbls.indexOf(lbl) < 0) {
                  lbls.push(lbl);
                  const p = lbl.indexOf('.');
                  if (!order && !optionNoexp && ((p > gStyle.fAxisMaxDigits) || ((p < 0) && (lbl.length > gStyle.fAxisMaxDigits)))) {
                     totallen += 1e10; // do not use order = 0 when too many digits are there
                     exclorder3 = false;
                  }
                  totallen += lbl.length;
                  indx++;
                  continue;
               }
               if (++this.ndig > 15) break; // not too many digits, anyway it will be exponential
               lbls = []; indx = 0; totallen = 0;
            }

            // for order === 0 we should virtually remove '0.' and extra label on top
            if (!order && (this.ndig < 4))
               totallen -= handle.major.length * 2 + 3;

            if (totallen < bestlen) {
               bestlen = totallen;
               bestorder = this.order;
               bestndig = this.ndig;
            }
         }

         this.order = bestorder;
         this.ndig = bestndig;

         if (optionInt) {
            if (this.order) console.warn(`Axis painter - integer labels are configured, but axis order ${this.order} is preferable`);
            if (this.ndig) console.warn(`Axis painter - integer labels are configured, but ${this.ndig} decimal digits are required`);
            this.ndig = 0;
            this.order = 0;
         }
      }

      return handle;
   }

   /** @summary Is labels should be centered */
   isCenteredLabels() {
      if (this.kind === kAxisLabels) return true;
      if (this.log) return false;
      return this.getObject()?.TestBit(EAxisBits.kCenterLabels);
   }

   /** @summary Add interactive elements to draw axes title */
   addTitleDrag(title_g, vertical, offset_k, reverse, axis_length) {
      if (!settings.MoveResize || this.isBatchMode()) return;

      let drag_rect = null,
          acc_x, acc_y, new_x, new_y, sign_0, alt_pos, curr_indx;
      const drag_move = d3_drag().subject(Object);

      drag_move.on('start', evnt => {
         evnt.sourceEvent.preventDefault();
         evnt.sourceEvent.stopPropagation();

         const box = title_g.node().getBBox(), // check that elements visible, request precise value
             title_length = vertical ? box.height : box.width;

         new_x = acc_x = title_g.property('shift_x');
         new_y = acc_y = title_g.property('shift_y');

         sign_0 = vertical ? (acc_x > 0) : (acc_y > 0); // sign should remain

         alt_pos = vertical ? [axis_length, axis_length/2, 0] : [0, axis_length/2, axis_length]; // possible positions
         const off = vertical ? -title_length/2 : title_length/2;
         if (this.title_align === 'middle') {
            alt_pos[0] += off;
            alt_pos[2] -= off;
         } else if (this.title_align === 'begin') {
            alt_pos[1] -= off;
            alt_pos[2] -= 2*off;
         } else { // end
            alt_pos[0] += 2*off;
            alt_pos[1] += off;
         }

         if (this.titleCenter)
            curr_indx = 1;
         else if (reverse ^ this.titleOpposite)
            curr_indx = 0;
         else
            curr_indx = 2;

         alt_pos[curr_indx] = vertical ? acc_y : acc_x;

         drag_rect = title_g.append('rect')
            .attr('x', box.x)
            .attr('y', box.y)
            .attr('width', box.width)
            .attr('height', box.height)
            .style('cursor', 'move')
            .call(addHighlightStyle, true);
         //   .style('pointer-events','none'); // let forward double click to underlying elements
      }).on('drag', evnt => {
         if (!drag_rect) return;

         evnt.sourceEvent.preventDefault();
         evnt.sourceEvent.stopPropagation();

         acc_x += evnt.dx;
         acc_y += evnt.dy;

         let set_x, set_y, besti = 0;
         const p = vertical ? acc_y : acc_x;

         for (let i = 1; i < 3; ++i)
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
            makeTranslate(title_g, new_x, new_y);
         }
      }).on('end', evnt => {
         if (!drag_rect) return;

         evnt.sourceEvent.preventDefault();
         evnt.sourceEvent.stopPropagation();

         title_g.property('shift_x', new_x)
                .property('shift_y', new_y);

         const axis = this.getObject(), axis2 = this.source_axis,
               setBit = (bit, on) => {
                  if (axis && axis.TestBit(bit) !== on) axis.InvertBit(bit);
                  if (axis2 && axis2.TestBit(bit) !== on) axis2.InvertBit(bit);
               };

         this.titleOffset = (vertical ? new_x : new_y) / offset_k;
         const offset = this.titleOffset / this.offsetScaling / this.titleSize;
         if (axis) axis.fTitleOffset = offset;
         if (axis2) axis2.fTitleOffset = offset;

         if (curr_indx === 1) {
            setBit(EAxisBits.kCenterTitle, true); this.titleCenter = true;
            setBit(EAxisBits.kOppositeTitle, false); this.titleOpposite = false;
         } else if (curr_indx === 0) {
            setBit(EAxisBits.kCenterTitle, false); this.titleCenter = false;
            setBit(EAxisBits.kOppositeTitle, true); this.titleOpposite = true;
         } else {
            setBit(EAxisBits.kCenterTitle, false); this.titleCenter = false;
            setBit(EAxisBits.kOppositeTitle, false); this.titleOpposite = false;
         }

         this.submitAxisExec(`SetTitleOffset(${offset});;SetBit(${EAxisBits.kCenterTitle},${this.titleCenter?1:0})`);

         drag_rect.remove();
         drag_rect = null;
      });

      title_g.style('cursor', 'move').call(drag_move);
   }

   /** @summary Configure hist painter which creates axis - to be able submit execs
     * @private */
   setHistPainter(hist_painter, axis_name) {
      this.hist_painter = hist_painter;
      this.hist_axis = axis_name;
   }

   /** @summary Submit exec for the axis - if possible
     * @private */
   submitAxisExec(exec, only_gaxis) {
      const snapid = this.hist_painter?.snapid;
      if (snapid && this.hist_axis && !only_gaxis)
         this.submitCanvExec(exec, `${snapid}#${this.hist_axis}`);
      else if (this.is_gaxis)
         this.submitCanvExec(exec);
   }

   /** @summary Produce svg path for axis ticks */
   produceTicksPath(handle, side, tickSize, ticksPlusMinus, secondShift, real_draw) {
      let path1 = '', path2 = '';
      this.ticks = [];

      while (handle.next(true)) {
         let h1 = Math.round(tickSize/4), h2 = 0;

         if (handle.kind < 3)
            h1 = Math.round(tickSize/2);

         if (handle.kind === 1) {
            // if not showing labels, not show large tick
            // FIXME: for labels last tick is smaller,
            if (/* (this.kind === kAxisLabels) || */ (this.format(handle.tick, true) !== null)) h1 = tickSize;
            this.ticks.push(handle.grpos); // keep graphical positions of major ticks
         }

         if (ticksPlusMinus > 0)
            h2 = -h1;
          else if (side < 0) {
            h2 = -h1; h1 = 0;
         }

         path1 += this.vertical ? `M${h1},${handle.grpos}H${h2}` : `M${handle.grpos},${-h1}V${-h2}`;

         if (secondShift)
            path2 += this.vertical ? `M${secondShift-h1},${handle.grpos}H${secondShift-h2}` : `M${handle.grpos},${secondShift+h1}V${secondShift+h2}`;
      }

      return real_draw ? path1 + path2 : '';
   }

   /** @summary Returns modifier for axis label */
   findLabelModifier(axis, nlabel, positions) {
      if (!axis.fModLabs) return null;
      for (let n = 0; n < axis.fModLabs.arr.length; ++n) {
         const mod = axis.fModLabs.arr[n];

         if ((mod.fLabValue !== undefined) && (mod.fLabNum === 0)) {
            const eps = this.log ? positions[nlabel]*1e-6 : (this.scale_max - this.scale_min)*1e-6;
            if (Math.abs(mod.fLabValue - positions[nlabel]) < eps)
               return mod;
         }

         if ((mod.fLabNum === nlabel + 1) ||
             ((mod.fLabNum < 0) && (nlabel === positions.length + mod.fLabNum)))
                return mod;
      }
      return null;
   }

   /** @summary Draw axis labels
     * @return {Promise} with array label size and max width */
   async drawLabels(axis_g, axis, w, h, handle, side, labelsFont, labeloffset, tickSize, ticksPlusMinus, max_text_width, frame_ygap) {
      const center_lbls = this.isCenteredLabels(),
            rotate_lbls = axis.TestBit(EAxisBits.kLabelsVert),
            label_g = [axis_g.append('svg:g').attr('class', 'axis_labels')],
            lbl_pos = handle.lbl_pos || handle.major,
            tilt_angle = gStyle.AxisTiltAngle ?? 25;
      let textscale = 1, maxtextlen = 0, applied_scale = 0,
          lbl_tilt = false, any_modified = false, max_textwidth = 0, max_tiltsize = 0;

      if (this.lbls_both_sides)
         label_g.push(axis_g.append('svg:g').attr('class', 'axis_labels').attr('transform', this.vertical ? `translate(${w})` : `translate(0,${-h})`));

       if (frame_ygap > 0)
          max_tiltsize = frame_ygap / Math.sin(tilt_angle/180*Math.PI) - Math.tan(tilt_angle/180*Math.PI);

      // function called when text is drawn to analyze width, required to correctly scale all labels
      // must be function to correctly handle 'this' argument
      function process_drawtext_ready(painter) {
         const textwidth = this.result_width;
         max_textwidth = Math.max(max_textwidth, textwidth);

         if (textwidth && ((!painter.vertical && !rotate_lbls) || (painter.vertical && rotate_lbls)) && !painter.log) {
            let maxwidth = this.gap_before*0.45 + this.gap_after*0.45;
            if (!this.gap_before)
               maxwidth = 0.9*this.gap_after;
            else if (!this.gap_after)
               maxwidth = 0.9*this.gap_before;
            textscale = Math.min(textscale, maxwidth / textwidth);
         } else if (painter.vertical && max_text_width && this.normal_side && (max_text_width - labeloffset > 20) && (textwidth > max_text_width - labeloffset))
            textscale = Math.min(textscale, (max_text_width - labeloffset) / textwidth);

         if ((textscale > 0.0001) && (textscale < 0.7) && !any_modified &&
              !painter.vertical && !rotate_lbls && (maxtextlen > 5) && (label_g.length === 1) && (lbl_tilt === false))
            lbl_tilt = true;

         let scale = textscale;

         if (lbl_tilt) {
            if (max_tiltsize && max_textwidth) {
               scale = Math.min(1, 0.8*max_tiltsize/max_textwidth);
               if (scale < textscale) {
                  // if due to tilt scale is even smaller - ignore tilting
                  lbl_tilt = 0;
                  scale = textscale;
               }
            } else
               scale *= 3;
         }

         if (((scale > 0.0001) && (scale < 1)) || (lbl_tilt !== false)) {
            applied_scale = 1/scale;
            painter.scaleTextDrawing(applied_scale, label_g[0]);
         }
      }

      for (let lcnt = 0; lcnt < label_g.length; ++lcnt) {
         if (lcnt > 0) side = -side;

         let lastpos = 0;
         const fix_coord = this.vertical ? -labeloffset * side : labeloffset * side + ticksPlusMinus * tickSize;

         this.startTextDrawing(labelsFont, 'font', label_g[lcnt]);

         for (let nmajor = 0; nmajor < lbl_pos.length; ++nmajor) {
            let text = this.format(lbl_pos[nmajor], true);
            if (text === null) continue;

            const mod = this.findLabelModifier(axis, nmajor, lbl_pos);
            if (mod?.fTextSize === 0) continue;

            if (mod) any_modified = true;
            if (mod?.fLabText) text = mod.fLabText;

            const arg = { text, color: labelsFont.color, latex: 1, draw_g: label_g[lcnt], normal_side: (lcnt === 0) };
            let pos = Math.round(this.func(lbl_pos[nmajor]));

            if (mod?.fTextColor > 0) arg.color = this.getColor(mod.fTextColor);

            arg.gap_before = (nmajor > 0) ? Math.abs(Math.round(pos - this.func(lbl_pos[nmajor - 1]))) : 0;

            arg.gap_after = (nmajor < lbl_pos.length - 1) ? Math.abs(Math.round(this.func(lbl_pos[nmajor + 1]) - pos)) : 0;

            if (center_lbls) {
               const gap = arg.gap_after || arg.gap_before;
               pos = Math.round(pos - ((this.vertical !== this.reverse) ? 0.5 * gap : -0.5 * gap));
               if ((pos < -5) || (pos > (this.vertical ? h : w) + 5)) continue;
            }

            maxtextlen = Math.max(maxtextlen, text.length);

            if (this.vertical) {
               arg.x = fix_coord;
               arg.y = pos;
               arg.align = rotate_lbls ? ((side < 0) ? 23 : 20) : ((side < 0) ? 12 : 32);
            } else {
               arg.x = pos;
               arg.y = fix_coord;
               arg.align = rotate_lbls ? ((side < 0) ? 12 : 32) : ((side < 0) ? 20 : 23);
               if (this.log && !this.noexp && !this.vertical && arg.align === 23) {
                  arg.align = 21;
                  arg.y += labelsFont.size;
               } else if (arg.align % 10 === 3)
                  arg.y -= labelsFont.size*0.1; // font takes 10% more by top align
            }

            if (rotate_lbls)
               arg.rotate = 270;
            else if (mod && mod.fTextAngle !== -1)
               arg.rotate = -mod.fTextAngle;

            // only for major text drawing scale factor need to be checked
            if (lcnt === 0) arg.post_process = process_drawtext_ready;

            this.drawText(arg);

            // workaround for symlog where labels can be compressed to close
            if (this.symlog && lastpos && (pos !== lastpos) && ((this.vertical && !rotate_lbls) || (!this.vertical && rotate_lbls))) {
               const axis_step = Math.abs(pos - lastpos);
               textscale = Math.min(textscale, 1.1*axis_step/labelsFont.size);
            }

            lastpos = pos;
         }

         if (this.order) {
            let xoff = 0, yoff = 0;
            if (this.name === 'xaxis') {
               xoff = gStyle.fXAxisExpXOffset || 0;
               yoff = gStyle.fXAxisExpYOffset || 0;
            } else if (this.name === 'yaxis') {
               xoff = gStyle.fYAxisExpXOffset || 0;
               yoff = gStyle.fYAxisExpYOffset || 0;
            }

            if (xoff) xoff = Math.round(xoff * (this.getPadPainter()?.getPadWidth() ?? 0));
            if (yoff) yoff = Math.round(yoff * (this.getPadPainter()?.getPadHeight() ?? 0));

            this.drawText({ color: labelsFont.color,
                            x: xoff + (this.vertical ? side*5 : w+5),
                            y: yoff + (this.has_obstacle ? fix_coord : (this.vertical ? -3 : -3*side)),
                            align: this.vertical ? ((side < 0) ? 30 : 10) : ((this.has_obstacle ^ (side < 0)) ? 13 : 10),
                            latex: 1,
                            text: '#times' + this.formatExp(10, this.order),
                            draw_g: label_g[lcnt] });
         }
      }

      // first complete major labels drawing
      return this.finishTextDrawing(label_g[0], true).then(() => {
         if (label_g.length > 1) {
            // now complete drawing of second half with scaling if necessary
            if (applied_scale)
               this.scaleTextDrawing(applied_scale, label_g[1]);
            return this.finishTextDrawing(label_g[1], true);
         }
      }).then(() => {
         if (lbl_tilt) {
            label_g[0].selectAll('text').each(function() {
               const txt = d3_select(this), tr = txt.attr('transform');
               txt.attr('transform', `${tr} rotate(${tilt_angle})`).style('text-anchor', 'start');
            });
         }

         return max_textwidth;
      });
   }

   /** @summary Extract major draw attributes, which are also used in interactive operations
     * @private  */
   extractDrawAttributes(scalingSize, w, h) {
      const axis = this.getObject();
      let pp = this.getPadPainter();
      if (axis.$use_top_pad)
         pp = pp?.getPadPainter(); // workaround for ratio plot
      const pad_w = pp?.getPadWidth() || scalingSize || w/0.8, // use factor 0.8 as ratio between frame and pad size
            pad_h = pp?.getPadHeight() || scalingSize || h/0.8,
            // if no external scaling size use scaling as in TGaxis.cxx:1448 - NDC axis length is in the scaling factor
            tickScalingSize = scalingSize || (this.vertical ? h/pad_h*pad_w : w/pad_w*pad_h);

      let tickSize = 0, titleColor, titleFontId, offset;

      this.scalingSize = scalingSize || Math.max(Math.min(pad_w, pad_h), 10);

      if (this.is_gaxis) {
         const optionSize = axis.fChopt.indexOf('S') >= 0;
         this.optionUnlab = axis.fChopt.indexOf('U') >= 0;
         this.optionMinus = (axis.fChopt.indexOf('-') >= 0) || axis.TestBit(EAxisBits.kTickMinus);
         this.optionPlus = (axis.fChopt.indexOf('+') >= 0) || axis.TestBit(EAxisBits.kTickPlus);
         this.optionNoopt = (axis.fChopt.indexOf('N') >= 0);  // no ticks position optimization
         this.optionInt = (axis.fChopt.indexOf('I') >= 0);  // integer labels
         this.optionText = (axis.fChopt.indexOf('T') >= 0);  // text scaling?
         this.createAttLine({ attr: axis });
         tickSize = optionSize ? axis.fTickSize : 0.03;
         titleColor = this.getColor(axis.fTextColor);
         titleFontId = axis.fTextFont;
         offset = axis.fLabelOffset;
         if ((this.vertical && axis.fY1 > axis.fY2 && !this.optionMinus) || (!this.vertical && axis.fX1 > axis.fX2))
            offset = -offset;
      } else {
         this.optionUnlab = false;
         this.optionMinus = this.vertical ^ this.invert_side;
         this.optionPlus = !this.optionMinus;
         this.optionNoopt = false;  // no ticks position optimization
         this.optionInt = false;  // integer labels
         this.optionText = false;
         this.createAttLine({ color: axis.fAxisColor, width: 1, style: 1 });
         tickSize = axis.fTickLength;
         titleColor = this.getColor(axis.fTitleColor);
         titleFontId = axis.fTitleFont;
         offset = axis.fLabelOffset;
      }

      offset += (this.vertical ? 0.002 : 0.005);

      if (this.kind === kAxisLabels)
         this.optionText = true;

      this.optionNoexp = axis.TestBit(EAxisBits.kNoExponent);

      this.ticksSize = Math.round(tickSize * tickScalingSize);
      if (scalingSize && (this.ticksSize < 0))
         this.ticksSize = -this.ticksSize;

      if (this.maxTickSize && (this.ticksSize > this.maxTickSize)) this.ticksSize = this.maxTickSize;

      // now used only in 3D drawing
      this.ticksColor = this.lineatt.color;
      this.ticksWidth = this.lineatt.width;

      const k = this.optionText ? 0.66666 : 1; // set TGaxis.cxx, line 1504
      this.labelSize = Math.round((axis.fLabelSize < 1) ? k * axis.fLabelSize * this.scalingSize : k * axis.fLabelSize);
      this.labelsOffset = Math.round(offset * this.scalingSize);
      this.labelsFont = new FontHandler(axis.fLabelFont, this.labelSize, scalingSize);
      if ((this.labelSize <= 0) || (Math.abs(axis.fLabelOffset) > 1.1)) this.optionUnlab = true; // disable labels when size not specified
      this.labelsFont.setColor(this.getColor(axis.fLabelColor));

      this.fTitle = axis.fTitle;
      if (this.fTitle) {
         this.titleSize = (axis.fTitleSize >= 1) ? axis.fTitleSize : Math.round(axis.fTitleSize * this.scalingSize);
         this.titleFont = new FontHandler(titleFontId, this.titleSize, scalingSize);
         this.titleFont.setColor(titleColor);
         this.offsetScaling = (axis.fTitleSize >= 1) ? 1 : (this.vertical ? pad_w : pad_h) / this.scalingSize;
         this.titleOffset = axis.fTitleOffset;
         if (!this.titleOffset && this.name[0] === 'x')
            this.titleOffset = gStyle.fXaxis.fTitleOffset;
         this.titleOffset *= this.titleSize * this.offsetScaling;
         this.titleCenter = axis.TestBit(EAxisBits.kCenterTitle);
         this.titleOpposite = axis.TestBit(EAxisBits.kOppositeTitle);
      } else {
         delete this.titleSize;
         delete this.titleFont;
         delete this.offsetScaling;
         delete this.titleOffset;
         delete this.titleCenter;
         delete this.titleOpposite;
      }
   }

   /** @summary function draws TAxis or TGaxis object
     * @return {Promise} for drawing ready */
   async drawAxis(layer, w, h, transform, secondShift, disable_axis_drawing, max_text_width, calculate_position, frame_ygap) {
      const axis = this.getObject(),
            swap_side = this.swap_side || false;
      let axis_g = layer, draw_lines = true;

      // shift for second ticks set (if any)
      if (!secondShift)
         secondShift = 0;
      else if (this.invert_side)
         secondShift = -secondShift;

      this.extractDrawAttributes(undefined, w, h);

      if (this.is_gaxis)
         draw_lines = axis.fLineColor !== 0;

      // indicate that attributes created not for TAttLine, therefore cannot be updated as TAttLine in GED
      this.lineatt.not_standard = true;

      if (!this.is_gaxis || (this.name === 'zaxis')) {
         axis_g = layer.selectChild(`.${this.name}_container`);
         if (axis_g.empty())
            axis_g = layer.append('svg:g').attr('class', `${this.name}_container`);
         else
            axis_g.selectAll('*').remove();
      }

      let axis_lines = '';
      if (draw_lines) {
         axis_lines = 'M0,0' + (this.vertical ? `v${h}` : `h${w}`);
         if (secondShift)
            axis_lines += this.vertical ? `M${secondShift},0v${h}` : `M0,${secondShift}h${w}`;
      }

      axis_g.attr('transform', transform);

      let side = 1, ticksPlusMinus = 0;

      if (this.optionPlus && this.optionMinus) {
         side = 1; ticksPlusMinus = 1;
      } else if (this.optionMinus)
         side = (swap_side ^ this.vertical) ? 1 : -1;
      else if (this.optionPlus)
         side = (swap_side ^ this.vertical) ? -1 : 1;

      // first draw ticks

      const handle = this.createTicks(false, this.optionNoexp, this.optionNoopt, this.optionInt);

      axis_lines += this.produceTicksPath(handle, side, this.ticksSize, ticksPlusMinus, secondShift, draw_lines && !disable_axis_drawing && !this.disable_ticks);

      if (!disable_axis_drawing && axis_lines && !this.lineatt.empty()) {
         axis_g.append('svg:path')
               .attr('d', axis_lines)
               .call(this.lineatt.func);
      }

      let title_shift_x = 0, title_shift_y = 0, title_g = null, labelsMaxWidth = 0;
      // draw labels (sometime on both sides)
      const pr = (disable_axis_drawing || this.optionUnlab)
                ? Promise.resolve(0)
                : this.drawLabels(axis_g, axis, w, h, handle, side, this.labelsFont, this.labelsOffset, this.ticksSize, ticksPlusMinus, max_text_width, frame_ygap);

      return pr.then(maxw => {
         labelsMaxWidth = maxw;

         if (settings.Zooming && !this.disable_zooming && !this.isBatchMode()) {
            const labelSize = Math.max(this.labelsFont.size, 5),
                  r = axis_g.append('svg:rect')
                            .attr('class', 'axis_zoom')
                            .style('opacity', '0')
                            .style('cursor', 'crosshair');

            if (this.vertical) {
               const rw = (labelsMaxWidth || 2*labelSize) + 3;
               r.attr('x', (side > 0) ? -rw : 0).attr('y', 0)
                .attr('width', rw).attr('height', h);
            } else {
               r.attr('x', 0).attr('y', (side > 0) ? 0 : -labelSize - 3)
                .attr('width', w).attr('height', labelSize + 3);
            }
         }

         this.position = 0;

         if (calculate_position) {
            const node1 = axis_g.node(), node2 = this.getPadSvg().node();
            if (isFunc(node1?.getBoundingClientRect) && isFunc(node2?.getBoundingClientRect)) {
               const rect1 = node1.getBoundingClientRect(),
                  rect2 = node2.getBoundingClientRect();
               this.position = rect1.left - rect2.left; // use to control left position of Y scale
            }
            if (node1 && !node2)
               console.warn('Why PAD element missing when search for position');
         }

         if (!this.fTitle || disable_axis_drawing) return true;

         title_g = axis_g.append('svg:g').attr('class', 'axis_title');

         let title_offest_k = side;
         const rotate = axis.TestBit(EAxisBits.kRotateTitle) ? -1 : 1;

         this.startTextDrawing(this.titleFont, 'font', title_g);

         const xor_reverse = swap_side ^ this.titleOpposite, myxor = (rotate < 0) ^ xor_reverse;

         this.title_align = this.titleCenter ? 'middle' : (myxor ? 'begin' : 'end');

         if (this.vertical) {
            title_offest_k *= -1.6;

            title_shift_x = Math.round(title_offest_k * this.titleOffset);

            // if ((this.name === 'zaxis') && this.is_gaxis && ('getBoundingClientRect' in axis_g.node())) {
            //    // special handling for color palette labels - draw them always on right side
            //   const rect = axis_g.node().getBoundingClientRect();
            //   if (title_shift_x < rect.width - this.ticksSize)
            //      title_shift_x = Math.round(rect.width - this.ticksSize);
            // }

            title_shift_y = Math.round(this.titleCenter ? h/2 : (xor_reverse ? h : 0));

            this.drawText({ align: this.title_align+';middle',
                            rotate: (rotate < 0) ? 90 : 270,
                            text: this.fTitle, color: this.titleFont.color, draw_g: title_g });
         } else {
            title_offest_k *= 1.6;

            title_shift_x = Math.round(this.titleCenter ? w/2 : (xor_reverse ? 0 : w));
            title_shift_y = Math.round(title_offest_k * this.titleOffset);
            this.drawText({ align: this.title_align+';middle',
                            rotate: (rotate < 0) ? 180 : 0,
                            text: this.fTitle, color: this.titleFont.color, draw_g: title_g });
         }

         this.addTitleDrag(title_g, this.vertical, title_offest_k, swap_side, this.vertical ? h : w);

         return this.finishTextDrawing(title_g);
      }).then(() => {
         if (title_g) {
            if (!this.titleOffset && this.vertical && labelsMaxWidth)
               title_shift_x = Math.round(-side * (labelsMaxWidth + 0.7*this.offsetScaling*this.titleSize));
            makeTranslate(title_g, title_shift_x, title_shift_y);
            title_g.property('shift_x', title_shift_x)
                   .property('shift_y', title_shift_y);
         }

         return this;
      });
   }

} // class TAxisPainter

export { EAxisBits, chooseTimeFormat, AxisPainterMethods, TAxisPainter };
