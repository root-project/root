import { settings, isBatchMode } from '../core.mjs';
import { select as d3_select, pointer as d3_pointer,
         drag as d3_drag, timeFormat as d3_timeFormat,
         scaleTime as d3_scaleTime, scaleSymlog as d3_scaleSymlog,
         scaleLog as d3_scaleLog, scaleLinear as d3_scaleLinear } from '../d3.mjs';
import { AxisPainterMethods, chooseTimeFormat } from './TAxisPainter.mjs';
import { createMenu } from '../gui/menu.mjs';
import { addDragHandler } from './TFramePainter.mjs';
import { RObjectPainter } from '../base/RObjectPainter.mjs';


/**
 * @summary Axis painter for v7
 *
 * @private
 */

class RAxisPainter extends RObjectPainter {

   /** @summary constructor */
   constructor(dom, arg1, axis, cssprefix) {
      let drawable = cssprefix ? arg1.getObject() : arg1;
      super(dom, drawable, "", cssprefix ? arg1.csstype : "axis");
      Object.assign(this, AxisPainterMethods);
      this.initAxisPainter();

      this.axis = axis;
      if (cssprefix) { // drawing from the frame
         this.embedded = true; // indicate that painter embedded into the histo painter
         //this.csstype = arg1.csstype; // for the moment only via frame one can set axis attributes
         this.cssprefix = cssprefix;
         this.rstyle = arg1.rstyle;
      } else {
         // this.csstype = "axis";
         this.cssprefix = "axis_";
      }
   }

   /** @summary cleanup painter */
   cleanup() {
      delete this.axis;
      delete this.axis_g;
      this.cleanupAxisPainter();
      super.cleanup();
   }

   /** @summary Use in GED to identify kind of axis */
   getAxisType() { return "RAttrAxis"; }

   /** @summary Configure only base parameters, later same handle will be used for drawing  */
   configureZAxis(name, fp) {
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
   configureAxis(name, min, max, smin, smax, vertical, frame_range, axis_range, opts) {
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
         this.func = d3_scaleTime().domain([this.convertDate(smin), this.convertDate(smax)]);
      } else if (_symlog && (_symlog > 0)) {
         this.symlog = _symlog;
         this.func = d3_scaleSymlog().constant(_symlog).domain([smin,smax]);
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
         this.func = d3_scaleLog().base(this.logbase).domain([smin,smax]);
      } else {
         this.func = d3_scaleLinear().domain([smin,smax]);
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

      if (this.nticks > 20) this.nticks = 20;

      let gr_range = Math.abs(this.gr_range) || 100;

      if (this.kind == 'time') {
         if (this.nticks > 8) this.nticks = 8;

         let scale_range = this.scale_max - this.scale_min,
             tf1 = this.v7EvalAttr("timeFormat", ""),
             tf2 = chooseTimeFormat(scale_range / gr_range, false);

         if (!tf1 || (scale_range < 0.1 * (this.full_max - this.full_min)))
            tf1 = chooseTimeFormat(scale_range / this.nticks, true);

         this.tfunc1 = this.tfunc2 = d3_timeFormat(tf1);
         if (tf2!==tf1)
            this.tfunc2 = d3_timeFormat(tf2);

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
   createTicks(only_major_as_array, optionNoexp, optionNoopt, optionInt) {

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
   isCenteredLabels() {
      if (this.kind === 'labels') return true;
      if (this.kind === 'log') return false;
      return this.v7EvalAttr("labels_center", false);
   }

   /** @summary Used to move axis labels instead of zooming
     * @private */
   processLabelsMove(arg, pos) {
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
         label_g.attr('transform', `translate(${offset})`);
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
            this.changeAxisAttr(1, "labels_offset", this.labelsOffset / this.scalingSize);
         }
      }

      return true;
   }

   /** @summary Add interactive elements to draw axes title */
   addTitleDrag(title_g, side) {
      if (!settings.MoveResize || isBatchMode()) return;

      let drag_rect = null,
          acc_x, acc_y, new_x, new_y, alt_pos, curr_indx,
          drag_move = d3_drag().subject(Object);

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

               this.changeAxisAttr(0, "title_position", this.titlePos, "title_offset", this.titleOffset / this.scalingSize);

               drag_rect.remove();
               drag_rect = null;
            });

      title_g.style("cursor", "move").call(drag_move);
   }

   /** @summary checks if value inside graphical range, taking into account delta */
   isInsideGrRange(pos, delta1, delta2) {
      if (!delta1) delta1 = 0;
      if (delta2 === undefined) delta2 = delta1;
      if (this.gr_range < 0)
         return (pos >= this.gr_range - delta2) && (pos <= delta1);
      return (pos >= -delta1) && (pos <= this.gr_range + delta2);
   }

   /** @summary returns graphical range */
   getGrRange(delta) {
      if (!delta) delta = 0;
      if (this.gr_range < 0)
         return this.gr_range - delta;
      return this.gr_range + delta;
   }

   /** @summary If axis direction is negative coordinates direction */
   isReverseAxis() {
      return !this.vertical !== (this.getGrRange() > 0);
   }

   /** @summary Draw axis ticks
     * @private */
   drawMainLine(axis_g) {
      let ending = "";

      if (this.endingSize && this.endingStyle) {
         let sz = (this.gr_range > 0) ? -this.endingSize : this.endingSize,
             sz7 = Math.round(sz*0.7);
         sz = Math.round(sz);
         if (this.vertical)
            ending = `l${sz7},${sz}M0,${this.gr_range}l${-sz7},${sz}`;
         else
            ending = `l${sz},${sz7}M${this.gr_range},0l${sz},${-sz7}`;
      }

      axis_g.append("svg:path")
            .attr("d","M0,0" + (this.vertical ? "v" : "h") + this.gr_range + ending)
            .call(this.lineatt.func)
            .style('fill', ending ? "none" : null);
   }

   /** @summary Draw axis ticks
     * @returns {Promise} with gaps on left and right side
     * @private */
   drawTicks(axis_g, side, main_draw) {
      if (main_draw) this.ticks = [];

      this.handle.reset();

      let res = "", ticks_plusminus = 0;
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

         if (ticks_plusminus > 0) {
            h2 = -h1;
         } else if (side < 0) {
            h2 = -h1; h1 = 0;
         } else {
            h2 = 0;
         }

         res += this.vertical ? `M${h1},${grpos}H${h2}` : `M${grpos},${-h1}V${-h2}`;
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
   drawLabels(axis_g, side, gaps) {
      let center_lbls = this.isCenteredLabels(),
          rotate_lbls = this.labelsFont.angle != 0,
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

         if ((textscale > 0.0001) && (textscale < 0.8) && !painter.vertical && !rotate_lbls && (maxtextlen > 5) && (side > 0))
            lbls_tilt = true;

         let scale = textscale * (lbls_tilt ? 3 : 1);
         if ((scale > 0.0001) && (scale < 1))
            painter.scaleTextDrawing(1/scale, label_g);
      }

      let lastpos = 0,
          fix_offset = Math.round((this.vertical ? -side : side) * this.labelsOffset),
          fix_coord = Math.round((this.vertical ? -side : side) * gaps[side]);

      if (fix_offset)
         label_g.attr('transform', this.vertical ? `translate(${fix_offset})` : `translate(0,${fix_offset})`);

      label_g.property('fix_offset', fix_offset);

      this.startTextDrawing(this.labelsFont, 'font', label_g);

      for (let nmajor = 0; nmajor < lbl_pos.length; ++nmajor) {

         let lbl = this.format(lbl_pos[nmajor], true);
         if (lbl === null) continue;

         let pos = Math.round(this.func(lbl_pos[nmajor])),
             arg = { text: lbl, latex: 1, draw_g: label_g };

         arg.gap_before = (nmajor > 0) ? Math.abs(Math.round(pos - this.func(lbl_pos[nmajor-1]))) : 0,
         arg.gap_after = (nmajor < lbl_pos.length-1) ? Math.abs(Math.round(this.func(lbl_pos[nmajor+1])-pos)) : 0;

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
                         align: this.vertical ? ((side < 0) ? 30 : 10) : ((this.has_obstacle ^ (side < 0)) ? 13 : 10),
                         latex: 1,
                         text: '#times' + this.formatExp(10, this.order),
                         draw_g: label_g
         });

      return this.finishTextDrawing(label_g).then(() => {

        if (lbls_tilt)
           label_g.selectAll("text").each(function () {
               let txt = d3_select(this), tr = txt.attr("transform");
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
   addZoomingRect(axis_g, side, lgaps) {
      if (settings.Zooming && !this.disable_zooming && !isBatchMode()) {
         let sz = Math.max(lgaps[side], 10),
             d = this.vertical ? `v${this.gr_range}h${-side*sz}v${-this.gr_range}`
                               : `h${this.gr_range}v${side*sz}h${-this.gr_range}`;
         axis_g.append("svg:path")
               .attr("d",`M0,0${d}z`)
               .attr("class", "axis_zoom")
               .style("opacity", "0")
               .style("cursor", "crosshair");
      }
   }

   /** @summary Returns true if axis title is rotated */
   isTitleRotated() {
      return this.titleFont && (this.titleFont.angle != (this.vertical ? 270 : 0));
   }

   /** @summary Draw axis title */
   drawTitle(axis_g, side, lgaps) {
      if (!this.fTitle)
         return Promise.resolve(this);

      let title_g = axis_g.append("svg:g").attr("class", "axis_title"),
          title_shift_x = 0, title_shift_y = 0, title_basepos = 0;

      let rotated = this.isTitleRotated();

      this.startTextDrawing(this.titleFont, 'font', title_g);

      this.title_align = this.titleCenter ? "middle" : (this.titleOpposite ^ (this.isReverseAxis() || rotated) ? "begin" : "end");

      if (this.vertical) {
         title_basepos = Math.round(-side*(lgaps[side]));
         title_shift_x = title_basepos + Math.round(-side*this.titleOffset);
         title_shift_y = Math.round(this.titleCenter ? this.gr_range/2 : (this.titleOpposite ? 0 : this.gr_range));
         this.drawText({ align: [this.title_align, ((side < 0) ^ rotated ? 'top' : 'bottom')],
                         text: this.fTitle, draw_g: title_g });
      } else {
         title_shift_x = Math.round(this.titleCenter ? this.gr_range/2 : (this.titleOpposite ? 0 : this.gr_range));
         title_basepos = Math.round(side*lgaps[side]);
         title_shift_y = title_basepos + Math.round(side*this.titleOffset);
         this.drawText({ align: [this.title_align, ((side > 0) ^ rotated ? 'top' : 'bottom')],
                         text: this.fTitle, draw_g: title_g });
      }

      title_g.attr('transform', `translate(${title_shift_x},${title_shift_y})`)
             .property('basepos', title_basepos)
             .property('shift_x', title_shift_x)
             .property('shift_y', title_shift_y);

      this.addTitleDrag(title_g, side);

      return this.finishTextDrawing(title_g);
   }

   /** @summary Extract major draw attributes, which are also used in interactive operations
     * @private  */
   extractDrawAttributes(scalingSize) {
      let pp = this.getPadPainter(),
          rect = pp?.getPadRect() || { width: 10, height: 10 };

      this.scalingSize = scalingSize || (this.vertical ? rect.width : rect.height);

      this.createv7AttLine("line_");

      this.optionUnlab = this.v7EvalAttr("labels_hide", false);

      this.endingStyle = this.v7EvalAttr("ending_style", "");
      this.endingSize = Math.round(this.v7EvalLength("ending_size", this.scalingSize, this.endingStyle ? 0.02 : 0));
      this.startingSize = Math.round(this.v7EvalLength("starting_size", this.scalingSize, 0));
      this.ticksSize = this.v7EvalLength("ticks_size", this.scalingSize, 0.02);
      this.ticksSide = this.v7EvalAttr("ticks_side", "normal");
      this.ticksColor = this.v7EvalColor("ticks_color", "");
      this.ticksWidth = this.v7EvalAttr("ticks_width", 1);
      if (scalingSize && (this.ticksSize < 0))
         this.ticksSize = -this.ticksSize;

      this.fTitle = this.v7EvalAttr("title_value", "");

      if (this.fTitle) {
         this.titleFont = this.v7EvalFont("title", { size: 0.03 }, scalingSize || pp?.getPadHeight() || 10);
         this.titleFont.roundAngle(180, this.vertical ? 270 : 0);

         this.titleOffset = this.v7EvalLength("title_offset", this.scalingSize, 0);
         this.titlePos = this.v7EvalAttr("title_position", "right");
         this.titleCenter = (this.titlePos == "center");
         this.titleOpposite = (this.titlePos == "left");
      } else {
         delete this.titleFont;
         delete this.titleOffset;
         delete this.titlePos;
      }

      // TODO: remove old scaling factors for labels and ticks
      this.labelsFont = this.v7EvalFont("labels", { size: scalingSize ? 0.05 : 0.03 });
      this.labelsFont.roundAngle(180);
      if (this.labelsFont.angle) this.labelsFont.angle = 270;
      this.labelsOffset = this.v7EvalLength("labels_offset", this.scalingSize, 0);

      if (scalingSize) this.ticksSize = this.labelsFont.size*0.5; // old lego scaling factor

      if (this.maxTickSize && (this.ticksSize > this.maxTickSize))
         this.ticksSize = this.maxTickSize;
   }

   /** @summary Performs axis drawing
     * @returns {Promise} which resolved when drawing is completed */
   drawAxis(layer, transform, side) {
      let axis_g = layer;

      if (side === undefined) side = 1;

      if (!this.standalone) {
         axis_g = layer.select("." + this.name + "_container");
         if (axis_g.empty())
            axis_g = layer.append("svg:g").attr("class", this.name + "_container");
         else
            axis_g.selectAll("*").remove();
      }

      axis_g.attr("transform", transform || null);

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
   setAfterDrawHandler(handler) {
      this._afterDrawAgain = handler;
   }

   /** @summary Draw axis with the same settings, used by interactive changes */
   drawAxisAgain() {
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
   drawAxisOtherPlace(layer, transform, side, only_ticks) {
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
   zoomStandalone(min,max) {
      this.changeAxisAttr(1, "zoomMin", min, "zoomMax", max);
   }

   /** @summary Redraw axis, used in standalone mode for RAxisDrawable */
   redraw() {

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

      this.configureAxis("axis", min, max, smin, smax, drawable.fVertical, undefined, len, { reverse, labels: labels_len > 0 });

      this.createG();

      this.standalone = true;  // no need to clean axis container

      let promise = this.drawAxis(this.draw_g, `translate(${pos.x},${pos.y})`);

      if (isBatchMode()) return promise;

      return promise.then(() => {
         if (settings.ContextMenu)
            this.draw_g.on("contextmenu", evnt => {
               evnt.stopPropagation(); // disable main context menu
               evnt.preventDefault();  // disable browser context menu
               createMenu(evnt, this).then(menu => {
                 menu.add("header:RAxisDrawable");
                 menu.add("Unzoom", () => this.zoomStandalone());
                 this.fillAxisContextMenu(menu, "");
                 menu.show();
               });
            });

         addDragHandler(this, { x: pos.x, y: pos.y, width: this.vertical ? 10 : len, height: this.vertical ? len : 10,
                                only_move: true, redraw: d => this.positionChanged(d) });

         this.draw_g.on("dblclick", () => this.zoomStandalone());

         if (settings.ZoomWheel)
            this.draw_g.on("wheel", evnt => {
               evnt.stopPropagation();
               evnt.preventDefault();

               let pos = d3_pointer(evnt, this.draw_g.node()),
                   coord = this.vertical ? (1 - pos[1] / len) : pos[0] / len,
                   item = this.analyzeWheelEvent(evnt, coord);

               if (item.changed) this.zoomStandalone(item.min, item.max);
            });
      });
   }

   /** @summary Process interactive moving of the axis drawing */
   positionChanged(drag) {
      let drawable = this.getObject(),
          rect = this.getPadPainter().getPadRect(),
          xn = drag.x / rect.width,
          yn = 1 - drag.y / rect.height;

      drawable.fPos.fHoriz.fArr = [ xn ];
      drawable.fPos.fVert.fArr = [ yn ];

      this.submitCanvExec(`SetPos({${xn.toFixed(4)},${yn.toFixed(4)}})`);
   }

   /** @summary Change axis attribute, submit changes to server and redraw axis when specified
     * @desc Arguments as redraw_mode, name1, value1, name2, value2, ... */
   changeAxisAttr(redraw_mode) {
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
   changeAxisLog(arg) {
      if ((this.kind == "labels") || (this.kind == 'time')) return;
      if (arg === 'toggle') arg = this.log ? 0 : 10;

      arg = parseFloat(arg);
      if (Number.isFinite(arg)) this.changeAxisAttr(2, "log", arg, "symlog", 0);
   }

   /** @summary Provide context menu for axis */
   fillAxisContextMenu(menu, kind) {

      if (kind) menu.add("Unzoom", () => this.getFramePainter().unzoom(kind));

      menu.add("sub:Log scale", () => this.changeAxisLog('toggle'));
      menu.addchk(!this.log && !this.symlog, "linear", 0, arg => this.changeAxisLog(arg));
      menu.addchk(this.log && !this.symlog && (this.logbase==10), "log10", () => this.changeAxisLog(10));
      menu.addchk(this.log && !this.symlog && (this.logbase==2), "log2", () => this.changeAxisLog(2));
      menu.addchk(this.log && !this.symlog && Math.abs(this.logbase - Math.exp(1)) < 0.1, "ln", () => this.changeAxisLog(Math.exp(1)));
      menu.addchk(!this.log && this.symlog, "symlog", 0, () =>
         menu.input("set symlog constant", this.symlog || 10, "float").then(v => this.changeAxisAttr(2,"symlog", v)));
      menu.add("endsub:");

      menu.add("Divisions", () => menu.input("Set axis devisions", this.v7EvalAttr("ndiv", 508), "int").then(val => this.changeAxisAttr(2, "ndiv", val)));

      menu.add("sub:Ticks");
      menu.addRColorMenu("color", this.ticksColor, col => this.changeAxisAttr(1, "ticks_color", col));
      menu.addSizeMenu("size", 0, 0.05, 0.01, this.ticksSize/this.scalingSize, sz => this.changeAxisAttr(1, "ticks_size", sz));
      menu.addSelectMenu("side", ["normal", "invert", "both"], this.ticksSide, side => this.changeAxisAttr(1, "ticks_side", side));
      menu.add("endsub:");

      if (!this.optionUnlab && this.labelsFont) {
         menu.add("sub:Labels");
         menu.addSizeMenu("offset", -0.05, 0.05, 0.01, this.labelsOffset/this.scalingSize,
                         offset => this.changeAxisAttr(1, "labels_offset", offset));
         menu.addRAttrTextItems(this.labelsFont, { noangle: 1, noalign: 1 },
               change => this.changeAxisAttr(1, "labels_" + change.name, change.value));
         menu.addchk(this.labelsFont.angle, "rotate", res => this.changeAxisAttr(1, "labels_angle", res ? 180 : 0));
         menu.add("endsub:");
      }

      menu.add("sub:Title", () => menu.input("Enter axis title", this.fTitle).then(t => this.changeAxisAttr(1, "title_value", t)));

      if (this.fTitle) {
         menu.addSizeMenu("offset", -0.05, 0.05, 0.01, this.titleOffset/this.scalingSize,
                           offset => this.changeAxisAttr(1, "title_offset", offset));

         menu.addSelectMenu("position", ["left", "center", "right"], this.titlePos,
                            pos => this.changeAxisAttr(1, "title_position", pos));

         menu.addchk(this.isTitleRotated(), "rotate", flag => this.changeAxisAttr(1, "title_angle", flag ? 180 : 0));

         menu.addRAttrTextItems(this.titleFont, { noangle: 1, noalign: 1 }, change => this.changeAxisAttr(1, "title_" + change.name, change.value));
      }

      menu.add("endsub:");
      return true;
   }

} // class RAxisPainter

export { RAxisPainter };
