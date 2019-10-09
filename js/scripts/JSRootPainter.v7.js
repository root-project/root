/// @file JSRootPainter.v7.js
/// JavaScript ROOT graphics for ROOT v7 classes

(function( factory ) {
   if ( typeof define === "function" && define.amd ) {
      define( ['JSRootPainter', 'd3'], factory );
   } else if (typeof exports === 'object' && typeof module !== 'undefined') {
       factory(require("./JSRootCore.js"), require("d3"));
   } else {
      if (typeof d3 != 'object')
         throw new Error('This extension requires d3.js', 'JSRootPainter.v7.js');
      if (typeof JSROOT == 'undefined')
         throw new Error('JSROOT is not defined', 'JSRootPainter.v7.js');
      if (typeof JSROOT.Painter != 'object')
         throw new Error('JSROOT.Painter not defined', 'JSRootPainter.v7.js');
      factory(JSROOT, d3);
   }
} (function(JSROOT, d3) {

   "use strict";

   JSROOT.sources.push("v7");

   JSROOT.v7 = {}; // placeholder for v7-relevant code

   /** Evalue attributes using fAttr storage and configured RStyle */
   JSROOT.TObjectPainter.prototype.v7EvalAttr = function(name, dflt) {
      var obj = this.GetObject();
      if (!obj) return dflt;

      if (obj.fAttr && obj.fAttr.m) {
         var value = obj.fAttr.m[name];
         if (value) return value.v; // found value direct in attributes
      }

      if (this.rstyle && this.rstyle.fBlocks) {
         var blks = this.rstyle.fBlocks;
         for (var k=0;k<blks.length;++k) {
            var block = blks[k];

            var match = (this.csstype && (block.selector == this.csstype)) ||
                        (obj.fId && (block.selector == ("#" + obj.fId))) ||
                        (obj.fCssClass && (block.selector == ("." + obj.fCssClass)));

            if (match && block.map && block.map.m) {
               var value = block.map.m[name];
               if (value) return value.v;
            }
         }
      }

      return dflt;
   }

   /** Evalue RColor using attribute storage and configured RStyle */
   JSROOT.TObjectPainter.prototype.v7EvalColor = function(name, dflt) {
      var rgb = this.v7EvalAttr(name + "_rgb", "");

      if (rgb)
         return "#" + rgb + this.v7EvalAttr(name + "_a", "");

      return this.v7EvalAttr(name + "_name", "") || dflt;
   }


   function TAxisPainter(axis, embedded) {
      JSROOT.TObjectPainter.call(this, axis);

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

   TAxisPainter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   TAxisPainter.prototype.Cleanup = function() {

      this.ticks = [];
      this.func = null;
      delete this.format;

      JSROOT.TObjectPainter.prototype.Cleanup.call(this);
   }

   TAxisPainter.prototype.SetAxisConfig = function(name, kind, func, min, max, smin, smax) {
      this.name = name;
      this.kind = kind;
      this.func = func;

      this.full_min = min;
      this.full_max = max;
      this.scale_min = smin;
      this.scale_max = smax;
   }

   TAxisPainter.prototype.format10Exp = function(order, value) {
      var res = "";
      if (value) {
         value = Math.round(value/Math.pow(10,order));
         if ((value!=0) && (value!=1)) res = value.toString() + (JSROOT.gStyle.Latex ? "#times" : "x");
      }
      res += "10";
      if (JSROOT.gStyle.Latex > 1) return res + "^{" + order + "}";
      var superscript_symbols = {
            '0': '\u2070', '1': '\xB9', '2': '\xB2', '3': '\xB3', '4': '\u2074', '5': '\u2075',
            '6': '\u2076', '7': '\u2077', '8': '\u2078', '9': '\u2079', '-': '\u207B'
         };
      var str = order.toString();
      for (var n=0;n<str.length;++n) res += superscript_symbols[str[n]];
      return res;
   }

   TAxisPainter.prototype.CreateFormatFuncs = function() {

      var axis = this.GetObject(),
          is_gaxis = (axis && axis._typename === 'TGaxis');

      delete this.format;// remove formatting func

      var ndiv = 508;
      if (is_gaxis) ndiv = axis.fNdiv; else
      if (axis) ndiv = Math.max(axis.fNdivisions, 4);

      this.nticks = ndiv % 100;
      this.nticks2 = (ndiv % 10000 - this.nticks) / 100;
      this.nticks3 = Math.floor(ndiv/10000);

      if (axis && !is_gaxis && (this.nticks > 7)) this.nticks = 7;

      var gr_range = Math.abs(this.func.range()[1] - this.func.range()[0]);
      if (gr_range<=0) gr_range = 100;

      if (this.kind == 'time') {
         if (this.nticks > 8) this.nticks = 8;

         var scale_range = this.scale_max - this.scale_min,
             tf1 = JSROOT.Painter.getTimeFormat(axis),
             tf2 = JSROOT.Painter.chooseTimeFormat(scale_range / gr_range, false);

         if ((tf1.length == 0) || (scale_range < 0.1 * (this.full_max - this.full_min)))
            tf1 = JSROOT.Painter.chooseTimeFormat(scale_range / this.nticks, true);

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
            var val = parseFloat(d), rnd = Math.round(val);
            if (!asticks)
               return ((rnd === val) && (Math.abs(rnd)<1e9)) ? rnd.toString() : JSROOT.FFormat(val, notickexp_fmt || JSROOT.gStyle.fStatFormat);

            if (val <= 0) return null;
            var vlog = JSROOT.log10(val);
            if (this.moreloglabels || (Math.abs(vlog - Math.round(vlog))<0.001)) {
               if (!this.noexp && !notickexp_fmt)
                  return this.format10Exp(Math.floor(vlog+0.01), val);

               return (vlog<0) ? val.toFixed(Math.round(-vlog+0.5)) : val.toFixed(0);
            }
            return null;
         }
      } else if (this.kind == 'labels') {
         this.nticks = 50; // for text output allow max 50 names
         var scale_range = this.scale_max - this.scale_min;
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
            var indx = parseFloat(d);
            if (!this.regular_labels)
               indx = (indx - this.axis.fXmin)/(this.axis.fXmax - this.axis.fXmin) * this.axis.fNbins;
            indx = Math.round(indx);
            if ((indx<0) || (indx>=this.axis.fNbins)) return null;
            for (var i = 0; i < this.axis.fLabels.arr.length; ++i) {
               var tstr = this.axis.fLabels.arr[i];
               if (tstr.fUniqueID === indx+1) return tstr.fString;
            }
            return null;
         }
      } else {

         this.order = 0;
         this.ndig = 0;

         this.format = function(d, asticks, fmt) {
            var val = parseFloat(d);
            if (asticks && this.order) val = val / Math.pow(10, this.order);

            if (val === Math.round(val))
               return (Math.abs(val)<1e9) ? val.toFixed(0) : val.toExponential(4);

            if (asticks) return (this.ndig>10) ? val.toExponential(this.ndig-11) : val.toFixed(this.ndig);

            return JSROOT.FFormat(val, fmt || JSROOT.gStyle.fStatFormat);
         }
      }
   }

   TAxisPainter.prototype.ProduceTicks = function(ndiv, ndiv2) {
      if (!this.noticksopt) return this.func.ticks(ndiv * (ndiv2 || 1));

      if (ndiv2) ndiv = (ndiv-1) * ndiv2;
      var dom = this.func.domain(), ticks = [];
      for (var n=0;n<=ndiv;++n)
         ticks.push((dom[0]*(ndiv-n) + dom[1]*n)/ndiv);
      return ticks;
   }

   TAxisPainter.prototype.CreateTicks = function(only_major_as_array, optionNoexp, optionNoopt, optionInt) {
      // function used to create array with minor/middle/major ticks

      if (optionNoopt && this.nticks && (this.kind == "normal")) this.noticksopt = true;

      var handle = { nminor: 0, nmiddle: 0, nmajor: 0, func: this.func };

      handle.minor = handle.middle = handle.major = this.ProduceTicks(this.nticks);

      if (only_major_as_array) {
         var res = handle.major, delta = (this.scale_max - this.scale_min)*1e-5;
         if (res[0] > this.scale_min + delta) res.unshift(this.scale_min);
         if (res[res.length-1] < this.scale_max - delta) res.push(this.scale_max);
         return res;
      }

      if ((this.kind == 'labels') && !this.regular_labels) {
         handle.lbl_pos = [];
         for (var n=0;n<this.axis.fNbins;++n) {
            var x = this.axis.fXmin + n / this.axis.fNbins * (this.axis.fXmax - this.axis.fXmin);
            if ((x >= this.scale_min) && (x < this.scale_max)) handle.lbl_pos.push(x);
         }
      }

      if (this.nticks2 > 1) {
         handle.minor = handle.middle = this.ProduceTicks(handle.major.length, this.nticks2);

         var gr_range = Math.abs(this.func.range()[1] - this.func.range()[0]);

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

         var maxorder = 0, minorder = 0, exclorder3 = false;

         if (!optionNoexp) {
            var maxtick = Math.max(Math.abs(handle.major[0]),Math.abs(handle.major[handle.major.length-1])),
                mintick = Math.min(Math.abs(handle.major[0]),Math.abs(handle.major[handle.major.length-1])),
                ord1 = (maxtick > 0) ? Math.round(JSROOT.log10(maxtick)/3)*3 : 0,
                ord2 = (mintick > 0) ? Math.round(JSROOT.log10(mintick)/3)*3 : 0;

             exclorder3 = (maxtick < 2e4); // do not show 10^3 for values below 20000

             if (maxtick || mintick) {
                maxorder = Math.max(ord1,ord2) + 3;
                minorder = Math.min(ord1,ord2) - 3;
             }
         }

         // now try to find best combination of order and ndig for labels

         var bestorder = 0, bestndig = this.ndig, bestlen = 1e10;

         for (var order = minorder; order <= maxorder; order+=3) {
            if (exclorder3 && (order===3)) continue;
            this.order = order;
            this.ndig = 0;
            var lbls = [], indx = 0, totallen = 0;
            while (indx<handle.major.length) {
               var lbl = this.format(handle.major[indx], true);
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

   TAxisPainter.prototype.IsCenterLabels = function() {
      if (this.kind === 'labels') return true;
      if (this.kind === 'log') return false;
      var axis = this.GetObject();
      return axis && axis.TestBit(JSROOT.EAxisBits.kCenterLabels);
   }

   TAxisPainter.prototype.AddTitleDrag = function(title_g, vertical, offset_k, reverse, axis_length) {
      if (!JSROOT.gStyle.MoveResize) return;

      var pthis = this,  drag_rect = null, prefix = "", drag_move,
          acc_x, acc_y, new_x, new_y, sign_0, center_0, alt_pos;
      if (JSROOT._test_d3_ === 3) {
         prefix = "drag";
         drag_move = d3.behavior.drag().origin(Object);
      } else {
         drag_move = d3.drag().subject(Object);
      }

      drag_move
         .on(prefix+"start",  function() {

            d3.event.sourceEvent.preventDefault();
            d3.event.sourceEvent.stopPropagation();

            var box = title_g.node().getBBox(), // check that elements visible, request precise value
                axis = pthis.GetObject();

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
          }).on("drag", function() {
               if (!drag_rect) return;

               d3.event.sourceEvent.preventDefault();
               d3.event.sourceEvent.stopPropagation();

               acc_x += d3.event.dx;
               acc_y += d3.event.dy;

               var set_x = title_g.property('shift_x'),
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

          }).on(prefix+"end", function() {
               if (!drag_rect) return;

               d3.event.sourceEvent.preventDefault();
               d3.event.sourceEvent.stopPropagation();

               title_g.property('shift_x', new_x)
                      .property('shift_y', new_y);

               var axis = pthis.GetObject();

               axis.fTitleOffset = (vertical ? new_x : new_y) / offset_k;
               if ((vertical ? new_y : new_x) === alt_pos) axis.InvertBit(JSROOT.EAxisBits.kCenterTitle);

               drag_rect.remove();
               drag_rect = null;
            });

      title_g.style("cursor", "move").call(drag_move);
   }

   TAxisPainter.prototype.DrawAxis = function(vertical, layer, w, h, transform, reverse, second_shift, disable_axis_drawing, max_text_width) {
      // function draws  TAxis or TGaxis object

      var axis = this.GetObject(), chOpt = "",
          is_gaxis = (axis && axis._typename === 'TGaxis'),
          axis_g = layer, tickSize = 0.03,
          scaling_size = 100, draw_lines = true,
          pad_w = this.pad_width() || 10,
          pad_h = this.pad_height() || 10;

      this.vertical = vertical;

      function myXor(a,b) { return ( a && !b ) || (!a && b); }

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
         chOpt = myXor(vertical, this.invert_side) ? "-S" : "+S";
         tickSize = axis.fTickLength;
         scaling_size = (vertical ? pad_w : pad_h);
      }

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

      var side = 1, ticks_plusminus = 0,
          text_scaling_size = Math.min(pad_w, pad_h),
          optionPlus = (chOpt.indexOf("+")>=0),
          optionMinus = (chOpt.indexOf("-")>=0),
          optionSize = (chOpt.indexOf("S")>=0),
          optionY = (chOpt.indexOf("Y")>=0),
          optionUp = (chOpt.indexOf("0")>=0),
          optionDown = (chOpt.indexOf("O")>=0),
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

      var res = "", res2 = "", lastpos = 0, lasth = 0;

      // first draw ticks

      this.ticks = [];

      var handle = this.CreateTicks(false, optionNoexp, optionNoopt, optionInt);

      while (handle.next(true)) {

         var h1 = Math.round(tickSize/4), h2 = 0;

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

      var labelsize = Math.round( (axis.fLabelSize < 1) ? axis.fLabelSize * text_scaling_size : axis.fLabelSize);
      if ((labelsize <= 0) || (Math.abs(axis.fLabelOffset) > 1.1)) optionUnlab = true; // disable labels when size not specified

      // draw labels (on both sides, when needed)
      if (!disable_axis_drawing && !optionUnlab) {

         var label_color = this.get_color(axis.fLabelColor),
             labeloffset = Math.round(axis.fLabelOffset*text_scaling_size /*+ 0.5*labelsize*/),
             center_lbls = this.IsCenterLabels(),
             rotate_lbls = axis.TestBit(JSROOT.EAxisBits.kLabelsVert),
             textscale = 1, maxtextlen = 0, lbls_tilt = false, labelfont = null,
             label_g = [ axis_g.append("svg:g").attr("class","axis_labels") ],
             lbl_pos = handle.lbl_pos || handle.major;

         if (this.lbls_both_sides)
            label_g.push(axis_g.append("svg:g").attr("class","axis_labels").attr("transform", vertical ? "translate(" + w + ",0)" : "translate(0," + (-h) + ")"));

         for (var lcnt = 0; lcnt < label_g.length; ++lcnt) {

            if (lcnt > 0) side = -side;

            var lastpos = 0,
                fix_coord = vertical ? -labeloffset*side : (labeloffset+2)*side + ticks_plusminus*tickSize;

            labelfont = JSROOT.Painter.getFontDetails(axis.fLabelFont, labelsize);

            this.StartTextDrawing(labelfont, 'font', label_g[lcnt]);

            for (var nmajor=0;nmajor<lbl_pos.length;++nmajor) {

               var lbl = this.format(lbl_pos[nmajor], true);
               if (lbl === null) continue;

               var pos = Math.round(this.func(lbl_pos[nmajor])),
                   gap_before = (nmajor>0) ? Math.abs(Math.round(pos - this.func(lbl_pos[nmajor-1]))) : 0,
                   gap_after = (nmajor<lbl_pos.length-1) ? Math.abs(Math.round(this.func(lbl_pos[nmajor+1])-pos)) : 0;

               if (center_lbls) {
                  var gap = gap_after || gap_before;
                  pos = Math.round(pos - (vertical ? 0.5*gap : -0.5*gap));
                  if ((pos < -5) || (pos > (vertical ? h : w) + 5)) continue;
               }

               var arg = { text: lbl, color: label_color, latex: 1, draw_g: label_g[lcnt] };

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

               var textwidth = this.DrawText(arg);

               if (textwidth && ((!vertical && !rotate_lbls) || (vertical && rotate_lbls)) && (this.kind != 'log')) {
                  var maxwidth = gap_before*0.45 + gap_after*0.45;
                  if (!gap_before) maxwidth = 0.9*gap_after; else
                  if (!gap_after) maxwidth = 0.9*gap_before;
                  textscale = Math.min(textscale, maxwidth / textwidth);
               } else if (vertical && max_text_width && !lcnt && (max_text_width - labeloffset > 20) && (textwidth > max_text_width - labeloffset)) {
                  textscale = Math.min(textscale, (max_text_width - labeloffset) / textwidth);
               }

               if (lastpos && (pos!=lastpos) && ((vertical && !rotate_lbls) || (!vertical && rotate_lbls))) {
                  var axis_step = Math.abs(pos-lastpos);
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

         if ((textscale > 0.01) && (textscale < 0.7) && !vertical && !rotate_lbls && (maxtextlen > 5) && !this.lbls_both_sides) {
            lbls_tilt = true;
            textscale *= 3;
         }

         for (var lcnt = 0; lcnt < label_g.length; ++lcnt) {
            if ((textscale > 0.01) && (textscale < 1))
               this.TextScaleFactor(1/textscale, label_g[lcnt]);

            this.FinishTextDrawing(label_g[lcnt]);
            if (lbls_tilt)
               label_g[lcnt].selectAll("text").each(function() {
                  var txt = d3.select(this), tr = txt.attr("transform");
                  txt.attr("transform", tr + " rotate(25)").style("text-anchor", "start");
               });
         }

         if (label_g.length > 1) side = -side;

         if (labelfont) labelsize = labelfont.size; // use real font size
      }

      if (JSROOT.gStyle.Zooming && !this.disable_zooming) {
         var r =  axis_g.append("svg:rect")
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
         var title_g = axis_g.append("svg:g").attr("class", "axis_title"),
             title_fontsize = (axis.fTitleSize >= 1) ? axis.fTitleSize : Math.round(axis.fTitleSize * text_scaling_size),
             title_offest_k = 1.6*(axis.fTitleSize<1 ? axis.fTitleSize : axis.fTitleSize/(this.pad_height("") || 10)),
             center = axis.TestBit(JSROOT.EAxisBits.kCenterTitle),
             rotate = axis.TestBit(JSROOT.EAxisBits.kRotateTitle) ? -1 : 1,
             title_color = this.get_color(axis.fTitleColor),
             shift_x = 0, shift_y = 0;

         this.StartTextDrawing(axis.fTitleFont, title_fontsize, title_g);

         var myxor = ((rotate<0) && !reverse) || ((rotate>=0) && reverse);

         if (vertical) {
            title_offest_k *= -side*pad_w;

            shift_x = Math.round(title_offest_k*axis.fTitleOffset);

            if ((this.name == "zaxis") && is_gaxis && ('getBoundingClientRect' in axis_g.node())) {
               // special handling for color palette labels - draw them always on right side
               var rect = axis_g.node().getBoundingClientRect();
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

         var axis_rect = null;
         if (vertical && (axis.fTitleOffset == 0) && ('getBoundingClientRect' in axis_g.node()))
            axis_rect = axis_g.node().getBoundingClientRect();

         this.FinishTextDrawing(title_g, function() {
            if (axis_rect) {
               var title_rect = title_g.node().getBoundingClientRect();
               shift_x = (side>0) ? Math.round(axis_rect.left - title_rect.right - title_fontsize*0.3) :
                                    Math.round(axis_rect.right - title_rect.left + title_fontsize*0.3);
            }

            title_g.attr('transform', 'translate(' + shift_x + ',' + shift_y +  ')')
                   .property('shift_x', shift_x)
                   .property('shift_y', shift_y);
         });


         this.AddTitleDrag(title_g, vertical, title_offest_k, reverse, vertical ? h : w);
      }

      this.position = 0;

      if ('getBoundingClientRect' in axis_g.node()) {
         var rect1 = axis_g.node().getBoundingClientRect(),
             rect2 = this.svg_pad().node().getBoundingClientRect();

         this.position = rect1.left - rect2.left; // use to control left position of Y scale
      }
   }

   TAxisPainter.prototype.Redraw = function() {

      var gaxis = this.GetObject(),
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
         this.toffset = JSROOT.Painter.getTimeOffset(gaxis);
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
            var d = y1; y1 = y2; y2 = d;
            h = -h; reverse = true;
            func.range([0,h]);
         }
      } else {
         if (w > 0) {
            func.range([0,w]);
         } else {
            var d = x1; x1 = x2; x2 = d;
            w = -w; reverse = true;
            func.range([w,0]);
         }
      }

      this.SetAxisConfig(vertical ? "yaxis" : "xaxis", kind, func, min, max, min, max);

      this.CreateG();

      this.DrawAxis(vertical, this.draw_g, w, h, "translate(" + x1 + "," + y2 +")", reverse);
   }

   // ==========================================================================================


   function TFramePainter(tframe) {
      JSROOT.TooltipHandler.call(this, tframe);
      this.mode3d = false;
      this.shrink_frame_left = 0.;
      this.x_kind = 'normal'; // 'normal', 'log', 'time', 'labels'
      this.y_kind = 'normal'; // 'normal', 'log', 'time', 'labels'
      this.xmin = this.xmax = 0; // no scale specified, wait for objects drawing
      this.ymin = this.ymax = 0; // no scale specified, wait for objects drawing
      this.axes_drawn = false;
      this.keys_handler = null;
      this.mode3d = false;
   }

   TFramePainter.prototype = Object.create(JSROOT.TooltipHandler.prototype);

   TFramePainter.prototype.frame_painter = function() {
      return this;
   }

   /** @summary Set active flag for frame - can block some events
    * @private */
   TFramePainter.prototype.SetActive = function(on) {
      // do nothing here - key handler is handled differently
   }

   TFramePainter.prototype.GetTipName = function(append) {
      var res = JSROOT.TooltipHandler.prototype.GetTipName.call(this) || "TFrame";
      if (append) res+=append;
      return res;
   }

   TFramePainter.prototype.Shrink = function(shrink_left, shrink_right) {
      this.fX1NDC += shrink_left;
      this.fX2NDC -= shrink_right;
   }

   TFramePainter.prototype.SetLastEventPos = function(pnt) {
      // set position of last context menu event, can be
      this.fLastEventPnt = pnt;
   }

   TFramePainter.prototype.GetLastEventPos = function() {
      // return position of last event
      return this.fLastEventPnt;
   }

   TFramePainter.prototype.UpdateAttributes = function(force) {
      var tframe = this.GetObject();

      if ((this.fX1NDC === undefined) || (force && !this.modified_NDC)) {
         JSROOT.extend(this, JSROOT.gStyle.FrameNDC);

         if (tframe && tframe.fPos && tframe.fSize) {
            this.fX1NDC = tframe.fPos.fHoriz.fNormal.fVal;
            this.fX2NDC = this.fX1NDC + tframe.fSize.fHoriz.fNormal.fVal;
            this.fY1NDC = tframe.fPos.fVert.fNormal.fVal;
            this.fY2NDC = this.fY1NDC + tframe.fSize.fVert.fNormal.fVal;
         }
      }

      if (this.fillatt === undefined) {
         this.createAttFill({ pattern: 1001, color: 0 });

         // TODO: provide real fill color
         this.fillatt.SetSolidColor('white');
      }

      this.createAttLine({ color: 'black' });
   }

   TFramePainter.prototype.ProjectAitoff2xy = function(l, b) {
      var DegToRad = Math.PI/180,
          alpha2 = (l/2)*DegToRad,
          delta  = b*DegToRad,
          r2     = Math.sqrt(2),
          f      = 2*r2/Math.PI,
          cdec   = Math.cos(delta),
          denom  = Math.sqrt(1. + cdec*Math.cos(alpha2)),
          res = {
             x: cdec*Math.sin(alpha2)*2.*r2/denom/f/DegToRad,
             y: Math.sin(delta)*r2/denom/f/DegToRad
          };
      //  x *= -1.; // for a skymap swap left<->right
      return res;
   }

   TFramePainter.prototype.ProjectMercator2xy = function(l, b) {
      var aid = Math.tan((Math.PI/2 + b/180*Math.PI)/2);
      return { x: l, y: Math.log(aid) };
   }

   TFramePainter.prototype.ProjectSinusoidal2xy = function(l, b) {
      return { x: l*Math.cos(b/180*Math.PI), y: b };
   }

   TFramePainter.prototype.ProjectParabolic2xy = function(l, b) {
      return {
         x: l*(2.*Math.cos(2*b/180*Math.PI/3) - 1),
         y: 180*Math.sin(b/180*Math.PI/3)
      };
   }

   TFramePainter.prototype.RecalculateRange = function(Proj) {
      // not yet used, could be useful in the future

      if (!Proj) return;

      var pnts = []; // all extremes which used to find
      if (Proj == 1) {
         // TODO : check x range not lower than -180 and not higher than 180
         pnts.push(this.ProjectAitoff2xy(this.scale_xmin, this.scale_ymin));
         pnts.push(this.ProjectAitoff2xy(this.scale_xmin, this.scale_ymax));
         pnts.push(this.ProjectAitoff2xy(this.scale_xmax, this.scale_ymax));
         pnts.push(this.ProjectAitoff2xy(this.scale_xmax, this.scale_ymin));
         if (this.scale_ymin<0 && this.scale_ymax>0) {
            // there is an  'equator', check its range in the plot..
            pnts.push(this.ProjectAitoff2xy(this.scale_xmin*0.9999, 0));
            pnts.push(this.ProjectAitoff2xy(this.scale_xmax*0.9999, 0));
         }
         if (this.scale_xmin<0 && this.scale_xmax>0) {
            pnts.push(this.ProjectAitoff2xy(0, this.scale_ymin));
            pnts.push(this.ProjectAitoff2xy(0, this.scale_ymax));
         }
      } else if (Proj == 2) {
         if (this.scale_ymin <= -90 || this.scale_ymax >=90) {
            console.warn("Mercator Projection", "Latitude out of range", this.scale_ymin, this.scale_ymax);
            this.options.Proj = 0;
            return;
         }
         pnts.push(this.ProjectMercator2xy(this.scale_xmin, this.scale_ymin));
         pnts.push(this.ProjectMercator2xy(this.scale_xmax, this.scale_ymax));

      } else if (Proj == 3) {
         pnts.push(this.ProjectSinusoidal2xy(this.scale_xmin, this.scale_ymin));
         pnts.push(this.ProjectSinusoidal2xy(this.scale_xmin, this.scale_ymax));
         pnts.push(this.ProjectSinusoidal2xy(this.scale_xmax, this.scale_ymax));
         pnts.push(this.ProjectSinusoidal2xy(this.scale_xmax, this.scale_ymin));
         if (this.scale_ymin<0 && this.scale_ymax>0) {
            pnts.push(this.ProjectSinusoidal2xy(this.scale_xmin, 0));
            pnts.push(this.ProjectSinusoidal2xy(this.scale_xmax, 0));
         }
         if (this.scale_xmin<0 && this.scale_xmax>0) {
            pnts.push(this.ProjectSinusoidal2xy(0, this.scale_ymin));
            pnts.push(this.ProjectSinusoidal2xy(0, this.scale_ymax));
         }
      } else if (Proj == 4) {
         pnts.push(this.ProjectParabolic2xy(this.scale_xmin, this.scale_ymin));
         pnts.push(this.ProjectParabolic2xy(this.scale_xmin, this.scale_ymax));
         pnts.push(this.ProjectParabolic2xy(this.scale_xmax, this.scale_ymax));
         pnts.push(this.ProjectParabolic2xy(this.scale_xmax, this.scale_ymin));
         if (this.scale_ymin<0 && this.scale_ymax>0) {
            pnts.push(this.ProjectParabolic2xy(this.scale_xmin, 0));
            pnts.push(this.ProjectParabolic2xy(this.scale_xmax, 0));
         }
         if (this.scale_xmin<0 && this.scale_xmax>0) {
            pnts.push(this.ProjectParabolic2xy(0, this.scale_ymin));
            pnts.push(this.ProjectParabolic2xy(0, this.scale_ymax));
         }
      }

      this.original_xmin = this.scale_xmin;
      this.original_xmax = this.scale_xmax;
      this.original_ymin = this.scale_ymin;
      this.original_ymax = this.scale_ymax;

      this.scale_xmin = this.scale_xmax = pnts[0].x;
      this.scale_ymin = this.scale_ymax = pnts[0].y;

      for (var n=1;n<pnts.length;++n) {
         this.scale_xmin = Math.min(this.scale_xmin, pnts[n].x);
         this.scale_xmax = Math.max(this.scale_xmax, pnts[n].x);
         this.scale_ymin = Math.min(this.scale_ymin, pnts[n].y);
         this.scale_ymax = Math.max(this.scale_ymax, pnts[n].y);
      }
   }

   TFramePainter.prototype.DrawGrids = function() {
      // grid can only be drawn by first painter

      var layer = this.svg_frame().select(".grid_layer");

      layer.selectAll(".xgrid").remove();
      layer.selectAll(".ygrid").remove();

      var h = this.frame_height(),
          w = this.frame_width(),
          grid, grid_style = JSROOT.gStyle.fGridStyle,
          grid_color = (JSROOT.gStyle.fGridColor > 0) ? this.get_color(JSROOT.gStyle.fGridColor) : "black";

      if ((grid_style < 0) || (grid_style >= JSROOT.Painter.root_line_styles.length)) grid_style = 11;

      // add a grid on x axis, if the option is set
      if (this.x_handle) {
         grid = "";
         for (var n=0;n<this.x_handle.ticks.length;++n)
            if (this.swap_xy)
               grid += "M0,"+this.x_handle.ticks[n]+"h"+w;
            else
               grid += "M"+this.x_handle.ticks[n]+",0v"+h;

         if (grid.length > 0)
          layer.append("svg:path")
               .attr("class", "xgrid")
               .attr("d", grid)
               .style('stroke',grid_color).style("stroke-width",JSROOT.gStyle.fGridWidth)
               .style("stroke-dasharray", JSROOT.Painter.root_line_styles[grid_style]);
      }

      // add a grid on y axis, if the option is set
      if (this.y_handle) {
         grid = "";
         for (var n=0;n<this.y_handle.ticks.length;++n)
            if (this.swap_xy)
               grid += "M"+this.y_handle.ticks[n]+",0v"+h;
            else
               grid += "M0,"+this.y_handle.ticks[n]+"h"+w;

         if (grid.length > 0)
          layer.append("svg:path")
               .attr("class", "ygrid")
               .attr("d", grid)
               .style('stroke',grid_color).style("stroke-width",JSROOT.gStyle.fGridWidth)
               .style("stroke-dasharray", JSROOT.Painter.root_line_styles[grid_style]);
      }
   }

   TFramePainter.prototype.AxisAsText = function(axis, value) {
      if (axis == "x") {
         if (this.x_kind == 'time')
            value = this.ConvertX(value);
         if (this.x_handle && ('format' in this.x_handle))
            return this.x_handle.format(value, false, JSROOT.gStyle.XValuesFormat);
      } else if (axis == "y") {
         if (this.y_kind == 'time')
            value = this.ConvertY(value);
         if (this.y_handle && ('format' in this.y_handle))
            return this.y_handle.format(value, false, JSROOT.gStyle.YValuesFormat);
      } else {
         if (this.z_handle && ('format' in this.z_handle))
            return this.z_handle.format(value, false, JSROOT.gStyle.ZValuesFormat);
      }

      return value.toPrecision(4);
   }

   TFramePainter.prototype.SetAxesRanges = function(xmin, xmax, ymin, ymax) {
      if (this.axes_drawn) return;

      if ((this.xmin == this.xmax) && (xmin!==xmax)) {
         this.xmin = xmin;
         this.xmax = xmax;
      }
      if ((this.ymin == this.ymax) && (ymin!==ymax)) {
         this.ymin = ymin;
         this.ymax = ymax;
      }
   }

   TFramePainter.prototype.DrawAxes = function(shrink_forbidden) {
      // axes can be drawn only for main histogram

      if (this.axes_drawn) return true;

      if ((this.xmin==this.xmax) || (this.ymin==this.ymax)) return false;

      this.CleanupAxes();
      this.CleanXY();

      this.CreateXY();

      var layer = this.svg_frame().select(".axis_layer"),
          w = this.frame_width(),
          h = this.frame_height(),
          axisx = JSROOT.Create("TAxis"), // temporary object for different attributes
          axisy = JSROOT.Create("TAxis");

      this.x_handle = new JSROOT.TAxisPainter(axisx, true);
      this.x_handle.SetDivId(this.divid, -1);
      this.x_handle.pad_name = this.pad_name;

      this.x_handle.SetAxisConfig("xaxis",
                                  (this.logx && (this.x_kind !== "time")) ? "log" : this.x_kind,
                                  this.x, this.xmin, this.xmax, this.scale_xmin, this.scale_xmax);
      this.x_handle.invert_side = false;
      this.x_handle.lbls_both_sides = false;
      this.x_handle.has_obstacle = false;

      this.y_handle = new JSROOT.TAxisPainter(axisy, true);
      this.y_handle.SetDivId(this.divid, -1);
      this.y_handle.pad_name = this.pad_name;

      this.y_handle.SetAxisConfig("yaxis",
                                  (this.logy && this.y_kind !== "time") ? "log" : this.y_kind,
                                  this.y, this.ymin, this.ymax, this.scale_ymin, this.scale_ymax);
      this.y_handle.invert_side = false; // ((this.options.AxisPos % 10) === 1) || (pad.fTicky > 1);
      this.y_handle.lbls_both_sides = false;

      var draw_horiz = this.swap_xy ? this.y_handle : this.x_handle,
          draw_vertical = this.swap_xy ? this.x_handle : this.y_handle,
          disable_axis_draw = false, show_second_ticks = false;

      if (!disable_axis_draw) {
         var pp = this.pad_painter();
         if (pp && pp._fast_drawing) disable_axis_draw = true;
      }

      if (!disable_axis_draw) {
         draw_horiz.DrawAxis(false, layer, w, h,
                             draw_horiz.invert_side ? undefined : "translate(0," + h + ")",
                             false, show_second_ticks ? -h : 0, disable_axis_draw);

         draw_vertical.DrawAxis(true, layer, w, h,
                                draw_vertical.invert_side ? "translate(" + w + ",0)" : undefined,
                                false, show_second_ticks ? w : 0, disable_axis_draw,
                             draw_vertical.invert_side ? 0 : this.frame_x());

         this.DrawGrids();
      }

      if (!shrink_forbidden && JSROOT.gStyle.CanAdjustFrame && !disable_axis_draw) {

         var shrink = 0., ypos = draw_vertical.position;

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
            this.DrawAxes(true);
         }
      }

      this.axes_drawn = true;

      return true;
   }

   TFramePainter.prototype.SizeChanged = function() {
      // function called at the end of resize of frame
      // One should apply changes to the pad

    /*  var pad = this.root_pad();

      if (pad) {
         pad.fLeftMargin = this.fX1NDC;
         pad.fRightMargin = 1 - this.fX2NDC;
         pad.fBottomMargin = this.fY1NDC;
         pad.fTopMargin = 1 - this.fY2NDC;
         this.SetRootPadRange(pad);
      }
      */

      this.RedrawPad();
   }

   TFramePainter.prototype.CleanXY = function() {
      // remove all kinds of X/Y function for axes transformation
      delete this.x; delete this.grx;
      delete this.ConvertX; delete this.RevertX;
      delete this.y; delete this.gry;
      delete this.ConvertY; delete this.RevertY;
      delete this.z; delete this.grz;
   }

   TFramePainter.prototype.CleanupAxes = function() {
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

   /** Removes all drawn elements of the frame @private */
   TFramePainter.prototype.CleanFrameDrawings = function() {
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

   TFramePainter.prototype.Cleanup = function() {

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

      JSROOT.TooltipHandler.prototype.Cleanup.call(this);
   }

   TFramePainter.prototype.Redraw = function() {

      var pp = this.pad_painter();
      if (pp) pp.frame_painter_ref = this;

      if (this.mode3d) return;

      // first update all attributes from objects
      this.UpdateAttributes();

      var width = this.pad_width(),
          height = this.pad_height(),
          lm = Math.round(width * this.fX1NDC),
          w = Math.round(width * (this.fX2NDC - this.fX1NDC)),
          tm = Math.round(height * (1 - this.fY2NDC)),
          h = Math.round(height * (this.fY2NDC - this.fY1NDC)),
          rotate = false, fixpos = false;

      if (pp && pp.options) {
         if (pp.options.RotateFrame) rotate = true;
         if (pp.options.FixFrame) fixpos = true;
      }

      // this is svg:g object - container for every other items belonging to frame
      this.draw_g = this.svg_layer("primitives_layer").select(".root_frame");

      var top_rect, main_svg;

      if (this.draw_g.empty()) {

         var layer = this.svg_layer("primitives_layer");

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

      var trans = "translate(" + lm + "," + tm + ")";
      if (rotate) {
         trans += " rotate(-90) " + "translate(" + -h + ",0)";
         var d = w; w = h; h = d;
      }

      this._frame_x = lm;
      this._frame_y = tm;
      this._frame_width = w;
      this._frame_height = h;

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

      // var tooltip_rect = this.draw_g.select(".interactive_rect");
      // if (JSROOT.BatchMode) return tooltip_rect.remove();
      if (JSROOT.BatchMode) return;

      this.draw_g.attr("x", lm)
                 .attr("y", tm)
                 .attr("width", w)
                 .attr("height", h);

      if (!rotate && !fixpos)
         this.AddDrag({ obj: this, only_resize: true, minwidth: 20, minheight: 20,
                        redraw: this.SizeChanged.bind(this) });

      var tooltip_rect = main_svg;
      tooltip_rect.style("pointer-events","visibleFill")
                  .property('handlers_set', 0);

      //if (tooltip_rect.empty())
      //   tooltip_rect =
      //      this.draw_g
      //          .append("rect")
      //          .attr("class","interactive_rect")
      //          .style('opacity',0)
      //          .style('fill',"none")
      //          .style("pointer-events","visibleFill")
      //          .property('handlers_set', 0);

      var handlers_set = (pp && pp._fast_drawing) ? 0 : 1;

      if (tooltip_rect.property('handlers_set') != handlers_set) {
         var close_handler = handlers_set ? this.ProcessTooltipEvent.bind(this, null) : null,
              mouse_handler = handlers_set ? this.ProcessTooltipEvent.bind(this, { handler: true, touch: false }) : null;

         tooltip_rect.property('handlers_set', handlers_set)
                     .on('mouseenter', mouse_handler)
                     .on('mousemove', mouse_handler)
                     .on('mouseleave', close_handler);

         if (JSROOT.touches) {
            var touch_handler = handlers_set ? this.ProcessTooltipEvent.bind(this, { handler: true, touch: true }) : null;

            tooltip_rect.on("touchstart", touch_handler)
                        .on("touchmove", touch_handler)
                        .on("touchend", close_handler)
                        .on("touchcancel", close_handler);
         }
      }

      tooltip_rect.attr("x", 0)
                  .attr("y", 0)
                  .attr("width", w)
                  .attr("height", h);

      var hintsg = this.hints_layer().select(".objects_hints");
      // if tooltips were visible before, try to reconstruct them after short timeout
      if (!hintsg.empty() && this.IsTooltipAllowed() && (hintsg.property("hints_pad") == this.pad_name))
         setTimeout(this.ProcessTooltipEvent.bind(this, hintsg.property('last_point')), 10);
   }

   /** Returns frame rectangle plus extra info for hint display */
   TFramePainter.prototype.GetFrameRect = function() {
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

   /** Function called when frame is clicked and object selection can be performed
     * such event can be used to select objects */
   TFramePainter.prototype.ProcessFrameClick = function(pnt, dblckick) {

      var pp = this.pad_painter();
      if (!pp) return;

      pnt.painters = true; // provide painters reference in the hints
      pnt.disabled = true; // do not invoke graphics

      // collect tooltips from pad painter - it has list of all drawn objects
      var hints = pp.GetTooltips(pnt), exact = null;
      for (var k=0; (k<hints.length) && !exact; ++k)
         if (hints[k] && hints[k].exact) exact = hints[k];
      //if (exact) console.log('Click exact', pnt, exact.painter.GetTipName());
      //      else console.log('Click frame', pnt);

      var res;

      if (exact) {
         var handler = dblckick ? this._dblclick_handler : this._click_handler;
         if (handler) res = handler(exact.user_info, pnt);
      }

      if (!dblckick)
         pp.SelectObjectPainter(exact ? exact.painter : this,
               { x: pnt.x + (this._frame_x || 0),  y: pnt.y + (this._frame_y || 0) });

      return res;
   }

   TFramePainter.prototype.ConfigureUserClickHandler = function(handler) {
      this._click_handler = handler && (typeof handler == 'function') ? handler : null;
   }

   TFramePainter.prototype.ConfigureUserDblclickHandler = function(handler) {
      this._dblclick_handler = handler && (typeof handler == 'function') ? handler : null;
   }

   TFramePainter.prototype.Zoom = function(xmin, xmax, ymin, ymax, zmin, zmax) {
      // function can be used for zooming into specified range
      // if both limits for each axis 0 (like xmin==xmax==0), axis will be unzoomed

      // disable zooming when axis conversion is enabled
      if (this.options && this.options.Proj) return false;

      if (xmin==="x") { xmin = xmax; xmax = ymin; ymin = undefined; } else
      if (xmin==="y") { ymax = ymin; ymin = xmax; xmin = xmax = undefined; } else
      if (xmin==="z") { zmin = xmax; zmax = ymin; xmin = xmax = ymin = undefined; }

      var zoom_x = (xmin !== xmax), zoom_y = (ymin !== ymax), zoom_z = (zmin !== zmax),
          unzoom_x = false, unzoom_y = false, unzoom_z = false;

      if (zoom_x) {
         var cnt = 0;
         if (xmin <= this.xmin) { xmin = this.xmin; cnt++; }
         if (xmax >= this.xmax) { xmax = this.xmax; cnt++; }
         if (cnt === 2) { zoom_x = false; unzoom_x = true; }
      } else {
         unzoom_x = (xmin === xmax) && (xmin === 0);
      }

      if (zoom_y) {
         var cnt = 0;
         if (ymin <= this.ymin) { ymin = this.ymin; cnt++; }
         if (ymax >= this.ymax) { ymax = this.ymax; cnt++; }
         if (cnt === 2) { zoom_y = false; unzoom_y = true; }
      } else {
         unzoom_y = (ymin === ymax) && (ymin === 0);
      }

      if (zoom_z) {
         var cnt = 0;
         // if (this.logz && this.ymin_nz && this.Dimension()===2) main_zmin = 0.3*this.ymin_nz;
         if (zmin <= this.zmin) { zmin = this.zmin; cnt++; }
         if (zmax >= this.zmax) { zmax = this.zmax; cnt++; }
         if (cnt === 2) { zoom_z = false; unzoom_z = true; }
      } else {
         unzoom_z = (zmin === zmax) && (zmin === 0);
      }

      var changed = false, fp = this;

      // first process zooming (if any)
      if (zoom_x || zoom_y || zoom_z)
         this.ForEachPainter(function(obj) {
            if (zoom_x && obj.CanZoomIn("x", xmin, xmax)) {
               fp.zoom_xmin = xmin;
               fp.zoom_xmax = xmax;
               changed = true;
               zoom_x = false;
            }
            if (zoom_y && obj.CanZoomIn("y", ymin, ymax)) {
               fp.zoom_ymin = ymin;
               fp.zoom_ymax = ymax;
               changed = true;
               zoom_y = false;
            }
            if (zoom_z && obj.CanZoomIn("z", zmin, zmax)) {
               fp.zoom_zmin = zmin;
               fp.zoom_zmax = zmax;
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
      }

      if (changed) this.RedrawPad();

      return changed;
   }

   TFramePainter.prototype.IsAxisZoomed = function(axis) {
      return this['zoom_'+axis+'min'] !== this['zoom_'+axis+'max'];
   }

   TFramePainter.prototype.Unzoom = function(dox, doy, doz) {
      if (typeof dox === 'undefined') { dox = true; doy = true; doz = true; } else
      if (typeof dox === 'string') { doz = dox.indexOf("z")>=0; doy = dox.indexOf("y")>=0; dox = dox.indexOf("x")>=0; }

      var last = this.zoom_changed_interactive;

      if (dox || doy || doz) this.zoom_changed_interactive = 2;

      var changed = this.Zoom(dox ? 0 : undefined, dox ? 0 : undefined,
                              doy ? 0 : undefined, doy ? 0 : undefined,
                              doz ? 0 : undefined, doz ? 0 : undefined);

      // if unzooming has no effect, decrease counter
      if ((dox || doy || doz) && !changed)
         this.zoom_changed_interactive = (!isNaN(last) && (last>0)) ? last - 1 : 0;

      return changed;

   }

   TFramePainter.prototype.clearInteractiveElements = function() {
      JSROOT.Painter.closeMenu();
      if (this.zoom_rect) { this.zoom_rect.remove(); this.zoom_rect = null; }
      this.zoom_kind = 0;

      // enable tooltip in frame painter
      this.SwitchTooltip(true);
   }

   TFramePainter.prototype.mouseDoubleClick = function() {
      d3.event.preventDefault();
      var m = d3.mouse(this.svg_frame().node());
      this.clearInteractiveElements();

      var valid_x = (m[0] >= 0) && (m[0] <= this.frame_width()),
          valid_y = (m[1] >= 0) && (m[1] <= this.frame_height());

      if (valid_x && valid_y && this._dblclick_handler)
         if (this.ProcessFrameClick({ x: m[0], y: m[1] }, true)) return;

      var kind = "xyz";
      if (!valid_x) kind = this.swap_xy ? "x" : "y"; else
      if (!valid_y) kind = this.swap_xy ? "y" : "x";
      if (this.Unzoom(kind)) return;
   }

   TFramePainter.prototype.startRectSel = function() {
      // ignore when touch selection is activated

      if (this.zoom_kind > 100) return;

      // ignore all events from non-left button
      if ((d3.event.which || d3.event.button) !== 1) return;

      d3.event.preventDefault();

      var pos = d3.mouse(this.svg_frame().node());

      this.clearInteractiveElements();
      this.zoom_origin = pos;

      var w = this.frame_width(), h = this.frame_height();

      this.zoom_curr = [ Math.max(0, Math.min(w, this.zoom_origin[0])),
                         Math.max(0, Math.min(h, this.zoom_origin[1])) ];

      if ((this.zoom_origin[0] < 0) || (this.zoom_origin[0] > w)) {
         this.zoom_kind = 3; // only y
         this.zoom_origin[0] = 0;
         this.zoom_origin[1] = this.zoom_curr[1];
         this.zoom_curr[0] = w;
         this.zoom_curr[1] += 1;
      } else if ((this.zoom_origin[1] < 0) || (this.zoom_origin[1] > h)) {
         this.zoom_kind = 2; // only x
         this.zoom_origin[0] = this.zoom_curr[0];
         this.zoom_origin[1] = 0;
         this.zoom_curr[0] += 1;
         this.zoom_curr[1] = h;
      } else {
         this.zoom_kind = 1; // x and y
         this.zoom_origin[0] = this.zoom_curr[0];
         this.zoom_origin[1] = this.zoom_curr[1];
      }

      d3.select(window).on("mousemove.zoomRect", this.moveRectSel.bind(this))
                       .on("mouseup.zoomRect", this.endRectSel.bind(this), true);

      this.zoom_rect = null;

      // disable tooltips in frame painter
      this.SwitchTooltip(false);

      d3.event.stopPropagation();
   }

   TFramePainter.prototype.moveRectSel = function() {

      if ((this.zoom_kind == 0) || (this.zoom_kind > 100)) return;

      d3.event.preventDefault();
      var m = d3.mouse(this.svg_frame().node());

      m[0] = Math.max(0, Math.min(this.frame_width(), m[0]));
      m[1] = Math.max(0, Math.min(this.frame_height(), m[1]));

      switch (this.zoom_kind) {
         case 1: this.zoom_curr[0] = m[0]; this.zoom_curr[1] = m[1]; break;
         case 2: this.zoom_curr[0] = m[0]; break;
         case 3: this.zoom_curr[1] = m[1]; break;
      }

      if (this.zoom_rect===null)
         this.zoom_rect = this.svg_frame()
                              .append("rect")
                              .attr("class", "zoom")
                              .attr("pointer-events","none");

      this.zoom_rect.attr("x", Math.min(this.zoom_origin[0], this.zoom_curr[0]))
                    .attr("y", Math.min(this.zoom_origin[1], this.zoom_curr[1]))
                    .attr("width", Math.abs(this.zoom_curr[0] - this.zoom_origin[0]))
                    .attr("height", Math.abs(this.zoom_curr[1] - this.zoom_origin[1]));
   }

   TFramePainter.prototype.endRectSel = function() {
      if ((this.zoom_kind == 0) || (this.zoom_kind > 100)) return;

      d3.event.preventDefault();

      d3.select(window).on("mousemove.zoomRect", null)
                       .on("mouseup.zoomRect", null);

      var m = d3.mouse(this.svg_frame().node()), changed = [true, true];
      m[0] = Math.max(0, Math.min(this.frame_width(), m[0]));
      m[1] = Math.max(0, Math.min(this.frame_height(), m[1]));

      switch (this.zoom_kind) {
         case 1: this.zoom_curr[0] = m[0]; this.zoom_curr[1] = m[1]; break;
         case 2: this.zoom_curr[0] = m[0]; changed[1] = false; break; // only X
         case 3: this.zoom_curr[1] = m[1]; changed[0] = false; break; // only Y
      }

      var xmin, xmax, ymin, ymax, isany = false,
          idx = this.swap_xy ? 1 : 0, idy = 1 - idx;

      if (changed[idx] && (Math.abs(this.zoom_curr[idx] - this.zoom_origin[idx]) > 10)) {
         xmin = Math.min(this.RevertX(this.zoom_origin[idx]), this.RevertX(this.zoom_curr[idx]));
         xmax = Math.max(this.RevertX(this.zoom_origin[idx]), this.RevertX(this.zoom_curr[idx]));
         isany = true;
      }

      if (changed[idy] && (Math.abs(this.zoom_curr[idy] - this.zoom_origin[idy]) > 10)) {
         ymin = Math.min(this.RevertY(this.zoom_origin[idy]), this.RevertY(this.zoom_curr[idy]));
         ymax = Math.max(this.RevertY(this.zoom_origin[idy]), this.RevertY(this.zoom_curr[idy]));
         isany = true;
      }

      var kind = this.zoom_kind, pnt = (kind===1) ? { x: this.zoom_origin[0], y: this.zoom_origin[1] } : null;

      this.clearInteractiveElements();

      if (isany) {
         this.zoom_changed_interactive = 2;
         this.Zoom(xmin, xmax, ymin, ymax);
      } else {
         switch (kind) {
            case 1:
               var fp = this.frame_painter();
               if (fp) fp.ProcessFrameClick(pnt);
               break;
            case 2:
               var pp = this.pad_painter();
               if (pp) pp.SelectObjectPainter(this.x_handle);
               break;
            case 3:
               var pp = this.pad_painter();
               if (pp) pp.SelectObjectPainter(this.y_handle);
               break;
         }
      }

      this.zoom_kind = 0;
   }

   TFramePainter.prototype.startTouchZoom = function() {
      // in case when zooming was started, block any other kind of events
      if (this.zoom_kind != 0) {
         d3.event.preventDefault();
         d3.event.stopPropagation();
         return;
      }

      var arr = d3.touches(this.svg_frame().node());
      this.touch_cnt+=1;

      // normally double-touch will be handled
      // touch with single click used for context menu
      if (arr.length == 1) {
         // this is touch with single element

         var now = new Date(), diff = now.getTime() - this.last_touch.getTime();
         this.last_touch = now;

         if ((diff < 300) && this.zoom_curr
              && (Math.abs(this.zoom_curr[0] - arr[0][0]) < 30)
              && (Math.abs(this.zoom_curr[1] - arr[0][1]) < 30)) {

            d3.event.preventDefault();
            d3.event.stopPropagation();

            this.clearInteractiveElements();
            this.Unzoom("xyz");

            this.last_touch = new Date(0);

            this.svg_frame().on("touchcancel", null)
                            .on("touchend", null, true);
         } else
         if (JSROOT.gStyle.ContextMenu) {
            this.zoom_curr = arr[0];
            this.svg_frame().on("touchcancel", this.endTouchSel.bind(this))
                            .on("touchend", this.endTouchSel.bind(this));
            d3.event.preventDefault();
            d3.event.stopPropagation();
         }
      }

      if ((arr.length != 2) || !JSROOT.gStyle.Zooming || !JSROOT.gStyle.ZoomTouch) return;

      d3.event.preventDefault();
      d3.event.stopPropagation();

      this.clearInteractiveElements();

      this.svg_frame().on("touchcancel", null)
                      .on("touchend", null);

      var pnt1 = arr[0], pnt2 = arr[1], w = this.frame_width(), h = this.frame_height();

      this.zoom_curr = [ Math.min(pnt1[0], pnt2[0]), Math.min(pnt1[1], pnt2[1]) ];
      this.zoom_origin = [ Math.max(pnt1[0], pnt2[0]), Math.max(pnt1[1], pnt2[1]) ];

      if ((this.zoom_curr[0] < 0) || (this.zoom_curr[0] > w)) {
         this.zoom_kind = 103; // only y
         this.zoom_curr[0] = 0;
         this.zoom_origin[0] = w;
      } else if ((this.zoom_origin[1] > h) || (this.zoom_origin[1] < 0)) {
         this.zoom_kind = 102; // only x
         this.zoom_curr[1] = 0;
         this.zoom_origin[1] = h;
      } else {
         this.zoom_kind = 101; // x and y
      }

      this.SwitchTooltip(false);

      this.zoom_rect = this.svg_frame().append("rect")
            .attr("class", "zoom")
            .attr("id", "zoomRect")
            .attr("x", this.zoom_curr[0])
            .attr("y", this.zoom_curr[1])
            .attr("width", this.zoom_origin[0] - this.zoom_curr[0])
            .attr("height", this.zoom_origin[1] - this.zoom_curr[1]);

      d3.select(window).on("touchmove.zoomRect", this.moveTouchSel.bind(this))
                       .on("touchcancel.zoomRect", this.endTouchSel.bind(this))
                       .on("touchend.zoomRect", this.endTouchSel.bind(this));
   }

   TFramePainter.prototype.moveTouchSel = function() {
      if (this.zoom_kind < 100) return;

      d3.event.preventDefault();

      var arr = d3.touches(this.svg_frame().node());

      if (arr.length != 2)
         return this.clearInteractiveElements();

      var pnt1 = arr[0], pnt2 = arr[1];

      if (this.zoom_kind != 103) {
         this.zoom_curr[0] = Math.min(pnt1[0], pnt2[0]);
         this.zoom_origin[0] = Math.max(pnt1[0], pnt2[0]);
      }
      if (this.zoom_kind != 102) {
         this.zoom_curr[1] = Math.min(pnt1[1], pnt2[1]);
         this.zoom_origin[1] = Math.max(pnt1[1], pnt2[1]);
      }

      this.zoom_rect.attr("x", this.zoom_curr[0])
                     .attr("y", this.zoom_curr[1])
                     .attr("width", this.zoom_origin[0] - this.zoom_curr[0])
                     .attr("height", this.zoom_origin[1] - this.zoom_curr[1]);

      if ((this.zoom_origin[0] - this.zoom_curr[0] > 10)
           || (this.zoom_origin[1] - this.zoom_curr[1] > 10))
         this.SwitchTooltip(false);

      d3.event.stopPropagation();
   }

   TFramePainter.prototype.endTouchSel = function() {

      this.svg_frame().on("touchcancel", null)
                      .on("touchend", null);

      if (this.zoom_kind === 0) {
         // special case - single touch can ends up with context menu

         d3.event.preventDefault();

         var now = new Date();

         var diff = now.getTime() - this.last_touch.getTime();

         if ((diff > 500) && (diff<2000) && !this.frame_painter().IsTooltipShown()) {
            this.ShowContextMenu('main', { clientX: this.zoom_curr[0], clientY: this.zoom_curr[1] });
            this.last_touch = new Date(0);
         } else {
            this.clearInteractiveElements();
         }
      }

      if (this.zoom_kind < 100) return;

      d3.event.preventDefault();
      d3.select(window).on("touchmove.zoomRect", null)
                       .on("touchend.zoomRect", null)
                       .on("touchcancel.zoomRect", null);

      var xmin, xmax, ymin, ymax, isany = false,
          xid = this.swap_xy ? 1 : 0, yid = 1 - xid,
          changed = [true, true];
      if (this.zoom_kind === 102) changed[1] = false;
      if (this.zoom_kind === 103) changed[0] = false;

      if (changed[xid] && (Math.abs(this.zoom_curr[xid] - this.zoom_origin[xid]) > 10)) {
         xmin = Math.min(this.RevertX(this.zoom_origin[xid]), this.RevertX(this.zoom_curr[xid]));
         xmax = Math.max(this.RevertX(this.zoom_origin[xid]), this.RevertX(this.zoom_curr[xid]));
         isany = true;
      }

      if (changed[yid] && (Math.abs(this.zoom_curr[yid] - this.zoom_origin[yid]) > 10)) {
         ymin = Math.min(this.RevertY(this.zoom_origin[yid]), this.RevertY(this.zoom_curr[yid]));
         ymax = Math.max(this.RevertY(this.zoom_origin[yid]), this.RevertY(this.zoom_curr[yid]));
         isany = true;
      }

      this.clearInteractiveElements();
      this.last_touch = new Date(0);

      if (isany) {
         this.zoom_changed_interactive = 2;
         this.Zoom(xmin, xmax, ymin, ymax);
      }

      d3.event.stopPropagation();
   }

   TFramePainter.prototype.ShowContextMenu = function(kind, evnt, obj) {
      // ignore context menu when touches zooming is ongoing
      if (('zoom_kind' in this) && (this.zoom_kind > 100)) return;

      // this is for debug purposes only, when context menu is where, close is and show normal menu
      //if (!evnt && !kind && document.getElementById('root_ctx_menu')) {
      //   var elem = document.getElementById('root_ctx_menu');
      //   elem.parentNode.removeChild(elem);
      //   return;
      //}

      var menu_painter = this, frame_corner = false, fp = this; // object used to show context menu

      if (!evnt) {
         d3.event.preventDefault();
         d3.event.stopPropagation(); // disable main context menu
         evnt = d3.event;

         if (kind === undefined) {
            var ms = d3.mouse(this.svg_frame().node()),
                tch = d3.touches(this.svg_frame().node()),
                pp = this.pad_painter(),
                pnt = null, sel = null;

            if (tch.length === 1) pnt = { x: tch[0][0], y: tch[0][1], touch: true }; else
            if (ms.length === 2) pnt = { x: ms[0], y: ms[1], touch: false };

            if ((pnt !== null) && (pp !== null)) {
               pnt.painters = true; // assign painter for every tooltip
               var hints = pp.GetTooltips(pnt), bestdist = 1000;
               for (var n=0;n<hints.length;++n)
                  if (hints[n] && hints[n].menu) {
                     var dist = ('menu_dist' in hints[n]) ? hints[n].menu_dist : 7;
                     if (dist < bestdist) { sel = hints[n].painter; bestdist = dist; }
                  }
            }

            if (sel!==null) menu_painter = sel;

            if (pnt!==null) frame_corner = (pnt.x>0) && (pnt.x<20) && (pnt.y>0) && (pnt.y<20);

            this.SetLastEventPos(pnt);
         }
      }

      // one need to copy event, while after call back event may be changed
      menu_painter.ctx_menu_evnt = evnt;

      JSROOT.Painter.createMenu(menu_painter, function(menu) {
         var domenu = menu.painter.FillContextMenu(menu, kind, obj);

         // fill frame menu by default - or append frame elements when activated in the frame corner
         if (fp && (!domenu || (frame_corner && (kind!=="frame") && (fp!=menu.painter))))
            domenu = fp.FillContextMenu(menu);

         if (domenu)
            menu.painter.FillObjectExecMenu(menu, kind, function() {
                // suppress any running zooming
                menu.painter.SwitchTooltip(false);
                menu.show(menu.painter.ctx_menu_evnt, menu.painter.SwitchTooltip.bind(menu.painter, true) );
            });

      });  // end menu creation
   }

   TFramePainter.prototype.FillContextMenu = function(menu, kind, obj) {

      // when fill and show context menu, remove all zooming
      this.clearInteractiveElements();

      if ((kind=="x") || (kind=="y")) {
         var faxis = null;
         //this.histo.fXaxis;
         //if (kind=="y") faxis = this.histo.fYaxis;  else
         //if (kind=="z") faxis = obj ? obj : this.histo.fZaxis;
         menu.add("header: " + kind.toUpperCase() + " axis");
         menu.add("Unzoom", this.Unzoom.bind(this, kind));

         if (this[kind+"_kind"] == "normal")
           menu.addchk(this["log"+kind], "SetLog"+kind, this.ToggleLog.bind(this, kind) );

         // if ((kind === "z") && this.options.Zscale)
         //   if (this.FillPaletteMenu) this.FillPaletteMenu(menu);

         if (faxis) {
            menu.addchk(faxis.TestBit(JSROOT.EAxisBits.kMoreLogLabels), "More log",
                  function() { faxis.InvertBit(JSROOT.EAxisBits.kMoreLogLabels); this.RedrawPad(); });
            menu.addchk(faxis.TestBit(JSROOT.EAxisBits.kNoExponent), "No exponent",
                  function() { faxis.InvertBit(JSROOT.EAxisBits.kNoExponent); this.RedrawPad(); });
            menu.add("sub:Labels");
            menu.addchk(faxis.TestBit(JSROOT.EAxisBits.kCenterLabels), "Center",
                  function() { faxis.InvertBit(JSROOT.EAxisBits.kCenterLabels); this.RedrawPad(); });
            menu.addchk(faxis.TestBit(JSROOT.EAxisBits.kLabelsVert), "Rotate",
                  function() { faxis.InvertBit(JSROOT.EAxisBits.kLabelsVert); this.RedrawPad(); });
            this.AddColorMenuEntry(menu, "Color", faxis.fLabelColor,
                  function(arg) { faxis.fLabelColor = parseInt(arg); this.RedrawPad(); });
            this.AddSizeMenuEntry(menu,"Offset", 0, 0.1, 0.01, faxis.fLabelOffset,
                  function(arg) { faxis.fLabelOffset = parseFloat(arg); this.RedrawPad(); } );
            this.AddSizeMenuEntry(menu,"Size", 0.02, 0.11, 0.01, faxis.fLabelSize,
                  function(arg) { faxis.fLabelSize = parseFloat(arg); this.RedrawPad(); } );
            menu.add("endsub:");
            menu.add("sub:Title");
            menu.add("SetTitle", function() {
               var t = prompt("Enter axis title", faxis.fTitle);
               if (t!==null) { faxis.fTitle = t; this.RedrawPad(); }
            });
            menu.addchk(faxis.TestBit(JSROOT.EAxisBits.kCenterTitle), "Center",
                  function() { faxis.InvertBit(JSROOT.EAxisBits.kCenterTitle); this.RedrawPad(); });
            menu.addchk(faxis.TestBit(JSROOT.EAxisBits.kRotateTitle), "Rotate",
                  function() { faxis.InvertBit(JSROOT.EAxisBits.kRotateTitle); this.RedrawPad(); });
            this.AddColorMenuEntry(menu, "Color", faxis.fTitleColor,
                  function(arg) { faxis.fTitleColor = parseInt(arg); this.RedrawPad(); });
            this.AddSizeMenuEntry(menu,"Offset", 0, 3, 0.2, faxis.fTitleOffset,
                                  function(arg) { faxis.fTitleOffset = parseFloat(arg); this.RedrawPad(); } );
            this.AddSizeMenuEntry(menu,"Size", 0.02, 0.11, 0.01, faxis.fTitleSize,
                  function(arg) { faxis.fTitleSize = parseFloat(arg); this.RedrawPad(); } );
            menu.add("endsub:");
            menu.add("sub:Ticks");
            this.AddColorMenuEntry(menu, "Color", faxis.fAxisColor,
                        function(arg) { faxis.fAxisColor = parseInt(arg); this.RedrawPad(); });
            this.AddSizeMenuEntry(menu, "Size", -0.05, 0.055, 0.01, faxis.fTickLength,
                      function(arg) { faxis.fTickLength = parseFloat(arg); this.RedrawPad(); } );
            menu.add("endsub:");
         }
         return true;
      }

      var alone = menu.size()==0;

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
      this.FillAttContextMenu(menu,alone ? "" : "Frame ");
      menu.add("separator");
      menu.add("Save as frame.png", function() { this.pad_painter().SaveAs("png", 'frame', 'frame.png'); });
      menu.add("Save as frame.svg", function() { this.pad_painter().SaveAs("svg", 'frame', 'frame.svg'); });

      return true;
   }

   /** @summary Show axis status message
   *
   * @desc method called normally when mouse enter main object element
   * @private
   */
   TFramePainter.prototype.ShowAxisStatus = function(axis_name) {
      // method called normally when mouse enter main object element

      var status_func = this.GetShowStatusFunc();

      if (typeof status_func != "function") return;

      var taxis = null, hint_name = axis_name, hint_title = "TAxis",
          m = d3.mouse(this.svg_frame().node()), id = (axis_name=="x") ? 0 : 1;

      if (taxis) { hint_name = taxis.fName; hint_title = taxis.fTitle || "histogram TAxis object"; }

      if (this.swap_xy) id = 1-id;

      var axis_value = (axis_name=="x") ? this.RevertX(m[id]) : this.RevertY(m[id]);

      status_func(hint_name, hint_title, axis_name + " : " + this.AxisAsText(axis_name, axis_value), m[0]+","+m[1]);
   }

   TFramePainter.prototype.AddInteractive = function() {
      // only first painter in list allowed to add interactive functionality to the frame

      if (JSROOT.BatchMode || (!JSROOT.gStyle.Zooming && !JSROOT.gStyle.ContextMenu)) return;

      var pp = this.pad_painter();
      if (pp && pp._fast_drawing) return;

      var svg = this.svg_frame();

      if (svg.empty()) return;

      var svg_x = svg.selectAll(".xaxis_container"),
          svg_y = svg.selectAll(".yaxis_container");

      if (!svg.property('interactive_set')) {
         this.AddKeysHandler();

         this.last_touch = new Date(0);
         this.zoom_kind = 0; // 0 - none, 1 - XY, 2 - only X, 3 - only Y, (+100 for touches)
         this.zoom_rect = null;
         this.zoom_origin = null;  // original point where zooming started
         this.zoom_curr = null;    // current point for zooming
         this.touch_cnt = 0;
      }

      if (JSROOT.gStyle.Zooming && (!this.options || !this.options.Proj)) {
         if (JSROOT.gStyle.ZoomMouse) {
            svg.on("mousedown", this.startRectSel.bind(this));
            svg.on("dblclick", this.mouseDoubleClick.bind(this));
         }
         if (JSROOT.gStyle.ZoomWheel) {
            svg.on("wheel", this.mouseWheel.bind(this));
         }
      }

      if (JSROOT.touches && ((JSROOT.gStyle.Zooming && JSROOT.gStyle.ZoomTouch) || JSROOT.gStyle.ContextMenu))
         svg.on("touchstart", this.startTouchZoom.bind(this));

      if (JSROOT.gStyle.ContextMenu) {
         if (JSROOT.touches) {
            svg_x.on("touchstart", this.startTouchMenu.bind(this,"x"));
            svg_y.on("touchstart", this.startTouchMenu.bind(this,"y"));
         }
         svg.on("contextmenu", this.ShowContextMenu.bind(this));
         svg_x.on("contextmenu", this.ShowContextMenu.bind(this,"x"));
         svg_y.on("contextmenu", this.ShowContextMenu.bind(this,"y"));
      }

      svg_x.on("mousemove", this.ShowAxisStatus.bind(this,"x"));
      svg_y.on("mousemove", this.ShowAxisStatus.bind(this,"y"));

      svg.property('interactive_set', true);
   }

   TFramePainter.prototype.mouseWheel = function() {
      d3.event.stopPropagation();

      d3.event.preventDefault();
      this.clearInteractiveElements();

      var itemx = { name: "x", ignore: false },
          itemy = { name: "y", ignore: !this.AllowDefaultYZooming() },
          cur = d3.mouse(this.svg_frame().node()),
          w = this.frame_width(), h = this.frame_height();

      this.AnalyzeMouseWheelEvent(d3.event, this.swap_xy ? itemy : itemx, cur[0] / w, (cur[1] >=0) && (cur[1] <= h));

      this.AnalyzeMouseWheelEvent(d3.event, this.swap_xy ? itemx : itemy, 1 - cur[1] / h, (cur[0] >= 0) && (cur[0] <= w));

      this.Zoom(itemx.min, itemx.max, itemy.min, itemy.max);

      if (itemx.changed || itemy.changed) this.zoom_changed_interactive = 2;
   }

   TFramePainter.prototype.AllowDefaultYZooming = function() {
      // return true if default Y zooming should be enabled
      // it is typically for 2-Dim histograms or
      // when histogram not draw, defined by other painters

      var pad_painter = this.pad_painter();
      if (pad_painter &&  pad_painter.painters)
         for (var k = 0; k < pad_painter.painters.length; ++k) {
            var subpainter = pad_painter.painters[k];
            if (subpainter && (subpainter.wheel_zoomy!==undefined))
               return subpainter.wheel_zoomy;
         }

      return false;
   }


   TFramePainter.prototype.AnalyzeMouseWheelEvent = function(event, item, dmin, ignore) {

      item.min = item.max = undefined;
      item.changed = false;
      if (ignore && item.ignore) return;

      var delta = 0, delta_left = 1, delta_right = 1;

      if ('dleft' in item) { delta_left = item.dleft; delta = 1; }
      if ('dright' in item) { delta_right = item.dright; delta = 1; }

      if ('delta' in item) {
         delta = item.delta;
      } else if (event && event.wheelDelta !== undefined ) {
         // WebKit / Opera / Explorer 9
         delta = -event.wheelDelta;
      } else if (event && event.deltaY !== undefined ) {
         // Firefox
         delta = event.deltaY;
      } else if (event && event.detail !== undefined) {
         delta = event.detail;
      }

      if (delta===0) return;
      delta = (delta<0) ? -0.2 : 0.2;

      delta_left *= delta
      delta_right *= delta;

      var lmin = item.min = this["scale_"+item.name+"min"],
          lmax = item.max = this["scale_"+item.name+"max"],
          gmin = this[item.name+"min"],
          gmax = this[item.name+"max"];

      if ((item.min === item.max) && (delta<0)) {
         item.min = gmin;
         item.max = gmax;
      }

      if (item.min >= item.max) return;

      if ((dmin>0) && (dmin<1)) {
         if (this['log'+item.name]) {
            var factor = (item.min>0) ? JSROOT.log10(item.max/item.min) : 2;
            if (factor>10) factor = 10; else if (factor<0.01) factor = 0.01;
            item.min = item.min / Math.pow(10, factor*delta_left*dmin);
            item.max = item.max * Math.pow(10, factor*delta_right*(1-dmin));
         } else {
            var rx_left = (item.max - item.min), rx_right = rx_left;
            if (delta_left>0) rx_left = 1.001 * rx_left / (1-delta_left);
            item.min += -delta_left*dmin*rx_left;

            if (delta_right>0) rx_right = 1.001 * rx_right / (1-delta_right);

            item.max -= -delta_right*(1-dmin)*rx_right;
         }
         if (item.min >= item.max)
            item.min = item.max = undefined;
         else
         if (delta_left !== delta_right) {
            // extra check case when moving left or right
            if (((item.min < gmin) && (lmin===gmin)) ||
                ((item.max > gmax) && (lmax==gmax)))
                   item.min = item.max = undefined;
         }

      } else {
         item.min = item.max = undefined;
      }

      item.changed = ((item.min !== undefined) && (item.max !== undefined));
   }

   TFramePainter.prototype.AddKeysHandler = function() {
      if (this.keys_handler || JSROOT.BatchMode || (typeof window == 'undefined')) return;

      this.keys_handler = this.ProcessKeyPress.bind(this);

      window.addEventListener('keydown', this.keys_handler, false);
   }

   TFramePainter.prototype.ProcessKeyPress = function(evnt) {

      var main = this.select_main();
      if (main.empty()) return;

      var key = "";
      switch (evnt.keyCode) {
         case 33: key = "PageUp"; break;
         case 34: key = "PageDown"; break;
         case 37: key = "ArrowLeft"; break;
         case 38: key = "ArrowUp"; break;
         case 39: key = "ArrowRight"; break;
         case 40: key = "ArrowDown"; break;
         case 42: key = "PrintScreen"; break;
         case 106: key = "*"; break;
         default: return false;
      }

      if (evnt.shiftKey) key = "Shift " + key;
      if (evnt.altKey) key = "Alt " + key;
      if (evnt.ctrlKey) key = "Ctrl " + key;

      var zoom = { name: "x", dleft: 0, dright: 0 };

      switch (key) {
         case "ArrowLeft":  zoom.dleft = -1; zoom.dright = 1; break;
         case "ArrowRight":  zoom.dleft = 1; zoom.dright = -1; break;
         case "Ctrl ArrowLeft": zoom.dleft = zoom.dright = -1; break;
         case "Ctrl ArrowRight": zoom.dleft = zoom.dright = 1; break;
         case "ArrowUp":  zoom.name = "y"; zoom.dleft = 1; zoom.dright = -1; break;
         case "ArrowDown":  zoom.name = "y"; zoom.dleft = -1; zoom.dright = 1; break;
         case "Ctrl ArrowUp": zoom.name = "y"; zoom.dleft = zoom.dright = 1; break;
         case "Ctrl ArrowDown": zoom.name = "y"; zoom.dleft = zoom.dright = -1; break;
      }

      if (zoom.dleft || zoom.dright) {
         if (!JSROOT.gStyle.Zooming) return false;
         // in 3dmode with orbit control ignore simple arrows
         if (this.mode3d && (key.indexOf("Ctrl")!==0)) return false;
         this.AnalyzeMouseWheelEvent(null, zoom, 0.5);
         this.Zoom(zoom.name, zoom.min, zoom.max);
         if (zoom.changed) this.zoom_changed_interactive = 2;
         evnt.stopPropagation();
         evnt.preventDefault();
      } else {
         var pp = this.pad_painter(),
             func = pp ? pp.FindButton(key) : "";
         if (func) {
            pp.PadButtonClick(func);
            evnt.stopPropagation();
            evnt.preventDefault();
         }
      }

      return true; // just process any key press
   }

   TFramePainter.prototype.CreateXY = function() {
      // here we create x,y objects which maps our physical coordinates into pixels
      // while only first painter really need such object, all others just reuse it
      // following functions are introduced
      //    this.GetBin[X/Y]  return bin coordinate
      //    this.Convert[X/Y]  converts root value in JS date when date scale is used
      //    this.[x,y]  these are d3.scale objects
      //    this.gr[x,y]  converts root scale into graphical value
      //    this.Revert[X/Y]  converts graphical coordinates to root scale value

      this.swap_xy = false;
      this.reverse_x = false;
      this.reverse_y = false;

      // if (this.options.BarStyle>=20) this.swap_xy = true;
      this.logx = this.logy = false;

      var w = this.frame_width(), h = this.frame_height();

      this.scale_xmin = this.xmin;
      this.scale_xmax = this.xmax;

      this.scale_ymin = this.ymin;
      this.scale_ymax = this.ymax;

      // if (opts.extra_y_space) {
      //    var log_scale = this.swap_xy ? pad.fLogx : pad.fLogy;
      //    if (log_scale && (this.scale_ymax > 0))
      //       this.scale_ymax = Math.exp(Math.log(this.scale_ymax)*1.1);
      //    else
      //       this.scale_ymax += (this.scale_ymax - this.scale_ymin) * 0.1;
      // }

      //if (typeof this.RecalculateRange == "function")
      //   this.RecalculateRange();

      if (this._xaxis_timedisplay) {
         this.x_kind = 'time';
         this.timeoffsetx = JSROOT.Painter.getTimeOffset(/*this.histo.fXaxis*/);
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

      var gr_range_x = this.reverse_x ? [ w, 0 ] : [ 0, w ],
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
         this.timeoffsety = JSROOT.Painter.getTimeOffset(/*this.histo.fYaxis*/);
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
   TFramePainter.prototype.SetRootPadRange = function(pad, is3d) {
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
         pad.fUxmin = JSROOT.log10(pad.fUxmin);
         pad.fUxmax = JSROOT.log10(pad.fUxmax);
      }
      if (pad.fLogy) {
         pad.fUymin = JSROOT.log10(pad.fUymin);
         pad.fUymax = JSROOT.log10(pad.fUymax);
      }

      var rx = pad.fUxmax - pad.fUxmin,
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

   TFramePainter.prototype.ToggleLog = function(axis) {
      var painter = this.main_painter() || this,
          pad = this.root_pad();
      var curr = pad["fLog" + axis];
      // do not allow log scale for labels
      if (!curr) {
         var kind = this[axis+"_kind"];
         if (this.swap_xy && axis==="x") kind = this["y_kind"]; else
         if (this.swap_xy && axis==="y") kind = this["x_kind"];
         if (kind === "labels") return;
      }

      var pp = this.pad_painter(), canp = this.canv_painter();
      if (pp && pp.snapid && canp && canp._websocket) {
         console.warn('Change log scale on server here!!!!');
         // canp.SendWebsocket("OBJEXEC:" + pp.snapid + ":SetLog" + axis + (curr ? "(0)" : "(1)"));
      } else {
         pad["fLog" + axis] = curr ? 0 : 1;
         painter.RedrawPad();
      }
   }

   function drawFrame(divid, obj, opt) {
      var p = new TFramePainter(obj);
      if (opt == "3d") p.mode3d = true;
      p.SetDivId(divid, 2);
      p.Redraw();
      return p.DrawingReady();
   }

   // ===========================================================================

   function TPadPainter(pad, iscan) {
      JSROOT.TObjectPainter.call(this, pad);
      this.csstype = "pad";
      this.pad = pad;
      this.iscan = iscan; // indicate if working with canvas
      this.this_pad_name = "";
      if (!this.iscan && (pad !== null)) {
         if (pad.fObjectID)
            this.this_pad_name = "pad" + pad.fObjectID; // use objectid as padname
         else
            this.this_pad_name = "ppp" + JSROOT.id_counter++; // artificical name
      }
      this.painters = []; // complete list of all painters in the pad
      this.has_canvas = true;
   }

   TPadPainter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   TPadPainter.prototype.Cleanup = function() {
      // cleanup only pad itself, all child elements will be collected and cleanup separately

      for (var k=0;k<this.painters.length;++k)
         this.painters[k].Cleanup();

      var svg_p = this.svg_pad(this.this_pad_name);
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

      JSROOT.Painter.SelectActivePad({ pp: this, active: false });

      JSROOT.TObjectPainter.prototype.Cleanup.call(this);
   }

   /** @summary Cleanup primitives from pad - selector lets define which painters to remove
    * @private
    */

   TPadPainter.prototype.CleanPrimitives = function(selector) {
      if (!selector || (typeof selector !== 'function')) return;

      for (var k = this.painters.length-1; k >= 0; --k)
         if (selector(this.painters[k])) {
            this.painters[k].Cleanup();
            this.painters.splice(k, 1);
         }
   }

   /// call function for each painter
   /// kind == "all" for all objects (default)
   /// kind == "pads" only pads and subpads
   /// kind == "objects" only for object in current pad
   TPadPainter.prototype.ForEachPainterInPad = function(userfunc, kind) {
      if (!kind) kind = "all";
      if (kind!="objects") userfunc(this);
      for (var k = 0; k < this.painters.length; ++k) {
         var sub = this.painters[k];
         if (typeof sub.ForEachPainterInPad === 'function') {
            if (kind!="objects") sub.ForEachPainterInPad(userfunc, kind);
         } else if (kind != "pads") userfunc(sub);
      }
   }

   TPadPainter.prototype.ButtonSize = function(fact) {
      return Math.round((!fact ? 1 : fact) * (this.iscan || !this.has_canvas ? 16 : 12));
   }

   TPadPainter.prototype.RegisterForPadEvents = function(receiver) {
      this.pad_events_receiver = receiver;
   }

   TPadPainter.prototype.SelectObjectPainter = function(_painter, pos) {
      // dummy function, redefined in the TCanvasPainter

      var istoppad = (this.iscan || !this.has_canvas),
          canp = istoppad ? this : this.canv_painter(),
          pp = _painter instanceof TPadPainter ? _painter : _painter.pad_painter();

      if (pos && !istoppad)
          this.CalcAbsolutePosition(this.svg_pad(this.this_pad_name), pos);

      JSROOT.Painter.SelectActivePad({ pp: pp, active: true });

      if (typeof canp.SelectActivePad == "function")
          canp.SelectActivePad(pp, _painter, pos);

      if (canp.pad_events_receiver)
         canp.pad_events_receiver({ what: "select", padpainter: pp, painter: _painter, position: pos });
   }

   /** @brief Called by framework when pad is supposed to be active and get focus
    * @private */
   TPadPainter.prototype.SetActive = function(on) {
      var fp = this.frame_painter();
      if (fp && (typeof fp.SetActive == 'function')) fp.SetActive(on);
   }

   TPadPainter.prototype.CreateCanvasSvg = function(check_resize, new_size) {

      var factor = null, svg = null, lmt = 5, rect = null, btns;

      if (check_resize > 0) {

         if (this._fixed_size) return (check_resize > 1); // flag used to force re-drawing of all subpads

         svg = this.svg_canvas();

         if (svg.empty()) return false;

         factor = svg.property('height_factor');

         rect = this.check_main_resize(check_resize, null, factor);

         if (!rect.changed) return false;

         btns = this.svg_layer("btns_layer");

      } else {

         var render_to = this.select_main();

         if (render_to.style('position')=='static')
            render_to.style('position','relative');

         svg = render_to.append("svg")
             .attr("class", "jsroot root_canvas")
             .property('pad_painter', this) // this is custom property
             .property('mainpainter', null) // this is custom property
             .property('current_pad', "") // this is custom property
             .property('redraw_by_resize', false); // could be enabled to force redraw by each resize

         svg.append("svg:title").text("ROOT canvas");
         var frect = svg.append("svg:rect").attr("class","canvas_fillrect")
                               .attr("x",0).attr("y",0);
         if (!JSROOT.BatchMode)
            frect.style("pointer-events", "visibleFill")
                 .on("dblclick", this.EnlargePad.bind(this))
                 .on("click", this.SelectObjectPainter.bind(this, this))
                 .on("mouseenter", this.ShowObjectStatus.bind(this));

         svg.append("svg:g").attr("class","primitives_layer");
         svg.append("svg:g").attr("class","info_layer");
         btns = svg.append("svg:g").attr("class","btns_layer")
                                   .property('leftside', JSROOT.gStyle.ToolBarSide == 'left')
                                   .property('vertical', JSROOT.gStyle.ToolBarVert);

         if (JSROOT.gStyle.ContextMenu)
            svg.select(".canvas_fillrect").on("contextmenu", this.ShowContextMenu.bind(this));

         factor = 0.66;
         if (this.pad && this.pad.fCw && this.pad.fCh && (this.pad.fCw > 0)) {
            factor = this.pad.fCh / this.pad.fCw;
            if ((factor < 0.1) || (factor > 10)) factor = 0.66;
         }

         if (this._fixed_size) {
            render_to.style("overflow","auto");
            rect = { width: this.pad.fCw, height: this.pad.fCh };
         } else {
            rect = this.check_main_resize(2, new_size, factor);
         }
      }

      // this.createAttFill({ attr: this.pad });

      if ((rect.width<=lmt) || (rect.height<=lmt)) {
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

      // console.log('CANVAS SVG width = ' + rect.width + " height = " + rect.height);

      svg.attr("viewBox", "0 0 " + rect.width + " " + rect.height)
         .attr("preserveAspectRatio", "none")  // we do not preserve relative ratio
         .property('height_factor', factor)
         .property('draw_x', 0)
         .property('draw_y', 0)
         .property('draw_width', rect.width)
         .property('draw_height', rect.height);

      //svg.select(".canvas_fillrect")
      //   .attr("width", rect.width)
      //   .attr("height", rect.height)
      //   .call(this.fillatt.func);

      this._fast_drawing = JSROOT.gStyle.SmallPad && ((rect.width < JSROOT.gStyle.SmallPad.width) || (rect.height < JSROOT.gStyle.SmallPad.height));

      this.AlignBtns(btns, rect.width, rect.height, svg);

      return true;
   }

   TPadPainter.prototype.EnlargePad = function() {

      if (d3.event) {
         d3.event.preventDefault();
         d3.event.stopPropagation();
      }

      var svg_can = this.svg_canvas(),
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

      var was_fast = this._fast_drawing;

      this.CheckResize({ force: true });

      if (this._fast_drawing != was_fast)
         this.ShowButtons();
   }

   TPadPainter.prototype.CreatePadSvg = function(only_resize) {
      // returns true when pad is displayed and all its items should be redrawn

      if (!this.has_canvas) {
         this.CreateCanvasSvg(only_resize ? 2 : 0);
         return true;
      }

      var svg_parent = this.svg_pad(),
          svg_can = this.svg_canvas(),
          width = svg_parent.property("draw_width"),
          height = svg_parent.property("draw_height"),
          pad_enlarged = svg_can.property("pad_enlarged"),
          pad_visible = !pad_enlarged || (pad_enlarged === this.pad),
          w = width, h = height, x = 0, y = 0,
          svg_pad = null, svg_rect = null, btns = null;

      if (this.pad && this.pad.fPos && this.pad.fSize) {
         x = Math.round(width * this.pad.fPos.fHoriz.fArr[0]);
         y = Math.round(height * this.pad.fPos.fVert.fArr[0]);
         w = Math.round(width * this.pad.fSize.fHoriz.fArr[0]);
         h = Math.round(height * this.pad.fSize.fVert.fArr[0]);
      }

      if (pad_enlarged === this.pad) { w = width; h = height; x = y = 0; }

      if (only_resize) {
         svg_pad = this.svg_pad(this.this_pad_name);
         svg_rect = svg_pad.select(".root_pad_border");
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
         btns = svg_pad.append("svg:g").attr("class","btns_layer")
                                       .property('leftside', JSROOT.gStyle.ToolBarSide != 'left')
                                       .property('vertical', JSROOT.gStyle.ToolBarVert);

         if (JSROOT.gStyle.ContextMenu)
            svg_rect.on("contextmenu", this.ShowContextMenu.bind(this));

         if (!JSROOT.BatchMode)
            svg_rect.attr("pointer-events", "visibleFill") // get events also for not visible rect
                    .on("dblclick", this.EnlargePad.bind(this))
                    .on("click", this.SelectObjectPainter.bind(this, this))
                    .on("mouseenter", this.ShowObjectStatus.bind(this));
      }

      this.createAttFill({ attr: this.pad });

      this.createAttLine({ attr: this.pad, color0: this.pad.fBorderMode == 0 ? 'none' : '' });

      svg_pad
              //.attr("transform", "translate(" + x + "," + y + ")") // is not handled for SVG
             .attr("display", pad_visible ? null : "none")
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

      this._fast_drawing = JSROOT.gStyle.SmallPad && ((w < JSROOT.gStyle.SmallPad.width) || (h < JSROOT.gStyle.SmallPad.height));

      if (svg_pad.property('can3d') === 1)
         // special case of 3D canvas overlay
          this.select_main()
              .select(".draw3d_" + this.this_pad_name)
              .style('display', pad_visible ? '' : 'none');

      this.AlignBtns(btns, w, h);

      return pad_visible;
   }

   TPadPainter.prototype.RemovePrimitive = function(obj) {
      if (!this.pad || !this.pad.fPrimitives) return;
      var indx = this.pad.fPrimitives.arr.indexOf(obj);
      if (indx>=0) this.pad.fPrimitives.RemoveAt(indx);
   }

   TPadPainter.prototype.FindPrimitive = function(exact_obj, classname, name) {
      if (!this.pad || !this.pad.fPrimitives) return null;

      for (var i=0; i < this.pad.fPrimitives.arr.length; i++) {
         var obj = this.pad.fPrimitives.arr[i];

         if ((exact_obj!==null) && (obj !== exact_obj)) continue;

         if ((classname !== undefined) && (classname !== null))
            if (obj._typename !== classname) continue;

         if ((name !== undefined) && (name !== null))
            if (obj.fName !== name) continue;

         return obj;
      }

      return null;
   }

   TPadPainter.prototype.HasObjectsToDraw = function() {
      // return true if any objects beside sub-pads exists in the pad

      var arr = this.pad ? this.pad.fPrimitives : null;

      if (arr)
         for (var n=0;n<arr.length;++n)
            if (arr[n] && arr[n]._typename != "ROOT::Experimental::RPadDisplayItem") return true;

      return false;
   }

   TPadPainter.prototype.DrawPrimitives = function(indx, callback, ppainter) {

      if (indx===0) {
         // flag used to prevent immediate pad redraw during normal drawing sequence
         this._doing_pad_draw = true;

         if (this.iscan)
            this._start_tm = this._lasttm_tm =  new Date().getTime();

         // set number of primitves
         this._num_primitives = this.pad && this.pad.fPrimitives ? this.pad.fPrimitives.length : 0;
      }

      while (true) {
         if (ppainter && (typeof ppainter=='object')) ppainter._primitive = true; // mark painter as belonging to primitives

         if (!this.pad || (indx >= this.pad.fPrimitives.length)) {
            delete this._doing_pad_draw;
            delete this._current_primitive_indx;

            if (this._start_tm) {
               var spenttm = new Date().getTime() - this._start_tm;
               if (spenttm > 3000) console.log("Canvas drawing took " + (spenttm*1e-3).toFixed(2) + "s");
               delete this._start_tm;
               delete this._lasttm_tm;
            }

            return JSROOT.CallBack(callback);
         }

         // handle use to invoke callback only when necessary
         var handle = { func: this.DrawPrimitives.bind(this, indx+1, callback) };

         // set current index
         this._current_primitive_indx = indx;

         ppainter = JSROOT.draw(this.divid, this.pad.fPrimitives[indx], "", handle);

         indx++;

         if (!handle.completed) return;

         if (!JSROOT.BatchMode && this.iscan) {
            var curtm = new Date().getTime();
            if (curtm > this._lasttm_tm + 500) {
               this._lasttm_tm = curtm;
               ppainter._primitive = true; // mark primitive ourself
               return requestAnimationFrame(handle.func);
            }
         }
      }
   }

   TPadPainter.prototype.GetTooltips = function(pnt) {
      var painters = [], hints = [];

      // first count - how many processors are there
      if (this.painters !== null)
         this.painters.forEach(function(obj) {
            if ('ProcessTooltip' in obj) painters.push(obj);
         });

      if (pnt) pnt.nproc = painters.length;

      painters.forEach(function(obj) {
         var hint = obj.ProcessTooltip(pnt);
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

         function ToggleGridField(arg) {
            this.pad[arg] = this.pad[arg] ? 0 : 1;
            var main = this.svg_pad(this.this_pad_name).property('mainpainter');
            if (main && (typeof main.DrawGrids == 'function')) main.DrawGrids();
         }

         function SetTickField(arg) {
            this.pad[arg.substr(1)] = parseInt(arg[0]);

            var main = this.svg_pad(this.this_pad_name).property('mainpainter');
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

         this.FillAttContextMenu(menu);
      }

      menu.add("separator");

      if (this.ToggleEventStatus)
         menu.addchk(this.HasEventStatus(), "Event status", this.ToggleEventStatus.bind(this));

      if (this.enlarge_main() || (this.has_canvas && this.HasObjectsToDraw()))
         menu.addchk((this.enlarge_main('state')=='on'), "Enlarge " + (this.iscan ? "canvas" : "pad"), this.EnlargePad.bind(this));

      var fname = this.this_pad_name;
      if (fname.length===0) fname = this.iscan ? "canvas" : "pad";
      menu.add("Save as "+fname+".png", fname+".png", this.SaveAs.bind(this, "png", false));
      menu.add("Save as "+fname+".svg", fname+".svg", this.SaveAs.bind(this, "svg", false));

      return true;
   }

   TPadPainter.prototype.ShowContextMenu = function(evnt) {
      if (!evnt) {
         // for debug purposes keep original context menu for small region in top-left corner
         var pos = d3.mouse(this.svg_pad(this.this_pad_name).node());

         if (pos && (pos.length==2) && (pos[0]>0) && (pos[0]<10) && (pos[1]>0) && pos[1]<10) return;

         d3.event.stopPropagation(); // disable main context menu
         d3.event.preventDefault();  // disable browser context menu

         // one need to copy event, while after call back event may be changed
         evnt = d3.event;

         var fp = this.frame_painter();
         if (fp) fp.SetLastEventPos();
      }

      JSROOT.Painter.createMenu(this, function(menu) {

         menu.painter.FillContextMenu(menu);

         menu.painter.FillObjectExecMenu(menu, "", function() { menu.show(evnt); });
      }); // end menu creation
   }

   TPadPainter.prototype.Redraw = function(resize) {

      // prevent redrawing
      if (this._doing_pad_draw) return console.log('Prevent redrawing', this.pad.fName);

      var showsubitems = true;

      if (this.iscan) {
         this.CreateCanvasSvg(2);
      } else {
         showsubitems = this.CreatePadSvg(true);
      }

      // even sub-pad is not visible, we should redraw sub-sub-pads to hide them as well
      for (var i = 0; i < this.painters.length; ++i) {
         var sub = this.painters[i];
         if (showsubitems || sub.this_pad_name) sub.Redraw(resize);
      }
   }

   TPadPainter.prototype.NumDrawnSubpads = function() {
      if (this.painters === undefined) return 0;

      var num = 0;

      for (var i = 0; i < this.painters.length; ++i) {
         var obj = this.painters[i].GetObject();
         if (obj && (obj._typename === "TPad")) num++;
      }

      return num;
   }

   TPadPainter.prototype.RedrawByResize = function() {
      if (this.access_3d_kind() === 1) return true;

      for (var i = 0; i < this.painters.length; ++i)
         if (typeof this.painters[i].RedrawByResize === 'function')
            if (this.painters[i].RedrawByResize()) return true;

      return false;
   }

   TPadPainter.prototype.CheckCanvasResize = function(size, force) {

      if (!this.iscan && this.has_canvas) return false;

      if ((size === true) || (size === false)) { force = size; size = null; }

      if (size && (typeof size === 'object') && size.force) force = true;

      if (!force) force = this.RedrawByResize();

      var changed = this.CreateCanvasSvg(force ? 2 : 1, size);

      // if canvas changed, redraw all its subitems.
      // If redrawing was forced for canvas, same applied for sub-elements
      if (changed)
         for (var i = 0; i < this.painters.length; ++i)
            this.painters[i].Redraw(force ? false : true);

      return changed;
   }

   TPadPainter.prototype.UpdateObject = function(obj) {
      if (!obj) return false;

      this.pad.fCw = obj.fCw;
      this.pad.fCh = obj.fCh;
      this.pad.fTitle = obj.fTitle;

      return true;
   }

   TPadPainter.prototype.DrawNextSnap = function(lst, indx, call_back, objpainter) {
      // function called when drawing next snapshot from the list
      // it is also used as callback for drawing of previous snap

      if (indx===-1) {
         // flag used to prevent immediate pad redraw during first draw
         this._doing_pad_draw = true;
         this._snaps_map = {}; // to control how much snaps are drawn
         this._num_primitives = lst ? lst.length : 0;
      }

      // workaround to insert v6 frame in list of primitives
      if (objpainter === "workaround") { --indx; objpainter = null; }

      while (true) {

         if (objpainter && lst && lst[indx] && objpainter.snapid === undefined) {
            // keep snap id in painter, will be used for the
            if (this.painters.indexOf(objpainter)<0) this.painters.push(objpainter);
            objpainter.snapid = lst[indx].fObjectID;
            objpainter.rstyle = lst[indx].fStyle;
         }

         objpainter = null;

         ++indx; // change to the next snap

         if (!lst || indx >= lst.length) {
            delete this._doing_pad_draw;
            delete this._snaps_map;
            delete this._current_primitive_indx;
            return JSROOT.CallBack(call_back, this);
         }

         var snap = lst[indx],
             snapid = snap.fObjectID,
             cnt = this._snaps_map[snapid];

         if (cnt) cnt++; else cnt=1;
         this._snaps_map[snapid] = cnt; // check how many objects with same snapid drawn, use them again

         this._current_primitive_indx = indx;

         // first appropriate painter for the object
         // if same object drawn twice, two painters will exists
         for (var k=0; k<this.painters.length; ++k) {
            if (this.painters[k].snapid === snapid)
               if (--cnt === 0) { objpainter = this.painters[k]; break;  }
         }

         // function which should be called when drawing of next item finished
         var draw_callback = this.DrawNextSnap.bind(this, lst, indx, call_back);

         if (objpainter) {

            if (snap._typename == "ROOT::Experimental::RPadDisplayItem")  // subpad
               return objpainter.RedrawPadSnap(snap, draw_callback);

            if (objpainter.UpdateObject(snap.fDrawable || snap.fObject, snap.fOption || ""))
               objpainter.Redraw();

            continue; // call next
         }

         if (snap._typename == "ROOT::Experimental::RPadDisplayItem") { // subpad

            var subpad = snap; // not subpad, but just attributes

            var padpainter = new TPadPainter(subpad, false);
            padpainter.DecodeOptions("");
            padpainter.SetDivId(this.divid); // pad painter will be registered in the canvas painters list
            padpainter.snapid = snap.fObjectID;
            padpainter.rstyle = snap.fStyle;

            padpainter.CreatePadSvg();

            if (snap.fPrimitives && snap.fPrimitives.length > 0) {
               padpainter.AddButton(JSROOT.ToolbarIcons.camera, "Create PNG", "PadSnapShot");
               padpainter.AddButton(JSROOT.ToolbarIcons.circle, "Enlarge pad", "EnlargePad");

               if (JSROOT.gStyle.ContextMenu)
                  padpainter.AddButton(JSROOT.ToolbarIcons.question, "Access context menus", "PadContextMenus");
            }

            // we select current pad, where all drawing is performed
            var prev_name = padpainter.CurrentPadName(padpainter.this_pad_name);

            padpainter.DrawNextSnap(snap.fPrimitives, -1, function() {
               padpainter.CurrentPadName(prev_name);
               draw_callback(padpainter);
            });
            return;
         }

         var handle = { func: draw_callback };

         if (snap._typename === "ROOT::Experimental::RObjectDisplayItem")
            if (!this.frame_painter())
               return JSROOT.draw(this.divid, { _typename: "TFrame", $dummy: true }, "", function() {
                  handle.func("workaround"); // call function with "workaround" as argument
               });

         // TODO - fDrawable is v7, fObject from v6, maybe use same data member?
         objpainter = JSROOT.draw(this.divid, snap.fDrawable || snap.fObject, snap.fOption || "", handle);

         if (!handle.completed) return; // if callback will be invoked, break while loop
      }
   }

   TPadPainter.prototype.FindSnap = function(snapid) {

      if (this.snapid === snapid) return this;

      if (!this.painters) return null;

      for (var k=0;k<this.painters.length;++k) {
         var sub = this.painters[k];

         if (typeof sub.FindSnap === 'function') sub = sub.FindSnap(snapid);
         else if (sub.snapid !== snapid) sub = null;

         if (sub) return sub;
      }

      return null;
   }

   TPadPainter.prototype.AddOnlineButtons = function() {
      this.AddButton(JSROOT.ToolbarIcons.camera, "Create PNG", "CanvasSnapShot", "Ctrl PrintScreen");
      if (JSROOT.gStyle.ContextMenu)
         this.AddButton(JSROOT.ToolbarIcons.question, "Access context menus", "PadContextMenus");

      if (this.enlarge_main('verify'))
         this.AddButton(JSROOT.ToolbarIcons.circle, "Enlarge canvas", "EnlargePad");
   }

   TPadPainter.prototype.RedrawPadSnap = function(snap, call_back) {
      // for the pad/canvas display item contains list of primitives plus pad attributes

      if (!snap || !snap.fPrimitives) return;

      // for the moment only window size attributes are provided
      var padattr = { fCw: snap.fWinSize[0], fCh: snap.fWinSize[1], fTitle: snap.fTitle };

      // if canvas size not specified in batch mode, temporary use 900x700 size
      if (this.batch_mode && this.iscan && (!padattr.fCw || !padattr.fCh)) { padattr.fCw = 900; padattr.fCh = 700; }

      if (this.iscan && snap.fTitle && document)
         document.title = snap.fTitle;

      if (this.iscan && snap.fTitle && document)
         document.title = snap.fTitle;

      if (this.snapid === undefined) {
         // first time getting snap, create all gui elements first

         this.snapid = snap.fObjectID;

         this.draw_object = padattr;
         this.pad = padattr;
         this.pad_frame = snap.fFrame;

         if (this.batch_mode && this.iscan)
             this._fixed_size = true;

         this.CreateCanvasSvg(0);
         this.SetDivId(this.divid);  // now add to painters list
         this.AddOnlineButtons();

         this.DrawNextSnap(snap.fPrimitives, -1, call_back);

         return;
      }

      // update only pad/canvas attributes
      this.UpdateObject(padattr);

      // apply all changes in the object (pad or canvas)
      if (this.iscan) {
         this.CreateCanvasSvg(2);
      } else {
         this.CreatePadSvg(true);
      }

      var isanyfound = false, isanyremove = false;

      // find and remove painters which no longer exists in the list
      for (var k=0;k<this.painters.length;++k) {
         var sub = this.painters[k];
         if (sub.snapid===undefined) continue; // look only for painters with snapid

         for (var i=0;i<snap.fPrimitives.length;++i)
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
         var svg_p = this.svg_pad(this.this_pad_name),
             fp = this.frame_painter();
         if (svg_p && !svg_p.empty())
            svg_p.property('mainpainter', null);
         for (var k=0;k<this.painters.length;++k)
            if (fp !== this.painters[k])
               this.painters[k].Cleanup();
         this.painters = [];
         if (fp) {
            this.painters.push(fp);
            fp.CleanFrameDrawings();
         }
         this.RemoveButtons();
         this.AddOnlineButtons();
      }

      var padpainter = this,
          prev_name = padpainter.CurrentPadName(padpainter.this_pad_name);

      padpainter.DrawNextSnap(snap.fPrimitives, -1, function() {
         padpainter.CurrentPadName(prev_name);
         call_back(padpainter);
      });
   }

   TPadPainter.prototype.CreateImage = function(format, call_back) {
      if (format=="pdf") {
         // use https://github.com/MrRio/jsPDF in the future here
         JSROOT.CallBack(call_back, btoa("dummy PDF file"));
      } else if ((format=="png") || (format=="jpeg") || (format=="svg")) {
         this.ProduceImage(true, format, function(res) {
            if ((format=="svg") || !res)
               return JSROOT.CallBack(call_back, res);
            var separ = res.indexOf("base64,");
            JSROOT.CallBack(call_back, (separ>0) ? res.substr(separ+7) : "");
         });
      } else {
         JSROOT.CallBack(call_back, "");
      }
   }

   TPadPainter.prototype.ItemContextMenu = function(name) {
       var rrr = this.svg_pad(this.this_pad_name).node().getBoundingClientRect();
       var evnt = { clientX: rrr.left+10, clientY: rrr.top + 10 };

       // use timeout to avoid conflict with mouse click and automatic menu close
       if (name=="pad")
          return setTimeout(this.ShowContextMenu.bind(this, evnt), 50);

       var selp = null, selkind;

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
             var indx = parseInt(name);
             if (!isNaN(indx)) selp = this.painters[indx];
          }
       }

       if (!selp || (typeof selp.FillContextMenu !== 'function')) return;

       JSROOT.Painter.createMenu(selp, function(menu) {
          if (selp.FillContextMenu(menu,selkind))
             setTimeout(menu.show.bind(menu, evnt), 50);
       });
   }

   TPadPainter.prototype.SaveAs = function(kind, full_canvas, filename) {
      if (!filename) {
         filename = this.this_pad_name;
         if (filename.length === 0) filename = this.iscan ? "canvas" : "pad";
         filename += "." + kind;
      }
      this.ProduceImage(full_canvas, kind, function(imgdata) {
         var a = document.createElement('a');
         a.download = filename;
         a.href = (kind != "svg") ? imgdata : "data:image/svg+xml;charset=utf-8,"+encodeURIComponent(imgdata);
         document.body.appendChild(a);
         a.addEventListener("click", function(e) {
            a.parentNode.removeChild(a);
         });
         a.click();
      });
   }

   TPadPainter.prototype.ProduceImage = function(full_canvas, file_format, call_back) {

      var use_frame = (full_canvas === "frame");

      var elem = use_frame ? this.svg_frame() : (full_canvas ? this.svg_canvas() : this.svg_pad(this.this_pad_name));

      if (elem.empty()) return JSROOT.CallBack(call_back);

      var painter = (full_canvas && !use_frame) ? this.canv_painter() : this;

      var items = []; // keep list of replaced elements, which should be moved back at the end

//      document.body.style.cursor = 'wait';

      if (!use_frame) // do not make transformations for the frame
      painter.ForEachPainterInPad(function(pp) {

         // console.log('Check painter pp', pp.this_pad_name);

         var item = { prnt: pp.svg_pad(pp.this_pad_name) };
         items.push(item);

         // remove buttons from each subpad
         var btns = pp.svg_layer("btns_layer", pp.this_pad_name);
         item.btns_node = btns.node();
         if (item.btns_node) {
            item.btns_prnt = item.btns_node.parentNode;
            item.btns_next = item.btns_node.nextSibling;
            btns.remove();
         }

         var main = pp.frame_painter_ref;
         if (!main || (typeof main.Render3D !== 'function')) return;

         var can3d = main.access_3d_kind();

         if ((can3d !== 1) && (can3d !== 2)) return;

         var sz2 = main.size_for_3d(2); // get size of DOM element as it will be embed

         var sz = (can3d == 2) ? sz : main.size_for_3d(1);

         // console.log('Render 3D', sz2);

         var canvas = main.renderer.domElement;
         main.Render3D(0); // WebGL clears buffers, therefore we should render scene and convert immediately
         var dataUrl = canvas.toDataURL("image/png");

         // console.log('canvas width height', canvas.width, canvas.height);

         // console.log('produced png image len = ', dataUrl.length, 'begin', dataUrl.substr(0,100));

         // remove 3D drawings

         if (can3d == 2) {
            item.foreign = item.prnt.select("." + sz2.clname);
            item.foreign.remove();
         }

         var svg_frame = main.svg_frame();
         item.frame_node = svg_frame.node();
         if (item.frame_node) {
            item.frame_next = item.frame_node.nextSibling;
            svg_frame.remove();
         }

         //var origin = main.apply_3d_size(sz3d, true);
         //origin.remove();

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
           var c = String.fromCharCode('0x'+p1);
           return c === '%' ? '%25' : c;
         });
         return decodeURIComponent(data);
      }

      function reconstruct(res) {
         for (var k=0;k<items.length;++k) {
            var item = items[k];

            if (item.img)
               item.img.remove(); // delete embed image

            var prim = item.prnt.select(".primitives_layer");

            if (item.foreign) // reinsert foreign object
               item.prnt.node().insertBefore(item.foreign.node(), prim.node());

            if (item.frame_node) // reinsert frame as first in list of primitives
               prim.node().insertBefore(item.frame_node, item.frame_next);

            if (item.btns_node) // reinsert buttons
               item.btns_prnt.insertBefore(item.btns_node, item.btns_next);
         }

         JSROOT.CallBack(call_back, res);
      }

      var width = elem.property('draw_width'), height = elem.property('draw_height');
      if (use_frame) { width = this.frame_width(); height = this.frame_height(); }

      var svg = '<svg width="' + width + '" height="' + height + '" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">' +
                 elem.node().innerHTML +
                 '</svg>';

      if (file_format == "svg")
         return reconstruct(svg); // return SVG file as is

      var doctype = '<?xml version="1.0" standalone="no"?><!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">';

      var image = new Image();
      image.onload = function() {
         // if (options.result==="image") return JSROOT.CallBack(call_back, image);

         // console.log('GOT IMAGE', image.width, image.height);

         var canvas = document.createElement('canvas');
         canvas.width = image.width;
         canvas.height = image.height;
         var context = canvas.getContext('2d');
         context.drawImage(image, 0, 0);

         reconstruct(canvas.toDataURL('image/' + file_format));
      }

      image.onerror = function(arg) {
         console.log('IMAGE ERROR', arg);
         reconstruct(null);
      }

      image.src = 'data:image/svg+xml;base64,' + window.btoa(reEncode(doctype + svg));
   }


   TPadPainter.prototype.PadButtonClick = function(funcname) {

      if (funcname == "CanvasSnapShot") return this.SaveAs("png", true);

      if (funcname == "EnlargePad") return this.EnlargePad();

      if (funcname == "PadSnapShot") return this.SaveAs("png", false);

      if (funcname == "PadContextMenus") {

         d3.event.preventDefault();
         d3.event.stopPropagation();

         if (JSROOT.Painter.closeMenu()) return;

         var pthis = this, evnt = d3.event;

         JSROOT.Painter.createMenu(pthis, function(menu) {
            menu.add("header:Menus");

            if (pthis.iscan)
               menu.add("Canvas", "pad", pthis.ItemContextMenu);
            else
               menu.add("Pad", "pad", pthis.ItemContextMenu);

            if (pthis.frame_painter())
               menu.add("Frame", "frame", pthis.ItemContextMenu);

            var main = pthis.main_painter();

            if (main) {
               menu.add("X axis", "xaxis", pthis.ItemContextMenu);
               menu.add("Y axis", "yaxis", pthis.ItemContextMenu);
               if ((typeof main.Dimension === 'function') && (main.Dimension() > 1))
                  menu.add("Z axis", "zaxis", pthis.ItemContextMenu);
            }

            if (pthis.painters && (pthis.painters.length>0)) {
               menu.add("separator");
               var shown = [];
               for (var n=0;n<pthis.painters.length;++n) {
                  var pp = pthis.painters[n];
                  var obj = pp ? pp.GetObject() : null;
                  if (!obj || (shown.indexOf(obj)>=0)) continue;

                  var name = ('_typename' in obj) ? (obj._typename + "::") : "";
                  if ('fName' in obj) name += obj.fName;
                  if (name.length==0) name = "item" + n;
                  menu.add(name, n, pthis.ItemContextMenu);
               }
            }

            menu.show(evnt);
         });

         return;
      }

      // click automatically goes to all sub-pads
      // if any painter indicates that processing completed, it returns true
      var done = false;

      for (var i = 0; i < this.painters.length; ++i) {
         var pp = this.painters[i];

         if (typeof pp.PadButtonClick == 'function')
            pp.PadButtonClick(funcname);

         if (!done && (typeof pp.ButtonClick == 'function'))
            done = pp.ButtonClick(funcname);
      }
   }

   TPadPainter.prototype.FindButton = function(keyname) {
      var group = this.svg_layer("btns_layer", this.this_pad_name), found_func = "";
      if (!group.empty())
         group.selectAll("svg").each(function() {
            if (d3.select(this).attr("key") === keyname)
               found_func = d3.select(this).attr("name");
         });
      return found_func;
   }

   TPadPainter.prototype.toggleButtonsVisibility = function(action) {
      var group = this.svg_layer("btns_layer", this.this_pad_name),
          btn = group.select("[name='Toggle']");

      if (btn.empty()) return;

      var state = btn.property('buttons_state');

      if (btn.property('timout_handler')) {
         if (action!=='timeout') clearTimeout(btn.property('timout_handler'));
         btn.property('timout_handler', null);
      }

      var is_visible = false;
      switch(action) {
         case 'enable': is_visible = true; break;
         case 'enterbtn': return; // do nothing, just cleanup timeout
         case 'timeout': is_visible = false; break;
         case 'toggle':
            state = !state;
            btn.property('buttons_state', state);
            is_visible = state;
            break;
         case 'disable':
         case 'leavebtn':
            if (!state) btn.property('timout_handler', setTimeout(this.toggleButtonsVisibility.bind(this,'timeout'), 500));
            return;
      }

      group.selectAll('svg').each(function() {
         if (this===btn.node()) return;
         d3.select(this).style('display', is_visible ? "" : "none");
      });
   }

   TPadPainter.prototype.RemoveButtons = function() {
      var group = this.svg_layer("btns_layer", this.this_pad_name);
      if (!group.empty()) {
         group.selectAll("*").remove();
         group.property("nextx", null);
      }
   }

   TPadPainter.prototype.RemoveButtons = function() {
      var group = this.svg_layer("btns_layer", this.this_pad_name);
      if (!group.empty()) {
         group.selectAll("*").remove();
         group.property("nextx", null);
      }
   }

   TPadPainter.prototype.AddButton = function(_btn, _tooltip, _funcname, _keyname) {
      if (!JSROOT.gStyle.ToolBar) return;

      if (!this._buttons) this._buttons = [];
      // check if there are duplications

      for (var k=0;k<this._buttons.length;++k)
         if (this._buttons[k].funcname == _funcname) return;

      this._buttons.push({ btn: _btn, tooltip: _tooltip, funcname: _funcname, keyname: _keyname });

      var iscan = this.iscan || !this.has_canvas;
      if (!iscan && (_funcname.indexOf("Pad")!=0) && (_funcname !== "EnlargePad")) {
         var cp = this.canv_painter();
         if (cp && (cp!==this)) cp.AddButton(_btn, _tooltip, _funcname);
      }
   }

   TPadPainter.prototype.ShowButtons = function() {

      if (!this._buttons) return;

      var group = this.svg_layer("btns_layer", this.this_pad_name);
      if (group.empty()) return;

      // clean all previous buttons
      group.selectAll("*").remove();

      var iscan = this.iscan || !this.has_canvas, ctrl,
          x = group.property('leftside') ? this.ButtonSize(1.25) : 0, y = 0;

      if (this._fast_drawing) {
         ctrl = JSROOT.ToolbarIcons.CreateSVG(group, JSROOT.ToolbarIcons.circle, this.ButtonSize(), "EnlargePad");
         ctrl.attr("name", "Enlarge").attr("x", 0).attr("y", 0)
             // .property("buttons_state", (JSROOT.gStyle.ToolBar!=='popup'))
             .on("click", this.PadButtonClick.bind(this, "EnlargePad"));
      } else {
         ctrl = JSROOT.ToolbarIcons.CreateSVG(group, JSROOT.ToolbarIcons.rect, this.ButtonSize(), "Toggle tool buttons");

         ctrl.attr("name", "Toggle").attr("x", 0).attr("y", 0)
             .property("buttons_state", (JSROOT.gStyle.ToolBar!=='popup'))
             .on("click", this.toggleButtonsVisibility.bind(this, 'toggle'))
             .on("mouseenter", this.toggleButtonsVisibility.bind(this, 'enable'))
             .on("mouseleave", this.toggleButtonsVisibility.bind(this, 'disable'));

         for (var k=0;k<this._buttons.length;++k) {
            var item = this._buttons[k];

            var svg = JSROOT.ToolbarIcons.CreateSVG(group, item.btn, this.ButtonSize(),
                        item.tooltip + (iscan ? "" : (" on pad " + this.this_pad_name)) + (item.keyname ? " (keyshortcut " + item.keyname + ")" : ""));

            if (group.property('vertical'))
                svg.attr("x", y).attr("y", x);
            else
               svg.attr("x", x).attr("y", y);

            svg.attr("name", item.funcname)
               .style('display', (ctrl.property("buttons_state") ? '' : 'none'))
               .on("mouseenter", this.toggleButtonsVisibility.bind(this, 'enterbtn'))
               .on("mouseleave", this.toggleButtonsVisibility.bind(this, 'leavebtn'));

            if (item.keyname) svg.attr("key", item.keyname);

            svg.on("click", this.PadButtonClick.bind(this, item.funcname));

            x += this.ButtonSize(1.25);
         }
      }

      group.property("nextx", x);

      this.AlignBtns(group, this.pad_width(this.this_pad_name), this.pad_height(this.this_pad_name));

      if (group.property('vertical')) ctrl.attr("y", x);
      else if (!group.property('leftside')) ctrl.attr("x", x);
   }

   TPadPainter.prototype.AlignBtns = function(btns, width, height, svg) {
      var sz0 = this.ButtonSize(1.25), nextx = (btns.property('nextx') || 0) + sz0, btns_x, btns_y;
      if (btns.property('vertical')) {
          btns_x = btns.property('leftside') ? 2 : (width - sz0);
          btns_y = height - nextx;
      } else {
          btns_x = btns.property('leftside') ? 2 : (width - nextx);
          btns_y = height - sz0;
      }

      btns.attr("transform","translate("+btns_x+","+btns_y+")");
   }

   TPadPainter.prototype.GetCoordinate = function(pos) {
      var res = { x: 0, y: 0 };

      if (!pos) return res;

      function GetV(len, indx, dflt) {
         return (len.fArr && (len.fArr.length>indx)) ? len.fArr[indx] : dflt;
      }

      var w = this.pad_width(this.this_pad_name),
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


//   TPadPainter.prototype.DrawingReady = function(res_painter) {
//      var main = this.main_painter();
//      if (main && main.mode3d && typeof main.Render3D == 'function') main.Render3D(-2222);
//      TBasePainter.prototype.DrawingReady.call(this, res_painter);
//   }

   TPadPainter.prototype.DecodeOptions = function(opt) {
      var pad = this.GetObject();
      if (!pad) return;

      var d = new JSROOT.DrawOptions(opt);

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

   function drawPad(divid, pad, opt) {
      var painter = new TPadPainter(pad, false);
      painter.DecodeOptions(opt);

      painter.SetDivId(divid); // pad painter will be registered in the canvas painters list

      if (painter.svg_canvas().empty()) {
         painter.has_canvas = false;
         painter.this_pad_name = "";
      }

      painter.CreatePadSvg();

      if (painter.MatchObjectType("TPad") && (!painter.has_canvas || painter.HasObjectsToDraw())) {
         painter.AddButton(JSROOT.ToolbarIcons.camera, "Create PNG", "PadSnapShot");

         if ((painter.has_canvas && painter.HasObjectsToDraw()) || painter.enlarge_main('verify'))
            painter.AddButton(JSROOT.ToolbarIcons.circle, "Enlarge pad", "EnlargePad");

         if (JSROOT.gStyle.ContextMenu)
            painter.AddButton(JSROOT.ToolbarIcons.question, "Access context menus", "PadContextMenus");
      }

      // we select current pad, where all drawing is performed
      var prev_name = painter.has_canvas ? painter.CurrentPadName(painter.this_pad_name) : undefined;

      JSROOT.Painter.SelectActivePad({ pp: painter, active: false });

      // flag used to prevent immediate pad redraw during first draw
      painter.DrawPrimitives(0, function() {
         painter.ShowButtons();
         // we restore previous pad name
         painter.CurrentPadName(prev_name);
         painter.DrawingReady();
      });

      return painter;
   }

   // ==========================================================================================

   function TCanvasPainter(canvas) {
      // used for online canvas painter
      TPadPainter.call(this, canvas, true);
      this._websocket = null;
      this.tooltip_allowed = (JSROOT.gStyle.Tooltip > 0);
   }

   TCanvasPainter.prototype = Object.create(TPadPainter.prototype);

   TCanvasPainter.prototype.ChangeLayout = function(layout_kind, call_back) {
      var current = this.get_layout_kind();
      if (current == layout_kind) return JSROOT.CallBack(call_back, true);

      var origin = this.select_main('origin'),
          sidebar = origin.select('.side_panel'),
          main = this.select_main(), lst = [];

      while (main.node().firstChild)
         lst.push(main.node().removeChild(main.node().firstChild));

      if (!sidebar.empty()) JSROOT.cleanup(sidebar.node());

      this.set_layout_kind("simple"); // restore defaults
      origin.html(""); // cleanup origin

      if (layout_kind == 'simple') {
         main = origin;
         for (var k=0;k<lst.length;++k)
            main.node().appendChild(lst[k]);
         this.set_layout_kind(layout_kind);
         JSROOT.resize(main.node());
         return JSROOT.CallBack(call_back, true);
      }

      var pthis = this;

      JSROOT.AssertPrerequisites("jq2d", function() {

         var grid = new JSROOT.GridDisplay(origin.node(), layout_kind);

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
         for (var k=0;k<lst.length;++k)
            main.node().appendChild(lst[k]);

         pthis.set_layout_kind(layout_kind, ".central_panel");

         // remove reference to MDIDisplay, solves resize problem
         origin.property('mdi', null);

         // resize main drawing and let draw extras
         JSROOT.resize(main.node());

         JSROOT.CallBack(call_back, true);
      });
   }

   TCanvasPainter.prototype.ToggleProjection = function(kind, call_back) {
      delete this.proj_painter;

      if (kind) this.proj_painter = 1; // just indicator that drawing can be preformed

      if (this.ShowUI5ProjectionArea)
         return this.ShowUI5ProjectionArea(kind, call_back);

      var layout = 'simple';

      if (kind == "X") layout = 'vert2_31'; else
      if (kind == "Y") layout = 'horiz2_13';

      this.ChangeLayout(layout, call_back);
   }

   TCanvasPainter.prototype.DrawProjection = function(kind,hist) {
      if (!this.proj_painter) return; // ignore drawing if projection not configured

      if (this.proj_painter === 1) {

         var canv = JSROOT.Create("TCanvas"), pthis = this, pad = this.root_pad(), main = this.main_painter(), drawopt;

         if (kind == "X") {
            canv.fLeftMargin = pad.fLeftMargin;
            canv.fRightMargin = pad.fRightMargin;
            canv.fLogx = main.logx ? 1 : 0;
            canv.fUxmin = main.logx ? JSROOT.log10(main.scale_xmin) : main.scale_xmin;
            canv.fUxmax = main.logx ? JSROOT.log10(main.scale_xmax) : main.scale_xmax;
            drawopt = "fixframe";
         } else {
            canv.fBottomMargin = pad.fBottomMargin;
            canv.fTopMargin = pad.fTopMargin;
            canv.fLogx = main.logy ? 1 : 0;
            canv.fUxmin = main.logy ? JSROOT.log10(main.scale_ymin) : main.scale_ymin;
            canv.fUxmax = main.logy ? JSROOT.log10(main.scale_ymax) : main.scale_ymax;
            drawopt = "rotate";
         }

         canv.fPrimitives.Add(hist, "hist");

         if (this.DrawInUI5ProjectionArea) {
            // copy frame attributes
            this.DrawInUI5ProjectionArea(canv, drawopt, function(painter) { pthis.proj_painter = painter; })
         } else {
            this.DrawInSidePanel(canv, drawopt, function(painter) { pthis.proj_painter = painter; })
         }
      } else {
         var hp = this.proj_painter.main_painter();
         if (hp) hp.UpdateObject(hist, "hist");
         this.proj_painter.RedrawPad();
      }
   }

   TCanvasPainter.prototype.DrawInSidePanel = function(canv, opt, call_back) {
      var side = this.select_main('origin').select(".side_panel");
      if (side.empty()) return JSROOT.CallBack(call_back, null);
      JSROOT.draw(side.node(), canv, opt, call_back);
   }

   TCanvasPainter.prototype.ShowMessage = function(msg) {
      JSROOT.progress(msg, 7000);
   }

   /// function called when canvas menu item Save is called
   TCanvasPainter.prototype.SaveCanvasAsFile = function(fname) {
      var pthis = this, pnt = fname.indexOf(".");
      this.CreateImage(fname.substr(pnt+1), function(res) {
         pthis.SendWebsocket("SAVE:" + fname + ":" + res);
      })
   }

   TCanvasPainter.prototype.SendSaveCommand = function(fname) {
      this.SendWebsocket("PRODUCE:" + fname);
   }

   TCanvasPainter.prototype.SendWebsocket = function(msg, chid) {
      if (this._websocket)
         this._websocket.Send(msg, chid);
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
      console.log('Use websocket', this._websocket.key);
      this._websocket.SetReceiver(this);
      this._websocket.Connect(href);
   }

   TCanvasPainter.prototype.WindowBeforeUnloadHanlder = function() {
      // when window closed, close socket
      this.CloseWebsocket(true);
   }

   TCanvasPainter.prototype.OnWebsocketOpened = function(handle) {
      // indicate that we are ready to recieve any following commands
   }

   TCanvasPainter.prototype.OnWebsocketClosed = function(handle) {
      JSROOT.CloseCurrentWindow();
   }

   TCanvasPainter.prototype.OnWebsocketMsg = function(handle, msg) {
      console.log("GET MSG " + msg.substr(0,30));

      if (msg == "CLOSE") {
         this.OnWebsocketClosed();
         this.CloseWebsocket(true);
      } else if (msg.substr(0,5)=='SNAP:') {
         msg = msg.substr(5);
         var p1 = msg.indexOf(":"),
             snapid = msg.substr(0,p1),
             snap = JSROOT.parse(msg.substr(p1+1));
         this.RedrawPadSnap(snap, function() {
            handle.Send("SNAPDONE:" + snapid); // send ready message back when drawing completed
         });
      } else if (msg.substr(0,4)=='JSON') {
         var obj = JSROOT.parse(msg.substr(4));
         // console.log("get JSON ", msg.length-4, obj._typename);
         this.RedrawObject(obj);

      } else if (msg.substr(0,5)=='MENU:') {
         // this is container with object id and list of menu items
         var lst = JSROOT.parse(msg.substr(5));
         // console.log("get MENUS ", typeof lst, 'nitems', lst.length, msg.length-4);
         if (typeof this._getmenu_callback == 'function')
            this._getmenu_callback(lst);
      } else if (msg.substr(0,4)=='CMD:') {
         msg = msg.substr(4);
         var p1 = msg.indexOf(":"),
             cmdid = msg.substr(0,p1),
             cmd = msg.substr(p1+1),
             reply = "REPLY:" + cmdid + ":";
         if ((cmd == "SVG") || (cmd == "PNG") || (cmd == "JPEG")) {
            this.CreateImage(cmd.toLowerCase(), function(res) {
               handle.Send(reply + res);
            });
         } else if (cmd.indexOf("ADDPANEL:") == 0) {
            var relative_path = cmd.substr(9);
            console.log('request panel = ' + relative_path);
            if (!this.ShowUI5Panel) {
               handle.Send(reply + "false");
            } else {

               var conn = new JSROOT.WebWindowHandle(handle.kind);

               // set interim receiver until first message arrives
               conn.SetReceiver({
                  cpainter: this,

                  OnWebsocketOpened: function(hhh) {
                     console.log('Panel socket connected');
                  },

                  OnWebsocketMsg: function(panel_handle, msg) {

                     var panel_name = (msg.indexOf("SHOWPANEL:")==0) ? msg.substr(10) : "";
                     console.log('Panel get message ' + msg + " show " + panel_name);

                     this.cpainter.ShowUI5Panel(panel_name, panel_handle, function(res) {
                        handle.Send(reply + (res ? "true" : "false"));
                     });
                  },

                  OnWebsocketClosed: function(hhh) {
                     // if connection failed,
                     handle.Send(reply + "false");
                  },

                  OnWebsocketError: function(hhh) {
                     // if connection failed,
                     handle.Send(reply + "false");
                  }

               });

               var addr = handle.href;
               if (relative_path.indexOf("../")==0) {
                  var ddd = addr.lastIndexOf("/",addr.length-2);
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
         var kind = msg[1],
             hist = JSROOT.parse(msg.substr(7));
         this.DrawProjection(kind, hist);
      } else if (msg.substr(0,5)=='SHOW:') {
         var that = msg.substr(5),
             on = that[that.length-1] == '1';
         this.ShowSection(that.substr(0,that.length-2), on);
      } else {
         console.log("unrecognized msg len:" + msg.length + " msg:" + msg.substr(0,20));
      }
   }

   TCanvasPainter.prototype.ShowSection = function(that, on) {
      switch(that) {
         case "Menu": break;
         case "StatusBar": break;
         case "Editor": break;
         case "ToolBar": break;
         case "ToolTips": this.SetTooltipAllowed(on); break;
      }
   }

   JSROOT.TCanvasStatusBits = {
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

   TCanvasPainter.prototype.CompeteCanvasSnapDrawing = function() {
      if (!this.pad) return;

      if (document) document.title = this.pad.fTitle;

      if (this._all_sections_showed) return;
      this._all_sections_showed = true;
      this.ShowSection("Menu", this.pad.TestBit(JSROOT.TCanvasStatusBits.kMenuBar));
      this.ShowSection("StatusBar", this.pad.TestBit(JSROOT.TCanvasStatusBits.kShowEventStatus));
      this.ShowSection("ToolBar", this.pad.TestBit(JSROOT.TCanvasStatusBits.kShowToolBar));
      this.ShowSection("Editor", this.pad.TestBit(JSROOT.TCanvasStatusBits.kShowEditor));
      this.ShowSection("ToolTips", this.pad.TestBit(JSROOT.TCanvasStatusBits.kShowToolTips));
   }

   TCanvasPainter.prototype.HasEventStatus = function() {
      return this.has_event_status;
   }

   function drawCanvas(divid, can, opt) {
      var nocanvas = !can;
      if (nocanvas) {
         console.log("No canvas specified");
         return null;
         // can = JSROOT.Create("ROOT::Experimental::TCanvas");
      }

      var painter = new TCanvasPainter(can);
      painter.normal_canvas = !nocanvas;

      painter.SetDivId(divid, -1); // just assign id
      painter.CreateCanvasSvg(0);
      painter.SetDivId(divid);  // now add to painters list

      painter.AddButton(JSROOT.ToolbarIcons.camera, "Create PNG", "CanvasSnapShot", "Ctrl PrintScreen");
      if (JSROOT.gStyle.ContextMenu)
         painter.AddButton(JSROOT.ToolbarIcons.question, "Access context menus", "PadContextMenus");

      if (painter.enlarge_main('verify'))
         painter.AddButton(JSROOT.ToolbarIcons.circle, "Enlarge canvas", "EnlargePad");

      JSROOT.Painter.SelectActivePad({ pp: painter, active: false });

      painter.DrawPrimitives(0, function() {
         painter.ShowButtons();
         painter.DrawingReady();
      });

      return painter;
   }

   // JSROOT.addDrawFunc({ name: "ROOT::Experimental::RPadDisplayItem", icon: "img_canvas", func: drawPad, opt: "" });

   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RHistDrawable<1>", icon: "img_histo1d", prereq: "v7hist", func: "JSROOT.v7.drawHist1", opt: "" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RHistDrawable<2>", icon: "img_histo2d", prereq: "v7hist", func: "JSROOT.v7.drawHist2", opt: "" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RText", icon: "img_text", prereq: "v7more", func: "JSROOT.v7.drawText", opt: "", direct: true, csstype: "text" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RLine", icon: "img_graph", prereq: "v7more", func: "JSROOT.v7.drawLine", opt: "", direct: true, csstype: "line" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RBox", icon: "img_graph", prereq: "v7more", func: "JSROOT.v7.drawBox", opt: "", direct: true, csstype: "box" });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::RMarker", icon: "img_graph", prereq: "v7more", func: "JSROOT.v7.drawMarker", opt: "", direct: true, csstype: "marker" });

   JSROOT.v7.TAxisPainter = TAxisPainter;
   JSROOT.v7.TFramePainter = TFramePainter;
   JSROOT.v7.TPadPainter = TPadPainter;
   JSROOT.v7.TCanvasPainter = TCanvasPainter;
   JSROOT.v7.drawFrame = drawFrame;
   JSROOT.v7.drawPad = drawPad;
   JSROOT.v7.drawCanvas = drawCanvas;

   return JSROOT;

}));