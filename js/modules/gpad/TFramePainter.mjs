/// TCanvas painting

import { gStyle, settings, isBatchMode, browser } from '../core.mjs';

import { select as d3_select, pointer as d3_pointer, pointers as d3_pointers, drag as d3_drag } from '../d3.mjs';

import { getActivePad, ObjectPainter } from '../base/ObjectPainter.mjs';

import { getSvgLineStyle } from '../base/TAttLineHandler.mjs';

import { EAxisBits, TAxisPainter } from './TAxisPainter.mjs';

import { getElementRect, getAbsPosInCanvas } from '../base/BasePainter.mjs';

import { FontHandler } from '../base/FontHandler.mjs';

import { createMenu, closeMenu } from '../gui/menu.mjs';

import { detectRightButton, injectStyle } from '../gui/utils.mjs';

function setPainterTooltipEnabled(painter, on) {
   if (!painter) return;

   let fp = painter.getFramePainter();
   if (fp && typeof fp.setTooltipEnabled == 'function') {
      fp.setTooltipEnabled(on);
      fp.processFrameTooltipEvent(null);
   }
   // this is 3D control object
   if (painter.control && (typeof painter.control.setTooltipEnabled == 'function'))
      painter.control.setTooltipEnabled(on);
}

/** @summary Add drag for interactive rectangular elements for painter */
function addDragHandler(_painter, arg) {
   if (!settings.MoveResize || isBatchMode()) return;

   let painter = _painter, drag_rect = null, pp = painter.getPadPainter();
   if (pp && pp._fast_drawing) return;

   function makeResizeElements(group, handler) {
      function addElement(cursor, d) {
         let clname = "js_" + cursor.replace(/[-]/g, '_'),
            elem = group.select('.' + clname);
         if (elem.empty()) elem = group.append('path').classed(clname, true);
         elem.style('opacity', 0).style('cursor', cursor).attr('d', d);
         if (handler) elem.call(handler);
      }

      addElement("nw-resize", "M2,2h15v-5h-20v20h5Z");
      addElement("ne-resize", `M${arg.width - 2},2h-15v-5h20v20h-5 Z`);
      addElement("sw-resize", `M2,${arg.height - 2}h15v5h-20v-20h5Z`);
      addElement("se-resize", `M${arg.width - 2},${arg.height - 2}h-15v5h20v-20h-5Z`);

      if (!arg.no_change_x) {
         addElement("w-resize", `M-3,18h5v${Math.max(0, arg.height - 2 * 18)}h-5Z`);
         addElement("e-resize", `M${arg.width + 3},18h-5v${Math.max(0, arg.height - 2 * 18)}h5Z`);
      }
      if (!arg.no_change_y) {
         addElement("n-resize", `M18,-3v5h${Math.max(0, arg.width - 2 * 18)}v-5Z`);
         addElement("s-resize", `M18,${arg.height + 3}v-5h${Math.max(0, arg.width - 2 * 18)}v5Z`);
      }
   }

   const complete_drag = (newx, newy, newwidth, newheight) => {
      drag_rect.style("cursor", "auto");

      if (!painter.draw_g) {
         drag_rect.remove();
         drag_rect = null;
         return false;
      }

      let oldx = arg.x, oldy = arg.y;

      if (arg.minwidth && newwidth < arg.minwidth) newwidth = arg.minwidth;
      if (arg.minheight && newheight < arg.minheight) newheight = arg.minheight;

      let change_size = (newwidth !== arg.width) || (newheight !== arg.height),
          change_pos = (newx !== oldx) || (newy !== oldy);

      arg.x = newx; arg.y = newy; arg.width = newwidth; arg.height = newheight;

      painter.draw_g.attr("transform", `translate(${newx},${newy})`);

      drag_rect.remove();
      drag_rect = null;

      setPainterTooltipEnabled(painter, true);

      makeResizeElements(painter.draw_g);

      if (change_size || change_pos) {
         if (change_size && ('resize' in arg)) arg.resize(newwidth, newheight);
         if (change_pos && ('move' in arg)) arg.move(newx, newy, newx - oldxx, newy - oldy);

         if (change_size || change_pos) {
            if ('obj' in arg) {
               let rect = pp.getPadRect();
               arg.obj.fX1NDC = newx / rect.width;
               arg.obj.fX2NDC = (newx + newwidth) / rect.width;
               arg.obj.fY1NDC = 1 - (newy + newheight) / rect.height;
               arg.obj.fY2NDC = 1 - newy / rect.height;
               arg.obj.modified_NDC = true; // indicate that NDC was interactively changed, block in updated
            }
            if ('redraw' in arg) arg.redraw(arg);
         }
      }

      return change_size || change_pos;
   };

   let drag_move = d3_drag().subject(Object);

   drag_move
      .on("start", function(evnt) {
         if (detectRightButton(evnt.sourceEvent)) return;

         closeMenu(); // close menu

         setPainterTooltipEnabled(painter, false); // disable tooltip

         evnt.sourceEvent.preventDefault();
         evnt.sourceEvent.stopPropagation();

         let pad_rect = pp.getPadRect();

         let handle = {
            x: arg.x, y: arg.y, width: arg.width, height: arg.height,
            acc_x1: arg.x, acc_y1: arg.y,
            pad_w: pad_rect.width - arg.width,
            pad_h: pad_rect.height - arg.height,
            drag_tm: new Date(),
            path: `v${arg.height}h${arg.width}v${-arg.height}z`
         };

         drag_rect = d3_select(painter.draw_g.node().parentNode).append("path")
            .classed("zoom", true)
            .attr("d", `M${handle.acc_x1},${handle.acc_y1}${handle.path}`)
            .style("cursor", "move")
            .style("pointer-events", "none") // let forward double click to underlying elements
            .property('drag_handle', handle);


      }).on("drag", function(evnt) {
         if (!drag_rect) return;

         evnt.sourceEvent.preventDefault();
         evnt.sourceEvent.stopPropagation();

         let handle = drag_rect.property('drag_handle');

         if (!arg.no_change_x)
            handle.acc_x1 += evnt.dx;
         if (!arg.no_change_y)
            handle.acc_y1 += evnt.dy;

         handle.x = Math.min(Math.max(handle.acc_x1, 0), handle.pad_w);
         handle.y = Math.min(Math.max(handle.acc_y1, 0), handle.pad_h);

         drag_rect.attr("d", `M${handle.x},${handle.y}${handle.path}`);

      }).on("end", function(evnt) {
         if (!drag_rect) return;

         evnt.sourceEvent.preventDefault();

         let handle = drag_rect.property('drag_handle');

         if (complete_drag(handle.x, handle.y, arg.width, arg.height) === false) {
            let spent = (new Date()).getTime() - handle.drag_tm.getTime();
            if (arg.ctxmenu && (spent > 600) && painter.showContextMenu) {
               let rrr = resize_se.node().getBoundingClientRect();
               painter.showContextMenu('main', { clientX: rrr.left, clientY: rrr.top });
            } else if (arg.canselect && (spent <= 600)) {
               let pp = painter.getPadPainter();
               if (pp) pp.selectObjectPainter(painter);
            }
         }
      });

   let drag_resize = d3_drag().subject(Object);

   drag_resize
      .on("start", function(evnt) {
         if (detectRightButton(evnt.sourceEvent)) return;

         evnt.sourceEvent.stopPropagation();
         evnt.sourceEvent.preventDefault();

         setPainterTooltipEnabled(painter, false); // disable tooltip

         let pad_rect = pp.getPadRect();

         let handle = {
            x: arg.x, y: arg.y, width: arg.width, height: arg.height,
            acc_x1: arg.x, acc_y1: arg.y,
            pad_w: pad_rect.width,
            pad_h: pad_rect.height
         };

         handle.acc_x2 = handle.acc_x1 + arg.width;
         handle.acc_y2 = handle.acc_y1 + arg.height;

         drag_rect = d3_select(painter.draw_g.node().parentNode)
            .append("rect")
            .classed("zoom", true)
            .style("cursor", d3_select(this).style("cursor"))
            .attr("x", handle.acc_x1)
            .attr("y", handle.acc_y1)
            .attr("width", handle.acc_x2 - handle.acc_x1)
            .attr("height", handle.acc_y2 - handle.acc_y1)
            .property('drag_handle', handle);

      }).on("drag", function(evnt) {
         if (!drag_rect) return;

         evnt.sourceEvent.preventDefault();
         evnt.sourceEvent.stopPropagation();

         let handle = drag_rect.property('drag_handle'),
            dx = evnt.dx, dy = evnt.dy, elem = d3_select(this);

         if (arg.no_change_x) dx = 0;
         if (arg.no_change_y) dy = 0;

         if (elem.classed('js_nw_resize')) { handle.acc_x1 += dx; handle.acc_y1 += dy; }
         else if (elem.classed('js_ne_resize')) { handle.acc_x2 += dx; handle.acc_y1 += dy; }
         else if (elem.classed('js_sw_resize')) { handle.acc_x1 += dx; handle.acc_y2 += dy; }
         else if (elem.classed('js_se_resize')) { handle.acc_x2 += dx; handle.acc_y2 += dy; }
         else if (elem.classed('js_w_resize')) { handle.acc_x1 += dx; }
         else if (elem.classed('js_n_resize')) { handle.acc_y1 += dy; }
         else if (elem.classed('js_e_resize')) { handle.acc_x2 += dx; }
         else if (elem.classed('js_s_resize')) { handle.acc_y2 += dy; }

         let x1 = Math.max(0, handle.acc_x1), x2 = Math.min(handle.acc_x2, handle.pad_w),
             y1 = Math.max(0, handle.acc_y1), y2 = Math.min(handle.acc_y2, handle.pad_h);

         handle.x = Math.min(x1, x2);
         handle.y = Math.min(y1, y2);
         handle.width = Math.abs(x2 - x1);
         handle.height = Math.abs(y2 - y1);

         drag_rect.attr("x", handle.x).attr("y", handle.y).attr("width", handle.width).attr("height", handle.height);

      }).on("end", function(evnt) {
         if (!drag_rect) return;
         evnt.sourceEvent.preventDefault();

         let handle = drag_rect.property('drag_handle');

         complete_drag(handle.x, handle.y, handle.width, handle.height);
      });

   if (!arg.only_resize)
      painter.draw_g.style("cursor", "move").call(drag_move);

   if (!arg.only_move)
      makeResizeElements(painter.draw_g, drag_resize);
}

const TooltipHandler = {

   /** @desc only canvas info_layer can be used while other pads can overlay
     * @returns layer where frame tooltips are shown */
   hints_layer() {
      let pp = this.getCanvPainter();
      return pp ? pp.getLayerSvg("info_layer") : d3_select(null);
   },

   /** @returns true if tooltip is shown, use to prevent some other action */
   isTooltipShown() {
      if (!this.tooltip_enabled || !this.isTooltipAllowed()) return false;
      let hintsg = this.hints_layer().select(".objects_hints");
      return hintsg.empty() ? false : hintsg.property("hints_pad") == this.getPadName();
   },

   setTooltipEnabled(enabled) {
      if (enabled !== undefined) this.tooltip_enabled = enabled;
   },

   /** @summary central function which let show selected hints for the object */
   processFrameTooltipEvent(pnt, evnt) {
      if (pnt && pnt.handler) {
         // special use of interactive handler in the frame painter
         let rect = this.draw_g ? this.draw_g.select(".main_layer") : null;
         if (!rect || rect.empty()) {
            pnt = null; // disable
         } else if (pnt.touch && evnt) {
            let pos = d3_pointers(evnt, rect.node());
            pnt = (pos && pos.length == 1) ? { touch: true, x: pos[0][0], y: pos[0][1] } : null;
         } else if (evnt) {
            let pos = d3_pointer(evnt, rect.node());
            pnt = { touch: false, x: pos[0], y: pos[1] };
         }
      }

      let hints = [], nhints = 0, nexact = 0, maxlen = 0, lastcolor1 = 0, usecolor1 = false,
         textheight = 11, hmargin = 3, wmargin = 3, hstep = 1.2,
         frame_rect = this.getFrameRect(),
         pp = this.getPadPainter(),
         pad_width = pp.getPadWidth(),
         font = new FontHandler(160, textheight),
         disable_tootlips = !this.isTooltipAllowed() || !this.tooltip_enabled;

      if (pnt && disable_tootlips) pnt.disabled = true; // indicate that highlighting is not required
      if (pnt) pnt.painters = true; // get also painter

      // collect tooltips from pad painter - it has list of all drawn objects
      if (pp) hints = pp.processPadTooltipEvent(pnt);

      if (pnt && pnt.touch) textheight = 15;

      for (let n = 0; n < hints.length; ++n) {
         let hint = hints[n];
         if (!hint) continue;

         if (hint.painter && (hint.user_info !== undefined))
            hint.painter.provideUserTooltip(hint.user_info);

         if (!hint.lines || (hint.lines.length === 0)) {
            hints[n] = null; continue;
         }

         // check if fully duplicated hint already exists
         for (let k = 0; k < n; ++k) {
            let hprev = hints[k], diff = false;
            if (!hprev || (hprev.lines.length !== hint.lines.length)) continue;
            for (let l = 0; l < hint.lines.length && !diff; ++l)
               if (hprev.lines[l] !== hint.lines[l]) diff = true;
            if (!diff) { hints[n] = null; break; }
         }
         if (!hints[n]) continue;

         nhints++;

         if (hint.exact) nexact++;

         for (let l = 0; l < hint.lines.length; ++l)
            maxlen = Math.max(maxlen, hint.lines[l].length);

         hint.height = Math.round(hint.lines.length * textheight * hstep + 2 * hmargin - textheight * (hstep - 1));

         if ((hint.color1 !== undefined) && (hint.color1 !== 'none')) {
            if ((lastcolor1 !== 0) && (lastcolor1 !== hint.color1)) usecolor1 = true;
            lastcolor1 = hint.color1;
         }
      }

      let layer = this.hints_layer(),
          hintsg = layer.select(".objects_hints"), // group with all tooltips
          title = "", name = "", info = "",
          hint = null, best_dist2 = 1e10, best_hint = null, show_only_best = nhints > 15,
          coordinates = pnt ? Math.round(pnt.x) + "," + Math.round(pnt.y) : "";

      // try to select hint with exact match of the position when several hints available
      for (let k = 0; k < (hints ? hints.length : 0); ++k) {
         if (!hints[k]) continue;
         if (!hint) hint = hints[k];

         // select exact hint if this is the only one
         if (hints[k].exact && (nexact < 2) && (!hint || !hint.exact)) { hint = hints[k]; break; }

         if (!pnt || (hints[k].x === undefined) || (hints[k].y === undefined)) continue;

         let dist2 = (pnt.x - hints[k].x) * (pnt.x - hints[k].x) + (pnt.y - hints[k].y) * (pnt.y - hints[k].y);
         if (dist2 < best_dist2) { best_dist2 = dist2; best_hint = hints[k]; }
      }

      if ((!hint || !hint.exact) && (best_dist2 < 400)) hint = best_hint;

      if (hint) {
         name = (hint.lines && hint.lines.length > 1) ? hint.lines[0] : hint.name;
         title = hint.title || "";
         info = hint.line;
         if (!info && hint.lines) info = hint.lines.slice(1).join(' ');
      }

      this.showObjectStatus(name, title, info, coordinates);


      // end of closing tooltips
      if (!pnt || disable_tootlips || (hints.length === 0) || (maxlen === 0) || (show_only_best && !best_hint)) {
         hintsg.remove();
         return;
      }

      // we need to set pointer-events=none for all elements while hints
      // placed in front of so-called interactive rect in frame, used to catch mouse events

      if (hintsg.empty())
         hintsg = layer.append("svg:g")
            .attr("class", "objects_hints")
            .style("pointer-events", "none");

      let frame_shift = { x: 0, y: 0 }, trans = frame_rect.transform || "";
      if (!pp.iscan) {
         frame_shift = getAbsPosInCanvas(this.getPadSvg(), frame_shift);
         trans = "translate(" + frame_shift.x + "," + frame_shift.y + ") " + trans;
      }

      // copy transform attributes from frame itself
      hintsg.attr("transform", trans)
         .property("last_point", pnt)
         .property("hints_pad", this.getPadName());

      let viewmode = hintsg.property('viewmode') || "",
         actualw = 0, posx = pnt.x + frame_rect.hint_delta_x;

      if (show_only_best || (nhints == 1)) {
         viewmode = "single";
         posx += 15;
      } else {
         // if there are many hints, place them left or right

         let bleft = 0.5, bright = 0.5;

         if (viewmode == "left")
            bright = 0.7;
         else if (viewmode == "right")
            bleft = 0.3;

         if (posx <= bleft * frame_rect.width) {
            viewmode = "left";
            posx = 20;
         } else if (posx >= bright * frame_rect.width) {
            viewmode = "right";
            posx = frame_rect.width - 60;
         } else {
            posx = hintsg.property('startx');
         }
      }

      if (viewmode !== hintsg.property('viewmode')) {
         hintsg.property('viewmode', viewmode);
         hintsg.selectAll("*").remove();
      }

      let curry = 10, // normal y coordinate
          gapy = 10,  // y coordinate, taking into account all gaps
          gapminx = -1111, gapmaxx = -1111,
          minhinty = -frame_shift.y,
          cp = this.getCanvPainter(),
          maxhinty = cp.getPadHeight() - frame_rect.y - frame_shift.y;

      const FindPosInGap = y => {
         for (let n = 0; (n < hints.length) && (y < maxhinty); ++n) {
            let hint = hints[n];
            if (!hint) continue;
            if ((hint.y >= y - 5) && (hint.y <= y + hint.height + 5)) {
               y = hint.y + 10;
               n = -1;
            }
         }
         return y;
      };

      for (let n = 0; n < hints.length; ++n) {
         let hint = hints[n],
            group = hintsg.select(".painter_hint_" + n);

         if (show_only_best && (hint !== best_hint)) hint = null;

         if (hint === null) {
            group.remove();
            continue;
         }

         let was_empty = group.empty();

         if (was_empty)
            group = hintsg.append("svg:svg")
               .attr("class", "painter_hint_" + n)
               .attr('opacity', 0) // use attribute, not style to make animation with d3.transition()
               .style('overflow', 'hidden')
               .style("pointer-events", "none");

         if (viewmode == "single") {
            curry = pnt.touch ? (pnt.y - hint.height - 5) : Math.min(pnt.y + 15, maxhinty - hint.height - 3) + frame_rect.hint_delta_y;
         } else {
            gapy = FindPosInGap(gapy);
            if ((gapminx === -1111) && (gapmaxx === -1111)) gapminx = gapmaxx = hint.x;
            gapminx = Math.min(gapminx, hint.x);
            gapmaxx = Math.min(gapmaxx, hint.x);
         }

         group.attr("x", posx)
            .attr("y", curry)
            .property("curry", curry)
            .property("gapy", gapy);

         curry += hint.height + 5;
         gapy += hint.height + 5;

         if (!was_empty)
            group.selectAll("*").remove();

         group.attr("width", 60)
            .attr("height", hint.height);

         let r = group.append("rect")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", 60)
            .attr("height", hint.height)
            .style("fill", "lightgrey")
            .style("pointer-events", "none");

         if (nhints > 1) {
            let col = usecolor1 ? hint.color1 : hint.color2;
            if (col && (col !== 'none'))
               r.style("stroke", col);
         }
         r.attr("stroke-width", hint.exact ? 3 : 1);

         for (let l = 0; l < (hint.lines ? hint.lines.length : 0); l++)
            if (hint.lines[l] !== null) {
               let txt = group.append("svg:text")
                  .attr("text-anchor", "start")
                  .attr("x", wmargin)
                  .attr("y", hmargin + l * textheight * hstep)
                  .attr("dy", ".8em")
                  .style("fill", "black")
                  .style("pointer-events", "none")
                  .call(font.func)
                  .text(hint.lines[l]);

               let box = getElementRect(txt, 'bbox');

               actualw = Math.max(actualw, box.width);
            }

         function translateFn() {
            // We only use 'd', but list d,i,a as params just to show can have them as params.
            // Code only really uses d and t.
            return function(/*d, i, a*/) {
               return function(t) {
                  return t < 0.8 ? "0" : (t - 0.8) * 5;
               };
            };
         }

         if (was_empty)
            if (settings.TooltipAnimation > 0)
               group.transition().duration(settings.TooltipAnimation).attrTween("opacity", translateFn());
            else
               group.attr('opacity', 1);
      }

      actualw += 2 * wmargin;

      let svgs = hintsg.selectAll("svg");

      if ((viewmode == "right") && (posx + actualw > frame_rect.width - 20)) {
         posx = frame_rect.width - actualw - 20;
         svgs.attr("x", posx);
      }

      if ((viewmode == "single") && (posx + actualw > pad_width - frame_rect.x) && (posx > actualw + 20)) {
         posx -= (actualw + 20);
         svgs.attr("x", posx);
      }

      // if gap not very big, apply gapy coordinate to open view on the histogram
      if ((viewmode !== "single") && (gapy < maxhinty) && (gapy !== curry)) {
         if ((gapminx <= posx + actualw + 5) && (gapmaxx >= posx - 5))
            svgs.attr("y", function() { return d3_select(this).property('gapy'); });
      } else if ((viewmode !== 'single') && (curry > maxhinty)) {
         let shift = Math.max((maxhinty - curry - 10), minhinty);
         if (shift < 0)
            svgs.attr("y", function() { return d3_select(this).property('curry') + shift; });
      }

      if (actualw > 10)
         svgs.attr("width", actualw)
            .select('rect').attr("width", actualw);

      hintsg.property('startx', posx);

      if (cp._highlight_connect && (typeof cp.processHighlightConnect == 'function'))
         cp.processHighlightConnect(hints);
   },

   /** @summary Assigns tooltip methods */
   assign(painter) {
      Object.assign(painter, this, { tooltip_enabled: true });
   }

} // TooltipHandler


const FrameInteractive = {

   addBasicInteractivity() {

      TooltipHandler.assign(this);

      if (!this._frame_rotate && !this._frame_fixpos)
         addDragHandler(this, { obj: this, x: this._frame_x, y: this._frame_y, width: this.getFrameWidth(), height: this.getFrameHeight(),
                                only_resize: true, minwidth: 20, minheight: 20, redraw: () => this.sizeChanged() });

      injectStyle(`
.jsroot rect.h1bin { stroke: #4572A7; fill: #4572A7; opacity: 0; }
.jsroot rect.zoom { stroke: steelblue; fill-opacity: 0.1; }
.jsroot path.zoom { stroke: steelblue; fill-opacity: 0.1; }
.jsroot svg:not(:root) { overflow: hidden; }`, this.draw_g.node());

      let main_svg = this.draw_g.select(".main_layer");

      main_svg.style("pointer-events","visibleFill")
              .property('handlers_set', 0);

      let pp = this.getPadPainter(),
          handlers_set = (pp && pp._fast_drawing) ? 0 : 1;

      if (main_svg.property('handlers_set') != handlers_set) {
         let close_handler = handlers_set ? this.processFrameTooltipEvent.bind(this, null) : null,
             mouse_handler = handlers_set ? this.processFrameTooltipEvent.bind(this, { handler: true, touch: false }) : null;

         main_svg.property('handlers_set', handlers_set)
                 .on('mouseenter', mouse_handler)
                 .on('mousemove', mouse_handler)
                 .on('mouseleave', close_handler);

         if (browser.touches) {
            let touch_handler = handlers_set ? this.processFrameTooltipEvent.bind(this, { handler: true, touch: true }) : null;

            main_svg.on("touchstart", touch_handler)
                    .on("touchmove", touch_handler)
                    .on("touchend", close_handler)
                    .on("touchcancel", close_handler);
         }
      }

      main_svg.attr("x", 0)
              .attr("y", 0)
              .attr("width", this.getFrameWidth())
              .attr("height", this.getFrameHeight());

      let hintsg = this.hints_layer().select(".objects_hints");
      // if tooltips were visible before, try to reconstruct them after short timeout
      if (!hintsg.empty() && this.isTooltipAllowed() && (hintsg.property("hints_pad") == this.getPadName()))
         setTimeout(this.processFrameTooltipEvent.bind(this, hintsg.property('last_point'), null), 10);
   },

   /** @summary Add interactive handlers */
   addFrameInteractivity(for_second_axes) {

      let pp = this.getPadPainter(),
          svg = this.getFrameSvg();
      if ((pp && pp._fast_drawing) || svg.empty())
         return Promise.resolve(this);

      if (for_second_axes) {

         // add extra handlers for second axes
         let svg_x2 = svg.selectAll(".x2axis_container"),
             svg_y2 = svg.selectAll(".y2axis_container");
         if (settings.ContextMenu) {
            svg_x2.on("contextmenu", evnt => this.showContextMenu("x2", evnt));
            svg_y2.on("contextmenu", evnt => this.showContextMenu("y2", evnt));
         }
         svg_x2.on("mousemove", evnt => this.showAxisStatus("x2", evnt));
         svg_y2.on("mousemove", evnt => this.showAxisStatus("y2", evnt));
         return Promise.resolve(this);
      }

      let svg_x = svg.selectAll(".xaxis_container"),
          svg_y = svg.selectAll(".yaxis_container");

      this.can_zoom_x = this.can_zoom_y = settings.Zooming;

      if (pp && pp.options) {
         if (pp.options.NoZoomX) this.can_zoom_x = false;
         if (pp.options.NoZoomY) this.can_zoom_y = false;
      }

      if (!svg.property('interactive_set')) {
         this.addFrameKeysHandler();

         this.last_touch = new Date(0);
         this.zoom_kind = 0; // 0 - none, 1 - XY, 2 - only X, 3 - only Y, (+100 for touches)
         this.zoom_rect = null;
         this.zoom_origin = null;  // original point where zooming started
         this.zoom_curr = null;    // current point for zooming
         this.touch_cnt = 0;
      }

      if (settings.Zooming && !this.projection) {
         if (settings.ZoomMouse) {
            svg.on("mousedown", this.startRectSel.bind(this));
            svg.on("dblclick", this.mouseDoubleClick.bind(this));
         }
         if (settings.ZoomWheel)
            svg.on("wheel", this.mouseWheel.bind(this));
      }

      if (browser.touches && ((settings.Zooming && settings.ZoomTouch && !this.projection) || settings.ContextMenu))
         svg.on("touchstart", this.startTouchZoom.bind(this));

      if (settings.ContextMenu) {
         if (browser.touches) {
            svg_x.on("touchstart", this.startTouchMenu.bind(this,"x"));
            svg_y.on("touchstart", this.startTouchMenu.bind(this,"y"));
         }
         svg.on("contextmenu", evnt => this.showContextMenu("", evnt));
         svg_x.on("contextmenu", evnt => this.showContextMenu("x", evnt));
         svg_y.on("contextmenu", evnt => this.showContextMenu("y", evnt));
      }

      svg_x.on("mousemove", evnt => this.showAxisStatus("x", evnt));
      svg_y.on("mousemove", evnt => this.showAxisStatus("y", evnt));

      svg.property('interactive_set', true);

      return Promise.resolve(this);
   },

   /** @summary Add keys handler */
   addFrameKeysHandler() {
      if (this.keys_handler || (typeof window == 'undefined')) return;

      this.keys_handler = evnt => this.processKeyPress(evnt);

      window.addEventListener('keydown', this.keys_handler, false);
   },

   /** @summary Handle key press */
   processKeyPress(evnt) {
      let main = this.selectDom();
      if (!settings.HandleKeys || main.empty() || (this.enabledKeys === false)) return;

      let key = "";
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

      let pp = this.getPadPainter();
      if (getActivePad() !== pp) return;

      if (evnt.shiftKey) key = "Shift " + key;
      if (evnt.altKey) key = "Alt " + key;
      if (evnt.ctrlKey) key = "Ctrl " + key;

      let zoom = { name: "x", dleft: 0, dright: 0 };

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
         if (!settings.Zooming) return false;
         // in 3dmode with orbit control ignore simple arrows
         if (this.mode3d && (key.indexOf("Ctrl")!==0)) return false;
         this.analyzeMouseWheelEvent(null, zoom, 0.5);
         this.zoom(zoom.name, zoom.min, zoom.max);
         if (zoom.changed) this.zoomChangedInteractive(zoom.name, true);
         evnt.stopPropagation();
         evnt.preventDefault();
      } else {
         let func = pp && pp.findPadButton ? pp.findPadButton(key) : "";
         if (func) {
            pp.clickPadButton(func);
            evnt.stopPropagation();
            evnt.preventDefault();
         }
      }

      return true; // just process any key press
   },

   /** @summary Function called when frame is clicked and object selection can be performed
     * @desc such event can be used to select */
   processFrameClick(pnt, dblckick) {

      let pp = this.getPadPainter();
      if (!pp) return;

      pnt.painters = true; // provide painters reference in the hints
      pnt.disabled = true; // do not invoke graphics

      // collect tooltips from pad painter - it has list of all drawn objects
      let hints = pp.processPadTooltipEvent(pnt), exact = null, res;
      for (let k = 0; (k <hints.length) && !exact; ++k)
         if (hints[k] && hints[k].exact)
            exact = hints[k];

      if (exact) {
         let handler = dblckick ? this._dblclick_handler : this._click_handler;
         if (handler) res = handler(exact.user_info, pnt);
      }

      if (!dblckick)
         pp.selectObjectPainter(exact ? exact.painter : this,
               { x: pnt.x + (this._frame_x || 0),  y: pnt.y + (this._frame_y || 0) });

      return res;
   },

   /** @summary Start mouse rect zooming */
   startRectSel(evnt) {
      // ignore when touch selection is activated

      if (this.zoom_kind > 100) return;

      // ignore all events from non-left button
      if ((evnt.which || evnt.button) !== 1) return;

      evnt.preventDefault();

      let frame = this.getFrameSvg(),
          pos = d3_pointer(evnt, frame.node());

      this.clearInteractiveElements();

      let w = this.getFrameWidth(), h = this.getFrameHeight();

      this.zoom_lastpos = pos;
      this.zoom_curr = [ Math.max(0, Math.min(w, pos[0])),
                         Math.max(0, Math.min(h, pos[1])) ];

      this.zoom_origin = [0,0];
      this.zoom_second = false;

      if ((pos[0] < 0) || (pos[0] > w)) {
         this.zoom_second = (pos[0] > w) && this.y2_handle;
         this.zoom_kind = 3; // only y
         this.zoom_origin[1] = this.zoom_curr[1];
         this.zoom_curr[0] = w;
         this.zoom_curr[1] += 1;
      } else if ((pos[1] < 0) || (pos[1] > h)) {
         this.zoom_second = (pos[1] < 0) && this.x2_handle;
         this.zoom_kind = 2; // only x
         this.zoom_origin[0] = this.zoom_curr[0];
         this.zoom_curr[0] += 1;
         this.zoom_curr[1] = h;
      } else {
         this.zoom_kind = 1; // x and y
         this.zoom_origin[0] = this.zoom_curr[0];
         this.zoom_origin[1] = this.zoom_curr[1];
      }

      d3_select(window).on("mousemove.zoomRect", this.moveRectSel.bind(this))
                       .on("mouseup.zoomRect", this.endRectSel.bind(this), true);

      this.zoom_rect = null;

      // disable tooltips in frame painter
      setPainterTooltipEnabled(this, false);

      evnt.stopPropagation();

      if (this.zoom_kind != 1)
         setTimeout(() => this.startLabelsMove(), 500);
   },

   /** @summary Starts labels move */
   startLabelsMove() {
      if (this.zoom_rect) return;

      let handle = this.zoom_kind == 2 ? this.x_handle : this.y_handle;

      if (!handle || (typeof handle.processLabelsMove != 'function') || !this.zoom_lastpos) return;

      if (handle.processLabelsMove('start', this.zoom_lastpos)) {
         this.zoom_labels = handle;
      }
   },

   /** @summary Process mouse rect zooming */
   moveRectSel(evnt) {

      if ((this.zoom_kind == 0) || (this.zoom_kind > 100)) return;

      evnt.preventDefault();
      let m = d3_pointer(evnt, this.getFrameSvg().node());

      if (this.zoom_labels)
         return this.zoom_labels.processLabelsMove('move', m);

      this.zoom_lastpos[0] = m[0];
      this.zoom_lastpos[1] = m[1];

      m[0] = Math.max(0, Math.min(this.getFrameWidth(), m[0]));
      m[1] = Math.max(0, Math.min(this.getFrameHeight(), m[1]));

      switch (this.zoom_kind) {
         case 1: this.zoom_curr[0] = m[0]; this.zoom_curr[1] = m[1]; break;
         case 2: this.zoom_curr[0] = m[0]; break;
         case 3: this.zoom_curr[1] = m[1]; break;
      }

      let x = Math.min(this.zoom_origin[0], this.zoom_curr[0]),
          y = Math.min(this.zoom_origin[1], this.zoom_curr[1]),
          w = Math.abs(this.zoom_curr[0] - this.zoom_origin[0]),
          h = Math.abs(this.zoom_curr[1] - this.zoom_origin[1]);

      if (!this.zoom_rect) {
         // ignore small changes, can be switching to labels move
         if ((this.zoom_kind != 1) && ((w < 2) || (h < 2))) return;

         this.zoom_rect = this.getFrameSvg()
                              .append("rect")
                              .attr("class", "zoom")
                              .style("pointer-events","none");
      }

      this.zoom_rect.attr("x", x).attr("y", y).attr("width", w).attr("height", h);
   },

   /** @summary Finish mouse rect zooming */
   endRectSel(evnt) {
      if ((this.zoom_kind == 0) || (this.zoom_kind > 100)) return;

      evnt.preventDefault();

      d3_select(window).on("mousemove.zoomRect", null)
                       .on("mouseup.zoomRect", null);

      let m = d3_pointer(evnt, this.getFrameSvg().node()), kind = this.zoom_kind;

      if (this.zoom_labels) {
         this.zoom_labels.processLabelsMove('stop', m);
      } else {
         let changed = [this.can_zoom_x, this.can_zoom_y];
         m[0] = Math.max(0, Math.min(this.getFrameWidth(), m[0]));
         m[1] = Math.max(0, Math.min(this.getFrameHeight(), m[1]));

         switch (this.zoom_kind) {
            case 1: this.zoom_curr[0] = m[0]; this.zoom_curr[1] = m[1]; break;
            case 2: this.zoom_curr[0] = m[0]; changed[1] = false; break; // only X
            case 3: this.zoom_curr[1] = m[1]; changed[0] = false; break; // only Y
         }

         let xmin, xmax, ymin, ymax, isany = false,
             idx = this.swap_xy ? 1 : 0, idy = 1 - idx,
             namex = "x", namey = "y";

         if (changed[idx] && (Math.abs(this.zoom_curr[idx] - this.zoom_origin[idx]) > 10)) {
            if (this.zoom_second && (this.zoom_kind == 2)) namex = "x2";
            xmin = Math.min(this.revertAxis(namex, this.zoom_origin[idx]), this.revertAxis(namex, this.zoom_curr[idx]));
            xmax = Math.max(this.revertAxis(namex, this.zoom_origin[idx]), this.revertAxis(namex, this.zoom_curr[idx]));
            isany = true;
         }

         if (changed[idy] && (Math.abs(this.zoom_curr[idy] - this.zoom_origin[idy]) > 10)) {
            if (this.zoom_second && (this.zoom_kind == 3)) namey = "y2";
            ymin = Math.min(this.revertAxis(namey, this.zoom_origin[idy]), this.revertAxis(namey, this.zoom_curr[idy]));
            ymax = Math.max(this.revertAxis(namey, this.zoom_origin[idy]), this.revertAxis(namey, this.zoom_curr[idy]));
            isany = true;
         }

         if (namex == "x2") {
            this.zoomChangedInteractive(namex, true);
            this.zoomSingle(namex, xmin, xmax);
            kind = 0;
         } else if (namey == "y2") {
            this.zoomChangedInteractive(namey, true);
            this.zoomSingle(namey, ymin, ymax);
            kind = 0;
         } else if (isany) {
            this.zoomChangedInteractive("x", true);
            this.zoomChangedInteractive("y", true);
            this.zoom(xmin, xmax, ymin, ymax);
            kind = 0;
         }
      }

      let pnt = (kind===1) ? { x: this.zoom_origin[0], y: this.zoom_origin[1] } : null;

      this.clearInteractiveElements();

      // if no zooming was done, select active object instead
      switch (kind) {
         case 1:
            this.processFrameClick(pnt);
            break;
         case 2: {
            let pp = this.getPadPainter();
            if (pp) pp.selectObjectPainter(this, null, "xaxis");
            break;
         }
         case 3: {
            let pp = this.getPadPainter();
            if (pp) pp.selectObjectPainter(this, null, "yaxis");
            break;
         }
      }

   },

   /** @summary Handle mouse double click on frame */
   mouseDoubleClick(evnt) {
      evnt.preventDefault();
      let m = d3_pointer(evnt, this.getFrameSvg().node()),
          fw = this.getFrameWidth(), fh = this.getFrameHeight();
      this.clearInteractiveElements();

      let valid_x = (m[0] >= 0) && (m[0] <= fw),
          valid_y = (m[1] >= 0) && (m[1] <= fh);

      if (valid_x && valid_y && this._dblclick_handler)
         if (this.processFrameClick({ x: m[0], y: m[1] }, true)) return;

      let kind = (this.can_zoom_x ? "x" : "") + (this.can_zoom_y ? "y" : "") + "z";
      if (!valid_x) {
         if (!this.can_zoom_y) return;
         kind = this.swap_xy ? "x" : "y";
         if ((m[0] > fw) && this[kind+"2_handle"]) kind += "2"; // let unzoom second axis
      } else if (!valid_y) {
         if (!this.can_zoom_x) return;
         kind = this.swap_xy ? "y" : "x";
         if ((m[1] < 0) && this[kind+"2_handle"]) kind += "2"; // let unzoom second axis
      }
      this.unzoom(kind).then(changed => {
         if (changed) return;
         let pp = this.getPadPainter(), rect = this.getFrameRect();
         if (pp) pp.selectObjectPainter(pp, { x: m[0] + rect.x, y: m[1] + rect.y, dbl: true });
      });
   },

   /** @summary Start touch zoom */
   startTouchZoom(evnt) {
      // in case when zooming was started, block any other kind of events
      if (this.zoom_kind != 0) {
         evnt.preventDefault();
         evnt.stopPropagation();
         return;
      }

      let arr = d3_pointers(evnt, this.getFrameSvg().node());
      this.touch_cnt+=1;

      // normally double-touch will be handled
      // touch with single click used for context menu
      if (arr.length == 1) {
         // this is touch with single element

         let now = new Date(), diff = now.getTime() - this.last_touch.getTime();
         this.last_touch = now;

         if ((diff < 300) && this.zoom_curr
             && (Math.abs(this.zoom_curr[0] - arr[0][0]) < 30)
             && (Math.abs(this.zoom_curr[1] - arr[0][1]) < 30)) {

            evnt.preventDefault();
            evnt.stopPropagation();

            this.clearInteractiveElements();
            this.unzoom("xyz");

            this.last_touch = new Date(0);

            this.getFrameSvg().on("touchcancel", null)
                            .on("touchend", null, true);
         } else if (settings.ContextMenu) {
            this.zoom_curr = arr[0];
            this.getFrameSvg().on("touchcancel", this.endTouchSel.bind(this))
                            .on("touchend", this.endTouchSel.bind(this));
            evnt.preventDefault();
            evnt.stopPropagation();
         }
      }

      if ((arr.length != 2) || !settings.Zooming || !settings.ZoomTouch) return;

      evnt.preventDefault();
      evnt.stopPropagation();

      this.clearInteractiveElements();

      this.getFrameSvg().on("touchcancel", null)
                      .on("touchend", null);

      let pnt1 = arr[0], pnt2 = arr[1], w = this.getFrameWidth(), h = this.getFrameHeight();

      this.zoom_curr = [ Math.min(pnt1[0], pnt2[0]), Math.min(pnt1[1], pnt2[1]) ];
      this.zoom_origin = [ Math.max(pnt1[0], pnt2[0]), Math.max(pnt1[1], pnt2[1]) ];
      this.zoom_second = false;

      if ((this.zoom_curr[0] < 0) || (this.zoom_curr[0] > w)) {
         this.zoom_second = (this.zoom_curr[0] > w) && this.y2_handle;
         this.zoom_kind = 103; // only y
         this.zoom_curr[0] = 0;
         this.zoom_origin[0] = w;
      } else if ((this.zoom_origin[1] > h) || (this.zoom_origin[1] < 0)) {
         this.zoom_second = (this.zoom_origin[1] < 0) && this.x2_handle;
         this.zoom_kind = 102; // only x
         this.zoom_curr[1] = 0;
         this.zoom_origin[1] = h;
      } else {
         this.zoom_kind = 101; // x and y
      }

      setPainterTooltipEnabled(this, false);

      this.zoom_rect = this.getFrameSvg().append("rect")
            .attr("class", "zoom")
            .attr("id", "zoomRect")
            .attr("x", this.zoom_curr[0])
            .attr("y", this.zoom_curr[1])
            .attr("width", this.zoom_origin[0] - this.zoom_curr[0])
            .attr("height", this.zoom_origin[1] - this.zoom_curr[1]);

      d3_select(window).on("touchmove.zoomRect", this.moveTouchZoom.bind(this))
                       .on("touchcancel.zoomRect", this.endTouchZoom.bind(this))
                       .on("touchend.zoomRect", this.endTouchZoom.bind(this));
   },

   /** @summary Move touch zooming */
   moveTouchZoom(evnt) {
      if (this.zoom_kind < 100) return;

      evnt.preventDefault();

      let arr = d3_pointers(evnt, this.getFrameSvg().node());

      if (arr.length != 2)
         return this.clearInteractiveElements();

      let pnt1 = arr[0], pnt2 = arr[1];

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
         setPainterTooltipEnabled(this, false);

      evnt.stopPropagation();
   },

   /** @summary End touch zooming handler */
   endTouchZoom(evnt) {

      this.getFrameSvg().on("touchcancel", null)
                      .on("touchend", null);

      if (this.zoom_kind === 0) {
         // special case - single touch can ends up with context menu

         evnt.preventDefault();

         let now = new Date();

         let diff = now.getTime() - this.last_touch.getTime();

         if ((diff > 500) && (diff < 2000) && !this.isTooltipShown()) {
            this.showContextMenu('main', { clientX: this.zoom_curr[0], clientY: this.zoom_curr[1] });
            this.last_touch = new Date(0);
         } else {
            this.clearInteractiveElements();
         }
      }

      if (this.zoom_kind < 100) return;

      evnt.preventDefault();
      d3_select(window).on("touchmove.zoomRect", null)
                       .on("touchend.zoomRect", null)
                       .on("touchcancel.zoomRect", null);

      let xmin, xmax, ymin, ymax, isany = false,
          xid = this.swap_xy ? 1 : 0, yid = 1 - xid,
          changed = [true, true], namex = "x", namey = "y";

      if (this.zoom_kind === 102) changed[1] = false;
      if (this.zoom_kind === 103) changed[0] = false;

      if (changed[xid] && (Math.abs(this.zoom_curr[xid] - this.zoom_origin[xid]) > 10)) {
         if (this.zoom_second && (this.zoom_kind == 102)) namex = "x2";
         xmin = Math.min(this.revertAxis(namex, this.zoom_origin[xid]), this.revertAxis(namex, this.zoom_curr[xid]));
         xmax = Math.max(this.revertAxis(namex, this.zoom_origin[xid]), this.revertAxis(namex, this.zoom_curr[xid]));
         isany = true;
      }

      if (changed[yid] && (Math.abs(this.zoom_curr[yid] - this.zoom_origin[yid]) > 10)) {
         if (this.zoom_second && (this.zoom_kind == 103)) namey = "y2";
         ymin = Math.min(this.revertAxis(namey, this.zoom_origin[yid]), this.revertAxis(namey, this.zoom_curr[yid]));
         ymax = Math.max(this.revertAxis(namey, this.zoom_origin[yid]), this.revertAxis(namey, this.zoom_curr[yid]));
         isany = true;
      }

      this.clearInteractiveElements();
      this.last_touch = new Date(0);

      if (namex == "x2") {
         this.zoomChangedInteractive(namex, true);
         this.zoomSingle(namex, xmin, xmax);
      } else if (namey == "y2") {
         this.zoomChangedInteractive(namey, true);
         this.zoomSingle(namey, ymin, ymax);
      } else if (isany) {
         this.zoomChangedInteractive('x', true);
         this.zoomChangedInteractive('y', true);
         this.zoom(xmin, xmax, ymin, ymax);
      }

      evnt.stopPropagation();
   },

   /** @summary Analyze zooming with mouse wheel */
   analyzeMouseWheelEvent(event, item, dmin, test_ignore, second_side) {
      // if there is second handle, use it
      let handle2 = second_side ? this[item.name + "2_handle"] : null;
      if (handle2) {
         item.second = Object.assign({}, item);
         return handle2.analyzeWheelEvent(event, dmin, item.second, test_ignore);
      }
      let handle = this[item.name + "_handle"];
      if (handle) return handle.analyzeWheelEvent(event, dmin, item, test_ignore);
      console.error('Fail to analyze zooming event for ', item.name);
   },

    /** @summary return true if default Y zooming should be enabled
      * @desc it is typically for 2-Dim histograms or
      * when histogram not draw, defined by other painters */
   isAllowedDefaultYZooming() {

      if (this.self_drawaxes) return true;

      let pad_painter = this.getPadPainter();
      if (pad_painter && pad_painter.painters)
         for (let k = 0; k < pad_painter.painters.length; ++k) {
            let subpainter = pad_painter.painters[k];
            if (subpainter && (subpainter.wheel_zoomy !== undefined))
               return subpainter.wheel_zoomy;
         }

      return false;
   },

   /** @summary Handles mouse wheel event */
   mouseWheel(evnt) {
      evnt.stopPropagation();
      evnt.preventDefault();
      this.clearInteractiveElements();

      let itemx = { name: "x", reverse: this.reverse_x },
          itemy = { name: "y", reverse: this.reverse_y, ignore: !this.isAllowedDefaultYZooming() },
          cur = d3_pointer(evnt, this.getFrameSvg().node()),
          w = this.getFrameWidth(), h = this.getFrameHeight();

      if (this.can_zoom_x)
         this.analyzeMouseWheelEvent(evnt, this.swap_xy ? itemy : itemx, cur[0] / w, (cur[1] >=0) && (cur[1] <= h), cur[1] < 0);

      if (this.can_zoom_y)
         this.analyzeMouseWheelEvent(evnt, this.swap_xy ? itemx : itemy, 1 - cur[1] / h, (cur[0] >= 0) && (cur[0] <= w), cur[0] > w);

      this.zoom(itemx.min, itemx.max, itemy.min, itemy.max);

      if (itemx.changed) this.zoomChangedInteractive('x', true);
      if (itemy.changed) this.zoomChangedInteractive('y', true);

      if (itemx.second) {
         this.zoomSingle("x2", itemx.second.min, itemx.second.max);
         if (itemx.second.changed) this.zoomChangedInteractive('x2', true);
      }
      if (itemy.second) {
         this.zoomSingle("y2", itemy.second.min, itemy.second.max);
         if (itemy.second.changed) this.zoomChangedInteractive('y2', true);
      }
   },

   /** @summary Show frame context menu */
   showContextMenu(kind, evnt, obj) {

      // ignore context menu when touches zooming is ongoing
      if (('zoom_kind' in this) && (this.zoom_kind > 100)) return;

      // this is for debug purposes only, when context menu is where, close is and show normal menu
      //if (!evnt && !kind && document.getElementById('root_ctx_menu')) {
      //   let elem = document.getElementById('root_ctx_menu');
      //   elem.parentNode.removeChild(elem);
      //   return;
      //}

      let menu_painter = this, exec_painter = null, frame_corner = false, fp = null; // object used to show context menu

      if (evnt.stopPropagation) {
         evnt.preventDefault();
         evnt.stopPropagation(); // disable main context menu

         if (kind == 'painter' && obj) {
            menu_painter = obj;
            kind = "";
         } else if (!kind) {
            let ms = d3_pointer(evnt, this.getFrameSvg().node()),
                tch = d3_pointers(evnt, this.getFrameSvg().node()),
                pp = this.getPadPainter(),
                pnt = null, sel = null;

            fp = this;

            if (tch.length === 1) pnt = { x: tch[0][0], y: tch[0][1], touch: true }; else
            if (ms.length === 2) pnt = { x: ms[0], y: ms[1], touch: false };

            if ((pnt !== null) && (pp !== null)) {
               pnt.painters = true; // assign painter for every tooltip
               let hints = pp.processPadTooltipEvent(pnt), bestdist = 1000;
               for (let n=0;n<hints.length;++n)
                  if (hints[n] && hints[n].menu) {
                     let dist = ('menu_dist' in hints[n]) ? hints[n].menu_dist : 7;
                     if (dist < bestdist) { sel = hints[n].painter; bestdist = dist; }
                  }
            }

            if (sel) menu_painter = sel; else kind = "frame";

            if (pnt) frame_corner = (pnt.x>0) && (pnt.x<20) && (pnt.y>0) && (pnt.y<20);

            fp.setLastEventPos(pnt);
         } else if (!this.v7_frame && ((kind=="x") || (kind=="y") || (kind=="z"))) {
            exec_painter = this.getMainPainter(); // histogram painter delivers items for axis menu
         }
      } else if (kind == 'painter' && obj) {
         // this is used in 3D context menu to show special painter
         menu_painter = obj;
         kind = "";
      }

      if (!exec_painter) exec_painter = menu_painter;

      if (!menu_painter || !menu_painter.fillContextMenu) return;

      this.clearInteractiveElements();

      createMenu(evnt, menu_painter).then(menu => {
         let domenu = menu.painter.fillContextMenu(menu, kind, obj);

         // fill frame menu by default - or append frame elements when activated in the frame corner
         if (fp && (!domenu || (frame_corner && (kind!=="frame"))))
            domenu = fp.fillContextMenu(menu);

         if (domenu)
            exec_painter.fillObjectExecMenu(menu, kind).then(menu => {
                // suppress any running zooming
                setPainterTooltipEnabled(menu.painter, false);
                menu.show().then(() => setPainterTooltipEnabled(menu.painter, true));
            });
      });
   },

  /** @summary Activate context menu handler via touch events
    * @private */
   startTouchMenu(kind, evnt) {
      // method to let activate context menu via touch handler

      let arr = d3_pointers(evnt, this.getFrameSvg().node());
      if (arr.length != 1) return;

      if (!kind || (kind=="")) kind = "main";
      let fld = "touch_" + kind;

      evnt.sourceEvent.preventDefault();
      evnt.sourceEvent.stopPropagation();

      this[fld] = { dt: new Date(), pos: arr[0] };

      let handler = this.endTouchMenu.bind(this, kind);

      this.getFrameSvg().on("touchcancel", handler)
                      .on("touchend", handler);
   },

   /** @summary Process end-touch event, which can cause content menu to appear
    * @private */
   endTouchMenu(kind, evnt) {
      let fld = "touch_" + kind;

      if (! (fld in this)) return;

      evnt.sourceEvent.preventDefault();
      evnt.sourceEvent.stopPropagation();

      let diff = new Date().getTime() - this[fld].dt.getTime();

      this.getFrameSvg().on("touchcancel", null)
                      .on("touchend", null);

      if (diff > 500) {
         let rect = this.getFrameSvg().node().getBoundingClientRect();
         this.showContextMenu(kind, { clientX: rect.left + this[fld].pos[0],
                                      clientY: rect.top + this[fld].pos[1] } );
      }

      delete this[fld];
   },

   /** @summary Clear frame interactive elements */
   clearInteractiveElements() {
      closeMenu();
      this.zoom_kind = 0;
      if (this.zoom_rect) { this.zoom_rect.remove(); delete this.zoom_rect; }
      delete this.zoom_curr;
      delete this.zoom_origin;
      delete this.zoom_lastpos;
      delete this.zoom_labels;

      // enable tooltip in frame painter
      setPainterTooltipEnabled(this, true);
   },

   /** @summary Assign frame interactive methods */
   assign(painter) {
      Object.assign(painter, this);
   }

} // FrameInterative



/**
 * @summary Painter class for TFrame, main handler for interactivity
 */

class TFramePainter extends ObjectPainter {

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
      this.enabledKeys = on && settings.HandleKeys ? true : false;
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

            if (axis && axis.TestBit(EAxisBits.kAxisRange)) {
               if ((axis.fFirst !== axis.fLast) && ((axis.fFirst > 1) || (axis.fLast < axis.fNbins))) {
                  this[`zoom_${name}min`] = axis.fFirst > 1 ? axis.GetBinLowEdge(axis.fFirst) : axis.fXmin;
                  this[`zoom_${name}max`] = axis.fLast < axis.fNbins ? axis.GetBinLowEdge(axis.fLast + 1) : axis.fXmax;
                  // reset user range for main painter
                  axis.InvertBit(EAxisBits.kAxisRange);
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
          grid_style = gStyle.fGridStyle;

      // add a grid on x axis, if the option is set
      if (pad && pad.fGridx && this.x_handle) {
         let gridx = "";
         for (let n = 0; n < this.x_handle.ticks.length; ++n)
            if (this.swap_xy)
               gridx += `M0,${this.x_handle.ticks[n]}h${w}`;
            else
               gridx += `M${this.x_handle.ticks[n]},0v${h}`;

         let colid = (gStyle.fGridColor > 0) ? gStyle.fGridColor : (this.getAxis("x") ? this.getAxis("x").fAxisColor : 1),
             grid_color = this.getColor(colid) || "black";

         if (gridx.length > 0)
           layer.append("svg:path")
                .attr("class", "xgrid")
                .attr("d", gridx)
                .style("stroke", grid_color)
                .style("stroke-width", gStyle.fGridWidth)
                .style("stroke-dasharray", getSvgLineStyle(grid_style));
      }

      // add a grid on y axis, if the option is set
      if (pad && pad.fGridy && this.y_handle) {
         let gridy = "";
         for (let n = 0; n < this.y_handle.ticks.length; ++n)
            if (this.swap_xy)
               gridy += `M${this.y_handle.ticks[n]},0v${h}`;
            else
               gridy += `M0,${this.y_handle.ticks[n]}h${w}`;

         let colid = (gStyle.fGridColor > 0) ? gStyle.fGridColor : (this.getAxis("y") ? this.getAxis("y").fAxisColor : 1),
             grid_color = this.getColor(colid) || "black";

         if (gridy.length > 0)
           layer.append("svg:path")
                .attr("class", "ygrid")
                .attr("d", gridy)
                .style("stroke", grid_color)
                .style("stroke-width",gStyle.fGridWidth)
                .style("stroke-dasharray", getSvgLineStyle(grid_style));
      }
   }

   /** @summary Converts "raw" axis value into text */
   axisAsText(axis, value) {
      let handle = this[axis+"_handle"];

      if (handle)
         return handle.axisAsText(value, settings[axis.toUpperCase() + "ValuesFormat"]);

      return value.toPrecision(4);
   }

   /** @summary Identify if requested axes are drawn
     * @desc Checks if x/y axes are drawn. Also if second side is already there */
   hasDrawnAxes(second_x, second_y) {
      return !second_x && !second_y ? this.axes_drawn : false;
   }

   /** @summary draw axes, return Promise which ready when drawing is completed  */
   drawAxes(shrink_forbidden, disable_x_draw, disable_y_draw,
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
          draw_vertical = this.swap_xy ? this.x_handle : this.y_handle;

      if (!disable_x_draw || !disable_y_draw) {
         let pp = this.getPadPainter();
         if (pp && pp._fast_drawing) disable_x_draw = disable_y_draw = true;
      }

      let pr = Promise.resolve(true);

      if (!disable_x_draw || !disable_y_draw) {

         let can_adjust_frame = !shrink_forbidden && settings.CanAdjustFrame;

         let pr1 = draw_horiz.drawAxis(layer, w, h,
                                   draw_horiz.invert_side ? undefined : `translate(0,${h})`,
                                   pad && pad.fTickx ? -h : 0, disable_x_draw,
                                   undefined, false);

         let pr2 = draw_vertical.drawAxis(layer, w, h,
                                      draw_vertical.invert_side ? `translate(${w})` : undefined,
                                      pad && pad.fTicky ? w : 0, disable_y_draw,
                                      draw_vertical.invert_side ? 0 : this._frame_x, can_adjust_frame);

         pr = Promise.all([pr1,pr2]).then(() => {

            this.drawGrids();

            if (!can_adjust_frame) return;

            let shrink = 0., ypos = draw_vertical.position;

            if ((-0.2 * w < ypos) && (ypos < 0)) {
               shrink = -ypos / w + 0.001;
               this.shrink_frame_left += shrink;
            } else if ((ypos > 0) && (ypos < 0.3 * w) && (this.shrink_frame_left > 0) && (ypos / w > this.shrink_frame_left)) {
               shrink = -this.shrink_frame_left;
               this.shrink_frame_left = 0.;
            }

            if (!shrink) return;

            this.shrinkFrame(shrink, 0);
            return this.redraw().then(() => this.drawAxes(true));
         });
      }

     return pr.then(() => {
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

      let pr1, pr2;

      if (draw_horiz)
         pr1 = draw_horiz.drawAxis(layer, w, h,
                                   draw_horiz.invert_side ? undefined : `translate(0,${h})`,
                                   pad && pad.fTickx ? -h : 0, false,
                                   undefined, false);

      if (draw_vertical)
         pr2 = draw_vertical.drawAxis(layer, w, h,
                                      draw_vertical.invert_side ? `translate(${w})` : undefined,
                                      pad && pad.fTicky ? w : 0, false,
                                      draw_vertical.invert_side ? 0 : this._frame_x, false);

       return Promise.all([pr1, pr2]);
   }


   /** @summary Update frame attributes
     * @private */
   updateAttributes(force) {
      let pp = this.getPadPainter(),
          pad = pp ? pp.getRootPad(true) : null,
          tframe = this.getObject();

      if ((this.fX1NDC === undefined) || (force && !this.modified_NDC)) {
         if (!pad) {
            Object.assign(this, settings.FrameNDC);
         } else {
            Object.assign(this, {
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

      if (this.mode3d) return this; // no need to create any elements in 3d mode

      // this is svg:g object - container for every other items belonging to frame
      this.draw_g = this.getFrameSvg();

      let top_rect, main_svg;

      if (this.draw_g.empty()) {

         this.draw_g = this.getLayerSvg("primitives_layer").append("svg:g").attr("class", "root_frame");

         // empty title on the frame required to suppress title of the canvas
         if (!isBatchMode())
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

      if (!isBatchMode()) {
         top_rect.style("pointer-events", "visibleFill"); // let process mouse events inside frame
         FrameInteractive.assign(this);
         this.addBasicInteractivity();
      }

      return this;
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
         menu.addchk(faxis.TestBit(EAxisBits.kMoreLogLabels), "More log",
               () => { faxis.InvertBit(EAxisBits.kMoreLogLabels); this.redrawPad(); });
         menu.addchk(faxis.TestBit(EAxisBits.kNoExponent), "No exponent",
               () => { faxis.InvertBit(EAxisBits.kNoExponent); this.redrawPad(); });

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

            menu.addTAxisMenu(EAxisBits, main || this, faxis, kind);
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
          m = d3_pointer(evnt, this.getFrameSvg().node()), id = (axis_name=="x") ? 0 : 1;

      if (taxis) { hint_name = taxis.fName; hint_title = taxis.fTitle || ("TAxis object for " + axis_name); }
      if (this.swap_xy) id = 1-id;

      let axis_value = this.revertAxis(axis_name, m[id]);

      this.showObjectStatus(hint_name, hint_title, axis_name + " : " + this.axisAsText(axis_name, axis_value), m[0]+","+m[1]);
   }

   /** @summary Add interactive keys handlers
    * @private */
   addKeysHandler() {
      if (isBatchMode()) return;
      FrameInteractive.assign(this);
      this.addFrameKeysHandler();
   }

   /** @summary Add interactive functionality to the frame
     * @private */
   addInteractivity(for_second_axes) {
      if (isBatchMode() || (!settings.Zooming && !settings.ContextMenu))
         return false;

      FrameInteractive.assign(this);
      return this.addFrameInteractivity(for_second_axes);
   }

} // class TFramePainter

export { addDragHandler, TooltipHandler, FrameInteractive, TFramePainter };

