import { gStyle, settings, isFunc, isStr, postponePromise, browser, clTAxis, kNoZoom } from '../core.mjs';
import { select as d3_select, pointer as d3_pointer, pointers as d3_pointers, drag as d3_drag } from '../d3.mjs';
import { getElementRect, getAbsPosInCanvas, makeTranslate, addHighlightStyle } from '../base/BasePainter.mjs';
import { getActivePad, ObjectPainter, EAxisBits, kAxisLabels } from '../base/ObjectPainter.mjs';
import { getSvgLineStyle } from '../base/TAttLineHandler.mjs';
import { TAxisPainter } from './TAxisPainter.mjs';
import { FontHandler } from '../base/FontHandler.mjs';
import { createMenu, closeMenu, showPainterMenu, hasMenu } from '../gui/menu.mjs';
import { detectRightButton } from '../gui/utils.mjs';


const logminfactorX = 0.0001, logminfactorY = 3e-4;

/** @summary Configure tooltip enable flag for painter
  * @private */
function setPainterTooltipEnabled(painter, on) {
   if (!painter) return;

   const fp = painter.getFramePainter();
   if (isFunc(fp?.setTooltipEnabled)) {
      fp.setTooltipEnabled(on);
      fp.processFrameTooltipEvent(null);
   }
   // this is 3D control object
   if (isFunc(painter.control?.setTooltipEnabled))
      painter.control.setTooltipEnabled(on);
}

/** @summary Return pointers on touch event
  * @private */
function get_touch_pointers(event, node) {
   return event.$touch_arr ?? d3_pointers(event, node);
}

/** @summary Returns coordinates transformation func
  * @private */
function getEarthProjectionFunc(id) {
   switch (id) {
      // Aitoff2xy
      case 1: return (l, b) => {
         const DegToRad = Math.PI/180,
               alpha2 = (l/2)*DegToRad,
               delta = b*DegToRad,
               r2 = Math.sqrt(2),
               f = 2*r2/Math.PI,
               cdec = Math.cos(delta),
               denom = Math.sqrt(1.0 + cdec*Math.cos(alpha2));
         return {
            x: cdec*Math.sin(alpha2)*2.0*r2/denom/f/DegToRad,
            y: Math.sin(delta)*r2/denom/f/DegToRad
         };
      };
      // mercator
      case 2: return (l, b) => { return { x: l, y: Math.log(Math.tan((Math.PI/2 + b/180*Math.PI)/2)) }; };
      // sinusoidal
      case 3: return (l, b) => { return { x: l*Math.cos(b/180*Math.PI), y: b }; };
      // parabolic
      case 4: return (l, b) => { return { x: l*(2.0*Math.cos(2*b/180*Math.PI/3) - 1), y: 180*Math.sin(b/180*Math.PI/3) }; };
      // Mollweide projection
      case 5: return (l, b) => {
         const theta0 = b * Math.PI/180;
         let theta = theta0, num, den;
         for (let i = 0; i < 100; i++) {
            num = 2 * theta + Math.sin(2 * theta) - Math.PI * Math.sin(theta0);
            den = 4 * (Math.cos(theta)**2);
            if (den < 1e-20) {
               theta = theta0;
               break;
            }
            theta -= num / den;
            if (Math.abs(num / den) < 1e-4) break;
         }
         return {
            x: l * Math.cos(theta),
            y: 90 * Math.sin(theta)
         };
      };
   }
}

/** @summary Unzoom preselected range for main histogram painter
  * @desc Used with TGraph where Y zooming selected with fMinimum/fMaximum but histogram
  * axis range can be wider. Or for normal histogram drawing when preselected range smaller than histogram range
  * @private */
function unzoomHistogramYRange(main) {
    if (!isFunc(main?.getDimension) || main.getDimension() !== 1) return;

    const ymin = main.draw_content ? main.hmin : main.ymin,
          ymax = main.draw_content ? main.hmax : main.ymax;

    if ((main.zoom_ymin !== main.zoom_ymax) && (ymin !== ymax) &&
        (ymin <= main.zoom_ymin) && (main.zoom_ymax <= ymax))
       main.zoom_ymin = main.zoom_ymax = 0;
}

// global, allow single drag at once
let drag_rect = null, drag_kind = '', drag_painter = null;

/** @summary Check if dragging performed currently
  * @private */
function is_dragging(painter, kind) {
   return drag_rect && (drag_painter === painter) && (drag_kind === kind);
}

/** @summary Add drag for interactive rectangular elements for painter
  * @private */
function addDragHandler(_painter, arg) {
   if (!settings.MoveResize) return;

   const painter = _painter, pp = painter.getPadPainter();
   if (pp?._fast_drawing || pp?.isBatchMode()) return;
   // cleanup all drag elements when canvas is not ediatable
   if (pp?.isEditable() === false)
      arg.cleanup = true;

   if (!isFunc(arg.getDrawG))
      arg.getDrawG = () => painter?.draw_g;

   function makeResizeElements(group, handler) {
      function addElement(cursor, d) {
         const clname = 'js_' + cursor.replace(/[-]/g, '_');
         let elem = group.selectChild('.' + clname);
         if (arg.cleanup) return elem.remove();
         if (elem.empty()) elem = group.append('path').classed(clname, true);
         elem.style('opacity', 0).style('cursor', cursor).attr('d', d);
         if (handler) elem.call(handler);
      }

      addElement('nw-resize', 'M2,2h15v-5h-20v20h5Z');
      addElement('ne-resize', `M${arg.width-2},2h-15v-5h20v20h-5 Z`);
      addElement('sw-resize', `M2,${arg.height-2}h15v5h-20v-20h5Z`);
      addElement('se-resize', `M${arg.width-2},${arg.height-2}h-15v5h20v-20h-5Z`);

      if (!arg.no_change_x) {
         addElement('w-resize', `M-3,18h5v${Math.max(0, arg.height-2*18)}h-5Z`);
         addElement('e-resize', `M${arg.width+3},18h-5v${Math.max(0, arg.height-2*18)}h5Z`);
      }
      if (!arg.no_change_y) {
         addElement('n-resize', `M18,-3v5h${Math.max(0, arg.width-2*18)}v-5Z`);
         addElement('s-resize', `M18,${arg.height+3}v-5h${Math.max(0, arg.width-2*18)}v5Z`);
      }
   }

   const complete_drag = (newx, newy, newwidth, newheight) => {
      drag_painter = null;
      drag_kind = '';
      if (drag_rect) {
         drag_rect.remove();
         drag_rect = null;
      }

      const draw_g = arg.getDrawG();

      if (!draw_g)
         return false;

      const oldx = arg.x, oldy = arg.y;

      if (arg.minwidth && newwidth < arg.minwidth) newwidth = arg.minwidth;
      if (arg.minheight && newheight < arg.minheight) newheight = arg.minheight;

      const change_size = (newwidth !== arg.width) || (newheight !== arg.height),
            change_pos = (newx !== oldx) || (newy !== oldy);

      arg.x = newx; arg.y = newy; arg.width = newwidth; arg.height = newheight;

      if (!arg.no_transform)
         makeTranslate(draw_g, newx, newy);

      setPainterTooltipEnabled(painter, true);

      makeResizeElements(draw_g);

      if (change_size || change_pos) {
         if (change_size && isFunc(arg.resize))
            arg.resize(newwidth, newheight);

         if (change_pos && isFunc(arg.move))
            arg.move(newx, newy, newx - oldx, newy - oldy);

         if (change_size || change_pos) {
            if (arg.obj) {
               const rect = arg.pad_rect ?? pp.getPadRect();
               arg.obj.fX1NDC = newx / rect.width;
               arg.obj.fX2NDC = (newx + newwidth) / rect.width;
               arg.obj.fY1NDC = 1 - (newy + newheight) / rect.height;
               arg.obj.fY2NDC = 1 - newy / rect.height;
               arg.obj.modified_NDC = true; // indicate that NDC was interactively changed, block in updated
            } else if (isFunc(arg.move_resize))
               arg.move_resize(newx, newy, newwidth, newheight);

            if (isFunc(arg.redraw))
               arg.redraw(arg);
         }
      }

      return change_size || change_pos;
   },
   drag_move = d3_drag().subject(Object),
   drag_move_off = d3_drag().subject(Object);

   drag_move_off.on('start', null).on('drag', null).on('end', null);

   drag_move
      .on('start', function(evnt) {
         if (detectRightButton(evnt.sourceEvent) || drag_kind) return;
         if (isFunc(arg.is_disabled) && arg.is_disabled('move')) return;

         closeMenu(); // close menu
         setPainterTooltipEnabled(painter, false); // disable tooltip

         evnt.sourceEvent.preventDefault();
         evnt.sourceEvent.stopPropagation();

         const pad_rect = arg.pad_rect ?? pp.getPadRect(), handle = {
            x: arg.x, y: arg.y, width: arg.width, height: arg.height,
            acc_x1: arg.x, acc_y1: arg.y,
            pad_w: pad_rect.width - arg.width,
            pad_h: pad_rect.height - arg.height,
            drag_tm: new Date(),
            path: `v${arg.height}h${arg.width}v${-arg.height}z`,
            evnt_x: evnt.x, evnt_y: evnt.y
         };

         drag_painter = painter;
         drag_kind = 'move';
         drag_rect = d3_select(arg.getDrawG().node().parentNode).append('path')
            .attr('d', `M${handle.acc_x1},${handle.acc_y1}${handle.path}`)
            .style('cursor', 'move')
            .style('pointer-events', 'none') // let forward double click to underlying elements
            .property('drag_handle', handle)
            .call(addHighlightStyle, true);
      }).on('drag', function(evnt) {
         if (!is_dragging(painter, 'move')) return;

         evnt.sourceEvent.preventDefault();
         evnt.sourceEvent.stopPropagation();

         const handle = drag_rect.property('drag_handle');

         if (!arg.no_change_x)
            handle.acc_x1 += evnt.dx;
         if (!arg.no_change_y)
            handle.acc_y1 += evnt.dy;

         handle.x = Math.min(Math.max(handle.acc_x1, 0), handle.pad_w);
         handle.y = Math.min(Math.max(handle.acc_y1, 0), handle.pad_h);

         drag_rect.attr('d', `M${handle.x},${handle.y}${handle.path}`);
      }).on('end', function(evnt) {
         if (!is_dragging(painter, 'move')) return;

         evnt.sourceEvent.stopPropagation();
         evnt.sourceEvent.preventDefault();

         const handle = drag_rect.property('drag_handle');

         if (complete_drag(handle.x, handle.y, arg.width, arg.height) === false) {
            const spent = (new Date()).getTime() - handle.drag_tm.getTime();

            if (arg.ctxmenu && (spent > 600))
               showPainterMenu({ clientX: handle.evnt_x, clientY: handle.evnt_y, skip_close: 1 }, painter);
            else if (arg.canselect && (spent <= 600))
               painter.getPadPainter()?.selectObjectPainter(painter);
         }
      });

   const drag_resize = d3_drag().subject(Object);

   drag_resize
      .on('start', function(evnt) {
         if (detectRightButton(evnt.sourceEvent) || drag_kind) return;
         if (isFunc(arg.is_disabled) && arg.is_disabled('resize')) return;

         closeMenu(); // close menu
         setPainterTooltipEnabled(painter, false); // disable tooltip

         evnt.sourceEvent.stopPropagation();
         evnt.sourceEvent.preventDefault();

         const pad_rect = arg.pad_rect ?? pp.getPadRect(), handle = {
            x: arg.x, y: arg.y, width: arg.width, height: arg.height,
            acc_x1: arg.x, acc_y1: arg.y,
            acc_x2: arg.x + arg.width, acc_y2: arg.y + arg.height,
            pad_w: pad_rect.width, pad_h: pad_rect.height
         };

         drag_painter = painter;
         drag_kind = 'resize';
         drag_rect = d3_select(arg.getDrawG().node().parentNode)
            .append('rect')
            .style('cursor', d3_select(this).style('cursor'))
            .attr('x', handle.acc_x1)
            .attr('y', handle.acc_y1)
            .attr('width', handle.acc_x2 - handle.acc_x1)
            .attr('height', handle.acc_y2 - handle.acc_y1)
            .property('drag_handle', handle)
            .call(addHighlightStyle, true);
      }).on('drag', function(evnt) {
         if (!is_dragging(painter, 'resize')) return;

         evnt.sourceEvent.preventDefault();
         evnt.sourceEvent.stopPropagation();

         const handle = drag_rect.property('drag_handle'),
               elem = d3_select(this);
         let dx = evnt.dx, dy = evnt.dy;

         if (arg.no_change_x) dx = 0;
         if (arg.no_change_y) dy = 0;

         if (elem.classed('js_nw_resize')) {
            handle.acc_x1 += dx; handle.acc_y1 += dy;
         } else if (elem.classed('js_ne_resize')) {
            handle.acc_x2 += dx; handle.acc_y1 += dy;
         } else if (elem.classed('js_sw_resize')) {
            handle.acc_x1 += dx; handle.acc_y2 += dy;
         } else if (elem.classed('js_se_resize')) {
            handle.acc_x2 += dx; handle.acc_y2 += dy;
         } else if (elem.classed('js_w_resize'))
            handle.acc_x1 += dx;
         else if (elem.classed('js_n_resize'))
            handle.acc_y1 += dy;
         else if (elem.classed('js_e_resize'))
            handle.acc_x2 += dx;
         else if (elem.classed('js_s_resize'))
            handle.acc_y2 += dy;

         const x1 = Math.max(0, handle.acc_x1), x2 = Math.min(handle.acc_x2, handle.pad_w),
               y1 = Math.max(0, handle.acc_y1), y2 = Math.min(handle.acc_y2, handle.pad_h);

         handle.x = Math.min(x1, x2);
         handle.y = Math.min(y1, y2);
         handle.width = Math.abs(x2 - x1);
         handle.height = Math.abs(y2 - y1);

         drag_rect.attr('x', handle.x).attr('y', handle.y).attr('width', handle.width).attr('height', handle.height);
      }).on('end', function(evnt) {
         if (!is_dragging(painter, 'resize')) return;

         evnt.sourceEvent.preventDefault();

         const handle = drag_rect.property('drag_handle');

         complete_drag(handle.x, handle.y, handle.width, handle.height);
      });

   if (!arg.only_resize)
      arg.getDrawG().style('cursor', arg.cleanup ? null : 'move').call(arg.cleanup ? drag_move_off : drag_move);

   if (!arg.only_move)
      makeResizeElements(arg.getDrawG(), drag_resize);
}

const TooltipHandler = {

   /** @desc only canvas info_layer can be used while other pads can overlay
     * @return layer where frame tooltips are shown */
   hints_layer() {
      return this.getCanvPainter()?.getLayerSvg('info_layer') ?? d3_select(null);
   },

   /** @return true if tooltip is shown, use to prevent some other action */
   isTooltipShown() {
      if (!this.tooltip_enabled || !this.isTooltipAllowed())
         return false;
      const hintsg = this.hints_layer().selectChild('.objects_hints');
      return hintsg.empty() ? false : hintsg.property('hints_pad') === this.getPadName();
   },

   /** @summary set tooltips enabled on/off */
   setTooltipEnabled(enabled) {
      if (enabled !== undefined)
         this.tooltip_enabled = enabled;
   },

   /** @summary central function which let show selected hints for the object */
   processFrameTooltipEvent(pnt, evnt) {
      if (pnt?.handler) {
         // special use of interactive handler in the frame painter
         const rect = this.draw_g?.selectChild('.main_layer');
         if (!rect || rect.empty())
            pnt = null; // disable
         else if (pnt.touch && evnt) {
            const pos = get_touch_pointers(evnt, rect.node());
            pnt = (pos && pos.length === 1) ? { touch: true, x: pos[0][0], y: pos[0][1] } : null;
         } else if (evnt) {
            const pos = d3_pointer(evnt, rect.node());
            pnt = { touch: false, x: pos[0], y: pos[1] };
         }
      }

      let nhints = 0, nexact = 0, maxlen = 0, lastcolor1 = 0, usecolor1 = false, textheight = 11;
      const hmargin = 3, wmargin = 3, hstep = 1.2,
            frame_rect = this.getFrameRect(),
            pp = this.getPadPainter(),
            pad_width = pp?.getPadWidth(),
            font = new FontHandler(160, textheight),
            disable_tootlips = !this.isTooltipAllowed() || !this.tooltip_enabled;

      if (pnt) {
         pnt.disabled = disable_tootlips; // indicate that highlighting is not required
         pnt.painters = true; // get also painter
      }

      // collect tooltips from pad painter - it has list of all drawn objects
      const hints = pp?.processPadTooltipEvent(pnt) ?? [];

      if (pp?._deliver_webcanvas_events && pp?.is_active_pad && pnt && isFunc(pp?.deliverWebCanvasEvent))
         pp.deliverWebCanvasEvent('move', frame_rect.x + pnt.x, frame_rect.y + pnt.y, hints);

      if (pnt?.touch) textheight = 15;

      for (let n = 0; n < hints.length; ++n) {
         const hint = hints[n];
         if (!hint) continue;

         if (hint.user_info !== undefined)
            hint.painter?.provideUserTooltip(hint.user_info);

         if (!hint.lines || (hint.lines.length === 0)) {
            hints[n] = null;
            continue;
         }

         // check if fully duplicated hint already exists
         for (let k = 0; k < n; ++k) {
            const hprev = hints[k];
            let diff = false;
            if (!hprev || (hprev.lines.length !== hint.lines.length)) continue;
            for (let l = 0; l < hint.lines.length && !diff; ++l)
               if (hprev.lines[l] !== hint.lines[l]) diff = true;
            if (!diff) { hints[n] = null; break; }
         }
         if (!hints[n]) continue;

         nhints++;

         if (hint.exact) nexact++;

         hint.lines.forEach(line => { maxlen = Math.max(maxlen, line.length); });

         hint.height = Math.round(hint.lines.length * textheight * hstep + 2 * hmargin - textheight * (hstep - 1));

         if ((hint.color1 !== undefined) && (hint.color1 !== 'none')) {
            if ((lastcolor1 !== 0) && (lastcolor1 !== hint.color1)) usecolor1 = true;
            lastcolor1 = hint.color1;
         }
      }

      let path_name = null, same_path = hints.length > 1;
      for (let n = 0; n < hints.length; ++n) {
         const hint = hints[n], p = hint?.lines ? hint.lines[0]?.lastIndexOf('/') : -1;
         if (p > 0) {
            const path = hint.lines[0].slice(0, p + 1);
            if (path_name === null)
               path_name = path;
            else if (path_name !== path)
               same_path = false;
         } else
            same_path = false;
      }

      const layer = this.hints_layer(),
            show_only_best = nhints > 15,
            coordinates = pnt ? Math.round(pnt.x) + ',' + Math.round(pnt.y) : '';
      let hintsg = layer.selectChild('.objects_hints'), // group with all tooltips
          title = '', name = '', info = '',
          hint = null, best_dist2 = 1e10, best_hint = null;

      // try to select hint with exact match of the position when several hints available
      for (let k = 0; k < hints.length; ++k) {
         if (!hints[k]) continue;
         if (!hint) hint = hints[k];

         // select exact hint if this is the only one
         if (hints[k].exact && (nexact < 2) && (!hint || !hint.exact)) { hint = hints[k]; break; }

         if (!pnt || (hints[k].x === undefined) || (hints[k].y === undefined)) continue;

         const dist2 = (pnt.x - hints[k].x) ** 2 + (pnt.y - hints[k].y) ** 2;
         if (dist2 < best_dist2) { best_dist2 = dist2; best_hint = hints[k]; }
      }

      if ((!hint || !hint.exact) && (best_dist2 < 400)) hint = best_hint;

      if (hint) {
         name = (hint.lines && hint.lines.length > 1) ? hint.lines[0] : hint.name;
         title = hint.title || '';
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

      if (hintsg.empty()) {
         hintsg = layer.append('svg:g')
            .attr('class', 'objects_hints')
            .style('pointer-events', 'none');
      }

      let frame_shift = { x: 0, y: 0 }, trans = frame_rect.transform || '';
      if (!pp.iscan) {
         frame_shift = getAbsPosInCanvas(this.getPadSvg(), frame_shift);
         trans = `translate(${frame_shift.x},${frame_shift.y}) ${trans}`;
      }

      // copy transform attributes from frame itself
      hintsg.attr('transform', trans)
            .property('last_point', pnt)
            .property('hints_pad', this.getPadName());

      let viewmode = hintsg.property('viewmode') || '',
         actualw = 0, posx = pnt.x + frame_rect.hint_delta_x;

      if (show_only_best || (nhints === 1)) {
         viewmode = 'single';
         posx += 15;
      } else {
         // if there are many hints, place them left or right

         let bleft = 0.5, bright = 0.5;

         if (viewmode === 'left')
            bright = 0.7;
         else if (viewmode === 'right')
            bleft = 0.3;

         if (posx <= bleft * frame_rect.width) {
            viewmode = 'left';
            posx = 20;
         } else if (posx >= bright * frame_rect.width) {
            viewmode = 'right';
            posx = frame_rect.width - 60;
         } else
            posx = hintsg.property('startx');
      }

      if (viewmode !== hintsg.property('viewmode')) {
         hintsg.property('viewmode', viewmode);
         hintsg.selectAll('*').remove();
      }

      let curry = 10, // normal y coordinate
          gapy = 10,  // y coordinate, taking into account all gaps
          gapminx = -1111, gapmaxx = -1111;
      const minhinty = -frame_shift.y,
            cp = this.getCanvPainter(),
            maxhinty = cp.getPadHeight() - frame_rect.y - frame_shift.y;

      for (let n = 0; n < hints.length; ++n) {
         let hint = hints[n],
             group = hintsg.selectChild(`.painter_hint_${n}`);

         if (show_only_best && (hint !== best_hint))
            hint = null;

         if (hint === null) {
            group.remove();
            continue;
         }

         const was_empty = group.empty();

         if (was_empty) {
            group = hintsg.append('svg:svg')
               .attr('class', `painter_hint_${n}`)
               .attr('opacity', 0) // use attribute, not style to make animation with d3.transition()
               .style('overflow', 'hidden')
               .style('pointer-events', 'none');
          }

         if (viewmode === 'single')
            curry = pnt.touch ? (pnt.y - hint.height - 5) : Math.min(pnt.y + 15, maxhinty - hint.height - 3) + frame_rect.hint_delta_y;
          else {
            for (let n = 0; (n < hints.length) && (gapy < maxhinty); ++n) {
               const hint = hints[n];
               if (!hint) continue;
               if ((hint.y >= gapy - 5) && (hint.y <= gapy + hint.height + 5)) {
                  gapy = hint.y + 10;
                  n = -1;
               }
            }
            if ((gapminx === -1111) && (gapmaxx === -1111)) gapminx = gapmaxx = hint.x;
            gapminx = Math.min(gapminx, hint.x);
            gapmaxx = Math.min(gapmaxx, hint.x);
         }

         group.attr('x', posx)
              .attr('y', curry)
              .property('curry', curry)
              .property('gapy', gapy);

         curry += hint.height + 5;
         gapy += hint.height + 5;

         if (!was_empty)
            group.selectAll('*').remove();

         group.attr('width', 60)
              .attr('height', hint.height);

         const r = group.append('rect')
            .attr('x', 0)
            .attr('y', 0)
            .attr('width', 60)
            .attr('height', hint.height)
            .style('fill', 'lightgrey')
            .style('pointer-events', 'none');

         if (nhints > 1) {
            const col = usecolor1 ? hint.color1 : hint.color2;
            if (col && (col !== 'none'))
               r.style('stroke', col);
         }
         r.attr('stroke-width', hint.exact ? 3 : 1);

         for (let l = 0; l < (hint.lines?.length ?? 0); l++) {
            let line = hint.lines[l];
            if (l === 0 && path_name && same_path)
               line = line.slice(path_name.length);
            if (line) {
               const txt = group.append('svg:text')
                  .attr('text-anchor', 'start')
                  .attr('x', wmargin)
                  .attr('y', hmargin + l * textheight * hstep)
                  .attr('dy', '.8em')
                  .style('fill', 'black')
                  .style('pointer-events', 'none')
                  .call(font.func)
                  .text(line),
               box = getElementRect(txt, 'bbox');

               actualw = Math.max(actualw, box.width);
            }
         }

         function translateFn() {
            // We only use 'd', but list d,i,a as params just to show can have them as params.
            // Code only really uses d and t.
            return function(/* d, i, a */) {
               return function(t) {
                  return t < 0.8 ? '0' : (t - 0.8) * 5;
               };
            };
         }

         if (was_empty) {
            if (settings.TooltipAnimation > 0)
               group.transition().duration(settings.TooltipAnimation).attrTween('opacity', translateFn());
            else
               group.attr('opacity', 1);
         }
      }

      actualw += 2 * wmargin;

      const svgs = hintsg.selectAll('svg');

      if ((viewmode === 'right') && (posx + actualw > frame_rect.width - 20)) {
         posx = frame_rect.width - actualw - 20;
         svgs.attr('x', posx);
      }

      if ((viewmode === 'single') && (posx + actualw > pad_width - frame_rect.x) && (posx > actualw + 20)) {
         posx -= (actualw + 20);
         svgs.attr('x', posx);
      }

      // if gap not very big, apply gapy coordinate to open view on the histogram
      if ((viewmode !== 'single') && (gapy < maxhinty) && (gapy !== curry)) {
         if ((gapminx <= posx + actualw + 5) && (gapmaxx >= posx - 5))
            svgs.attr('y', function() { return d3_select(this).property('gapy'); });
      } else if ((viewmode !== 'single') && (curry > maxhinty)) {
         const shift = Math.max((maxhinty - curry - 10), minhinty);
         if (shift < 0)
            svgs.attr('y', function() { return d3_select(this).property('curry') + shift; });
      }

      if (actualw > 10)
         svgs.attr('width', actualw).select('rect').attr('width', actualw);

      hintsg.property('startx', posx);

      if (cp._highlight_connect && isFunc(cp.processHighlightConnect))
         cp.processHighlightConnect(hints);
   },

   /** @summary Assigns tooltip methods */
   assign(painter) {
      Object.assign(painter, this, { tooltip_enabled: true });
   }

}, // TooltipHandler


/** @summary Set of frame interactivity methods
  * @private */

 FrameInteractive = {

   /** @summary Adding basic interactivity */
   addBasicInteractivity() {
      TooltipHandler.assign(this);

      if (!this._frame_rotate && !this._frame_fixpos) {
         addDragHandler(this, { obj: this, x: this._frame_x, y: this._frame_y, width: this.getFrameWidth(), height: this.getFrameHeight(),
                                is_disabled: kind => { return (kind === 'move') && this.mode3d; },
                                only_resize: true, minwidth: 20, minheight: 20, redraw: () => this.sizeChanged() });
      }

      const top_rect = this.draw_g.selectChild('path'),
            main_svg = this.draw_g.selectChild('.main_layer');

      top_rect.style('pointer-events', 'visibleFill')  // let process mouse events inside frame
              .style('cursor', 'default');             // show normal cursor

      main_svg.style('pointer-events', 'visibleFill')
              .style('cursor', 'default')
              .property('handlers_set', 0);

      const pp = this.getPadPainter(),
          handlers_set = pp?._fast_drawing ? 0 : 1;

      if (main_svg.property('handlers_set') !== handlers_set) {
         const close_handler = handlers_set ? this.processFrameTooltipEvent.bind(this, null) : null,
               mouse_handler = handlers_set ? this.processFrameTooltipEvent.bind(this, { handler: true, touch: false }) : null;

         main_svg.property('handlers_set', handlers_set)
                 .on('mouseenter', mouse_handler)
                 .on('mousemove', mouse_handler)
                 .on('mouseleave', close_handler);

         if (browser.touches) {
            const touch_handler = handlers_set ? this.processFrameTooltipEvent.bind(this, { handler: true, touch: true }) : null;

            main_svg.on('touchstart', touch_handler)
                    .on('touchmove', touch_handler)
                    .on('touchend', close_handler)
                    .on('touchcancel', close_handler);
         }
      }

      main_svg.attr('x', 0)
              .attr('y', 0)
              .attr('width', this.getFrameWidth())
              .attr('height', this.getFrameHeight());

      const hintsg = this.hints_layer().selectChild('.objects_hints');
      // if tooltips were visible before, try to reconstruct them after short timeout
      if (!hintsg.empty() && this.isTooltipAllowed() && (hintsg.property('hints_pad') === this.getPadName()))
         setTimeout(this.processFrameTooltipEvent.bind(this, hintsg.property('last_point'), null), 10);
   },

   /** @summary Add interactive handlers */
   async addFrameInteractivity(for_second_axes) {
      const pp = this.getPadPainter(),
            svg = this.getFrameSvg();
      if (pp?._fast_drawing || svg.empty())
         return this;

      if (for_second_axes) {
         // add extra handlers for second axes
         const svg_x2 = svg.selectAll('.x2axis_container'),
               svg_y2 = svg.selectAll('.y2axis_container');
         if (settings.ContextMenu) {
            svg_x2.on('contextmenu', evnt => this.showContextMenu('x2', evnt));
            svg_y2.on('contextmenu', evnt => this.showContextMenu('y2', evnt));
         }
         svg_x2.on('mousemove', evnt => this.showAxisStatus('x2', evnt));
         svg_y2.on('mousemove', evnt => this.showAxisStatus('y2', evnt));
         return this;
      }

      const svg_x = svg.selectAll('.xaxis_container'),
            svg_y = svg.selectAll('.yaxis_container');

      this.can_zoom_x = this.can_zoom_y = settings.Zooming;

      if (pp?.options) {
         if (pp.options.NoZoomX) this.can_zoom_x = false;
         if (pp.options.NoZoomY) this.can_zoom_y = false;
      }

      if (!svg.property('interactive_set')) {
         this.addFrameKeysHandler();

         this.zoom_kind = 0; // 0 - none, 1 - XY, 2 - only X, 3 - only Y, (+100 for touches)
         this.zoom_rect = null;
         this.zoom_origin = null;  // original point where zooming started
         this.zoom_curr = null;    // current point for zooming
      }

      if (settings.Zooming) {
         if (settings.ZoomMouse) {
            svg.on('mousedown', evnt => this.startRectSel(evnt));
            svg.on('dblclick', evnt => this.mouseDoubleClick(evnt));
         }
         if (settings.ZoomWheel)
            svg.on('wheel', evnt => this.mouseWheel(evnt));
      }

      if (browser.touches && ((settings.Zooming && settings.ZoomTouch) || settings.ContextMenu))
         svg.on('touchstart', evnt => this.startTouchZoom(evnt));

      if (settings.ContextMenu) {
         if (browser.touches) {
            svg_x.on('touchstart', evnt => this.startSingleTouchHandling('x', evnt));
            svg_y.on('touchstart', evnt => this.startSingleTouchHandling('y', evnt));
         }
         svg.on('contextmenu', evnt => this.showContextMenu('', evnt));
         svg_x.on('contextmenu', evnt => this.showContextMenu('x', evnt));
         svg_y.on('contextmenu', evnt => this.showContextMenu('y', evnt));
      }

      svg_x.on('mousemove', evnt => this.showAxisStatus('x', evnt));
      svg_y.on('mousemove', evnt => this.showAxisStatus('y', evnt));

      svg.property('interactive_set', true);

      return this;
   },

   /** @summary Add keys handler */
   addFrameKeysHandler() {
      if (this.keys_handler || (typeof window === 'undefined')) return;

      this.keys_handler = evnt => this.processKeyPress(evnt);

      window.addEventListener('keydown', this.keys_handler, false);
   },

   /** @summary Handle key press */
   processKeyPress(evnt) {
      // no custom keys handling when menu is present
      if (hasMenu())
         return true;

      const allowed = ['PageUp', 'PageDown', 'ArrowLeft', 'ArrowUp', 'ArrowRight', 'ArrowDown', 'PrintScreen', 'Escape', '*'],
            main = this.selectDom(),
            pp = this.getPadPainter();
      let key = evnt.key;

      if (!settings.HandleKeys || main.empty() || (this.enabledKeys === false) ||
          (getActivePad() !== pp) || (allowed.indexOf(key) < 0)) return false;

      if (evnt.shiftKey) key = `Shift ${key}`;
      if (evnt.altKey) key = `Alt ${key}`;
      if (evnt.ctrlKey) key = `Ctrl ${key}`;

      const zoom = { name: 'x', dleft: 0, dright: 0 };

      switch (key) {
         case 'ArrowLeft': zoom.dleft = -1; zoom.dright = 1; break;
         case 'ArrowRight': zoom.dleft = 1; zoom.dright = -1; break;
         case 'Ctrl ArrowLeft': zoom.dleft = zoom.dright = -1; break;
         case 'Ctrl ArrowRight': zoom.dleft = zoom.dright = 1; break;
         case 'ArrowUp': zoom.name = 'y'; zoom.dleft = 1; zoom.dright = -1; break;
         case 'ArrowDown': zoom.name = 'y'; zoom.dleft = -1; zoom.dright = 1; break;
         case 'Ctrl ArrowUp': zoom.name = 'y'; zoom.dleft = zoom.dright = 1; break;
         case 'Ctrl ArrowDown': zoom.name = 'y'; zoom.dleft = zoom.dright = -1; break;
         case 'Escape': pp?.enlargePad(null, false, true); return true;
      }

      if (zoom.dleft || zoom.dright) {
         if (!settings.Zooming) return false;
         // in 3dmode with orbit control ignore simple arrows
         if (this.mode3d && (key.indexOf('Ctrl') !== 0)) return false;
         this.analyzeMouseWheelEvent(null, zoom, 0.5);
         if (zoom.changed) {
            this.zoom(zoom.name, zoom.min, zoom.max);
            this.zoomChangedInteractive(zoom.name, true);
         }
         evnt.stopPropagation();
         evnt.preventDefault();
      } else {
         const func = pp?.findPadButton(key);
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
      const pp = this.getPadPainter();
      if (!pp) return;

      pnt.painters = true; // provide painters reference in the hints
      pnt.disabled = true; // do not invoke graphics

      // collect tooltips from pad painter - it has list of all drawn objects
      const hints = pp.processPadTooltipEvent(pnt);
      let exact = null, res;
      for (let k = 0; (k < hints.length) && !exact; ++k) {
         if (hints[k] && hints[k].exact)
            exact = hints[k];
      }

      if (exact) {
         const handler = dblckick ? this._dblclick_handler : this._click_handler;
         if (handler) res = handler(exact.user_info, pnt);
      }

      if (!dblckick) {
         pp.selectObjectPainter(exact ? exact.painter : this,
               { x: pnt.x + (this._frame_x || 0), y: pnt.y + (this._frame_y || 0) });
      }

      return res;
   },

   /** @summary Check mouse moving  */
   shiftMoveHanlder(evnt, pos0) {
      if (evnt.buttons === this._shifting_buttons) {
         const frame = this.getFrameSvg(),
             pos = d3_pointer(evnt, frame.node()),
             main_svg = this.draw_g.selectChild('.main_layer'),
             dx = pos0[0] - pos[0],
             dy = (this.scales_ndim === 1) ? 0 : pos0[1] - pos[1],
             w = this.getFrameWidth(), h = this.getFrameHeight();

         this._shifting_dx = dx;
         this._shifting_dy = dy;

         main_svg.attr('viewBox', `${dx} ${dy} ${w} ${h}`);

         evnt.preventDefault();
         evnt.stopPropagation();
      }
   },

   /** @summary mouse up handler for shifting */
   shiftUpHanlder(evnt) {
      evnt.preventDefault();

      d3_select(window).on('mousemove.shiftHandler', null)
                       .on('mouseup.shiftHandler', null);

      if ((this._shifting_dx !== undefined) && (this._shifting_dy !== undefined))
         this.performScalesShift();
    },

    /** @summary Shift scales on defined positions */
   performScalesShift() {
      const w = this.getFrameWidth(), h = this.getFrameHeight(),
          main_svg = this.draw_g.selectChild('.main_layer'),
          gr = this.getGrFuncs(),
          xmin = gr.revertAxis('x', this._shifting_dx),
          xmax = gr.revertAxis('x', this._shifting_dx + w),
          ymin = gr.revertAxis('y', this._shifting_dy + h),
          ymax = gr.revertAxis('y', this._shifting_dy);

      main_svg.attr('viewBox', `0 0 ${w} ${h}`);

      delete this._shifting_dx;
      delete this._shifting_dy;

      setPainterTooltipEnabled(this, true);

      if (this.scales_ndim === 1)
         this.zoomSingle('x', xmin, xmax);
      else
         this.zoom(xmin, xmax, ymin, ymax);
   },

   /** @summary Start mouse rect zooming */
   startRectSel(evnt) {
      // ignore when touch selection is activated
      if (this.zoom_kind > 100) return;

      const frame = this.getFrameSvg(),
            pos = d3_pointer(evnt, frame.node());

      if ((evnt.buttons === 3) || (evnt.button === 1)) {
         this.clearInteractiveElements();
         this._shifting_buttons = evnt.buttons;

         if (!evnt.$emul) {
            d3_select(window).on('mousemove.shiftHandler', evnt => this.shiftMoveHanlder(evnt, pos))
                             .on('mouseup.shiftHandler', evnt => this.shiftUpHanlder(evnt), true);
         }

         setPainterTooltipEnabled(this, false);
         evnt.preventDefault();
         evnt.stopPropagation();
         return;
      }

      // ignore all events from non-left button
      if (evnt.button !== 0) return;

      evnt.preventDefault();

      this.clearInteractiveElements();

      const w = this.getFrameWidth(), h = this.getFrameHeight();

      this.zoom_lastpos = pos;
      this.zoom_curr = [Math.max(0, Math.min(w, pos[0])), Math.max(0, Math.min(h, pos[1]))];

      this.zoom_origin = [0, 0];
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

      if (!evnt.$emul) {
         d3_select(window).on('mousemove.zoomRect', evnt => this.moveRectSel(evnt))
                          .on('mouseup.zoomRect', evnt => this.endRectSel(evnt), true);
      }

      this.zoom_rect = null;

      // disable tooltips in frame painter
      setPainterTooltipEnabled(this, false);

      evnt.stopPropagation();

      if (this.zoom_kind !== 1)
         return postponePromise(() => this.startLabelsMove(), 500);
   },

   /** @summary Starts labels move */
   startLabelsMove() {
      if (this.zoom_rect) return;

      const handle = (this.zoom_kind === 2) ? this.x_handle : this.y_handle;

      if (!isFunc(handle?.processLabelsMove) || !this.zoom_lastpos) return;

      if (handle.processLabelsMove('start', this.zoom_lastpos))
         this.zoom_labels = handle;
   },

   /** @summary Process mouse rect zooming */
   moveRectSel(evnt) {
      if ((this.zoom_kind === 0) || (this.zoom_kind > 100)) return;

      evnt.preventDefault();
      const m = d3_pointer(evnt, this.getFrameSvg().node());

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

      const x = Math.min(this.zoom_origin[0], this.zoom_curr[0]),
            y = Math.min(this.zoom_origin[1], this.zoom_curr[1]),
            w = Math.abs(this.zoom_curr[0] - this.zoom_origin[0]),
            h = Math.abs(this.zoom_curr[1] - this.zoom_origin[1]);

      if (!this.zoom_rect) {
         // ignore small changes, can be switching to labels move
         if ((this.zoom_kind !== 1) && ((w < 2) || (h < 2))) return;

         this.zoom_rect = this.getFrameSvg()
                              .append('rect')
                              .style('pointer-events', 'none')
                              .call(addHighlightStyle, true);
      }

      this.zoom_rect.attr('x', x).attr('y', y).attr('width', w).attr('height', h);
   },

   /** @summary Finish mouse rect zooming */
   endRectSel(evnt) {
      if ((this.zoom_kind === 0) || (this.zoom_kind > 100)) return;

      evnt.preventDefault();

      if (!evnt.$emul) {
         d3_select(window).on('mousemove.zoomRect', null)
                          .on('mouseup.zoomRect', null);
      }

      const m = d3_pointer(evnt, this.getFrameSvg().node());
      let kind = this.zoom_kind, pr;

      if (this.zoom_labels)
         this.zoom_labels.processLabelsMove('stop', m);
       else {
         const changed = [this.can_zoom_x, this.can_zoom_y];
         m[0] = Math.max(0, Math.min(this.getFrameWidth(), m[0]));
         m[1] = Math.max(0, Math.min(this.getFrameHeight(), m[1]));

         switch (this.zoom_kind) {
            case 1: this.zoom_curr[0] = m[0]; this.zoom_curr[1] = m[1]; break;
            case 2: this.zoom_curr[0] = m[0]; changed[1] = false; break; // only X
            case 3: this.zoom_curr[1] = m[1]; changed[0] = false; break; // only Y
         }

         const idx = this.swap_xy ? 1 : 0, idy = 1 - idx;
         let xmin, xmax, ymin, ymax, isany = false,
             namex = 'x', namey = 'y';

         if (changed[idx] && (Math.abs(this.zoom_curr[idx] - this.zoom_origin[idx]) > 10)) {
            if (this.zoom_second && (this.zoom_kind === 2)) namex = 'x2';
            xmin = Math.min(this.revertAxis(namex, this.zoom_origin[idx]), this.revertAxis(namex, this.zoom_curr[idx]));
            xmax = Math.max(this.revertAxis(namex, this.zoom_origin[idx]), this.revertAxis(namex, this.zoom_curr[idx]));
            isany = true;
         }

         if (changed[idy] && (Math.abs(this.zoom_curr[idy] - this.zoom_origin[idy]) > 10)) {
            if (this.zoom_second && (this.zoom_kind === 3)) namey = 'y2';
            ymin = Math.min(this.revertAxis(namey, this.zoom_origin[idy]), this.revertAxis(namey, this.zoom_curr[idy]));
            ymax = Math.max(this.revertAxis(namey, this.zoom_origin[idy]), this.revertAxis(namey, this.zoom_curr[idy]));
            isany = true;
         }

         if (namex === 'x2') {
            this.zoomChangedInteractive(namex, true);
            pr = this.zoomSingle(namex, xmin, xmax);
            kind = 0;
         } else if (namey === 'y2') {
            this.zoomChangedInteractive(namey, true);
            pr = this.zoomSingle(namey, ymin, ymax);
            kind = 0;
         } else if (isany) {
            this.zoomChangedInteractive('x', true);
            this.zoomChangedInteractive('y', true);
            pr = this.zoom(xmin, xmax, ymin, ymax);
            kind = 0;
         }
      }

      const pnt = (kind === 1) ? { x: this.zoom_origin[0], y: this.zoom_origin[1] } : null;

      this.clearInteractiveElements();

      // if no zooming was done, select active object instead
      switch (kind) {
         case 1:
            this.processFrameClick(pnt);
            break;
         case 2:
            this.getPadPainter()?.selectObjectPainter(this, null, 'xaxis');
            break;
         case 3:
            this.getPadPainter()?.selectObjectPainter(this, null, 'yaxis');
            break;
      }

      // return promise - if any
      return pr;
   },

   /** @summary Handle mouse double click on frame */
   mouseDoubleClick(evnt) {
      evnt.preventDefault();
      const m = d3_pointer(evnt, this.getFrameSvg().node()),
            fw = this.getFrameWidth(), fh = this.getFrameHeight();
      this.clearInteractiveElements();

      const valid_x = (m[0] >= 0) && (m[0] <= fw),
            valid_y = (m[1] >= 0) && (m[1] <= fh);

      if (valid_x && valid_y && this._dblclick_handler)
         if (this.processFrameClick({ x: m[0], y: m[1] }, true)) return;

      let kind = (this.can_zoom_x ? 'x' : '') + (this.can_zoom_y ? 'y' : '') + 'z';
      if (!valid_x) {
         if (!this.can_zoom_y) return;
         kind = this.swap_xy ? 'x' : 'y';
         if ((m[0] > fw) && this[kind+'2_handle']) kind += '2'; // let unzoom second axis
      } else if (!valid_y) {
         if (!this.can_zoom_x) return;
         kind = this.swap_xy ? 'y' : 'x';
         if ((m[1] < 0) && this[kind+'2_handle']) kind += '2'; // let unzoom second axis
      }
      return this.unzoom(kind).then(changed => {
         if (changed) return;
         const pp = this.getPadPainter(), rect = this.getFrameRect();
         return pp?.selectObjectPainter(pp, { x: m[0] + rect.x, y: m[1] + rect.y, dbl: true });
      });
   },

   /** @summary Start touch zoom */
   startTouchZoom(evnt) {
      evnt.preventDefault();
      evnt.stopPropagation();

      // in case when zooming was started, block any other kind of events
      // also prevent zooming together with active dragging
      if ((this.zoom_kind !== 0) || drag_kind)
         return;

      const arr = get_touch_pointers(evnt, this.getFrameSvg().node());

      // normally double-touch will be handled
      // touch with single click used for context menu
      if (arr.length === 1) {
         // this is touch with single element

         const now = new Date().getTime();
         let tmdiff = 1e10, dx = 100, dy = 100;

         if (this.last_touch_time && this.last_touch_pos) {
            tmdiff = now - this.last_touch_time;
            dx = Math.abs(arr[0][0] - this.last_touch_pos[0]);
            dy = Math.abs(arr[0][1] - this.last_touch_pos[1]);
         }

         this.last_touch_time = now;
         this.last_touch_pos = arr[0];

         if ((tmdiff < 500) && (dx < 20) && (dy < 20)) {
            this.clearInteractiveElements();
            this.unzoom('xyz');

            delete this.last_touch_time;
         } else if (settings.ContextMenu)
            this.startSingleTouchHandling('', evnt);
      }

      if ((arr.length !== 2) || !settings.Zooming || !settings.ZoomTouch)
         return;

      this.clearInteractiveElements();

      // clear single touch handler
      this.endSingleTouchHandling(null);

      const pnt1 = arr[0], pnt2 = arr[1], w = this.getFrameWidth(), h = this.getFrameHeight();

      this.zoom_curr = [Math.min(pnt1[0], pnt2[0]), Math.min(pnt1[1], pnt2[1])];
      this.zoom_origin = [Math.max(pnt1[0], pnt2[0]), Math.max(pnt1[1], pnt2[1])];
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
      } else
         this.zoom_kind = 101; // x and y

      drag_kind = 'zoom'; // block other possible dragging

      setPainterTooltipEnabled(this, false);

      this.zoom_rect = this.getFrameSvg().append('rect')
            .attr('id', 'zoomRect')
            .attr('x', this.zoom_curr[0])
            .attr('y', this.zoom_curr[1])
            .attr('width', this.zoom_origin[0] - this.zoom_curr[0])
            .attr('height', this.zoom_origin[1] - this.zoom_curr[1])
            .call(addHighlightStyle, true);

      if (!evnt.$emul) {
         d3_select(window).on('touchmove.zoomRect', evnt => this.moveTouchZoom(evnt))
                          .on('touchcancel.zoomRect', evnt => this.endTouchZoom(evnt))
                          .on('touchend.zoomRect', evnt => this.endTouchZoom(evnt));
      }
   },

   /** @summary Move touch zooming */
   moveTouchZoom(evnt) {
      if (this.zoom_kind < 100) return;

      evnt.preventDefault();

      const arr = get_touch_pointers(evnt, this.getFrameSvg().node());

      if (arr.length !== 2)
         return this.clearInteractiveElements();

      const pnt1 = arr[0], pnt2 = arr[1];

      if (this.zoom_kind !== 103) {
         this.zoom_curr[0] = Math.min(pnt1[0], pnt2[0]);
         this.zoom_origin[0] = Math.max(pnt1[0], pnt2[0]);
      }
      if (this.zoom_kind !== 102) {
         this.zoom_curr[1] = Math.min(pnt1[1], pnt2[1]);
         this.zoom_origin[1] = Math.max(pnt1[1], pnt2[1]);
      }

      this.zoom_rect.attr('x', this.zoom_curr[0])
                     .attr('y', this.zoom_curr[1])
                     .attr('width', this.zoom_origin[0] - this.zoom_curr[0])
                     .attr('height', this.zoom_origin[1] - this.zoom_curr[1]);

      if ((this.zoom_origin[0] - this.zoom_curr[0] > 10) || (this.zoom_origin[1] - this.zoom_curr[1] > 10))
         setPainterTooltipEnabled(this, false);

      evnt.stopPropagation();
   },

   /** @summary End touch zooming handler */
   endTouchZoom(evnt) {
      if (this.zoom_kind < 100) return;

      drag_kind = ''; // reset global flag

      evnt.preventDefault();
      if (!evnt.$emul) {
         d3_select(window).on('touchmove.zoomRect', null)
                          .on('touchend.zoomRect', null)
                          .on('touchcancel.zoomRect', null);
      }

      let xmin, xmax, ymin, ymax, isany = false, namex = 'x', namey = 'y';
      const xid = this.swap_xy ? 1 : 0, yid = 1 - xid, changed = [true, true];

      if (this.zoom_kind === 102) changed[1] = false;
      if (this.zoom_kind === 103) changed[0] = false;

      if (changed[xid] && (Math.abs(this.zoom_curr[xid] - this.zoom_origin[xid]) > 10)) {
         if (this.zoom_second && (this.zoom_kind === 102)) namex = 'x2';
         xmin = Math.min(this.revertAxis(namex, this.zoom_origin[xid]), this.revertAxis(namex, this.zoom_curr[xid]));
         xmax = Math.max(this.revertAxis(namex, this.zoom_origin[xid]), this.revertAxis(namex, this.zoom_curr[xid]));
         isany = true;
      }

      if (changed[yid] && (Math.abs(this.zoom_curr[yid] - this.zoom_origin[yid]) > 10)) {
         if (this.zoom_second && (this.zoom_kind === 103)) namey = 'y2';
         ymin = Math.min(this.revertAxis(namey, this.zoom_origin[yid]), this.revertAxis(namey, this.zoom_curr[yid]));
         ymax = Math.max(this.revertAxis(namey, this.zoom_origin[yid]), this.revertAxis(namey, this.zoom_curr[yid]));
         isany = true;
      }

      this.clearInteractiveElements();
      delete this.last_touch_time;

      if (namex === 'x2') {
         this.zoomChangedInteractive(namex, true);
         this.zoomSingle(namex, xmin, xmax);
      } else if (namey === 'y2') {
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
      const handle2 = second_side ? this[item.name + '2_handle'] : null;
      if (handle2) {
         item.second = Object.assign({}, item);
         return handle2.analyzeWheelEvent(event, dmin, item.second, test_ignore);
      }
      const handle = this[item.name + '_handle'];
      return handle?.analyzeWheelEvent(event, dmin, item, test_ignore);
   },

    /** @summary return true if default Y zooming should be enabled
      * @desc it is typically for 2-Dim histograms or
      * when histogram not draw, defined by other painters */
   isAllowedDefaultYZooming() {
      if (this.self_drawaxes) return true;

      const pad_painter = this.getPadPainter();
      if (pad_painter?.painters) {
         for (let k = 0; k < pad_painter.painters.length; ++k) {
            const subpainter = pad_painter.painters[k];
            if (subpainter?.wheel_zoomy !== undefined)
               return subpainter.wheel_zoomy;
         }
      }

      return false;
   },

   /** @summary Handles mouse wheel event */
   mouseWheel(evnt) {
      evnt.stopPropagation();
      evnt.preventDefault();
      this.clearInteractiveElements();

      const itemx = { name: 'x', reverse: this.reverse_x },
            itemy = { name: 'y', reverse: this.reverse_y, ignore: !this.isAllowedDefaultYZooming() },
            cur = d3_pointer(evnt, this.getFrameSvg().node()),
            w = this.getFrameWidth(), h = this.getFrameHeight();

      if (this.can_zoom_x)
         this.analyzeMouseWheelEvent(evnt, this.swap_xy ? itemy : itemx, cur[0] / w, (cur[1] >= 0) && (cur[1] <= h), cur[1] < 0);

      if (this.can_zoom_y)
         this.analyzeMouseWheelEvent(evnt, this.swap_xy ? itemx : itemy, 1 - cur[1] / h, (cur[0] >= 0) && (cur[0] <= w), cur[0] > w);

      let pr = this.zoom(itemx.min, itemx.max, itemy.min, itemy.max);

      if (itemx.changed) this.zoomChangedInteractive('x', true);
      if (itemy.changed) this.zoomChangedInteractive('y', true);

      if (itemx.second) {
         pr = pr.then(() => this.zoomSingle('x2', itemx.second.min, itemx.second.max));
         if (itemx.second.changed) this.zoomChangedInteractive('x2', true);
      }
      if (itemy.second) {
         pr = pr.then(() => this.zoomSingle('y2', itemy.second.min, itemy.second.max));
         if (itemy.second.changed) this.zoomChangedInteractive('y2', true);
      }

      return pr;
   },

   /** @summary Show frame context menu */
   showContextMenu(kind, evnt, obj) {
      // disable context menu left/right buttons clicked
      if (evnt?.buttons === 3)
         return evnt.preventDefault();

      // ignore context menu when touches zooming is ongoing or
      if (('zoom_kind' in this) && (this.zoom_kind > 100)) return;

      let pnt, menu_painter = this, exec_painter = null,
          frame_corner = false, fp = null; // object used to show context menu
      const svg_node = this.getFrameSvg().node();

      if (isFunc(evnt?.stopPropagation)) {
         evnt.preventDefault();
         evnt.stopPropagation(); // disable main context menu
         const ms = d3_pointer(evnt, svg_node),
               tch = get_touch_pointers(evnt, svg_node);
         if (tch.length === 1)
             pnt = { x: tch[0][0], y: tch[0][1], touch: true };
         else if (ms.length === 2)
             pnt = { x: ms[0], y: ms[1], touch: false };
       } else if ((evnt?.x !== undefined) && (evnt?.y !== undefined) && (evnt?.clientX === undefined)) {
          pnt = evnt;
          const rect = svg_node.getBoundingClientRect();
          evnt = { clientX: rect.left + pnt.x, clientY: rect.top + pnt.y };
       }

       if ((kind === 'painter') && obj) {
          menu_painter = obj;
          kind = '';
       } else if (kind === 'main') {
          menu_painter = this.getMainPainter(true);
          kind = '';
       } else if (!kind) {
         const pp = this.getPadPainter();
         let sel = null;

         fp = this;
         if (pnt && pp) {
            pnt.painters = true; // assign painter for every tooltip
            const hints = pp.processPadTooltipEvent(pnt);
            let bestdist = 1000;
            for (let n = 0; n < hints.length; ++n) {
               if (hints[n]?.menu) {
                  const dist = hints[n].menu_dist ?? 7;
                  if (dist < bestdist) { sel = hints[n].painter; bestdist = dist; }
               }
            }
         }

         if (sel) menu_painter = sel;
             else kind = 'frame';

         if (pnt) frame_corner = (pnt.x > 0) && (pnt.x < 20) && (pnt.y > 0) && (pnt.y < 20);

         fp.setLastEventPos(pnt);
      } else if ((kind === 'x') || (kind === 'y') || (kind === 'z') || (kind === 'pal')) {
         exec_painter = this.getMainPainter(true); // histogram painter delivers items for axis menu

         if (this.v7_frame && isFunc(exec_painter?.v7EvalAttr))
            exec_painter = null;
      }

      if (!exec_painter) exec_painter = menu_painter;

      if (!isFunc(menu_painter?.fillContextMenu)) return;

      this.clearInteractiveElements();

      return createMenu(evnt, menu_painter).then(menu => {
         let domenu = menu.painter.fillContextMenu(menu, kind, obj);

         // fill frame menu by default - or append frame elements when activated in the frame corner
         if (fp && (!domenu || (frame_corner && (kind !== 'frame'))))
            domenu = fp.fillContextMenu(menu);

         if (domenu) {
            return exec_painter.fillObjectExecMenu(menu, kind).then(menu => {
                // suppress any running zooming
                setPainterTooltipEnabled(menu.painter, false);
                return menu.show().then(() => setPainterTooltipEnabled(menu.painter, true));
            });
         }
      });
   },

  /** @summary Activate touch handling on frame
    * @private */
   startSingleTouchHandling(kind, evnt) {
      const arr = get_touch_pointers(evnt, this.getFrameSvg().node());
      if (arr.length !== 1) return;

      evnt.preventDefault();
      evnt.stopPropagation();
      closeMenu();

      const tm = new Date().getTime();

      this._shifting_dx = 0;
      this._shifting_dy = 0;

      setPainterTooltipEnabled(this, false);

      d3_select(window).on('touchmove.singleTouch', kind ? null : evnt => this.moveTouchHandling(evnt, kind, arr[0]))
                       .on('touchcancel.singleTouch', evnt => this.endSingleTouchHandling(evnt, kind, arr[0], tm))
                       .on('touchend.singleTouch', evnt => this.endSingleTouchHandling(evnt, kind, arr[0], tm));
   },

   /** @summary Moving of touch pointer
    * @private */
   moveTouchHandling(evnt, kind, pos0) {
      const frame = this.getFrameSvg(),
            main_svg = this.draw_g.selectChild('.main_layer');
      let pos;

      try {
        pos = get_touch_pointers(evnt, frame.node())[0];
      } catch (err) {
        pos = [0, 0];
        if (evnt?.changedTouches)
           pos = [evnt.changedTouches[0].clientX, evnt.changedTouches[0].clientY];
      }

      const dx = pos0[0] - pos[0],
            dy = (this.scales_ndim === 1) ? 0 : pos0[1] - pos[1],
            w = this.getFrameWidth(), h = this.getFrameHeight();

      this._shifting_dx = dx;
      this._shifting_dy = dy;

      main_svg.attr('viewBox', `${dx} ${dy} ${w} ${h}`);
   },

   /** @summary Process end-touch event, which can cause content menu to appear
    * @private */
   endSingleTouchHandling(evnt, kind, pos, tm) {
      evnt?.preventDefault();
      evnt?.stopPropagation();

      setPainterTooltipEnabled(this, true);

      d3_select(window).on('touchmove.singleTouch', null)
                       .on('touchcancel.singleTouch', null)
                       .on('touchend.singleTouch', null);

      if (evnt === null) return;

      if (Math.abs(this._shifting_dx) > 2 || Math.abs(this._shifting_dy) > 2)
         this.performScalesShift();
       else if (new Date().getTime() - tm > 700)
         this.showContextMenu(kind, { x: pos[0], y: pos[1] });
   },

   /** @summary Clear frame interactive elements */
   clearInteractiveElements() {
      closeMenu();
      this.zoom_kind = 0;
      this.zoom_rect?.remove();
      delete this.zoom_rect;
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

}; // FrameInterative


/**
 * @summary Painter class for TFrame, main handler for interactivity
 * @private
 */

class TFramePainter extends ObjectPainter {

   /** @summary constructor
     * @param {object|string} dom - DOM element for drawing or element id
     * @param {object} tframe - TFrame object */
   constructor(dom, tframe) {
      super(dom, (tframe && tframe.$dummy) ? null : tframe);
      this.zoom_kind = 0;
      this.mode3d = false;
      this.shrink_frame_left = 0.0;
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
      if (place === 'xaxis') return this.xaxis;
      if (place === 'yaxis') return this.yaxis;
      return super.getObject();
   }

   /** @summary Set active flag for frame - can block some events
     * @private */
   setFrameActive(on) {
      this.enabledKeys = on && settings.HandleKeys;
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
   getProjectionFunc() { return getEarthProjectionFunc(this.projection); }

   /** @summary Rcalculate frame ranges using specified projection functions */
   recalculateRange(Proj, change_x, change_y) {
      this.projection = Proj || 0;

      if ((this.projection === 2) && ((this.scale_ymin <= -90) || (this.scale_ymax >= 90))) {
         console.warn(`Mercator Projection: Latitude out of range ${this.scale_ymin} ${this.scale_ymax}`);
         this.projection = 0;
      }

      const func = this.getProjectionFunc();
      if (!func) return;

      const pnts = [func(this.scale_xmin, this.scale_ymin),
                   func(this.scale_xmin, this.scale_ymax),
                   func(this.scale_xmax, this.scale_ymax),
                   func(this.scale_xmax, this.scale_ymin)];
      if (this.scale_xmin < 0 && this.scale_xmax > 0) {
         pnts.push(func(0, this.scale_ymin));
         pnts.push(func(0, this.scale_ymax));
      }
      if (this.scale_ymin < 0 && this.scale_ymax > 0) {
         pnts.push(func(this.scale_xmin, 0));
         pnts.push(func(this.scale_xmax, 0));
      }

      this.original_xmin = this.scale_xmin;
      this.original_xmax = this.scale_xmax;
      this.original_ymin = this.scale_ymin;
      this.original_ymax = this.scale_ymax;

      if (change_x)
         this.scale_xmin = this.scale_xmax = pnts[0].x;
      if (change_y)
         this.scale_ymin = this.scale_ymax = pnts[0].y;

      for (let n = 1; n < pnts.length; ++n) {
         if (change_x) {
            this.scale_xmin = Math.min(this.scale_xmin, pnts[n].x);
            this.scale_xmax = Math.max(this.scale_xmax, pnts[n].x);
         }
         if (change_y) {
            this.scale_ymin = Math.min(this.scale_ymin, pnts[n].y);
            this.scale_ymax = Math.max(this.scale_ymax, pnts[n].y);
         }
      }
   }

   /** @summary Configure frame axes ranges */
   setAxesRanges(xaxis, xmin, xmax, yaxis, ymin, ymax, zaxis, zmin, zmax, hpainter) {
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

      if (hpainter?.check_pad_range) {
         delete hpainter.check_pad_range;
         const ndim = hpainter.getDimension();
         this.applyAxisZoom('x');
         if (ndim > 1)
            this.applyAxisZoom('y');
         if (ndim > 2)
            this.applyAxisZoom('z');
      }

      if (hpainter && !hpainter._checked_zooming) {
         hpainter._checked_zooming = true;

         if (hpainter.options.minimum !== kNoZoom) {
            this.zoom_zmin = hpainter.options.minimum;
            this.zoom_zmax = this.zmax;
         }
         if (hpainter.options.maximum !== kNoZoom) {
            this.zoom_zmax = hpainter.options.maximum;
            if (this.zoom_zmin === undefined) this.zoom_zmin = this.zmin;
         }
      }
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
      switch (name) {
         case 'x': return this.xaxis;
         case 'y': return this.yaxis;
         case 'z': return this.zaxis;
         case 'x2': return this.x2axis;
         case 'y2': return this.y2axis;
      }
      return null;
   }

   /** @summary Apply axis zooming from pad user range
     * @private */
   applyPadUserRange(pad, name) {
      if (!pad) return;

      // seems to be, not allways user range calculated
      let umin = pad[`fU${name}min`],
          umax = pad[`fU${name}max`],
          eps = 1e-7;

            if (name === 'x') {
         if ((Math.abs(pad.fX1) > eps) || (Math.abs(pad.fX2 - 1) > eps)) {
            const dx = pad.fX2 - pad.fX1;
            umin = pad.fX1 + dx*pad.fLeftMargin;
            umax = pad.fX2 - dx*pad.fRightMargin;
         }
      } else {
         if ((Math.abs(pad.fY1) > eps) || (Math.abs(pad.fY2 - 1) > eps)) {
            const dy = pad.fY2 - pad.fY1;
            umin = pad.fY1 + dy*pad.fBottomMargin;
            umax = pad.fY2 - dy*pad.fTopMargin;
         }
      }

      if ((umin >= umax) || (Math.abs(umin) < eps && Math.abs(umax-1) < eps)) return;

      if (pad[`fLog${name}`] > 0) {
         umin = Math.exp(umin * Math.log(10));
         umax = Math.exp(umax * Math.log(10));
      }

      let aname = name;
      if (this.swap_xy) aname = (name === 'x') ? 'y' : 'x';
      const smin = this[`scale_${aname}min`],
            smax = this[`scale_${aname}max`];

      eps = (smax - smin) * 1e-7;

      if ((Math.abs(umin - smin) > eps) || (Math.abs(umax - smax) > eps)) {
         this[`zoom_${aname}min`] = umin;
         this[`zoom_${aname}max`] = umax;
      }
   }

   /** @summary Apply zooming from TAxis attributes */
   applyAxisZoom(name) {
      if (this.zoomChangedInteractive(name)) return;
      this[`zoom_${name}min`] = this[`zoom_${name}max`] = 0;

      const axis = this.getAxis(name);

      if (axis?.TestBit(EAxisBits.kAxisRange)) {
         if ((axis.fFirst !== axis.fLast) && ((axis.fFirst > 1) || (axis.fLast < axis.fNbins))) {
            this[`zoom_${name}min`] = axis.fFirst > 1 ? axis.GetBinLowEdge(axis.fFirst) : axis.fXmin;
            this[`zoom_${name}max`] = axis.fLast < axis.fNbins ? axis.GetBinLowEdge(axis.fLast + 1) : axis.fXmax;
            // reset user range for main painter
            axis.InvertBit(EAxisBits.kAxisRange);
            axis.fFirst = 1; axis.fLast = axis.fNbins;
         }
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

      const w = this.getFrameWidth(), h = this.getFrameHeight(),
            pp = this.getPadPainter(), pad = pp.getRootPad(),
            pad_logx = pad.fLogx,
            pad_logy = (opts.ndim === 1 ? pad.fLogv : undefined) ?? pad.fLogy;

      this.scales_ndim = opts.ndim;

      this.scale_xmin = this.xmin;
      this.scale_xmax = this.xmax;

      this.scale_ymin = this.ymin;
      this.scale_ymax = this.ymax;

      if (opts.extra_y_space) {
         const log_scale = this.swap_xy ? pad_logx : pad_logy;
         if (log_scale && (this.scale_ymax > 0))
            this.scale_ymax = Math.exp(Math.log(this.scale_ymax)*1.1);
         else
            this.scale_ymax += (this.scale_ymax - this.scale_ymin)*0.1;
      }

      if (opts.check_pad_range) {
         // take zooming out of pad or axis attributes
         this.applyAxisZoom('x');
         if (opts.ndim > 1) this.applyAxisZoom('y');
         if (opts.ndim > 2) this.applyAxisZoom('z');

         // Use configured pad range - only when main histogram drawn with SAME draw option
         if (opts.check_pad_range === 'pad_range') {
            this.applyPadUserRange(pad, 'x');
            this.applyPadUserRange(pad, 'y');
         }
      }

      if ((opts.zoom_xmin !== opts.zoom_xmax) && ((this.zoom_xmin === this.zoom_xmax) || !this.zoomChangedInteractive('x'))) {
         this.zoom_xmin = opts.zoom_xmin;
         this.zoom_xmax = opts.zoom_xmax;
      }

      if ((opts.zoom_ymin !== opts.zoom_ymax) && ((this.zoom_ymin === this.zoom_ymax) || !this.zoomChangedInteractive('y'))) {
         this.zoom_ymin = opts.zoom_ymin;
         this.zoom_ymax = opts.zoom_ymax;
      }

      let orig_x = true, orig_y = true;

      if (this.zoom_xmin !== this.zoom_xmax) {
         this.scale_xmin = this.zoom_xmin;
         this.scale_xmax = this.zoom_xmax;
         orig_x = false;
      }

      if (this.zoom_ymin !== this.zoom_ymax) {
         this.scale_ymin = this.zoom_ymin;
         this.scale_ymax = this.zoom_ymax;
         orig_y = false;
      }

      // projection should be assigned
      this.recalculateRange(opts.Proj, orig_x, orig_y);

      this.x_handle = new TAxisPainter(this.getDom(), this.xaxis, true);
      this.x_handle.setPadName(this.getPadName());
      this.x_handle.setHistPainter(opts.hist_painter, 'x');

      this.x_handle.configureAxis('xaxis', this.xmin, this.xmax, this.scale_xmin, this.scale_xmax, this.swap_xy, this.swap_xy ? [0, h] : [0, w],
                                      { reverse: this.reverse_x,
                                        log: this.swap_xy ? pad_logy : pad_logx,
                                        noexp_changed: this.x_noexp_changed,
                                        symlog: this.swap_xy ? opts.symlog_y : opts.symlog_x,
                                        logcheckmin: this.swap_xy,
                                        logminfactor: logminfactorX });

      this.x_handle.assignFrameMembers(this, 'x');

      this.y_handle = new TAxisPainter(this.getDom(), this.yaxis, true);
      this.y_handle.setPadName(this.getPadName());
      this.y_handle.setHistPainter(opts.hist_painter, 'y');

      this.y_handle.configureAxis('yaxis', this.ymin, this.ymax, this.scale_ymin, this.scale_ymax, !this.swap_xy, this.swap_xy ? [0, w] : [0, h],
                                      { reverse: this.reverse_y,
                                        log: this.swap_xy ? pad_logx : pad_logy,
                                        noexp_changed: this.y_noexp_changed,
                                        symlog: this.swap_xy ? opts.symlog_x : opts.symlog_y,
                                        logcheckmin: (opts.ndim < 2) || this.swap_xy,
                                        log_min_nz: opts.ymin_nz && (opts.ymin_nz <= this.ymax) ? 0.5*opts.ymin_nz : 0,
                                        logminfactor: logminfactorY });

      this.y_handle.assignFrameMembers(this, 'y');

      this.setRootPadRange(pad);
   }

   /** @summary Create x,y objects for drawing of second axes
     * @private */
   createXY2(opts) {
      if (!opts) opts = { ndim: this.scales_ndim ?? 1 };

      this.reverse_x2 = opts.reverse_x || false;
      this.reverse_y2 = opts.reverse_y || false;

      this.logx2 = this.logy2 = 0;

      const w = this.getFrameWidth(), h = this.getFrameHeight(),
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
         const log_scale = this.swap_xy ? pad.fLogx : pad.fLogy;
         if (log_scale && (this.scale_y2max > 0))
            this.scale_y2max = Math.exp(Math.log(this.scale_y2max)*1.1);
         else
            this.scale_y2max += (this.scale_y2max - this.scale_y2min)*0.1;
      }

      if ((this.zoom_x2min !== this.zoom_x2max) && opts.second_x) {
         this.scale_x2min = this.zoom_x2min;
         this.scale_x2max = this.zoom_x2max;
      }

      if ((this.zoom_y2min !== this.zoom_y2max) && opts.second_y) {
         this.scale_y2min = this.zoom_y2min;
         this.scale_y2max = this.zoom_y2max;
      }

      if (opts.second_x) {
         this.x2_handle = new TAxisPainter(this.getDom(), this.x2axis, true);
         this.x2_handle.setPadName(this.getPadName());
         this.x2_handle.setHistPainter(opts.hist_painter, 'x');

         this.x2_handle.configureAxis('x2axis', this.x2min, this.x2max, this.scale_x2min, this.scale_x2max, this.swap_xy, this.swap_xy ? [0, h] : [0, w],
                                         { reverse: this.reverse_x2,
                                           log: this.swap_xy ? pad.fLogy : pad.fLogx,
                                           noexp_changed: this.x2_noexp_changed,
                                           logcheckmin: this.swap_xy,
                                           logminfactor: logminfactorX });

         this.x2_handle.assignFrameMembers(this, 'x2');
      }

      if (opts.second_y) {
         this.y2_handle = new TAxisPainter(this.getDom(), this.y2axis, true);
         this.y2_handle.setPadName(this.getPadName());
         this.y2_handle.setHistPainter(opts.hist_painter, 'y');

         this.y2_handle.configureAxis('y2axis', this.y2min, this.y2max, this.scale_y2min, this.scale_y2max, !this.swap_xy, this.swap_xy ? [0, w] : [0, h],
                                         { reverse: this.reverse_y2,
                                           log: this.swap_xy ? pad.fLogx : pad.fLogy,
                                           noexp_changed: this.y2_noexp_changed,
                                           logcheckmin: (opts.ndim < 2) || this.swap_xy,
                                           log_min_nz: opts.ymin_nz && (opts.ymin_nz < this.y2max) ? 0.5 * opts.ymin_nz : 0,
                                           logminfactor: logminfactorY });

         this.y2_handle.assignFrameMembers(this, 'y2');
      }
   }

   /** @summary Return functions to create x/y points based on coordinates
     * @desc In default case returns frame painter itself
     * @private */
   getGrFuncs(second_x, second_y) {
      const use_x2 = second_x && this.grx2,
            use_y2 = second_y && this.gry2;
      if (!use_x2 && !use_y2) return this;

      return {
         use_x2,
         grx: use_x2 ? this.grx2 : this.grx,
         logx: this.logx,
         x_handle: use_x2 ? this.x2_handle : this.x_handle,
         scale_xmin: use_x2 ? this.scale_x2min : this.scale_xmin,
         scale_xmax: use_x2 ? this.scale_x2max : this.scale_xmax,
         use_y2,
         gry: use_y2 ? this.gry2 : this.gry,
         logy: this.logy,
         y_handle: use_y2 ? this.y2_handle : this.y_handle,
         scale_ymin: use_y2 ? this.scale_y2min : this.scale_ymin,
         scale_ymax: use_y2 ? this.scale_y2max : this.scale_ymax,
         swap_xy: this.swap_xy,
         fp: this,
         revertAxis(name, v) {
            if ((name === 'x') && this.use_x2) name = 'x2';
            if ((name === 'y') && this.use_y2) name = 'y2';
            return this.fp.revertAxis(name, v);
         },
         axisAsText(name, v) {
            if ((name === 'x') && this.use_x2) name = 'x2';
            if ((name === 'y') && this.use_y2) name = 'y2';
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

      const rx = pad.fUxmax - pad.fUxmin,
            ry = pad.fUymax - pad.fUymin;
      let mx = 1 - pad.fLeftMargin - pad.fRightMargin,
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
   drawGrids(draw_grids) {
      const layer = this.getFrameSvg().selectChild('.axis_layer');

      layer.selectAll('.xgrid').remove();
      layer.selectAll('.ygrid').remove();

      const pp = this.getPadPainter(),
         pad = pp?.getRootPad(true),
         h = this.getFrameHeight(),
         w = this.getFrameWidth(),
         grid_style = gStyle.fGridStyle;

      // add a grid on x axis, if the option is set
      if (pad?.fGridx && draw_grids && this.x_handle?.ticks) {
         const colid = (gStyle.fGridColor > 0) ? gStyle.fGridColor : (this.getAxis('x')?.fAxisColor ?? 1);
         let gridx = '';

         this.x_handle.ticks.forEach(pos => {
            gridx += this.swap_xy ? `M0,${pos}h${w}` : `M${pos},0v${h}`;
         });

         layer.append('svg:path')
              .attr('class', 'xgrid')
              .attr('d', gridx)
              .style('stroke', this.getColor(colid) || 'black')
              .style('stroke-width', gStyle.fGridWidth)
              .style('stroke-dasharray', getSvgLineStyle(grid_style));
      }

      // add a grid on y axis, if the option is set
      if (pad?.fGridy && draw_grids && this.y_handle?.ticks) {
         const colid = (gStyle.fGridColor > 0) ? gStyle.fGridColor : (this.getAxis('y')?.fAxisColor ?? 1);
         let gridy = '';

         this.y_handle.ticks.forEach(pos => {
            gridy += this.swap_xy ? `M${pos},0v${h}` : `M0,${pos}h${w}`;
         });

         layer.append('svg:path')
              .attr('class', 'ygrid')
              .attr('d', gridy)
              .style('stroke', this.getColor(colid) || 'black')
              .style('stroke-width', gStyle.fGridWidth)
              .style('stroke-dasharray', getSvgLineStyle(grid_style));
      }
   }

   /** @summary Converts 'raw' axis value into text */
   axisAsText(axis, value) {
      const handle = this[`${axis}_handle`];

      if (handle)
         return handle.axisAsText(value, settings[axis.toUpperCase() + 'ValuesFormat']);

      return value.toPrecision(4);
   }

   /** @summary Identify if requested axes are drawn
     * @desc Checks if x/y axes are drawn. Also if second side is already there */
   hasDrawnAxes(second_x, second_y) {
      return !second_x && !second_y ? this.axes_drawn : false;
   }

   /** @summary draw axes,
     * @return {Promise} which ready when drawing is completed  */
   async drawAxes(shrink_forbidden, disable_x_draw, disable_y_draw,
                  AxisPos, has_x_obstacle, has_y_obstacle, enable_grids) {
      this.cleanAxesDrawings();

      if ((this.xmin === this.xmax) || (this.ymin === this.ymax))
         return false;

      if (AxisPos === undefined) AxisPos = 0;

      const layer = this.getFrameSvg().selectChild('.axis_layer'),
            w = this.getFrameWidth(),
            h = this.getFrameHeight(),
            pp = this.getPadPainter(),
            pad = pp.getRootPad(true),
            draw_grids = enable_grids && (pad?.fGridx || pad?.fGridy);

      this.x_handle.invert_side = (AxisPos >= 10);
      this.x_handle.lbls_both_sides = !this.x_handle.invert_side && (pad?.fTickx > 1); // labels on both sides
      this.x_handle.has_obstacle = has_x_obstacle;

      this.y_handle.invert_side = ((AxisPos % 10) === 1);
      this.y_handle.lbls_both_sides = !this.y_handle.invert_side && (pad?.fTicky > 1); // labels on both sides
      this.y_handle.has_obstacle = has_y_obstacle;

      const draw_horiz = this.swap_xy ? this.y_handle : this.x_handle,
            draw_vertical = this.swap_xy ? this.x_handle : this.y_handle;

      if ((!disable_x_draw || !disable_y_draw) && pp._fast_drawing)
         disable_x_draw = disable_y_draw = true;

      let pr = Promise.resolve(true);

      if (!disable_x_draw || !disable_y_draw || draw_grids) {
         const can_adjust_frame = !shrink_forbidden && settings.CanAdjustFrame,

         pr1 = draw_horiz.drawAxis(layer, w, h,
                                   draw_horiz.invert_side ? null : `translate(0,${h})`,
                                   pad?.fTickx ? -h : 0, disable_x_draw,
                                   undefined, false, pp.getPadHeight() - h - this.getFrameY()),

         pr2 = draw_vertical.drawAxis(layer, w, h,
                                      draw_vertical.invert_side ? `translate(${w})` : null,
                                      pad?.fTicky ? w : 0, disable_y_draw,
                                      draw_vertical.invert_side ? 0 : this._frame_x, can_adjust_frame);

         pr = Promise.all([pr1, pr2]).then(() => {
            this.drawGrids(draw_grids);

            if (!can_adjust_frame) return;

            let shrink = 0.0;
            const ypos = draw_vertical.position;

            if ((-0.2 * w < ypos) && (ypos < 0)) {
               shrink = -ypos / w + 0.001;
               this.shrink_frame_left += shrink;
            } else if ((ypos > 0) && (ypos < 0.3 * w) && (this.shrink_frame_left > 0) && (ypos / w > this.shrink_frame_left)) {
               shrink = -this.shrink_frame_left;
               this.shrink_frame_left = 0.0;
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
      const layer = this.getFrameSvg().selectChild('.axis_layer'),
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

      if ((draw_horiz || draw_vertical) && pp._fast_drawing)
         draw_horiz = draw_vertical = null;

      let pr1, pr2;

      if (draw_horiz) {
         pr1 = draw_horiz.drawAxis(layer, w, h,
                                   draw_horiz.invert_side ? null : `translate(0,${h})`,
                                   pad?.fTickx ? -h : 0, false,
                                   undefined, false);
      }

      if (draw_vertical) {
         pr2 = draw_vertical.drawAxis(layer, w, h,
                                      draw_vertical.invert_side ? `translate(${w})` : null,
                                      pad?.fTicky ? w : 0, false,
                                      draw_vertical.invert_side ? 0 : this._frame_x, false);
      }

       return Promise.all([pr1, pr2]);
   }


   /** @summary Update frame attributes
     * @private */
   updateAttributes(force) {
      const pp = this.getPadPainter(),
          pad = pp?.getRootPad(true),
          tframe = this.getObject();

      if ((this.fX1NDC === undefined) || (force && !this.modified_NDC)) {
         if (!pad) {
            this.fX1NDC = gStyle.fPadLeftMargin;
            this.fX2NDC = 1 - gStyle.fPadRightMargin;
            this.fY1NDC = gStyle.fPadBottomMargin;
            this.fY2NDC = 1 - gStyle.fPadTopMargin;
         } else {
            this.fX1NDC = pad.fLeftMargin;
            this.fX2NDC = 1 - pad.fRightMargin;
            this.fY1NDC = pad.fBottomMargin;
            this.fY2NDC = 1 - pad.fTopMargin;
         }
      }

      if (this.fillatt === undefined) {
         if (tframe)
            this.createAttFill({ attr: tframe });
         else if (pad?.fFrameFillColor)
            this.createAttFill({ pattern: pad.fFrameFillStyle, color: pad.fFrameFillColor });
         else if (pad)
            this.createAttFill({ attr: pad });
         else
            this.createAttFill({ pattern: 1001, color: 0 });

         // force white color for the canvas frame
         if (!tframe && this.fillatt.empty() && pp?.iscan)
            this.fillatt.setSolidColor('white');
         else if ((pad?.fFillStyle === 4000) && !this.fillatt.empty()) // special case of transpad.C macro, which set transparent pad
            this.fillatt.setOpacity(0);
      }

      if (!tframe && (pad?.fFrameLineColor !== undefined))
         this.createAttLine({ color: pad.fFrameLineColor, width: pad.fFrameLineWidth, style: pad.fFrameLineStyle });
      else
         this.createAttLine({ attr: tframe, color: 'black' });
   }

   /** @summary Function called at the end of resize of frame
     * @desc One should apply changes to the pad
     * @private */
   sizeChanged() {
      const pad = this.getPadPainter()?.getRootPad(true);

      if (pad) {
         pad.fLeftMargin = this.fX1NDC;
         pad.fRightMargin = 1 - this.fX2NDC;
         pad.fBottomMargin = this.fY1NDC;
         pad.fTopMargin = 1 - this.fY2NDC;
         this.setRootPadRange(pad);
      }

      this.interactiveRedraw('pad', 'frame');
   }

    /** @summary Remove all kinds of X/Y function for axes transformation */
   cleanXY() {
      delete this.grx;
      delete this.gry;
      delete this.grz;
      delete this.grx2;
      delete this.gry2;

      this.x_handle?.cleanup();
      this.y_handle?.cleanup();
      this.z_handle?.cleanup();
      this.x2_handle?.cleanup();
      this.y2_handle?.cleanup();

      delete this.x_handle;
      delete this.y_handle;
      delete this.z_handle;
      delete this.x2_handle;
      delete this.y2_handle;
   }

   /** @summary remove all axes drawings */
   cleanAxesDrawings() {
      this.x_handle?.removeG();
      this.y_handle?.removeG();
      this.z_handle?.removeG();
      this.x2_handle?.removeG();
      this.y2_handle?.removeG();

      this.draw_g?.selectChild('.axis_layer').selectAll('*').remove();
      this.axes_drawn = false;
   }

   /** @summary Returns frame rectangle plus extra info for hint display */
   cleanFrameDrawings() {
      // cleanup all 3D drawings if any
      if (isFunc(this.create3DScene))
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

      this.draw_g?.selectChild('.main_layer').selectAll('*').remove();
      this.draw_g?.selectChild('.upper_layer').selectAll('*').remove();

      this.xaxis = null;
      this.yaxis = null;
      this.zaxis = null;

      if (this.draw_g) {
         this.draw_g.selectAll('*').remove();
         this.draw_g.on('mousedown', null)
                    .on('dblclick', null)
                    .on('wheel', null)
                    .on('contextmenu', null)
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

      const pp = this.getPadPainter();
      if (pp?.frame_painter_ref === this)
         delete pp.frame_painter_ref;

      super.cleanup();
   }

   /** @summary Redraw TFrame */
   redraw(/* reason */) {
      const pp = this.getPadPainter();
      if (pp) pp.frame_painter_ref = this; // keep direct reference to the frame painter

      // first update all attributes from objects
      this.updateAttributes();

      const rect = pp?.getPadRect() ?? { width: 10, height: 10 },
            lm = Math.round(rect.width * this.fX1NDC),
            tm = Math.round(rect.height * (1 - this.fY2NDC));
      let w = Math.round(rect.width * (this.fX2NDC - this.fX1NDC)),
          h = Math.round(rect.height * (this.fY2NDC - this.fY1NDC)),
          rotate = false, fixpos = false, trans;

      if (pp?.options) {
         if (pp.options.RotateFrame) rotate = true;
         if (pp.options.FixFrame) fixpos = true;
      }

      if (rotate) {
         trans = `rotate(-90,${lm},${tm}) translate(${lm-h},${tm})`;
         [w, h] = [h, w];
      } else
         trans = makeTranslate(lm, tm);

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
         this.draw_g = this.getLayerSvg('primitives_layer').append('svg:g').attr('class', 'root_frame');

         // empty title on the frame required to suppress title of the canvas
         if (!this.isBatchMode())
            this.draw_g.append('svg:title').text('');

         top_rect = this.draw_g.append('svg:path');

         main_svg = this.draw_g.append('svg:svg')
                           .attr('class', 'main_layer')
                           .attr('x', 0)
                           .attr('y', 0)
                           .attr('overflow', 'hidden');

         this.draw_g.append('svg:g').attr('class', 'axis_layer');
         this.draw_g.append('svg:g').attr('class', 'upper_layer');
      } else {
         top_rect = this.draw_g.selectChild('path');
         main_svg = this.draw_g.selectChild('.main_layer');
      }

      this.axes_drawn = false;

      this.draw_g.attr('transform', trans);

      top_rect.attr('d', `M0,0H${w}V${h}H0Z`)
              .call(this.fillatt.func)
              .call(this.lineatt.func);

      main_svg.attr('width', w)
              .attr('height', h)
              .attr('viewBox', `0 0 ${w} ${h}`);

      return this;
   }

   /** @summary Change log state of specified axis
     * @param {number} value - 0 (linear), 1 (log) or 2 (log2) */
   changeAxisLog(axis, value) {
      const pp = this.getPadPainter(),
            pad = pp?.getRootPad(true);
      if (!pad) return;

      pp._interactively_changed = true;

      const name = `fLog${axis}`;

      // do not allow log scale for labels
      if (!pad[name]) {
         if (this.swap_xy && axis === 'x')
            axis = 'y';
         else if (this.swap_xy && axis === 'y')
            axis = 'x';
         const handle = this[`${axis}_handle`];
         if (handle?.kind === kAxisLabels) return;
      }

      if ((value === 'toggle') || (value === undefined))
         value = pad[name] ? 0 : 1;

      // directly change attribute in the pad
      pad[name] = value;

      return this.interactiveRedraw('pad', `log${axis}`);
   }

   /** @summary Toggle log state on the specified axis */
   toggleAxisLog(axis) {
      return this.changeAxisLog(axis, 'toggle');
   }

   /** @summary Fill context menu for the frame
     * @desc It could be appended to the histogram menus */
   fillContextMenu(menu, kind, obj) {
      const main = this.getMainPainter(true),
          pp = this.getPadPainter(),
          pad = pp?.getRootPad(true),
          is_pal = kind === 'pal';

      if (is_pal) kind = 'z';

      if ((kind === 'x') || (kind === 'y') || (kind === 'z') || (kind === 'x2') || (kind === 'y2')) {
         const faxis = obj || this[kind+'axis'],
               handle = this[`${kind}_handle`];
        if (!isFunc(faxis?.TestBit))
           return false;

         menu.add(`header: ${kind.toUpperCase()} axis`);
         menu.add('Unzoom', () => this.unzoom(kind));
         if (pad) {
            const member = 'fLog'+kind[0];
            menu.add('sub:SetLog '+kind[0], () => {
               menu.input('Enter log kind: 0 - off, 1 - log10, 2 - log2, 3 - ln, ...', pad[member], 'int', 0, 10000).then(v => {
                  this.changeAxisLog(kind[0], v);
               });
            });
            menu.addchk(pad[member] === 0, 'linear', () => this.changeAxisLog(kind[0], 0));
            menu.addchk(pad[member] === 1, 'log10', () => this.changeAxisLog(kind[0], 1));
            menu.addchk(pad[member] === 2, 'log2', () => this.changeAxisLog(kind[0], 2));
            menu.addchk(pad[member] === 3, 'ln', () => this.changeAxisLog(kind[0], 3));
            menu.addchk(pad[member] === 4, 'log4', () => this.changeAxisLog(kind[0], 4));
            menu.addchk(pad[member] === 8, 'log8', () => this.changeAxisLog(kind[0], 8));
            menu.add('endsub:');
         }
         menu.addchk(faxis.TestBit(EAxisBits.kMoreLogLabels), 'More log', flag => {
            faxis.InvertBit(EAxisBits.kMoreLogLabels);
            if (main?.snapid && (kind.length === 1))
               main.interactiveRedraw('pad', `exec:SetMoreLogLabels(${flag})`, kind);
            else
               this.interactiveRedraw('pad');
         });
         menu.addchk(handle?.noexp ?? faxis.TestBit(EAxisBits.kNoExponent), 'No exponent', flag => {
            if (flag !== faxis.TestBit(EAxisBits.kNoExponent))
               faxis.InvertBit(EAxisBits.kNoExponent);
            if (handle) handle.noexp_changed = true;
            this[`${kind}_noexp_changed`] = true;
            if (main?.snapid && (kind.length === 1))
               main.interactiveRedraw('pad', `exec:SetNoExponent(${flag})`, kind);
            else
               this.interactiveRedraw('pad');
         });

         if ((kind === 'z') && isFunc(main?.fillPaletteMenu))
            main.fillPaletteMenu(menu, !is_pal);

         if ((handle?.kind === kAxisLabels) && (faxis.fNbins > 20)) {
            menu.add('Find label', () => menu.input('Label id').then(id => {
               if (!id) return;
               for (let bin = 0; bin < faxis.fNbins; ++bin) {
                  const lbl = handle.formatLabels(bin);
                  if (lbl === id)
                     return this.zoom(kind, Math.max(0, bin - 4), Math.min(faxis.fNbins, bin+5));
                }
            }));
         }

         menu.addTAxisMenu(EAxisBits, main || this, faxis, kind);
         return true;
      }

      const alone = menu.size() === 0;

      if (alone)
         menu.add('header:Frame');
      else
         menu.add('separator');

      if (this.zoom_xmin !== this.zoom_xmax)
         menu.add('Unzoom X', () => this.unzoom('x'));
      if (this.zoom_ymin !== this.zoom_ymax)
         menu.add('Unzoom Y', () => this.unzoom('y'));
      if (this.zoom_zmin !== this.zoom_zmax)
         menu.add('Unzoom Z', () => this.unzoom('z'));
      if (this.zoom_x2min !== this.zoom_x2max)
         menu.add('Unzoom X2', () => this.unzoom('x2'));
      if (this.zoom_y2min !== this.zoom_y2max)
         menu.add('Unzoom Y2', () => this.unzoom('y2'));
      menu.add('Unzoom all', () => this.unzoom('all'));

      if (pad) {
         menu.addchk(pad.fLogx, 'SetLogx', () => this.toggleAxisLog('x'));
         menu.addchk(pad.fLogy, 'SetLogy', () => this.toggleAxisLog('y'));

         if (isFunc(main?.getDimension) && (main.getDimension() > 1))
            menu.addchk(pad.fLogz, 'SetLogz', () => this.toggleAxisLog('z'));
         menu.add('separator');
      }

      menu.addchk(this.isTooltipAllowed(), 'Show tooltips', () => this.setTooltipAllowed('toggle'));
      menu.addAttributesMenu(this, alone ? '' : 'Frame ');
      menu.add('Save to gStyle', () => {
         gStyle.fPadBottomMargin = this.fY1NDC;
         gStyle.fPadTopMargin = 1 - this.fY2NDC;
         gStyle.fPadLeftMargin = this.fX1NDC;
         gStyle.fPadRightMargin = 1 - this.fX2NDC;
         this.fillatt?.saveToStyle('fFrameFillColor', 'fFrameFillStyle');
         this.lineatt?.saveToStyle('fFrameLineColor', 'fFrameLineWidth', 'fFrameLineStyle');
      }, 'Store frame position and graphical attributes to gStyle');

      menu.add('separator');

      menu.add('sub:Save as');
      ['svg', 'png', 'jpeg', 'pdf', 'webp'].forEach(fmt => menu.add(`frame.${fmt}`, () => pp.saveAs(fmt, 'frame', `frame.${fmt}`)));
      menu.add('endsub:');

      return true;
   }

   /** @summary Fill option object used in TWebCanvas
     * @private */
   fillWebObjectOptions(res) {
      res.fcust = 'frame';
      res.fopt = [this.scale_xmin || 0, this.scale_ymin || 0, this.scale_xmax || 0, this.scale_ymax || 0];
   }

   /** @summary Returns frame X position */
   getFrameX() { return this._frame_x || 0; }

   /** @summary Returns frame Y position */
   getFrameY() { return this._frame_y || 0; }

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
         transform: this.draw_g?.attr('transform') || '',
         hint_delta_x: 0,
         hint_delta_y: 0
      };
   }

   /** @summary Configure user-defined click handler
     * @desc Function will be called every time when frame click was perfromed
     * As argument, tooltip object with selected bins will be provided
     * If handler function returns true, default handling of click will be disabled */
   configureUserClickHandler(handler) {
      this._click_handler = isFunc(handler) ? handler : null;
   }

   /** @summary Configure user-defined dblclick handler
     * @desc Function will be called every time when double click was called
     * As argument, tooltip object with selected bins will be provided
     * If handler function returns true, default handling of dblclick (unzoom) will be disabled */
   configureUserDblclickHandler(handler) {
      this._dblclick_handler = isFunc(handler) ? handler : null;
   }

   /** @summary Function can be used for zooming into specified range
     * @desc if both limits for each axis 0 (like xmin === xmax === 0), axis will be unzoomed
     * @param {number} xmin
     * @param {number} xmax
     * @param {number} [ymin]
     * @param {number} [ymax]
     * @param {number} [zmin]
     * @param {number} [zmax]
     * @return {Promise} with boolean flag if zoom operation was performed */
   async zoom(xmin, xmax, ymin, ymax, zmin, zmax) {
      if (xmin === 'x') { xmin = xmax; xmax = ymin; ymin = undefined; } else
      if (xmin === 'y') { ymax = ymin; ymin = xmax; xmin = xmax = undefined; } else
      if (xmin === 'z') { zmin = xmax; zmax = ymin; xmin = xmax = ymin = undefined; }

      let zoom_x = (xmin !== xmax), zoom_y = (ymin !== ymax), zoom_z = (zmin !== zmax),
          unzoom_x = false, unzoom_y = false, unzoom_z = false;

      if (zoom_x) {
         let cnt = 0;
         if (xmin <= this.xmin) { xmin = this.xmin; cnt++; }
         if (xmax >= this.xmax) { xmax = this.xmax; cnt++; }
         if (cnt === 2) { zoom_x = false; unzoom_x = true; }
      } else
         unzoom_x = (xmin === xmax) && (xmin === 0);

      if (zoom_y) {
         let cnt = 0;
         if ((ymin <= this.ymin) || (!this.ymin && this.logy &&
              ((!this.y_handle?.log_min_nz && ymin < logminfactorY*this.ymax) || (ymin < this.y_handle?.log_min_nz)))) {
                 ymin = this.ymin;
                 cnt++;
              }
         if (ymax >= this.ymax) { ymax = this.ymax; cnt++; }
         if ((cnt === 2) && (this.scales_ndim !== 1)) {
            zoom_y = false;
            unzoom_y = true;
         }
      } else
         unzoom_y = (ymin === ymax) && (ymin === 0);

      if (zoom_z) {
         let cnt = 0;
         if (zmin <= this.zmin) { zmin = this.zmin; cnt++; }
         if (zmax >= this.zmax) { zmax = this.zmax; cnt++; }
         if ((cnt === 2) && (this.scales_ndim > 2)) { zoom_z = false; unzoom_z = true; }
      } else
         unzoom_z = (zmin === zmax) && (zmin === 0);


      let changed = false;

      // first process zooming (if any)
      if (zoom_x || zoom_y || zoom_z) {
         this.forEachPainter(obj => {
            if (!isFunc(obj.canZoomInside)) return;
            if (zoom_x && obj.canZoomInside('x', xmin, xmax)) {
               this.zoom_xmin = xmin;
               this.zoom_xmax = xmax;
               changed = true;
               zoom_x = false;
            }
            if (zoom_y && obj.canZoomInside('y', ymin, ymax)) {
               this.zoom_ymin = ymin;
               this.zoom_ymax = ymax;
               changed = true;
               zoom_y = false;
            }
            if (zoom_z && obj.canZoomInside('z', zmin, zmax)) {
               this.zoom_zmin = zmin;
               this.zoom_zmax = zmax;
               changed = true;
               zoom_z = false;
            }
         });
      }

      // and process unzoom, if any
      if (unzoom_x || unzoom_y || unzoom_z) {
         if (unzoom_x) {
            if (this.zoom_xmin !== this.zoom_xmax) changed = true;
            this.zoom_xmin = this.zoom_xmax = 0;
         }
         if (unzoom_y) {
            if (this.zoom_ymin !== this.zoom_ymax) {
               changed = true;
               unzoomHistogramYRange(this.getMainPainter());
            }
            this.zoom_ymin = this.zoom_ymax = 0;
         }
         if (unzoom_z) {
            if (this.zoom_zmin !== this.zoom_zmax) changed = true;
            this.zoom_zmin = this.zoom_zmax = 0;
         }

         // than try to unzoom all overlapped objects
         if (!changed) {
            this.getPadPainter()?.painters?.forEach(painter => {
               if (isFunc(painter?.unzoomUserRange)) {
                  if (painter.unzoomUserRange(unzoom_x, unzoom_y, unzoom_z))
                     changed = true;
                  }
            });
         }
      }

      return changed ? this.interactiveRedraw('pad', 'zoom').then(() => true) : false;
   }

   /** @summary Provide zooming of single axis
     * @desc One can specify names like x/y/z but also second axis x2 or y2
     * @private */
   async zoomSingle(name, vmin, vmax) {
      if (!this[name+'_handle'])
         return false;

      let zoom_v = (vmin !== vmax), unzoom_v = false;

      if (zoom_v) {
         let cnt = 0;
         if (vmin <= this[name+'min']) { vmin = this[name+'min']; cnt++; }
         if (vmax >= this[name+'max']) { vmax = this[name+'max']; cnt++; }
         if (cnt === 2) { zoom_v = false; unzoom_v = true; }
      } else
         unzoom_v = (vmin === vmax) && (vmin === 0);


      let changed = false;

      // first process zooming
      if (zoom_v) {
         this.forEachPainter(obj => {
            if (!isFunc(obj.canZoomInside)) return;
            if (zoom_v && obj.canZoomInside(name[0], vmin, vmax)) {
               this[`zoom_${name}min`] = vmin;
               this[`zoom_${name}max`] = vmax;
               changed = true;
               zoom_v = false;
            }
         });
      }

      // and process unzoom, if any
      if (unzoom_v) {
         if (this[`zoom_${name}min`] !== this[`zoom_${name}max`]) {
            changed = true;
            if (name === 'y') unzoomHistogramYRange(this.getMainPainter());
         }
         this[`zoom_${name}min`] = this[`zoom_${name}max`] = 0;
      }

      if (!changed) return false;

      return this.interactiveRedraw('pad', 'zoom').then(() => true);
   }

   /** @summary Checks if specified axis zoomed */
   isAxisZoomed(axis) {
      return this[`zoom_${axis}min`] !== this[`zoom_${axis}max`];
   }

   /** @summary Unzoom speicified axes
     * @return {Promise} with boolean flag if zooming changed */
   async unzoom(dox, doy, doz) {
      if (dox === 'all')
         return this.unzoom('x2').then(() => this.unzoom('y2')).then(() => this.unzoom('xyz'));

      if ((dox === 'x2') || (dox === 'y2')) {
         return this.zoomSingle(dox, 0, 0).then(changed => {
            if (changed) this.zoomChangedInteractive(dox, 'unzoom');
            return changed;
         });
      }

      if (typeof dox === 'undefined') dox = doy = doz = true; else
      if (isStr(dox)) { doz = dox.indexOf('z') >= 0; doy = dox.indexOf('y') >= 0; dox = dox.indexOf('x') >= 0; }

      return this.zoom(dox ? 0 : undefined, dox ? 0 : undefined,
                       doy ? 0 : undefined, doy ? 0 : undefined,
                       doz ? 0 : undefined, doz ? 0 : undefined).then(changed => {
         if (changed && dox) this.zoomChangedInteractive('x', 'unzoom');
         if (changed && doy) this.zoomChangedInteractive('y', 'unzoom');
         if (changed && doz) this.zoomChangedInteractive('z', 'unzoom');

         return changed;
      });
   }

   /** @summary Mark/check if zoom for specific axis was changed interactively
     * @private */
   zoomChangedInteractive(axis, value) {
      if (axis === 'reset') {
         this.zoom_changed_x = this.zoom_changed_y = this.zoom_changed_z = undefined;
         return;
      }
      if (!axis || axis === 'any')
         return this.zoom_changed_x || this.zoom_changed_y || this.zoom_changed_z;

      if ((axis !== 'x') && (axis !== 'y') && (axis !== 'z')) return;

      const fld = 'zoom_changed_' + axis;
      if (value === undefined)
         return this[fld];

      if (value === 'unzoom') {
         // special handling of unzoom, only if was never changed before flag set to true
         this[fld] = (this[fld] === undefined);
         return;
      }

      if (value)
         this[fld] = true;
   }

   /** @summary Convert graphical coordinate into axis value */
   revertAxis(axis, pnt) { return this[`${axis}_handle`]?.revertPoint(pnt) ?? 0; }

   /** @summary Show axis status message
     * @desc method called normally when mouse enter main object element
     * @private */
   showAxisStatus(axis_name, evnt) {
      const taxis = this.getAxis(axis_name),
            m = d3_pointer(evnt, this.getFrameSvg().node());
      let hint_name = axis_name,
          hint_title = clTAxis,
          id = (axis_name === 'x') ? 0 : 1;

      if (taxis) {
         hint_name = taxis.fName;
         hint_title = taxis.fTitle || `TAxis object for ${axis_name}`;
      }
      if (this.swap_xy) id = 1 - id;

      const axis_value = this.revertAxis(axis_name, m[id]);

      this.showObjectStatus(hint_name, hint_title, `${axis_name} : ${this.axisAsText(axis_name, axis_value)}`, `${m[0]},${m[1]}`);
   }

   /** @summary Add interactive keys handlers
    * @private */
   addKeysHandler() {
      if (this.isBatchMode()) return;
      FrameInteractive.assign(this);
      this.addFrameKeysHandler();
   }

   /** @summary Add interactive functionality to the frame
     * @private */
   addInteractivity(for_second_axes) {
      if (this.isBatchMode() || (!settings.Zooming && !settings.ContextMenu))
         return false;

      FrameInteractive.assign(this);
      if (!for_second_axes)
         this.addBasicInteractivity();

      return this.addFrameInteractivity(for_second_axes);
   }

} // class TFramePainter

export { addDragHandler, TooltipHandler, FrameInteractive, TFramePainter, getEarthProjectionFunc };
