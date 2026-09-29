/** Annotations -- the hover tooltip, and annotations kept from it.
 *
 * GlViewerRCore owns one Annotations, tells it what is hovered and calls
 * layout() once per frame. The tooltip and every kept annotation are ZTexts in
 * the viewer's overlay scene. They are therefore in the framebuffer and appear
 * in screen captures, and the Overlay class moves and resizes them.
 *
 * Text is plain: ZText preserves newlines and spaces but renders no markup.
 */

sap.ui.define([], function() {

   "use strict";

   class Annotations {

      /** @param viewer the owning GlViewerRCore.
       *  @param RC the RenderCore module. */
      constructor(viewer, RC) {
         this.viewer = viewer;
         this.RC = RC;

         /** Size and placement of the hover tooltip. font_size is a fraction of
          * viewport height, as everywhere in ZText's screen modes; the gap is in
          * CSS pixels and keeps the box clear of the cursor itself. */
         this.font_size = 0.017;
         this.cursor_gap_px = 14;

         this.font_name = "LiberationSerif-Regular";

         /** Button font size, as a fraction of the annotation's font size. */
         this.btn_scale = 0.8;

         /** Slack, in overlay coordinates, around an annotation's outer edges
          * within which the pointer still counts as on it. */
         this.hover_margin = 0.008;

         /** Frame line width, as a fraction of the line height. REveText's
          * 0.05 is sub-pixel at tooltip font sizes, and a sub-pixel frame
          * loses its edges one at a time. */
         this.frame_line = 0.14;

         /** Plate opacity, streamed from REveViewer::fTooltipAlpha. */
         this.plate_alpha = 0.85;

         /** Synthetic bold for small text, ramped by cap height in CSS pixels.
          * An SDF glyph a few pixels tall never reaches full coverage, so it
          * tints what is behind it instead of covering it. A positive ZText
          * `weight` dilates the glyph until coverage saturates. Large text
          * gets none, since it would look overweight. */
         this.weight_max    = 0.07;
         this.weight_px_lo  = 6;    ///< full weight at or below this cap height
         this.weight_px_hi  = 14;   ///< no weight at or above it

         /** Connector line width, in CSS pixels. */
         this.conn_width_px = 1.5;

         this._font = null;      // {texture, metrics}, cached once delivered
         this._tip = null;       // the hover tooltip ZText
         this._pending = null;   // text asked for before the font arrived
         this._kept = [];        // Annotation instances
         this._keep_pending = null;   // keepAt() request waiting for the font
         this._font_requested = false;
         this._watched = {};     // ids of the scenes registered with, see _watchScene()
         this._last = null;      // last tooltip text and position, see updateText()
      }

      //-----------------------------------------------------------------------
      // Kept annotations
      //-----------------------------------------------------------------------

      /** Keep an annotation with `text` at (x, y), CSS pixels from the canvas
       * top-left. `anchor3d` is a world point for a connector, or null.
       * `target` is {elementId, sceneId} of the annotated element, or null.
       * Used by the context menu, which supplies text from its own pick. If
       * the font is not loaded yet, the annotation is created when it arrives
       * and null is returned. */
      keepAt(text, x, y, anchor3d, target) {
         if (!text) return null;
         if (!this._font) {
            // Only the latest request is remembered. Recorded before asking for
            // the font, because a cached font is delivered synchronously.
            this._keep_pending = { text: text, x: x, y: y, a3: anchor3d, tgt: target };
            this._ensureFont();
            return null;
         }
         return this._keepAtNow(text, x, y, anchor3d, target);
      }

      _keepAtNow(text, x, y, anchor3d, target) {
         const W = this.viewer.canvas.width, H = this.viewer.canvas.height;
         const px = (this.viewer.canvas.pixelRatio || 1);
         const pos = [(x * px) / W, 1.0 - (y * px) / H];
         const a = new Annotation(this, text, pos, this.font_size, anchor3d);
         a.target = target || null;    // { elementId, sceneId }
         this._kept.push(a);
         this._watchScene(a);
         this.viewer.request_render();
         return a;
      }

      /** Register as a receiver of the annotated element's scene, once per
       * scene, so that elementsRemoved() can drop annotations whose subject is
       * deleted. EveManager.ImportSceneChangeJson calls elementsRemoved before
       * it removes the elements from its map. */
      _watchScene(a) {
         const sid = a.target && a.target.sceneId;
         if (!sid) return;
         if (this._watched[sid]) return;
         const mgr = this.viewer.controller ? this.viewer.controller.mgr : null;
         if (!mgr || typeof mgr.RegisterSceneReceiver !== "function") return;
         mgr.RegisterSceneReceiver(sid, this);
         this._watched[sid] = true;
      }

      /** Scene-receiver callback: drop annotations whose subject, or any
       * ancestor of it, is in `ids`. A removal can be reported for a container
       * only, so matching the subject's own id is not sufficient. */
      elementsRemoved(ids) {
         if (!ids || !ids.length || !this._kept.length) return;
         const mgr = this.viewer.controller ? this.viewer.controller.mgr : null;
         if (!mgr) return;
         const dead = new Set(ids);

         const subjectIsGone = (id) => {
            // The callback runs before removeElements, so the mother chain is intact.
            let guard = 0;
            while (id !== undefined && id !== null && ++guard < 64) {
               if (dead.has(id)) return true;
               const el = mgr.GetElement(id);
               if (!el) return false;
               id = el.fMotherId;
            }
            return false;
         };

         for (const a of this._kept.slice())
            if (a.target && subjectIsGone(a.target.elementId)) a.remove();
      }

      /** World-space point under a pick, or null. Unprojects the pixel's near-
       * and far-plane points and interpolates between them by
       * `pstate.depth`, the eye-space distance from RendeQuTor.pick_low_level. */
      worldFromPick(pstate) {
         const RC = this.RC, cam = this.viewer.camera;
         if (!pstate || !cam || !(pstate.depth > 0)) return null;
         const nx = pstate.mouse.x, ny = pstate.mouse.y;
         const pn = new RC.Vector3(nx, ny, -1).unproject(cam);
         const pf = new RC.Vector3(nx, ny,  1).unproject(cam);
         const span = cam.far - cam.near;
         if (!(Math.abs(span) > 1e-9)) return null;
         const t = (pstate.depth - cam.near) / span;
         return [pn.x + (pf.x - pn.x) * t,
                 pn.y + (pf.y - pn.y) * t,
                 pn.z + (pf.z - pn.z) * t];
      }

      _forget(a) {
         const i = this._kept.indexOf(a);
         if (i >= 0) this._kept.splice(i, 1);
         this.viewer.request_render();
      }

      /** Buttons sit on the box, so they have to be re-placed whenever it moves
       * or resizes. Called from the viewer's render loop: a drag moves the box
       * through ovlSetPos without telling anyone, so there is nothing to hook. */
      layout() {
         // Buttons show while the pointer is on the annotation or its buttons,
         // or while one of its parts is being dragged. Testing the box alone
         // would hide the buttons as soon as the pointer moved onto one.
         const ovl = this.viewer.overlay;
         const dragged = ovl.drag ? ovl.drag.obj : null;
         const nx = ovl.nx, ny = ovl.ny;
         for (const a of this._kept) {
            // Geometry first: the test below is against the laid-out rects.
            a.syncWeight();
            a.layout();
            a.updateConnector();
            a.setButtonsVisible(a.owns(dragged) || a.containsPointer(nx, ny));
         }
      }

      //-----------------------------------------------------------------------
      // The hover tooltip
      //-----------------------------------------------------------------------

      /** Show `text` with its top-left corner near (x, y), in CSS pixels from
       * the top-left of the canvas. */
      showTooltip(text, x, y) {
         if (!text) return this.hideTooltip();

         if (!this._tip) {
            // First call: the font is fetched once and cached. Remember what was
            // asked for, so the tooltip appears as soon as the font lands rather
            // than only on the next hover.
            this._pending = { text: text, x: x, y: y };
            this._ensureFont();
            return;
         }
         this._apply(text, x, y);
      }

      /** Tooltip cap height, as a fraction of viewport height, streamed from
       * REveViewer::fTooltipFontSize. Kept annotations keep the size they
       * were made at. */
      setFontSize(sz) {
         if (!(sz > 0) || sz === this.font_size) return;
         this.font_size = sz;
         if (this._tip) {
            this._tip.fontSize = sz;
            this._tip.fontWeight = this.weightFor(sz);
            this.viewer.request_render();
         }
      }

      hideTooltip() {
         this._pending = null;
         if (this._tip && this._tip.visible) {
            this._tip.visible = false;
            this.viewer.request_render();
         }
      }

      // Recolouring is done by GlViewerRCore.recolourFgElements(), which calls
      // setColors() on every overlay object flagged use_fg_color.

      //-----------------------------------------------------------------------

      _ensureFont() {
         if (this._font_requested) return;
         this._font_requested = true;

         // top_path, not eve_path: REveText registers the atlas directory with
         // gEve->AddLocation("sdf-fonts/", ...), at the server's top level.
         const url_base = this.viewer.top_path + 'sdf-fonts/' + this.font_name;
         this.viewer.tex_cache.deliver_font(url_base,
            (texture, font_metrics) => {
               this._font = { texture: texture, metrics: font_metrics };
               this._build();
               if (this._pending) {
                  const p = this._pending;
                  this._pending = null;
                  this._apply(p.text, p.x, p.y);
               }
               if (this._keep_pending) {
                  const k = this._keep_pending;
                  this._keep_pending = null;
                  this._keepAtNow(k.text, k.x, k.y, k.a3, k.tgt);
               }
            },
            (img) => this.RC.ZText.createDefaultTexture(img),
            () => this.viewer.request_render()
         );
      }

      /** Give a new overlay object the viewer's CSS-pixel scale. ZText floors
       * its frame line, grip and font size at that many pixels.
       * Overlay.updatePixelScale() only pushes the scale when it changes, so an
       * object added later keeps ZText.PX_TO_SCREEN_SPACE until told. */
      _fixPixelScale(obj) {
         const v = this.viewer;
         if (typeof obj.setPixelScale !== "function") return;
         if (!(v.overlay.px_to_screen > 0)) return;
         obj.setPixelScale(v.overlay.px_to_screen, v.canvas.width, v.canvas.height);
      }

      /** Synthetic bold for `font_size` (a fraction of viewport height), in
       * ZText `weight` units. The ramp is evaluated in CSS pixels, since
       * stroke thinness depends on pixel size, not on the viewport fraction. */
      weightFor(font_size) {
         const px_scale = this.viewer.overlay.px_to_screen ||
                          this.RC.ZText.PX_TO_SCREEN_SPACE;
         const px = font_size / px_scale;
         const t = Math.min(1, Math.max(0,
                     (this.weight_px_hi - px) /
                     (this.weight_px_hi - this.weight_px_lo)));
         return this.weight_max * t;
      }

      /** Give `obj` the annotation plate: background-colour fill, foreground
       * frame, pixel scale and weight.
       * @param line_frac frame width as a fraction of obj's line height;
       * defaults to frame_line. Buttons pass Annotation._frameFrac(). */
      _plate(obj, line_frac) {
         // The plate takes the background colour, so it and its hover
         // highlight (fill_alpha raised towards opaque) read correctly on both
         // light and dark backgrounds.
         obj.setupFrameStuff(1.0, true,
                             this.viewer.bgCol, this.plate_alpha,
                             this.viewer.fgCol, 1.0, 0.25,
                             (line_frac === undefined) ? this.frame_line : line_frac);
         obj.use_fg_color = true;

         // recolourFgElements() traverses the overlay scene calling
         // setColors(fg, fg) on anything flagged use_fg_color. ZText's own
         // version sets the text and line colours but knows nothing about the
         // fill, so the plate would keep the old background after a flip.
         const owner = this;
         if (!obj._ann_recolour) {
            obj._ann_recolour = true;
            const base = obj.setColors.bind(obj);
            obj.setColors = function(text_col, line_col) {
               base(text_col, line_col);
               this.fill_color = owner.viewer.bgCol;
            };
         }

         this._fixPixelScale(obj);
         obj.fontWeight = this.weightFor(obj.fontSize);
      }

      /** Set the plate opacity of the tooltip and every kept annotation. Sets
       * _norm_fill_alpha too, because ZText.setHighlight(false) restores
       * fill_alpha from it. */
      setPlateAlpha(a) {
         if (!(a >= 0) || a === this.plate_alpha) return;
         this.plate_alpha = a;

         const put = (o) => {
            if (!o) return;
            o.fill_alpha = a;
            o._norm_fill_alpha = a;
         };
         put(this._tip);
         for (const an of this._kept) {
            put(an.text_obj); put(an.btn_close); put(an.btn_edit);
         }
         this.viewer.request_render();
      }

      _build() {
         const RC = this.RC;

         this._tip = new RC.ZText({
            text: " ",
            fontTexture: this._font.texture,
            font: this._font.metrics,
            fontSize: this.font_size,
            mode: RC.TEXT2D_SPACE_SCREEN,
            fontHinting: 1.0,
            color: this.viewer.fgCol,
            alignH: RC.ZText.ALIGN_H.LEFT,
            alignV: RC.ZText.ALIGN_V.TOP
         });

         // The tooltip is passive: not pickable, so the overlay neither hovers
         // nor drags it, and not resizable. Kept annotations are separate
         // ZTexts with both enabled.
         this._tip.pickable  = false;
         this._tip.resizable = false;

         // A plate behind the text keeps it readable over busy geometry.
         this._plate(this._tip);
         this._tip.visible = false;

         this.viewer.overlay_scene.add(this._tip);
      }

      _apply(text, x, y) {
         const RC = this.RC;
         const t = this._tip;
         if (t.text !== text) t.text = text;

         const W = this.viewer.canvas.width, H = this.viewer.canvas.height;
         if (!(W > 0 && H > 0)) return;

         this._last = { text: text, x: x, y: y };

         const px  = (this.viewer.canvas.pixelRatio || 1);
         const gap = this.cursor_gap_px * px;
         const sx  = x * px, sy = y * px;

         // Put the box on the side of the cursor facing the canvas centre, so
         // it stays on the canvas. ZText aligns against its laid-out box, so
         // the choice is made through setAlign().
         const right_half = sx > 0.5 * W;
         const lower_half = sy > 0.5 * H;

         t.setAlign(right_half ? RC.ZText.ALIGN_H.RIGHT : RC.ZText.ALIGN_H.LEFT,
                    lower_half ? RC.ZText.ALIGN_V.BOTTOM : RC.ZText.ALIGN_V.TOP);

         // ZText screen mode is (0,1) from the bottom-left; the pointer is in
         // CSS pixels from the top-left.
         const ox = (sx + (right_half ? -gap : gap)) / W;
         const oy = 1.0 - (sy + (lower_half ? -gap : gap)) / H;
         t.setOffset([ox, oy]);

         t.visible = true;
         this.viewer.request_render();
      }

      /** Re-show the last tooltip with new text at its last position. Called
       * by GlViewerRCore.remoteToolTip() with the tooltip that REveSelection
       * streams in the highlight record, which carries no position. */
      updateText(text) {
         if (!this._last) return;
         this.showTooltip(text, this._last.x, this._last.y);
      }
   }

   /** One kept annotation: a text box, its X (close) and E (edit) buttons,
    * and, when it has a 3D anchor, a connector line. The three ZTexts are
    * ordinary overlay objects, dragged and highlighted by Overlay. The buttons
    * act on the client through `onOverlayClick`, which Overlay.click() calls in
    * place of sending a MIR.
    */
   class Annotation {

      constructor(owner, text, pos, font_size, anchor3d) {
         this.owner = owner;
         /** World point this annotation points at, or null for a plain one. */
         this.anchor3d = anchor3d || null;
         const RC = owner.RC, f = owner._font;

         this.text_obj = new RC.ZText({
            text: text,
            fontTexture: f.texture, font: f.metrics,
            fontSize: font_size,
            mode: RC.TEXT2D_SPACE_SCREEN,
            fontHinting: 1.0,
            color: owner.viewer.fgCol,
            alignH: RC.ZText.ALIGN_H.LEFT,
            alignV: RC.ZText.ALIGN_V.TOP
         });
         this.text_obj.setOffset(pos.slice());
         this.text_obj.pickable  = true;
         this.text_obj.resizable = true;
         owner._plate(this.text_obj);

         this.btn_close = this._makeButton("X", () => this.remove());
         this.btn_edit  = this._makeButton("E", () => this.edit());

         // Hidden until the pointer enters the annotation; layout() decides.
         this.btn_close.visible = false;
         this.btn_edit.visible  = false;
         this._btns_on = false;

         const os = owner.viewer.overlay_scene;
         if (this.anchor3d) {
            this.conn = this._makeConnector();
            os.add(this.conn);          // first, so the plate draws over it
         }
         os.add(this.text_obj);
         os.add(this.btn_close);
         os.add(this.btn_edit);

         this.layout();
         this.updateConnector();
      }

      _makeButton(label, onclick) {
         const RC = this.owner.RC, f = this.owner._font;
         const b = new RC.ZText({
            text: label,
            fontTexture: f.texture, font: f.metrics,
            fontSize: this.text_obj.fontSize * this.owner.btn_scale,
            mode: RC.TEXT2D_SPACE_SCREEN,
            fontHinting: 1.0,
            color: this.owner.viewer.fgCol,
            alignH: RC.ZText.ALIGN_H.LEFT,
            alignV: RC.ZText.ALIGN_V.BOTTOM
         });
         b.pickable  = true;
         // Not resizable: a button with a resize grip would hand over half its
         // own hit area to the grip, and there is nothing to resize.
         b.resizable = false;
         this.owner._plate(b, this._frameFrac());
         b.onOverlayClick = onclick;
         return b;
      }

      /** Place the buttons on top of the box, X at its left edge and E to the
       * right of X, sharing frame lines with the box. Recomputed each frame
       * from the laid-out rect, so the buttons follow moves and resizes. */
      layout() {
         const v = this.owner.viewer;
         if (!v.canvas || !v.canvas.width) return;
         const aspect = v.canvas.width / v.canvas.height;
         const r = this.text_obj.getScreenRect(aspect);
         if (!r) return;

         // Buttons follow the plate's font size. Guarded, because assigning
         // fontSize rebuilds the glyph geometry and this runs every frame.
         const want = this.text_obj.fontSize * this.owner.btn_scale;
         if (Math.abs(this.btn_close.fontSize - want) > 1e-9) {
            this.btn_close.fontSize = want;
            this.btn_edit.fontSize  = want;
            // Buttons are smaller than the plate, so they cross the
            // thin-stroke threshold sooner and need their own weight.
            this.btn_close.fontWeight = this.owner.weightFor(want);
            this.btn_edit.fontWeight  = this.owner.weightFor(want);
            // Rescale the frame fraction so the button frames keep the plate's
            // absolute width; see _frameFrac().
            this.owner._plate(this.btn_close, this._frameFrac());
            this.owner._plate(this.btn_edit,  this._frameFrac());
         }

         // Frames share lines rather than stacking: X's left frame continues
         // the plate's left frame, both buttons' bottom frames lie on the
         // plate's top frame, and E's left frame is X's right frame. All
         // positions are outer edges, from _outer().
         const P = this._outer(this.text_obj, aspect);
         if (!P) return;
         const wf  = this._frameWidthOf(this.text_obj);   // == screen y units
         const wfx = wf / aspect;                         // x is aspect-divided

         // Measure, then correct. The offset means different things for
         // different alignH/alignV; a measured rect does not.
         const place = (obj, out_l, out_b) => {
            const o = this._outer(obj, aspect);
            if (!o) return null;
            const p = obj.ovlGetPos();
            obj.setOffset([p[0] + (out_l - o.l), p[1] + (out_b - o.b)]);
            return this._outer(obj, aspect);
         };

         // The plate's top frame bar spans [P.t - wf, P.t]; a button's bottom
         // bar spans [out_b, out_b + wf]. Coincident means out_b = P.t - wf.
         const bottom = P.t - wf;
         const oc = place(this.btn_close, P.l, bottom);
         if (oc) place(this.btn_edit, oc.r - wfx, bottom);
      }

      /** Is `o` one of this annotation's three objects? */
      owns(o) {
         return !!o && (o === this.text_obj || o === this.btn_close || o === this.btn_edit);
      }

      //-----------------------------------------------------------------------
      // The connector
      //-----------------------------------------------------------------------

      /** A two-triangle quad, rewritten every frame. The overlay camera is an
       * orthographic (0,1) box, so its coordinates are the same screen
       * fractions everything else here uses. */
      _makeConnector() {
         const RC = this.owner.RC;
         const g = new RC.Geometry();
         g.vertices = new RC.Float32Attribute(new Float32Array(18), 3);
         const m = new RC.MeshBasicMaterial();
         m.color = this.owner.viewer.fgCol;
         m.diffuse = this.owner.viewer.fgCol;
         m.lights = false;
         m.depthTest = false;
         // Both faces: the quad's winding follows the line direction, so it
         // flips as the annotation moves around its anchor.
         m.side = RC.FRONT_AND_BACK_SIDE;
         const mesh = new RC.Mesh(g, m);
         mesh.frustumCulled = false;
         mesh.pickable = false;

         // A plain Mesh has no setColors(). This one lets
         // GlViewerRCore.recolourFgElements() recolour the connector.
         mesh.use_fg_color = true;
         mesh.setColors = function(text_col, line_col) {
            m.color = line_col;
            m.diffuse = line_col;
         };
         return mesh;
      }

      /** Draw the connector from the box to the projected 3D anchor. The line
       * starts at a corner or edge midpoint of the box, chosen by which side
       * of the box the anchor falls on. No line is drawn when the anchor is
       * inside the box or outside the depth range. */
      updateConnector() {
         if (!this.conn) return;
         const v = this.owner.viewer, RC = this.owner.RC;
         const cam = v.camera;
         if (!cam || !v.canvas || !v.canvas.width) { this.conn.visible = false; return; }

         const W = v.canvas.width, H = v.canvas.height;
         const aspect = W / H;

         // 3D anchor -> overlay coordinates.
         const p = new RC.Vector3(this.anchor3d[0], this.anchor3d[1], this.anchor3d[2]);
         p.project(cam);
         if (p.z < -1 || p.z > 1) { this.conn.visible = false; return; }  // behind or beyond
         const tx = 0.5 * (p.x + 1.0), ty = 0.5 * (p.y + 1.0);

         const o = this._outer(this.text_obj, aspect);
         if (!o) { this.conn.visible = false; return; }

         const fx = tx < o.l ? 0.0 : (tx > o.r ? 1.0 : 0.5);
         const fy = ty < o.b ? 0.0 : (ty > o.t ? 1.0 : 0.5);
         if (fx === 0.5 && fy === 0.5) { this.conn.visible = false; return; }

         const ax = o.l + fx * (o.r - o.l);
         const ay = o.b + fy * (o.t - o.b);

         // Constant pixel width: overlay x and y span different pixel counts,
         // so the perpendicular is computed in pixels and converted back.
         const dx = (tx - ax) * W, dy = (ty - ay) * H;
         const len = Math.hypot(dx, dy);
         if (!(len > 1e-6)) { this.conn.visible = false; return; }
         const hw = 0.5 * this.owner.conn_width_px * (v.canvas.pixelRatio || 1);
         const nx = (-dy / len) * hw / W, ny = (dx / len) * hw / H;

         const a = this.conn.geometry.vertices;
         const V = a.array;
         let i = 0;
         const put = (x, y) => { V[i++] = x; V[i++] = y; V[i++] = 0; };
         put(ax - nx, ay - ny); put(ax + nx, ay + ny); put(tx + nx, ty + ny);
         put(ax - nx, ay - ny); put(tx + nx, ty + ny); put(tx - nx, ty - ny);
         // A same-length assignment bumps the attribute's version, so every
         // context re-uploads it.
         a.array = V;

         this.conn.visible = true;
      }

      /** Frame fraction for a button that gives the plate's absolute frame
       * width: frame_line / btn_scale, because line height scales with font
       * size. */
      _frameFrac() { return this.owner.frame_line / this.owner.btn_scale; }

      /** An object's frame width in ZText geometry units: a fraction of
       * viewport height, to be divided by aspect for x. Mirrors setText2D:
       * line_width * line_height, floored at MIN_FRAME_LINE_PX. Per object,
       * because the floor can apply to a button and not to the plate. */
      _frameWidthOf(obj) {
         const RC = this.owner.RC;
         const f = this.owner._font;
         if (!f) return 0;
         const fm = RC.ZText._fontMetrics(f.metrics, obj.fontSize, 0.0);
         const w  = obj.line_width * fm.line_height;
         return Math.max(w, RC.ZText.MIN_FRAME_LINE_PX * obj._pxToScreen);
      }

      /** Outer edges of an object's box: what getScreenRect reports, grown by
       * one frame width, since that rect is the inner fill. */
      _outer(obj, aspect) {
         const r = obj.getScreenRect(aspect);
         if (!r) return null;
         const wf = this._frameWidthOf(obj), wfx = wf / aspect;
         return { l: Math.min(r.x0, r.x1) - wfx, r: Math.max(r.x0, r.x1) + wfx,
                  b: Math.min(r.y0, r.y1) - wf,  t: Math.max(r.y0, r.y1) + wf };
      }

      /** True if (nx, ny), in overlay coordinates, is inside the bounding box
       * of the plate and both buttons, grown by hover_margin. */
      containsPointer(nx, ny) {
         if (!(nx >= -1) || !(ny >= -1)) return false;   // no pointer seen yet
         const v = this.owner.viewer;
         if (!v.canvas || !v.canvas.width) return false;
         const aspect = v.canvas.width / v.canvas.height;
         const m = this.owner.hover_margin;

         let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
         for (const o of [this.text_obj, this.btn_close, this.btn_edit]) {
            const r = this._outer(o, aspect);
            if (!r) continue;
            x0 = Math.min(x0, r.l); x1 = Math.max(x1, r.r);
            y0 = Math.min(y0, r.b); y1 = Math.max(y1, r.t);
         }
         if (!(x1 > x0)) return false;
         return nx >= x0 - m && nx <= x1 + m && ny >= y0 - m && ny <= y1 + m;
      }

      setButtonsVisible(on) {
         if (this._btns_on === on) return;
         this._btns_on = on;
         this.btn_close.visible = on;
         this.btn_edit.visible  = on;
         this.owner.viewer.request_render();
      }

      /** Re-weight after a resize: the drag changes fontSize directly through
       * ovlSetSize, so nothing else would notice it crossing the threshold. */
      syncWeight() {
         const w = this.owner.weightFor(this.text_obj.fontSize);
         if (Math.abs(this.text_obj.fontWeight - w) > 1e-6)
            this.text_obj.fontWeight = w;
      }

      setText(t) {
         this.text_obj.text = t;
         this.layout();
         this.owner.viewer.request_render();
      }

      remove() {
         const os = this.owner.viewer.overlay_scene;
         if (this.conn) os.remove(this.conn);
         os.remove(this.text_obj);
         os.remove(this.btn_close);
         os.remove(this.btn_edit);
         this.owner._forget(this);
      }

      /** Edit the text in an HTML textarea over the annotation. Text entry is
       * left to the DOM for caret, selection, IME and clipboard; the result is
       * still rendered by ZText as plain text.
       */
      edit() {
         if (this._editor) return;
         const v = this.owner.viewer;
         const dom = v.canvas.parentDOM;
         if (!dom) return;

         // Save and Discard buttons sit under the textarea, so the editor can
         // be left without knowing the keys. Their tooltips name the keys.
         const box = document.createElement('div');
         box.style.position = "absolute";
         box.style.zIndex = 1000;
         box.style.display = "flex";
         box.style.flexDirection = "column";
         box.style.background = "#fff";
         box.style.border = "1px solid #888";
         box.style.boxShadow = "0 2px 8px rgba(0,0,0,0.35)";

         const ta = document.createElement('textarea');
         ta.value = this.text_obj.text;
         ta.style.font = "13px monospace";
         ta.style.minWidth = "220px";
         ta.style.minHeight = "70px";
         ta.style.border = "none";
         ta.style.outline = "none";
         ta.style.margin = "0";
         ta.style.padding = "4px";
         ta.style.resize = "both";

         const row = document.createElement('div');
         row.style.display = "flex";
         row.style.justifyContent = "flex-end";
         row.style.gap = "4px";
         row.style.padding = "4px";
         row.style.borderTop = "1px solid #ddd";

         // Over the annotation itself: the thing being edited should not be
         // somewhere else on the screen while it is edited.
         const aspect = v.canvas.width / v.canvas.height;
         const r = this.text_obj.getScreenRect(aspect);
         const px = (v.canvas.pixelRatio || 1);
         if (r) {
            box.style.left = (Math.min(r.x0, r.x1) * v.canvas.width / px) + "px";
            box.style.top  = ((1 - Math.max(r.y0, r.y1)) * v.canvas.height / px) + "px";
         }

         // Runs once. Removing the box fires focusout, which would otherwise
         // call close(true) again, after Escape as well.
         let closed = false;
         const close = (apply) => {
            if (closed) return;
            closed = true;
            if (apply) this.setText(ta.value);
            box.remove();
            this._editor = null;
         };

         const button = (label, title, apply) => {
            const b = document.createElement('button');
            b.textContent = label;
            b.title = title;
            b.style.font = "12px sans-serif";
            b.style.padding = "2px 10px";
            b.style.cursor = "pointer";
            // Take the click without taking the focus, so the textarea never
            // blurs and focusout below stays a click-away-to-save.
            b.addEventListener('mousedown', (e) => e.preventDefault());
            b.addEventListener('click', (e) => { e.stopPropagation(); close(apply); });
            return b;
         };
         row.appendChild(button("Discard", "Esc", false));
         row.appendChild(button("Save", "Ctrl-Enter", true));

         ta.addEventListener('keydown', (e) => {
            // Every key is stopped here. The viewer keeps a window-level keydown
            // handler in which bare t, e and r rescale line widths, so typing any
            // of the three into the editor would also rescale the scene.
            e.stopPropagation();
            // Esc discards and Ctrl/Cmd-Enter saves. Plain Enter is a newline.
            if (e.key === "Escape") { e.preventDefault(); close(false); }
            else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
               e.preventDefault(); close(true);
            }
         });

         // Focus leaving the box saves. focusout with a relatedTarget check,
         // so tabbing between the textarea and the buttons is not an exit.
         box.addEventListener('focusout', (e) => {
            if (box.contains(e.relatedTarget)) return;
            close(true);
         });

         box.appendChild(ta);
         box.appendChild(row);
         dom.appendChild(box);
         this._editor = box;
         ta.focus();
         ta.select();
      }
   }

   return Annotations;
});
