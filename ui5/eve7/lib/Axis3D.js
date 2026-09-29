/** Axis3D -- world-space axes for a 3D viewer, in origin or box style.
 *
 * GlViewerRCore owns one Axis3D. It sets style, up axis, font size and
 * attenuation from REveViewer, passes the scene box to setBBox(), widens its
 * clip box by getRenderMargin(), and calls updateForCamera() from render().
 *
 * The axis is client-local. REveProjectionAxis lives on the server because
 * only the server knows the projection. A 3D axis takes its ticks from the
 * bounding box alone, and the camera decides only the presentation: which box
 * faces are drawn, which edges carry numbers, and which axes are edge-on.
 */

sap.ui.define([], function() {

   "use strict";

   /** Round tick values for an axis range, from d3's scaleLinear().ticks().
    *
    * This is the 1/2/5 x 10^n picker JSROOT's own axes use
    * (TAxisPainter.produceTicks() calls this.func.ticks()), so a REve 3D axis
    * picks the same numbers a TAxis would. JSROOT has already loaded d3, so the
    * import resolves without a fetch.
    */
   class TickSource {

      /** Resolves once the generator is usable. Everything that builds geometry
       * waits on this, the same way the font is waited on. */
      init() {
         if (this._ready) return this._ready;
         this._ready = import('jsrootsys/modules/d3.mjs').then(d3 => {
            this._scale = d3.scaleLinear;
            return this;
         });
         return this._ready;
      }

      /** Round values in [min, max], about n of them, and a formatter for them.
       * d3 picks the precision from the step, so all labels on an axis have
       * the same number of decimals and no floating-point noise. */
      ticks(min, max, n) {
         if (!this._scale || !(max > min)) return { values: [], format: String };
         const s = this._scale().domain([min, max]);
         const v = s.ticks(n);
         const d3f = s.tickFormat(n);

         // d3 formats negatives with U+2212 MINUS SIGN. The generated SDF atlases
         // lack that glyph and ZText substitutes "?", so fold it to ASCII '-'.
         const f = x => d3f(x).replace(/\u2212/g, '-');
         return { values: v, format: f };
      }
   }

   /** Axis style. The values match REveViewer::EAxesType (kAxesNone,
    * kAxesOrigin, kAxesEdge), which GlViewerRCore passes to setStyle() as is. */
   const STYLE = { NONE: 0, ORIGIN: 1, BOX: 2 };

   class Axis3D {

      /** @param viewer the owning GlViewerRCore; used for its camera, texture
       * cache, stripe factory, foreground colour, top_path and request_render.
       * @param RC the RenderCore module. */
      constructor(viewer, RC) {
         this.viewer = viewer;
         this.RC = RC;

         this.group = new RC.Group();
         this.group.name = "Axis3D";

         this.style = STYLE.NONE;
         this.atten = RC.Z3DAxis.ATTEN_FIXED;

         /** Which axis points up, or -1 for no opinion. See setUpAxis(). */
         this.up_axis = -1;

         /** Target number of labelled ticks per axis. Unlike the projected axis
          * this is not over-provided: there is no client-side filtering step to
          * feed, because a 3D tick's position is known exactly here. */
         this.n_ticks = 5;

         /** Tick length, as a fraction of the bounding-box diagonal. It is also
          * the unit of num_out and name_out. World-space, so ticks scale with
          * the scene; the Z3DAxis class doc says what fixed-pixel ticks need. */
         this.tick_frac = 0.02;

         /** Label placement, in world space.
          *
          * name_frac: where the box-style axis name sits along its edge, as a
          * fraction of the axis extent. The middle keeps it away from the
          * corners, where the three axes meet and names collide with the end
          * numbers.
          *
          * num_out, name_out: how far numbers and names stand out, in tick
          * lengths. Numbers, and box-style names, step out along both axes
          * perpendicular to their own, which lifts them off the panel planes and
          * their grids. Origin-style names sit on the axis, name_out past each
          * end. */
         this.name_frac = 0.5;
         this.num_out   = 2.2;
         this.name_out  = 5.0;

         /** Label size, as a fraction of viewport height. */
         this.font_size = 0.018;
         this.font_name = "LiberationSerif-Regular";

         /** An axis whose full extent projects shorter than this in NDC (the
          * viewport spans 2) is edge-on and not drawn, since it has no screen
          * direction to lay ticks along. The exact case is an orthographic
          * camera looking down the axis, such as z in an XOY view. Kept small so
          * that a foreshortened perspective axis is still drawn. */
         this.min_axis_ndc = 0.02;

         this.ticks = new TickSource();
         this.bbox = null;
         this.labels_obj = null;
         /** Per axis: is it edge-on for the current camera? Set at build time
          * by _degenerate(); the camera check rebuilds when the set changes. */
         this._degen = [false, false, false];

         /** Cached once delivered. The box style rebuilds whenever the camera
          * crosses one of the box's mid-planes, too often to re-request a font. */
         this._font = null;
         /** Edge-on key plus octant key the current geometry was built for,
          * from _degenKey() and _octantKey(). updateForCamera() rebuilds when
          * it changes. Null means not built for any camera. */
         this._camSig = null;
      }

      /** Origin, box, or nothing. */
      setStyle(style) {
         if (style === this.style) return;
         this.style = style;
         this.rebuild();
      }

      /** Label size attenuation with distance: 0 keeps a constant pixel size,
       * 1 shrinks like geometry. A uniform, so no rebuild. */
      setAttenuation(k) {
         this.atten = k;
         if (this.labels_obj) this.labels_obj.setAttenuation(k);
         this.viewer.request_render();
      }

      getAttenuation() { return this.atten; }

      /** Label size, as a fraction of viewport height. It is baked into the
       * glyph quads, so this rebuilds the label geometry only, not through
       * rebuild(). */
      setFontSize(sz) {
         if (!(sz > 0)) return;
         this.font_size = sz;
         // No early return on an unchanged size. GlViewerRCore assigns
         // font_size directly before setStyle(), so the field can already hold
         // the new value while labels_obj still has the old one.
         // Z3DAxis.setFontSize() compares against the size it was built with.
         if (this.labels_obj) {
            this.labels_obj.setFontSize(sz);
            this.viewer.request_render();
         }
      }

      /** How far past the bounding box the camera's clip box must extend for
       * this axis, in world units.
       *
       * GlViewerRCore fits the perspective near and far planes to the bounding
       * box, and the box excludes the axis, which is built from it. The margin
       * is one tick length, for the tick stubs, which are ordinary Stripes.
       * Numbers and names stand further out but are ZText glyphs, whose anchor
       * shader clamps their depth into the frustum. A margin sized to the names
       * would be a tenth of the diagonal and cost depth precision everywhere. */
      getRenderMargin() {
         if (!this.bbox || this.style === STYLE.NONE) return 0;
         const b = this.bbox;
         const diag = Math.hypot(b.max.x - b.min.x,
                                 b.max.y - b.min.y,
                                 b.max.z - b.min.z);
         return this.tick_frac * diag;
      }

      /** The scene extent the axis describes. Rebuilds only on a real change:
       * recalcSceneBBox runs often and an identical box must not throw the
       * geometry away. */
      setBBox(bbox) {
         if (!bbox) return;
         const b = this.bbox;
         if (b && b.min.equals(bbox.min) && b.max.equals(bbox.max)) return;
         this.bbox = { min: bbox.min.clone(), max: bbox.max.clone() };
         this.rebuild();
      }

      clear() {
         this.group.clear();
         this.labels_obj = null;
      }

      rebuild() {
         this.clear();
         this._camSig = null;
         if (this.style === STYLE.NONE || !this.bbox) return;

         // The tick generator and the font both load asynchronously. The build
         // waits for the ticks, then for the font. A second rebuild while one is
         // in flight is harmless because _build() starts by clearing.
         this.ticks.init().then(() => {
            if (this._font) return this._build();
            this._withFont(f => { this._font = f; this._build(); });
         });
      }

      _withFont(cb) {
         // top_path, not eve_path: REveText registers the font directory with
         // gEve->AddLocation("sdf-fonts/", ...), at the server's top level.
         const url_base = this.viewer.top_path + 'sdf-fonts/' + this.font_name;
         this.viewer.tex_cache.deliver_font(url_base,
            (texture, font_metrics) => { cb({ texture, metrics: font_metrics }); },
            (img) => this.RC.ZText.createDefaultTexture(img),
            () => this.viewer.request_render()
         );
      }

      _build() {
         this.clear();
         const font = this._font;
         if (this.style === STYLE.NONE || !this.bbox || !font) return;

         const RC = this.RC;
         const lines = [];    // flat [x0,y0,z0, x1,y1,z1] runs, one per segment
         const labels = [];   // for Z3DAxis

         // Which axes are edge-on for this camera. Both builders skip those
         // entirely -- there is no screen direction to lay a scale along.
         this._degen = this._degenerate(this.viewer.camera);

         if (this.style === STYLE.ORIGIN) {
            this._buildOrigin(lines, labels);
         } else {
            this._buildBox(lines, labels);
         }

         // Record what this geometry was built for, so the camera check does
         // not immediately rebuild the very thing it just triggered.
         if (this.viewer.camera) {
            const ok = (this.style === STYLE.BOX) ? this._octantKey(this.viewer.camera) : "";
            this._camSig = this._degenKey(this.viewer.camera) + "/" + ok;
         }

         // ---- lines and ticks ------------------------------------------------
         // Stripes, not the label buffer: a 3D line of constant pixel width
         // needs the screen-space perpendicular, which depends on the camera.
         // Stripes computes it in its vertex shader.
         //
         // Batched by width and colour. Stripes draws each consecutive vertex
         // pair as one instanced segment, so all lines of one style share one
         // buffer and one draw call, as in makeStraightLineSet().
         const groups = new Map();
         for (const seg of lines) {
            const key = seg.width + "|" + seg.color.getHex();
            let g = groups.get(key);
            if (!g) { g = { width: seg.width, color: seg.color, pts: [] }; groups.set(key, g); }
            for (const c of seg.pts) g.pts.push(c);
         }

         for (const g of groups.values()) {
            const geom = new RC.Geometry();
            geom.vertices = new RC.Float32Attribute(new Float32Array(g.pts), 3);
            const ss = this.viewer.creator.RcMakeStripes(geom, g.width, g.color);
            this.group.add(ss);
         }

         // ---- labels ---------------------------------------------------------
         // One object for every string on every axis: the anchor is per vertex,
         // so N labels are one draw call and no per-label matrix.
         if (labels.length) {
            const lo = new RC.Z3DAxis({
               text: "",
               fontTexture: font.texture,
               font: font.metrics,
               fontSize: this.font_size,
               fontHinting: 1.0,
               color: this.viewer.fgCol,
               atten: this.atten
            });
            lo.material.side = RC.FRONT_SIDE;
            // The axis must stay legible when the background flips.
            // GlViewerRCore.recolourFgElements() calls setColors() on every
            // object with use_fg_color set, so the viewer needs no special case.
            lo.use_fg_color = true;
            lo.setLabels(labels);
            this.labels_obj = lo;
            this.group.add(lo);
         }

         this.viewer.request_render();
      }

      /** Lines through the origin along each axis, ticked and labelled.
       *
       * Each line spans [min(0, box min), max(0, box max)] along its axis, so it
       * covers the box extent and always includes the origin. */
      _buildOrigin(lines, labels) {
         const RC = this.RC;
         const b = this.bbox;
         const mn = [b.min.x, b.min.y, b.min.z];
         const mx = [b.max.x, b.max.y, b.max.z];

         const diag = Math.hypot(mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]);
         const tick_len = this.tick_frac * diag;
         if (!(diag > 0)) return;

         // One neutral grey for all three axes, as in the box style. Every ray
         // is named at its ends, so colour does not have to tell them apart.
         const col = new RC.Color(0.55, 0.55, 0.55);

         const AX = [
            { i: 0, name: "x" },
            { i: 1, name: "y" },
            { i: 2, name: "z" }
         ];

         const pt = (i, v) => { const p = [0, 0, 0]; p[i] = v; return p; };

         for (const ax of AX) {
            const i = ax.i;
            if (this._degen[i]) continue;   // edge-on: nothing to lay out along
            const lo = Math.min(0, mn[i]), hi = Math.max(0, mx[i]);
            if (!(hi > lo)) continue;

            lines.push({ pts: [...pt(i, lo), ...pt(i, hi)], width: 1.5, color: col });

            const t = this.ticks.ticks(lo, hi, this.n_ticks);

            // Ticks go out along the next axis round the cycle: x ticks lie
            // along y, y along z, z along x. Any fixed choice is arbitrary, but
            // cycling keeps the three from all landing in one plane, where two
            // of them would overlap edge-on from the commonest viewpoints.
            const j = (i + 1) % 3, k = (i + 2) % 3;

            // Numbers step away along both perpendicular axes, so they do not lie
            // in the plane of the tick marks. An origin axis runs through the
            // scene and has no outward side, so the same diagonal is used for
            // every tick on the axis.
            const away = (p, f) => {
               const q = p.slice();
               q[j] += f * tick_len;
               q[k] += f * tick_len;
               return q;
            };

            for (const v of t.values) {
               if (Math.abs(v) < 1e-12) continue;    // the origin needs no tick

               const p0 = pt(i, v);
               const p1 = pt(i, v); p1[j] += tick_len;
               lines.push({ pts: [...p0, ...p1], width: 1.2, color: col });

               labels.push({
                  text: t.format(v),
                  pos: away(p0, this.num_out),
                  px: 0, py: 0,
                  ah: RC.ZText.ALIGN_H.CENTER,
                  av: RC.ZText.ALIGN_V.MIDDLE
               });
            }

            // Names sit on the axis, name_out tick lengths past its ends: "x"
            // past the positive end, and "-x" past the negative end when there
            // is one. Each name continues its own ray, which makes the sign
            // unambiguous. The ends are clear of the scene, so no perpendicular
            // offset is added.
            const nout = this.name_out * tick_len;

            labels.push({
               text: ax.name,
               pos: pt(i, hi + nout),
               px: 0, py: 0,
               ah: RC.ZText.ALIGN_H.CENTER,
               av: RC.ZText.ALIGN_V.MIDDLE
            });

            // Only when the axis actually reaches negative values: lo is
            // min(0, bbox min), so a scene sitting entirely on the positive
            // side has no negative end to name.
            if (lo < 0) {
               labels.push({
                  text: "-" + ax.name,
                  pos: pt(i, lo - nout),
                  px: 0, py: 0,
                  ah: RC.ZText.ALIGN_H.CENTER,
                  av: RC.ZText.ALIGN_V.MIDDLE
               });
            }
         }
      }

      //-----------------------------------------------------------------------
      // Box style (kAxesEdge)
      //-----------------------------------------------------------------------

      /** Camera position in world space, from the view matrix.
       *
       * Uses matrixWorldInverse, the VMat the renderer hands to shaders and the
       * matrix Vector3.project() applies in _degenerate() and _buildBox(), so
       * the eye point agrees with those projections.
       *
       * For V = [R|t] the camera sits at -R^T t. With column-major elements
       * that is minus the dot of each column of R with t. For an orthographic
       * camera the result still lies on the view axis on the near side, which
       * is all the back-face test needs. */
      _camPos(camera) {
         const e = camera.matrixWorldInverse.elements;
         const tx = e[12], ty = e[13], tz = e[14];
         return [-(e[0]*tx + e[1]*ty + e[2] *tz),
                 -(e[4]*tx + e[5]*ty + e[6] *tz),
                 -(e[8]*tx + e[9]*ty + e[10]*tz)];
      }

      /** Which axes are edge-on for this camera, as three booleans.
       *
       * Measured by projecting each axis's box extent and comparing its screen
       * length with min_axis_ndc. Unlike reading the camera type, this also
       * covers a camera the user has rotated. */
      _degenerate(camera) {
         const RC = this.RC, b = this.bbox;
         const out = [false, false, false];
         if (!camera || !b) return out;

         const mid = [0.5 * (b.min.x + b.max.x),
                      0.5 * (b.min.y + b.max.y),
                      0.5 * (b.min.z + b.max.z)];
         const ext = [b.max.x - b.min.x, b.max.y - b.min.y, b.max.z - b.min.z];

         const proj = (p) => {
            const v = new RC.Vector3(p[0], p[1], p[2]);
            v.project(camera);
            return v;
         };
         for (let a = 0; a < 3; ++a) {
            if (!(ext[a] > 0)) { out[a] = true; continue; }
            const p0 = mid.slice(), p1 = mid.slice();
            p0[a] -= 0.5 * ext[a];
            p1[a] += 0.5 * ext[a];
            const s0 = proj(p0), s1 = proj(p1);
            out[a] = Math.hypot(s1.x - s0.x, s1.y - s0.y) < this.min_axis_ndc;
         }
         return out;
      }

      _degenKey(camera) {
         return this._degenerate(camera).map(d => d ? "1" : "0").join("");
      }

      /** Which axis points up, from REveViewer::SetAxesUpAxis. -1 for none. */
      setUpAxis(a) {
         a = (a >= 0 && a <= 2) ? a : -1;
         if (a === this.up_axis) return;
         this.up_axis = a;
         this.rebuild();
      }

      /** Which side of the box centre the camera is on along each axis, as a
       * 3-character key such as "+-+".
       *
       * The back panels, labelled edges and tick directions of the box style
       * depend on the camera only through this key, so the geometry is rebuilt
       * when it changes rather than every frame. */
      _octantKey(camera) {
         const c = this._camPos(camera), b = this.bbox;
         const mid = [0.5 * (b.min.x + b.max.x),
                      0.5 * (b.min.y + b.max.y),
                      0.5 * (b.min.z + b.max.z)];
         let k = "";
         for (let a = 0; a < 3; ++a) k += (c[a] > mid[a]) ? "+" : "-";
         return k;
      }

      /** A box round the scene: the three faces away from the camera, ruled at
       * the tick values, with numbers along three silhouette edges. Only the
       * far faces are drawn, so the panels stay behind the geometry. The drawn
       * set flips as the camera crosses the box's mid-planes. */
      _buildBox(lines, labels) {
         const RC = this.RC;
         const b = this.bbox;
         const mn = [b.min.x, b.min.y, b.min.z];
         const mx = [b.max.x, b.max.y, b.max.z];
         const camera = this.viewer.camera;
         if (!camera) return;

         const diag = Math.hypot(mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]);
         if (!(diag > 0)) return;

         const cam = this._camPos(camera);
         const mid = [0.5*(mn[0]+mx[0]), 0.5*(mn[1]+mx[1]), 0.5*(mn[2]+mx[2])];

         // back[a] is the coordinate of the face pointing away from the camera
         // along axis a; front[a] is the near one.
         const back = [], front = [];
         for (let a = 0; a < 3; ++a) {
            const camOnMaxSide = cam[a] > mid[a];
            back[a]  = camOnMaxSide ? mn[a] : mx[a];
            front[a] = camOnMaxSide ? mx[a] : mn[a];
         }

         // Along the viewer's up axis the floor (min face) is always the back
         // panel. From eye height inside a scene the far face along up is the
         // ceiling, which would leave the floor unruled. A floor panel never
         // comes between the camera and content resting on it; a near side wall
         // would, so the other axes keep the back-face rule.
         if (this.up_axis >= 0 && this.up_axis <= 2) {
            back[this.up_axis]  = mn[this.up_axis];
            front[this.up_axis] = mx[this.up_axis];
         }

         const col  = new RC.Color(0.55, 0.55, 0.55);
         const gcol = new RC.Color(0.75, 0.75, 0.75);
         const at = (a, v, j, jv, k, kv) => {
            const p = [0, 0, 0];
            p[a] = v; p[j] = jv; p[k] = kv;
            return p;
         };

         const tick_sets = [];
         for (let a = 0; a < 3; ++a)
            tick_sets.push(this.ticks.ticks(mn[a], mx[a], this.n_ticks));

         // ---- the three back panels: border plus grid ------------------------
         // Grid lines use the same tick values as the numbers, so each grid line
         // continues a tick across the panel.
         for (let a = 0; a < 3; ++a) {
            const j = (a + 1) % 3, k = (a + 2) % 3;
            const v = back[a];

            // border
            const corners = [
               at(a, v, j, mn[j], k, mn[k]),
               at(a, v, j, mx[j], k, mn[k]),
               at(a, v, j, mx[j], k, mx[k]),
               at(a, v, j, mn[j], k, mx[k])
            ];
            for (let i = 0; i < 4; ++i)
               lines.push({ pts: [...corners[i], ...corners[(i+1)%4]],
                            width: 1.5, color: col });

            // grid, ruled in both in-plane directions
            for (const t of tick_sets[j].values) {
               if (t <= mn[j] || t >= mx[j]) continue;
               lines.push({ pts: [...at(a, v, j, t, k, mn[k]),
                                  ...at(a, v, j, t, k, mx[k])],
                            width: 1, color: gcol });
            }
            for (const t of tick_sets[k].values) {
               if (t <= mn[k] || t >= mx[k]) continue;
               lines.push({ pts: [...at(a, v, j, mn[j], k, t),
                                  ...at(a, v, j, mx[j], k, t)],
                            width: 1, color: gcol });
            }
         }

         // ---- numbers, on three silhouette edges -----------------------------
         // For each axis there are two candidate edges on the boundary of the
         // drawn panels: back in one of the other two axes and front in the
         // other. The one whose midpoint projects further from the box centre
         // on screen is used, so the numbers sit outside the silhouette and not
         // in the crease where the panels meet, where geometry would cross them.
         const scr = (p) => {
            const v = new RC.Vector3(p[0], p[1], p[2]);
            v.project(camera);
            return [v.x, v.y];
         };
         const midScr = scr(mid);

         const tick_len = this.tick_frac * diag;

         for (let a = 0; a < 3; ++a) {
            if (this._degen[a]) continue;   // edge-on: no scale to draw
            const j = (a + 1) % 3, k = (a + 2) % 3;

            // `out` is the axis the tick runs along. Each candidate edge sits
            // ON one back panel (the axis held at its back coordinate) and at
            // the far rim of it (the axis held at its front coordinate). The
            // tick therefore has to step along the FRONT one: that keeps it in
            // the plane of its own panel and takes it out past the silhouette.
            // Stepping along both would send it diagonally out of the corner,
            // in the plane of neither.
            const cands = [
               { jv: back[j],  kv: front[k], out: k },
               { jv: front[j], kv: back[k],  out: j }
            ];
            let best = null, bestD = -1;
            for (const c of cands) {
               const m = scr(at(a, 0.5*(mn[a]+mx[a]), j, c.jv, k, c.kv));
               const d = Math.hypot(m[0]-midScr[0], m[1]-midScr[1]);
               if (d > bestD) { bestD = d; best = c; }
            }

            const o = best.out;
            const ov = (o === j) ? best.jv : best.kv;
            const dir = (Math.sign(ov - mid[o]) || 1) * tick_len;
            // The tick mark itself: short, and along `out` only, so it stays in
            // the plane of its own panel and reads as attached to the edge.
            const step = (p, f) => { const q = p.slice(); q[o] += f * dir; return q; };

            // The numbers stand further out, along both perpendicular axes, so
            // they sit off the panel planes and their grid lines.
            const dj = (Math.sign(best.jv - mid[j]) || 1) * tick_len;
            const dk = (Math.sign(best.kv - mid[k]) || 1) * tick_len;
            const away = (p, f) => {
               const q = p.slice();
               q[j] += f * dj;
               q[k] += f * dk;
               return q;
            };

            const t = tick_sets[a];
            for (const val of t.values) {
               if (val < mn[a] || val > mx[a]) continue;

               const p0 = at(a, val, j, best.jv, k, best.kv);
               lines.push({ pts: [...p0, ...step(p0, 1)], width: 1.2, color: col });

               labels.push({
                  text: t.format(val),
                  pos: away(p0, this.num_out),
                  px: 0, py: 0,
                  ah: RC.ZText.ALIGN_H.CENTER,
                  av: RC.ZText.ALIGN_V.MIDDLE
               });
            }

            // Axis name: at the middle of the labelled edge, which is as far
            // from both corners as it gets, and standing further out again than
            // the numbers along the same diagonal.
            const nm = ["x", "y", "z"][a];
            const nv = mn[a] + this.name_frac * (mx[a] - mn[a]);
            const e = away(at(a, nv, j, best.jv, k, best.kv), this.name_out);
            labels.push({
               text: nm, pos: e, px: 0, py: 0,
               ah: RC.ZText.ALIGN_H.CENTER, av: RC.ZText.ALIGN_V.MIDDLE
            });
         }
      }

      /** Per-frame camera work, called from GlViewerRCore.render().
       *
       * Sets the attenuation reference, the clip-space w of the box centre,
       * where labels are drawn at their nominal size. Under an orthographic
       * camera w is 1 and attenuation has no effect. Then rebuilds the geometry
       * if the edge-on axes or, in box style, the camera octant have changed.
       * Returns true when it rebuilt. */
      updateForCamera(camera) {
         if (!this.labels_obj || !this.bbox || !camera) return false;

         const RC = this.RC;
         const c = new RC.Vector3(0.5 * (this.bbox.min.x + this.bbox.max.x),
                                  0.5 * (this.bbox.min.y + this.bbox.max.y),
                                  0.5 * (this.bbox.min.z + this.bbox.max.z));

         // w of the centre under the current view-projection. A Vector4, since
         // Vector3.project() divides by w and w is the value wanted.
         const v = new RC.Vector4(c.x, c.y, c.z, 1.0);
         v.applyMatrix4(camera.matrixWorldInverse);
         v.applyMatrix4(camera.projectionMatrix);
         this.labels_obj.setReferenceW(v.w);

         // The geometry depends on the camera only through the edge-on axes
         // (both styles) and the octant (box style), so it is rebuilt when
         // their combined key changes. The key costs six point projections and
         // three comparisons per frame.
         const dk = this._degenKey(camera);
         const ok = (this.style === STYLE.BOX) ? this._octantKey(camera) : "";
         const sig = dk + "/" + ok;
         if (sig !== this._camSig) {
            this._camSig = sig;
            this._build();
            return true;
         }
         return false;
      }
   }

   Axis3D.STYLE = STYLE;
   return Axis3D;
});
