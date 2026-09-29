/** Overlay -- the viewer's overlay scene: pointer interaction with its
 * elements (hover highlight, move and resize by drag, click), and per-frame
 * upkeep (the CSS-pixel scale, and projection-axis layout for the camera).
 *
 * Moving and resizing are client-local and send no MIR, so other clients do
 * not see them. A click on an element with a click action sends one, see
 * click().
 */

sap.ui.define([], function() {

   "use strict";

   class Overlay {

      /** @param viewer a GlViewerRCore; used for the overlay scene, the canvas,
       * the orbit controls, overlay picking and request_render. */
      constructor(viewer, RC) {
         this.viewer = viewer;
         this.RC = RC;

         this.drag = null;      // set from mouse-down on an element until mouse-up
         this.hover = null;     // element currently highlighted
         this.nx = NaN;         // last pointer position, overlay coordinates;
         this.ny = NaN;         // NaN until the pointer has been seen
         this.px_to_screen = 0; // CSS pixel in screen space, 0 until known
      }

      /** Pointer position as canvas pixels and as overlay coordinates, (0,0)
       * bottom-left to (1,1) top-right, the space of ZText's `offset`. */
      normCoords(event) {
         const c = this.viewer.canvas;
         const x = event.offsetX * c.pixelRatio;
         const y = event.offsetY * c.pixelRatio;
         return { px: x, py: y, nx: x / c.width, ny: 1.0 - y / c.height };
      }

      /** "resize" if the pointer is on a resizable element's grip, the square the
       * element draws at its bottom-right corner, otherwise "move". */
      grabZone(obj, nx, ny) {
         if (!obj.resizable || typeof obj.getScreenRect !== "function") return "move";

         const c = this.viewer.canvas;
         const r = obj.getScreenRect(c.width / c.height);
         if (!r || !(r.grip_x > 0) || !(r.grip_y > 0)) return "move";

         const xmax = Math.max(r.x0, r.x1), ymin = Math.min(r.y0, r.y1);
         return (nx > xmax - r.grip_x && ny < ymin + r.grip_y) ? "resize" : "move";
      }

      /** Re-lay the projection axes for the current orthographic camera from the
       * unprojected NDC corners, so zooming needs no round trip. Returns true if
       * any axis was rebuilt. */
      updateProjectionAxes() {
         const v = this.viewer;
         if (!v.overlay_scene || !v.camera) return false;

         const p0 = new this.RC.Vector3(-1, -1, 0).unproject(v.camera);
         const p1 = new this.RC.Vector3( 1,  1, 0).unproject(v.camera);
         const l = Math.min(p0.x, p1.x), r = Math.max(p0.x, p1.x);
         const b = Math.min(p0.y, p1.y), t = Math.max(p0.y, p1.y);
         const aspect = v.canvas.width / v.canvas.height;

         let rebuilt = false;
         v.overlay_scene.traverse(o => {
            if (o.type === "ZTextAxis" && o.updateForCamera(l, r, b, t, aspect))
               rebuilt = true;
         });
         return rebuilt;
      }

      /** Pass the size of a CSS pixel in screen space to every overlay element
       * with setPixelScale(), for its pixel floors: grip, frame line and font
       * size. Only when it changes, i.e. on a resize or a display-scale
       * change, because the elements rebuild their vertices. */
      updatePixelScale() {
         const v = this.viewer;
         if (!v.canvas || !v.canvas.height) return;
         const f = (v.canvas.pixelRatio || 1) / v.canvas.height;
         if (f === this.px_to_screen) return;
         this.px_to_screen = f;

         if (v.overlay_scene) {
            const W = v.canvas.width, H = v.canvas.height;
            v.overlay_scene.traverse(o => {
               if (typeof o.setPixelScale === "function") o.setPixelScale(f, W, H);
            });
         }
      }

      /** The topmost pickable overlay element whose screen rect contains the
       * point. A rect test rather than GPU picking, so it can run on every
       * mouse move; mouse-down uses GPU picking. */
      hoverTest(nx, ny) {
         const c = this.viewer.canvas;
         const aspect = c.width / c.height;
         let hit = null;
         this.viewer.overlay_scene.traverse(o => {
            if (!o.pickable || !o.visible || typeof o.getScreenRect !== "function") return;
            const r = o.getScreenRect(aspect);
            if (!r) return;
            if (nx >= Math.min(r.x0, r.x1) && nx <= Math.max(r.x0, r.x1) &&
                ny >= Math.min(r.y0, r.y1) && ny <= Math.max(r.y0, r.y1))
               hit = o;   // later in the traversal is drawn on top
         });
         return hit;
      }

      /** Highlight the element under the pointer, at most one at a time. */
      updateHover(event) {
         if (this.drag) return;

         const c = this.normCoords(event);
         // Kept for Annotations, which tests against a region wider than the
         // element itself: an annotation plus its buttons.
         this.nx = c.nx;
         this.ny = c.ny;

         const hit = this.hoverTest(c.nx, c.ny);
         if (hit === this.hover) return;

         this.setHighlight(this.hover, false);
         this.hover = hit;
         this.setHighlight(hit, true);
         this.viewer.request_render();
      }

      clearHover() {
         if (!this.hover) return;
         this.setHighlight(this.hover, false);
         this.hover = null;
         this.viewer.request_render();
      }

      setHighlight(obj, on) {
         if (obj && typeof obj.setHighlight === "function") obj.setHighlight(on);
      }

      /** Start a drag if the press is on a movable overlay element. Returns true
       * if it is, and then disables pan and rotate until mouse-up. */
      onMouseDown(event) {
         if (this.drag) return false;

         const v = this.viewer;
         const c = this.normCoords(event);

         // RCore-side pick diagnostics are gated on window.__RC_PICKDBG; see
         // MeshRenderer._renderPickableObjects.
         if (v._logLevel >= 3) window.__RC_PICKDBG = true;
         const pstate = v.render_for_Overlay_picking(c.px, c.py, false);
         window.__RC_PICKDBG = false;
         if (v._logLevel >= 3)
            console.log("overlay pick at " + c.px + "," + c.py +
                        " hit=" + (!!(pstate && pstate.object)));
         if (!pstate || !pstate.object) return false;

         const obj = pstate.object;
         if (typeof obj.ovlGetPos !== "function") return false;
         const rect = (typeof obj.getScreenRect === "function")
                    ? obj.getScreenRect(v.canvas.width / v.canvas.height) : null;

         this.drag = {
            obj:       obj,
            zone:      this.grabZone(obj, c.nx, c.ny),
            grab_nx:   c.nx,
            grab_ny:   c.ny,
            orig_x:    obj.ovlGetPos()[0],
            orig_y:    obj.ovlGetPos()[1],
            orig_size: obj.ovlGetSize(),
            // Resize anchors on the left edge. The reference width runs to the
            // grab point rather than to the right edge, so the scale is 1 at
            // the grab and the box does not jump.
            anchor_x:  rect ? Math.min(rect.x0, rect.x1) : 0,
            grab_w:    rect ? (c.nx - Math.min(rect.x0, rect.x1)) : 0,
            // Still false at mouse-up means a click. Overlay elements are all
            // movable, so a click can only be told from a drag on release.
            moved:     false
         };

         v.controls.enablePan = false;
         v.controls.enableRotate = false;
         return true;
      }

      onMouseMove(event) {
         const d = this.drag;
         if (!d) { this.updateHover(event); return; }

         const c = this.normCoords(event);

         if (Math.abs(c.nx - d.grab_nx) > Overlay.CLICK_SLOP ||
             Math.abs(c.ny - d.grab_ny) > Overlay.CLICK_SLOP)
            d.moved = true;

         if (d.zone === "move") {
            d.obj.ovlSetPos(d.orig_x + (c.nx - d.grab_nx),
                            d.orig_y + (c.ny - d.grab_ny));
         } else if (d.grab_w > 1e-4) {
            const f = (c.nx - d.anchor_x) / d.grab_w;
            d.obj.ovlSetSize(Math.max(d.orig_size * f, 1e-4));
         }

         this.viewer.request_render();
      }

      onMouseUp() {
         if (!this.drag) return;

         if (!this.drag.moved) this.click(this.drag.obj);

         this.drag = null;
         const v = this.viewer;
         v.controls.enablePan = true;
         v.controls.enableRotate = true;
         v.request_render();
      }

      /** A click on an overlay element. A local `onOverlayClick` handler wins, which
       * is how annotation buttons act on the client alone. Otherwise, if the
       * streamed element carries a click action, send its MIR: a button exists
       * to change server state, so the result reaches every client. */
      click(obj) {
         if (obj && typeof obj.onOverlayClick === "function") { obj.onOverlayClick(); return; }

         const el = obj ? obj.eve_el : null;
         if (!el || !el.fClickMir) return;

         const mgr = this.viewer.controller ? this.viewer.controller.mgr : null;
         if (!mgr) return;

         // fClickTargetId 0 means the element itself.
         const tid = el.fClickTargetId || el.fElementId;
         const tgt = mgr.GetElement(tid);
         if (!tgt) {
            console.error("Overlay.click: no element", tid, "for MIR", el.fClickMir);
            return;
         }
         mgr.SendMIR(el.fClickMir, tid, tgt._typename);
      }

      /** Click/drag threshold in overlay coordinates, about 3 px on a typical
       * view: above pointer jitter, below a deliberate drag. */
      static CLICK_SLOP = 0.004;
   }

   return Overlay;
});
