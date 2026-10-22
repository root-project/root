sap.ui.define([], function() {

   "use strict";

   /** Client-side extrapolation of streamed motion, one per GlViewerRCore.
     *
     * REveTrans::SetMotion() gives an element's transformation a velocity, an
     * acceleration, a spin axis and rate in the local frame, a start time t0
     * and a trust window max_dt. Every animation frame this places the object at
     *
     *     p(t) = p0 + v*dt + 0.5*a*dt^2,   dt = clamp(now - t0, 0, max_dt) in s
     *
     * and turns its basis by rate*dt about the spin axis. p0 and the basis come
     * from the matrix in the same update. The spin composes on the right, so
     * it preserves the object only while the basis scale is uniform.
     */
   class Motion {

      constructor(viewer) {
         this.viewer  = viewer;
         this.objects = new Map();   // element id -> entry
         this.raf     = 0;
         this.enabled = true;

         /** Cap on animation-driven redraws per second; 0 is uncapped. */
         this.render_max_hz = 0;
         this._last_render  = 0;
         this._pending      = 0;
      }

      /** How often the animation loop may redraw, from
        * REveViewer::SetRenderMaxHz. Zero is uncapped. Positions are evaluated
        * at draw time, so a capped viewer is coarser but not behind.
        */
      setRenderMaxHz(hz) {
         this.render_max_hz = (hz > 0) ? hz : 0;
      }

      /** Ask for a redraw, subject to render_max_hz.
        *
        * The animation loop and arriving updates (EveScene.sceneElementMotion)
        * both redraw through here, so the cap covers both. A suppressed request
        * is deferred to a trailing timer rather than dropped, so the last update
        * before the scene goes still is drawn, also with extrapolation off.
        */
      requestRender() {
         if (this.render_max_hz <= 0) { this.viewer.request_render(); return; }

         const now  = performance.now();
         const gap  = 1000 / this.render_max_hz;
         const wait = gap - (now - this._last_render);

         if (wait <= 0) {
            this._last_render = now;
            this.viewer.request_render();
            return;
         }

         if (!this._pending) {
            this._pending = setTimeout(() => {
               this._pending = 0;
               this._last_render = performance.now();
               this.viewer.request_render();
            }, wait);
         }
      }

      /** Per-viewer switch for extrapolation, from
        * REveViewer::SetExtrapolateMotion.
        *
        * Off draws every object where its last update put it, which shows the
        * real update rate. Turning it off restores each object's position and
        * basis to the last values the server sent.
        */
      setEnabled(on) {
         on = !!on;
         if (on === this.enabled) return;
         this.enabled = on;

         if (on) {
            this.start();
         } else {
            this.stop();
            for (const e of this.objects.values()) {
               const m = e.obj3d._matrix.elements, r = e.r0;
               m[12] = e.p0[0]; m[13] = e.p0[1]; m[14] = e.p0[2];
               for (let cIdx = 0; cIdx < 3; ++cIdx) {
                  const o = cIdx * 4;
                  m[o] = r[cIdx*3]; m[o+1] = r[cIdx*3+1]; m[o+2] = r[cIdx*3+2];
               }
               e.obj3d.matrixChanged();
               if (e.scene) ++e.scene.trans_stamp;
            }
            this.viewer.request_render();
         }
      }

      /** The server's time now, in ms, from EveManager's clock. */
      serverNow() {
         return this.viewer.get_manager().serverNow();
      }

      //--------------------------------------------------------------------

      /** Take the `motion` block of a transformation update. Null clears it.
        * `scene` is the EveScene of `obj3d`; moving the object raises its
        * trans_stamp. */
      update(id, obj3d, motion, scene) {
         if (!motion) { this.remove(id); return; }

         const m = obj3d._matrix ? obj3d._matrix.elements : null;
         if (!m) return;

         this.objects.set(id, {
            obj3d:  obj3d,
            scene:  scene,
            // Where the server says it was at t0 -- the matrix it sent in the
            // same message, before anything here has touched it. The rotation
            // is kept as the three basis columns, scale and all.
            p0:     [m[12], m[13], m[14]],
            r0:     [m[0], m[1], m[2],  m[4], m[5], m[6],  m[8], m[9], m[10]],
            t0:     motion.t0,
            vel:    motion.vel,
            acc:    motion.acc,
            axis:   motion.axis  || [0, 0, 1],
            rate:   motion.rate  || 0,
            max_dt: motion.max_dt
         });

         this.start();
      }

      remove(id) {
         this.objects.delete(id);
      }

      /** Move the trajectory of element `id` to `obj3d`, which takes over the
        * element's matrix, as when EveScene gives a leaf a group node. */
      retarget(id, obj3d) {
         const e = this.objects.get(id);
         if (e) e.obj3d = obj3d;
      }

      clear() {
         this.objects.clear();
         this.stop();
      }

      //--------------------------------------------------------------------

      start() {
         if (this.raf || this.objects.size === 0 || !this.enabled) return;
         this.raf = requestAnimationFrame(this.tick.bind(this));
      }

      stop() {
         if (this.raf) { cancelAnimationFrame(this.raf); this.raf = 0; }
         if (this._pending) { clearTimeout(this._pending); this._pending = 0; }
      }

      tick() {
         this.raf = 0;
         if (this.objects.size === 0) return;

         // Skip evaluation on a frame the render cap will not draw. The next
         // drawn frame evaluates for its own instant.
         const now = performance.now();
         const due = (this.render_max_hz === 0) ||
                     (now - this._last_render >= 1000 / this.render_max_hz);

         if (due && this.apply())
            this.requestRender();

         this.raf = requestAnimationFrame(this.tick.bind(this));
      }

      /** Place every moving object for the current instant.
        * Returns true if anything actually moved. */
      apply() {
         const now = this.serverNow();
         let moved = false;

         for (const e of this.objects.values()) {
            let dt = (now - e.t0) / 1000;         // seconds

            // Before t0 means the estimate is running ahead of the server;
            // hold at the start of the trajectory rather than run it backwards.
            if (dt < 0) dt = 0;

            // Past max_dt the object stops at the last point the server vouched
            // for, so a stalled link freezes it rather than letting it coast.
            if (dt > e.max_dt) dt = e.max_dt;

            const m = e.obj3d._matrix.elements;
            const x = e.p0[0] + e.vel[0] * dt + 0.5 * e.acc[0] * dt * dt;
            const y = e.p0[1] + e.vel[1] * dt + 0.5 * e.acc[1] * dt * dt;
            const z = e.p0[2] + e.vel[2] * dt + 0.5 * e.acc[2] * dt * dt;

            let touched = false;

            if (m[12] !== x || m[13] !== y || m[14] !== z) {
               m[12] = x; m[13] = y; m[14] = z;
               touched = true;
            }

            // Spin: turn the stored basis by rate*dt (rad/s times s) about the
            // axis, as one Rodrigues rotation, so there is no axis ordering. The
            // axis is in the local frame, so R composes on the right of the base
            // basis: each new column is a combination of the old columns.
            // REveTrans::SetMotion normalises the axis.
            if (e.rate !== 0) {
               const th = e.rate * dt;
               const ux = e.axis[0], uy = e.axis[1], uz = e.axis[2];
               const c = Math.cos(th), s = Math.sin(th), t = 1 - c;

               // Rodrigues, row-major.
               const R = [t*ux*ux + c,      t*ux*uy - s*uz,  t*ux*uz + s*uy,
                          t*ux*uy + s*uz,   t*uy*uy + c,     t*uy*uz - s*ux,
                          t*ux*uz - s*uy,   t*uy*uz + s*ux,  t*uz*uz + c];

               // new_col[j] = sum_k old_col[k] * R[k][j]. Column lengths survive
               // while the basis scale is uniform, which is what a local-frame
               // spin needs; a non-uniform scale would shear.
               const r = e.r0;
               for (let j = 0; j < 3; ++j) {
                  const o = j * 4;
                  for (let i = 0; i < 3; ++i)
                     m[o + i] = r[i] * R[j] + r[3 + i] * R[3 + j] + r[6 + i] * R[6 + j];
               }
               touched = true;
            }

            if (touched) {
               e.obj3d.matrixChanged();
               if (e.scene) ++e.scene.trans_stamp;
               moved = true;
            }
         }
         return moved;
      }
   }

   return Motion;
});
