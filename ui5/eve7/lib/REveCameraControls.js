/** Based on three.js OrbitControls, extended for ROOT REve.
 * Copyright (c) 2010-2024 three.js authors
 * MIT Licensed
 */

// This set of controls performs orbiting, dollying (zooming), and panning.
// Unlike TrackballControls, it maintains the "up" direction object.up (+Y by default).
//
//    Orbit - left mouse / touch: one-finger move
//    Zoom - middle mouse, or mousewheel / touch: two-finger spread or squish
//    Pan - right mouse, or left mouse + ctrl/meta/shiftKey, or arrow keys / touch: two-finger move

// Requires the following to be set on Camera passed.
//   this.camera.isPerspectiveCamera = true;
// or
//   this.camera.isOrthographicCamera = true;

// Math/core primitives come through RenderCore.js, so this works with both
// the bundled and the dev-mode RenderCore.
import { EventDispatcher, Vector3, Vector2, Spherical, Matrix4 } from './RenderCore.js';

const MOUSE = { ROTATE: 0, DOLLY: 1, PAN: 2 };
const TOUCH = { ROTATE: 0, PAN: 1, DOLLY_PAN: 2, DOLLY_ROTATE: 3 };

const STATE = {
   NONE: -1,
   ROTATE: 0,
   DOLLY: 1,
   PAN: 2,
   TOUCH_ROTATE: 3,
   TOUCH_PAN: 4,
   TOUCH_DOLLY_PAN: 5,
   TOUCH_DOLLY_ROTATE: 6
};

const EPS = 0.000001;

const changeEvent = { type: 'change' };
const startEvent = { type: 'start' };
const endEvent = { type: 'end' };

// Matrix4 helpers for ROOT GL-style camera matrices; base vectors are 1-based columns.
Object.assign(Matrix4.prototype, {
   lookAtMt(eye, fwd, up) {
      const _x = new Vector3(), _y = new Vector3(), _z = new Vector3();
      const te = this.elements;

      _z.copy(fwd);
      // _z.negate();
      _z.normalize();
      _x.crossVectors(up, _z);
      _x.normalize();
      _y.crossVectors(_z, _x);

      te[0] = _x.x; te[4] = _y.x; te[8] = _z.x;
      te[1] = _x.y; te[5] = _y.y; te[9] = _z.y;
      te[2] = _x.z; te[6] = _y.z; te[10] = _z.z;

      return this;
   },

   getBaseVector(idx) {
      const off = 4 * (idx - 1), C = this.elements;
      return new Vector3(C[off], C[off + 1], C[off + 2]);
   },

   getTranslation() {
      const C = this.elements;
      return new Vector3(C[12], C[13], C[14]);
   },

   setBaseVector(idx, x, y, z) {
      const off = 4 * (idx - 1), C = this.elements;
      if (x.isVector3) {
         C[off] = x.x; C[off + 1] = x.y; C[off + 2] = x.z;
      } else {
         C[off] = x; C[off + 1] = y; C[off + 2] = z;
      }
   },

   rotateIP(v) {
      const M = this.elements, r = v.clone();
      v.x = M[0] * r.x + M[4] * r.y + M[8] * r.z;
      v.y = M[1] * r.x + M[5] * r.y + M[9] * r.z;
      v.z = M[2] * r.x + M[6] * r.y + M[10] * r.z;
   },

   rotateLF(i1, i2, amount) {
      if (i1 == i2) return;

      const cos = Math.cos(amount), sin = Math.sin(amount), C = this.elements;
      i1 = (i1 - 1) * 4;
      i2 = (i2 - 1) * 4; // column major
      for (let off = 0; off < 4; ++off) {
         const b1 = cos * C[i1 + off] + sin * C[i2 + off];
         const b2 = cos * C[i2 + off] - sin * C[i1 + off];
         C[i1 + off] = b1; C[i2 + off] = b2;
      }
   },

   rotatePF(i1, i2, amount) {
      if (i1 == i2) return;

      const cos = Math.cos(amount), sin = Math.sin(amount), C = this.elements;
      --i1; --i2;
      for (let c = 0; c < 4; ++c) {
         const off = c * 4;
         const b1 = cos * C[i1 + off] - sin * C[i2 + off];
         const b2 = cos * C[i2 + off] + sin * C[i1 + off];
         C[i1 + off] = b1; C[i2 + off] = b2;
      }
   },

   moveLF(ai, amount) {
      const C = this.elements, off = 4 * (ai - 1);
      C[12] += amount * C[off];
      C[13] += amount * C[off + 1];
      C[14] += amount * C[off + 2];
   },

   // debugging only, see the commented-out call in GlViewerRCore.js
   dump() {
      for (let x = 0; x < 4; x++) {
         let row = '[ ';
         for (let y = 0; y < 4; y++)
            row += this.elements[y * 4 + x].toFixed(2) + ' ';
         console.log(row + ']');
      }
   }
});

// Midpoint of the first one or two touches, or the single touch itself.
function touchCenter(event) {
   const t = event.touches;
   if (t.length == 1)
      return [t[0].pageX, t[0].pageY];
   return [0.5 * (t[0].pageX + t[1].pageX), 0.5 * (t[0].pageY + t[1].pageY)];
}

function touchDistance(event) {
   const t = event.touches;
   return Math.hypot(t[0].pageX - t[1].pageX, t[0].pageY - t[1].pageY);
}

export class REveCameraControls extends EventDispatcher {

   // Set to false to disable this control
   enabled = true;

   // Unused, left from OrbitControls: orbiting pivots around cameraCenter.
   // "target" sets the location of focus, where the object orbits around
   // target = new Vector3();
   cameraCenter = new Vector3();

   // Set to true to have setFromBBox() also re-point the orbit-rotation
   // pivot (see setCameraCenter()) at the framed bbox's center. Off by
   // default: the pivot stays wherever it was (the origin, until something
   // moves it), so content far from the origin swings out of frame on rotation.
   // GlViewerRCore sets it from REveViewer::SetCameraCenter() on camera reset.
   centerCameraOnBBox = false;

   // Unused, left from OrbitControls.
   // How far you can dolly in and out ( PerspectiveCamera only )
   // minDistance = 0;
   // maxDistance = Infinity;

   // How far you can zoom in and out ( OrthographicCamera only )
   minZoom = 0;
   maxZoom = Infinity;

   // Unused, left from OrbitControls: rotateRad() has its own fixed up-vector lock.
   // How far you can orbit vertically, upper and lower limits.
   // Range is 0 to Math.PI radians.
   // minPolarAngle = 0; // radians
   // maxPolarAngle = Math.PI; // radians

   // How far you can orbit horizontally, upper and lower limits.
   // If set, must be a sub-interval of the interval [ - Math.PI, Math.PI ].
   // minAzimuthAngle = -Infinity; // radians
   // maxAzimuthAngle = Infinity; // radians

   // Unused, left from OrbitControls.
   // Set to true to enable damping (inertia)
   // If damping is enabled, you must call controls.update() in your animation loop
   // enableDamping = false;
   // dampingFactor = 0.05;

   // This option actually enables dollying in and out; left as "zoom" for backwards compatibility.
   // Set to false to disable zooming
   enableZoom = true;
   zoomSpeed = 1.0;

   // Set to false to disable rotating
   enableRotate = true;
   rotateSpeed = 1.0;

   // Set to false to disable panning
   enablePan = true;
   panSpeed = 1.0;
   // Unused: panning is always in screen space. GlViewerRCore still sets it.
   // screenSpacePanning = false; // if true, pan in screen-space
   keyPanSpeed = 7.0; // pixels moved per arrow key push

   // Unused, left from OrbitControls.
   // Set to true to automatically rotate around the target
   // If auto-rotate is enabled, you must call controls.update() in your animation loop
   // autoRotate = false;
   // autoRotateSpeed = 2.0; // 30 seconds per round when fps is 60

   // Set to false to disable use of the keys
   enableKeys = true;

   // The four arrow keys
   keys = { LEFT: 37, UP: 38, RIGHT: 39, BOTTOM: 40 };

   // Mouse buttons
   mouseButtons = { LEFT: MOUSE.ROTATE, MIDDLE: MOUSE.DOLLY, RIGHT: MOUSE.PAN };

   // Touch fingers
   touches = { ONE: TOUCH.ROTATE, TWO: TOUCH.DOLLY_PAN };

   #state = STATE.NONE;

   // Unused: never set, so getPolarAngle()/getAzimuthalAngle() would always return 0.
   // current position in spherical coordinates
   // #spherical = new Spherical();

   // last rotation step, only used to decide whether update() fires 'change'
   #sphericalDelta = new Spherical();

   // camera matrix is camBase * camTrans
   #camBase = new Matrix4();
   #camTrans = new Matrix4();

   #scale = 1;
   #panOffset = new Vector3();
   #zoomChanged = false;
   #lastPosition = new Vector3();

   #rotateStart = new Vector2();
   #rotateEnd = new Vector2();
   #rotateDelta = new Vector2();

   #panStart = new Vector2();
   #panEnd = new Vector2();
   #panDelta = new Vector2();

   #dollyStart = new Vector2();
   #dollyEnd = new Vector2();
   #dollyDelta = new Vector2();

   constructor(object, domElement) {
      super();

      this.object = object;
      this.domElement = domElement ?? document;

      this.domElement.addEventListener('contextmenu', this.#onContextMenu, false);

      this.domElement.addEventListener('mousedown', this.#onMouseDown, false);
      this.domElement.addEventListener('wheel', this.#onMouseWheel, false);

      this.domElement.addEventListener('touchstart', this.#onTouchStart, false);
      this.domElement.addEventListener('touchend', this.#onTouchEnd, false);
      this.domElement.addEventListener('touchmove', this.#onTouchMove, false);

      this.domElement.addEventListener('mouseenter', this.#onMouseEnter);
      this.domElement.addEventListener('mouseleave', this.#onMouseLeave);

      // force an update at start
      this.update();
   }

   //
   // public methods
   //

   setCamBaseMtx(hAxis, vAxis) {
      const camBase = this.#camBase;
      camBase.identity();

      camBase.setBaseVector(1, hAxis);
      camBase.setBaseVector(3, vAxis);

      const y = new Vector3();
      y.crossVectors(vAxis, hAxis);
      camBase.setBaseVector(2, y);
   }

   setFromBBox(bbox) {
      const bb_center = new Vector3();
      bbox.getCenter(bb_center);

      // Half the box's own diagonal, independent of where the box sits.
      const bb_size = new Vector3();
      bbox.getSize(bb_size);
      const bb_R = 0.5 * bb_size.length();

      // The camera matrix is camBase * camTrans (see update()), so camTrans's
      // translation is in camBase's local frame -- pull bb_center back through
      // camBase's inverse, as setCameraCenter() does.
      const camBaseInv = new Matrix4();
      camBaseInv.getInverse(this.#camBase);
      const localCenter = bb_center.clone().applyMatrix4(camBaseInv);

      const camTrans = this.#camTrans;
      camTrans.identity();
      camTrans.setPosition(localCenter);

      if (this.object.isPerspectiveCamera) {
         const fovDefault = 30;
         const fov = Math.min(fovDefault, this.object.aspect * fovDefault);
         const dollyDefault = bb_R / (2.0 * Math.tan(fov * Math.PI * 0.8 / 180));

         camTrans.moveLF(1, dollyDefault);
      } else {
         const dollyDefault = 1.25 * 0.5 * Math.sqrt(3) * bb_R;
         camTrans.moveLF(1, dollyDefault);
         this.object._near = 0.05 * dollyDefault;
         this.object._far = 2 * dollyDefault;
         this.object.updateProjectionMatrix();
      }

      // Orbit-rotation pivots around cameraCenter, which setCamBaseMtx() resets
      // to the origin. Re-pointing it at bb_center keeps the current view but
      // changes how it rotates, hence opt-in.
      if (this.centerCameraOnBBox)
         this.setCameraCenter(bb_center.x, bb_center.y, bb_center.z);
   }

   getCamTrans() {
      return this.#camTrans;
   }

   setCamTrans(elements) {
      this.#camTrans.elements = elements;
   }

   getCamBase() {
      return this.#camBase;
   }

   setCameraCenter(x, y, z) {
      const camBase = this.#camBase, camTrans = this.#camTrans;
      this.cameraCenter.set(x, y, z);

      const bt = new Matrix4();
      bt.multiplyMatrices(camBase, camTrans);
      camBase.setBaseVector(4, this.cameraCenter);
      const binv = camBase.clone();
      binv.getInverse(camBase);
      camTrans.multiplyMatrices(binv, bt);
   }

   // Unused, see #spherical.
   // getPolarAngle() {
   //    return this.#spherical.phi;
   // }
   //
   // getAzimuthalAngle() {
   //    return this.#spherical.theta;
   // }

   // osschar - need to reset internal panOffset
   resetOrthoPanZoom() {
      this.#panOffset.set(0, 0, 0);
      this.object.zoom = 0.78; // AMT default ortho camera zoom value in ROOT GL
      this.object.updateProjectionMatrix();
      this.#zoomChanged = true;
   }

   // Unused: nothing calls these.
   // osschar - fake mouse up event, needed for context menu on M3 down
   // resetMouseDown(event) {
   //    this.#onMouseUp(event);
   // }
   //
   // testCameraMenu() {
   //    this.#dollyIn(0.5);
   // }

   // alja - set camera matrix from two operations camBase and camTrans
   update() {
      const camTrans = this.#camTrans, obj = this.object;

      // dolly scale
      if (this.#scale != 1.0) {
         const b1 = camTrans.getBaseVector(1);
         const b4 = camTrans.getBaseVector(4);
         const lookAtDist = Math.sqrt(b1.dot(b4));
         b4.multiplyScalar(this.#scale);
         camTrans.setBaseVector(4, b4);
         this.#scale = 1.0;

         obj.near = Math.min(lookAtDist * 0.1, 20);
      }

      // pan/ truck
      if (this.#panOffset.x || this.#panOffset.y) {
         camTrans.moveLF(2, this.#panOffset.x);
         camTrans.moveLF(3, this.#panOffset.y);
         this.#panOffset.set(0, 0, 0);
      }

      const cam = new Matrix4();
      cam.multiplyMatrices(this.#camBase, camTrans);

      // matrix needed to transform position from the picking
      obj.testMtx = cam;

      const eye = new Vector3(); eye.setFromMatrixPosition(cam);
      const fwd = new Vector3(); fwd.setFromMatrixColumn(cam, 0);
      const up = new Vector3(); up.setFromMatrixColumn(cam, 2);

      obj._matrix.lookAtMt(eye, fwd, up);
      obj._matrix.setPosition(eye);

      // camera matrix auto update is disabled to prevent reading from quaternions
      obj.matrixWorld.copy(obj.matrix);

      // update condition is:
      // min(camera displacement, camera rotation in radians)^2 > EPS
      // using small-angle approximation
      const sd = this.#sphericalDelta;
      if (this.#zoomChanged ||
          this.#lastPosition.distanceToSquared(cam.getBaseVector(4)) > EPS ||
          10 * (sd.phi + sd.theta) > EPS) {

         this.dispatchEvent(changeEvent);

         this.#lastPosition.copy(cam.getBaseVector(4));
         this.#zoomChanged = false;

         sd.theta = 0;
         sd.phi = 0;
      }

      return true;
   }

   dispose() {
      this.domElement.removeEventListener('contextmenu', this.#onContextMenu, false);
      this.domElement.removeEventListener('mousedown', this.#onMouseDown, false);
      this.domElement.removeEventListener('wheel', this.#onMouseWheel, false);

      this.domElement.removeEventListener('touchstart', this.#onTouchStart, false);
      this.domElement.removeEventListener('touchend', this.#onTouchEnd, false);
      this.domElement.removeEventListener('touchmove', this.#onTouchMove, false);

      this.domElement.removeEventListener('mouseenter', this.#onMouseEnter);
      this.domElement.removeEventListener('mouseleave', this.#onMouseLeave);

      document.removeEventListener('mousemove', this.#onMouseMove, false);
      document.removeEventListener('mouseup', this.#onMouseUp, false);

      window.removeEventListener('keydown', this.#onKeyDown, false);

      //this.dispatchEvent( { type: 'dispose' } ); // should this be added here?
   }

   //
   // internals
   //

   get #element() {
      return this.domElement === document ? this.domElement.body : this.domElement;
   }

   // Mouse Control / Shift scaling factor, set on MouseButtonDown, reset on MouseButtonUp.
   // Ctrl -> 0.1, Ctrl-Shift -> 0.01, Shift -> 10.0
   #mouseCSScale(event) {
      if (event.ctrlKey)
         return event.shiftKey ? 0.01 : 0.1;
      if (event.shiftKey)
         return 10.0; // was also stored in this.MouseCSScaleFactor, never read
      return 1.0;
   }

   #dollyCSScale(event) {
      if (event.ctrlKey)
         return event.shiftKey ? 0.01 : 0.1;
      if (event.shiftKey)
         return 5.0; // was also stored in this.MouseCSScaleFactor, never read
      return 1.0;
   }

   #getZoomScale(fac) {
      return Math.pow(0.978, fac * this.zoomSpeed);
   }

   #rotateRad(hRotate, vRotate) {
      const fVAxisMinAngle = 0.01;
      const camTrans = this.#camTrans;

      if (hRotate != 0.0) {
         const fwd = camTrans.getBaseVector(1);
         const up = camTrans.getBaseVector(3);
         const pos = camTrans.getTranslation();

         const deltaF = pos.dot(fwd);
         const deltaU = pos.dot(up);

         // up vector lock
         const zdir = this.#camBase.getBaseVector(3);
         this.#camBase.rotateIP(fwd);
         const theta = Math.acos(fwd.dot(zdir));
         if (theta + hRotate < fVAxisMinAngle)
            hRotate = fVAxisMinAngle - theta;
         else if (theta + hRotate > Math.PI - fVAxisMinAngle)
            hRotate = Math.PI - fVAxisMinAngle - theta;

         camTrans.moveLF(1, -deltaF);
         camTrans.moveLF(3, -deltaU);
         camTrans.rotateLF(3, 1, hRotate);
         camTrans.moveLF(3, deltaU);
         camTrans.moveLF(1, deltaF);
      }
      if (vRotate != 0.0)
         camTrans.rotatePF(1, 2, -vRotate);

      this.#sphericalDelta.phi = hRotate;
      this.#sphericalDelta.theta = vRotate;
   }

   #panFwdBkwStepPersp(element) {
      // Same for Pan and FwdBkw
      const obj = this.object;
      const targetDistance = 0.5 * (obj.far + obj.near) * Math.tan(0.5 * obj.fov * Math.PI / 180.0);
      return targetDistance / element.clientHeight;
   }

   #panStepOrtho(element) {
      const obj = this.object;
      return [ (obj.right - obj.left) / obj.zoom / element.clientWidth,
               (obj.top - obj.bottom) / obj.zoom / element.clientHeight ];
   }

   #fwdBkwStepOrtho(element) {
      // Average of dx/dy
      const sxy = this.#panStepOrtho(element);
      return 0.5 * (sxy[0] + sxy[1]);
   }

   // deltaX and deltaY are in pixels; right and down are positive
   #pan(deltaX, deltaY) {
      // amt, x seem to be negatd
      deltaX = -deltaX;

      const element = this.#element;

      if (this.object.isPerspectiveCamera) {
         const step = this.#panFwdBkwStepPersp(element);
         // We use only clientHeight for both here so aspect ratio does not distort speed.
         this.#panOffset.setX(deltaX * step);
         this.#panOffset.setY(deltaY * step);
      } else if (this.object.isOrthographicCamera) {
         const step = this.#panStepOrtho(element);
         this.#panOffset.setX(deltaX * step[0]);
         this.#panOffset.setY(deltaY * step[1]);
      } else {
         // camera neither orthographic nor perspective
         console.warn('WARNING: REveCameraControls encountered an unknown camera type - pan disabled.');
         this.enablePan = false;
      }
   }

   #fwdBkw(delta) {
      const element = this.#element;
      if (this.object.isPerspectiveCamera) {
         this.#camTrans.moveLF(1, -delta * this.#panFwdBkwStepPersp(element));
      } else if (this.object.isOrthographicCamera) {
         this.#camTrans.moveLF(1, -delta * this.#fwdBkwStepOrtho(element));
      } else {
         // camera neither orthographic nor perspective
         console.warn('WARNING: REveCameraControls encountered an unknown camera type - fwd-bkw disabled.');
         this.enableZoom = false;
      }
   }

   #dollyIn(dollyScale) {
      const obj = this.object;
      if (obj.isPerspectiveCamera) {
         this.#scale /= dollyScale;
      } else if (obj.isOrthographicCamera) {
         obj.zoom = Math.max(this.minZoom, Math.min(this.maxZoom, obj.zoom * dollyScale));
         obj.updateProjectionMatrix();
         this.#zoomChanged = true;
      } else {
         console.warn('WARNING: REveCameraControls encountered an unknown camera type - dolly/zoom disabled.');
         this.enableZoom = false;
      }
      this.update();
   }

   #dollyOut(dollyScale) {
      const obj = this.object;
      if (obj.isPerspectiveCamera) {
         this.#scale *= dollyScale;
      } else if (obj.isOrthographicCamera) {
         obj.zoom = Math.max(this.minZoom, Math.min(this.maxZoom, obj.zoom / dollyScale));
         obj.updateProjectionMatrix();
         this.#zoomChanged = true;
      } else {
         console.warn('WARNING: REveCameraControls encountered an unknown camera type - dolly/zoom disabled.');
         this.enableZoom = false;
      }
      this.update();
   }

   //
   // event callbacks - update the object state
   //

   #handleMouseMoveRotate(event) {
      this.#rotateEnd.set(event.clientX, event.clientY);

      this.#rotateDelta.subVectors(this.#rotateEnd, this.#rotateStart).multiplyScalar(this.rotateSpeed);
      this.#rotateDelta.multiplyScalar(this.#mouseCSScale(event));
      const element = this.#element;

      this.#rotateRad(-2 * Math.PI * this.#rotateDelta.y / element.clientHeight,
                      2 * Math.PI * this.#rotateDelta.x / element.clientWidth);

      this.#rotateStart.copy(this.#rotateEnd);

      this.update();
   }

   #handleMouseMoveDolly(event) {
      this.#dollyEnd.set(event.clientX, event.clientY);
      this.#dollyDelta.subVectors(this.#dollyEnd, this.#dollyStart);

      if (this.object.isPerspectiveCamera) {
         this.#fwdBkw(this.#dollyDelta.y * this.#mouseCSScale(event));
      } else if (this.#dollyDelta.y < 0) {
         this.#dollyIn(this.#getZoomScale(this.#dollyCSScale(event)));
      } else if (this.#dollyDelta.y > 0) {
         this.#dollyOut(this.#getZoomScale(this.#dollyCSScale(event)));
      }
      this.#dollyStart.copy(this.#dollyEnd);
      this.update();
   }

   #handleMouseMovePan(event) {
      this.#panEnd.set(event.clientX, event.clientY);
      this.#panDelta.subVectors(this.#panEnd, this.#panStart).multiplyScalar(this.panSpeed);
      this.#panDelta.multiplyScalar(this.#mouseCSScale(event));

      this.#pan(this.#panDelta.x, this.#panDelta.y);

      this.#panStart.copy(this.#panEnd);
      this.update();
   }

   #handleMouseWheel(event) {
      if (this.object.isPerspectiveCamera) {
         let step;
         if (event.deltaMode == 0) {
            step = event.deltaY;
         } else { // 1 is lines, 2 pages -- we don't care, just take 1/50 of height
            step = event.deltaY * 0.05 * this.#element.clientHeight;
         }
         this.#fwdBkw(0.5 * this.#mouseCSScale(event) * step);
      } else if (event.deltaY > 0) {
         this.#dollyOut(this.#getZoomScale(this.#dollyCSScale(event)));
      } else if (event.deltaY < 0) {
         this.#dollyIn(this.#getZoomScale(this.#dollyCSScale(event)));
      }
      this.update();
   }

   #handleKeyDown(event) {
      const step = this.keyPanSpeed * this.#mouseCSScale(event);

      switch (event.keyCode) {
         case this.keys.UP:     this.#pan(0, step); break;
         case this.keys.BOTTOM: this.#pan(0, -step); break;
         case this.keys.LEFT:   this.#pan(step, 0); break;
         case this.keys.RIGHT:  this.#pan(-step, 0); break;
         default: return;
      }

      // prevent the browser from scrolling on cursor keys
      event.preventDefault();
      // and prevent others from consuming the event
      event.stopImmediatePropagation();
      this.dispatchEvent(endEvent);
      this.update();
   }

   #handleTouchStartDolly(event) {
      this.#dollyStart.set(0, touchDistance(event));
   }

   #handleTouchMoveRotate(event) {
      this.#rotateEnd.set(...touchCenter(event));

      this.#rotateDelta.subVectors(this.#rotateEnd, this.#rotateStart).multiplyScalar(this.rotateSpeed);

      const element = this.#element;

      this.#rotateRad(-2 * Math.PI * this.#rotateDelta.y / element.clientHeight,
                      2 * Math.PI * this.#rotateDelta.x / element.clientWidth);

      this.#rotateStart.copy(this.#rotateEnd);
   }

   #handleTouchMovePan(event) {
      this.#panEnd.set(...touchCenter(event));

      this.#panDelta.subVectors(this.#panEnd, this.#panStart).multiplyScalar(this.panSpeed);

      this.#pan(this.#panDelta.x, this.#panDelta.y);

      this.#panStart.copy(this.#panEnd);
   }

   #handleTouchMoveDolly(event) {
      this.#dollyEnd.set(0, touchDistance(event));
      this.#dollyDelta.set(0, Math.pow(this.#dollyEnd.y / this.#dollyStart.y, this.zoomSpeed));
      this.#dollyIn(this.#dollyDelta.y);
      this.#dollyStart.copy(this.#dollyEnd);
   }

   //
   // event handlers - FSM: listen for events and reset state
   // (arrow-function fields, so they can be passed to add/removeEventListener as is)
   //

   #onMouseDown = (event) => {
      if (this.enabled === false) return;

      // Prevent the browser from scrolling.
      event.preventDefault();

      // Manually set the focus since calling preventDefault above
      // prevents the browser from setting it automatically.
      this.domElement.focus ? this.domElement.focus() : window.focus();

      const action = [this.mouseButtons.LEFT, this.mouseButtons.MIDDLE, this.mouseButtons.RIGHT][event.button];
      if (action === undefined) return;

      switch (action) {
         case MOUSE.ROTATE:
            if (this.enableRotate === false) return;
            this.#rotateStart.set(event.clientX, event.clientY);
            this.#state = STATE.ROTATE;
            break;

         case MOUSE.DOLLY:
            if (this.enableZoom === false) return;
            this.#dollyStart.set(event.clientX, event.clientY);
            this.#state = STATE.DOLLY;
            break;

         case MOUSE.PAN:
            if (this.enablePan === false) return;
            this.#panStart.set(event.clientX, event.clientY);
            this.#state = STATE.PAN;
            break;

         default:
            this.#state = STATE.NONE;
      }

      if (this.#state !== STATE.NONE) {
         document.addEventListener('mousemove', this.#onMouseMove, false);
         document.addEventListener('mouseup', this.#onMouseUp, false);

         this.dispatchEvent(startEvent);
      }
   };

   #onMouseMove = (event) => {
      if (this.enabled === false) return;

      event.preventDefault();

      switch (this.#state) {
         case STATE.ROTATE:
            if (this.enableRotate) this.#handleMouseMoveRotate(event);
            break;

         case STATE.DOLLY:
            if (this.enableZoom) this.#handleMouseMoveDolly(event);
            break;

         case STATE.PAN:
            if (this.enablePan) this.#handleMouseMovePan(event);
            break;
      }
   };

   #onMouseUp = (/*event*/) => {
      if (this.enabled === false) return;

      document.removeEventListener('mousemove', this.#onMouseMove, false);
      document.removeEventListener('mouseup', this.#onMouseUp, false);

      this.dispatchEvent(endEvent);

      this.#state = STATE.NONE;
   };

   #onMouseWheel = (event) => {
      if (this.enabled === false || this.enableZoom === false ||
          (this.#state !== STATE.NONE && this.#state !== STATE.ROTATE)) return;

      event.preventDefault();
      event.stopPropagation();

      this.dispatchEvent(startEvent);

      this.#handleMouseWheel(event);

      this.dispatchEvent(endEvent);
   };

   #onKeyDown = (event) => {
      if (this.enabled === false || this.enableKeys === false || this.enablePan === false) return;

      this.#handleKeyDown(event);
   };

   #onMouseEnter = () => {
      window.addEventListener('keydown', this.#onKeyDown);
   };

   #onMouseLeave = () => {
      window.removeEventListener('keydown', this.#onKeyDown);
   };

   #onTouchStart = (event) => {
      if (this.enabled === false) return;

      event.preventDefault();

      switch (event.touches.length) {
         case 1:
            switch (this.touches.ONE) {
               case TOUCH.ROTATE:
                  if (this.enableRotate === false) return;
                  this.#rotateStart.set(...touchCenter(event));
                  this.#state = STATE.TOUCH_ROTATE;
                  break;

               case TOUCH.PAN:
                  if (this.enablePan === false) return;
                  this.#panStart.set(...touchCenter(event));
                  this.#state = STATE.TOUCH_PAN;
                  break;

               default:
                  this.#state = STATE.NONE;
            }
            break;

         case 2:
            switch (this.touches.TWO) {
               case TOUCH.DOLLY_PAN:
                  if (this.enableZoom === false && this.enablePan === false) return;
                  if (this.enableZoom) this.#handleTouchStartDolly(event);
                  if (this.enablePan) this.#panStart.set(...touchCenter(event));
                  this.#state = STATE.TOUCH_DOLLY_PAN;
                  break;

               case TOUCH.DOLLY_ROTATE:
                  if (this.enableZoom === false && this.enableRotate === false) return;
                  if (this.enableZoom) this.#handleTouchStartDolly(event);
                  if (this.enableRotate) this.#rotateStart.set(...touchCenter(event));
                  this.#state = STATE.TOUCH_DOLLY_ROTATE;
                  break;

               default:
                  this.#state = STATE.NONE;
            }
            break;

         default:
            this.#state = STATE.NONE;
      }

      if (this.#state !== STATE.NONE)
         this.dispatchEvent(startEvent);
   };

   #onTouchMove = (event) => {
      if (this.enabled === false) return;

      event.preventDefault();
      event.stopPropagation();

      switch (this.#state) {
         case STATE.TOUCH_ROTATE:
            if (this.enableRotate === false) return;
            this.#handleTouchMoveRotate(event);
            break;

         case STATE.TOUCH_PAN:
            if (this.enablePan === false) return;
            this.#handleTouchMovePan(event);
            break;

         case STATE.TOUCH_DOLLY_PAN:
            if (this.enableZoom === false && this.enablePan === false) return;
            if (this.enableZoom) this.#handleTouchMoveDolly(event);
            if (this.enablePan) this.#handleTouchMovePan(event);
            break;

         case STATE.TOUCH_DOLLY_ROTATE:
            if (this.enableZoom === false && this.enableRotate === false) return;
            if (this.enableZoom) this.#handleTouchMoveDolly(event);
            if (this.enableRotate) this.#handleTouchMoveRotate(event);
            break;

         default:
            this.#state = STATE.NONE;
            return;
      }

      this.update();
   };

   #onTouchEnd = (/*event*/) => {
      if (this.enabled === false) return;

      this.dispatchEvent(endEvent);

      this.#state = STATE.NONE;
   };

   #onContextMenu = (event) => {
      if (this.enabled === false) return;

      event.preventDefault();
   };
}
