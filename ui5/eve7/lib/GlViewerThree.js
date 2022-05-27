sap.ui.define([
   'rootui5/eve7/lib/GlViewer',
   'rootui5/eve7/lib/EveElements',
   'rootui5/eve7/lib/OutlinePassEve',
   'rootui5/eve7/lib/FXAAShader'
], function(GlViewer, EveElements) {

   "use strict";

   let g_global_init_done = false;

   class GlViewerThree extends GlViewer {

      g_highlight_update(mgr) {
         let sa = THREE.OutlinePassEve.selection_atts;
         let gs = mgr.GetElement(mgr.global_selection_id);
         let gh = mgr.GetElement(mgr.global_highlight_id);

         if (gs && gh) {
            sa[0].visibleEdgeColor.setStyle(EVE.JSR.getColor(gs.fVisibleEdgeColor));
            sa[0].hiddenEdgeColor.setStyle(EVE.JSR.getColor(gs.fHiddenEdgeColor));
            sa[1].visibleEdgeColor.setStyle(EVE.JSR.getColor(gh.fVisibleEdgeColor));
            sa[1].hiddenEdgeColor.setStyle(EVE.JSR.getColor(gh.fHiddenEdgeColor));
         }
      }

      init(controller) {
         super.init(controller);

         this.creator = new EveElements(controller);
         // MT-RCORE indices now work, we probably don't need this option anymore.
         // this.creator.useIndexAsIs = EVE.JSR.decodeUrl().has('useindx');
         this.creator.useIndexAsIs = true;

         if (!g_global_init_done) {
            g_global_init_done = true;

            this.controller.mgr.RegisterSelectionChangeFoo(this.g_highlight_update.bind(this));
            this.g_highlight_update(this.controller.mgr);

            ApplyThreeHacks();
         }

         this.createThreejsRenderer();
         this.controller.createScenes();
         this.controller.redrawScenes();
         this.setupThreejsDomAndEventHandlers();

         this.controller.glViewerInitDone();
      }

      cleanup() {
         if (this.controller) this.controller.removeScenes();
         this.destroyThreejsRenderer();
         super.cleanup();
      }

      //==============================================================================

      make_object(/*name*/) {
         return new THREE.Object3D;
      }

      get_top_scene() {
         return this.scene;
      }

      //==============================================================================
      // THREE renderer creation, DOM/event handler setup, reset
      //==============================================================================

      createThreejsRenderer() {
         let w = this.get_width(), h = this.get_height();

         // console.log("createThreejsRenderer", this.controller.kind, "w=", w, "h=", h);

         this.scene = new THREE.Scene();
         // this.scene.fog = new THREE.FogExp2( 0xaaaaaa, 0.05 );

         if (this.controller.isEveCameraPerspective())
            this.camera = new THREE.PerspectiveCamera(75, w / h, 1, 5000);
         else
            this.camera = new THREE.OrthographicCamera(-w / 2, w / 2, -h / 2, h / 2, 0, 2000);

         this.scene.add(this.camera);

         this.rot_center = new THREE.Vector3(0, 0, 0);

         this.renderer = new THREE.WebGLRenderer();
         this.renderer.setPixelRatio(window.devicePixelRatio);
         this.renderer.setSize(w, h);

         this.renderer.setClearColor(0xffffff, 1);

         // -------- Raycaster, lights, composer & FXAA and Outline passes.

         this.raycaster = new THREE.Raycaster();
         this.raycaster.params.Points.threshold = 4;   // ???
         this.raycaster.params.Line.threshold = 2.5;   // new three.js r121

         // Lights are positioned in resetRenderer

         this.point_lights = new THREE.Object3D;
         this.point_lights.add(new THREE.PointLight(0xffffff, 0.7)); // R
         this.point_lights.add(new THREE.PointLight(0xffffff, 0.7)); // G
         this.point_lights.add(new THREE.PointLight(0xffffff, 0.7)); // B
         this.scene.add(this.point_lights);

         // let plane = new THREE.GridHelper(20, 20, 0x80d080, 0x8080d0);
         // this.scene.add(plane);

         this.composer = new THREE.EffectComposer(this.renderer);
         this.composer.addPass(new THREE.RenderPass(this.scene, this.camera));

         this.outline_pass = new THREE.OutlinePassEve(new THREE.Vector2(w, h), this.scene, this.camera);
         this.outline_pass.edgeStrength = 5.5;
         this.outline_pass.edgeGlow = 0.7;
         this.outline_pass.edgeThickness = 1.5;
         this.outline_pass.usePatternTexture = false;
         this.outline_pass.downSampleRatio = 1;
         this.outline_pass.glowDownSampleRatio = 3;
         this.outline_pass.id2obj_map = {};


         // This does not work ... seems it is not standard pass?
         // this.outline_pass.renderToScreen = true;
         // Tried hacking with this, but would apparently need to load it somehow, sigh.
         // let copyPass = new ShaderPass( CopyShader );
         // this.composer.addPass( new THREE.ShaderPass(CopyShader) );

         this.composer.addPass(this.outline_pass);

         this.fxaa_pass = new THREE.ShaderPass(THREE.FXAAShader);
         this.fxaa_pass.uniforms.resolution.value.set(0.5 / w, 0.5 / h);
         this.fxaa_pass.renderToScreen = true;

         this.composer.addPass(this.fxaa_pass);
      }

      destroyThreejsRenderer() {
         if (this.renderer) {
            this.get_view().getDomRef().removeChild(this.renderer.domElement);
            this.renderer.domElement.removeEventListener('pointermove', this.mousemove_func);
            this.renderer.domElement.removeEventListener('pointerleave', this.mouseleave_func);
            this.renderer.domElement.removeEventListener('pointerdown', this.mousedown_func);
            this.renderer.domElement.removeEventListener('dblclick', this.dblclick_func);
            window.removeEventListener('keydown', this.keydown_func);
         }

         this.removeMouseMoveTimeout();
         delete this.renderer;
         delete this.scene;
         delete this.composer;
         if (this.controls) {
            this.controls.dispose();
            delete this.controls;
         }
      }

      mouseMoveHandler(event) {
         if (event.movementX == 0 && event.movementY == 0)
            return;

         if (event.buttons === 0) {
            this.removeMouseMoveTimeout();
            this.mousemove_timeout = setTimeout(this.onMouseMoveTimeout.bind(this, event.offsetX, event.offsetY), this.controller.htimeout);
         } else {
            this.clearHighlight();
         }
      }

      mouseLeaveHandler(/* event */) {
         this.removeMouseMoveTimeout();
         this.clearHighlight();
      }

      mouseDownHandler(event) {
         this.removeMouseMoveTimeout();
         if (event.buttons != 1 && event.buttons != 2)
            this.clearHighlight();
         else if (this.renderer) {
            // keep track which buttons and where are clicked
            this.click_event = event;
            this.click_buttons = event.buttons;
            this.click_intersect = this.getIntersectAt(event.offsetX, event.offsetY);
         }
      }

      clearClickedButtons() {
         delete this.click_buttons;
         delete this.click_intersect;
      }

      processControlEnd() {
         if (this.click_buttons == 1) {
            // handle left mouse button click
            if (this.click_intersect) {
               let c = this.click_intersect.object.get_ctrl();
               c.event = this.click_event;
               c.elementSelected(c.extractIndex(this.click_intersect));
               this.highlighted_scene = this.click_intersect.object.scene;
            } else {
               // XXXX HACK - handlersMIR senders should really be in the mgr

               this.controller.created_scenes[0].processElementSelected(null, [], this.click_event);
            }
         } else if (this.click_buttons == 2) {
            let intersect = this.click_intersect;
            EVE.JSR.createMenu(this.click_event, this).then(menu => this.showContextMenu(intersect, menu));
         }
      }

      dblClickHandler(/* event */) {
         if (this.controller.dblclick_action == "Reset")
            this.resetThreejsRenderer();
      }

      keyDownHandler(event) {
         let handled = true;

         if (event.key == "t") {
            this.scene.traverse(function(node) {

               if (node.material && node.material.linewidth) {
                  if (!node.material.linewidth_orig) node.material.linewidth_orig = node.material.linewidth;

                  node.material.linewidth *= 1.2;
               }
            });
         }
         else if (event.key == "e") {
            this.scene.traverse(function(node) {

               if (node.material && node.material.linewidth) {
                  if (!node.material.linewidth_orig) node.material.linewidth_orig = node.material.linewidth;

                  node.material.linewidth *= 0.8;
               }
            });
         }
         else if (event.key == "r") {
            this.scene.traverse(function(node) {

               if (node.material && node.material.linewidth && node.material.linewidth_orig) {
                  node.material.linewidth = node.material.linewidth_orig;
               }
            });
         }
         else {
            handled = false;
         }

         if (handled) {
            // // // event.stopPropagation();
            // event.preventDefault();
            // event.stopImmediatePropagation();

            this.render();
         }
      }

      setupThreejsDomAndEventHandlers() {
         this.get_view().getDomRef().appendChild(this.renderer.domElement);

         // Setup tooltip
         this.ttip = document.createElement('div');
         this.ttip.setAttribute('class', 'eve_tooltip');
         this.ttip_text = document.createElement('div');
         this.ttip.appendChild(this.ttip_text);
         this.get_view().getDomRef().appendChild(this.ttip);

         // Setup some event pre-handlers
         this.mousemove_func = this.mouseMoveHandler.bind(this);
         this.mouseleave_func = this.mouseLeaveHandler.bind(this);
         this.mousedown_func = this.mouseDownHandler.bind(this);
         this.dblclick_func = this.dblClickHandler.bind(this);
         this.keydown_func = this.keyDownHandler.bind(this);

         this.renderer.domElement.addEventListener('pointermove', this.mousemove_func);
         this.renderer.domElement.addEventListener('pointerleave', this.mouseleave_func);
         this.renderer.domElement.addEventListener('pointerdown', this.mousedown_func);
         this.renderer.domElement.addEventListener('dblclick', this.dblclick_func);

         // Key-handlers go on window ...
         window.addEventListener('keydown', this.keydown_func);

         // Setup controls
         this.controls = new THREE.OrbitControls(this.camera, this.get_view().getDomRef());
         this.controls.addEventListener('change', () => {
            this.clearClickedButtons();
            this.render();
         })
         this.controls.addEventListener('end', () => {
            this.processControlEnd();
            this.clearClickedButtons();
         });

         // This will also call render().
         this.resetThreejsRenderer();
      }

      /** Reset camera, lights based on scene bounding box. */
      resetThreejsRenderer() {
         let sbbox = new THREE.Box3();
         sbbox.setFromObject(this.scene);

         let posV = new THREE.Vector3; posV.subVectors(sbbox.max, this.rot_center);
         let negV = new THREE.Vector3; negV.subVectors(sbbox.min, this.rot_center);

         let extV = new THREE.Vector3; extV = negV; extV.negate(); extV.max(posV);
         let extR = extV.length();

         let lc = this.point_lights.children;
         lc[0].position.set(extR, extR, -extR);
         lc[1].position.set(-extR, extR, extR);
         lc[2].position.set(extR, extR, extR);

         let s = 1.02;
         if (this.camera.isPerspectiveCamera) {
            let posC = new THREE.Vector3(-s * extR, s * extR, -s * extR);

            this.camera.position.copy(posC);

            this.controls.screenSpacePanning = true;

            // console.log("resetThreejsRenderer 3D scene bbox ", sbbox, ", camera_pos ", posC, ", look_at ", this.rot_center);
         }
         else {
            let posC = new THREE.Vector3(0, 0, 1000);

            this.camera.position.copy(posC);
            let ex, ey;
            if (extV.x > extV.y) {
               ex = extV.x;
               ey = ex / this.get_width() * this.get_height();
               if (ey < extV.y)
                  s *= extV.y / ey;
            }
            else {
               ey = extV.y;
               ex = ey / this.get_height() * this.get_width();
               if (ex < extV.x)
                  s *= extV.x / ex;
            }

            ex *= s;
            ey *= s;
            this.camera.left = -ex;
            this.camera.right = ex;
            this.camera.top = ey;
            this.camera.bottom = -ey;

            if (typeof this.controls.resetOrthoPanZoom == 'function')
               this.controls.resetOrthoPanZoom();

            this.controls.screenSpacePanning = true;
            this.controls.enableRotate = false;

            // console.log("resetThreejsRenderer 2D scene bbox ex ey", sbbox, ex, ey, ", camera_pos ", posC, ", look_at ", this.rot_center);
         }
         this.controls.target.copy(this.rot_center);

         this.composer.reset();

         this.controls.update();
      }


      //==============================================================================

      render() {
         // AMT check if controller is attached in the splitter
         let v = this.get_manager().GetElement(this.controller.eveViewerId);
         if (!v.fRnrSelf)
            return;

         // Render through composer:
         this.composer.render(this.scene, this.camera);

         // or directly through renderer:
         // this.renderer.render( this.scene, this.camera );
      }

      //==============================================================================

      onResizeTimeout() {
         let w = this.get_width();
         let h = this.get_height();

         if (this.camera.isPerspectiveCamera) {
            this.camera.aspect = w / h;
         }
         else {
            this.camera.left = this.camera.bottom / h * w;
            this.camera.right = -this.camera.left;
            this.camera.updateProjectionMatrix();
         }
         this.camera.updateProjectionMatrix();

         this.renderer.setSize(w, h);
         this.outline_pass.setSize(w, h);
         this.fxaa_pass.uniforms.resolution.value.set(0.5 / w, 0.5 / h);

         this.composer.reset();
         this.controls.update();
         this.render();
      }


      //==============================================================================
      // THREE renderer event handlers etc.
      //==============================================================================

      //------------------------------------------------------------------------------
      // Highlight & Mouse move timeout handling
      //------------------------------------------------------------------------------

      clearHighlight() {
         if (this.highlighted_scene) {
            this.highlighted_scene.clearHighlight(); // XXXX should go through manager
            this.highlighted_scene = 0;

            this.ttip.style.display = "none";
         }
      }

      removeMouseMoveTimeout() {
         if (this.mousemove_timeout) {
            clearTimeout(this.mousemove_timeout);
            delete this.mousemove_timeout;
         }
      }

      /** Get three.js intersect object at specified mouse position */
      getIntersectAt(x, y) {
         let w = this.get_width();
         let h = this.get_height();

         // console.log("GLC::onMouseMoveTimeout", this, event, x, y);

         let mouse = new THREE.Vector2(((x + 0.5) / w) * 2 - 1, -((y + 0.5) / h) * 2 + 1);

         this.raycaster.setFromCamera(mouse, this.camera);

         let intersects = this.raycaster.intersectObjects(this.scene.children, true);
         for (let i = 0; i < intersects.length; ++i)
         {
            if (!intersects[i].object.get_ctrl)
               intersects[i].object = intersects[i].object.parent;

            if (intersects[i].object.visible && intersects[i].object.get_ctrl)
            {
               let ctrl = intersects[i].object.get_ctrl();
               if (ctrl && ctrl.obj3d && ctrl.obj3d.eve_el)
               {
                  let el = ctrl.obj3d.eve_el;
                  if (el && el.fPickable)
                  {
                     intersects[i].mouse = mouse;
                     intersects[i].w = w;
                     intersects[i].h = h;
                     return intersects[i];
                  }
               }
            }
         }
         return null;
      }

      onMouseMoveTimeout(x, y) {
         delete this.mousemove_timeout;

         let intersect = this.getIntersectAt(x, y);
         if (!intersect)
            return this.clearHighlight();

         let c = intersect.object.get_ctrl();

         let mouse = intersect.mouse;

         c.elementHighlighted(c.extractIndex(intersect));

         this.highlighted_scene = c.obj3d.scene;

         if (c.obj3d && c.obj3d.eve_el)
            this.ttip_text.innerHTML = c.getTooltipText(intersect);
         else
            this.ttip_text.innerHTML = "";

         let del = this.controller.getView().getDomRef();
         let offs = (mouse.x > 0 || mouse.y < 0) ? this.getRelativeOffsets(del) : null;

         if (mouse.x <= 0) {
            this.ttip.style.left = (x + del.offsetLeft + 10) + "px";
            this.ttip.style.right = null;
         } else {
            this.ttip.style.right = (intersect.w - x + offs.right + 10) + "px";
            this.ttip.style.left = null;
         }
         if (mouse.y >= 0) {
            this.ttip.style.top = (y + del.offsetTop + 10) + "px";
            this.ttip.style.bottom = null;
         } else {
            this.ttip.style.bottom = (intersect.h - y + offs.bottom + 10) + "px";
            this.ttip.style.top = null;
         }

         this.ttip.style.display = "block";
      }

      remoteToolTip(msg) {
         if (this.ttip_text && this.ttip_text.innerHTML)
            this.ttip_text.innerHTML = msg;
         if (this.highlighted_scene && this.ttip)
            this.ttip.style.display = "block";
      }

      getRelativeOffsets(elem) {
         // Based on:
         // https://stackoverflow.com/questions/3000887/need-to-calculate-offsetright-in-javascript

         let r = { left: 0, right: 0, top: 0, bottom: 0 };

         let parent = elem.offsetParent;

         while (parent && getComputedStyle(parent).position === 'relative') {
            r.top += elem.offsetTop;
            r.left += elem.offsetLeft;
            r.right += parent.offsetWidth - (elem.offsetLeft + elem.offsetWidth);
            r.bottom += parent.offsetHeight - (elem.offsetTop + elem.offsetHeight);

            elem = parent;
            parent = parent.offsetParent;
         }

         return r;
      }

      //------------------------------------------------------------------------------
      // Mouse button handlers, selection, context menu
      //------------------------------------------------------------------------------

      showContextMenu(intersect, menu) {
         // console.log("GLC::showContextMenu", this, menu)

         // See js/modules/gui/menu.mjs createMenu(), menu.add()

         menu.add("header:Context Menu");

         if (intersect) {
            let el = intersect.object.get_ctrl().obj3d.eve_el;
            menu.add("Browse to " + (el.fName || "element"), el.fElementId, this.controller.invokeBrowseOf.bind(this.controller));
         }

         menu.add("Reset camera", this.resetThreejsRenderer);

         menu.add("separator");

         let fff = this.defaultContextMenuAction;
         menu.add("sub:Sub Test");
         menu.add("Foo", 'foo', fff);
         menu.add("Bar", 'bar', fff);
         menu.add("Baz", 'baz', fff);
         menu.add("endsub:");

         menu.show();
      }

      defaultContextMenuAction(arg) {
         console.log("GLC::defaultContextMenuAction", this, arg);
      }

      handleMouseSelect(event) {
         let intersect = this.getIntersectAt(event.offsetX, event.offsetY);

         if (intersect) {
            let c = intersect.object.get_ctrl();
            c.event = event;
            c.elementSelected(c.extractIndex(intersect));
            this.highlighted_scene = intersect.object.scene;
         } else {
            // XXXX HACK - handlersMIR senders should really be in the mgr

            this.controller.created_scenes[0].processElementSelected(null, [], event);
         }
      }

   } // class GlViewerThree

   //==============================================================================
   // THREE.js hacks
   //==============================================================================

   function ApplyThreeHacks() {
      console.log("GlViewerThree.ApplyThreeHacks()");

      THREE.BufferGeometry.prototype.computeVertexNormalsIdxRange = function(start, count) {

         if (!this.attributes.position || !this.index) return;

         let index = this.index;
         let attributes = this.attributes;
         let positions = attributes.position.array;
         if (attributes.normal === undefined) {
            this.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(positions.length), 3));
         } else {
            // reset existing normals to zero
            let array = attributes.normal.array;
            for (let i = 0, il = array.length; i < il; i++) {
               array[i] = 0;
            }
         }
         let normals = attributes.normal.array;

         let vA, vB, vC;
         let pA = new THREE.Vector3(), pB = new THREE.Vector3(), pC = new THREE.Vector3();
         let cb = new THREE.Vector3(), ab = new THREE.Vector3();

         let indices = index.array;

         for (let i = start, i_end = start + count; i < i_end; i += 3) {

            vA = indices[i + 0] * 3;
            vB = indices[i + 1] * 3;
            vC = indices[i + 2] * 3;

            pA.fromArray(positions, vA);
            pB.fromArray(positions, vB);
            pC.fromArray(positions, vC);

            cb.subVectors(pC, pB);
            ab.subVectors(pA, pB);
            cb.cross(ab);

            normals[vA] += cb.x;
            normals[vA + 1] += cb.y;
            normals[vA + 2] += cb.z;

            normals[vB] += cb.x;
            normals[vB + 1] += cb.y;
            normals[vB + 2] += cb.z;

            normals[vC] += cb.x;
            normals[vC + 1] += cb.y;
            normals[vC + 2] += cb.z;

         }
         this.normalizeNormals();

         attributes.normal.needsUpdate = true;

      };


   }

   //==============================================================================

   return GlViewerThree;

});
