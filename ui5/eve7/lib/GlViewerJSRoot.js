sap.ui.define([
   'rootui5/eve7/lib/GlViewer',
   'rootui5/eve7/lib/EveElements',
   'rootui5/eve7/lib/OutlinePassEve',
   'rootui5/eve7/lib/FXAAShader'
], function(GlViewer, EveElements) {

   "use strict";

   class GlViewerJSRoot extends GlViewer {

      init(controller)
      {
         super.init(controller);

         this.creator = new EveElements(controller);
         this.creator.useIndexAsIs = EVE.JSR.decodeUrl().has('useindx');

         let msg = this.getGeomDescription();


         this.createGeoPainter(msg);
      }

      getGeomDescription() {
         let element = this.controller.mgr.GetElement(this.controller.eveViewerId);
         for (let scene of element.childs)      {
            let prnt = this.controller.mgr.GetElement(scene.fSceneId);
            if (prnt?.childs)
              for (let k = 0; k < prnt.childs.length; ++k)
              {
                let elem = prnt.childs[k];
                if (elem?.geomDescription) {
                  let json = atob(elem.geomDescription);
                  return EVE.JSR.parse(json);
                }
              }
         }

      }

      cleanupGeoPainter()
      {
         this.geo_painter?.cleanup();
         delete this.geo_painter;
         delete this.normal_drawing;
      }

      cleanup()
      {
         this.cleanupGeoPainter();

         super.cleanup();
      }

      //==============================================================================

      make_object(/* name */)
      {
         return new THREE.Object3D;
      }

      get_top_scene()
      {
         return this.geo_painter.getExtrasContainer();
      }

      get outline_map() { return this.outline_pass.id2obj_map; }

      //==============================================================================

      createGeoPainter(msg)
      {
         let options = "outline";
         options += ", mouse_click"; // process mouse click events
         // options += ", ambient"; // use ambient light
         // options += " black, ";
         if (!this.controller.isEveCameraPerspective())
            options += ", ortho_camera";

         // TODO: should be specified somehow in XML file
         // MT-RCORE - why have I removed this ???
         this.get_view().$().css("overflow", "hidden").css("width", "100%").css("height", "100%");

         this.geo_painter = EVE.JSR.createGeoPainter(this.get_view().getDomRef(), null, options);

         // function used by TGeoPainter to create OutlineShader - for the moment remove from JSROOT
         this.geo_painter.createOutline = function(scene, camera, w, h) {
            // 'this' here will be TGeoPainter!
            this.outline_pass = new THREE.OutlinePassEve( new THREE.Vector2( w, h ), scene, camera );
            this.outline_pass.edgeStrength = 5.5;
            this.outline_pass.edgeGlow = 0.7;
            this.outline_pass.edgeThickness = 1.5;
            this.outline_pass.usePatternTexture = false;
            this.outline_pass.downSampleRatio = 1;
            this.outline_pass.glowDownSampleRatio = 3;
            this.getEffectComposer().addPass( this.outline_pass );

            this.fxaa_pass = new THREE.ShaderPass( THREE.FXAAShader );
            this.fxaa_pass.uniforms[ 'resolution' ].value.set( 1 / w, 1 / h );
            this.fxaa_pass.renderToScreen = true;
            this.getEffectComposer().addPass( this.fxaa_pass );
         };

         this.geo_painter.setMouseTmout(this.controller.htimeout);

         this.geo_painter.addOrbitControls();

         if (!msg) {
            this.geo_painter.assignObject(null);

            this.geo_painter.setGeomViewer(true); // disable several JSROOT features

            this.geo_painter.prepareObjectDraw(null) // and now start everything
               .then(() => this.onGeoPainterReady(this.geo_painter));
         } else {
            this.normal_drawing = true;

            this.geo_painter.extractRawShapes(msg, true);

            // assign configuration to the control ??
            if (msg.cfg) {
               this.geo_painter.ctrl.cfg = msg.cfg;
               this.geo_painter.ctrl.show_config = true;
            }
            this.geo_painter.prepareObjectDraw(msg.visibles, '__geom_viewer_selection__') // and now start everything
                .then(() => this.onGeoPainterReady(this.geo_painter));
         }
      }

      onGeoPainterReady(painter)
      {
         // AMT temporary here, should be set in camera instantiation time
         const camera = this.geo_painter?.getCamera();
         if (camera?.isOrthographicCamera) {
            camera.left   = -this.get_width();
            camera.right  =  this.get_width();
            camera.top    =  this.get_height();
            camera.bottom = -this.get_height();
            camera.updateProjectionMatrix();
         }

         painter.eveGLcontroller = this.controller;

         const ctrls = this.geo_painter?.getControls();

         /** Handler for single mouse click, provided by basic control, used in GeoPainter */
         if (ctrls)
            ctrls.processSingleClick = function(intersects) {
               if (!intersects) return;
               let intersect = null;
               for (let k=0;k<intersects.length;++k) {
                  if (intersects[k].object.get_ctrl) {
                     intersect = intersects[k];
                     break;
                  }
               }
               if (intersect) {
                  let c = intersect.object.get_ctrl();
                  c.elementSelected(c.extractIndex(intersect));
               }
            };

         /** Handler of mouse double click - either ignore or reset camera position */
         if ((this.controller.dblclick_action != "Reset") && ctrls)
            ctrls.processDblClick = function() { }

         if (ctrls)
            ctrls.processMouseMove = function(intersects) {
               let active_mesh = null, tooltip = null, resolve = null, names = [], geo_object, geo_index;

               // try to find mesh from intersections
               for (let k = 0; k < intersects.length; ++k) {
                  let obj = intersects[k].object, info = null;
                  if (!obj) continue;
                  if (obj.geo_object)
                     info = obj.geo_name;
                  else if (obj.stack)
                     info = painter.getStackFullName(obj.stack);
                  if (info === null)
                     continue;

                  if (info.indexOf("<prnt>")==0)
                     info = painter.getItemName() + info.substr(6);

                  names.push(info);

                  if (!active_mesh) {
                     active_mesh = obj;
                     tooltip = info;
                     geo_object = obj.geo_object;
                     if (obj.get_ctrl) {
                        geo_index = obj.get_ctrl().extractIndex(intersects[k]);
                        if ((geo_index !== undefined) && (typeof tooltip == "string"))
                           tooltip += " indx:" + JSON.stringify(geo_index);
                     }
                     if (active_mesh.stack)
                        resolve = painter.resolveStack(active_mesh.stack);
                  }
               }

               // painter.highlightMesh(active_mesh, undefined, geo_object, geo_index); AMT override
               if (active_mesh && active_mesh.get_ctrl())
               {
                  active_mesh.get_ctrl().elementHighlighted(geo_index);
               }
               else
               {
                  let sl = painter.eveGLcontroller.created_scenes;
                  for (let k=0; k < sl.length; ++k)
                     sl[k].clearHighlight();
               }

               if (painter.options.update_browser) {
                  if (painter.options.highlight && tooltip) names = [ tooltip ];
                  painter.activateInBrowser(names);
               }

               if (!resolve || !resolve.obj) return tooltip;

               let lines = EVE.JSR.provideObjectInfo(resolve.obj);
               lines.unshift(tooltip);

               return { name: resolve.obj.fName, title: resolve.obj.fTitle || resolve.obj._typename, lines: lines };
            }

         // this.geo_painter.addHighlightHandler(this); // register ourself for highlight handling
         this.last_highlight = null;

         // outline_pass passthrough
         this.outline_pass = this.geo_painter.outline_pass;

         let sz = this.geo_painter.getSizeFor3d();
         this.geo_painter.getEffectComposer()?.setSize( sz.width, sz.height);
         if (this.geo_painter.fxaa_pass)
            this.geo_painter.fxaa_pass.uniforms[ 'resolution' ].value.set( 1 / sz.width, 1 / sz.height );

         if (ctrls)
            ctrls.contextMenu = this.jsrootOrbitContext.bind(this);

         if (this.normal_drawing) {
            // TODO: create scene objects to controller to correctly update geom drawing

            const pthis = this;

            // we need scene objects, but need to manipulate it
            this.controller.createScenes();

            this.controller.created_scenes.forEach(scene => {
               scene.redrawScene = function() { console.log("do nothing with redraw"); }
               scene.sceneElementChange = function(msg) {
                  if (!msg.geomDescription)
                     return;

                  let json = atob(msg.geomDescription);
                  let draw_data = EVE.JSR.parse(json);

                  // delete all drawings with geometry painter
                  pthis.cleanupGeoPainter();

                  // create from scratch
                  pthis.createGeoPainter(draw_data);
               }
            });


         } else {
            // create only when geo painter is ready
            this.controller.createScenes();
            this.controller.redrawScenes();

            // is it too early?
            this.render();
         }
         this.geo_painter.adjustCameraPosition(true);

         this.controller.glViewerInitDone();
      }

      /** @summary Used together with the geo painter for processing context menu */
      jsrootOrbitContext(evnt, intersects) {

         let browseHandler = this.controller.invokeBrowseOf.bind(this.controller);

         EVE.JSR.createMenu(evnt, this.geo_painter).then(menu => {
            let numitems = 0;
            if (intersects)
               for (let n=0;n<intersects.length;++n)
                  if (intersects[n].object.geo_name) numitems++;

            if (numitems === 0) {
               // default JSROOT context menu
               menu.painter.fillContextMenu(menu);
            } else {
               let many = numitems > 1;

               if (many) menu.add("header: Items");

               for (let n=0;n<intersects.length;++n) {
                  let obj = intersects[n].object;
                  if (!obj.geo_name) continue;

                  menu.add((many ? "sub:" : "header:") + obj.geo_name, obj.geo_object, browseHandler);

                  menu.add("Browse", obj.geo_object, browseHandler);

                  let wireframe = menu.painter.accessObjectWireFrame(obj);

                  if (wireframe!==undefined)
                     menu.addchk(wireframe, "Wireframe", n, function(indx) {
                        let m = intersects[indx].object.material;
                        m.wireframe = !m.wireframe;
                        this.render3D();
                     });


                  // not yet working
                  // menu.add("Focus", n, function(indx) { this.focusCamera(intersects[indx].object); });

                  if (many) menu.add("endsub:");
               }
            }

            // show menu
            menu.show();
         });
      }

      //==============================================================================
      remoteToolTip()
      {
         // to be implemented
      }

      //==============================================================================

      render()
      {
         //let outline_pass = this.geo_painter.outline_pass;
         //if (outline_pass) outline_pass._selectedObjects = Object.values(outline_pass.id2obj_map).flat();

         this.geo_painter.render3D();
      }

      //==============================================================================

      onResizeTimeout()
      {
         this.geo_painter.checkResize();
         if (this.geo_painter.fxaa_pass) {
            const sz = this.geo_painter.getSizeFor3d()
            this.geo_painter.fxaa_pass.uniforms[ 'resolution' ].value.set( 1 / sz.width, 1 / sz.height );
         }
      }

   } // class GlViewerJSRoot

   return GlViewerJSRoot;
});
