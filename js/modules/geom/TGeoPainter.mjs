import { httpRequest, decodeUrl, browser, source_dir,
         settings, internals, constants, create, clone,
         findFunction, isBatchMode, isNodeJs, getDocument, isPromise } from '../core.mjs';
import { REVISION, DoubleSide, FrontSide,
         Color, Vector2, Vector3, Matrix4, Object3D, Box3, Group, Plane,
         Euler, Quaternion, MathUtils,
         Mesh, MeshLambertMaterial, MeshBasicMaterial,
         LineSegments, LineBasicMaterial, BufferAttribute,
         TextGeometry, BufferGeometry, BoxBufferGeometry, CircleBufferGeometry,
         SphereBufferGeometry, SphereGeometry, WireframeGeometry,
         Scene, Fog, BoxHelper, AxesHelper, GridHelper, OrthographicCamera, PerspectiveCamera,
         TransformControls, PointLight, AmbientLight, HemisphereLight,
         EffectComposer, RenderPass, SSAOPass, UnrealBloomPass } from '../three.mjs';
import { showProgress, injectStyle, ToolbarIcons } from '../gui/utils.mjs';
import { assign3DHandler, disposeThreejsObject, createOrbitControl,
         createLineSegments, InteractiveControl, PointsCreator,
         createRender3D, beforeRender3D, afterRender3D, getRender3DKind, cleanupRender3D,
         HelveticerRegularFont } from '../base/base3d.mjs';
import { getColor, getRootColors } from '../base/colors.mjs';
import { DrawOptions } from '../base/BasePainter.mjs';
import { ObjectPainter } from '../base/ObjectPainter.mjs';
import { createMenu, closeMenu } from '../gui/menu.mjs';
import { ensureTCanvas } from '../gpad/TCanvasPainter.mjs';
import { kindGeo, kindEve, geoCfg, geoBITS, ClonedNodes, testGeoBit, setGeoBit, toggleGeoBit, setInvisibleAll,
         countNumShapes, getNodeKind, produceRenderOrder, createFlippedMesh,
         projectGeometry, countGeometryFaces, createFrustum, createProjectionMatrix,
         getBoundingBox, provideObjectInfo, isSameStack, checkDuplicates, getObjectName, cleanupShape } from './geobase.mjs';


const _ENTIRE_SCENE = 0, _BLOOM_SCENE = 1;

/** @summary Function used to build hierarchy of elements of overlap object
  * @private */
function buildOverlapVolume(overlap) {

   let vol = create("TGeoVolume");

   setGeoBit(vol, geoBITS.kVisDaughters, true);
   vol.$geoh = true; // workaround, let know browser that we are in volumes hierarchy
   vol.fName = "";

   let node1 = create("TGeoNodeMatrix");
   node1.fName = overlap.fVolume1.fName || "Overlap1";
   node1.fMatrix = overlap.fMatrix1;
   node1.fVolume = overlap.fVolume1;
   // node1.fVolume.fLineColor = 2; // color assigned with _splitColors

   let node2 = create("TGeoNodeMatrix");
   node2.fName = overlap.fVolume2.fName || "Overlap2";
   node2.fMatrix = overlap.fMatrix2;
   node2.fVolume = overlap.fVolume2;
   // node2.fVolume.fLineColor = 3;  // color assigned with _splitColors

   vol.fNodes = create("TList");
   vol.fNodes.Add(node1);
   vol.fNodes.Add(node2);

   return vol;
}

let $comp_col_cnt = 0;

/** @summary Function used to build hierarchy of elements of composite shapes
  * @private */
function buildCompositeVolume(comp, maxlvl, side) {

   if (maxlvl === undefined) maxlvl = 1;
   if (!side) {
      $comp_col_cnt = 0;
      side = "";
   }

   let vol = create("TGeoVolume");
   setGeoBit(vol, geoBITS.kVisThis, true);
   setGeoBit(vol, geoBITS.kVisDaughters, true);

   if ((side && (comp._typename!=='TGeoCompositeShape')) || (maxlvl<=0)) {
      vol.fName = side;
      vol.fLineColor = ($comp_col_cnt++ % 8) + 2;
      vol.fShape = comp;
      return vol;
   }

   if (side) side += "/";
   vol.$geoh = true; // workaround, let know browser that we are in volumes hierarchy
   vol.fName = "";

   let node1 = create("TGeoNodeMatrix");
   setGeoBit(node1, geoBITS.kVisThis, true);
   setGeoBit(node1, geoBITS.kVisDaughters, true);
   node1.fName = "Left";
   node1.fMatrix = comp.fNode.fLeftMat;
   node1.fVolume = buildCompositeVolume(comp.fNode.fLeft, maxlvl-1, side + "Left");

   let node2 = create("TGeoNodeMatrix");
   setGeoBit(node2, geoBITS.kVisThis, true);
   setGeoBit(node2, geoBITS.kVisDaughters, true);
   node2.fName = "Right";
   node2.fMatrix = comp.fNode.fRightMat;
   node2.fVolume = buildCompositeVolume(comp.fNode.fRight, maxlvl-1, side + "Right");

   vol.fNodes = create("TList");
   vol.fNodes.Add(node1);
   vol.fNodes.Add(node2);

   if (!side) $comp_col_cnt = 0;

   return vol;
}


/** @summary create list entity for geo object
  * @private */
function createList(parent, lst, name, title) {

   if (!lst || !('arr' in lst) || (lst.arr.length==0)) return;

   let item = {
       _name: name,
       _kind: "ROOT.TList",
       _title: title,
       _more: true,
       _geoobj: lst,
       _parent: parent,
   };

   item._get = function(item /*, itemname */) {
      return Promise.resolve(item._geoobj || null);
   };

   item._expand = function(node, lst) {
      // only childs

      if ('fVolume' in lst)
         lst = lst.fVolume.fNodes;

      if (!('arr' in lst)) return false;

      node._childs = [];

      checkDuplicates(null, lst.arr);

      for (let n in lst.arr)
         createItem(node, lst.arr[n]);

      return true;
   };

   if (!parent._childs) parent._childs = [];
   parent._childs.push(item);
}


/** @summary Expand geo object
  * @private */
function expandGeoObject(parent, obj) {
   injectGeoStyle();

   if (!parent || !obj) return false;

   let isnode = (obj._typename.indexOf('TGeoNode') === 0),
       isvolume = (obj._typename.indexOf('TGeoVolume') === 0),
       ismanager = (obj._typename === 'TGeoManager'),
       iseve = ((obj._typename === 'TEveGeoShapeExtract') || (obj._typename === 'ROOT::Experimental::REveGeoShapeExtract')),
       isoverlap = (obj._typename === 'TGeoOverlap');

   if (!isnode && !isvolume && !ismanager && !iseve && !isoverlap) return false;

   if (parent._childs) return true;

   if (ismanager) {
      createList(parent, obj.fMaterials, "Materials", "list of materials");
      createList(parent, obj.fMedia, "Media", "list of media");
      createList(parent, obj.fTracks, "Tracks", "list of tracks");
      createList(parent, obj.fOverlaps, "Overlaps", "list of detected overlaps");
      createItem(parent, obj.fMasterVolume);
      return true;
   }

   if (isoverlap) {
      createItem(parent, obj.fVolume1);
      createItem(parent, obj.fVolume2);
      createItem(parent, obj.fMarker, 'Marker');
      return true;
   }

   let volume, subnodes, shape;

   if (iseve) {
      subnodes = obj.fElements?.arr;
      shape = obj.fShape;
   } else {
      volume = isnode ? obj.fVolume : obj;
      subnodes = volume?.fNodes?.arr;
      shape = volume?.fShape;
   }

   if (!subnodes && (shape?._typename === "TGeoCompositeShape") && shape?.fNode) {
      if (!parent._childs) {
         createItem(parent, shape.fNode.fLeft, 'Left');
         createItem(parent, shape.fNode.fRight, 'Right');
      }

      return true;
   }

   if (!subnodes) return false;

   checkDuplicates(obj, subnodes);

   for (let i = 0; i < subnodes.length; ++i)
      createItem(parent, subnodes[i]);

   return true;
}


/** @summary find item with 3d painter
  * @private */
function findItemWithPainter(hitem, funcname) {
   while (hitem) {
      if (hitem._painter?._camera) {
         if (funcname && typeof hitem._painter[funcname] == 'function')
            hitem._painter[funcname]();
         return hitem;
      }
      hitem = hitem._parent;
   }
   return null;
}

/** @summary provide css style for geo object
  * @private */
function provideVisStyle(obj) {
   if ((obj._typename === 'TEveGeoShapeExtract') || (obj._typename === 'ROOT::Experimental::REveGeoShapeExtract'))
      return obj.fRnrSelf ? " geovis_this" : "";

   let vis = !testGeoBit(obj, geoBITS.kVisNone) &&
              testGeoBit(obj, geoBITS.kVisThis),
       chld = testGeoBit(obj, geoBITS.kVisDaughters);

   if (chld && (!obj.fNodes || (obj.fNodes.arr.length === 0))) chld = false;

   if (vis && chld) return " geovis_all";
   if (vis) return " geovis_this";
   if (chld) return " geovis_daughters";
   return "";
}


/** @summary update icons
  * @private */
function updateBrowserIcons(obj, hpainter) {
   if (!obj || !hpainter) return;

   hpainter.forEachItem(m => {
      // update all items with that volume
      if ((obj === m._volume) || (obj === m._geoobj)) {
         m._icon = m._icon.split(" ")[0] + provideVisStyle(obj);
         hpainter.updateTreeNode(m);
      }
   });
}


/** @summary provide icon name for the shape
  * @private */
function getShapeIcon(shape) {
   switch (shape._typename) {
      case "TGeoArb8": return "img_geoarb8";
      case "TGeoCone": return "img_geocone";
      case "TGeoConeSeg": return "img_geoconeseg";
      case "TGeoCompositeShape": return "img_geocomposite";
      case "TGeoTube": return "img_geotube";
      case "TGeoTubeSeg": return "img_geotubeseg";
      case "TGeoPara": return "img_geopara";
      case "TGeoParaboloid": return "img_geoparab";
      case "TGeoPcon": return "img_geopcon";
      case "TGeoPgon": return "img_geopgon";
      case "TGeoShapeAssembly": return "img_geoassembly";
      case "TGeoSphere": return "img_geosphere";
      case "TGeoTorus": return "img_geotorus";
      case "TGeoTrd1": return "img_geotrd1";
      case "TGeoTrd2": return "img_geotrd2";
      case "TGeoXtru": return "img_geoxtru";
      case "TGeoTrap": return "img_geotrap";
      case "TGeoGtra": return "img_geogtra";
      case "TGeoEltu": return "img_geoeltu";
      case "TGeoHype": return "img_geohype";
      case "TGeoCtub": return "img_geoctub";
   }
   return "img_geotube";
}


/**
  * @summary Toolbar for geometry painter
  *
  * @private
  */

class Toolbar {

   /** @summary constructor */
   constructor(container, bright) {
      this.bright = bright;

      this.element = container.append("div").attr('class','geo_toolbar_group');

      injectStyle(
         `.geo_toolbar_group { float: left; box-sizing: border-box; position: relative; bottom: 23px; vertical-align: middle; white-space: nowrap; }
          .geo_toolbar_group:first-child { margin-left: 2px; }
          .geo_toolbar_group a { position: relative; font-size: 16px; padding: 3px 1px; cursor: pointer; line-height: normal; box-sizing: border-box; }
          .geo_toolbar_group a svg { position: relative; top: 2px; }
          .geo_toolbar_btn path { fill: rgba(0, 31, 95, 0.2); }
          .geo_toolbar_btn path .active,
          .geo_toolbar_btn path:hover { fill: rgba(0, 22, 72, 0.5); }
          .geo_toolbar_btn_bright path { fill: rgba(255, 224, 160, 0.2); }
          .geo_toolbar_btn_bright path .active,
          .geo_toolbar_btn_bright path:hover { fill: rgba(255, 233, 183, 0.5); }`, this.element.node());
   }

   /** @summary add buttons */
   addButtons(buttons) {
      this.buttonsNames = [];

      buttons.forEach(buttonConfig => {
         let buttonName = buttonConfig.name;
         if (!buttonName)
            throw new Error(`must provide button ${name} in button config`);
         if (this.buttonsNames.indexOf(buttonName) !== -1)
            throw new Error(`button name ${buttonName} is taken`);

         this.buttonsNames.push(buttonName);

         let title = buttonConfig.title || buttonConfig.name;

         if (typeof buttonConfig.click !== 'function')
            throw new Error('must provide button "click" function in button config');

         let button = this.element.append('a')
                           .attr('class', this.bright ? 'geo_toolbar_btn_bright' : 'geo_toolbar_btn')
                           .attr('rel', 'tooltip')
                           .attr('data-title', title)
                           .on('click', buttonConfig.click);

         ToolbarIcons.createSVG(button, ToolbarIcons[buttonConfig.icon], 16, title);
      });

   }

   /** @summary change brightness */
   changeBrightness(bright) {
      this.bright = bright;
      if (this.element)
         this.element.selectAll(bright ? '.geo_toolbar_btn' : ".geo_toolbar_btn_bright")
                     .attr("class", !bright ? 'geo_toolbar_btn' : "geo_toolbar_btn_bright");
   }

   /** @summary cleanup toolbar */
   cleanup() {
      if (this.element) {
         this.element.remove();
         delete this.element;
      }
   }

} // class ToolBar


/**
  * @summary geometry drawing control
  *
  * @private
  */

class GeoDrawingControl extends InteractiveControl {

   constructor(mesh, bloom) {
      super();
      this.mesh = (mesh && mesh.material) ? mesh : null;
      this.bloom = bloom;
   }

   /** @summary set highlight */
   setHighlight(col, indx) {
      return this.drawSpecial(col, indx);
   }

   /** @summary draw special */
   drawSpecial(col /*, indx*/) {
      let c = this.mesh;
      if (!c || !c.material) return;

      if (col) {
         if (!c.origin)
            c.origin = {
              color: c.material.color,
              emissive: c.material.emissive,
              opacity: c.material.opacity,
              width: c.material.linewidth,
              size: c.material.size
           };
         if (this.bloom) {
            c.layers.enable(_BLOOM_SCENE);
            c.material.emissive = new Color(0x00ff00);
         } else {
            c.material.color = new Color( col );
            c.material.opacity = 1.;
         }

         if (c.hightlightWidthScale && !browser.isWin)
            c.material.linewidth = c.origin.width * c.hightlightWidthScale;
         if (c.highlightScale)
            c.material.size = c.origin.size * c.highlightScale;
         return true;
      } else if (c.origin) {
         if (this.bloom) {
            c.material.emissive = c.origin.emissive;
            c.layers.enable(_ENTIRE_SCENE);
         } else {
            c.material.color = c.origin.color;
            c.material.opacity = c.origin.opacity;
         }
         if (c.hightlightWidthScale)
            c.material.linewidth = c.origin.width;
         if (c.highlightScale)
            c.material.size = c.origin.size;
         return true;
      }
   }

} // class GeoDrawingControl


const stageInit = 0, stageCollect = 1, stageWorkerCollect = 2, stageAnalyze = 3, stageCollShapes = 4,
      stageStartBuild = 5, stageWorkerBuild = 6, stageBuild = 7, stageBuildReady = 8, stageWaitMain = 9, stageBuildProj = 10;

/**
 * @summary Painter class for geometries drawing
 *
 * @private
 */

class TGeoPainter extends ObjectPainter {

   /** @summary Constructor
     * @param {object|string} dom - DOM element for drawing or element id
     * @param {object} obj - supported TGeo object */
   constructor(dom, obj) {

      let gm;
      if (obj && (obj._typename === "TGeoManager")) {
         gm = obj;
         obj = obj.fMasterVolume;
      }

      if (obj && (obj._typename.indexOf('TGeoVolume') === 0))
         obj = { _typename:"TGeoNode", fVolume: obj, fName: obj.fName, $geoh: obj.$geoh, _proxy: true };

      super(dom, obj);

      if (gm) this.geo_manager = gm;

      this.no_default_title = true; // do not set title to main DIV
      this.mode3d = true; // indication of 3D mode
      this.drawing_stage = stageInit; //
      this.drawing_log = "Init";
      this.ctrl = {
         clipIntersect: true,
         clip: [{ name:"x", enabled: false, value: 0, min: -100, max: 100}, { name:"y", enabled: false, value: 0, min: -100, max: 100}, { name:"z", enabled: false, value: 0, min: -100, max: 100}],
         ssao: { enabled: false, output: SSAOPass.OUTPUT.Default, kernelRadius: 0, minDistance: 0.001, maxDistance: 0.1 },
         bloom: { enabled: true, strength: 1.5 },
         info: { num_meshes: 0, num_faces: 0, num_shapes: 0 },
         highlight: false,
         highlight_scene: false,
         depthTest: true,
         depthMethod: "dflt",
         select_in_view: false,
         update_browser: true,
         light: { kind: "points", top: false, bottom: false, left: false, right: false, front: false, specular: true, power: 1 },
         trans_radial: 0,
         trans_z: 0
      };

      this.ctrl.depthMethodItems = [
         { name: 'Default', value: "dflt" },
         { name: 'Raytraicing', value: "ray" },
         { name: 'Boundary box', value: "box" },
         { name: 'Mesh size', value: "size" },
         { name: 'Central point', value: "pnt" }
       ];

      this.ctrl.ssao.outputItems = [
         { name: 'Default', value: SSAOPass.OUTPUT.Default },
         { name: 'SSAO Only', value: SSAOPass.OUTPUT.SSAO },
         { name: 'SSAO Only + Blur', value: SSAOPass.OUTPUT.Blur },
         { name: 'Beauty', value: SSAOPass.OUTPUT.Beauty },
         { name: 'Depth', value: SSAOPass.OUTPUT.Depth },
         { name: 'Normal', value: SSAOPass.OUTPUT.Normal }
      ];

      this.cleanup(true);
   }

   /** @summary Change drawing stage
     * @private */
   changeStage(value, msg) {
      this.drawing_stage = value;
      if (!msg)
         switch(value) {
            case stageInit: msg = "Building done"; break;
            case stageCollect: msg = "collect visibles"; break;
            case stageWorkerCollect: msg = "worker collect visibles"; break;
            case stageAnalyze: msg = "Analyse visibles"; break;
            case stageCollShapes: msg = "collect shapes for building"; break;
            case stageStartBuild: msg = "Start build shapes"; break;
            case stageWorkerBuild: msg = "Worker build shapes"; break;
            case stageBuild: msg = "Build shapes"; break;
            case stageBuildReady: msg = "Build ready"; break;
            case stageWaitMain: msg = "Wait for main painter"; break;
            case stageBuildProj: msg = "Build projection"; break;
            default: msg = `stage ${value}`;
         }
      this.drawing_log = msg;
   }

   /** @summary Check drawing stage */
   isStage(value) { return value === this.drawing_stage; }

   /** @summary Create toolbar */
   createToolbar() {
      if (this._toolbar || !this._webgl || this.ctrl.notoolbar || isBatchMode()) return;
      let buttonList = [{
         name: 'toImage',
         title: 'Save as PNG',
         icon: "camera",
         click: () => this.createSnapshot()
      }, {
         name: 'control',
         title: 'Toggle control UI',
         icon: "rect",
         click: () => this.showControlOptions('toggle')
      }, {
         name: 'enlarge',
         title: 'Enlarge geometry drawing',
         icon: "circle",
         click: () => this.toggleEnlarge()
      }];

      // Only show VR icon if WebVR API available.
      if (navigator.getVRDisplays) {
         buttonList.push({
            name: 'entervr',
            title: 'Enter VR (It requires a VR Headset connected)',
            icon: "vrgoggles",
            click: () => this.toggleVRMode()
         });
         this.initVRMode();
      }

      if (settings.ContextMenu)
      buttonList.push({
         name: 'menu',
         title: 'Show context menu',
         icon: "question",
         click: evnt => {

            evnt.preventDefault();
            evnt.stopPropagation();

            if (closeMenu()) return;

            createMenu(evnt, this).then(menu => {
                menu.painter.fillContextMenu(menu);
                menu.show();
            });
         }
      });

      let bkgr = new Color(this.ctrl.background);

      this._toolbar = new Toolbar(this.selectDom(), (bkgr.r + bkgr.g + bkgr.b) < 1);

      this._toolbar.addButtons(buttonList);
   }

   /** @summary Initialize VR mode */
   initVRMode() {
      // Dolly contains camera and controllers in VR Mode
      // Allows moving the user in the scene
      this._dolly = new Group();
      this._scene.add(this._dolly);
      this._standingMatrix = new Matrix4();

      // Raycaster temp variables to avoid one per frame allocation.
      this._raycasterEnd = new Vector3();
      this._raycasterOrigin = new Vector3();

      navigator.getVRDisplays().then(displays => {
         let vrDisplay = displays[0];
         if (!vrDisplay) return;
         this._renderer.vr.setDevice(vrDisplay);
         this._vrDisplay = vrDisplay;
         if (vrDisplay.stageParameters) {
            this._standingMatrix.fromArray(vrDisplay.stageParameters.sittingToStandingTransform);
         }
         this.initVRControllersGeometry();
      });
   }

   /** @summary Init VR controllers geometry
     * @private */
   initVRControllersGeometry() {
      let geometry = new SphereGeometry(0.025, 18, 36),
          material = new MeshBasicMaterial({ color: 'grey', vertexColors: false }),
          rayMaterial = new MeshBasicMaterial({ color: 'fuchsia', vertexColors: false }),
          rayGeometry = new BoxBufferGeometry(0.001, 0.001, 2),
          ray1Mesh = new Mesh(rayGeometry, rayMaterial),
          ray2Mesh = new Mesh(rayGeometry, rayMaterial),
          sphere1 = new Mesh(geometry, material),
          sphere2 = new Mesh(geometry, material);

      this._controllersMeshes = [];
      this._controllersMeshes.push(sphere1);
      this._controllersMeshes.push(sphere2);
      ray1Mesh.position.z -= 1;
      ray2Mesh.position.z -= 1;
      sphere1.add(ray1Mesh);
      sphere2.add(ray2Mesh);
      this._dolly.add(sphere1);
      this._dolly.add(sphere2);
      // Controller mesh hidden by default
      sphere1.visible = false;
      sphere2.visible = false;
   }

   /** @summary Update VR controllers list
     * @private */
   updateVRControllersList() {
      let gamepads = navigator.getGamepads && navigator.getGamepads();
      // Has controller list changed?
      if (this.vrControllers && (gamepads.length === this.vrControllers.length)) { return; }
      // Hide meshes.
      this._controllersMeshes.forEach(mesh => { mesh.visible = false; });
      this._vrControllers = [];
      for (let i = 0; i < gamepads.length; ++i) {
         if (!gamepads[i] || !gamepads[i].pose) { continue; }
         this._vrControllers.push({
            gamepad: gamepads[i],
            mesh: this._controllersMeshes[i]
         });
         this._controllersMeshes[i].visible = true;
      }
   }

   /** @summary Process VR controller intersection
     * @private */
   processVRControllerIntersections() {
      let intersects = []
      for (let i = 0; i < this._vrControllers.length; ++i) {
         let controller = this._vrControllers[i].mesh,
             end = controller.localToWorld(this._raycasterEnd.set(0, 0, -1)),
             origin = controller.localToWorld(this._raycasterOrigin.set(0, 0, 0));
         end.sub(origin).normalize();
         intersects = intersects.concat(this._controls.getOriginDirectionIntersects(origin, end));
      }
      // Remove duplicates.
      intersects = intersects.filter(function (item, pos) {return intersects.indexOf(item) === pos});
      this._controls.processMouseMove(intersects);
   }

   /** @summary Update VR controllers
     * @private */
   updateVRControllers() {
      this.updateVRControllersList();
      // Update pose.
      for (let i = 0; i < this._vrControllers.length; ++i) {
         let controller = this._vrControllers[i],
             orientation = controller.gamepad.pose.orientation,
             position = controller.gamepad.pose.position,
             controllerMesh = controller.mesh;
         if (orientation) { controllerMesh.quaternion.fromArray(orientation); }
         if (position) { controllerMesh.position.fromArray(position); }
         controllerMesh.updateMatrix();
         controllerMesh.applyMatrix4(this._standingMatrix);
         controllerMesh.matrixWorldNeedsUpdate = true;
      }
      this.processVRControllerIntersections();
   }

   /** @summary Toggle VR mode
     * @private */
   toggleVRMode() {
      if (!this._vrDisplay) return;
      // Toggle VR mode off
      if (this._vrDisplay.isPresenting) {
         this.exitVRMode();
         return;
      }
      this._previousCameraPosition = this._camera.position.clone();
      this._previousCameraRotation = this._camera.rotation.clone();
      this._vrDisplay.requestPresent([{ source: this._renderer.domElement }]).then(() => {
         this._previousCameraNear = this._camera.near;
         this._dolly.position.set(this._camera.position.x/4, - this._camera.position.y/8, - this._camera.position.z/4);
         this._camera.position.set(0,0,0);
         this._dolly.add(this._camera);
         this._camera.near = 0.1;
         this._camera.updateProjectionMatrix();
         this._renderer.vr.enabled = true;
         this._renderer.setAnimationLoop(() => {
            this.updateVRControllers();
            this.render3D(0);
         });
      });
      this._renderer.vr.enabled = true;

      window.addEventListener( 'keydown', evnt => {
         // Esc Key turns VR mode off
         if (evnt.code == 'Escape') this.exitVRMode();
      });
   }

   /** @summary Exit VR mode
     * @private */
   exitVRMode() {
      if (!this._vrDisplay.isPresenting) return;
      this._renderer.vr.enabled = false;
      this._dolly.remove(this._camera);
      this._scene.add(this._camera);
      // Restore Camera pose
      this._camera.position.copy(this._previousCameraPosition);
      this._previousCameraPosition = undefined;
      this._camera.rotation.copy(this._previousCameraRotation);
      this._previousCameraRotation = undefined;
      this._camera.near = this._previousCameraNear;
      this._camera.updateProjectionMatrix();
      this._vrDisplay.exitPresent();
   }

   /** @summary Returns main geometry object */
   getGeometry() {
      return this.getObject();
   }

   /** @summary Modify visibility of provided node by name */
   modifyVisisbility(name, sign) {
      if (getNodeKind(this.getGeometry()) !== 0) return;

      if (name == "")
         return setGeoBit(this.getGeometry().fVolume, geoBITS.kVisThis, (sign === "+"));

      let regexp, exact = false;

      //arg.node.fVolume
      if (name.indexOf("*") < 0) {
         regexp = new RegExp("^"+name+"$");
         exact = true;
      } else {
         regexp = new RegExp("^" + name.split("*").join(".*") + "$");
         exact = false;
      }

      this.findNodeWithVolume(regexp, function(arg) {
         setInvisibleAll(arg.node.fVolume, (sign !== "+"));
         return exact ? arg : null; // continue search if not exact expression provided
      });
   }

   /** @summary Decode drawing options */
   decodeOptions(opt) {
      if (typeof opt != "string") opt = "";

      let res = { _grid: false, _bound: false, _debug: false,
                  _full: false, _axis: 0,
                  _count: false, wireframe: false,
                   scale: new Vector3(1,1,1), zoom: 1.0, rotatey: 0, rotatez: 0,
                   more: 1, maxlimit: 100000,
                   vislevel: undefined, maxnodes: undefined, dflt_colors: false,
                   use_worker: false, show_controls: false,
                   highlight: false, highlight_scene: false, no_screen: false,
                   project: '', is_main: false, tracks: false, showtop: false, can_rotate: true, ortho_camera: false,
                   clipx: false, clipy: false, clipz: false, usessao: false, usebloom: true, outline: false,
                   script_name: "", transparency: 0, rotate: false, background: '#FFFFFF',
                   depthMethod: "dflt", mouse_tmout: 50, trans_radial: 0, trans_z: 0 };

      let dd = decodeUrl();
      if (dd.get('_grid') == "true") res._grid = true;
      let _opt = dd.get('_debug');
      if (_opt == "true") { res._debug = true; res._grid = true; }
      if (_opt == "bound") { res._debug = true; res._grid = true; res._bound = true; }
      if (_opt == "full") { res._debug = true; res._grid = true; res._full = true; res._bound = true; }

      let macro = opt.indexOf("macro:");
      if (macro >= 0) {
         let separ = opt.indexOf(";", macro+6);
         if (separ<0) separ = opt.length;
         res.script_name = opt.slice(macro+6, separ);
         opt = opt.slice(0, macro) + opt.slice(separ+1);
         console.log('script', res.script_name, 'rest', opt);
      }

      while (true) {
         let pp = opt.indexOf("+"), pm = opt.indexOf("-");
         if ((pp<0) && (pm<0)) break;
         let p1 = pp, sign = "+";
         if ((p1<0) || ((pm>=0) && (pm<pp))) { p1 = pm; sign = "-"; }

         let p2 = p1+1, regexp = new RegExp('[,; .]');
         while ((p2<opt.length) && !regexp.test(opt[p2]) && (opt[p2]!='+') && (opt[p2]!='-')) p2++;

         let name = opt.substring(p1+1, p2);
         opt = opt.slice(0,p1) + opt.slice(p2);
         // console.log("Modify visibility", sign,':',name);

         this.modifyVisisbility(name, sign);
      }

      let d = new DrawOptions(opt);

      if (d.check("MAIN")) res.is_main = true;

      if (d.check("TRACKS")) res.tracks = true; // only for TGeoManager
      if (d.check("SHOWTOP")) res.showtop = true; // only for TGeoManager
      if (d.check("NO_SCREEN")) res.no_screen = true; // ignore kVisOnScreen bits for visibility

      if (d.check("ORTHO_CAMERA_ROTATE")) { res.ortho_camera = true; res.can_rotate = true; }
      if (d.check("ORTHO_CAMERA")) { res.ortho_camera = true; res.can_rotate = false; }
      if (d.check("MOUSE_CLICK")) res.mouse_click = true;

      if (d.check("DEPTHRAY") || d.check("DRAY")) res.depthMethod = "ray";
      if (d.check("DEPTHBOX") || d.check("DBOX")) res.depthMethod = "box";
      if (d.check("DEPTHPNT") || d.check("DPNT")) res.depthMethod = "pnt";
      if (d.check("DEPTHSIZE") || d.check("DSIZE")) res.depthMethod = "size";
      if (d.check("DEPTHDFLT") || d.check("DDFLT")) res.depthMethod = "dflt";

      if (d.check("ZOOM", true)) res.zoom = d.partAsFloat(0, 100) / 100;
      if (d.check("ROTY", true)) res.rotatey = d.partAsFloat();
      if (d.check("ROTZ", true)) res.rotatez = d.partAsFloat();
      if (d.check("VISLVL", true)) res.vislevel = d.partAsInt();

      if (d.check('BLACK')) res.background = "#000000";
      if (d.check('WHITE')) res.background = "#FFFFFF";

      if (d.check('BKGR_', true)) {
         let bckgr = null;
         if (d.partAsInt(1) > 0) {
           bckgr = getColor(d.partAsInt());
         } else {
            for (let col = 0; col < 8; ++col)
               if (getColor(col).toUpperCase() === d.part)
                  bckgr = getColor(col);
         }
         if (bckgr) res.background = "#" + new Color(bckgr).getHexString();
      }

      if (d.check('R3D_', true))
         res.Render3D = constants.Render3D.fromString(d.part.toLowerCase());

      if (d.check("MORE3")) res.more = 3;
      if (d.check("MORE")) res.more = 2;
      if (d.check("ALL")) { res.more = 10; res.vislevel = 9; }

      if (d.check("CONTROLS") || d.check("CTRL")) res.show_controls = true;

      if (d.check("CLIPXYZ")) res.clipx = res.clipy = res.clipz = true;
      if (d.check("CLIPX")) res.clipx = true;
      if (d.check("CLIPY")) res.clipy = true;
      if (d.check("CLIPZ")) res.clipz = true;
      if (d.check("CLIP")) res.clipx = res.clipy = res.clipz = true;

      if (d.check("PROJX", true)) { res.project = 'x'; if (d.partAsInt(1)>0) res.projectPos = d.partAsInt(); res.can_rotate = false; }
      if (d.check("PROJY", true)) { res.project = 'y'; if (d.partAsInt(1)>0) res.projectPos = d.partAsInt(); res.can_rotate = false; }
      if (d.check("PROJZ", true)) { res.project = 'z'; if (d.partAsInt(1)>0) res.projectPos = d.partAsInt(); res.can_rotate = false; }

      if (d.check("DFLT_COLORS") || d.check("DFLT")) res.dflt_colors = true;
      if (d.check("SSAO")) res.usessao = true;
      if (d.check("NOBLOOM")) res.usebloom = false;
      if (d.check("BLOOM")) res.usebloom = true;
      if (d.check("OUTLINE")) res.outline = true;

      if (d.check("NOWORKER")) res.use_worker = -1;
      if (d.check("WORKER")) res.use_worker = 1;

      if (d.check("NOHIGHLIGHT") || d.check("NOHIGH")) res.highlight_scene = res.highlight = 0;
      if (d.check("HIGHLIGHT")) res.highlight_scene = res.highlight = true;
      if (d.check("HSCENEONLY")) { res.highlight_scene = true; res.highlight = 0; }
      if (d.check("NOHSCENE")) res.highlight_scene = 0;
      if (d.check("HSCENE")) res.highlight_scene = true;

      if (d.check("WIREFRAME") || d.check("WIRE")) res.wireframe = true;
      if (d.check("ROTATE")) res.rotate = true;

      if (d.check("INVX") || d.check("INVERTX")) res.scale.x = -1;
      if (d.check("INVY") || d.check("INVERTY")) res.scale.y = -1;
      if (d.check("INVZ") || d.check("INVERTZ")) res.scale.z = -1;

      if (d.check("COUNT")) res._count = true;

      if (d.check('TRANSP',true))
         res.transparency = d.partAsInt(0,100)/100;

      if (d.check('OPACITY',true))
         res.transparency = 1 - d.partAsInt(0,100)/100;

      if (d.check("AXISCENTER") || d.check("AC")) res._axis = 2;

      if (d.check('TRR',true)) res.trans_radial = d.partAsInt()/100;
      if (d.check('TRZ',true)) res.trans_z = d.partAsInt()/100;

      if (d.check("AXIS") || d.check("A")) res._axis = true;

      if (d.check("D")) res._debug = true;
      if (d.check("G")) res._grid = true;
      if (d.check("B")) res._bound = true;
      if (d.check("W")) res.wireframe = true;
      if (d.check("F")) res._full = true;
      if (d.check("Y")) res._yup = true;
      if (d.check("Z")) res._yup = false;

      // when drawing geometry without TCanvas, yup = true by default
      if (res._yup === undefined)
         res._yup = this.getCanvSvg().empty();

      return res;
   }

   /** @summary Activate specified items in the browser */
   activateInBrowser(names, force) {
      // if (this.getItemName() === null) return;

      if (typeof names == 'string') names = [ names ];

      if (this._hpainter) {
         // show browser if it not visible

         this._hpainter.activateItems(names, force);

         // if highlight in the browser disabled, suppress in few seconds
         if (!this.ctrl.update_browser)
            setTimeout(() => this._hpainter.activateItems([]), 2000);
      }
   }

   /** @summary  method used to check matrix calculations performance with current three.js model */
   testMatrixes() {

      let errcnt = 0, totalcnt = 0, totalmax = 0;

      let arg = {
            domatrix: true,
            func: (/*node*/) => {

               let m2 = this.getmatrix();

               let entry = this.CopyStack();

               let mesh = this._clones.createObject3D(entry.stack, this._toplevel, 'mesh');

               if (!mesh) return true;

               totalcnt++;

               let m1 = mesh.matrixWorld, flip;

               if (m1.equals(m2)) return true;
               if ((m1.determinant()>0) && (m2.determinant()<-0.9)) {
                  flip = new Vector3(1,1,-1);
                  m2 = m2.clone().scale(flip);
                  if (m1.equals(m2)) return true;
               }

               let max = 0;
               for (let k = 0; k < 16; ++k)
                  max = Math.max(max, Math.abs(m1.elements[k] - m2.elements[k]));

               totalmax = Math.max(max, totalmax);

               if (max < 1e-4) return true;

               console.log(this._clones.resolveStack(entry.stack).name, 'maxdiff', max, 'determ', m1.determinant(), m2.determinant());

               errcnt++;

               return false;
            }
         };


      let tm1 = new Date().getTime();

      /* let cnt = */ this._clones.scanVisible(arg);

      let tm2 = new Date().getTime();

      console.log(`Compare matrixes total ${totalcnt} errors ${errcnt} takes ${tm2-tm1} maxdiff ${totalmax}`);
   }

   /** @summary Fills context menu */
   fillContextMenu(menu) {
      menu.add("header: Draw options");

      menu.addchk(this.ctrl.update_browser, "Browser update", () => {
         this.ctrl.update_browser = !this.ctrl.update_browser;
         if (!this.ctrl.update_browser) this.activateInBrowser([]);
      });
      menu.addchk(this.ctrl.show_controls, "Show Controls", () => this.showControlOptions('toggle'));

      menu.addchk(this.ctrl._axis, "Show axes", () => this.setAxesDraw('toggle'));

      if (this.geo_manager)
         menu.addchk(this.ctrl.showtop, "Show top volume", () => this.setShowTop(!this.ctrl.showtop));

      menu.addchk(this.ctrl.wireframe, "Wire frame", () => this.toggleWireFrame());

      menu.addchk(this.ctrl.highlight, "Highlight volumes", () => {
         this.ctrl.highlight = !this.ctrl.highlight;
      });
      menu.addchk(this.ctrl.highlight_scene, "Highlight scene", () => {
         this.ctrl.highlight_scene = !this.ctrl.highlight_scene;
      });
      menu.add("Reset camera position", () => this.focusCamera());

      if (!this._geom_viewer)
         menu.add("Get camera position", () => menu.info("Position (as url)", "&opt=" + this.produceCameraUrl()));

      if (!this.ctrl.project)
         menu.addchk(this.ctrl.rotate, "Autorotate", () => this.setAutoRotate(!this.ctrl.rotate));
      menu.addchk(this.ctrl.select_in_view, "Select in view", () => {
         this.ctrl.select_in_view = !this.ctrl.select_in_view;
         if (this.ctrl.select_in_view) this.startDrawGeometry();
      });
   }

   /** @summary Method used to set transparency for all geometrical shapes
     * @param {number|Function} transparency - one could provide function
     * @param {boolean} [skip_render] - if specified, do not perform rendering */
   changedGlobalTransparency(transparency, skip_render) {
      let func = (typeof transparency == 'function') ? transparency : null;
      if (func || (transparency === undefined)) transparency = this.ctrl.transparency;
      this._toplevel.traverse( node => {
         if (node?.material?.inherentOpacity !== undefined) {
            let t = func ? func(node) : undefined;
            if (t !== undefined)
               node.material.opacity = 1 - t;
            else
               node.material.opacity = Math.min(1 - (transparency || 0), node.material.inherentOpacity);
            node.material.transparent = node.material.opacity < 1;
         }
      });
      if (!skip_render)
         this.render3D(-1);
   }

   /** @summary Reset transformation */
   resetTransformation() {
      this.changedTransformation("reset");
   }

   /** @summary Method should be called when transformation parameters were changed */
   changedTransformation(arg) {
      if (!this._toplevel) return;

      let ctrl = this.ctrl,
          translation = new Matrix4(),
          vect2 = new Vector3();

      if (arg == "reset")
         ctrl.trans_z = ctrl.trans_radial = 0;

      this._toplevel.traverse(mesh => {
         if (mesh.stack === undefined) return;

         let node = mesh.parent;

         if (arg == "reset") {
            if (node.matrix0) {
               node.matrix.copy(node.matrix0);
               node.matrix.decompose( node.position, node.quaternion, node.scale );
               node.matrixWorldNeedsUpdate = true;
            }
            delete node.matrix0;
            delete node.vect0;
            delete node.vect1;
            delete node.minvert;
            return;
         }

         if (node.vect0 === undefined) {
            node.matrix0 = node.matrix.clone();
            node.minvert = new Matrix4().copy(node.matrixWorld).invert();

            let box3 = getBoundingBox(mesh, null, true),
                signz = mesh._flippedMesh ? -1 : 1;

            // real center of mesh in local coordinates
            node.vect0 = new Vector3((box3.max.x  + box3.min.x) / 2, (box3.max.y  + box3.min.y) / 2, signz * (box3.max.z  + box3.min.z) / 2).applyMatrix4(node.matrixWorld);
            node.vect1 = new Vector3(0,0,0).applyMatrix4(node.minvert);
         }

         vect2.set(ctrl.trans_radial * node.vect0.x, ctrl.trans_radial * node.vect0.y, ctrl.trans_z * node.vect0.z).applyMatrix4(node.minvert).sub(node.vect1);

         node.matrix.multiplyMatrices(node.matrix0, translation.makeTranslation(vect2.x, vect2.y, vect2.z));
         node.matrix.decompose( node.position, node.quaternion, node.scale );
         node.matrixWorldNeedsUpdate = true;
      });

      this._toplevel.updateMatrixWorld();

      // axes drawing always triggers rendering
      if (arg != "norender")
         this.drawSimpleAxis();
   }

   /** @summary Should be called when autorotate property changed */
   changedAutoRotate() {
      this.autorotate(2.5);
   }

   /** @summary Method should be called when changing axes drawing */
   changedAxes() {
      if (typeof this.ctrl._axis == 'string')
         this.ctrl._axis = parseInt(this.ctrl._axis);

      this.drawSimpleAxis();
   }

   /** @summary Method should be called to change background color */
   changedBackground(val) {
      if (val !== undefined)
         this.ctrl.background = val;
      this._renderer.setClearColor(this.ctrl.background, 1);
      this.render3D(0);

      if (this._toolbar) {
         let bkgr = new Color(this.ctrl.background);
         this._toolbar.changeBrightness((bkgr.r + bkgr.g + bkgr.b) < 1);
      }
   }

   /** @summary Method called when SSAO configuration changed via GUI */
   changedSSAO() {
      if (!this.ctrl.ssao.enabled) {
         this.removeSSAO();
      } else {
         this.createSSAO();

         this._ssaoPass.output = parseInt(this.ctrl.ssao.output);
         this._ssaoPass.kernelRadius = this.ctrl.ssao.kernelRadius;
         this._ssaoPass.minDistance = this.ctrl.ssao.minDistance;
         this._ssaoPass.maxDistance = this.ctrl.ssao.maxDistance;
      }

      this.updateClipping();

      if (this._slave_painters)
         this._slave_painters.forEach(p => {
            Object.assign(p.ctrl.ssao, this.ctrl.ssao);
            p.changedSSAO();
         });
   }

   /** @summary Display control GUI */
   showControlOptions(on) {
      // while complete geo drawing can be removed until dat is loaded - just check and ignore callback
      if (!this.ctrl) return;

      if (on === 'toggle') {
         on = !this._datgui;
      } else if (on === undefined) {
         on = this.ctrl.show_controls;
      }

      this.ctrl.show_controls = on;

      if (this._datgui) {
         if (!on) {
            this._datgui.domElement.remove();
            this._datgui.destroy();
            delete this._datgui;
         }
         return;
      }

      if (on)
         import('../gui/dat.gui.mjs').then(h => this.buildDatGui(h));
   }

   /** @summary build dat.gui elements
     * @private */
   buildDatGui(dat) {
      // can happen when dat gui loaded after drawing is already cleaned
      if (!this._renderer) return;

      if (!dat)
         throw Error('Fail to load dat.gui');

      this._datgui = new dat.GUI({ autoPlace: false, width: Math.min(650, this._renderer.domElement.width / 2) });

      let main = this.selectDom();
      if (main.style('position')=='static') main.style('position','relative');

      let dom = this._datgui.domElement;
      dom.style.position = 'absolute';
      dom.style.top = 0;
      dom.style.right = 0;
      main.node().appendChild(dom);

      this._datgui.painter = this;

      if (this.ctrl.project) {

         let bound = this.getGeomBoundingBox(this.getProjectionSource(), 0.01),
             axis = this.ctrl.project;

         if (this.ctrl.projectPos === undefined)
            this.ctrl.projectPos = (bound.min[axis] + bound.max[axis])/2;

         this._datgui.add(this.ctrl, 'projectPos', bound.min[axis], bound.max[axis])
             .name(axis.toUpperCase() + ' projection')
             .onChange(() => this.startDrawGeometry());

      } else {
         // Clipping Options

         let clipFolder = this._datgui.addFolder('Clipping'),
             clip_handler = () => this.changedClipping(-1);

         for (let naxis = 0; naxis < 3; ++naxis) {
            let cc = this.ctrl.clip[naxis],
                axisC = cc.name.toUpperCase();

            clipFolder.add(cc, 'enabled')
                .name('Enable ' + axisC)
                .listen() // react if option changed outside
                .onChange(clip_handler);

            clipFolder.add(cc, "value", cc.min, cc.max)
                .name(axisC + ' position')
                .onChange(this.changedClipping.bind(this, naxis));
         }

         clipFolder.add(this.ctrl, 'clipIntersect').name("Clip intersection")
                   .listen().onChange(clip_handler);

      }

      // Appearance Options

      let appearance = this._datgui.addFolder('Appearance');

      appearance.add(this.ctrl, 'highlight').name('Highlight Selection')
                .listen().onChange(() => this.changedHighlight());

      appearance.add(this.ctrl, 'transparency', 0.0, 1.0, 0.001)
                     .listen().onChange(value => this.changedGlobalTransparency(value));

      appearance.addColor(this.ctrl, 'background').name('Background')
                .onChange(col => this.changedBackground(col));

      appearance.add(this.ctrl, 'wireframe').name('Wireframe')
                     .listen().onChange(() => this.changedWireFrame());

      this.ctrl._axis_cfg = 0;
      appearance.add(this.ctrl, '_axis', { "none": 0, "show": 1, "center": 2 }).name('Axes')
                    .onChange(() => this.changedAxes());

      if (!this.ctrl.project)
         appearance.add(this.ctrl, 'rotate').name("Autorotate")
                      .listen().onChange(() => this.changedAutoRotate());

      appearance.add(this, 'focusCamera').name('Reset camera position');

      // Advanced Options

      if (this._webgl) {
         let advanced = this._datgui.addFolder('Advanced'), depthcfg = {};
         this.ctrl.depthMethodItems.forEach(i => { depthcfg[i.name] = i.value; });

         advanced.add(this.ctrl, 'depthTest').name("Depth test")
            .listen().onChange(() => this.changedDepthTest());

         advanced.add( this.ctrl, 'depthMethod', depthcfg)
             .name("Rendering order")
             .onChange(method => this.changedDepthMethod(method));

         advanced.add(this.ctrl, 'ortho_camera').name("Orhographic camera")
                 .listen().onChange(() => this.changeCamera());

        advanced.add(this, 'resetAdvanced').name('Reset');
      }

      // Transformation Options
      if (!this.ctrl.project) {
         let transform = this._datgui.addFolder('Transform');
         transform.add(this.ctrl, 'trans_z', 0., 3., 0.01)
                     .name('Z axis')
                     .listen().onChange(() => this.changedTransformation());
         transform.add(this.ctrl, 'trans_radial', 0., 3., 0.01)
                  .name('Radial')
                  .listen().onChange(() => this.changedTransformation());

         transform.add(this, 'resetTransformation').name('Reset');

         if (this.ctrl.trans_z || this.ctrl.trans_radial) transform.open();
      }

      // no SSAO folder if outline is enabled
      if (this.ctrl.outline) return;

      let ssaofolder = this._datgui.addFolder('Smooth Lighting (SSAO)'),
          ssao_handler = () => this.changedSSAO(), ssaocfg = {};

      this.ctrl.ssao.outputItems.forEach(i => { ssaocfg[i.name] = i.value; });

      ssaofolder.add(this.ctrl.ssao, 'enabled').name('Enable SSAO')
                .listen().onChange(ssao_handler);

      ssaofolder.add( this.ctrl.ssao, 'output', ssaocfg)
                .listen().onChange(ssao_handler);

      ssaofolder.add( this.ctrl.ssao, 'kernelRadius', 0, 32)
                .listen().onChange(ssao_handler);

      ssaofolder.add( this.ctrl.ssao, 'minDistance', 0.001, 0.02)
                .listen().onChange(ssao_handler);

      ssaofolder.add( this.ctrl.ssao, 'maxDistance', 0.01, 0.3)
                .listen().onChange(ssao_handler);

      let blooming = this._datgui.addFolder('Unreal Bloom'),
          bloom_handler = () => this.changedBloomSettings();

      blooming.add(this.ctrl.bloom, 'enabled').name('Enable Blooming')
              .listen().onChange(bloom_handler);

      blooming.add( this.ctrl.bloom, 'strength', 0.0, 3.0).name("Strength")
               .listen().onChange(bloom_handler);
   }

   /** @summary Method called when bloom configuration changed via GUI */
   changedBloomSettings() {
      if (this.ctrl.bloom.enabled) {
         this.createBloom();
         this._bloomPass.strength = this.ctrl.bloom.strength;
      } else {
         this.removeBloom();
      }

      if (this._slave_painters)
         this._slave_painters.forEach(p => {
            Object.assign(p.ctrl.bloom, this.ctrl.bloom);
            p.changedBloomSettings();
         });
   }

   /** @summary Handle change of camera kind */
   changeCamera() {
      // force control recreation
      if (this._controls) {
          this._controls.cleanup();
          delete this._controls;
       }

       this.removeBloom();
       this.removeSSAO();

      // recreate camera
      this.createCamera();

      this.createSpecialEffects();

      this._first_drawing = true;
      this.startDrawGeometry(true);
   }

   /** @summary create bloom effect */
   createBloom() {
      if (this._bloomPass) return;

      this._camera.layers.enable( _BLOOM_SCENE );
      this._bloomComposer = new EffectComposer( this._renderer );
      this._bloomComposer.addPass( new RenderPass( this._scene, this._camera ) );
      this._bloomPass = new UnrealBloomPass(new Vector2( window.innerWidth, window.innerHeight ), 1.5, 0.4, 0.85);
      this._bloomPass.threshold = 0;
      this._bloomPass.strength = this.ctrl.bloom.strength;
      this._bloomPass.radius = 0;
      this._bloomPass.renderToScreen = true;
      this._bloomComposer.addPass( this._bloomPass );
      this._renderer.autoClear = false;
   }

   /** @summary Remove bloom highlight */
   removeBloom() {
      if (!this._bloomPass) return;
      delete this._bloomPass;
      delete this._bloomComposer;
      this._renderer.autoClear = true;
      this._camera.layers.disable( _BLOOM_SCENE );
   }

   /** @summary Remove composer */
   removeSSAO() {
      // we cannot remove pass from composer - just disable it
      delete this._ssaoPass;
      delete this._effectComposer;
   }

   /** @summary create SSAO */
   createSSAO() {
      if (!this._webgl) return;

      // this._depthRenderTarget = new WebGLRenderTarget( this._scene_width, this._scene_height, { minFilter: LinearFilter, magFilter: LinearFilter } );
      // Setup SSAO pass
      if (!this._ssaoPass) {
         if (!this._effectComposer) {
            this._effectComposer = new EffectComposer( this._renderer );
            this._effectComposer.addPass(new RenderPass( this._scene, this._camera));
         }

         this._ssaoPass = new SSAOPass( this._scene, this._camera, this._scene_width, this._scene_height );
         this._ssaoPass.kernelRadius = 16;
         this._ssaoPass.renderToScreen = true;

         // Add pass to effect composer
         this._effectComposer.addPass( this._ssaoPass );
      }
   }

   /** @summary Show context menu for orbit control
     * @private */
   orbitContext(evnt, intersects) {

      createMenu(evnt, this).then(menu => {
         let numitems = 0, numnodes = 0, cnt = 0;
         if (intersects)
            for (let n = 0; n < intersects.length; ++n) {
               if (intersects[n].object.stack) numnodes++;
               if (intersects[n].object.geo_name) numitems++;
            }

         if (numnodes + numitems === 0) {
            this.fillContextMenu(menu);
         } else {
            let many = (numnodes + numitems) > 1;

            if (many) menu.add("header:" + ((numitems > 0) ? "Items" : "Nodes"));

            for (let n = 0; n < intersects.length; ++n) {
               let obj = intersects[n].object,
                   name, itemname, hdr;

               if (obj.geo_name) {
                  itemname = obj.geo_name;
                  if (itemname.indexOf("<prnt>") == 0)
                     itemname = (this.getItemName() || "top") + itemname.slice(6);
                  name = itemname.slice(itemname.lastIndexOf("/")+1);
                  if (!name) name = itemname;
                  hdr = name;
               } else if (obj.stack) {
                  name = this._clones.resolveStack(obj.stack).name;
                  itemname = this.getStackFullName(obj.stack);
                  hdr = this.getItemName();
                  if (name.indexOf("Nodes/") === 0)
                     hdr = name.slice(6);
                  else if (name.length > 0)
                     hdr = name;
                  else if (!hdr)
                     hdr = "header";

               } else
                  continue;

               menu.add((many ? "sub:" : "header:") + hdr, itemname, arg => this.activateInBrowser([arg], true));

               menu.add("Browse", itemname, arg => this.activateInBrowser([arg], true));

               if (this._hpainter)
                  menu.add("Inspect", itemname, arg => this._hpainter.display(arg, "inspect"));

               if (obj.geo_name) {
                  menu.add("Hide", n, indx => {
                     let mesh = intersects[indx].object;
                     mesh.visible = false; // just disable mesh
                     if (mesh.geo_object) mesh.geo_object.$hidden_via_menu = true; // and hide object for further redraw
                     menu.painter.render3D();
                  });

                  if (many) menu.add("endsub:");

                  continue;
               }

               let wireframe = this.accessObjectWireFrame(obj);

               if (wireframe !== undefined)
                  menu.addchk(wireframe, "Wireframe", n, function(indx) {
                     let m = intersects[indx].object.material;
                     m.wireframe = !m.wireframe;
                     this.render3D();
                  });

               if (++cnt > 1)
                  menu.add("Manifest", n, function(indx) {

                     if (this._last_manifest)
                        this._last_manifest.wireframe = !this._last_manifest.wireframe;

                     if (this._last_hidden)
                        this._last_hidden.forEach(obj => { obj.visible = true; });

                     this._last_hidden = [];

                     for (let i=0;i<indx;++i)
                        this._last_hidden.push(intersects[i].object);

                     this._last_hidden.forEach(obj => { obj.visible = false; });

                     this._last_manifest = intersects[indx].object.material;

                     this._last_manifest.wireframe = !this._last_manifest.wireframe;

                     this.render3D();
                  });


               menu.add("Focus", n, function(indx) {
                  this.focusCamera(intersects[indx].object);
               });

               if (!this._geom_viewer)
               menu.add("Hide", n, function(indx) {
                  let resolve = menu.painter._clones.resolveStack(intersects[indx].object.stack);
                  if (resolve.obj && (resolve.node.kind === kindGeo) && resolve.obj.fVolume) {
                     setGeoBit(resolve.obj.fVolume, geoBITS.kVisThis, false);
                     updateBrowserIcons(resolve.obj.fVolume, this._hpainter);
                  } else if (resolve.obj && (resolve.node.kind === kindEve)) {
                     resolve.obj.fRnrSelf = false;
                     updateBrowserIcons(resolve.obj, this._hpainter);
                  }

                  this.testGeomChanges();// while many volumes may disappear, recheck all of them
               });

               if (many) menu.add("endsub:");
            }
         }
         menu.show();
      });
   }

   /** @summary Filter some objects from three.js intersects array */
   filterIntersects(intersects) {

      if (!intersects.length) return intersects;

      // check redirections
      for (let n = 0; n < intersects.length; ++n)
         if (intersects[n].object.geo_highlight)
            intersects[n].object = intersects[n].object.geo_highlight;

      // remove all elements without stack - indicator that this is geometry object
      // also remove all objects which are mostly transparent
      for (let n = intersects.length - 1; n >= 0; --n) {

         let obj = intersects[n].object,
            unique = (obj.stack !== undefined) || (obj.geo_name !== undefined);

         if (unique && obj.material && (obj.material.opacity !== undefined))
            unique = (obj.material.opacity >= 0.1);

         if (obj.jsroot_special) unique = false;

         for (let k = 0; (k < n) && unique;++k)
            if (intersects[k].object === obj) unique = false;

         if (!unique) intersects.splice(n,1);
      }

      let clip = this.ctrl.clip;

      if (clip[0].enabled || clip[1].enabled || clip[2].enabled) {
         let clippedIntersects = [];

         for (let i = 0; i < intersects.length; ++i) {
            let point = intersects[i].point, special = (intersects[i].object.type == "Points"), clipped = true;

            if (clip[0].enabled && ((this._clipPlanes[0].normal.dot(point) > this._clipPlanes[0].constant) ^ special)) clipped = false;
            if (clip[1].enabled && ((this._clipPlanes[1].normal.dot(point) > this._clipPlanes[1].constant) ^ special)) clipped = false;
            if (clip[2].enabled && (this._clipPlanes[2].normal.dot(point) > this._clipPlanes[2].constant)) clipped = false;

            if (!clipped) clippedIntersects.push(intersects[i]);
         }

         intersects = clippedIntersects;
      }

      return intersects;
   }

   /** @summary test camera position
     * @desc function analyzes camera position and start redraw of geometry
     *  if objects in view may be changed */
   testCameraPositionChange() {

      if (!this.ctrl.select_in_view || this._draw_all_nodes) return;

      let matrix = createProjectionMatrix(this._camera),
          frustum = createFrustum(matrix);

      // check if overall bounding box seen
      if (!frustum.CheckBox(this.getGeomBoundingBox(this._toplevel)))
         this.startDrawGeometry();
   }

   /** @summary Resolve stack */
   resolveStack(stack) {
      return this._clones && stack ? this._clones.resolveStack(stack) : null;
   }

   /** @summary Returns stack full name
     * @desc Includes item name of top geo object */
   getStackFullName(stack) {
      let mainitemname = this.getItemName(),
          sub = this.resolveStack(stack);
      if (!sub || !sub.name) return mainitemname;
      return mainitemname ? (mainitemname + "/" + sub.name) : sub.name;
   }

   /** @summary Add handler which will be called when element is highlighted in geometry drawing
     * @desc Handler should have highlightMesh function with same arguments as TGeoPainter  */
   addHighlightHandler(handler) {
      if (typeof handler?.highlightMesh != 'function') return;
      if (!this._highlight_handlers) this._highlight_handlers = [];
      this._highlight_handlers.push(handler);
   }

   /** @summary perform mesh highlight */
   highlightMesh(active_mesh, color, geo_object, geo_index, geo_stack, no_recursive) {

      if (geo_object) {
         active_mesh = active_mesh ? [ active_mesh ] : [];
         let extras = this.getExtrasContainer();
         if (extras)
            extras.traverse(obj3d => {
               if ((obj3d.geo_object === geo_object) && (active_mesh.indexOf(obj3d)<0)) active_mesh.push(obj3d);
            });
      } else if (geo_stack && this._toplevel) {
         active_mesh = [];
         this._toplevel.traverse(mesh => {
            if ((mesh instanceof Mesh) && isSameStack(mesh.stack, geo_stack)) active_mesh.push(mesh);
         });
      } else {
         active_mesh = active_mesh ? [ active_mesh ] : [];
      }

      if (!active_mesh.length) active_mesh = null;

      if (active_mesh) {
         // check if highlight is disabled for correspondent objects kinds
         if (active_mesh[0].geo_object) {
            if (!this.ctrl.highlight_scene) active_mesh = null;
         } else {
            if (!this.ctrl.highlight) active_mesh = null;
         }
      }

      if (!no_recursive) {
         // check all other painters

         if (active_mesh) {
            if (!geo_object) geo_object = active_mesh[0].geo_object;
            if (!geo_stack) geo_stack = active_mesh[0].stack;
         }

         let lst = this._highlight_handlers || (!this._main_painter ? this._slave_painters : this._main_painter._slave_painters.concat([this._main_painter]));

         for (let k = 0; k < lst.length; ++k)
            if (lst[k] !== this)
               lst[k].highlightMesh(null, color, geo_object, geo_index, geo_stack, true);
      }

      let curr_mesh = this._selected_mesh, same = false;

      if (!curr_mesh && !active_mesh) return false;

      const get_ctrl = mesh => mesh.get_ctrl ? mesh.get_ctrl() : new GeoDrawingControl(mesh, this.ctrl.bloom.enabled);

      // check if selections are the same
      if (curr_mesh && active_mesh && (curr_mesh.length == active_mesh.length)) {
         same = true;
         for (let k = 0; (k < curr_mesh.length) && same; ++k) {
            if ((curr_mesh[k] !== active_mesh[k]) || get_ctrl(curr_mesh[k]).checkHighlightIndex(geo_index)) same = false;
         }
      }
      if (same) return !!curr_mesh;

      if (curr_mesh)
         for (let k = 0; k < curr_mesh.length; ++k)
            get_ctrl(curr_mesh[k]).setHighlight();

      this._selected_mesh = active_mesh;

      if (active_mesh)
         for (let k = 0; k < active_mesh.length; ++k)
            get_ctrl(active_mesh[k]).setHighlight(color || 0x00ff00, geo_index);

      this.render3D(0);

      return !!active_mesh;
   }

   /** @summary handle mouse click event */
   processMouseClick(pnt, intersects, evnt) {
      if (!intersects.length) return;

      let mesh = intersects[0].object;
      if (!mesh.get_ctrl) return;

      let ctrl = mesh.get_ctrl(),
          click_indx = ctrl.extractIndex(intersects[0]);

      ctrl.evnt = evnt;

      if (ctrl.setSelected("blue", click_indx))
         this.render3D();

      ctrl.evnt = null;
   }

   /** @summary Configure mouse delay, required for complex geometries */
   setMouseTmout(val) {
      if (this.ctrl)
         this.ctrl.mouse_tmout = val;

      if (this._controls)
         this._controls.mouse_tmout = val;
   }

   /** @summary Configure depth method, used for render order production.
     * @param {string} method - Allowed values: "ray", "box","pnt", "size", "dflt" */
   setDepthMethod(method) {
      if (this.ctrl)
         this.ctrl.depthMethod = method;
   }

   /** @summary Add orbit control */
   addOrbitControls() {

      if (this._controls || !this._webgl || isBatchMode()) return;

      this.setTooltipAllowed(settings.Tooltip);

      this._controls = createOrbitControl(this, this._camera, this._scene, this._renderer, this._lookat);

      this._controls.mouse_tmout = this.ctrl.mouse_tmout; // set larger timeout for geometry processing

      if (!this.ctrl.can_rotate) this._controls.enableRotate = false;

      this._controls.contextMenu = this.orbitContext.bind(this);

      this._controls.processMouseMove = intersects => {

         // painter already cleaned up, ignore any incoming events
         if (!this.ctrl || !this._controls) return;

         let active_mesh = null, tooltip = null, resolve = null, names = [], geo_object, geo_index;

         // try to find mesh from intersections
         for (let k = 0; k < intersects.length; ++k) {
            let obj = intersects[k].object, info = null;
            if (!obj) continue;
            if (obj.geo_object)
               info = obj.geo_name;
            else if (obj.stack)
               info = this.getStackFullName(obj.stack);
            if (!info) continue;

            if (info.indexOf("<prnt>") == 0)
               info = this.getItemName() + info.slice(6);

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
               if (active_mesh.stack) resolve = this.resolveStack(active_mesh.stack);
            }
         }

         this.highlightMesh(active_mesh, undefined, geo_object, geo_index);

         if (this.ctrl.update_browser) {
            if (this.ctrl.highlight && tooltip) names = [ tooltip ];
            this.activateInBrowser(names);
         }

         if (!resolve || !resolve.obj) return tooltip;

         let lines = provideObjectInfo(resolve.obj);
         lines.unshift(tooltip);

         return { name: resolve.obj.fName, title: resolve.obj.fTitle || resolve.obj._typename, lines: lines };
      }

      this._controls.processMouseLeave = function() {
         this.processMouseMove([]); // to disable highlight and reset browser
      }

      this._controls.processDblClick = () => {
         // painter already cleaned up, ignore any incoming events
         if (!this.ctrl || !this._controls) return;

         if (this._last_manifest) {
            this._last_manifest.wireframe = !this._last_manifest.wireframe;
            if (this._last_hidden)
               this._last_hidden.forEach(obj => { obj.visible = true; });
            delete this._last_hidden;
            delete this._last_manifest;
            this.render3D();
         } else {
            this.adjustCameraPosition();
         }
      }
   }

   /** @summary add transformation control */
   addTransformControl() {
      if (this._tcontrols) return;

      if (!this.ctrl._debug && !this.ctrl._grid) return;

      this._tcontrols = new TransformControls(this._camera, this._renderer.domElement);
      this._scene.add(this._tcontrols);
      this._tcontrols.attach(this._toplevel);
      //this._tcontrols.setSize( 1.1 );

      window.addEventListener( 'keydown', event => {
         switch ( event.key ) {
         case 'q':
            this._tcontrols.setSpace( this._tcontrols.space === "local" ? "world" : "local" );
            break;
         case 'Control':
            this._tcontrols.setTranslationSnap( Math.ceil( this._overall_size ) / 50 );
            this._tcontrols.setRotationSnap( MathUtils.degToRad( 15 ) );
            break;
         case 't': // Translate
            this._tcontrols.setMode( "translate" );
            break;
         case 'r': // Rotate
            this._tcontrols.setMode( "rotate" );
            break;
         case 's': // Scale
            this._tcontrols.setMode( "scale" );
            break;
         case '+':
            this._tcontrols.setSize(this._tcontrols.size + 0.1);
            break;
         case '-':
            this._tcontrols.setSize(Math.max(this._tcontrols.size - 0.1, 0.1));
            break;
         }
      });
      window.addEventListener( 'keyup', event => {
         if (event.key == 'Control') {
            this._tcontrols.setTranslationSnap(null);
            this._tcontrols.setRotationSnap(null);
         }
      });

      this._tcontrols.addEventListener('change', () => this.render3D(0));
   }

   /** @summary Main function in geometry creation loop
     * @desc Returns:
     * - false when nothing todo
     * - true if one could perform next action immediately
     * - 1 when call after short timeout required
     * - 2 when call must be done from processWorkerReply */
   nextDrawAction() {

      if (!this._clones || this.isStage(stageInit)) return false;

      if (this.isStage(stageCollect)) {

         if (this._geom_viewer) {
            this._draw_all_nodes = false;
            this.changeStage(stageAnalyze);
            return true;
         }

         // wait until worker is really started
         if (this.ctrl.use_worker > 0) {
            if (!this._worker) { this.startWorker(); return 1; }
            if (!this._worker_ready) return 1;
         }

         // first copy visibility flags and check how many unique visible nodes exists
         let numvis = this._first_drawing ? this._clones.countVisibles() : 0,
             matrix = null, frustum = null;

         if (!numvis)
            numvis = this._clones.markVisibles(false, false, !!this.geo_manager && !this.ctrl.showtop);

         if (this.ctrl.select_in_view && !this._first_drawing) {
            // extract camera projection matrix for selection

            matrix = createProjectionMatrix(this._camera);

            frustum = createFrustum(matrix);

            // check if overall bounding box seen
            if (frustum.CheckBox(this.getGeomBoundingBox(this._toplevel))) {
               matrix = null; // not use camera for the moment
               frustum = null;
            }
         }

         this._current_face_limit = this.ctrl.maxlimit;
         if (matrix) this._current_face_limit *= 1.25;

         // here we decide if we need worker for the drawings
         // main reason - too large geometry and large time to scan all camera positions
         let need_worker = !isBatchMode() && browser.isChrome && ((numvis > 10000) || (matrix && (this._clones.scanVisible() > 1e5)));

         // worker does not work when starting from file system
         if (need_worker && source_dir.indexOf("file://")==0) {
            console.log('disable worker for jsroot from file system');
            need_worker = false;
         }

         if (need_worker && !this._worker && (this.ctrl.use_worker >= 0))
            this.startWorker(); // we starting worker, but it may not be ready so fast

         if (!need_worker || !this._worker_ready) {
            let res = this._clones.collectVisibles(this._current_face_limit, frustum);
            this._new_draw_nodes = res.lst;
            this._draw_all_nodes = res.complete;
            this.changeStage(stageAnalyze);
            return true;
         }

         let job = {
            collect: this._current_face_limit,   // indicator for the command
            flags: this._clones.getVisibleFlags(),
            matrix: matrix ? matrix.elements : null
         };

         this.submitToWorker(job);

         this.changeStage(stageWorkerCollect);

         return 2; // we now waiting for the worker reply
      }

      if (this.isStage(stageWorkerCollect)) {
         // do nothing, we are waiting for worker reply
         return 2;
      }

      if (this.isStage(stageAnalyze)) {
         // here we merge new and old list of nodes for drawing,
         // normally operation is fast and can be implemented with one c

         if (this._new_append_nodes) {

            this._new_draw_nodes = this._draw_nodes.concat(this._new_append_nodes);

            delete this._new_append_nodes;

         } else if (this._draw_nodes) {

            let del;
            if (this._geom_viewer)
               del = this._draw_nodes;
            else
               del = this._clones.mergeVisibles(this._new_draw_nodes, this._draw_nodes);

            // remove should be fast, do it here
            for (let n = 0; n < del.length; ++n)
               this._clones.createObject3D(del[n].stack, this._toplevel, 'delete_mesh');

            if (del.length > 0)
               this.drawing_log = `Delete ${del.length} nodes`;
         }

         this._draw_nodes = this._new_draw_nodes;
         delete this._new_draw_nodes;
         this.changeStage(stageCollShapes);
         return true;
      }

      if (this.isStage(stageCollShapes)) {

         // collect shapes
         let shapes = this._clones.collectShapes(this._draw_nodes);

         // merge old and new list with produced shapes
         this._build_shapes = this._clones.mergeShapesLists(this._build_shapes, shapes);

         this.changeStage(stageStartBuild);
         return true;
      }

      if (this.isStage(stageStartBuild)) {
         // this is building of geometries,
         // one can ask worker to build them or do it ourself

         if (this.canSubmitToWorker()) {
            let job = { limit: this._current_face_limit, shapes: [] }, cnt = 0;
            for (let n = 0; n < this._build_shapes.length; ++n) {
               let cl = null, item = this._build_shapes[n];
               // only submit not-done items
               if (item.ready || item.geom) {
                  // this is place holder for existing geometry
                  cl = { id: item.id, ready: true, nfaces: countGeometryFaces(item.geom), refcnt: item.refcnt };
               } else {
                  cl = clone(item, null, true);
                  cnt++;
               }

               job.shapes.push(cl);
            }

            if (cnt > 0) {
               /// only if some geom missing, submit job to the worker
               this.submitToWorker(job);
               this.changeStage(stageWorkerBuild);
               return 2;
            }
         }

         this.changeStage(stageBuild);
      }

      if (this.isStage(stageWorkerBuild)) {
         // waiting shapes from the worker, worker should activate our code
         return 2;
      }

      if (this.isStage(stageBuild) || this.isStage(stageBuildReady)) {

         if (this.isStage(stageBuild)) {
            // building shapes
            let res = this._clones.buildShapes(this._build_shapes, this._current_face_limit, 500);
            if (res.done) {
               this.ctrl.info.num_shapes = this._build_shapes.length;
               this.changeStage(stageBuildReady);
            } else {
               this.ctrl.info.num_shapes = res.shapes;
               this.drawing_log = `Creating: ${res.shapes} / ${this._build_shapes.length} shapes,  ${res.faces} faces`;
               if (res.notusedshapes < 30) return true;
            }
         }

         // final stage, create all meshes

         let tm0 = new Date().getTime(), ready = true,
             toplevel = this.ctrl.project ? this._full_geom : this._toplevel;

         for (let n = 0; n < this._draw_nodes.length; ++n) {
            let entry = this._draw_nodes[n];
            if (entry.done) continue;

            /// shape can be provided with entry itself
            let shape = entry.server_shape || this._build_shapes[entry.shapeid];
            if (!shape.ready) {
               if (this.isStage(stageBuildReady)) console.warn('shape marked as not ready when should');
               ready = false;
               continue;
            }

            entry.done = true;
            shape.used = true; // indicate that shape was used in building

            if (this.createEntryMesh(entry, shape, toplevel)) {
               this.ctrl.info.num_meshes++;
               this.ctrl.info.num_faces += shape.nfaces;
            }

            let tm1 = new Date().getTime();
            if (tm1 - tm0 > 500) { ready = false; break; }
         }

         if (ready) {
            if (this.ctrl.project) {
               this.changeStage(stageBuildProj);
               return true;
            }
            this.changeStage(stageInit);
            return false;
         }

         if (!this.isStage(stageBuild))
            this.drawing_log = `Building meshes ${this.ctrl.info.num_meshes} / ${this.ctrl.info.num_faces}`;
         return true;
      }

      if (this.isStage(stageWaitMain)) {
         // wait for main painter to be ready

         if (!this._main_painter) {
            this.changeStage(stageInit, "Lost main painter");
            return false;
         }
         if (!this._main_painter._drawing_ready) return 1;

         this.changeStage(stageBuildProj); // just do projection
      }

      if (this.isStage(stageBuildProj)) {
         this.doProjection();
         this.changeStage(stageInit);
         return false;
      }

      console.error(`never come here, stage ${this.drawing_stage}`);

      return false;
   }

   /** @summary Insert appropriate mesh for given entry */
   createEntryMesh(entry, shape, toplevel) {

      if (!shape.geom || (shape.nfaces === 0)) {
         // node is visible, but shape does not created
         this._clones.createObject3D(entry.stack, toplevel, 'delete_mesh');
         return false;
      }

      // workaround for the TGeoOverlap, where two branches should get predefined color
      if (this._splitColors && entry.stack) {
         if (entry.stack[0] === 0)
            entry.custom_color = "green";
         else if (entry.stack[0] === 1)
            entry.custom_color = "blue";
      }

      let prop = this._clones.getDrawEntryProperties(entry, getRootColors()),
          obj3d = this._clones.createObject3D(entry.stack, toplevel, this.ctrl),
          matrix = obj3d.absMatrix || obj3d.matrixWorld, mesh;

      prop.material.wireframe = this.ctrl.wireframe;

      prop.material.side = this.ctrl.bothSides ? DoubleSide : FrontSide;

      if (matrix.determinant() > -0.9) {
         mesh = new Mesh(shape.geom, prop.material);
      } else {
         mesh = createFlippedMesh(shape, prop.material);
      }

      obj3d.add(mesh);

      if (obj3d.absMatrix) {
         mesh.matrix.copy(obj3d.absMatrix);
         mesh.matrix.decompose(mesh.position, mesh.quaternion, mesh.scale);
         mesh.updateMatrixWorld();
      }

      // keep full stack of nodes
      mesh.stack = entry.stack;
      mesh.renderOrder = this._clones.maxdepth - entry.stack.length; // order of transparency handling

      // keep hierarchy level
      mesh.$jsroot_order = obj3d.$jsroot_depth;

      // set initial render order, when camera moves, one must refine it
      //mesh.$jsroot_order = mesh.renderOrder =
      //   this._clones.maxdepth - ((obj3d.$jsroot_depth !== undefined) ? obj3d.$jsroot_depth : entry.stack.length);

      if (this.ctrl._debug || this.ctrl._full) {
         let wfg = new WireframeGeometry( mesh.geometry ),
             wfm = new LineBasicMaterial({ color: prop.fillcolor, linewidth: prop.linewidth || 1 }),
             helper = new LineSegments(wfg, wfm);
         obj3d.add(helper);
      }

      if (this.ctrl._bound || this.ctrl._full) {
         let boxHelper = new BoxHelper( mesh );
         obj3d.add( boxHelper );
      }

      return true;
   }

   /** @summary used by geometry viewer to show more nodes
     * @desc These nodes excluded from selection logic and always inserted into the model
     * Shape already should be created and assigned to the node */
   appendMoreNodes(nodes, from_drawing) {

      if (!this.isStage(stageInit) && !from_drawing) {
         this._provided_more_nodes = nodes;
         return;
      }

      // delete old nodes
      if (this._more_nodes)
         for (let n = 0; n < this._more_nodes.length; ++n) {
            let entry = this._more_nodes[n],
                obj3d = this._clones.createObject3D(entry.stack, this._toplevel, 'delete_mesh');
            disposeThreejsObject(obj3d);
            cleanupShape(entry.server_shape);
            delete entry.server_shape;
         }

      delete this._more_nodes;

      if (!nodes) return;

      let real_nodes = [];

      for (let k = 0; k < nodes.length; ++k) {
         let entry = nodes[k],
             shape = entry.server_shape;
         if (!shape?.ready) continue;

         entry.done = true;
         shape.used = true; // indicate that shape was used in building

         if (this.createEntryMesh(entry, shape, this._toplevel))
            real_nodes.push(entry);
      }

      // remember additional nodes only if they include shape - otherwise one can ignore them
      if (real_nodes.length > 0)
         this._more_nodes = real_nodes;

      if (!from_drawing) this.render3D();
   }

   /** @summary Returns hierarchy of 3D objects used to produce projection.
     * @desc Typically external master painter is used, but also internal data can be used */
   getProjectionSource() {
      if (this._clones_owner)
         return this._full_geom;
      if (!this._main_painter) {
         console.warn('MAIN PAINTER DISAPPER');
         return null;
      }
      if (!this._main_painter._drawing_ready) {
         console.warn('MAIN PAINTER NOT READY WHEN DO PROJECTION');
         return null;
      }
      return this._main_painter._toplevel;
   }

   /** @summary Calculate geometry bounding box */
   getGeomBoundingBox(topitem, scalar) {
      let box3 = new Box3(), check_any = !this._clones;

      if (!topitem) {
         box3.min.x = box3.min.y = box3.min.z = -1;
         box3.max.x = box3.max.y = box3.max.z = 1;
         return box3;
      }

      box3.makeEmpty();

      topitem.traverse(mesh => {
         if (check_any || (mesh.stack && (mesh instanceof Mesh)) ||
             (mesh.main_track && (mesh instanceof LineSegments)))
            getBoundingBox(mesh, box3);
      });

      if (scalar !== undefined) box3.expandByVector(box3.getSize(new Vector3()).multiplyScalar(scalar));

      return box3;
   }

   /** @summary Create geometry projection */
   doProjection() {
      let toplevel = this.getProjectionSource();

      if (!toplevel) return false;

      disposeThreejsObject(this._toplevel, true);

      // let axis = this.ctrl.project;

      if (this.ctrl.projectPos === undefined) {

         let bound = this.getGeomBoundingBox(toplevel),
             min = bound.min[this.ctrl.project], max = bound.max[this.ctrl.project],
             mean = (min+max)/2;

         if ((min < 0) && (max > 0) && (Math.abs(mean) < 0.2*Math.max(-min,max))) mean = 0; // if middle is around 0, use 0

         this.ctrl.projectPos = mean;
      }

      toplevel.traverse(mesh => {
         if (!(mesh instanceof Mesh) || !mesh.stack) return;

         let geom2 = projectGeometry(mesh.geometry, mesh.parent.absMatrix || mesh.parent.matrixWorld, this.ctrl.project, this.ctrl.projectPos, mesh._flippedMesh);

         if (!geom2) return;

         let mesh2 = new Mesh(geom2, mesh.material.clone());

         this._toplevel.add(mesh2);

         mesh2.stack = mesh.stack;
      });

      return true;
   }

   /** @summary Should be invoked when light configuration changed */
   changedLight(box) {
      if (!this._camera) return;

      let need_render = !box;

      if (!box) box = this.getGeomBoundingBox(this._toplevel);

      let sizex = box.max.x - box.min.x,
          sizey = box.max.y - box.min.y,
          sizez = box.max.z - box.min.z,
          plights = [], p = this.ctrl.light.power;

      if (p === undefined) p = 1;

      if (this._camera._lights != this.ctrl.light.kind) {
         // remove all childs and recreate only necessary lights
         disposeThreejsObject(this._camera, true);

         this._camera._lights = this.ctrl.light.kind;

         switch (this._camera._lights) {
            case "ambient" : this._camera.add(new AmbientLight(0xefefef, p)); break;
            case "hemisphere" : this._camera.add(new HemisphereLight(0xffffbb, 0x080820, p)); break;
            default: // 6 point lights
               for (let n = 0; n < 6; ++n)
                  this._camera.add(new PointLight(0xefefef, p));
         }
      }

      for (let k = 0; k < this._camera.children.length; ++k) {
         let light = this._camera.children[k], enabled = false;
         if (light.isAmbientLight || light.isHemisphereLight) {
            light.intensity = p;
            continue;
         }

         if (!light.isPointLight) continue;
         switch (k) {
            case 0: light.position.set(sizex/5, sizey/5, sizez/5); enabled = this.ctrl.light.specular; break;
            case 1: light.position.set(0, 0, sizez/2); enabled = this.ctrl.light.front; break;
            case 2: light.position.set(0, 2*sizey, 0); enabled = this.ctrl.light.top; break;
            case 3: light.position.set(0, -2*sizey, 0); enabled = this.ctrl.light.bottom; break;
            case 4: light.position.set(-2*sizex, 0, 0); enabled = this.ctrl.light.left; break;
            case 5: light.position.set(2*sizex, 0, 0); enabled = this.ctrl.light.right; break;
         }
         light.power = enabled ? p*Math.PI*4 : 0;
         if (enabled) plights.push(light);
      }

      // keep light power of all soources constant
      plights.forEach(ll => { ll.power = p*4*Math.PI/plights.length; });

      if (need_render) this.render3D();
   }

   /** @summary Create configured camera */
   createCamera() {

      if (this._camera) {
          this._scene.remove(this._camera);
          disposeThreejsObject(this._camera);
          delete this._camera;
       }

      if (this.ctrl.ortho_camera) {
         this._camera = new OrthographicCamera(-this._scene_width/2, this._scene_width/2, this._scene_height/2, -this._scene_height/2, 1, 10000);
      } else {
         this._camera = new PerspectiveCamera(25, this._scene_width / this._scene_height, 1, 10000);
         this._camera.up = this.ctrl._yup ? new Vector3(0,1,0) : new Vector3(0,0,1);
      }

      // Light - add default point light, adjust later
      let light = new PointLight(0xefefef, 1);
      light.position.set(10, 10, 10);
      this._camera.add( light );

      this._scene.add(this._camera);
   }

   /** @summary Create special effects */
   createSpecialEffects() {
      // Smooth Lighting Shader (Screen Space Ambient Occlusion)
      // http://threejs.org/examples/webgl_postprocessing_ssao.html

      if (this._webgl && (this.ctrl.ssao.enabled || this.ctrl.outline)) {

         if (this.ctrl.outline && (typeof this.createOutline == "function")) {
            this._effectComposer = new EffectComposer( this._renderer );
            this._effectComposer.addPass(new RenderPass( this._scene, this._camera));
            this.createOutline(this._scene_width, this._scene_height);
         } else if (this.ctrl.ssao.enabled) {
            this.createSSAO();
         }
      }

      if (this._webgl && this.ctrl.bloom.enabled)
         this.createBloom();
   }

   /** @summary Initial scene creation */
   createScene(w, h) {
      // three.js 3D drawing
      this._scene = new Scene();
      this._scene.fog = new Fog(0xffffff, 1, 10000);
      this._scene.overrideMaterial = new MeshLambertMaterial({ color: 0x7000ff, vertexColors: false, transparent: true, opacity: 0.2, depthTest: false });

      this._scene_width = w;
      this._scene_height = h;

      this.createCamera();

      this._selected_mesh = null;

      this._overall_size = 10;

      this._toplevel = new Object3D();

      this._scene.add(this._toplevel);

      return createRender3D(w, h, this.options.Render3D, { antialias: true, logarithmicDepthBuffer: false, preserveDrawingBuffer: true }).then(r => {

         this._renderer = r;

         this._webgl = (this._renderer.jsroot_render3d === constants.Render3D.WebGL);

         if (this._renderer.setPixelRatio && !isNodeJs())
            this._renderer.setPixelRatio(window.devicePixelRatio);
         this._renderer.setSize(w, h, !this._fit_main_area);
         this._renderer.localClippingEnabled = true;

         this._renderer.setClearColor(this.ctrl.background, 1);

         if (this._fit_main_area && this._webgl) {
            this._renderer.domElement.style.width = "100%";
            this._renderer.domElement.style.height = "100%";
            let main = this.selectDom();
            if (main.style('position')=='static') main.style('position','relative');
         }

         this._animating = false;

         // Clipping Planes

         this.ctrl.bothSides = false; // which material kind should be used
         this._clipPlanes = [ new Plane(new Vector3(1, 0, 0), 0),
                              new Plane(new Vector3(0, this.ctrl._yup ? -1 : 1, 0), 0),
                              new Plane(new Vector3(0, 0, this.ctrl._yup ? 1 : -1), 0) ];

         this.createSpecialEffects();

         if (this._fit_main_area && !this._webgl) {
            // create top-most SVG for geomtery drawings
            let doc = getDocument(),
                svg = doc.createElementNS("http://www.w3.org/2000/svg", "svg");
            svg.setAttribute("width", w);
            svg.setAttribute("height", h);
            svg.appendChild(this._renderer.jsroot_dom);
            return svg;
         }

         return this._renderer.jsroot_dom;
      });
   }

   /** @summary Start geometry drawing */
   startDrawGeometry(force) {

      if (!force && !this.isStage(stageInit)) {
         this._draw_nodes_again = true;
         return;
      }

      if (this._clones_owner && this._clones)
         this._clones.setDefaultColors(this.ctrl.dflt_colors);

      this._startm = new Date().getTime();
      this._last_render_tm = this._startm;
      this._last_render_meshes = 0;
      this.changeStage(stageCollect);
      this._drawing_ready = false;
      this.ctrl.info.num_meshes = 0;
      this.ctrl.info.num_faces = 0;
      this.ctrl.info.num_shapes = 0;
      this._selected_mesh = null;

      if (this.ctrl.project) {
         if (this._clones_owner) {
            if (this._full_geom) {
               this.changeStage(stageBuildProj);
            } else {
               this._full_geom = new Object3D();
            }
         } else {
            this.changeStage(stageWaitMain);
         }
      }

      delete this._last_manifest;
      delete this._last_hidden; // clear list of hidden objects

      delete this._draw_nodes_again; // forget about such flag

      this.continueDraw();
   }

   /** @summary reset all kind of advanced features like SSAO or depth test changes */
   resetAdvanced() {
      this.ctrl.ssao.kernelRadius = 16;
      this.ctrl.ssao.output = SSAOPass.OUTPUT.Default;

      this.ctrl.depthTest = true;
      this.ctrl.clipIntersect = true;
      this.ctrl.depthMethod = "ray";

      this.changedDepthMethod("norender");
      this.changedDepthTest();
   }

   /** @summary returns maximal dimension */
   getOverallSize(force) {
      if (!this._overall_size || force) {
         let box = this.getGeomBoundingBox(this._toplevel);

         // if detect of coordinates fails - ignore
         if (!Number.isFinite(box.min.x)) return 1000;

         let sizex = box.max.x - box.min.x,
             sizey = box.max.y - box.min.y,
             sizez = box.max.z - box.min.z;

         this._overall_size = 2 * Math.max(sizex, sizey, sizez);
      }

      return this._overall_size;
   }

   /** @summary Create png image with drawing snapshot. */
   createSnapshot(filename) {
      if (!this._renderer) return;
      this.render3D(0);
      let dataUrl = this._renderer.domElement.toDataURL("image/png");
      if (filename === "asis") return dataUrl;
      dataUrl.replace("image/png", "image/octet-stream");
      let doc = getDocument(),
          link = doc.createElement('a');
      if (typeof link.download === 'string') {
         doc.body.appendChild(link); //Firefox requires the link to be in the body
         link.download = filename || "geometry.png";
         link.href = dataUrl;
         link.click();
         doc.body.removeChild(link); //remove the link when done
      }
   }

   /** @summary Returns url parameters defining camera position.
     * @desc It is zoom, roty, rotz parameters
     * These parameters applied from default position which is shift along X axis */
   produceCameraUrl(prec) {

      if (!this._lookat || !this._camera0pos || !this._camera || !this.ctrl) return;

      let pos1 = new Vector3().add(this._camera0pos).sub(this._lookat),
          pos2 = new Vector3().add(this._camera.position).sub(this._lookat),
          len1 = pos1.length(), len2 = pos2.length(),
          zoom = this.ctrl.zoom * len2 / len1 * 100;

      if (zoom < 1) zoom = 1; else if (zoom>10000) zoom = 10000;

      pos1.normalize();
      pos2.normalize();

      let quat = new Quaternion(), euler = new Euler();

      quat.setFromUnitVectors(pos1, pos2);
      euler.setFromQuaternion(quat, "YZX");

      let roty = euler.y / Math.PI * 180,
          rotz = euler.z / Math.PI * 180;

      if (roty < 0) roty += 360;
      if (rotz < 0) rotz += 360;

      return "roty" + roty.toFixed(prec || 0) + ",rotz" + rotz.toFixed(prec || 0) + ",zoom" + zoom.toFixed(prec || 0);
   }

   /** @summary Calculates current zoom factor */
   calculateZoom() {
      if (this._camera0pos && this._camera && this._lookat) {
         let pos1 = new Vector3().add(this._camera0pos).sub(this._lookat),
             pos2 = new Vector3().add(this._camera.position).sub(this._lookat);
         return pos2.length() / pos1.length();
      }

      return 0;
   }

   /** @summary Place camera to default position */
   adjustCameraPosition(first_time, keep_zoom) {
      if (!this._toplevel) return;

      let box = this.getGeomBoundingBox(this._toplevel);

      // if detect of coordinates fails - ignore
      if (!Number.isFinite(box.min.x)) return;

      let sizex = box.max.x - box.min.x,
          sizey = box.max.y - box.min.y,
          sizez = box.max.z - box.min.z,
          midx = (box.max.x + box.min.x)/2,
          midy = (box.max.y + box.min.y)/2,
          midz = (box.max.z + box.min.z)/2;

      this._overall_size = 2 * Math.max(sizex, sizey, sizez);

      this._camera.near = this._overall_size / 350;
      this._camera.far = this._overall_size * 12;
      this._scene.fog.near = this._overall_size * 2;
      this._scene.fog.far = this._overall_size * 12;

      if (first_time)
         for (let naxis = 0; naxis < 3; ++naxis) {
            let cc = this.ctrl.clip[naxis];
            cc.min = box.min[cc.name];
            cc.max = box.max[cc.name];
            let sz = cc.max - cc.min;
            cc.max += sz*0.01;
            cc.min -= sz*0.01;
            if (!cc.value)
               cc.value = (cc.min + cc.max) / 2;
            else if (cc.value < cc.min)
               cc.value = cc.min;
            else if (cc.value > cc.max)
               cc.value = cc.max;
         }

      if (this.ctrl.ortho_camera) {
         this._camera.left = box.min.x;
         this._camera.right = box.max.x;
         this._camera.top = box.max.y;
         this._camera.bottom = box.min.y;
      }

      // this._camera.far = 100000000000;

      this._camera.updateProjectionMatrix();

      let k = 2*this.ctrl.zoom,
          max_all = Math.max(sizex,sizey,sizez);

      if ((this.ctrl.rotatey || this.ctrl.rotatez) && this.ctrl.can_rotate) {

         let prev_zoom = this.calculateZoom();
         if (keep_zoom && prev_zoom) k = 2*prev_zoom;

         let euler = new Euler( 0, this.ctrl.rotatey/180.*Math.PI, this.ctrl.rotatez/180.*Math.PI, 'YZX' );

         this._camera.position.set(-k*max_all, 0, 0);
         this._camera.position.applyEuler(euler);
         this._camera.position.add(new Vector3(midx,midy,midz));

         if (keep_zoom && prev_zoom) {
            let actual_zoom = this.calculateZoom();
            k *= prev_zoom/actual_zoom;

            this._camera.position.set(-k*max_all, 0, 0);
            this._camera.position.applyEuler(euler);
            this._camera.position.add(new Vector3(midx,midy,midz));
         }

      } else if (this.ctrl.ortho_camera) {
         this._camera.position.set(midx, midy, Math.max(sizex,sizey));
      } else if (this.ctrl.project) {
         switch (this.ctrl.project) {
            case 'x': this._camera.position.set(k*1.5*Math.max(sizey,sizez), 0, 0); break;
            case 'y': this._camera.position.set(0, k*1.5*Math.max(sizex,sizez), 0); break;
            case 'z': this._camera.position.set(0, 0, k*1.5*Math.max(sizex,sizey)); break;
         }
      } else if (this.ctrl._yup) {
         this._camera.position.set(midx-k*Math.max(sizex,sizez), midy+k*sizey, midz-k*Math.max(sizex,sizez));
      } else {
         this._camera.position.set(midx-k*Math.max(sizex,sizey), midy-k*Math.max(sizex,sizey), midz+k*sizez);
      }

      this._lookat = new Vector3(midx, midy, midz);
      this._camera0pos = new Vector3(-2*max_all, 0, 0); // virtual 0 position, where rotation starts
      this._camera.lookAt(this._lookat);

      this.changedLight(box);

      if (this._controls) {
         this._controls.target.copy(this._lookat);
         this._controls.update();
      }

      // recheck which elements to draw
      if (this.ctrl.select_in_view)
         this.startDrawGeometry();
   }

   /** @summary Specifies camera position */
   setCameraPosition(rotatey, rotatez, zoom) {
      if (!this.ctrl) return;
      this.ctrl.rotatey = rotatey || 0;
      this.ctrl.rotatez = rotatez || 0;
      let preserve_zoom = false;
      if (zoom && Number.isFinite(zoom)) {
         this.ctrl.zoom = zoom;
      } else {
         preserve_zoom = true;
      }
      this.adjustCameraPosition(false, preserve_zoom);
   }

   /** @summary focus on item */
   focusOnItem(itemname) {

      if (!itemname || !this._clones) return;

      let stack = this._clones.findStackByName(itemname);

      if (!stack) return;

      let info = this._clones.resolveStack(stack, true);

      this.focusCamera( info, false );
   }

   /** @summary focus camera on speicifed position */
   focusCamera( focus, autoClip ) {

      if (this.ctrl.project || this.ctrl.ortho_camera)
         return this.adjustCameraPosition();

      let box = new Box3();
      if (focus === undefined) {
         box = this.getGeomBoundingBox(this._toplevel);
      } else if (focus instanceof Mesh) {
         box.setFromObject(focus);
      } else {
         let center = new Vector3().setFromMatrixPosition(focus.matrix),
             node = focus.node,
             halfDelta = new Vector3( node.fDX, node.fDY, node.fDZ ).multiplyScalar(0.5);
         box.min = center.clone().sub(halfDelta);
         box.max = center.clone().add(halfDelta);
      }

      let sizex = box.max.x - box.min.x,
          sizey = box.max.y - box.min.y,
          sizez = box.max.z - box.min.z,
          midx = (box.max.x + box.min.x)/2,
          midy = (box.max.y + box.min.y)/2,
          midz = (box.max.z + box.min.z)/2;

      let position;
      if (this.ctrl._yup)
         position = new Vector3(midx-2*Math.max(sizex,sizez), midy+2*sizey, midz-2*Math.max(sizex,sizez));
      else
         position = new Vector3(midx-2*Math.max(sizex,sizey), midy-2*Math.max(sizex,sizey), midz+2*sizez);

      let target = new Vector3(midx, midy, midz),
          oldTarget = this._controls.target,
          // probably, reduce number of frames
          frames = 50, step = 0,
          // Amount to change camera position at each step
          posIncrement = position.sub(this._camera.position).divideScalar(frames),
          // Amount to change "lookAt" so it will end pointed at target
          targetIncrement = target.sub(oldTarget).divideScalar(frames);

      autoClip = autoClip && this._webgl;

      // Automatic Clipping
      if (autoClip) {
         for (let axis = 0; axis < 3; ++axis) {
            let cc = this.ctrl.clip[axis];
            if (!cc.enabled) { cc.value = cc.min; cc.enabled = true; }
            cc.inc = ((cc.min + cc.max) / 2 - cc.value) / frames;
         }
         this.updateClipping();
      }

      this._animating = true;

      // Interpolate //

      let animate = () => {
         if (this._animating === undefined) return;

         if (this._animating) {
            requestAnimationFrame( animate );
         } else {
            if (!this._geom_viewer)
               this.startDrawGeometry();
         }
         let smoothFactor = -Math.cos( ( 2.0 * Math.PI * step ) / frames ) + 1.0;
         this._camera.position.add( posIncrement.clone().multiplyScalar( smoothFactor ) );
         oldTarget.add( targetIncrement.clone().multiplyScalar( smoothFactor ) );
         this._lookat = oldTarget;
         this._camera.lookAt( this._lookat );
         this._camera.updateProjectionMatrix();

         let tm1 = new Date().getTime();
         if (autoClip) {
            for (let axis = 0; axis < 3; ++axis)
               this.ctrl.clip[axis].value += this.ctrl.clip[axis].inc * smoothFactor;
            this.updateClipping();
         } else {
            this.render3D(0);
         }
         let tm2 = new Date().getTime();
         if ((step==0) && (tm2-tm1 > 200)) frames = 20;
         step++;
         this._animating = step < frames;
      };

      animate();

   //   this._controls.update();
   }

   /** @summary actiavte auto rotate */
   autorotate(speed) {

      let rotSpeed = (speed === undefined) ? 2.0 : speed,
          last = new Date();

      const animate = () => {
         if (!this._renderer || !this.ctrl) return;

         let current = new Date();

         if ( this.ctrl.rotate ) requestAnimationFrame( animate );

         if (this._controls) {
            this._controls.autoRotate = this.ctrl.rotate;
            this._controls.autoRotateSpeed = rotSpeed * ( current.getTime() - last.getTime() ) / 16.6666;
            this._controls.update();
         }
         last = new Date();
         this.render3D(0);
      };

      if (this._webgl) animate();
   }

   /** @summary called at the end of scene drawing */
   completeScene() {

      if ( this.ctrl._debug || this.ctrl._grid ) {
         if ( this.ctrl._full ) {
            let boxHelper = new BoxHelper(this._toplevel);
            this._scene.add( boxHelper );
         }
         this._scene.add( new AxesHelper( 2 * this._overall_size ) );
         this._scene.add( new GridHelper( Math.ceil( this._overall_size), Math.ceil( this._overall_size ) / 50 ) );
         this.helpText("<font face='verdana' size='1' color='red'><center>Transform Controls<br>" +
               "'T' translate | 'R' rotate | 'S' scale<br>" +
               "'+' increase size | '-' decrease size<br>" +
               "'W' toggle wireframe/solid display<br>"+
         "keep 'Ctrl' down to snap to grid</center></font>");
      }
   }

   /** @summary Drawing with "count" option
     * @desc Scans hieararchy and check for unique nodes
     * @returns {Promise} with object drawing ready */
   drawCount(unqievis, clonetm) {

      const makeTime = tm => (isBatchMode() ? "anytime" : tm.toString()) + " ms";

      let res = [ 'Unique nodes: ' + this._clones.nodes.length,
                  'Unique visible: ' + unqievis,
                  'Time to clone: ' + makeTime(clonetm) ];

      // need to fill cached value line numvischld
      this._clones.scanVisible();

      let nshapes = 0, arg = {
         clones: this._clones,
         cnt: [],
         func(node) {
            if (this.cnt[this.last] === undefined)
               this.cnt[this.last] = 1;
            else
               this.cnt[this.last]++;

            nshapes += countNumShapes(this.clones.getNodeShape(node.id));
            return true;
         }
      };

      let tm1 = new Date().getTime(),
          numvis = this._clones.scanVisible(arg),
          tm2 = new Date().getTime();

      res.push('Total visible nodes: ' + numvis, 'Total shapes: ' + nshapes);

      for (let lvl = 0; lvl < arg.cnt.length; ++lvl) {
         if (arg.cnt[lvl] !== undefined)
            res.push('  lvl' + lvl + ': ' + arg.cnt[lvl]);
      }

      res.push("Time to scan: " + makeTime(tm2-tm1), "", "Check timing for matrix calculations ...");

      let elem = this.selectDom().style('overflow', 'auto');

      if (isBatchMode())
         elem.property("_json_object_", res);
      else
         res.forEach(str => elem.append("p").text(str));

      return new Promise(resolveFunc => {
         setTimeout(() => {
            arg.domatrix = true;
            tm1 = new Date().getTime();
            numvis = this._clones.scanVisible(arg);
            tm2 = new Date().getTime();

            let last_str = "Time to scan with matrix: " + makeTime(tm2-tm1);
            if (isBatchMode())
               res.push(last_str);
            else
               elem.append("p").text(last_str);
            resolveFunc(this);
         }, 100);
      });
   }

   /** @summary Handle drop operation
     * @desc opt parameter can include function name like opt$func_name
     * Such function should be possible to find via {@link findFunction}
     * Function has to return Promise with objects to draw on geometry
     * By default function with name "extract_geo_tracks" is checked
     * @returns {Promise} handling of drop operation */
   performDrop(obj, itemname, hitem, opt) {

      if (obj && (obj.$kind==='TTree')) {
         // drop tree means function call which must extract tracks from provided tree

         let funcname = "extract_geo_tracks";

         if (opt && opt.indexOf("$") > 0) {
            funcname = opt.slice(0, opt.indexOf("$"));
            opt = opt.slice(opt.indexOf("$")+1);
         }

         let func = findFunction(funcname);

         if (!func) return Promise.reject(Error(`Function ${funcname} not found`));

         return func(obj, opt).then(tracks => {
            if (!tracks) return this;

            // FIXME: probably tracks should be remembered?
            return this.drawExtras(tracks, "", false).then(()=> {
               this.updateClipping(true);

               return this.render3D(100);
            });
         });
      }

      return this.drawExtras(obj, itemname).then(is_any => {
         if (!is_any) return this;

         if (hitem) hitem._painter = this; // set for the browser item back pointer

         return this.render3D(100);
      });
   }

   /** @summary function called when mouse is going over the item in the browser */
   mouseOverHierarchy(on, itemname, hitem) {
      if (!this.ctrl) return; // protection for cleaned-up painter

      let obj = hitem._obj;
      if (this.ctrl._debug)
         console.log('Mouse over', on, itemname, (obj ? obj._typename : "---"));

      // let's highlight tracks and hits only for the time being
      if (!obj || (obj._typename !== "TEveTrack" && obj._typename !== "TEvePointSet" && obj._typename !== "TPolyMarker3D")) return;

      this.highlightMesh(null, 0x00ff00, on ? obj : null);
   }

   /** @summary clear extra drawn objects like tracks or hits */
   clearExtras() {
      this.getExtrasContainer("delete");
      delete this._extraObjects; // workaround, later will be normal function
      this.render3D();
   }

   /** @summary Register extra objects like tracks or hits
    * @desc Rendered after main geometry volumes are created
    * Check if object already exists to prevent duplication */
   addExtra(obj, itemname) {
      if (this._extraObjects === undefined)
         this._extraObjects = create("TList");

      if (this._extraObjects.arr.indexOf(obj) >= 0) return false;

      this._extraObjects.Add(obj, itemname);

      delete obj.$hidden_via_menu; // remove previous hidden property

      return true;
   }

   /** @summary manipulate visisbility of extra objects, used for HierarhcyPainter
     * @private */
   extraObjectVisible(hpainter, hitem, toggle) {
      if (!this._extraObjects) return;

      let itemname = hpainter.itemFullName(hitem),
          indx = this._extraObjects.opt.indexOf(itemname);

      if ((indx < 0) && hitem._obj) {
         indx = this._extraObjects.arr.indexOf(hitem._obj);
         // workaround - if object found, replace its name
         if (indx>=0) this._extraObjects.opt[indx] = itemname;
      }

      if (indx < 0) return;

      let obj = this._extraObjects.arr[indx],
          res = obj.$hidden_via_menu ? false : true;

      if (toggle) {
         obj.$hidden_via_menu = res; res = !res;

         let mesh = null;
         // either found painted object or just draw once again
         this._toplevel.traverse(node => { if (node.geo_object === obj) mesh = node; });

         if (mesh) {
            mesh.visible = res;
            this.render3D();
         } else if (res) {
            this.drawExtras(obj, "", false).then(()=> {
               this.updateClipping(true);
               this.render3D();
            });
         }
      }

      return res;
   }

   /** @summary Draw extra object like tracks
     * @returns {Promise} for ready */
   drawExtras(obj, itemname, add_objects) {
      // if object was hidden via menu, do not redraw it with next draw call
      if (!obj?._typename || (!add_objects && obj.$hidden_via_menu))
         return Promise.resolve(false);

      let do_render = false;
      if (add_objects === undefined) {
         add_objects = true;
         do_render = true;
      }

      let promise = false;

      if ((obj._typename === "TList") || (obj._typename === "TObjArray")) {
         if (!obj.arr) return false;
         let parr = [];
         for (let n = 0; n < obj.arr.length; ++n) {
            let sobj = obj.arr[n], sname = obj.opt ? obj.opt[n] : "";
            if (!sname) sname = (itemname || "<prnt>") + "/[" + n + "]";
            parr.push(this.drawExtras(sobj, sname, add_objects));
         }
         promise = Promise.all(parr).then(ress => ress.indexOf(true) >= 0);
      } else if (obj._typename === 'Mesh') {
         // adding mesh as is
         this.addToExtrasContainer(obj);
         promise = Promise.resolve(true);
      } else if (obj._typename === 'TGeoTrack') {
         if (!add_objects || this.addExtra(obj, itemname))
            promise = this.drawGeoTrack(obj, itemname);
      } else if (obj._typename === 'TPolyLine3D') {
         if (!add_objects || this.addExtra(obj, itemname))
            promise = this.drawPolyLine(obj, itemname);
      } else if ((obj._typename === 'TEveTrack') || (obj._typename === 'ROOT::Experimental::TEveTrack')) {
         if (!add_objects || this.addExtra(obj, itemname))
            promise = this.drawEveTrack(obj, itemname);
      } else if ((obj._typename === 'TEvePointSet') || (obj._typename === "ROOT::Experimental::TEvePointSet") || (obj._typename === "TPolyMarker3D")) {
         if (!add_objects || this.addExtra(obj, itemname))
            promise = this.drawHit(obj, itemname);
      } else if ((obj._typename === "TEveGeoShapeExtract") || (obj._typename === "ROOT::Experimental::REveGeoShapeExtract")) {
         if (!add_objects || this.addExtra(obj, itemname))
            promise = this.drawExtraShape(obj, itemname);
      }

      if (!isPromise(promise))
         promise = Promise.resolve(promise);

      if (!do_render)
         return promise;

      return promise.then(is_any => {
         if (!is_any) return false;

         this.updateClipping(true);
         return this.render3D(100);
      });
   }

   /** @summary returns container for extra objects */
   getExtrasContainer(action, name) {
      if (!this._toplevel) return null;

      if (!name) name = "tracks";

      let extras = null, lst = [];
      for (let n = 0; n < this._toplevel.children.length; ++n) {
         let chld = this._toplevel.children[n];
         if (!chld._extras) continue;
         if (action == 'collect') { lst.push(chld); continue; }
         if (chld._extras === name) { extras = chld; break; }
      }

      if (action == 'collect') {
         for (let k = 0; k < lst.length; ++k)
            this._toplevel.remove(lst[k]);
         return lst;
      }

      if (action == "delete") {
         if (extras) this._toplevel.remove(extras);
         disposeThreejsObject(extras);
         return null;
      }

      if ((action !== "get") && !extras) {
         extras = new Object3D();
         extras._extras = name;
         this._toplevel.add(extras);
      }

      return extras;
   }

   /** @summary add object to extras container.
     * @desc If fail, dispore object */
   addToExtrasContainer(obj, name) {
      let container = this.getExtrasContainer("", name);
      if (container) {
         container.add(obj);
      } else {
         console.warn('Fail to add object to extras');
         disposeThreejsObject(obj);
      }
   }

   /** @summary drawing TGeoTrack */
   drawGeoTrack(track, itemname) {
      if (!track || !track.fNpoints) return false;

      let linewidth = browser.isWin ? 1 : (track.fLineWidth || 1), // linew width not supported on windows
          color = getColor(track.fLineColor) || "#ff00ff",
          npoints = Math.round(track.fNpoints/4), // each track point has [x,y,z,t] coordinate
          buf = new Float32Array((npoints-1)*6),
          pos = 0, projv = this.ctrl.projectPos,
          projx = (this.ctrl.project === "x"),
          projy = (this.ctrl.project === "y"),
          projz = (this.ctrl.project === "z");

      for (let k = 0; k < npoints-1; ++k) {
         buf[pos]   = projx ? projv : track.fPoints[k*4];
         buf[pos+1] = projy ? projv : track.fPoints[k*4+1];
         buf[pos+2] = projz ? projv : track.fPoints[k*4+2];
         buf[pos+3] = projx ? projv : track.fPoints[k*4+4];
         buf[pos+4] = projy ? projv : track.fPoints[k*4+5];
         buf[pos+5] = projz ? projv : track.fPoints[k*4+6];
         pos+=6;
      }

      let lineMaterial = new LineBasicMaterial({ color, linewidth }),
          line = createLineSegments(buf, lineMaterial);

      line.defaultOrder = line.renderOrder = 1000000; // to bring line to the front
      line.geo_name = itemname;
      line.geo_object = track;
      line.hightlightWidthScale = 2;

      if (itemname && itemname.indexOf("<prnt>/Tracks")==0)
         line.main_track = true;

      this.addToExtrasContainer(line);

      return true;
   }

   /** @summary drawing TPolyLine3D */
   drawPolyLine(line, itemname) {
      if (!line) return false;

      let linewidth = browser.isWin ? 1 : (line.fLineWidth || 1),
          color = getColor(line.fLineColor) || "#ff00ff",
          npoints = line.fN,
          fP = line.fP,
          buf = new Float32Array((npoints-1)*6),
          pos = 0, projv = this.ctrl.projectPos,
          projx = (this.ctrl.project === "x"),
          projy = (this.ctrl.project === "y"),
          projz = (this.ctrl.project === "z");

      for (let k = 0; k < npoints-1; ++k) {
         buf[pos]   = projx ? projv : fP[k*3];
         buf[pos+1] = projy ? projv : fP[k*3+1];
         buf[pos+2] = projz ? projv : fP[k*3+2];
         buf[pos+3] = projx ? projv : fP[k*3+3];
         buf[pos+4] = projy ? projv : fP[k*3+4];
         buf[pos+5] = projz ? projv : fP[k*3+5];
         pos += 6;
      }

      let lineMaterial = new LineBasicMaterial({ color, linewidth }),
          line3d = createLineSegments(buf, lineMaterial);

      line3d.defaultOrder = line3d.renderOrder = 1000000; // to bring line to the front
      line3d.geo_name = itemname;
      line3d.geo_object = line;
      line3d.hightlightWidthScale = 2;

      this.addToExtrasContainer(line3d);

      return true;
   }

   /** @summary Drawing TEveTrack */
   drawEveTrack(track, itemname) {
      if (!track || (track.fN <= 0)) return false;

      let linewidth = browser.isWin ? 1 : (track.fLineWidth || 1),
          color = getColor(track.fLineColor) || "#ff00ff",
          buf = new Float32Array((track.fN-1)*6), pos = 0,
          projv = this.ctrl.projectPos,
          projx = (this.ctrl.project === "x"),
          projy = (this.ctrl.project === "y"),
          projz = (this.ctrl.project === "z");

      for (let k = 0; k < track.fN-1; ++k) {
         buf[pos]   = projx ? projv : track.fP[k*3];
         buf[pos+1] = projy ? projv : track.fP[k*3+1];
         buf[pos+2] = projz ? projv : track.fP[k*3+2];
         buf[pos+3] = projx ? projv : track.fP[k*3+3];
         buf[pos+4] = projy ? projv : track.fP[k*3+4];
         buf[pos+5] = projz ? projv : track.fP[k*3+5];
         pos+=6;
      }

      let lineMaterial = new LineBasicMaterial({ color, linewidth }),
          line = createLineSegments(buf, lineMaterial);

      line.defaultOrder = line.renderOrder = 1000000; // to bring line to the front
      line.geo_name = itemname;
      line.geo_object = track;
      line.hightlightWidthScale = 2;

      this.addToExtrasContainer(line);

      return true;
   }

   /** @summary Drawing different hits types like TPolyMarker3D */
   drawHit(hit, itemname) {
      if (!hit || !hit.fN || (hit.fN < 0))
         return Promise.resolve(false);

      // make hit size scaling factor of overall geometry size
      // otherwise it is not possible to correctly see hits at all
      let hit_size = Math.max(hit.fMarkerSize * this.getOverallSize() * 0.005, 0.2),
          nhits = hit.fN,
          projv = this.ctrl.projectPos,
          projx = (this.ctrl.project === "x"),
          projy = (this.ctrl.project === "y"),
          projz = (this.ctrl.project === "z"),
          style = hit.fMarkerStyle;

      // FIXME: styles 2 and 4 does not work properly, see Misc/basic3d demo
      // style 4 is very bad for hits representation
      if ((style == 4) || (style == 2)) { style = 7; hit_size *= 1.5; }

      let pnts = new PointsCreator(nhits, this._webgl, hit_size);

      for (let i = 0; i < nhits; i++)
         pnts.addPoint(projx ? projv : hit.fP[i*3],
                       projy ? projv : hit.fP[i*3+1],
                       projz ? projv : hit.fP[i*3+2]);

      return pnts.createPoints({ color: getColor(hit.fMarkerColor) || "#0000ff", style }).then(mesh => {
         mesh.defaultOrder = mesh.renderOrder = 1000000; // to bring points to the front
         mesh.highlightScale = 2;
         mesh.geo_name = itemname;
         mesh.geo_object = hit;
         this.addToExtrasContainer(mesh);
         return true; // indicate that rendering should be done
      });
   }

   /** @summary Draw extra shape on the geometry */
   drawExtraShape(obj, itemname) {
      let mesh = build(obj);
      if (!mesh) return false;

      mesh.geo_name = itemname;
      mesh.geo_object = obj;

      this.addToExtrasContainer(mesh);
      return true;
   }

   /** @summary Serach for specified node
     * @private */
   findNodeWithVolume(name, action, prnt, itemname, volumes) {

      let first_level = false, res = null;

      if (!prnt) {
         prnt = this.getGeometry();
         if (!prnt && (getNodeKind(prnt) !== 0)) return null;
         itemname = this.geo_manager ? prnt.fName : "";
         first_level = true;
         volumes = [];
      } else {
         if (itemname.length>0) itemname += "/";
         itemname += prnt.fName;
      }

      if (!prnt.fVolume || prnt.fVolume._searched) return null;

      if (name.test(prnt.fVolume.fName)) {
         res = action({ node: prnt, item: itemname });
         if (res) return res;
      }

      prnt.fVolume._searched = true;
      volumes.push(prnt.fVolume);

      if (prnt.fVolume.fNodes)
         for (let n = 0, len = prnt.fVolume.fNodes.arr.length; n < len; ++n) {
            res = this.findNodeWithVolume(name, action, prnt.fVolume.fNodes.arr[n], itemname, volumes);
            if (res) break;
         }

      if (first_level)
         for (let n = 0, len = volumes.length; n < len; ++n)
            delete volumes[n]._searched;

      return res;
   }

   /** @summary Process script option - load and execute some gGeoManager-related calls */
   loadMacro(script_name) {

      let result = { obj: this.getGeometry(), prefix: "" };

      if (this.geo_manager)
         result.prefix = result.obj.fName;

      if (!script_name || (script_name.length < 3) || (getNodeKind(result.obj) !== 0))
         return Promise.resolve(result);

      let mgr = {
            GetVolume: name => {
               let regexp = new RegExp("^"+name+"$"),
                   currnode = this.findNodeWithVolume(regexp, arg => arg);

               if (!currnode) console.log('Did not found '+name + ' volume');

               // return proxy object with several methods, typically used in ROOT geom scripts
               return {
                   found: currnode,
                   fVolume: currnode ? currnode.node.fVolume : null,
                   InvisibleAll(flag) {
                      setInvisibleAll(this.fVolume, flag);
                   },
                   Draw() {
                      if (!this.found || !this.fVolume) return;
                      result.obj = this.found.node;
                      result.prefix = this.found.item;
                      console.log('Select volume for drawing', this.fVolume.fName, result.prefix);
                   },
                   SetTransparency(lvl) {
                     if (this.fVolume && this.fVolume.fMedium && this.fVolume.fMedium.fMaterial)
                        this.fVolume.fMedium.fMaterial.fFillStyle = 3000+lvl;
                   },
                   SetLineColor(col) {
                      if (this.fVolume) this.fVolume.fLineColor = col;
                   }
                };
            },

            DefaultColors: () => {
               this.ctrl.dflt_colors = true;
            },

            SetMaxVisNodes: limit => {
               if (!this.ctrl.maxnodes)
                  this.ctrl.maxnodes = pasrseInt(limit) || 0;
            },

            SetVisLevel: limit => {
               if (!this.ctrl.vislevel)
                  this.ctrl.vislevel = parseInt(limit) || 0;
            }
          };

      showProgress('Loading macro ' + script_name);

      return httpRequest(script_name, "text").then(script => {
         let lines = script.split('\n'), indx = 0;

         while (indx < lines.length) {
            let line = lines[indx++].trim();

            if (line.indexOf('//')==0) continue;

            if (line.indexOf('gGeoManager') < 0) continue;
            line = line.replace('->GetVolume','.GetVolume');
            line = line.replace('->InvisibleAll','.InvisibleAll');
            line = line.replace('->SetMaxVisNodes','.SetMaxVisNodes');
            line = line.replace('->DefaultColors','.DefaultColors');
            line = line.replace('->Draw','.Draw');
            line = line.replace('->SetTransparency','.SetTransparency');
            line = line.replace('->SetLineColor','.SetLineColor');
            line = line.replace('->SetVisLevel','.SetVisLevel');
            if (line.indexOf('->') >= 0) continue;

            try {
               let func = new Function('gGeoManager', line);
               func(mgr);
            } catch(err) {
               console.error('Problem by processing ' + line);
            }
         }

         return result;
      }).catch(() => {
         console.error('Fail to load ' + script_name);
         return result;
      });
   }

   /** @summary Assign clones, created outside.
     * @desc Used by geometry painter, where clones are handled by the server */
   assignClones(clones) {
      this._clones_owner = true;
      this._clones = clones;
   }

   /** @summary Prepare drawings
     * @desc Return value used as promise for painter */
   prepareObjectDraw(draw_obj, name_prefix) {

      // if did cleanup - ignore all kind of activity
      if (this.did_cleanup)
         return Promise.resolve(null);

      if (name_prefix == "__geom_viewer_append__") {
         this._new_append_nodes = draw_obj;
         this.ctrl.use_worker = 0;
         this._geom_viewer = true; // indicate that working with geom viewer
      } else if ((name_prefix == "__geom_viewer_selection__") && this._clones) {
         // these are selection done from geom viewer
         this._new_draw_nodes = draw_obj;
         this.ctrl.use_worker = 0;
         this._geom_viewer = true; // indicate that working with geom viewer
      } else if (this._main_painter) {

         this._clones_owner = false;

         this._clones = this._main_painter._clones;

         console.log(`Reuse clones ${this._clones.nodes.length} from main painter`);

      } else if (!draw_obj) {

         this._clones_owner = false;

         this._clones = null;

      } else {

         this._start_drawing_time = new Date().getTime();

         this._clones_owner = true;

         this._clones = new ClonedNodes(draw_obj);

         let lvl = this.ctrl.vislevel, maxnodes = this.ctrl.maxnodes;
         if (this.geo_manager) {
            if (!lvl && this.geo_manager.fVisLevel)
               lvl = this.geo_manager.fVisLevel;
            if (!maxnodes)
               maxnodes = this.geo_manager.fMaxVisNodes;
         }

         this._clones.setVisLevel(lvl);
         this._clones.setMaxVisNodes(maxnodes);

         this._clones.name_prefix = name_prefix;

         let hide_top_volume = !!this.geo_manager && !this.ctrl.showtop,
             uniquevis = this.ctrl.no_screen ? 0 : this._clones.markVisibles(true, false, hide_top_volume);

         if (uniquevis <= 0)
            uniquevis = this._clones.markVisibles(false, false, hide_top_volume);
         else
            uniquevis = this._clones.markVisibles(true, true, hide_top_volume); // copy bits once and use normal visibility bits

         this._clones.produceIdShifts();

         let spent = new Date().getTime() - this._start_drawing_time;

         if (!this._scene)
            console.log(`Creating clones ${this._clones.nodes.length} takes ${spent} ms uniquevis ${uniquevis}`);

         if (this.options._count)
            return this.drawCount(uniquevis, spent);
      }

      let promise = Promise.resolve(true);

      if (!this._scene) {
         this._first_drawing = true;

         this._on_pad = !!this.getPadPainter();

         if (this._on_pad) {
            let size, render3d, fp;
            promise = ensureTCanvas(this,"3d").then(() => {

               fp = this.getFramePainter();

               render3d = getRender3DKind();
               assign3DHandler(fp);
               fp.mode3d = true;

               size = fp.getSizeFor3d(undefined, render3d);

               this._fit_main_area = (size.can3d === -1);

               return this.createScene(size.width, size.height)
                          .then(dom => fp.add3dCanvas(size, dom, render3d === constants.Render3D.WebGL));
            });

         } else {
            // activate worker
            if (this.ctrl.use_worker > 0) this.startWorker();

            assign3DHandler(this);

            let size = this.getSizeFor3d(undefined, getRender3DKind(this.options.Render3D));

            this._fit_main_area = (size.can3d === -1);

            promise = this.createScene(size.width, size.height)
                          .then(dom => this.add3dCanvas(size, dom, this._webgl));
         }
      }

      return promise.then(() => {

         // this is limit for the visible faces, number of volumes does not matter
         if (this._first_drawing)
            this.ctrl.maxlimit = (this._webgl ? 200000 : 100000) * this.ctrl.more;

         // set top painter only when first child exists
         this.setAsMainPainter();

         this.createToolbar();

         // just draw extras and complete drawing if there are no main model
         if (!this._clones)
            return this.completeDraw();

         return new Promise(resolveFunc => {
            this._resolveFunc = resolveFunc;
            this.showDrawInfo("Drawing geometry");
            this.startDrawGeometry(true);
         });
      });
   }

   /** @summary methods show info when first geometry drawing is performed */
   showDrawInfo(msg) {
      if (isBatchMode() || !this._first_drawing || !this._start_drawing_time) return;

      let main = this._renderer.domElement.parentNode,
          info = main.querySelector(".geo_info");

      if (!msg) {
         info.remove();
      } else {
         let spent = (new Date().getTime() - this._start_drawing_time)*1e-3;
         if (!info) {
            info = document.createElement("p");
            info.setAttribute("class", "geo_info");
            info.setAttribute("style",  "position: absolute; text-align: center; vertical-align: middle; top: 45%; left: 40%; color: red; font-size: 150%;");
            main.append(info);
         }
         info.innerHTML = `${msg}, ${spent.toFixed(1)}s`;
      }
   }

   /** @summary Reentrant method to perform geometry drawing step by step */
   continueDraw() {

      // nothing to do - exit
      if (this.isStage(stageInit)) return;

      let tm0 = new Date().getTime(),
          interval = this._first_drawing ? 1000 : 200,
          now = tm0;

      while(true) {

         let res = this.nextDrawAction();

         if (!res) break;

         now = new Date().getTime();

         // stop creation after 100 sec, render as is
         if (now - this._startm > 1e5) {
            this.changeStage(stageInit, "Abort build after 100s");
            break;
         }

         // if we are that fast, do next action
         if ((res === true) && (now - tm0 < interval)) continue;

         if ((now - tm0 > interval) || (res === 1) || (res === 2)) {

            showProgress(this.drawing_log);

            this.showDrawInfo(this.drawing_log);

            if (this._first_drawing && this._webgl && (this._num_meshes - this._last_render_meshes > 100) && (now - this._last_render_tm > 2.5*interval)) {
               this.adjustCameraPosition();
               this.render3D(-1);
               this._last_render_meshes = this.ctrl.info.num_meshes;
            }
            if (res !== 2) setTimeout(() => this.continueDraw(), (res === 1) ? 100 : 1);

            return;
         }
      }

      let take_time = now - this._startm;

      if (this._first_drawing || this._full_redrawing)
         console.log(`Create tm = ${take_time} meshes ${this.ctrl.info.num_meshes} faces ${this.ctrl.info.num_faces}`);

      if (take_time > 300) {
         showProgress('Rendering geometry');
         this.showDrawInfo("Rendering");
         return setTimeout(() => this.completeDraw(true), 10);
      }

      this.completeDraw(true);
   }

   /** @summary Checks camera position and recalculate rendering order if needed
     * @param force - if specified, forces calculations of render order */
   testCameraPosition(force) {
      this._camera.updateMatrixWorld();
      let origin = this._camera.position.clone();

      if (!force && this._last_camera_position) {
         // if camera position does not changed a lot, ignore such change
         let dist = this._last_camera_position.distanceTo(origin);
         if (dist < (this._overall_size || 1000)*1e-4) return;
      }

      this._last_camera_position = origin; // remember current camera position

      if (!this.ctrl.project)
         produceRenderOrder(this._toplevel, origin, this.ctrl.depthMethod, this._clones);
   }

   /** @summary Call 3D rendering of the geometry
     * @param tmout - specifies delay, after which actual rendering will be invoked
     * @param [measure] - when true, for the first time printout rendering time
     * @returns {Promise} when tmout bigger than 0 is specified
     * @desc Timeout used to avoid multiple rendering of the picture when several 3D drawings
     * superimposed with each other. If tmeout<=0, rendering performed immediately
     * Several special values are used:
     *   -1    - force recheck of rendering order based on camera position */
   render3D(tmout, measure) {

      if (!this._renderer) {
         if (!this.did_cleanup)
            console.warn('renderer object not exists - check code');
         else
            console.warn('try to render after cleanup');
         return this;
      }

      let ret_promise = (tmout !== undefined) && (tmout > 0);

      if (tmout === undefined) tmout = 5; // by default, rendering happens with timeout

      if ((tmout > 0) && this._webgl /* && !isBatchMode() */) {
         if (isBatchMode()) tmout = 1; // use minimal timeout in batch mode
         if (ret_promise)
            return new Promise(resolveFunc => {
               if (!this._render_resolveFuncs) this._render_resolveFuncs = [];
               this._render_resolveFuncs.push(resolveFunc);
               if (!this.render_tmout)
                  this.render_tmout = setTimeout(() => this.render3D(0, measure), tmout);
            });

         if (!this.render_tmout)
            this.render_tmout = setTimeout(() => this.render3D(0, measure), tmout);
         return this;
      }

      if (this.render_tmout) {
         clearTimeout(this.render_tmout);
         delete this.render_tmout;
      }

      beforeRender3D(this._renderer);

      let tm1 = new Date();

      this.testCameraPosition(tmout === -1);

      // its needed for outlinePass - do rendering, most consuming time
      if (this._webgl && this._effectComposer && (this._effectComposer.passes.length > 0)) {
         this._effectComposer.render();
      } else if (this._webgl && this._bloomComposer && (this._bloomComposer.passes.length > 0)) {
         this._renderer.clear();
         this._camera.layers.set( _BLOOM_SCENE );
         this._bloomComposer.render();
         this._renderer.clearDepth();
         this._camera.layers.set( _ENTIRE_SCENE );
         this._renderer.render(this._scene, this._camera);
      } else {
    //     this._renderer.logarithmicDepthBuffer = true;
         this._renderer.render(this._scene, this._camera);
      }

      let tm2 = new Date();

      this.last_render_tm = tm2.getTime();

      if ((this.first_render_tm === 0) && measure) {
         this.first_render_tm = tm2.getTime() - tm1.getTime();
         console.log(`three.js r${REVISION}, first render tm = ${this.first_render_tm}`);
      }

      afterRender3D(this._renderer);

      if (this._render_resolveFuncs) {
         this._render_resolveFuncs.forEach(func => func(this));
         delete this._render_resolveFuncs;
      }
   }

   /** @summary Start geo worker */
   startWorker() {

      if (this._worker) return;

      this._worker_ready = false;
      this._worker_jobs = 0; // counter how many requests send to worker

      // TODO: modules not yet working, see https://www.codedread.com/blog/archives/2017/10/19/web-workers-can-be-es6-modules-too/
      this._worker = new Worker(source_dir + "scripts/geoworker.js" /*, { type: "module" } */);

      this._worker.onmessage = e => {

         if (typeof e.data !== 'object') return;

         if ('log' in e.data)
            return console.log(`geo: ${e.data.log}`);

         if ('progress' in e.data)
            return showProgress(e.data.progress);

         e.data.tm3 = new Date().getTime();

         if ('init' in e.data) {
            this._worker_ready = true;
            console.log(`Worker ready: ${e.data.tm3 - e.data.tm0}`);
         } else {
            this.processWorkerReply(e.data);
         }
      };

      // send initialization message with clones
      this._worker.postMessage({
         init: true,   // indicate init command for worker
         browser,
         tm0: new Date().getTime(),
         vislevel: this._clones.getVisLevel(),
         maxvisnodes: this._clones.getMaxVisNodes(),
         clones: this._clones.nodes,
         sortmap: this._clones.sortmap
      });
   }

   /** @summary check if one can submit request to worker
     * @private */
   canSubmitToWorker(force) {
      if (!this._worker) return false;

      return this._worker_ready && ((this._worker_jobs == 0) || force);
   }

   /** @summary submit request to worker
     * @private */
   submitToWorker(job) {
      if (!this._worker) return false;

      this._worker_jobs++;

      job.tm0 = new Date().getTime();

      this._worker.postMessage(job);
   }

   /** @summary process reply from worker
     * @private */
   processWorkerReply(job) {
      this._worker_jobs--;

      if ('collect' in job) {
         this._new_draw_nodes = job.new_nodes;
         this._draw_all_nodes = job.complete;
         this.changeStage(stageAnalyze);
         // invoke methods immediately
         return this.continueDraw();
      }

      if ('shapes' in job) {

         for (let n=0;n<job.shapes.length;++n) {
            let item = job.shapes[n],
                origin = this._build_shapes[n];

            // let shape = this._clones.getNodeShape(item.nodeid);

            if (item.buf_pos && item.buf_norm) {
               if (item.buf_pos.length === 0) {
                  origin.geom = null;
               } else if (item.buf_pos.length !== item.buf_norm.length) {
                  console.error('item.buf_pos',item.buf_pos.length, 'item.buf_norm', item.buf_norm.length);
                  origin.geom = null;
               } else {
                  origin.geom = new BufferGeometry();

                  origin.geom.setAttribute('position', new BufferAttribute(item.buf_pos, 3));
                  origin.geom.setAttribute('normal', new BufferAttribute(item.buf_norm, 3));
               }

               origin.ready = true;
               origin.nfaces = item.nfaces;
            }
         }

         job.tm4 = new Date().getTime();

         this.changeStage(stageBuild); // first check which shapes are used, than build meshes

         // invoke methods immediately
         return this.continueDraw();
      }
   }

   /** @summary start draw geometries on master and all slaves
     * @private */
   testGeomChanges() {
      if (this._main_painter) {
         console.warn('Get testGeomChanges call for slave painter');
         return this._main_painter.testGeomChanges();
      }
      this.startDrawGeometry();
      for (let k = 0; k < this._slave_painters.length; ++k)
         this._slave_painters[k].startDrawGeometry();
   }

   /** @summary Draw axes if configured, otherwise just remove completely
     * @returns {Promise} when norender not specified */
   drawSimpleAxis(norender) {
      this.getExtrasContainer('delete', 'axis');

      if (!this.ctrl._axis)
         return norender ? null : this.render3D();

      let box = this.getGeomBoundingBox(this._toplevel),
          container = this.getExtrasContainer('create', 'axis'),
          text_size = 0.02 * Math.max((box.max.x - box.min.x), (box.max.y - box.min.y), (box.max.z - box.min.z)),
          center = [0,0,0],
          names = ['x','y','z'],
          labels = ['X','Y','Z'],
          colors = ["red","green","blue"],
          ortho = this.ctrl.ortho_camera,
          yup = [this.ctrl._yup, this.ctrl._yup, this.ctrl._yup],
          numaxis = 3;

      if (this.ctrl._axis == 2)
         for (let naxis = 0; naxis < 3; ++naxis) {
            let name = names[naxis];
            if ((box.min[name]<=0) && (box.max[name]>=0)) continue;
            center[naxis] = (box.min[name] + box.max[name])/2;
         }

      // only two dimensions are seen by ortho camera, X draws Z, can be configured better later
      if (this.ctrl.ortho_camera) {
         numaxis = 2;
         labels[0] = labels[2];
         colors[0] = colors[2];
         yup[0] = yup[2];
         ortho = true;
      }

      for (let naxis = 0; naxis < numaxis; ++naxis) {

         let buf = new Float32Array(6),
             color = colors[naxis],
             name = names[naxis];

         const Convert = value => {
            let range = box.max[name] - box.min[name];
            if (range < 2) return value.toFixed(3);
            return (Math.abs(value) > 1e5) ? value.toExponential(3) : Math.round(value).toString();
         };

         let lbl = Convert(box.max[name]);

         buf[0] = box.min.x;
         buf[1] = box.min.y;
         buf[2] = box.min.z;

         buf[3] = box.min.x;
         buf[4] = box.min.y;
         buf[5] = box.min.z;

         switch (naxis) {
           case 0: buf[3] = box.max.x; if (yup[0] && !ortho) lbl = labels[0] + " " + lbl; else lbl += " " + labels[0]; break;
           case 1: buf[4] = box.max.y; if (yup[1]) lbl += " " + labels[1]; else lbl = labels[1] + " " + lbl; break;
           case 2: buf[5] = box.max.z; lbl += " " + labels[2]; break;
         }

         if (this.ctrl._axis == 2)
            for (let k = 0; k < 6; ++k)
               if ((k % 3) !== naxis) buf[k] = center[k%3];

         let lineMaterial = new LineBasicMaterial({ color }),
             mesh = createLineSegments(buf, lineMaterial);

         container.add(mesh);

         let textMaterial = new MeshBasicMaterial({ color, vertexColors: false });

         if ((center[naxis]===0) && (center[naxis] >= box.min[name]) && (center[naxis] <= box.max[name]))
           if ((this.ctrl._axis != 2) || (naxis===0)) {
               let geom = ortho ? new CircleBufferGeometry(text_size*0.25) :
                                  new SphereBufferGeometry(text_size*0.25);
               mesh = new Mesh(geom, textMaterial);
               mesh.translateX((naxis===0) ? center[0] : buf[0]);
               mesh.translateY((naxis===1) ? center[1] : buf[1]);
               mesh.translateZ((naxis===2) ? center[2] : buf[2]);
               container.add(mesh);
           }

         let text3d = new TextGeometry(lbl, { font: HelveticerRegularFont, size: text_size, height: 0, curveSegments: 5 });
         mesh = new Mesh(text3d, textMaterial);
         let textbox = new Box3().setFromObject(mesh);

         mesh.translateX(buf[3]);
         mesh.translateY(buf[4]);
         mesh.translateZ(buf[5]);

         if (yup[naxis]) {
            switch (naxis) {
               case 0:
                  if (!ortho) {
                     mesh.rotateY(Math.PI);
                     mesh.translateX(-textbox.max.x-text_size*0.5);
                  } else {
                     mesh.translateX(text_size*0.5);
                  }
                  mesh.translateY(-textbox.max.y/2);
                  break;
               case 1:
                  if (!ortho) {
                     mesh.rotateX(-Math.PI/2);
                     mesh.rotateY(-Math.PI/2);
                  } else {
                     mesh.rotateZ(Math.PI/2);
                  }
                  mesh.translateX(text_size*0.5);
                  mesh.translateY(-textbox.max.y/2);
                  break;
               case 2: mesh.rotateY(-Math.PI/2); mesh.translateX(text_size*0.5); mesh.translateY(-textbox.max.y/2); break;
           }
         } else {
            switch (naxis) {
               case 0: mesh.rotateX(Math.PI/2); mesh.translateY(-textbox.max.y/2); mesh.translateX(text_size*0.5); break;
               case 1: mesh.rotateX(Math.PI/2); mesh.rotateY(-Math.PI/2); mesh.translateX(-textbox.max.x-text_size*0.5); mesh.translateY(-textbox.max.y/2); break;
               case 2: mesh.rotateX(Math.PI/2); mesh.rotateZ(Math.PI/2); mesh.translateX(text_size*0.5); mesh.translateY(-textbox.max.y/2); break;
            }
         }

         container.add(mesh);

         text3d = new TextGeometry(Convert(box.min[name]), { font: HelveticerRegularFont, size: text_size, height: 0, curveSegments: 5 });

         mesh = new Mesh(text3d, textMaterial);
         textbox = new Box3().setFromObject(mesh);

         mesh.translateX(buf[0]);
         mesh.translateY(buf[1]);
         mesh.translateZ(buf[2]);

         if (yup[naxis]) {
            switch (naxis) {
               case 0:
                  if (!ortho) {
                     mesh.rotateY(Math.PI);
                     mesh.translateX(text_size*0.5);
                  } else {
                     mesh.translateX(-textbox.max.x-text_size*0.5);
                  }
                  mesh.translateY(-textbox.max.y/2);
                  break;
               case 1:
                  if (!ortho) {
                     mesh.rotateX(-Math.PI/2);
                     mesh.rotateY(-Math.PI/2);
                  } else {
                     mesh.rotateZ(Math.PI/2);
                  }
                  mesh.translateY(-textbox.max.y/2);
                  mesh.translateX(-textbox.max.x-text_size*0.5);
                  break;
               case 2: mesh.rotateY(-Math.PI/2);  mesh.translateX(-textbox.max.x-text_size*0.5); mesh.translateY(-textbox.max.y/2); break;
            }
         } else {
            switch (naxis) {
               case 0: mesh.rotateX(Math.PI/2); mesh.translateX(-textbox.max.x-text_size*0.5); mesh.translateY(-textbox.max.y/2); break;
               case 1: mesh.rotateX(Math.PI/2); mesh.rotateY(-Math.PI/2); mesh.translateY(-textbox.max.y/2); mesh.translateX(text_size*0.5); break;
               case 2: mesh.rotateX(Math.PI/2); mesh.rotateZ(Math.PI/2);  mesh.translateX(-textbox.max.x-text_size*0.5); mesh.translateY(-textbox.max.y/2); break;
            }
         }

         container.add(mesh);
      }

      // after creating axes trigger rendering and recalculation of depth
      return this.changedDepthMethod(norender ? "norender" : undefined);
   }

   /** @summary Set axes visibility 0 - off, 1 - on, 2 - centered */
   setAxesDraw(on) {
      if (on === "toggle")
         this.ctrl._axis = this.ctrl._axis ? 0 : 1;
      else
         this.ctrl._axis = (typeof on == 'number') ? on : (on ? 1 : 0);
      return this.drawSimpleAxis();
   }

   /** @summary Set auto rotate mode */
   setAutoRotate(on) {
      if (this.ctrl.project) return;
      if (on !== undefined) this.ctrl.rotate = on;
      this.autorotate(2.5);
   }

   /** @summary Toggle wireframe mode */
   toggleWireFrame() {
      this.ctrl.wireframe = !this.ctrl.wireframe;
      this.changedWireFrame();
   }

   /** @summary Specify wireframe mode */
   setWireFrame(on) {
      this.ctrl.wireframe = on ? true : false;
      this.changedWireFrame();
   }

   /** @summary Specify showtop draw options, relevant only for TGeoManager */
   setShowTop(on) {
      this.ctrl.showtop = on ? true : false;
      this.redrawObject('same');
   }

   /** @summary Should be called when configuration of particular axis is changed */
   changedClipping(naxis) {
      let clip = this.ctrl.clip;

      if ((naxis !== undefined) && (naxis >= 0)) {
         if (!clip[naxis].enabled) return;
      }

      if (clip[0].enabled || clip[1].enabled || clip[2].enabled) {
         this.ctrl.ssao.enabled = false;
         this.removeSSAO();
      }

      this.updateClipping(false, true);
   }

   /** @summary Should be called when depth test flag is changed */
   changedDepthTest() {
      if (!this._toplevel) return;
      let flag = this.ctrl.depthTest;
      this._toplevel.traverse(node => {
         if (node instanceof Mesh) {
            node.material.depthTest = flag;
         }
      });

      this.render3D(0);
   }

   /** @summary Should be called when depth method is changed */
   changedDepthMethod(arg) {
      // force recalculatiion of render order
      delete this._last_camera_position;
      if (arg !== "norender")
         return this.render3D();
   }

   /** @summary Should be called when configuration of highlight is changed */
   changedHighlight() {
      if (!this.ctrl.highlight)
         this.highlightMesh(null);
   }

   /** @summary Assign clipping attributes to the meshes - supported only for webgl */
   updateClipping(without_render, force_traverse) {
      // do not try clipping with SVG renderer
      if (this._renderer && this._renderer.jsroot_render3d === constants.Render3D.SVG) return;

      let clip = this.ctrl.clip, panels = [], changed = false,
          clip_constants = [ clip[0].value, -1 * clip[1].value, (this.ctrl._yup ? -1 : 1) * clip[2].value ],
          clip_cfg = this.ctrl.clipIntersect ? 16 : 0;

      for (let k = 0; k < 3; ++k) {
         if (clip[k].enabled) clip_cfg += 2 << k;
         if (this._clipPlanes[k].constant != clip_constants[k]) {
            changed = true;
            this._clipPlanes[k].constant = clip_constants[k];
         }
      }

      if (!this.ctrl.ssao.enabled) {
         if (clip[0].enabled) panels.push(this._clipPlanes[0]);
         if (clip[1].enabled) panels.push(this._clipPlanes[1]);
         if (clip[2].enabled) panels.push(this._clipPlanes[2]);
         clip_cfg += panels.length*1000;
      }
      if (panels.length == 0) panels = null;

      if (this._clipCfg !== clip_cfg) changed = true;

      this._clipCfg = clip_cfg;

      let any_clipping = !!panels, ci = this.ctrl.clipIntersect,
          material_side = any_clipping ? DoubleSide : FrontSide;

      if (force_traverse || changed)
         this._scene.traverse(node => {
            if (node.hasOwnProperty("material") && node.material && (node.material.clippingPlanes !== undefined)) {

               if (node.material.clippingPlanes !== panels) {
                  node.material.clipIntersection = ci;
                  node.material.clippingPlanes = panels;
                  node.material.needsUpdate = true;
               }

               if (node.material.emissive !== undefined) {
                  if (node.material.side != material_side) {
                     node.material.side = material_side;
                     node.material.needsUpdate = true;
                  }
               }
            }
         });

      this.ctrl.bothSides = any_clipping;

      if (!without_render) this.render3D(0);

      return changed;
   }

   /** @summary Assign callback, invoked every time when drawing is completed
     * @desc Used together with web-based geometry viewer
     * @private */
   setCompleteHandler(callback) {
      this._complete_handler = callback;
   }

   /** @summary Completes drawing procedure
     * @returns {Promise} for ready */
   completeDraw(close_progress) {

      let first_time = false, full_redraw = false, check_extras = true;

      if (!this.ctrl) {
         console.warn('ctrl object does not exist in completeDraw - something went wrong');
         return Promise.resolve(this);
      }

      let promise = Promise.resolve(true);

      if (!this._clones) {
         check_extras = false;
         // if extra object where append, redraw them at the end
         this.getExtrasContainer("delete"); // delete old container
         let extras = (this._main_painter ? this._main_painter._extraObjects : null) || this._extraObjects;
         promise = this.drawExtras(extras, "", false);
      } else if (this._first_drawing || this._full_redrawing) {
         if (this.ctrl.tracks && this.geo_manager)
            promise = this.drawExtras(this.geo_manager.fTracks, "<prnt>/Tracks");
      }

      return promise.then(() => {

         if (this._full_redrawing) {
            this.adjustCameraPosition(true);
            this._full_redrawing = false;
            full_redraw = true;
            this.changedDepthMethod("norender");
         }

         if (this._first_drawing) {
            this.adjustCameraPosition(true);
            this.showDrawInfo();
            this._first_drawing = false;
            first_time = true;
            full_redraw = true;
         }

         if (this.ctrl.transparency !== 0)
            this.changedGlobalTransparency(this.ctrl.transparency, true);

         if (first_time)
            this.completeScene();

         if (full_redraw && (this.ctrl.trans_radial || this.ctrl.trans_z))
            this.changedTransformation("norender");

         if (full_redraw && this.ctrl._axis)
            this.drawSimpleAxis(true);

         this._scene.overrideMaterial = null;

         if (this._provided_more_nodes !== undefined) {
            this.appendMoreNodes(this._provided_more_nodes, true);
            delete this._provided_more_nodes;
         }

         if (check_extras) {
            // if extra object where append, redraw them at the end
            this.getExtrasContainer("delete"); // delete old container
            let extras = (this._main_painter ? this._main_painter._extraObjects : null) || this._extraObjects;
            return this.drawExtras(extras, "", false);
         }
      }).then(() => {

         this.updateClipping(true); // do not render

         this.render3D(0, true);

         if (close_progress) showProgress();

         this.addOrbitControls();

         this.addTransformControl();

         if (first_time) {

            // after first draw check if highlight can be enabled
            if (this.ctrl.highlight === false)
               this.ctrl.highlight = (this.first_render_tm < 1000);

            // also highlight of scene object can be assigned at the first draw
            if (this.ctrl.highlight_scene === false)
               this.ctrl.highlight_scene = this.ctrl.highlight;

            // if rotation was enabled, do it
            if (this._webgl && this.ctrl.rotate && !this.ctrl.project) this.autorotate(2.5);
            if (this._webgl && this.ctrl.show_controls && !isBatchMode()) this.showControlOptions(true);
         }

         this.setAsMainPainter();

         if (typeof this._resolveFunc == 'function') {
            this._resolveFunc(this);
            delete this._resolveFunc;
         }

         if (typeof this._complete_handler == 'function')
            this._complete_handler(this);

         if (this._draw_nodes_again)
            this.startDrawGeometry(); // relaunch drawing
         else
            this._drawing_ready = true; // indicate that drawing is completed

         return this;
      });
   }

   /** @summary Returns true if geometry drawing is completed */
   isDrawingReady() {
      return this._drawing_ready || false;
   }

   /** @summary Remove already drawn node. Used by geom viewer */
   removeDrawnNode(nodeid) {
      if (!this._draw_nodes) return;

      let new_nodes = [];

      for (let n = 0; n < this._draw_nodes.length; ++n) {
         let entry = this._draw_nodes[n];
         if ((entry.nodeid === nodeid) || this._clones.isIdInStack(nodeid, entry.stack)) {
            this._clones.createObject3D(entry.stack, this._toplevel, 'delete_mesh');
         } else {
            new_nodes.push(entry);
         }
      }

      if (new_nodes.length < this._draw_nodes.length) {
         this._draw_nodes = new_nodes;
         this.render3D();
      }
   }

   /** @summary Cleanup geometry painter */
   cleanup(first_time) {

      if (!first_time) {

         this.removeSSAO();

         this.clearTopPainter(); // remove as pointer

         let can3d = 0;
         if (this._on_pad) {
            let fp = this.getFramePainter();
            if (fp?.mode3d) {
               fp.clear3dCanvas();
               fp.mode3d = false;
            }
         } else {
            can3d = this.clear3dCanvas(); // remove 3d canvas from main HTML element
         }

         if (this._toolbar) this._toolbar.cleanup(); // remove toolbar

         this.helpText();

         disposeThreejsObject(this._scene);

         disposeThreejsObject(this._full_geom);

         if (this._tcontrols)
            this._tcontrols.dispose();

         if (this._controls)
            this._controls.cleanup();

         if (this._context_menu)
            this._renderer.domElement.removeEventListener( 'contextmenu', this._context_menu, false );

         if (this._datgui)
            this._datgui.destroy();

         if (this._worker) this._worker.terminate();

         delete this._animating;

         let obj = this.getGeometry();
         if (obj && this.ctrl.is_main) {
            if (obj.$geo_painter===this) delete obj.$geo_painter; else
            if (obj.fVolume && obj.fVolume.$geo_painter===this) delete obj.fVolume.$geo_painter;
         }

         if (this._main_painter) {
            let pos = this._main_painter._slave_painters.indexOf(this);
            if (pos >= 0) this._main_painter._slave_painters.splice(pos,1);
         }

         for (let k = 0; k < this._slave_painters.length;++k) {
            let slave = this._slave_painters[k];
            if (slave && (slave._main_painter===this)) slave._main_painter = null;
         }

         delete this.geo_manager;
         delete this._highlight_handlers;

         super.cleanup();

         delete this.ctrl;
         delete this.options;

         this.did_cleanup = true;

         if (can3d < 0) this.selectDom().html("");
      }

      if (this._slave_painters)
         for (let k in this._slave_painters) {
            let slave = this._slave_painters[k];
            slave._main_painter = null;
            if (slave._clones === this._clones) slave._clones = null;
         }

      this._main_painter = null;
      this._slave_painters = [];

      if (this._render_resolveFuncs) {
         this._render_resolveFuncs.forEach(func => func(this));
         delete this._render_resolveFuncs;
      }

      cleanupRender3D(this._renderer);

      delete this._scene;
      this._scene_width = 0;
      this._scene_height = 0;
      this._renderer = null;
      this._toplevel = null;
      delete this._full_geom;
      delete this._camera;
      delete this._camera0pos;
      delete this._lookat;
      delete this._selected_mesh;

      if (this._clones && this._clones_owner)
         this._clones.cleanup(this._draw_nodes, this._build_shapes);
      delete this._clones;
      delete this._clones_owner;
      delete this._draw_nodes;
      delete this._drawing_ready;
      delete this._build_shapes;
      delete this._new_draw_nodes;
      delete this._new_append_nodes;
      delete this._last_camera_position;

      this.first_render_tm = 0; // time needed for first rendering
      this.last_render_tm = 0;

      this.changeStage(stageInit, "cleanup");
      delete this.drawing_log;

      delete this._datgui;
      delete this._controls;
      delete this._context_menu;
      delete this._tcontrols;
      delete this._toolbar;

      delete this._worker;
   }

   /** @summary show message in progress area
     * @private */
   helpText(msg) {
      showProgress(msg);
   }

   /** @summary perform resize */
   performResize(width, height) {
      if ((this._scene_width === width) && (this._scene_height === height)) return false;
      if ((width < 10) || (height < 10)) return false;

      this._scene_width = width;
      this._scene_height = height;

      if (this._camera && this._renderer) {
         if (this._camera.type == "PerspectiveCamera")
            this._camera.aspect = this._scene_width / this._scene_height;
         this._camera.updateProjectionMatrix();
         this._renderer.setSize( this._scene_width, this._scene_height, !this._fit_main_area );
         if (this._effectComposer)
            this._effectComposer.setSize( this._scene_width, this._scene_height );
         if (this._bloomComposer)
            this._bloomComposer.setSize( this._scene_width, this._scene_height );

         if (this.isStage(stageInit))
            this.render3D();
      }

      return true;
   }

   /** @summary Check if HTML element was resized and drawing need to be adjusted */
   checkResize(arg) {
      let cp = this.getCanvPainter();

      // firefox is the only browser which correctly supports resize of embedded canvas,
      // for others we should force canvas redrawing at every step
      if (cp && !cp.checkCanvasResize(arg)) return false;

      let sz = this.getSizeFor3d();

      return this.performResize(sz.width, sz.height);
   }

   /** @summary Toggle enlarge state */
   toggleEnlarge() {
      if (this.enlargeMain('toggle'))
        this.checkResize();
   }

   /** @summary check if element belongs to trnasform control
     * @private */
   ownedByTransformControls(child) {
      let obj = child.parent;
      while (obj && !(obj instanceof TransformControls) )
         obj = obj.parent;
      return (obj && (obj instanceof TransformControls));
   }

   /** @summary either change mesh wireframe or return current value
     * @returns undefined when wireframe cannot be accessed
     * @private */
   accessObjectWireFrame(obj, on) {
      if (!obj.hasOwnProperty("material") || (obj instanceof GridHelper)) return;

      if (this.ownedByTransformControls(obj)) return;

      if ((on !== undefined) && obj.stack)
         obj.material.wireframe = on;

      return obj.material.wireframe;
   }

   /** @summary handle wireframe flag change in GUI
     * @private */
   changedWireFrame() {
      if (!this._scene) return;

      let on = this.ctrl.wireframe;

      this._scene.traverse(obj => this.accessObjectWireFrame(obj, on));

      this.render3D();
   }

   /** @summary Update object in geo painter */
   updateObject(obj) {
      if (obj === "same") return true;
      if (!obj?._typename) return false;
      if (obj === this.getObject()) return true;

      if (this.geo_manager && (obj._typename == "TGeoManager")) {
         this.geo_manager = obj;
         this.assignObject({ _typename:"TGeoNode", fVolume: obj.fMasterVolume, fName: obj.fMasterVolume.fName, $geoh: obj.fMasterVolume.$geoh, _proxy: true });
         return true;
      }

      if (!this.matchObjectType(obj._typename)) return false;

      this.assignObject(obj);
      return true;
   }

   /** @summary Cleanup TGeo drawings */
   clearDrawings() {
      if (this._clones && this._clones_owner)
         this._clones.cleanup(this._draw_nodes, this._build_shapes);
      delete this._clones;
      delete this._clones_owner;
      delete this._draw_nodes;
      delete this._drawing_ready;
      delete this._build_shapes;

      delete this._extraObjects;
      delete this._clipCfg;

      // only remove all childs from top level object
      disposeThreejsObject(this._toplevel, true);

      this._full_redrawing = true;
   }

    /** @summary Redraw TGeo object inside TPad */
   redraw() {
      if (!this._on_pad) return;

      let main = this.getFramePainter();
      if (!main) return;

      let sz = main.getSizeFor3d(main.access3dKind());

      main.apply3dSize(sz);

      return this.performResize(sz.width, sz.height);
   }

   /** @summary Redraw TGeo object */
   redrawObject(obj /*, opt */) {
      if (!this.updateObject(obj))
         return false;

      this.clearDrawings();

      let draw_obj = this.getGeometry(), name_prefix = "";
      if (this.geo_manager) name_prefix = draw_obj.fName;

      return this.prepareObjectDraw(draw_obj, name_prefix);
   }

  /** @summary draw TGeo object */
   static draw(dom, obj, opt) {
      if (!obj) return null;

      let shape = null, extras = null, extras_path = "", is_eve = false;

      if (('fShapeBits' in obj) && ('fShapeId' in obj)) {
         shape = obj; obj = null;
      } else if ((obj._typename === 'TGeoVolumeAssembly') || (obj._typename === 'TGeoVolume')) {
         shape = obj.fShape;
      } else if ((obj._typename === "TEveGeoShapeExtract") || (obj._typename === "ROOT::Experimental::REveGeoShapeExtract")) {
         shape = obj.fShape; is_eve = true;
      } else if (obj._typename === 'TGeoManager') {
         shape = obj.fMasterVolume.fShape;
      } else if (obj._typename === 'TGeoOverlap') {
         extras = obj.fMarker; extras_path = "<prnt>/Marker";
         obj = buildOverlapVolume(obj);
         if (!opt) opt = "wire";
      } else if ('fVolume' in obj) {
         if (obj.fVolume) shape = obj.fVolume.fShape;
      } else {
         obj = null;
      }

      if ((typeof opt == "string") && opt.indexOf("comp")==0 && shape && (shape._typename == 'TGeoCompositeShape') && shape.fNode) {
         let maxlvl = 1;
         opt = opt.slice(4);
         if (opt[0] == "x") {  maxlvl = 999; opt = opt.slice(1) + "_vislvl999"; }
         obj = buildCompositeVolume(shape, maxlvl);
      }

      if (!obj && shape)
         obj = Object.assign(create("TEveGeoShapeExtract"),
                   { fTrans: null, fShape: shape, fRGBA: [0, 1, 0, 1], fElements: null, fRnrSelf: true });

      if (!obj) return null;

      let painter = createGeoPainter(dom, obj, opt);

      if (painter.ctrl.is_main && !obj.$geo_painter)
         obj.$geo_painter = painter;

      if (!painter.ctrl.is_main && painter.ctrl.project && obj.$geo_painter) {
         painter._main_painter = obj.$geo_painter;
         painter._main_painter._slave_painters.push(painter);
      }

      if (is_eve && !painter.ctrl.vislevel || (painter.ctrl.vislevel < 9))
         painter.ctrl.vislevel = 9;

      if (extras) {
         painter._splitColors = true;
         painter.addExtra(extras, extras_path);
      }

      return painter.loadMacro(painter.ctrl.script_name).then(arg => painter.prepareObjectDraw(arg.obj, arg.prefix));
   }

} // class TGeoPainter


let add_settings = false;

/** @summary Create geo-related css entries
  * @private */
function injectGeoStyle() {

   if (!add_settings && typeof internals.addDrawFunc == 'function') {
      add_settings = true;
      // indication that draw and hierarchy is loaded, create css
      internals.addDrawFunc({ name: "TEvePointSet", icon_get: getBrowserIcon, icon_click: browserIconClick });
      internals.addDrawFunc({ name: "TEveTrack", icon_get: getBrowserIcon, icon_click: browserIconClick });
   }

   function img(name,code) {
      return `.jsroot .img_${name} { display: inline-block; height: 16px; width: 16px; background-image: url("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQ${code}"); }`;
   }

   injectStyle(`
${img("geoarb8","CAAAAAA6mKC9AAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAAJ0Uk5TAAB2k804AAAAAmJLR0QA/4ePzL8AAAAJcEhZcwAAAEgAAABIAEbJaz4AAAB1SURBVBjTdY6rEYAwEETTy6lzK8/Fo+Jj18dTAjUgaQGfGiggtRDE8RtY93Zu514If2nzk2ux9c5TZkwXbiWTUavzws69oBfpYBrMT4r0Jhsw+QfRgQSw+CaKRsKsnV+SaF8MN49RBSgPUxO85PMl5n4tfGUH2gghs2uPAeQAAAAldEVYdGRhdGU6Y3JlYXRlADIwMTUtMTItMDJUMTQ6MjY6MjkrMDE6MDDARtd2AAAAJXRFWHRkYXRlOm1vZGlmeQAyMDE0LTExLTEyVDA4OjM5OjE5KzAxOjAwO3ydwwAAAABJRU5ErkJggg==")}
${img("geocombi","CAQAAAC1+jfqAAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAAJiS0dEAP+Hj8y/AAAACXBIWXMAAABIAAAASABGyWs+AAAAlUlEQVQoz5VQMQ4CMQyzEUNnBqT7Bo+4nZUH8gj+welWJsQDkHoCEYakTXMHSFiq2jqu4xRAEl2A7w4myWzpzCSZRZ658ldKu1hPnFsequBIc/hcLli3l52MAIANtpWrDsv8waGTW6BPuFtsdZArXyFuj33TQpazGEQF38phipnLgItxRcAoOeNpzv4PTXnC42fb//AGI5YqfQAU8dkAAAAldEVYdGRhdGU6Y3JlYXRlADIwMTUtMTItMDJUMTQ6MjY6MjkrMDE6MDDARtd2AAAAJXRFWHRkYXRlOm1vZGlmeQAyMDE0LTExLTEyVDA4OjM5OjE5KzAxOjAwO3ydwwAAAABJRU5ErkJggg==")}
${img("geocone","CAAAAAA6mKC9AAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAAJ0Uk5TAAB2k804AAAAAmJLR0QA/4ePzL8AAAAJcEhZcwAAAEgAAABIAEbJaz4AAACRSURBVBjTdY+xDcNACEVvEm/ggo6Olva37IB0C3iEzJABvAHFTXBDeJRwthMnUvylk44vPjxK+afeokX0flQhJO7L4pafSOMxzaxIKc/Tc7SIjNLyieyZSjBzc4DqMZI0HTMonWPBNlogOLeuewbg9c0hOiIqH7DKmTCuFykjHe4XOzQ58XVMGxzt575tKzd6AX9yMkcWyPlsAAAAJXRFWHRkYXRlOmNyZWF0ZQAyMDE1LTEyLTAyVDE0OjI2OjI5KzAxOjAwwEbXdgAAACV0RVh0ZGF0ZTptb2RpZnkAMjAxNC0xMS0xMlQwODozOToxOSswMTowMDt8ncMAAAAASUVORK5CYII=")}
${img("geogtra","CAAAAAA6mKC9AAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAAJ0Uk5TAAB2k804AAAAAmJLR0QA/4ePzL8AAAAJcEhZcwAAAEgAAABIAEbJaz4AAACCSURBVBjTVc+hDQMxDAVQD1FyqCQk0MwsCwQEG3+eCW6B0FvheDboFMGepTlVitPP/Cz5y0S/mNkw8pySU9INJDDH4vM4Usm5OrQXasXtkA+tQF+zxfcDY8EVwgNeiwmA37TEccK5oLOwQtuCj7BM2Fq7iGrxVqJbSsH+GzXs+798AThwKMh3/6jDAAAAJXRFWHRkYXRlOmNyZWF0ZQAyMDE1LTEyLTAyVDE0OjI2OjI5KzAxOjAwwEbXdgAAACV0RVh0ZGF0ZTptb2RpZnkAMjAxNC0xMS0xMlQwODozOToxOSswMTowMDt8ncMAAAAASUVORK5CYII=")}
${img("geomedium","BAMAAADt3eJSAAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAABVQTFRFAAAAAAAAMDAww8PDWKj/////gICAG0/C4AAAAAF0Uk5TAEDm2GYAAAABYktHRAX4b+nHAAAACXBIWXMAAABIAAAASABGyWs+AAAAXElEQVQI102MwRGAMAgEuQ6IDwvQCjQdhAl/H7ED038JHhkd3dcOLAgESFARaAqnEB3yrj6QSEym1RbbOKinN+8q2Esui1GaX7VXSi4RUbxHRbER8X6O5Pg/fLgBBzMN8HfXD3AAAAAldEVYdGRhdGU6Y3JlYXRlADIwMTUtMTItMDJUMTQ6MjY6MjkrMDE6MDDARtd2AAAAJXRFWHRkYXRlOm1vZGlmeQAyMDE0LTExLTEyVDA4OjM5OjE5KzAxOjAwO3ydwwAAAABJRU5ErkJggg==")}
${img("geopara","CAAAAAA6mKC9AAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAAJ0Uk5TAAB2k804AAAAAmJLR0QA/4ePzL8AAAAJcEhZcwAAAEgAAABIAEbJaz4AAABtSURBVBjTY2DADq5MT7+CzD9kaKjp+QhJYIWqublhMbKAgpOnZxWSQJdsVJTndCSBKoWoAM/VSALpqlEBAYeQBKJAAsi2BGgCBZDdEWUYFZCOLFBlGOWJ7AyGFeaotjIccopageK3R12PGHABACTYHWd0tGw6AAAAJXRFWHRkYXRlOmNyZWF0ZQAyMDE1LTEyLTAyVDE0OjI2OjI5KzAxOjAwwEbXdgAAACV0RVh0ZGF0ZTptb2RpZnkAMjAxNC0xMS0xMlQwODozOToxOSswMTowMDt8ncMAAAAASUVORK5CYII=")}
${img("georotation","CAQAAAC1+jfqAAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAAJiS0dEAP+Hj8y/AAAACXBIWXMAAABIAAAASABGyWs+AAAAiklEQVQoz2NgYGBgYGDg+A/BmIAFIvyDEbs0AwMTAwHACLPiB5QVBTdpGSOSCZjScDcgc4z+32BgYGBgEGIQw3QDLkdCTZD8/xJFeBfDVxQT/j9n/MeIrMCNIRBJwX8GRuzGM/yHKMAljeILNFOuMTyEisEUMKIqucrwB2oyIhyQpH8y/MZrLWkAAHFzIHIc0Q5yAAAAJXRFWHRkYXRlOmNyZWF0ZQAyMDE1LTEyLTAyVDE0OjI2OjI5KzAxOjAwwEbXdgAAACV0RVh0ZGF0ZTptb2RpZnkAMjAxNC0xMS0xMlQwODozOToxOSswMTowMDt8ncMAAAAASUVORK5CYII=")}
${img("geotranslation","CAAAAAA6mKC9AAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAAJ0Uk5TAAB2k804AAAAAmJLR0QA/4ePzL8AAAAJcEhZcwAAAEgAAABIAEbJaz4AAABESURBVBjTY2DgYGAAYzjgAAIQgSLAgSwAAcrWUUCAJBAVhSpgBAQumALGCJPAAsriHIS0IAQ4UAU4cGphQBWwZSAOAADGJBKdZk/rHQAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAxNS0xMi0wMlQxNDoyNjoyOSswMTowMMBG13YAAAAldEVYdGRhdGU6bW9kaWZ5ADIwMTQtMTEtMTJUMDg6Mzk6MTkrMDE6MDA7fJ3DAAAAAElFTkSuQmCC")}
${img("geotrd2","CAAAAAA6mKC9AAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAAJ0Uk5TAAB2k804AAAAAmJLR0QA/4ePzL8AAAAJcEhZcwAAAEgAAABIAEbJaz4AAABsSURBVBjTbY+xDcAwCARZx6UraiaAmpoRvIIb75PWI2QITxIiRQKk0CCO/xcA/NZ9LRs7RkJEYg3QxczUwoGsXiMAoe8lAelqRWFNKpiNXZLAalRDd0f3TMgeMckABKsCDmu+442RddeHz9cf9jUkW8smGn8AAAAldEVYdGRhdGU6Y3JlYXRlADIwMTUtMTItMDJUMTQ6MjY6MjkrMDE6MDDARtd2AAAAJXRFWHRkYXRlOm1vZGlmeQAyMDE0LTExLTEyVDA4OjM5OjE5KzAxOjAwO3ydwwAAAABJRU5ErkJggg==")}
${img("geovolume","BAMAAADt3eJSAAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAB5QTFRFAAAAMDAw///Ay8uc/7+Q/4BgmJh4gIDgAAD/////CZb2ugAAAAF0Uk5TAEDm2GYAAAABYktHRAnx2aXsAAAACXBIWXMAAABIAAAASABGyWs+AAAAR0lEQVQI12NggAEBIBAEQgYGQUYQAyIGIhgwAZMSGCgwMJuEKimFOhswsKWAGG4JDGxJIBk1EEO9o6NIDVkEpgauC24ODAAASQ8Pkj/retYAAAAldEVYdGRhdGU6Y3JlYXRlADIwMTUtMTItMDJUMTQ6MjY6MjkrMDE6MDDARtd2AAAAJXRFWHRkYXRlOm1vZGlmeQAyMDE0LTExLTEyVDA4OjM5OjE5KzAxOjAwO3ydwwAAAABJRU5ErkJggg==")}
${img("geoassembly","BAMAAADt3eJSAAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAA9QTFRFAAAAMDAw/wAAAAD/////jEo0BQAAAAF0Uk5TAEDm2GYAAAABYktHRASPaNlRAAAACXBIWXMAAABIAAAASABGyWs+AAAAOklEQVQI12NggAFGRgEgEBRgEBSAMhgYGQQEgAR+oARGDIwCIAYjUL0A2DQQg9nY2ABVBKoGrgsDAADxzgNboMz8zQAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAxNS0xMi0wMlQxNDoyNjoyOSswMTowMMBG13YAAAAldEVYdGRhdGU6bW9kaWZ5ADIwMTQtMTEtMTJUMDg6Mzk6MTkrMDE6MDA7fJ3DAAAAAElFTkSuQmCC")}
${img("geocomposite","CAAAAAA6mKC9AAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAAJ0Uk5TAAB2k804AAAAAmJLR0QA/4ePzL8AAAAJcEhZcwAAAEgAAABIAEbJaz4AAABuSURBVBjTY2AgF2hqgQCCr+0V4O7hFmgCF7CJyKysKkmxhfGNLaw9SppqAi2gfMuY5Agrl+ZaC6iAUXRJZX6Ic0klTMA5urapPFY5NRcmYKFqWl8S5RobBRNg0PbNT3a1dDGH8RlM3LysTRjIBwAG6xrzJt11BAAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAxNS0xMi0wMlQxNDoyNjoyOSswMTowMMBG13YAAAAldEVYdGRhdGU6bW9kaWZ5ADIwMTQtMTEtMTJUMDg6Mzk6MTkrMDE6MDA7fJ3DAAAAAElFTkSuQmCC")}
${img("geoctub","CAAAAAA6mKC9AAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAAJ0Uk5TAAB2k804AAAAAmJLR0QA/4ePzL8AAAAJcEhZcwAAAEgAAABIAEbJaz4AAACESURBVBjTdc+xDcMwDARA7cKKHTuWX37LHaw+vQbQAJomA7j2DB7FhCMFCZB8pxPwJEv5kQcZW+3HencRBekak4aaMQIi8YJdAQ1CMeE0UBkuaLMETklQ9Alhka0JzzXWqLVBuQYPpWcVuBbZjZafNRYcDk9o/b07bvhINz+/zxu1/M0FSRcmAk/HaIcAAAAldEVYdGRhdGU6Y3JlYXRlADIwMTUtMTItMDJUMTQ6MjY6MjkrMDE6MDDARtd2AAAAJXRFWHRkYXRlOm1vZGlmeQAyMDE0LTExLTEyVDA4OjM5OjE5KzAxOjAwO3ydwwAAAABJRU5ErkJggg==")}
${img("geohype","CAAAAAA6mKC9AAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAAJ0Uk5TAAB2k804AAAAAmJLR0QA/4ePzL8AAAAJcEhZcwAAAEgAAABIAEbJaz4AAACKSURBVBjTbU+rFQQhDKQSDDISEYuMREfHx6eHKMpYuf5qoIQt5bgDblfcuJk3nySEhSvceDV3c/ejT66lspopE9pXyIlkCrHMBACpu1DClekQAREi/loviCnF/NhRwJLaQ6hVhPjB8bOCsjlnNnNl0FWJVWxAqGzHONRHpu5Ml+nQ+8GzNW9n+Is3eg80Nk0iiwoAAAAldEVYdGRhdGU6Y3JlYXRlADIwMTUtMTItMDJUMTQ6MjY6MjkrMDE6MDDARtd2AAAAJXRFWHRkYXRlOm1vZGlmeQAyMDE0LTExLTEyVDA4OjM5OjE5KzAxOjAwO3ydwwAAAABJRU5ErkJggg==")}
${img("geomixture","BAMAAADt3eJSAAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAACFQTFRFAAAAAAAAKysrVVUA//8B//8AgICAqqpV398gv79A////VYJtlwAAAAF0Uk5TAEDm2GYAAAABYktHRApo0PRWAAAACXBIWXMAAABIAAAASABGyWs+AAAAXklEQVQI12NgwASCQsJCgoZAhoADq1tKIJAhEpDGxpYIZKgxsLElgBhibAkOCY4gKTaGkPRGIEPUIYEBrEaAIY0tDawmgYWNgREkkjCVjRWkWCUhLY0FJCIIBljsBgCZTAykgaRiRwAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAxNS0xMi0wMlQxNDoyNjoyOSswMTowMMBG13YAAAAldEVYdGRhdGU6bW9kaWZ5ADIwMTQtMTEtMTJUMDg6Mzk6MTkrMDE6MDA7fJ3DAAAAAElFTkSuQmCC")}
${img("geopcon","CAAAAAA6mKC9AAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAAJ0Uk5TAAB2k804AAAAAmJLR0QA/4ePzL8AAAAJcEhZcwAAAEgAAABIAEbJaz4AAACJSURBVBjTdc+hGcQwCIZhhjl/rkgWiECj8XgGyAbZoD5LdIRMkEnKkV575n75Pp8AgLU54dmh6mauelyAL2Qzxfe2sklioq6FacFAcRFXYhwJHdU5rDD2hEYB/CmoJVRMiIJqgtENuoqA8ltAlYAqRH4d1tGkwzTqN2gA7Nv+fUwkgZ/3mg34txM+szzATJS1HQAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAxNS0xMi0wMlQxNDoyNjoyOSswMTowMMBG13YAAAAldEVYdGRhdGU6bW9kaWZ5ADIwMTQtMTEtMTJUMDg6Mzk6MTkrMDE6MDA7fJ3DAAAAAElFTkSuQmCC")}
${img("geosphere","CAAAAAA6mKC9AAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAAJ0Uk5TAAB2k804AAAAAmJLR0QA/4ePzL8AAAAJcEhZcwAAAEgAAABIAEbJaz4AAACFSURBVBjTdY+xEcQwCAQp5QNFjpQ5vZACFBFTADFFfKYCXINzlUAJruXll2ekxDAEt9zcANFbXb2mqm56dxsymAH0yccAJaeNi0h5QGyfxGJmivMPjj0nmLsbRmyFCss3rlbpcUjfS8wLUNRcJyCF6uqg2IvYCnoKC7f1kSbA6riTz7evfwj3Ml+H3KBqAAAAJXRFWHRkYXRlOmNyZWF0ZQAyMDE1LTEyLTAyVDE0OjI2OjI5KzAxOjAwwEbXdgAAACV0RVh0ZGF0ZTptb2RpZnkAMjAxNC0xMS0xMlQwODozOToxOSswMTowMDt8ncMAAAAASUVORK5CYII=")}
${img("geotrap","CAAAAAA6mKC9AAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAAJ0Uk5TAAB2k804AAAAAmJLR0QA/4ePzL8AAAAJcEhZcwAAAEgAAABIAEbJaz4AAAB5SURBVBjTbY+hFYAwDETZB1OJi4yNPp0JqjtAZ2AELL5DdABmIS2PtLxHXH7u7l2W5W+uHMHpGiCHLYR1yw4SCZMIXBOJWVSjK7QDDAu4g8OBmAKK4sAEDdR3rw8YmcUcrEijKKhl7lN1IQPn9ExlgU6/WEyc75+5AYK0KY5oHBDfAAAAJXRFWHRkYXRlOmNyZWF0ZQAyMDE1LTEyLTAyVDE0OjI2OjI5KzAxOjAwwEbXdgAAACV0RVh0ZGF0ZTptb2RpZnkAMjAxNC0xMS0xMlQwODozOToxOSswMTowMDt8ncMAAAAASUVORK5CYII=")}
${img("geotubeseg","CAAAAAA6mKC9AAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAAJ0Uk5TAAB2k804AAAAAmJLR0QA/4ePzL8AAAAJcEhZcwAAAEgAAABIAEbJaz4AAACBSURBVBjTdc+hEcQwDARA12P6QFBQ9LDwcXEVkA7SQTr4BlJBakgpsWdsh/wfux3NSCrlV86Mlrxmz1pBWq3bAHwETohxABVmDZADQp1BE+wDNnGywzHgmHDOreJNTDH3Xn3CVX0dpu2MHcIFBkYp/gKsQ8SCQ72V+36/+2aWf3kAQfgshnpXF0wAAAAldEVYdGRhdGU6Y3JlYXRlADIwMTUtMTItMDJUMTQ6MjY6MjkrMDE6MDDARtd2AAAAJXRFWHRkYXRlOm1vZGlmeQAyMDE0LTExLTEyVDA4OjM5OjE5KzAxOjAwO3ydwwAAAABJRU5ErkJggg==")}
${img("geoxtru","CAAAAAA6mKC9AAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAAJ0Uk5TAAB2k804AAAAAmJLR0QA/4ePzL8AAAAJcEhZcwAAAEgAAABIAEbJaz4AAABcSURBVBjTY2AgEmhpeZV56vmWwQW00QUYwAJlSAI6XmVqukh8PT1bT03PchhXX09Pr9wQIQDiJ+ZowgWAXD3bck+QQDlCQTkDQgCoxA/ERBKwhbDglgA1lDMQDwCc/Rvq8nYsWgAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAxNS0xMi0wMlQxNDoyNjoyOSswMTowMMBG13YAAAAldEVYdGRhdGU6bW9kaWZ5ADIwMTQtMTEtMTJUMDg6Mzk6MTkrMDE6MDA7fJ3DAAAAAElFTkSuQmCC")}
${img("geobbox","CAAAAAA6mKC9AAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAAJ0Uk5TAAB2k804AAAAAmJLR0QA/4ePzL8AAAAJcEhZcwAAAEgAAABIAEbJaz4AAAB/SURBVBjTVc+hEYAwDAXQLlNRF1tVGxn9NRswQiSSCdgDyQBM0FlIIb2WuL77uf6E8E0N02wKYRwDciTKREVvB04GuZSyOMCABRB1WGzF3uDNQTvs/RcDtJXT4fSEXA5XoiQt0ttVSm8Co2psIOvoimjAOqBmFtH5wEP2373TPIvTK1nrpULXAAAAJXRFWHRkYXRlOmNyZWF0ZQAyMDE1LTEyLTAyVDE0OjI2OjI5KzAxOjAwwEbXdgAAACV0RVh0ZGF0ZTptb2RpZnkAMjAxNC0xMS0xMlQwODozOToxOSswMTowMDt8ncMAAAAASUVORK5CYII=")}
${img("geoconeseg","CAAAAAA6mKC9AAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAAJ0Uk5TAAB2k804AAAAAmJLR0QA/4ePzL8AAAAJcEhZcwAAAEgAAABIAEbJaz4AAAB4SURBVBjTdc6hEcAgDAXQbFNZXHQkFlkd/30myAIMwAws0gmYpVzvoFyv/S5P/B+izzQ387ZA2pkDnvsU1SQLVIFrOM4JFmEaYp2gCQbmPEGODhJ8jt7Am47hwgrzInGAifa/elUZnQLY00iU30BZAV+BWi2VfnIBv1osbHH8jX0AAAAldEVYdGRhdGU6Y3JlYXRlADIwMTUtMTItMDJUMTQ6MjY6MjkrMDE6MDDARtd2AAAAJXRFWHRkYXRlOm1vZGlmeQAyMDE0LTExLTEyVDA4OjM5OjE5KzAxOjAwO3ydwwAAAABJRU5ErkJggg==")}
${img("geoeltu","CAAAAAA6mKC9AAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAAJ0Uk5TAAB2k804AAAAAmJLR0QA/4ePzL8AAAAJcEhZcwAAAEgAAABIAEbJaz4AAACGSURBVBjTdY+hFYUwDEU7xq9CIXC4uNjY6KczQXeoYgVMR2ABRmCGjvIp/6dgiEruueedvBDuOR57LQnKyc8CJmKO+N8bieIUPtmBWjIIx8XDBHYCipsnql1g2D0UP2OoDqwBncf+RdZmzFMHizRjog7KZYzawd4Ay93lEAPWR7WAvNbwMl/XwSxBV8qCjgAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAxNS0xMi0wMlQxNDoyNjoyOSswMTowMMBG13YAAAAldEVYdGRhdGU6bW9kaWZ5ADIwMTQtMTEtMTJUMDg6Mzk6MTkrMDE6MDA7fJ3DAAAAAElFTkSuQmCC")}
${img("geomaterial","CAQAAAC1+jfqAAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAAJiS0dEAP+Hj8y/AAAACXBIWXMAAABIAAAASABGyWs+AAAAbElEQVQoz62QMRbAIAhDP319Xon7j54qHSyCtaMZFCUkRjgDIdRU9yZUCfg8ut5aAHdcxtoNurmgA3ABNKIR9KimhSukPe2qxcCYC0pfFXx/aFWo7i42KKItOpopqvvnLzJmtlZTS7EfGAfwAM4EQbLIGV0sAAAAJXRFWHRkYXRlOmNyZWF0ZQAyMDE1LTEyLTAyVDE0OjI2OjI5KzAxOjAwwEbXdgAAACV0RVh0ZGF0ZTptb2RpZnkAMjAxNC0xMS0xMlQwODozOToxOSswMTowMDt8ncMAAAAASUVORK5CYII=")}
${img("geoparab","CAAAAAA6mKC9AAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAAJ0Uk5TAAB2k804AAAAAmJLR0QA/4ePzL8AAAAJcEhZcwAAAEgAAABIAEbJaz4AAAB/SURBVBjTbY+xDYAwDAQ9UAp3X7p0m9o9dUZgA9oMwAjpMwMzMAnYBAQSX9mn9+tN9KOtzsWsLOvYCziUGNX3nnCLJRzKPgeYrhPW7FJNLUB3YJazYKQKTnBaxgXRzNmJcrt7XCHQp9kEB1wfELEir/KGj4Foh8A+/zW1nf51AFabKZuWK+mNAAAAJXRFWHRkYXRlOmNyZWF0ZQAyMDE1LTEyLTAyVDE0OjI2OjI5KzAxOjAwwEbXdgAAACV0RVh0ZGF0ZTptb2RpZnkAMjAxNC0xMS0xMlQwODozOToxOSswMTowMDt8ncMAAAAASUVORK5CYII=")}
${img("geopgon","CAAAAAA6mKC9AAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAAJ0Uk5TAAB2k804AAAAAmJLR0QA/4ePzL8AAAAJcEhZcwAAAEgAAABIAEbJaz4AAABwSURBVBjTY2AgDlwAAzh3sX1sPRDEeuwDc+8V2dsHgQQ8LCzq74HkLSzs7Yva2tLt7S3sN4MNiDUGKQmysCi6BzWkzcI+PdY+aDPCljZlj1iFOUjW1tvHLjYuQhJIt5/DcAFZYLH9YnSn7iPST9gAACbsJth21haFAAAAJXRFWHRkYXRlOmNyZWF0ZQAyMDE1LTEyLTAyVDE0OjI2OjI5KzAxOjAwwEbXdgAAACV0RVh0ZGF0ZTptb2RpZnkAMjAxNC0xMS0xMlQwODozOToxOSswMTowMDt8ncMAAAAASUVORK5CYII=")}
${img("geotorus","CAAAAAA6mKC9AAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAAJ0Uk5TAAB2k804AAAAAmJLR0QA/4ePzL8AAAAJcEhZcwAAAEgAAABIAEbJaz4AAACGSURBVBjTjY+hFcMwDEQ9SkFggXGIoejhw+LiGkBDlHoAr+AhgjNL5byChuXeE7gvPelUyjOds/f5Zw0ggfj5KVCPMBWeyx+SbQ1XUriAC2XfpWWxjQQEZasRtRHiCUAj3qN4JaolUJppzh4q7dUTdHFXW/tH9OuswWm3nI7tc08+/eGLl758ey9KpKrNOQAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAxNS0xMi0wMlQxNDoyNjoyOSswMTowMMBG13YAAAAldEVYdGRhdGU6bW9kaWZ5ADIwMTQtMTEtMTJUMDg6Mzk6MTkrMDE6MDA7fJ3DAAAAAElFTkSuQmCC")}
${img("geotrd1","CAAAAAA6mKC9AAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAAJ0Uk5TAAB2k804AAAAAmJLR0QA/4ePzL8AAAAJcEhZcwAAAEgAAABIAEbJaz4AAAB/SURBVBjTbc6xDQMhDAVQ9qH6lUtal65/zQ5IDMAMmYAZrmKGm4FJzlEQQUo+bvwkG4fwm9lbodV7w40Y4WGfSxQiXiJlQfZOjWRb8Ioi3tKuBQMCo7+9N72BzPsfAuoTdUP9QN8wgOQwvsfWmHzpeT5BKydMNW0nhJGvGf7mAc5WKO9e5N2dAAAAJXRFWHRkYXRlOmNyZWF0ZQAyMDE1LTEyLTAyVDE0OjI2OjI5KzAxOjAwwEbXdgAAACV0RVh0ZGF0ZTptb2RpZnkAMjAxNC0xMS0xMlQwODozOToxOSswMTowMDt8ncMAAAAASUVORK5CYII=")}
${img("geotube","CAAAAAA6mKC9AAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAAJ0Uk5TAAB2k804AAAAAmJLR0QA/4ePzL8AAAAJcEhZcwAAAEgAAABIAEbJaz4AAACGSURBVBjTRc+tEcAwCAXgLFNbWeSzSDQazw5doWNUZIOM0BEyS/NHy10E30HyklKvWnJ+0le3sJoKn3X2z7GRuvG++YRyMMDt0IIKUXMzxbnugJi5m9K1gNnGBOUFElAWGMaKIKI4xoQggl00gT+A9hXWgDwnfqgsHRAx2m+8bfjfdyrx5AtsSjpwu+M2RgAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAxNS0xMi0wMlQxNDoyNjoyOSswMTowMMBG13YAAAAldEVYdGRhdGU6bW9kaWZ5ADIwMTQtMTEtMTJUMDg6Mzk6MTkrMDE6MDA7fJ3DAAAAAElFTkSuQmCC")}
${img("evepoints","BAMAAADt3eJSAAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAABJQTFRF////n4mJcEdKRDMzcEdH////lLE/CwAAAAF0Uk5TAEDm2GYAAAABYktHRACIBR1IAAAACXBIWXMAAABIAAAASABGyWs+AAAAI0lEQVQI12NgIAowIpgKEJIZLiAgAKWZGQzQ9UGlWIizBQgAN4IAvGtVrTcAAAAldEVYdGRhdGU6Y3JlYXRlADIwMTYtMDktMDJUMTU6MDQ6MzgrMDI6MDDPyc7hAAAAJXRFWHRkYXRlOm1vZGlmeQAyMDE2LTA5LTAyVDE1OjA0OjM4KzAyOjAwvpR2XQAAAABJRU5ErkJggg==")}
${img("evetrack", "CAQAAAC1+jfqAAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAAJiS0dEAP+Hj8y/AAAACXBIWXMAAABIAAAASABGyWs+AAAAqElEQVQoz32RMQrCQBBFf4IgSMB0IpGkMpVHCFh7BbHIGTyVhU0K8QYewEKsbVJZaCUiPAsXV8Puzhaz7H8zs5+JUDjikLilQr5zpCRl5xMXZNScQE5gSMGaz70jjUAJcw5c3UBMTsUe+9Kzf065SbropeLXimWfDIgoab/tOyPGzOhz53+oSWcSGh7UdB2ZNKXBZdgAuUdEKJYmrEILyVgG6pE2tEHgDfe42rbjYzSHAAAAJXRFWHRkYXRlOmNyZWF0ZQAyMDE2LTA5LTAyVDE1OjA0OjQ3KzAyOjAwM0S3EQAAACV0RVh0ZGF0ZTptb2RpZnkAMjAxNi0wOS0wMlQxNTowNDo0NyswMjowMEIZD60AAAAASUVORK5CYII=")}
.jsroot .geovis_this { background-color: lightgreen; }
.jsroot .geovis_daughters { background-color: lightblue; }
.jsroot .geovis_all { background-color: yellow; }`);
}


/** @summary Create geo painter
  * @private */
function createGeoPainter(dom, obj, opt) {

   injectGeoStyle();

   geoCfg("GradPerSegm", settings.GeoGradPerSegm);
   geoCfg("CompressComp", settings.GeoCompressComp);

   let painter = new TGeoPainter(dom, obj);

   painter.options = painter.decodeOptions(opt); // indicator of initialization

   // copy all attributes from options to control
   Object.assign(painter.ctrl, painter.options);

   painter.ctrl.ssao.enabled = painter.options.usessao;
   painter.ctrl.bloom.enabled = painter.options.usebloom;

   // special handling for array of clips
   painter.ctrl.clip[0].enabled = painter.options.clipx;
   painter.ctrl.clip[1].enabled = painter.options.clipy;
   painter.ctrl.clip[2].enabled = painter.options.clipz;

   return painter;
}


/** @summary provide menu for geo object
  * @private */
function provideMenu(menu, item, hpainter) {

   if (!item._geoobj) return false;

   let obj = item._geoobj, vol = item._volume,
       iseve = ((obj._typename === 'TEveGeoShapeExtract') || (obj._typename === 'ROOT::Experimental::REveGeoShapeExtract'));

   if (!vol && !iseve) return false;

   menu.add("separator");

   const ScanEveVisible = (obj, arg, skip_this) => {

      if (!arg) arg = { visible: 0, hidden: 0 };

      if (!skip_this) {
         if (arg.assign!==undefined)
            obj.fRnrSelf = arg.assign;
         else if (obj.fRnrSelf)
            arg.vis++;
         else
            arg.hidden++;
      }

      if (obj.fElements)
         for (let n = 0; n < obj.fElements.arr.length; ++n)
            ScanEveVisible(obj.fElements.arr[n], arg, false);

      return arg;

   }, ToggleEveVisibility = arg => {

      if (arg === 'self') {
         obj.fRnrSelf = !obj.fRnrSelf;
         item._icon = item._icon.split(" ")[0] + provideVisStyle(obj);
         hpainter.updateTreeNode(item);
      } else {
         ScanEveVisible(obj, { assign: (arg === "true") }, true);
         hpainter.forEachItem(m => {
            // update all child items
            if (m._geoobj && m._icon) {
               m._icon = item._icon.split(" ")[0] + provideVisStyle(m._geoobj);
               hpainter.updateTreeNode(m);
            }
         }, item);
      }

      findItemWithPainter(item, 'testGeomChanges');
   }, ToggleMenuBit = arg => {
      toggleGeoBit(vol, arg);
      let newname = item._icon.split(" ")[0] + provideVisStyle(vol);
      hpainter.forEachItem(m => {
         // update all items with that volume
         if (item._volume === m._volume) {
            m._icon = newname;
            hpainter.updateTreeNode(m);
         }
      });

      hpainter.updateTreeNode(item);
      findItemWithPainter(item, 'testGeomChanges');
   };

   if ((item._geoobj._typename.indexOf("TGeoNode")===0) && findItemWithPainter(item))
      menu.add("Focus", function() {

        let drawitem = findItemWithPainter(item);

        if (!drawitem) return;

        let fullname = hpainter.itemFullName(item, drawitem);

        if (typeof drawitem._painter?.focusOnItem == 'function')
           drawitem._painter.focusOnItem(fullname);
      });

   if (iseve) {
      menu.addchk(obj.fRnrSelf, "Visible", "self", ToggleEveVisibility);
      let res = ScanEveVisible(obj, undefined, true);

      if (res.hidden + res.visible > 0)
         menu.addchk((res.hidden==0), "Daughters", (res.hidden!=0) ? "true" : "false", ToggleEveVisibility);

   } else {
      menu.addchk(testGeoBit(vol, geoBITS.kVisNone), "Invisible",
            geoBITS.kVisNone, ToggleMenuBit);
      menu.addchk(testGeoBit(vol, geoBITS.kVisThis), "Visible",
            geoBITS.kVisThis, ToggleMenuBit);
      menu.addchk(testGeoBit(vol, geoBITS.kVisDaughters), "Daughters",
            geoBITS.kVisDaughters, ToggleMenuBit);
   }

   return true;
}

/** @summary handle click on browser icon
  * @private */
function browserIconClick(hitem, hpainter) {
   if (hitem._volume) {
      if (hitem._more && hitem._volume.fNodes && (hitem._volume.fNodes.arr.length > 0))
         toggleGeoBit(hitem._volume, geoBITS.kVisDaughters);
      else
         toggleGeoBit(hitem._volume, geoBITS.kVisThis);

      updateBrowserIcons(hitem._volume, hpainter);

      findItemWithPainter(hitem, 'testGeomChanges');
      return false; // no need to update icon - we did it ourself
   }

   if (hitem._geoobj && ((hitem._geoobj._typename == "TEveGeoShapeExtract") || (hitem._geoobj._typename == "ROOT::Experimental::REveGeoShapeExtract"))) {
      hitem._geoobj.fRnrSelf = !hitem._geoobj.fRnrSelf;

      updateBrowserIcons(hitem._geoobj, hpainter);
      findItemWithPainter(hitem, 'testGeomChanges');
      return false; // no need to update icon - we did it ourself
   }


   // first check that geo painter assigned with the item
   let drawitem = findItemWithPainter(hitem);
   if (!drawitem) return false;

   let newstate = drawitem._painter.extraObjectVisible(hpainter, hitem, true);

   // return true means browser should update icon for the item
   return (newstate!==undefined) ? true : false;
}


/** @summary Get icon for the browser
  * @private */
function getBrowserIcon(hitem, hpainter) {
   let icon = "";
   if (hitem._kind == 'ROOT.TEveTrack') icon = 'img_evetrack'; else
   if (hitem._kind == 'ROOT.TEvePointSet') icon = 'img_evepoints'; else
   if (hitem._kind == 'ROOT.TPolyMarker3D') icon = 'img_evepoints';
   if (icon.length > 0) {
      let drawitem = findItemWithPainter(hitem);
      if (drawitem)
         if (drawitem._painter.extraObjectVisible(hpainter, hitem))
            icon += " geovis_this";
   }
   return icon;
}


/** @summary create hierarchy item for geo object
  * @private */
function createItem(node, obj, name) {
   let sub = {
      _kind: "ROOT." + obj._typename,
      _name: name ? name : getObjectName(obj),
      _title: obj.fTitle,
      _parent: node,
      _geoobj: obj,
      _get(item /* ,itemname */) {
          // mark object as belong to the hierarchy, require to
          if (item._geoobj) item._geoobj.$geoh = true;
          return Promise.resolve(item._geoobj);
      }
   }, volume, shape, subnodes, iseve = false;

   if (obj._typename == "TGeoMaterial") sub._icon = "img_geomaterial"; else
   if (obj._typename == "TGeoMedium") sub._icon = "img_geomedium"; else
   if (obj._typename == "TGeoMixture") sub._icon = "img_geomixture"; else
   if ((obj._typename.indexOf("TGeoNode")===0) && obj.fVolume) {
      sub._title = "node:"  + obj._typename;
      if (obj.fTitle.length > 0) sub._title += " " + obj.fTitle;
      volume = obj.fVolume;
   } else if (obj._typename.indexOf("TGeoVolume")===0) {
      volume = obj;
   } else if ((obj._typename == "TEveGeoShapeExtract") || (obj._typename == "ROOT::Experimental::REveGeoShapeExtract")) {
      iseve = true;
      shape = obj.fShape;
      subnodes = obj.fElements ? obj.fElements.arr : null;
   } else if ((obj.fShapeBits !== undefined) && (obj.fShapeId !== undefined)) {
      shape = obj;
   }

   if (volume) {
      shape = volume.fShape;
      subnodes = volume.fNodes ? volume.fNodes.arr : null;
   }

   if (volume || shape || subnodes) {
      if (volume) sub._volume = volume;

      if (subnodes) {
         sub._more = true;
         sub._expand = expandGeoObject;
      } else
      if (shape && (shape._typename === "TGeoCompositeShape") && shape.fNode) {
         sub._more = true;
         sub._shape = shape;
         sub._expand = function(node /*, obj */) {
            createItem(node, node._shape.fNode.fLeft, 'Left');
            createItem(node, node._shape.fNode.fRight, 'Right');
            return true;
         };
      }

      if (!sub._title && (obj._typename != "TGeoVolume")) sub._title = obj._typename;

      if (shape) {
         if (sub._title == "")
            sub._title = shape._typename;

         sub._icon = getShapeIcon(shape);
      } else {
         sub._icon = sub._more ? "img_geocombi" : "img_geobbox";
      }

      if (volume)
         sub._icon += provideVisStyle(volume);
      else if (iseve)
         sub._icon += provideVisStyle(obj);

      sub._menu = provideMenu;
      sub._icon_click  = browserIconClick;
   }

   if (!node._childs) node._childs = [];

   if (!sub._name)
      if (typeof node._name === 'string') {
         sub._name = node._name;
         if (sub._name.lastIndexOf("s")===sub._name.length-1)
            sub._name = sub._name.slice(0, sub._name.length-1);
         sub._name += "_" + node._childs.length;
      } else {
         sub._name = "item_" + node._childs.length;
      }

   node._childs.push(sub);

   return sub;
}


/** @summary Draw dummy geometry
  * @private */
function drawDummy3DGeom(painter) {

   let extra = painter.getObject(),
       min = [-1, -1, -1], max = [1, 1, 1];

   if (extra.fP && extra.fP.length)
      for(let k = 0; k < extra.fP.length; k += 3)
         for (let i = 0; i < 3; ++i) {
            min[i] = Math.min(min[i], extra.fP[k+i]);
            max[i] = Math.max(max[i], extra.fP[k+i]);
         }


   let shape = create('TNamed');
   shape._typename = "TGeoBBox";
   shape.fDX = max[0] - min[0];
   shape.fDY = max[1] - min[1];
   shape.fDZ = max[2] - min[2];
   shape.fShapeId = 1;
   shape.fShapeBits = 0;
   shape.fOrigin = [0,0,0];

   let obj = create("TEveGeoShapeExtract");

   Object.assign(obj, { fTrans: [1,0,0,0, 0,1,0,0, 0,0,1,0, (min[0]+max[0])/2, (min[1]+max[1])/2, (min[2]+max[2])/2, 0],
                        fShape: shape, fRGBA: [0, 0, 0, 0], fElements: null, fRnrSelf: false });

   let opt = "", pp = painter.getPadPainter();

   if (pp?.pad?.fFillColor && pp?.pad?.fFillStyle > 1000)
      opt = "bkgr_" +  pp.pad.fFillColor;

   return TGeoPainter.draw(painter.getDom(), obj, opt)
                     .then(geop => geop.drawExtras(extra));
}

/** @summary Direct draw function for TAxis3D
  * @private */
function drawAxis3D() {
   let main = this.getMainPainter();

   if (typeof main?.setAxesDraw == 'function')
      return main.setAxesDraw(true);

   console.error('no geometry painter found to toggle TAxis3D drawing');
}

/** @summary Build three.js model for given geometry object
  * @param {Object} obj - TGeo-related object
  * @param {Object} [opt] - options
  * @param {Number} [opt.vislevel] - visibility level like TGeoManager, when not specified - show all
  * @param {Number} [opt.numnodes=1000] - maximal number of visible nodes
  * @param {Number} [opt.numfaces=100000] - approx maximal number of created triangles
  * @param {boolean} [opt.doubleside=false] - use double-side material
  * @param {boolean} [opt.wireframe=false] - show wireframe for created shapes
  * @param {boolean} [opt.dflt_colors=false] - use default ROOT colors
  * @returns {object} Object3D with created model
  * @example
  * import { build } from './path_to_jsroot/modules/geom/TGeoPainter.mjs';
  * let obj3d = build(obj);
  * // this is three.js object and can be now inserted in the scene
  */
function build(obj, opt) {

   if (!obj) return null;

   if (!opt) opt = {};
   if (!opt.numfaces) opt.numfaces = 100000;
   if (!opt.numnodes) opt.numnodes = 1000;
   if (!opt.frustum) opt.frustum = null;

   opt.res_mesh = opt.res_faces = 0;

   let shape = null, hide_top = false;

   if (('fShapeBits' in obj) && ('fShapeId' in obj)) {
      shape = obj; obj = null;
   } else if ((obj._typename === 'TGeoVolumeAssembly') || (obj._typename === 'TGeoVolume')) {
      shape = obj.fShape;
   } else if ((obj._typename === "TEveGeoShapeExtract") || (obj._typename === "ROOT::Experimental::REveGeoShapeExtract")) {
      shape = obj.fShape;
   } else if (obj._typename === 'TGeoManager') {
      obj = obj.fMasterVolume;
      hide_top = !opt.showtop;
      shape = obj.fShape;
   } else if (obj.fVolume) {
      shape = obj.fVolume.fShape;
   } else {
      obj = null;
   }

   if (opt.composite && shape && (shape._typename == 'TGeoCompositeShape') && shape.fNode)
      obj = buildCompositeVolume(shape);

   if (!obj && shape)
      obj = Object.assign(create("TEveGeoShapeExtract"),
                { fTrans: null, fShape: shape, fRGBA: [0, 1, 0, 1], fElements: null, fRnrSelf: true });

   if (!obj) return null;

   if (obj._typename.indexOf('TGeoVolume') === 0)
      obj = { _typename: "TGeoNode", fVolume: obj, fName: obj.fName, $geoh: obj.$geoh, _proxy: true };

   let clones = new ClonedNodes(obj);
   clones.setVisLevel(opt.vislevel);
   clones.setMaxVisNodes(opt.numnodes);

   if (opt.dflt_colors)
      clones.setDefaultColors(true);

   let uniquevis = opt.no_screen ? 0 : clones.markVisibles(true);
   if (uniquevis <= 0)
      clones.markVisibles(false, false, hide_top);
   else
      clones.markVisibles(true, true, hide_top); // copy bits once and use normal visibility bits

   clones.produceIdShifts();

   // collect visible nodes
   let res = clones.collectVisibles(opt.numfaces, opt.frustum);

   // collect shapes
   let shapes = clones.collectShapes(res.lst);

   clones.buildShapes(shapes, opt.numfaces);

   let toplevel = new Object3D();

   for (let n = 0; n < res.lst.length; ++n) {
      let entry = res.lst[n];
      if (entry.done) continue;

      let shape = shapes[entry.shapeid];
      if (!shape.ready) {
         console.warn('shape marked as not ready when should');
         break;
      }
      entry.done = true;
      shape.used = true; // indicate that shape was used in building

      if (!shape.geom || (shape.nfaces === 0)) {
         // node is visible, but shape does not created
         clones.createObject3D(entry.stack, toplevel, 'delete_mesh');
         continue;
      }

      let prop = clones.getDrawEntryProperties(entry, getRootColors());

      opt.res_mesh++;
      opt.res_faces += shape.nfaces;

      let obj3d = clones.createObject3D(entry.stack, toplevel, opt);

      prop.material.wireframe = opt.wireframe;

      prop.material.side = opt.doubleside ? DoubleSide : FrontSide;

      let mesh = null, matrix = obj3d.absMatrix || obj3d.matrixWorld;

      if (matrix.determinant() > -0.9) {
         mesh = new Mesh(shape.geom, prop.material);
      } else {
         mesh = createFlippedMesh(shape, prop.material);
      }

      mesh.name = clones.getNodeName(entry.nodeid);

      obj3d.add(mesh);

      if (obj3d.absMatrix) {
         mesh.matrix.copy(obj3d.absMatrix);
         mesh.matrix.decompose( obj3d.position, obj3d.quaternion, obj3d.scale );
         mesh.updateMatrixWorld();
      }

      // specify rendering order, required for transparency handling
      //if (obj3d.$jsroot_depth !== undefined)
      //   mesh.renderOrder = clones.maxdepth - obj3d.$jsroot_depth;
      //else
      //   mesh.renderOrder = clones.maxdepth - entry.stack.length;
   }

   return toplevel;
}

export { ClonedNodes, build, TGeoPainter, GeoDrawingControl,
         expandGeoObject, createGeoPainter, drawAxis3D, drawDummy3DGeom, produceRenderOrder };
