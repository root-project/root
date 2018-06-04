// import { THREE.BufferGeometry } from 'threejs';

// import { BufferGeometry } from '../core/BufferGeometry.js';
// import { Float32BufferAttribute } from '../core/BufferAttribute.js';
// import { Vector3 } from '../math/Vector3.js';
// import { Vector2 } from '../math/Vector2.js';

function EveJetConeGeometry(vertices)
{
    THREE.BufferGeometry.call( this );

    this.addAttribute( 'position', new THREE.BufferAttribute( vertices, 3 ) );

    var N = vertices.length / 3;
    var idcs = [];
    for (var i = 1; i < N - 1; ++i)
    {
        idcs.push( i ); idcs.push( 0 ); idcs.push( i + 1 );
    }
    this.setIndex( idcs );
}

EveJetConeGeometry.prototype = Object.create( THREE.BufferGeometry.prototype );
EveJetConeGeometry.prototype.constructor = EveJetConeGeometry;

export { EveJetConeGeometry };
