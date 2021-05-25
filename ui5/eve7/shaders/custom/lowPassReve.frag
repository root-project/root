#version 300 es
precision mediump float;


struct Material {
    #if (TEXTURE)
        sampler2D texture0;
    #fi
};


uniform Material material;

#if (TEXTURE)
    in vec2 fragUV;
#fi

//out vec4 color[2];
out vec4 color;


void main() {
	// check whether fragment output is higher than threshold, if so output as brightness color
	float brightness = dot(texture(material.texture0, fragUV).rgb, vec3(0.2126, 0.7152, 0.0722));

	if(brightness > 0.75){
		//color[0] = texture(material.texture0, fragUV); //texture(material.texture##I_TEX, fragUV)
		color = texture(material.texture0, fragUV); //texture(material.texture##I_TEX, fragUV)
	}else{
		//color[0] = vec4(0.0, 0.0, 0.0, 1.0);
		color = vec4(0.0, 0.0, 0.0, 1.0);
	}


	//COPY
	//color[1] = texture(material.texture0, fragUV);
}
