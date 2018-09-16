
#version 150

uniform float time;
uniform vec2 resolution;
uniform vec2 mouse;
uniform vec3 spectrum;

uniform sampler2D texture0;
uniform sampler2D texture1;
uniform sampler2D texture2;
uniform sampler2D texture3;
uniform sampler2D prevFrame;
uniform sampler2D prevPass;

in VertexData
{
    vec4 v_position;
    vec3 v_normal;
    vec2 v_texcoord;
} inData;

out vec4 fragColor;


const float epsilon = 0.001;
const float PI = 3.14159265;

struct Material{
    int id;
    vec3 albedo;
    vec3 metallic;
    float roughness;
    int texId;
};



struct DistVal {
    float dist;
    Material material;
};

// util func
//-------------------------------------------------------------------------------------
float deg2rad(float angle){
    return angle * PI / 180.0;
}

//x 0-1
float saturate(float x){
    return clamp(x,0.0,1.0);
}

vec3 LinearToGamma( in vec3 value, in float gammaFactor ) {
    return pow( value, vec3(1.0 / gammaFactor) );
}



//texture func
//-------------------------------------------------------------------------------------
vec3 texture(int texId, vec3 pos){
    vec3 col = vec3(1.0);
    if(texId == 1){
        float u = 1.0 - floor(mod(pos.x, 2.0));
        float v = 1.0 - floor(mod(pos.z, 2.0));
        if((u == 1.0 && v < 1.0) || (u < 1.0 && v == 1.0)){
            col *= 0.1;
        }
        return col;
    }
    
    return col;
}


//pbr
//-------------------------------------------------------------------------------------

struct GeometricContext {
    vec3 position;
    vec3 normal;
    vec3 viewDir;
};

struct IncidentLight {
    vec3 direction;
    vec3 color;
    bool visible;
};

struct DirectionalLight {
    vec3 direction;
    vec3 color;
};

struct PointLight {
    vec3 position;
    vec3 color;
    float visible_distance;
    float decay;
};

struct ReflectedLight {
    vec3 directDiffuse;
    vec3 directSpecular;
    vec3 indirectDiffuse;
    vec3 indirectSpecular;
};

//test lighth visible
bool testLightInRange(const in float lightDistance, const in float cutoffDistance) {
    return any(bvec2(cutoffDistance == 0.0, lightDistance < cutoffDistance));
}

//light decay by distanc
float punctualLightIntensityToIrradianceFactor(const in float lightDistance, const in float cutoffDistance, const in float decayExponent) {
    if (decayExponent > 0.0) {
        return pow(saturate(-lightDistance / cutoffDistance + 1.0), decayExponent);
    }
    
    return 1.0;
}

//directional light irradiance
void getDirectionalDirectLightIrradiance(const in DirectionalLight directionalLight, out IncidentLight directLight) {
    directLight.color = directionalLight.color;
    directLight.direction = directionalLight.direction;
    directLight.visible = true;
}

//point light irradiance
void getPointDirectLightIrradiance(const in PointLight pointLight, const in vec3 geometryPosition, out IncidentLight directLight) {
    vec3 L = pointLight.position - geometryPosition;
    directLight.direction = normalize(L);
    
    float lightDistance = length(L);
    if (testLightInRange(lightDistance, pointLight.visible_distance)) {
        directLight.color = pointLight.color;
        directLight.color *= punctualLightIntensityToIrradianceFactor(lightDistance, pointLight.visible_distance, pointLight.decay);
        directLight.visible = true;
    } else {
        directLight.color = vec3(0.0);
        directLight.visible = false;
    }
}

//material from colors
vec3 diffuseColor(vec3 albedo, vec3 metallic){
    return mix(albedo, vec3(0.0), metallic);
}

vec3 specularColor(vec3 albedo, vec3 metallic){
    return mix(vec3(0.04), albedo, metallic);
}

// Normalized Lambert
vec3 DiffuseBRDF(vec3 diffuseColor) {
    return diffuseColor / PI;
}



vec3 F_Schlick(vec3 specularColor, vec3 H, vec3 V) {
    return (specularColor + (1.0 - specularColor) * pow(1.0 - saturate(dot(V,H)), 5.0));
}

float D_GGX(float a, float dotNH) {
    float a2 = a*a;
    float dotNH2 = dotNH*dotNH;
    float d = dotNH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

float G_Smith_Schlick_GGX(float a, float dotNV, float dotNL) {
    float k = a*a*0.5 + epsilon;
    float gl = dotNL / (dotNL * (1.0 - k) + k);
    float gv = dotNV / (dotNV * (1.0 - k) + k);
    return gl*gv;
}

// Cook-Torrance
vec3 SpecularBRDF(const in IncidentLight directLight, const in GeometricContext geometry, vec3 specularColor, float roughnessFactor) {
    
    vec3 N = geometry.normal;
    vec3 V = geometry.viewDir;
    vec3 L = directLight.direction;
    
    float dotNL = saturate(dot(N,L));
    float dotNV = saturate(dot(N,V));
    vec3 H = normalize(L+V);
    float dotNH = saturate(dot(N,H));
    float dotVH = saturate(dot(V,H));
    float dotLV = saturate(dot(L,V));
    float a = roughnessFactor * roughnessFactor;
    
    float D = D_GGX(a, dotNH);
    float G = G_Smith_Schlick_GGX(a, dotNV, dotNL);
    vec3 F = F_Schlick(specularColor, V, H);
    
    return (F*(G*D))/(4.0*dotNL*dotNV+epsilon);
}

// RenderEquations(RE)
void RE_Direct(const in IncidentLight directLight, const in GeometricContext geometry, const in Material material, inout ReflectedLight reflectedLight) {
    
    float dotNL = saturate(dot(geometry.normal, directLight.direction));
    vec3 irradiance = dotNL * directLight.color;
    
    // punctual light
    irradiance *= PI;
    
    vec3 diffuse = diffuseColor(material.albedo, material.metallic);
    vec3 specular = specularColor(material.albedo, material.metallic);
    
    reflectedLight.directDiffuse += irradiance * DiffuseBRDF(diffuse);
    //    reflectedLight.directSpecular += specular;
    reflectedLight.directSpecular += irradiance * SpecularBRDF(directLight, geometry, specular, material.roughness);
}


// dist func
//-------------------------------------------------------------------------------------

// box distance fuction
float distBox(vec3 p, vec3 size){
    vec3 q = abs(p);
    return length(max(q - size, 0.0));
}


// shere distance fuction
float distSphere(vec3 pos, float size, vec3 sPos)
{
    return length(pos - sPos) - size;
}

vec3 onRep(vec3 p, float interval){
    return mod(p, interval) - interval * 0.5;
}



// scene
//-------------------------------------------------------------------------------------

const float rayLoopCount = 256;
const float shadowLoopCount = 64;
const float aoLoopCount = 64;
vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));

const int LIGHT_MAX = 4;
DirectionalLight directionalLights[LIGHT_MAX];
PointLight pointLights[LIGHT_MAX];
int numPointLights;
int numDirectionalLights;


struct Camera{
    vec3 position;
    float fov;
    float farClip;
};

uniform Camera cam;


DistVal minDist(float distA, float distB, Material matA, Material matB){
    DistVal val;
    val.dist = min(distA, distB);
    val.material = distA < distB ? matA : matB;
    return val;
}


DistVal sceneDraw(vec3 pos){
    Material sphereMat1;
    Material sphereMat2;
    Material sphereMat3;
    
    sphereMat1.id = 0;
    sphereMat1.albedo = vec3(1.0, 1.0, 1.0);
    sphereMat1.metallic = vec3(0.0);
    sphereMat1.roughness = 0.5;
    sphereMat1.texId = 0;
    
    
    
    sphereMat2 = sphereMat1;
    sphereMat2.metallic = vec3(0.2);
    sphereMat2.roughness = 0.6;
    sphereMat3 = sphereMat1;
    sphereMat3.metallic = vec3(1.0);
    sphereMat3.roughness = 0.5;
    
    
    Material floorMat;
    floorMat.id = 1;
    floorMat.albedo = vec3(1.0, 1.0, 1.0);
    floorMat.metallic = vec3(0.1);
    floorMat.roughness = 0.4;
    floorMat.texId = 1;
    
    float floor_ = distBox(pos, vec3(10, 0.1, 10));
    Material mat;
    float sphere1 = distSphere(onRep(pos, 0.0), 0.5, vec3(0.0, 0.5, 0.0));
    float sphere2 = distSphere(onRep(pos, 0.0), 0.5, vec3(-2.0, 0.5, 0.0));
    float sphere3 = distSphere(onRep(pos, 0.0), 0.5, vec3(2.0, 0.5, 0.0));
    DistVal sphere12 = minDist(sphere1, sphere2, sphereMat1, sphereMat2);
    DistVal sphere = minDist(sphere12.dist, sphere3, sphere12.material, sphereMat3);
    DistVal spereFloor = minDist(sphere.dist, floor_, sphere.material, floorMat);
    
    
    return spereFloor;
}

float sceneDist(vec3 pos){
    DistVal val = sceneDraw(pos);
    return val.dist;
}

vec3 getNormal(vec3 pos)
{
    return normalize(vec3(
                          sceneDist(pos + vec3(epsilon, 0., 0.)) - sceneDist(pos - vec3(epsilon, 0., 0.)),
                          sceneDist(pos + vec3(0., epsilon, 0.)) - sceneDist(pos - vec3(0., epsilon, 0.)),
                          sceneDist(pos + vec3(0., 0., epsilon)) - sceneDist(pos - vec3(0., 0., epsilon))
                          ));
}

float genShadow(vec3 ro, vec3 rd){
    float h = 0.0;
    float c = 0.001;
    float r = 1.0;
    float shadowCoef = 0.5;
    for(float t=0.0; t<shadowLoopCount; t++){
        h = sceneDist(ro+rd*c);
        if(h<epsilon){
            return shadowCoef;
        }
        r = min(r, h * 10.0 / c);
        c += h;
    }
    return 1.0 - shadowCoef + r * shadowCoef;
}

float genAo(vec3 ro, vec3 rd){
    float k = 1.0;
    float occ = 0.0;
    for(float i=0; i<aoLoopCount; i++){
        float len = 0.15 + i * 0.15;
        float dist = sceneDist(rd * len + ro);
        occ += (len - dist) * k;
        k *= 0.5;
    }
    return saturate(1.0-occ);
}




vec3 trace(vec2 uv){
    vec3 col = vec3(0.0);
    
    //trace
    float fov = deg2rad(cam.fov / 2);
    
    vec3 lookAt = vec3(0.,0.,0.);
    vec3 forward = normalize(lookAt-cam.position);
    vec3 right = normalize(vec3(forward.z, 0., -forward.x ));
    vec3 up = normalize(cross(forward,right));
    
    vec3 ray = normalize(forward + fov*uv.x*right + fov*uv.y*up);
    vec3 cur = cam.position;
    
    
    DistVal result;
    for (int i = 0; i < rayLoopCount; i++)
    {
        result = sceneDraw(cur);
        
        if (result.dist < epsilon)
        {
            
            break;
        }
        cur += ray * result.dist;
        
        if(distance(cur, cam.position) > cam.farClip){
            break;
        }
    }
    
    if(result.dist >= epsilon){
        return vec3(0.0);
    }
    
    vec3 normal = getNormal(cur);
    float depth = distance(cur, cam.position)/cam.farClip;
    
    vec3 surfacePos = cur + normal * epsilon;
    
    
    
    
    //new
    GeometricContext geometry;
    geometry.position = surfacePos;
    geometry.normal = normal;
    geometry.viewDir = normalize(cam.position-surfacePos);
    
    Material material = result.material;
    material.albedo *= texture(material.texId, surfacePos);
    ReflectedLight reflectedLight = ReflectedLight(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0));
    vec3 emissive = vec3(0.0);
    float opacity = 1.0;
    
    IncidentLight directLight;
    
    // point light
    for (int i=0; i<LIGHT_MAX; ++i) {
        if (i >= numPointLights) break;
        getPointDirectLightIrradiance(pointLights[i], geometry.position, directLight);
        if (directLight.visible) {
            RE_Direct(directLight, geometry, material, reflectedLight);
        }
    }
    
    
    
    // directional light
    for (int i=0; i<LIGHT_MAX; ++i) {
        if (i >= numDirectionalLights) break;
        getDirectionalDirectLightIrradiance(directionalLights[i], directLight);
        RE_Direct(directLight, geometry, material, reflectedLight);
    }
    
    
    vec3 specular = reflectedLight.directSpecular + reflectedLight.indirectSpecular;
    vec3 diffuse = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse;
    vec3 ambient = vec3(0.04);
    col = vec3(0.0);
    col += emissive;
    col += specular;
    col += diffuse;
    
    col += ambient;
    
    //ambient occlusion
    float ao = genAo(surfacePos, normal);
    col *= ao;
    //shadow
    float shadow = genShadow(surfacePos, lightDir);
    col = col * max(0.5, shadow);
    
    col = LinearToGamma(col, 1.1);
    
    return col;
}



void main(void)
{
    
    DirectionalLight directionalLight;
    directionalLight.direction = vec3(1.0, 1.3, 0.4);
    directionalLight.color = vec3(1.0);
    
    directionalLights[0] = directionalLight;
    numDirectionalLights = 1;
    
    PointLight pointLight;
    pointLight.visible_distance = 4.0;
    pointLight.decay = 1.1;
    for(int i=0; i<LIGHT_MAX; i++){
        pointLights[i] = pointLight;
        if(i==0){
            pointLights[i].position = vec3(0.0, 2.0, 0.0);
            pointLights[i].color = vec3(1.0, 0.0, 0.0);
        }
        if(i==1){
            pointLights[i].position = vec3(-3.0, 2.0, 0.0);
            pointLights[i].color = vec3(0.0, 1.0, 0.0);
        }
        if(i==2){
            pointLights[i].position = vec3(3.0, 2.0, 0.0);
            pointLights[i].color = vec3(0.0, 0.0, 1.0);
        }
        
        
    }
    numPointLights = 4;
    
    
    //setup
    vec2 texCoord = inData.v_texcoord;
    vec2 fragCoord = texCoord * resolution;
    vec2 uv = (fragCoord * 2.0 - resolution) / min(resolution.x, resolution.y);
    
    fragColor = vec4(trace(uv), 1.0);
}
