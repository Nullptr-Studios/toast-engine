// Environment lighting: the sky cubemap, and the two precomputed cubemaps derived from it
//
// Three entry points run once at startup; the fourth draws the sky each frame. Split-sum, because the
// integral depends only on the environment and so can be done ahead of time

struct EnvironmentParams
{
    // Basis of the cube face being rendered, so a full-screen triangle can be turned into directions.
    // Unused by the skybox entry point, which gets its direction from the camera instead
    float4 faceRight;
    float4 faceUp;
    float4 faceForward;

    // Prefilter only: which roughness this mip represents
    float roughness;

    // Sky only: how bright the sky is, folded in here so the whole chain rebuilds from one value
    float intensity;

    float2 _pad0;
};

[[vk::push_constant]]
EnvironmentParams gParams;

// The source cubemap for the convolution and prefilter steps
[[vk::binding(0,0)]]
SamplerCube gEnvironment;

// An imported equirectangular HDR, read only by fragmentEquirect while building the sky cubemap. Bound to a
// 1x1 placeholder when no environment has been imported, so the descriptor set is always complete
[[vk::binding(1,0)]]
Sampler2D gEquirectangular;

struct VSOutput
{
    float4 position : SV_Position;

    [[vk::location(0)]]
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VSOutput vertexMain(uint vertexId : SV_VertexID)
{
    VSOutput output;
    output.uv = float2((vertexId << 1) & 2, vertexId & 2);
    output.position = float4(output.uv * 2.0 - 1.0, 0.0, 1.0);
    return output;
}

// UV across a cube face -> the world direction that face's texel looks along
float3 directionForFace(float2 uv)
{
    float2 ndc = uv * 2.0 - 1.0;
    return normalize(gParams.faceForward.xyz + gParams.faceRight.xyz * ndc.x + gParams.faceUp.xyz * ndc.y);
}

static const float kPi = 3.14159265359;

// A plain gradient sky: ground, horizon, zenith. Deliberately simple - this exists so the whole IBL chain
// has something real to consume, and swapping it for an imported HDR later replaces this function alone
float3 skyRadiance(float3 direction)
{
    // Z-up, matching the engine's world convention
    float height = direction.z;

    const float3 groundColor = float3(0.12, 0.11, 0.10);
    const float3 horizonColor = float3(0.55, 0.60, 0.70);
    const float3 zenithColor = float3(0.18, 0.32, 0.62);

    // Sharper falloff below the horizon than above it, which is what keeps the ground from washing the
    // lower hemisphere out
    float3 color = height > 0.0
        ? lerp(horizonColor, zenithColor, pow(saturate(height), 0.45))
        : lerp(horizonColor, groundColor, saturate(-height * 4.0));

    return color * gParams.intensity;
}

[shader("fragment")]
float4 fragmentSky(VSOutput input) : SV_Target0
{
    return float4(skyRadiance(directionForFace(input.uv)), 1.0);
}

// Projects an imported equirectangular HDR onto one cube face
//
// Resolved once so everything downstream reads the same cubemap and none of it knows where the sky came
// from. atan2 wraps at the seam behind the camera, so this texture's U address mode is repeat rather than
// clamp-to-edge, which would smear that column
[shader("fragment")]
float4 fragmentEquirect(VSOutput input) : SV_Target0
{
    float3 direction = normalize(directionForFace(input.uv));

    const float k_inv_two_pi = 0.1591549431;
    const float k_inv_pi = 0.3183098862;

    float2 uv = float2(atan2(direction.z, direction.x) * k_inv_two_pi + 0.5, acos(clamp(direction.y, -1.0, 1.0)) * k_inv_pi);

    // SampleLevel, not Sample: this runs over a cube face whose UV derivatives jump across the seam, and a
    // mip chosen from those derivatives would band there. There is only one level anyway
    return float4(gEquirectangular.SampleLevel(uv, 0.0).rgb * gParams.intensity, 1.0);
}

// Same full-screen triangle, pinned to the far plane instead of the near one
//
// Recorded after the geometry, so it has to lose the depth test wherever something was drawn. At z = 1 with
// a less-or-equal test it survives exactly where depth is still cleared
[shader("vertex")]
VSOutput vertexSkybox(uint vertexId : SV_VertexID)
{
    VSOutput output;
    output.uv = float2((vertexId << 1) & 2, vertexId & 2);
    output.position = float4(output.uv * 2.0 - 1.0, 1.0, 1.0);
    return output;
}

// The sky as the scene's backdrop, drawn behind everything each frame
//
// Samples the cubemap rather than calling skyRadiance() again, so background and reflections are the same
// data by construction. The caller passes right/up pre-scaled by the projection, so directionForFace()
// turns the triangle into view rays with no extra maths
[shader("fragment")]
float4 fragmentSkybox(VSOutput input) : SV_Target0
{
    return float4(gEnvironment.SampleLevel(directionForFace(input.uv), 0.0).rgb, 1.0);
}

// Cosine-weighted convolution of the environment: for every direction, the light arriving from the whole
// hemisphere around it. This is the diffuse half of image-based lighting, and it is why a surface facing a
// bright sky picks up sky colour without any light source existing
[shader("fragment")]
float4 fragmentIrradiance(VSOutput input) : SV_Target0
{
    float3 normal = directionForFace(input.uv);

    // Tangent frame around the normal to walk the hemisphere in
    float3 up = abs(normal.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 right = normalize(cross(up, normal));
    up = cross(normal, right);

    float3 irradiance = float3(0.0, 0.0, 0.0);
    float sampleCount = 0.0;

    // Fixed-step spherical sweep. Runs once at startup on a 32x32 face, so brute force is fine and avoids
    // the noise an importance-sampled version would need many more samples to clear
    const float deltaPhi = 0.025;
    const float deltaTheta = 0.025;
    for (float phi = 0.0; phi < 2.0 * kPi; phi += deltaPhi)
    {
        for (float theta = 0.0; theta < 0.5 * kPi; theta += deltaTheta)
        {
            float3 tangentSample = float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
            float3 worldSample = tangentSample.x * right + tangentSample.y * up + tangentSample.z * normal;

            // cos(theta) for the projected area, sin(theta) for the shrinking area of each ring
            irradiance += gEnvironment.SampleLevel(worldSample, 0.0).rgb * cos(theta) * sin(theta);
            sampleCount += 1.0;
        }
    }

    return float4(kPi * irradiance / max(sampleCount, 1.0), 1.0);
}

float radicalInverseVdC(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

float2 hammersley(uint i, uint count)
{
    return float2(float(i) / float(count), radicalInverseVdC(i));
}

// Draws a direction from the GGX distribution for this roughness, around the given normal
float3 importanceSampleGGX(float2 random, float3 normal, float roughness)
{
    float a = roughness * roughness;

    float phi = 2.0 * kPi * random.x;
    float cosTheta = sqrt((1.0 - random.y) / (1.0 + (a * a - 1.0) * random.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    float3 halfVector = float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

    float3 up = abs(normal.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 tangent = normalize(cross(up, normal));
    float3 bitangent = cross(normal, tangent);

    return normalize(tangent * halfVector.x + bitangent * halfVector.y + normal * halfVector.z);
}

// Prefiltered radiance for one roughness level, written into one mip of the prefiltered chain
//
// Split-sum assumes view == normal, so the result depends only on reflection direction and roughness. It
// costs the stretched highlights a grazing view should show
[shader("fragment")]
float4 fragmentPrefilter(VSOutput input) : SV_Target0
{
    float3 normal = directionForFace(input.uv);
    float3 reflection = normal;
    float3 view = normal;

    // Mip 0 is the sky itself. Taken directly because at roughness 0 the GGX distribution collapses to a
    // delta and all 128 samples degenerate onto the normal through expressions only defined in the limit
    if (gParams.roughness <= 0.0)
    {
        return float4(gEnvironment.SampleLevel(normal, 0.0).rgb, 1.0);
    }

    const uint kSampleCount = 128u;
    float3 prefiltered = float3(0.0, 0.0, 0.0);
    float totalWeight = 0.0;

    for (uint i = 0u; i < kSampleCount; ++i)
    {
        float2 random = hammersley(i, kSampleCount);
        float3 halfVector = importanceSampleGGX(random, normal, gParams.roughness);
        float3 light = normalize(2.0 * dot(view, halfVector) * halfVector - view);

        float NdotL = dot(normal, light);
        if (NdotL <= 0.0)
        {
            continue;
        }

        // Weighted by NdotL: samples nearer the reflection direction matter more, and this is what keeps
        // the result from biasing towards the edges of the lobe
        prefiltered += gEnvironment.SampleLevel(light, 0.0).rgb * NdotL;
        totalWeight += NdotL;
    }

    return float4(prefiltered / max(totalWeight, 1e-4), 1.0);
}
