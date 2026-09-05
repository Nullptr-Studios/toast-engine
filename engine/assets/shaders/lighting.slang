// Scene lighting: the shared half of the shading model
//
// Takes a described Surface and returns it lit. mesh.slang samples that surface out of its material
// textures; anything else able to describe a surface the same way shades identically
//
// Set 0 below is the engine's; an importing shader inherits it at these numbers and owns set 1 itself

struct DirectionalLightGpu
{
    float4 direction;        // world-space, w unused
    float4 colorIntensity;   // rgb color, a intensity
};

// Reprojects the environment onto nearby geometry - see toast::ReflectionProbe
//
// Mirrors VulkanRenderer::IrradianceVolumeData
struct IrradianceVolumeGpu
{
    // xyz world position of the grid's minimum corner, w nominal probe spacing (the true per-axis step is
    // derived from extents and counts, so a grid always spans its box exactly)
    float4 minCornerSpacing;

    // xyz probe count per axis, w index of this volume's first probe in gIrradianceSH
    uint4 countsBase;

    // xyz half-extents, w intensity
    float4 extentsIntensity;

    // x = 1 once this volume's slice of gIrradianceSH holds its own coefficients, 0 while it is unbaked or
    // baked for a grid it no longer has. The buffer is shared and never cleared between scenes, so without
    // this a fresh volume samples whatever the previous scene left at its base index
    uint4 bakedPad;
};

// Mirrors VulkanRenderer::ReflectionProbeData
struct ReflectionProbeGpu
{
    // xyz world position, w influence radius
    float4 positionRadius;

    // xyz half-extents of the projection box, w intensity. Zero extents select sphere projection
    float4 boxExtentsIntensity;

    // x = which cubemap in gProbeMaps/gProbeIrradiance this probe owns. This array is sorted by camera
    // distance every frame while the cubemaps are not, so the slot a probe occupies here is not its identity
    float4 params;
};

struct CameraUBO
{
    float4x4 view;
    float4x4 projection;
    float4x4 viewProjection;

    float3 position;
    float time;

    float4 ambientColorIntensity;    // rgb color, a intensity; zero if there's no AmbientLight
    uint4 directionalLightCountPad;  // x = directionalLightCount, yzw unused padding
    DirectionalLightGpu directionalLights[4];
    uint4 renderModePad;             // x = render mode (0 Lit, 1 ClusterHeatmap), yzw unused padding

    // World -> shadow clip space per cascade of the shadow-casting DirectionalLight, fitted each frame in
    // VulkanRenderer::tick(). Must match renderer::shadows::k_cascade_count
    float4x4 cascadeViewProjection[4];

    float4 cascadeSplits;          // view-space far distance of each cascade
    float4 cascadeTexelWorldSize;  // world size of one texel in each cascade, scales the normal offset
    float4 cascadeDepthBias;       // constant bias per cascade, in that cascade's depth units

    // x = shadow-casting directional light index + 1, 0 when nothing casts this frame; y = cascade count;
    // z = one cascade texel in UV units; w = one punctual-map texel in UV units
    float4 shadowParams;

    // x = 1 when the environment cubemaps hold real data, 0 while they do not; y = how many roughness levels
    // gPrefilteredEnv has, which roughness is mapped onto; zw reserved
    float4 environmentParams;

    // x = probe count, y = roughness levels in a probe cubemap, z = bit per probe marking it baked, w padding
    uint4 reflectionProbeCountPad;

    // Must match VulkanRenderer::k_max_reflection_probes
    ReflectionProbeGpu reflectionProbes[8];

    // x = volume count, yzw padding
    uint4 irradianceVolumeCountPad;

    // Must match VulkanRenderer::k_max_irradiance_volumes
    IrradianceVolumeGpu irradianceVolumes[4];

    // x = 1 when gScene holds a built TLAS and traced shadows are on, 0 otherwise; yzw reserved. Appended
    // last so every offset above stays put for the shaders mirroring only a prefix of this
    float4 tracedShadowParams;
};

[[vk::binding(0,0)]]
ConstantBuffer<CameraUBO> gCamera;

struct ClusterParams
{
    float4x4 inverseProjection;
    uint4 clusterDims;         // x,y,z dims, w = maxLightsPerCluster
    float4 screenSizeNearFar;  // x,y screen dims, z near, w far
    uint lightCount;
    float3 _pad0;
};

struct GpuLight
{
    float4 worldPosRange;   // xyz world-space position, w = attenuation range
    float4 viewPosType;     // xyz view-space position (culling only), w = type (0=point,1=spot)
    float4 colorIntensity;  // rgb color, a intensity
    float4 directionPad;    // xyz world-space direction (spot only), w = shadow layer index, -1 = not casting
    float4 coneAngles;      // x = cos(outer), y = cos(inner), z = shadow near, w = shadow far

    // x = fraction of the shadow layer this light rendered into; a distant light gets a smaller sub-rect
    // (see renderer::shadows::punctualShadowResolution), and every lookup scales into it. yzw reserved
    float4 shadowAtlas;

    // Spotlight only; a point light instead reconstructs its face depth from coneAngles' near/far
    float4x4 shadowViewProjection;
};

[[vk::binding(1,0)]]
ConstantBuffer<ClusterParams> gClusterParams;
[[vk::binding(2,0)]]
StructuredBuffer<GpuLight> gLights;
[[vk::binding(3,0)]]
StructuredBuffer<uint> gClusterLightGrid;
[[vk::binding(4,0)]]
StructuredBuffer<uint> gLightIndexList;

// Binding 5 used to be the shared joint pool this shader blended per vertex. skinning.slang owns that now -
// it poses every skinned instance into a vertex buffer before any pass records, so what arrives here is
// already deformed. See vertexMain's note about the identity model matrix that comes with it

// gShadowMap: one layer per directional cascade. gPunctualShadowMaps: one per shadow-casting Spotlight,
// then six per PointLight. Both fall back to a 1x1 white array, which reads as fully lit
//
// Depth-comparison samplers: the hardware compares at each of a bilinear tap's four texels and returns the
// filtered fraction, so one tap is already a soft edge. Comparing by hand gives 0 or 1 and stair-steps
[[vk::binding(6,0)]]
Sampler2DArrayShadow gShadowMap;

[[vk::binding(7,0)]]
Sampler2DArrayShadow gPunctualShadowMaps;

// Environment lighting, precomputed once by renderer::EnvironmentPass. gIrradianceMap is the cosine-
// convolved diffuse half; gPrefilteredEnv holds one roughness level per mip for the specular half. Both fall
// back to a 1x1 black cube when no environment exists, and environmentParams.x says which case this is
[[vk::binding(8,0)]]
SamplerCube gIrradianceMap;

[[vk::binding(9,0)]]
SamplerCube gPrefilteredEnv;

// A descriptor array rather than a cube array, since imageCubeArray is not enabled here. An unbaked slot
// holds the 1x1 black cube and probeMipCountPad.y marks it, so the shader falls back to the environment
[[vk::binding(10,0)]]
SamplerCube gProbeMaps[8];

// Cosine-convolved counterpart of gProbeMaps: the diffuse half of what each probe captured. Without it a
// probe corrects only what a surface reflects, leaving what it is *lit by* coming from the sky - so an
// interior keeps being bounce-lit by the outdoors it cannot see
[[vk::binding(11,0)]]
SamplerCube gProbeIrradiance[8];

// L1 spherical harmonics for every irradiance probe across every volume, four float4s each in the order
// L0, L1y, L1z, L1x. Written by sh_project.slang at bake time; addressed as volume base + grid index
[[vk::binding(12,0)]]
StructuredBuffer<float4> gIrradianceSH;

// Compiled in only where the device has ray query - SPIR-V declaring the capability is rejected at pipeline
// creation elsewhere, so it cannot be a runtime branch. Partially bound, because a frame with nothing
// traceable writes no structure and every access is gated on tracedShadowParams.x
#if TOAST_RAY_QUERY
[[vk::binding(14,0)]]
RaytracingAccelerationStructure gScene;
#endif

static const float kPi = 3.14159265359;

// Trowbridge-Reitz GGX normal distribution
float distributionGGX(float3 N, float3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = kPi * denom * denom;
    return a2 / max(denom, 0.0000001);
}

// Schlick-GGX geometry term, one side
float geometrySchlickGGX(float NdotV, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    float denom = NdotV * (1.0 - k) + k;
    return NdotV / max(denom, 0.0000001);
}

// Smith geometry term, both view and light sides
float geometrySmith(float3 N, float3 V, float3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggxV = geometrySchlickGGX(NdotV, roughness);
    float ggxL = geometrySchlickGGX(NdotL, roughness);
    return ggxV * ggxL;
}

float3 fresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (float3(1.0, 1.0, 1.0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Fresnel for environment lighting, where the "half vector" is not a single direction but a whole cone of
// them. Rough surfaces gather from so many directions that the grazing-angle rise is largely averaged away,
// which the plain Schlick form does not know about and over-brightens the silhouette of rough materials
float3 fresnelSchlickRoughness(float cosTheta, float3 F0, float roughness)
{
    float3 smoothness = float3(1.0 - roughness, 1.0 - roughness, 1.0 - roughness);
    return F0 + (max(smoothness, F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Split-sum environment BRDF: the scale and bias on F0 when integrating a whole environment
//
// Lazarov's analytic fit rather than the usual roughness x NdotV LUT - within a fraction of a percent, and
// costs no texture, no precompute and no asset
// Re-aims a reflection ray at the geometry a probe stands in for
//
// Sampling a cubemap by direction alone asserts its contents are infinitely far away. Indoors they are not,
// so reflections slide across surfaces as the camera moves. Intersecting a volume standing in for the room
// and re-aiming from the probe's centre pins them down
//
// @param probeExtents Half-extents of the box; all-zero selects a sphere of @p radius instead
float3 parallaxCorrectedReflection(
    float3 reflection, float3 worldPosition, float3 probePosition, float3 probeExtents, float radius)
{
    float3 toSurface = worldPosition - probePosition;

    if (all(probeExtents <= float3(0.0, 0.0, 0.0)))
    {
        // Sphere: the positive root of |toSurface + t*R| = radius
        float b = dot(toSurface, reflection);
        float c = dot(toSurface, toSurface) - radius * radius;
        float discriminant = (b * b) - c;
        if (discriminant <= 0.0)
        {
            return reflection;
        }
        float t = -b + sqrt(discriminant);
        return normalize(toSurface + reflection * t);
    }

    // Box: distance along the ray to each slab, keeping the nearest exit. Rays parallel to an axis divide by
    // zero here and produce +inf, which min() then discards - exactly the wanted behaviour, and why this is
    // written without a branch per axis
    float3 firstPlane = (probeExtents - toSurface) / reflection;
    float3 secondPlane = (-probeExtents - toSurface) / reflection;
    float3 furthest = max(firstPlane, secondPlane);
    float distance = min(min(furthest.x, furthest.y), furthest.z);

    return normalize(toSurface + reflection * distance);
}

/// @brief Falloff of a probe's authority at a point; 0 outside its influence
///
/// Squared, because a linear ramp still creases visibly where it reaches zero
float reflectionProbeWeight(float3 worldPosition, ReflectionProbeGpu probe)
{
    float radius = probe.positionRadius.w;
    if (radius <= 0.0)
    {
        return 0.0;
    }

    float3 local = worldPosition - probe.positionRadius.xyz;
    float3 extents = probe.boxExtentsIntensity.xyz;

    if (any(extents > float3(0.0, 0.0, 0.0)))
    {
        // No authority outside its box, whatever the influence radius says: the projection re-aims at the
        // box's inside, which for a surface outside points the reflection at a wall it is not behind
        if (any(abs(local) > extents))
        {
            return 0.0;
        }

        // Fade across the outer quarter of each axis, so the handover to the global environment happens
        // inside the volume instead of exactly at the wall where the projection is least reliable
        float3 margin = max(extents * 0.25, float3(1e-4, 1e-4, 1e-4));
        float3 axisWeight = saturate((extents - abs(local)) / margin);
        float weight = min(min(axisWeight.x, axisWeight.y), axisWeight.z);
        return weight * weight;
    }

    float normalized = saturate(1.0 - (length(local) / radius));
    return normalized * normalized;
}

// Irradiance arriving along @p normal, from one probe's L1 coefficients
//
// The coefficients store *radiance*; convolving with the cosine lobe is a per-band scale in SH. Divided
// through by pi so the result matches gIrradianceMap, which stores E/pi, and both feed the same expression
float3 evaluateIrradianceSH(uint probeIndex, float3 normal)
{
    uint base = probeIndex * 4;
    float3 l0 = gIrradianceSH[base + 0].rgb;
    float3 l1y = gIrradianceSH[base + 1].rgb;
    float3 l1z = gIrradianceSH[base + 2].rgb;
    float3 l1x = gIrradianceSH[base + 3].rgb;

    // 0.282095 = Y00; 0.325735 = (2/3) * 0.488603, the band-1 basis already carrying its cosine scale
    float3 result = 0.282095 * l0 + 0.325735 * (l1x * normal.x + l1y * normal.y + l1z * normal.z);

    // A single L1 band can swing negative on the dark side of a strongly directional bounce, which would
    // subtract light rather than add none
    return max(result, float3(0.0, 0.0, 0.0));
}

// Trilinearly blends the eight probes surrounding @p worldPosition, weighted to keep light from leaking
//
// A probe on the far side of a wall is geometrically adjacent to a surface it cannot see, and interpolating
// it in lights the room next door. Two cheap, inexact defences - the exact answer needs per-probe
// visibility: the lookup is pushed along the normal before the grid coordinates are taken, and each corner
// is weighted by how much it lies in the hemisphere the surface faces
//
// @returns false when the position is outside every volume, leaving the caller on the global environment
bool sampleIrradianceVolumes(float3 worldPosition, float3 normal, out float3 irradiance)
{
    irradiance = float3(0.0, 0.0, 0.0);

    for (uint v = 0; v < gCamera.irradianceVolumeCountPad.x; ++v)
    {
        IrradianceVolumeGpu volume = gCamera.irradianceVolumes[v];

        // Unbaked, or baked for a grid this volume no longer has. Its slice of gIrradianceSH holds either
        // nothing or another scene's coefficients, and both read as light arriving from geometry that is not
        // there - so it contributes nothing until its own bake lands
        if (volume.bakedPad.x == 0)
        {
            continue;
        }

        uint3 counts = volume.countsBase.xyz;
        if (any(counts < uint3(2, 2, 2)))
        {
            continue;
        }

        float3 minCorner = volume.minCornerSpacing.xyz;
        float3 extents = volume.extentsIntensity.xyz;
        float3 step = (extents * 2.0) / max(float3(counts) - 1.0, float3(1.0, 1.0, 1.0));

        // A small fraction of a cell, not half of one: at 4m spacing half a cell moves the lookup into the
        // next room, which is the leak it was meant to prevent. This only lifts a surface off its own probe
        float3 samplePosition = worldPosition + normal * (min(min(step.x, step.y), step.z) * 0.2);

        float3 local = (samplePosition - minCorner) / step;
        if (any(local < float3(-0.5, -0.5, -0.5)) || any(local > float3(counts) - 0.5))
        {
            continue;
        }

        float3 clamped = clamp(local, float3(0.0, 0.0, 0.0), float3(counts) - 1.0001);
        int3 baseCell = int3(floor(clamped));
        float3 frac3 = clamped - float3(baseCell);

        float3 accumulated = float3(0.0, 0.0, 0.0);
        float totalWeight = 0.0;

        for (uint corner = 0; corner < 8; ++corner)
        {
            int3 offset = int3(int(corner & 1u), int((corner >> 1u) & 1u), int((corner >> 2u) & 1u));
            int3 cell = min(baseCell + offset, int3(counts) - 1);

            float3 blend = lerp(1.0 - frac3, frac3, float3(offset));
            float weight = blend.x * blend.y * blend.z;
            if (weight <= 0.0)
            {
                continue;
            }

            uint probeIndex = volume.countsBase.w + uint(cell.x) + uint(cell.y) * counts.x + uint(cell.z) * counts.x * counts.y;

            // Direction from the surface to this probe. Probes behind the surface are the ones that leak
            float3 probePosition = minCorner + float3(cell) * step;
            float3 toProbe = probePosition - worldPosition;
            float distanceToProbe = length(toProbe);

            // A probe practically on the surface has no meaningful direction, so it keeps its full weight
            // rather than being rejected by a normal test against a near-zero vector
            float facing = distanceToProbe > 1e-4 ? saturate(dot(normal, toProbe / distanceToProbe) * 0.5 + 0.5) : 1.0;

            weight *= facing * facing;
            if (weight <= 0.0)
            {
                continue;
            }

            accumulated += evaluateIrradianceSH(probeIndex, normal) * weight;
            totalWeight += weight;
        }

        if (totalWeight > 0.0)
        {
            irradiance = (accumulated / totalWeight) * volume.extentsIntensity.w;
            return true;
        }
    }

    return false;
}

float2 environmentBRDF(float roughness, float NdotV)
{
    const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
    const float4 c1 = float4(1.0, 0.0425, 1.04, -0.04);
    float4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NdotV)) * r.x + r.y;
    return float2(-1.04, 1.04) * a004 + r.zw;
}

// Cook-Torrance specular + Lambertian diffuse, energy-conserving split via Fresnel/metallic
float3 shadeLight(float3 N, float3 V, float3 L, float3 radiance, float3 albedo, float metallic, float roughness)
{
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0)
    {
        return float3(0.0, 0.0, 0.0);
    }

    float3 H = normalize(V + L);
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);

    float NDF = distributionGGX(N, H, roughness);
    float G = geometrySmith(N, V, L, roughness);
    float3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    float3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001;
    float3 specular = numerator / denominator;

    float3 kD = (float3(1.0, 1.0, 1.0) - F) * (1.0 - metallic);

    return (kD * albedo / kPi + specular) * radiance * NdotL;
}

// Inverse-square falloff, smoothly windowed to zero at range - avoids the classic hard cutoff pop
float attenuateRange(float dist, float range)
{
    float rangeFalloff = clamp(1.0 - pow(dist / max(range, 0.0001), 4.0), 0.0, 1.0);
    rangeFalloff = rangeFalloff * rangeFalloff;
    return rangeFalloff / max(dist * dist, 0.0001);
}

// Radius of the filter kernel, in shadow-map texels
static const float kShadowFilterRadiusTexels = 2.5;

// Far smaller than the filter radius: this moves the lookup sideways, and past about a texel that reads as
// the shadow pulling away from its caster. Acne is ShadowPass's slope-scaled bias; this only covers the
// depth quantization that bias cannot see
static const float kShadowNormalOffsetTexels = 0.5;

// Ceiling on the *lateral* displacement the normal offset produces on the receiving surface, in world
// units - see shadowNormalOffset() for why this and not a ceiling on the offset itself
static const float kMaxShadowContactOffset = 0.02;

/**
 * How far along the surface normal to push a shadow lookup, given the shadow map's texel size there
 *
 * The offset has to grow at grazing angles, where a texel spans the most depth - but pushing a point off a
 * surface by d displaces its shadow by d / tan(elevation), so the same angle turns it into the largest
 * visible shift. In a far cascade that is metres under a low sun
 *
 * So the cap bounds the *displacement*, not the offset. Where it bites, ShadowPass's slope-scaled bias
 * covers acne instead - the right tool anyway, being a pure depth shift
 */
float shadowNormalOffset(float texelWorldSize, float NdotL)
{
    float sinTheta = sqrt(saturate(1.0 - NdotL * NdotL));
    float offset = texelWorldSize * kShadowNormalOffsetTexels * sinTheta;

    float lateral = offset * sinTheta / max(NdotL, 1e-3);
    if (lateral > kMaxShadowContactOffset)
    {
        offset *= kMaxShadowContactOffset / lateral;
    }

    return offset;
}

// Poisson disc: evenly spread but irregular, so what little structure survives reads as noise rather than
// as the grid pattern a box kernel leaves behind
static const float2 kPoissonDisk[12] = {
    float2(-0.326, -0.406), float2(-0.840, -0.074), float2(-0.696,  0.457), float2(-0.203,  0.621),
    float2( 0.962, -0.195), float2( 0.473, -0.480), float2( 0.519,  0.767), float2( 0.185, -0.893),
    float2( 0.507,  0.064), float2( 0.896,  0.412), float2(-0.322, -0.933), float2(-0.792, -0.598),
};

// Interleaved gradient noise - one cheap per-pixel value used to rotate the kernel. Without it every pixel
// samples the same 12 offsets and the disc's own shape prints itself onto every shadow edge
float interleavedGradientNoise(float2 pixel)
{
    return frac(52.9829189 * frac(dot(pixel, float2(0.06711056, 0.00583715))));
}

/**
 * Rate at which the receiver's own depth changes per unit of shadow-map UV
 *
 * What makes a filter kernel work on a slope. Every tap reads a different texel but compares against the
 * *shaded point's* depth, so on a tilted floor the far taps are legitimately deeper and report themselves
 * shadowed. Biasing the reference until that stops trades acne for a shadow that has moved
 *
 * The gradient removes the trade: each tap compares against the depth the receiver plane has at that tap
 *
 * Screen-space derivatives, so this is garbage on the seam where neighbouring pixels pick different
 * cascades - hence the clamp at the point of use
 */
float2 receiverPlaneDepthGradient(float2 uv, float depth)
{
    float3 dx = float3(ddx(uv), ddx(depth));
    float3 dy = float3(ddy(uv), ddy(depth));

    float determinant = (dx.x * dy.y) - (dx.y * dy.x);
    if (abs(determinant) < 1e-12)
    {
        return float2(0.0, 0.0);
    }

    float inverse = 1.0 / determinant;
    return float2(
        ((dy.y * dx.z) - (dx.y * dy.z)) * inverse,
        ((dx.x * dy.z) - (dy.x * dx.z)) * inverse);
}

// Percentage-closer filter over one layer of a shadow map array, using hardware depth comparison
//
// @param pixel SV_Position.xy, only used to rotate the kernel per pixel
// @param gradient Receiver plane depth gradient, see receiverPlaneDepthGradient()
// @param maxGradientAdjust Ceiling on the gradient's adjustment. At a silhouette it describes a plane the
//        neighbouring geometry is not on, and unbounded it pushes the reference through a real occluder
float pcfShadow(
    Sampler2DArrayShadow shadowMaps, float2 uv, float layer, float compareDepth, float texelSize, float2 pixel,
    float2 gradient, float maxGradientAdjust)
{
    // Outside the fitted frustum there is no depth information at all, so anything there is treated as lit -
    // the alternative (clamping) smears the border texel across the whole rest of the world
    if (any(uv < float2(0.0, 0.0)) || any(uv > float2(1.0, 1.0)) || compareDepth > 1.0 || compareDepth < 0.0)
    {
        return 1.0;
    }

    float angle = interleavedGradientNoise(pixel) * 6.28318530718;
    float2 rotation = float2(cos(angle), sin(angle));
    float radius = texelSize * kShadowFilterRadiusTexels;

    float sum = 0.0;
    for (int i = 0; i < 12; ++i)
    {
        float2 tap = kPoissonDisk[i];
        float2 offset = float2(
            tap.x * rotation.x - tap.y * rotation.y,
            tap.x * rotation.y + tap.y * rotation.x) * radius;

        // Compare against the depth the receiver plane has where this tap lands, not where the shaded point
        // is - which is what lets the kernel widen without the surface shadowing itself
        float tapDepth = compareDepth + clamp(dot(gradient, offset), -maxGradientAdjust, maxGradientAdjust);

        // LevelZero, not the derivative-picking variant: these calls sit under branches that differ between
        // neighbouring pixels, where implicit derivatives are undefined
        sum += shadowMaps.SampleCmpLevelZero(float3(uv + offset, layer), tapDepth);
    }

    return sum / 12.0;
}

// Fraction of light reaching a fragment from one specific cascade, ignoring whether that cascade is the
// right one to be using - see sampleDirectionalShadow(), which picks and blends
float sampleCascade(uint cascade, float3 worldPosition, float3 N, float3 L, float2 pixel)
{
    // Cancels depth-quantization acne at grazing angles without the peter-panning a constant bias gives.
    // Scaled by obliqueness, by the cascade's texel size, and by the filter radius - a tap reaching further
    // has to clear proportionally more of the surface
    float NdotL = saturate(dot(N, L));
    float3 offsetPosition = worldPosition + N * shadowNormalOffset(gCamera.cascadeTexelWorldSize[cascade], NdotL);

    float4 lightClip = mul(gCamera.cascadeViewProjection[cascade], float4(offsetPosition, 1.0));
    if (lightClip.w <= 0.0)
    {
        return 1.0;
    }

    float3 projected = lightClip.xyz / lightClip.w;
    float2 uv = projected.xy * 0.5 + 0.5;

    float bias = gCamera.cascadeDepthBias[cascade];
    float2 gradient = receiverPlaneDepthGradient(uv, projected.z);

    return pcfShadow(
        gShadowMap,
        uv,
        float(cascade),
        projected.z - bias,
        gCamera.shadowParams.z,
        pixel,
        gradient,
        // Bounded relative to the constant bias: enough headroom for a genuinely sloped receiver, not enough
        // to walk the reference past an occluder at a silhouette
        bias * 8.0);
}

// @returns true when the engine has a usable acceleration structure and tracing is switched on
//
// Every gScene access is behind this: the binding is partially-bound, so an unwritten frame would be
// undefined to read. Constant-folds to false without ray query, removing the trace entirely
bool tracedShadowsEnabled()
{
#if TOAST_RAY_QUERY
    return gCamera.tracedShadowParams.x >= 0.5;
#else
    return false;
#endif
}

// Visibility along a shadow ray: 1 lit, 0 occluded
//
// Replaces the whole shadow-map family below behind the same interface. None of that machinery applies -
// no cascades, no slope-scaled bias, no atlas slot, no resolution budget. The one bias is a nudge along the
// normal to clear the surface's own triangle
//
// @param maxDistance Distance to the light for a punctual one, or a long constant for a directional
float traceShadowRay(float3 worldPosition, float3 N, float3 L, float maxDistance)
{
#if TOAST_RAY_QUERY
    RayDesc ray;
    ray.Origin = worldPosition + N * 0.02;
    ray.Direction = L;
    ray.TMin = 0.0;
    // Pulled in slightly so a ray aimed at a light sitting inside geometry does not hit the surface holding
    // it and shadow everything the light touches
    ray.TMax = max(maxDistance - 0.02, 0.0);

    RayQuery<RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
    query.TraceRayInline(gScene, RAY_FLAG_NONE, 0xFF, ray);
    query.Proceed();

    return query.CommittedStatus() == COMMITTED_NOTHING ? 1.0 : 0.0;
#else
    // Unreachable - tracedShadowsEnabled() is false in this build - but the call sites still have to compile
    return 1.0;
#endif
}

// Fraction of the shadow-casting DirectionalLight reaching a fragment: 1 fully lit, 0 fully shadowed
//
// @param viewDepth Positive distance from the camera along its forward axis, which is what the cascade
//                  splits are expressed in
// @param pixel SV_Position.xy, forwarded to the filter for its per-pixel kernel rotation
float sampleDirectionalShadow(float3 worldPosition, float3 N, float3 L, float viewDepth, float2 pixel)
{
    if (gCamera.shadowParams.x < 0.5)
    {
        return 1.0;
    }

    // Nearest cascade whose far distance still covers this fragment. Beyond the last one the light simply
    // stops casting rather than stretching a cascade to infinity
    uint cascadeCount = uint(gCamera.shadowParams.y);
    uint cascade = cascadeCount;
    for (uint i = 0; i < cascadeCount; ++i)
    {
        if (viewDepth <= gCamera.cascadeSplits[i])
        {
            cascade = i;
            break;
        }
    }
    if (cascade >= cascadeCount)
    {
        return 1.0;
    }

    float shadow = sampleCascade(cascade, worldPosition, N, L, pixel);

    // Cascades differ ~4x in texel density, so switching at an exact distance draws that difference as a
    // hard line across the floor. Fading over the last stretch costs a second filter inside the band only
    float cascadeFar = gCamera.cascadeSplits[cascade];
    float cascadeNear = cascade > 0 ? gCamera.cascadeSplits[cascade - 1] : 0.0;
    float span = max(cascadeFar - cascadeNear, 1e-4);
    float t = (viewDepth - cascadeNear) / span;

    const float kBlendBand = 0.15;
    if (t > (1.0 - kBlendBand) && (cascade + 1) < cascadeCount)
    {
        float nextShadow = sampleCascade(cascade + 1, worldPosition, N, L, pixel);
        shadow = lerp(shadow, nextShadow, (t - (1.0 - kBlendBand)) / kBlendBand);
    }

    return shadow;
}

// Fraction of a shadow-casting Spotlight reaching a fragment. The spot's own perspective matrix does all
// the work, exactly like the directional case with one cascade
float sampleSpotShadow(GpuLight light, float3 worldPosition, float3 N, float3 L, float2 pixel)
{
    float layer = light.directionPad.w;
    if (layer < 0.0)
    {
        return 1.0;
    }

    // No fitted world texel size here, so the offset comes from distance to the light - a perspective map's
    // texel covers ground in proportion to it. Divided by the atlas fraction too, or a light in a
    // quarter-size sub-rect self-shadows
    float NdotL = saturate(dot(N, L));
    float texelWorldSize =
        distance(worldPosition, light.worldPosRange.xyz) * (gCamera.shadowParams.w / max(light.shadowAtlas.x, 1e-4)) * 2.0;
    float3 offsetPosition = worldPosition + N * shadowNormalOffset(texelWorldSize, NdotL);

    float4 lightClip = mul(light.shadowViewProjection, float4(offsetPosition, 1.0));
    if (lightClip.w <= 0.0)
    {
        return 1.0;
    }

    float3 projected = lightClip.xyz / lightClip.w;
    // Scaled into the sub-rect this light was actually rendered into, anchored at the layer's origin
    float2 uv = (projected.xy * 0.5 + 0.5) * light.shadowAtlas.x;

    // Depth bias in normalized units shrinks as the projection's depth curve steepens, so it is derived
    // from the stored depth itself rather than being a flat constant across the whole range
    float bias = max(0.0005, 0.0025 * (1.0 - projected.z));
    float2 gradient = receiverPlaneDepthGradient(uv, projected.z);

    return pcfShadow(
        gPunctualShadowMaps, uv, layer, projected.z - bias, gCamera.shadowParams.w, pixel, gradient, bias * 8.0);
}

// Fraction of a shadow-casting PointLight reaching a fragment
//
// The cube is six ordinary array layers, so face and UVs come from the light-to-fragment direction in the
// +X,-X,+Y,-Y,+Z,-Z order cubeFaceBasis() writes. Depth is reconstructed from near/far, which is why no
// per-face matrix is uploaded
float samplePointShadow(GpuLight light, float3 worldPosition, float3 N, float3 L, float2 pixel)
{
    float baseLayer = light.directionPad.w;
    if (baseLayer < 0.0)
    {
        return 1.0;
    }

    float nearPlane = light.coneAngles.z;
    float farPlane = light.coneAngles.w;

    float NdotL = saturate(dot(N, L));
    float texelWorldSize =
        distance(worldPosition, light.worldPosRange.xyz) * (gCamera.shadowParams.w / max(light.shadowAtlas.x, 1e-4)) * 2.0;
    float3 offsetPosition = worldPosition + N * shadowNormalOffset(texelWorldSize, NdotL);

    float3 direction = offsetPosition - light.worldPosRange.xyz;
    float3 absDirection = abs(direction);
    float majorAxis = max(absDirection.x, max(absDirection.y, absDirection.z));
    if (majorAxis < 0.0001)
    {
        return 1.0;
    }

    // Each face is lookAt(lightPos, lightPos + faceDir, faceUp) plus a 90-degree square perspective, so its
    // NDC is (dot(d, right), dot(d, up)) / dot(d, faceDir). These swizzles are those dot products for the
    // basis cubeFaceBasis() builds - disagree and the shadow tears along the face seams
    uint face;
    float2 ndc;
    if (absDirection.x >= absDirection.y && absDirection.x >= absDirection.z)
    {
        face = direction.x > 0.0 ? 0 : 1;
        ndc = float2(direction.x > 0.0 ? -direction.y : direction.y, direction.z) / majorAxis;
    }
    else if (absDirection.y >= absDirection.z)
    {
        face = direction.y > 0.0 ? 2 : 3;
        ndc = float2(direction.y > 0.0 ? direction.x : -direction.x, direction.z) / majorAxis;
    }
    else
    {
        face = direction.z > 0.0 ? 4 : 5;
        ndc = float2(direction.z > 0.0 ? -direction.x : direction.x, direction.y) / majorAxis;
    }

    // Each face is rendered slightly wider than 90 degrees so taps near its edge still land on real
    // geometry rather than the cleared border - shadowAtlas.y scales this exact-90-degree ndc into that
    // widened frustum. Without it every face boundary shows a lit seam the width of the filter kernel
    ndc *= light.shadowAtlas.y;

    float2 uv = (ndc * 0.5 + 0.5) * light.shadowAtlas.x;

    // Depth this face's perspectiveRH_ZO projection writes for a point whose distance along the face axis is
    // majorAxis: z_ndc = f/(f-n) - (f*n)/((f-n)*majorAxis)
    float range = max(farPlane - nearPlane, 0.0001);
    float faceDepth = (farPlane / range) - ((farPlane * nearPlane) / (range * majorAxis));

    float bias = max(0.0005, 0.0025 * (1.0 - faceDepth));
    float2 gradient = receiverPlaneDepthGradient(uv, faceDepth);

    return pcfShadow(
        gPunctualShadowMaps, uv, baseLayer + float(face), faceDepth - bias, gCamera.shadowParams.w, pixel, gradient,
        bias * 8.0);
}

// Cluster light-count -> color ramp for the clustered-lighting debug view (blue = empty, red = hot).
// kHeatmapMaxLights is the count that reads as fully red; tune to whatever "too many lights in one
// cluster" means for the scene you're profiling
float3 clusterHeatmapColor(uint lightCount)
{
    const float kHeatmapMaxLights = 8.0;
    float t = clamp(float(lightCount) / kHeatmapMaxLights, 0.0, 1.0);

    float3 c0 = float3(0.0, 0.0, 1.0);
    float3 c1 = float3(0.0, 1.0, 0.0);
    float3 c2 = float3(1.0, 1.0, 0.0);
    float3 c3 = float3(1.0, 0.0, 0.0);

    if (t < 0.333) return lerp(c0, c1, t / 0.333);
    if (t < 0.667) return lerp(c1, c2, (t - 0.333) / 0.334);
    return lerp(c2, c3, (t - 0.667) / 0.333);
}

// The surface a shader hands to the lighting model

/// One shaded point in world space. From here down nothing knows what produced it
struct Surface
{
    float3 worldPosition;

    /// Shading normal: normal-mapped where a material has one
    float3 normal;

    /// Un-perturbed. Shadow lookups offset along this, not `normal` - a per-texel direction scatters the
    /// offset across a flat surface and reads as blotchy self-shadowing
    float3 geometricNormal;

    /// Surface to camera, normalized
    float3 viewDirection;

    float3 albedo;
    float  metallic;
    float  roughness;

    /// Baked ambient occlusion; attenuates indirect light only. 1 where a shader has none
    float  occlusion;

    /// View-space Z of the point, negative in front of the camera. Picks the cluster's depth slice
    float  viewSpaceZ;

    /// SV_Position.xy - the pixel being shaded. Seeds the shadow dither and picks the cluster's screen tile
    float2 pixel;
};

struct ShadingResult
{
    /// Linear HDR radiance, unbounded and un-encoded - TonemapPass maps the scene once at the end
    float3 color;

    /// Already occluded. Separate because SsaoPass subtracts it and adds it back weighted, which it cannot
    /// do once it is summed into `color`
    float3 indirectDiffuse;

    /// `color` is a debug visualisation rather than radiance. A caller must not add emission to it or
    /// otherwise treat it as a lit result
    bool isDebugView;
};

/// The cluster covering one shaded point
struct ClusterCell
{
    uint index;
    uint count;
    uint lightCount;

    /// Distance used to pick the depth slice, clamped to the near plane
    float viewDistance;
};

/// @brief Locates @p pixel at @p viewSpaceZ in the cluster grid
///
/// Linearization must match ClusterLightingPass (i + j*dimX + k*dimX*dimY), or a fragment reads another
/// part of the screen's lights
ClusterCell clusterCellAt(float viewSpaceZ, float2 pixel)
{
    Surface surface;
    surface.viewSpaceZ = viewSpaceZ;
    surface.pixel = pixel;

    float dist = max(-surface.viewSpaceZ, gClusterParams.screenSizeNearFar.z);
    float near = gClusterParams.screenSizeNearFar.z;
    float far = gClusterParams.screenSizeNearFar.w;
    uint dimX = gClusterParams.clusterDims.x;
    uint dimY = gClusterParams.clusterDims.y;
    uint dimZ = gClusterParams.clusterDims.z;

    uint zSlice = uint(clamp(log(dist / near) * float(dimZ) / log(far / near), 0.0, float(dimZ - 1)));

    float2 tileSize = gClusterParams.screenSizeNearFar.xy / float2(dimX, dimY);
    uint2 tile = uint2(clamp(surface.pixel / tileSize, float2(0.0, 0.0), float2(dimX - 1, dimY - 1)));

    uint clusterIndex = tile.x + tile.y * dimX + zSlice * dimX * dimY;
    uint clusterCount = dimX * dimY * dimZ;

    uint lightCountInCluster = 0;
    if (clusterIndex < clusterCount)
    {
        lightCountInCluster = gClusterLightGrid[clusterIndex];
    }

    ClusterCell cell;
    cell.index = clusterIndex;
    cell.count = clusterCount;
    cell.lightCount = lightCountInCluster;
    cell.viewDistance = dist;
    return cell;
}

// Debug views

/// @brief Renders whichever debug view gCamera.renderModePad.x selects
///
/// @returns the view in rgb with alpha 1, or alpha 0 when this is not a debug mode
///
/// Alpha carries the flag because every view returns opaque anyway, and a mode list beside the switch
/// drifts out of sync with it. These are all *inputs* to the lighting equation, not results - a black
/// surface could be black albedo, a garbage normal or missing ambient
float4 evaluateDebugView(Surface surface, ClusterCell cluster)
{
    float3 albedo = surface.albedo;
    float3 N = surface.normal;
    float3 V = surface.viewDirection;
    float metallic = surface.metallic;
    float roughness = surface.roughness;
    uint lightCountInCluster = cluster.lightCount;

    // Raw per-cluster light count, before any shading work
    if (gCamera.renderModePad.x == 1)
    {
        return float4(clusterHeatmapColor(lightCountInCluster), 1.0);
    }

    switch (gCamera.renderModePad.x)
    {
    case 2:
        return float4(albedo, 1.0);
    case 3:
        // Remapped to 0..1 - a normal is signed, and the negative half would otherwise clip to black
        return float4(N * 0.5 + 0.5, 1.0);
    case 4:
        return float4(metallic, roughness, 0.0, 1.0);
    case 5:
        // The ambient radiance the surface is being lit by, before albedo is applied
        return float4(
            gCamera.ambientColorIntensity.rgb * gCamera.ambientColorIntensity.a +
                (gCamera.environmentParams.x >= 0.5 ? gIrradianceMap.SampleLevel(N, 0.0).rgb : float3(0.0, 0.0, 0.0)),
            1.0);
    case 6:
        {
            // Prefiltered radiance along the reflection vector, at this surface's roughness. This is the only
            // thing lighting a metal in an otherwise unlit scene, so it is worth being able to look at on its
            // own - black here and black in Lit are the same bug, black here alone is not
            float3 debugReflection = reflect(-V, N);
            float debugLod = roughness * max(gCamera.environmentParams.y - 1.0, 0.0);
            return float4(
                gCamera.environmentParams.x >= 0.5
                    ? gPrefilteredEnv.SampleLevel(debugReflection, debugLod).rgb
                    : float3(0.0, 0.0, 0.0),
                1.0);
        }
    case 7:
        {
            // The same lookup as mode 6, pinned to the sharpest mip. Black here too means the environment is
            // black along this direction; black only in mode 6 means the coarse mips are the problem
            float3 baseReflection = reflect(-V, N);
            return float4(
                gCamera.environmentParams.x >= 0.5 ? gPrefilteredEnv.SampleLevel(baseReflection, 0.0).rgb
                                                   : float3(0.0, 0.0, 0.0),
                1.0);
        }
    case 8:
        // The reflection direction itself, remapped to 0..1. If the black regions share a direction, they
        // share a patch of the cubemap - and this says which patch
        return float4(reflect(-V, N) * 0.5 + 0.5, 1.0);
    case 9:
        {
            // Coverage is otherwise invisible: a badly-sized radius looks like a subtly wrong reflection.
            // Hue is the probe, brightness its falloff, grey means the global environment still owns it
            int debugProbe = -1;
            float debugWeight = 0.0;
            for (uint pi = 0; pi < gCamera.reflectionProbeCountPad.x; ++pi)
            {
                float weight = reflectionProbeWeight(surface.worldPosition, gCamera.reflectionProbes[pi]);
                if (weight > debugWeight)
                {
                    debugWeight = weight;
                    debugProbe = int(pi);
                }
            }

            if (debugProbe < 0)
            {
                return float4(0.05, 0.05, 0.05, 1.0);
            }

            // Keyed by cubemap index, not by the slot in this frame's sorted array. A probe keeps its colour as
            // the camera moves; if a surface's colour changes while nothing else did, probe identity and probe
            // data have come apart - which is a bug this view exists to make visible rather than hide
            float hue = frac(gCamera.reflectionProbes[debugProbe].params.x * 0.618033988);
            float3 tint = saturate(abs(frac(hue + float3(0.0, 0.6666, 0.3333)) * 6.0 - 3.0) - 1.0);
            return float4(tint * debugWeight, 1.0);
        }
    case 10:
        {
            // What a probe actually captured, sampled from its own cubemap rather than the global environment.
            // Magenta marks "no probe owns this surface" and yellow "owns it but never baked", so a black result
            // can only mean the capture itself is black - which is a different fault from either of those
            int capProbe = -1;
            float capWeight = 0.0;
            for (uint pi = 0; pi < gCamera.reflectionProbeCountPad.x; ++pi)
            {
                float weight = reflectionProbeWeight(surface.worldPosition, gCamera.reflectionProbes[pi]);
                if (weight > capWeight)
                {
                    capWeight = weight;
                    capProbe = int(pi);
                }
            }

            if (capProbe < 0)
            {
                return float4(1.0, 0.0, 1.0, 1.0);
            }

            uint capCube = uint(gCamera.reflectionProbes[capProbe].params.x);
            if ((gCamera.reflectionProbeCountPad.z & (1u << capCube)) == 0u)
            {
                return float4(1.0, 1.0, 0.0, 1.0);
            }

            // Corrected exactly as the lit path samples it, or this view disagrees with the shading it
            // exists to explain. The raw capture is on the mirror ball in Probe Cubemap
            ReflectionProbeGpu capProbeData = gCamera.reflectionProbes[capProbe];
            float3 capReflection = parallaxCorrectedReflection(
                reflect(-V, N),
                surface.worldPosition,
                capProbeData.positionRadius.xyz,
                capProbeData.boxExtentsIntensity.xyz,
                capProbeData.positionRadius.w);
            return float4(gProbeMaps[NonUniformResourceIndex(capCube)].SampleLevel(capReflection, 0.0).rgb, 1.0);
        }
    case 16:
        {
            // Magenta: no volume covers this point. Yellow: covered, but every probe around it is zero. An
            // unbaked grid and one that baked black are otherwise the same image
            if (gCamera.irradianceVolumeCountPad.x == 0)
            {
                return float4(0.15, 0.0, 0.15, 1.0);
            }

            float3 volumeIrradiance;
            if (!sampleIrradianceVolumes(surface.worldPosition, N, volumeIrradiance))
            {
                return float4(1.0, 0.0, 1.0, 1.0);
            }

            if (dot(volumeIrradiance, volumeIrradiance) < 1e-8)
            {
                return float4(1.0, 1.0, 0.0, 1.0);
            }

            return float4(volumeIrradiance, 1.0);
        }
    // Mode 11 falls through on purpose: ReflectionProbePass draws a mirror ball per probe over the lit
    // scene, and a misaligned seam is obvious on a sphere but invisible smeared across a wall
    default:
        break;
    }

    return float4(0.0, 0.0, 0.0, 0.0);
}

// Lighting

/// @brief Lights @p surface with every directional light and every punctual light in its cluster
///
/// Split out only so the cluster lookup is shared with the debug views; callers want shadeSurfaceLighting()
ShadingResult evaluateLighting(Surface surface, ClusterCell cluster)
{
    float3 N = surface.normal;
    float3 V = surface.viewDirection;
    float3 geometricNormal = surface.geometricNormal;
    float3 albedo = surface.albedo;
    float metallic = surface.metallic;
    float roughness = surface.roughness;
    float occlusion = surface.occlusion;

    uint clusterIndex = cluster.index;
    uint clusterCount = cluster.count;
    uint lightCountInCluster = cluster.lightCount;
    float dist = cluster.viewDistance;

    ShadingResult result;
    result.indirectDiffuse = float3(0.0, 0.0, 0.0);
    result.isDebugView = false;

    float3 outIndirectDiffuse = float3(0.0, 0.0, 0.0);

    float3 color = float3(0.0, 0.0, 0.0);

    // Global (unclustered) lights. Only one of them casts shadows - shadowParams.x holds its index + 1, and
    // ShadowPass fits its cascades to that light
    uint shadowCasterIndex = uint(gCamera.shadowParams.x) - 1;

    // For the Shadow Term view. The minimum, not an average: averaging against every unshadowed light is
    // what hides contact displacement. Only lights facing the surface update it, or a back-facing surface
    // fills the view with black that means nothing
    float debugMinShadow = 1.0;

    for (uint di = 0; di < gCamera.directionalLightCountPad.x; ++di)
    {
        DirectionalLightGpu light = gCamera.directionalLights[di];
        float3 L = normalize(-light.direction.xyz);
        float3 radiance = light.colorIntensity.rgb * light.colorIntensity.a;

        // Traced when available, which needs no shadow-caster slot: a shadow map exists for exactly one
        // directional light because there is one cascade set, while a ray costs the same for any of them
        float shadow = 1.0;
        if (tracedShadowsEnabled())
        {
            shadow = traceShadowRay(surface.worldPosition, geometricNormal, L, 1.0e5);
        }
        else if (gCamera.shadowParams.x >= 0.5 && di == shadowCasterIndex)
        {
            shadow = sampleDirectionalShadow(surface.worldPosition, geometricNormal, L, dist, surface.pixel);
        }

        if (dot(geometricNormal, L) > 0.0)
        {
            debugMinShadow = min(debugMinShadow, shadow);
        }

        color += shadeLight(N, V, L, radiance, albedo, metallic, roughness) * shadow;
    }

    if (clusterIndex < clusterCount)
    {
        uint indexBase = clusterIndex * gClusterParams.clusterDims.w;

        for (uint li = 0; li < lightCountInCluster; ++li)
        {
            GpuLight light = gLights[gLightIndexList[indexBase + li]];

            float3 toLight = light.worldPosRange.xyz - surface.worldPosition;
            float lightDist = length(toLight);
            float3 L = toLight / max(lightDist, 0.0001);

            float atten = attenuateRange(lightDist, light.worldPosRange.w);
            bool isSpot = light.viewPosType.w > 0.5;
            if (isSpot)
            {
                // Spotlight: cone smoothstep between inner/outer angle on top of the range falloff
                float cosAngle = dot(-L, normalize(light.directionPad.xyz));
                atten *= smoothstep(light.coneAngles.x, light.coneAngles.y, cosAngle);
            }

            // Traced, every punctual light gets a shadow. The atlas can only shadow the handful handed a
            // slot this frame, so a scene with more lights than slots has some casting nothing at all
            float shadow;
            if (tracedShadowsEnabled())
            {
                shadow = traceShadowRay(surface.worldPosition, geometricNormal, L, lightDist);
            }
            else
            {
                shadow = isSpot
                    ? sampleSpotShadow(light, surface.worldPosition, geometricNormal, L, surface.pixel)
                    : samplePointShadow(light, surface.worldPosition, geometricNormal, L, surface.pixel);
            }

            if (dot(geometricNormal, L) > 0.0)
            {
                debugMinShadow = min(debugMinShadow, shadow);
            }

            float3 radiance = light.colorIntensity.rgb * light.colorIntensity.a * atten;

            color += shadeLight(N, V, L, radiance, albedo, metallic, roughness) * shadow;
        }
    }

    // Contact displacement is buried under albedo and ambient in a lit image; as greyscale it is
    // measurable. Toggling traced shadows swaps in an exact reference, so any disagreement is the map's
    // error and nothing else
    if (gCamera.renderModePad.x == 19)
    {
        result.color = float3(debugMinShadow, debugMinShadow, debugMinShadow);
        result.isDebugView = true;
        return result;
    }

    // Both halves of split-sum, not diffuse alone: a fully metallic surface has no diffuse, so an
    // ambient-only metal rendered black without the specular half
    //
    // AmbientLight is a uniform environment - the same radiance in every direction, so it is its own
    // irradiance and its own prefiltered radiance. The cubemaps *add to* it rather than replacing it, or
    // generating a sky would silently disconnect the AmbientLight node
    float3 uniformRadiance = gCamera.ambientColorIntensity.rgb * gCamera.ambientColorIntensity.a;
    float NdotV = saturate(dot(N, V));
    bool hasEnvironment = gCamera.environmentParams.x >= 0.5;

    // Diffuse comes from the direction the surface faces; specular from where the view reflects off it
    float3 reflection = reflect(-V, N);

    // Roughness picks a mip: 0 is the mirror level, the last is fully rough. The sampler filters between
    // them, so roughness varies smoothly rather than stepping at each stored level
    float prefilteredLod = roughness * max(gCamera.environmentParams.y - 1.0, 0.0);

    // Strongest probe covering this point. Nearest-wins rather than a weighted blend of several: blending
    // reflections from differently-shaped volumes produces a double image wherever they overlap, which reads
    // worse than the seam the falloff already smooths over
    int bestProbe = -1;
    float bestProbeWeight = 0.0;
    for (uint pi = 0; pi < gCamera.reflectionProbeCountPad.x; ++pi)
    {
        float weight = reflectionProbeWeight(surface.worldPosition, gCamera.reflectionProbes[pi]);
        if (weight > bestProbeWeight)
        {
            bestProbeWeight = weight;
            bestProbe = int(pi);
        }
    }

    float3 irradiance = uniformRadiance;
    float3 prefiltered = uniformRadiance;

    // Outside the hasEnvironment branch on purpose: a baked probe is measured light, including the sky
    // where the sky was visible, so gating it on a cubemap would disable GI in any scene without one. It is
    // also the only source that knows what stands next to what - neither of the others can tint a floor
    // from the red wall beside it
    float3 volumeIrradiance;
    bool hasVolumeIrradiance = sampleIrradianceVolumes(surface.worldPosition, N, volumeIrradiance);

    if (hasEnvironment)
    {
        // Diffuse ambient, corrected by the probe where one has authority. No parallax here: irradiance is
        // integrated over the whole hemisphere around the normal, so there is no single ray to re-aim
        float3 globalIrradiance = gIrradianceMap.SampleLevel(N, 0.0).rgb;
        uint bestProbeCube = bestProbe >= 0 ? uint(gCamera.reflectionProbes[bestProbe].params.x) : 0u;
        if (bestProbe >= 0 && (gCamera.reflectionProbeCountPad.z & (1u << bestProbeCube)) != 0u)
        {
            float3 probeIrradiance =
                gProbeIrradiance[NonUniformResourceIndex(bestProbeCube)].SampleLevel(N, 0.0).rgb *
                gCamera.reflectionProbes[bestProbe].boxExtentsIntensity.w;
            globalIrradiance = lerp(globalIrradiance, probeIrradiance, bestProbeWeight);
        }

        irradiance += hasVolumeIrradiance ? volumeIrradiance : globalIrradiance;

        // The global environment is the fallback the probe blends over, so it is always sampled - a probe is a
        // correction to something already correct, never a replacement for it
        float3 globalSpecular = gPrefilteredEnv.SampleLevel(reflection, prefilteredLod).rgb;

        if (bestProbe >= 0)
        {
            ReflectionProbeGpu probe = gCamera.reflectionProbes[bestProbe];
            float3 corrected = parallaxCorrectedReflection(
                reflection,
                surface.worldPosition,
                probe.positionRadius.xyz,
                probe.boxExtentsIntensity.xyz,
                probe.positionRadius.w);

            // A baked probe holds the scene around it, which is the entire point - the global environment is
            // only the stand-in until then. Its chain has its own length, so roughness maps onto a different
            // number of levels than gPrefilteredEnv's
            uint probeCube = uint(probe.params.x);
            bool probeBaked = (gCamera.reflectionProbeCountPad.z & (1u << probeCube)) != 0u;
            float3 probeSpecular;
            if (probeBaked)
            {
                float probeLod = roughness * max(float(gCamera.reflectionProbeCountPad.y) - 1.0, 0.0);
                probeSpecular = gProbeMaps[NonUniformResourceIndex(probeCube)].SampleLevel(corrected, probeLod).rgb;
            }
            else
            {
                probeSpecular = gPrefilteredEnv.SampleLevel(corrected, prefilteredLod).rgb;
            }
            probeSpecular *= probe.boxExtentsIntensity.w;
            globalSpecular = lerp(globalSpecular, probeSpecular, bestProbeWeight);
        }

        prefiltered += globalSpecular;
    }
    else if (hasVolumeIrradiance)
    {
        // No environment cubemap, but a baked volume still knows what arrived here. Without this branch a
        // scene with GI baked and no sky generated falls back to the flat AmbientLight value and every probe
        // in it does nothing
        irradiance = volumeIrradiance;
    }

    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);
    float3 specularFresnel = fresnelSchlickRoughness(NdotV, F0, roughness);

    // Energy conservation: what is not reflected specularly is available to scatter diffusely, and a metal
    // has no diffuse at all
    float3 diffuseWeight = (float3(1.0, 1.0, 1.0) - specularFresnel) * (1.0 - metallic);

    float2 envBrdf = environmentBRDF(roughness, NdotV);
    float3 ambientDiffuse = irradiance * albedo * diffuseWeight;
    float3 ambientSpecular = prefiltered * (specularFresnel * envBrdf.x + envBrdf.y);

    // Baked occlusion attenuates indirect light only - it approximates nearby geometry blocking the
    // environment, which says nothing about a light the surface can plainly see
    color += (ambientDiffuse + ambientSpecular) * occlusion;

    // Handed out separately as well as summed in. Screen-space occlusion is allowed to scale this term and
    // nothing else, and once it is added to `color` there is no way to recover it - SsaoPass subtracts what
    // is here and adds it back weighted. Occlusion is folded in first so the two agree on what was applied
    outIndirectDiffuse = ambientDiffuse * occlusion;

    result.color = color;
    result.indirectDiffuse = outIndirectDiffuse;
    return result;
}

/// @brief Shades @p surface: a debug view where one is selected, otherwise the full lighting model
ShadingResult shadeSurfaceLighting(Surface surface)
{
    ClusterCell cluster = clusterCellAt(surface.viewSpaceZ, surface.pixel);

    float4 debug = evaluateDebugView(surface, cluster);
    if (debug.a > 0.0)
    {
        ShadingResult result;
        result.color = debug.rgb;
        result.indirectDiffuse = float3(0.0, 0.0, 0.0);
        result.isDebugView = true;
        return result;
    }

    return evaluateLighting(surface, cluster);
}
