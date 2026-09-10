import toast;

// The scene lighting model, and with it every set-0 binding the engine fills in. Anything else that can
// describe a Surface imports the same module and resolves through identical code
import lighting;

struct MaterialUBO
{
    [Reflect] [Color] [Name("Tint")]
    float4 tint;

    [Reflect] [Name("Metallic")] [Range(0.0, 1.0)]
    float metallic;

    [Reflect] [Name("Roughness")] [Range(0.045, 1.0)]
    float roughness;

    // 0 disables the test, so a material only pays for it by opting in. Cutout is order-independent and
    // still writes depth, which is what makes it right for foliage and grate cards where blending is not
    [Reflect] [Name("Alpha Cutoff")] [Range(0.0, 1.0)]
    float alphaCutoff;

    // How much of gOcclusionMap's baked ambient occlusion to apply, glTF's occlusionTexture.strength. 0
    // ignores the map entirely, which is what a material authored before these parameters existed bakes to
    [Reflect] [Name("Occlusion Strength")] [Range(0.0, 1.0)]
    float occlusionStrength;

    // Self-illumination added after shading and before tonemapping, so it survives being in shadow but is
    // still rolled off by the Reinhard curve. Zero intensity (the pre-existing-material default) is no emission
    [Reflect] [Color] [Name("Emissive Color")]
    float3 emissiveColor;

    [Reflect] [Name("Emissive Intensity")] [Range(0.0, 64.0)]
    float emissiveIntensity;

    // 0=R, 1=G, 2=B, 3=A. R is the default so materials predating these keep working; glTF packs metal and
    // roughness into one texture, and a channel selector is what lets it bind to several slots at once
    //
    // No [Group]: a grouped parameter is looked up inside the group's TOML table, and these sit at the top
    // level beside the factors they pair with
    [Reflect] [Name("Metallic Channel")] [Range(0.0, 3.0)]
    float metallicChannel;

    [Reflect] [Name("Roughness Channel")] [Range(0.0, 3.0)]
    float roughnessChannel;

    [Reflect] [Name("Occlusion Channel")] [Range(0.0, 3.0)]
    float occlusionChannel;
};

[[vk::binding(0,1)]]
ConstantBuffer<MaterialUBO> gMaterial;

[Reflect] [Name("Albedo")]
[[vk::binding(1,1)]]
Sampler2D gAlbedo;

// Tangent-space normal map; unbound materials fall back to a flat (0.5,0.5,1.0) "no perturbation" normal
[Reflect] [Linear] [Name("Normal")] [Default("flat_normal")]
[[vk::binding(2,1)]]
Sampler2D gNormal;

// Unbound falls back to white, the identity multiplier, so the factor alone drives a uniform material.
// Black would look equally plausible and silently zero the product, making Metallic do nothing
[Reflect] [Linear] [Name("Metallic Map")]
[[vk::binding(3,1)]]
Sampler2D gMetallicMap;

// Roughness map, multiplied by gMaterial.roughness and read from gMaterial.roughnessChannel; unbound falls
// back to white so the factor alone controls uniformly-rough materials
[Reflect] [Linear] [Name("Roughness Map")]
[[vk::binding(4,1)]]
Sampler2D gRoughnessMap;

// Baked ambient occlusion read from gMaterial.occlusionChannel, glTF's occlusionTexture. Attenuates the ambient term only -
// occlusion is an approximation of indirect light being blocked, and applying it to direct lighting darkens
// surfaces a light source can plainly see. Unbound falls back to white (nothing occluded)
[Reflect] [Linear] [Name("Occlusion Map")]
[[vk::binding(5,1)]]
Sampler2D gOcclusionMap;

// Multiplied by emissiveColor * emissiveIntensity; unbound falls back to white so the factors alone drive
// uniformly-emissive materials
[Reflect] [Name("Emissive Map")]
[[vk::binding(6,1)]]
Sampler2D gEmissiveMap;

// Picks one channel out of a sampled texel. The index is uniform across a draw, so the branch costs nothing
// beyond the compare - and doing it this way lets a packed texture serve several material slots at once
float channelOf(float4 value, float index)
{
    if (index >= 2.5)
    {
        return value.a;
    }
    if (index >= 1.5)
    {
        return value.b;
    }
    if (index >= 0.5)
    {
        return value.g;
    }
    return value.r;
}

// One entry per drawn instance, filled by the engine each frame. Mirrors VulkanRenderer::InstanceData
struct InstanceData
{
    float4x4 model;

    // Index of this instance's first matrix in gJointMatrices - see resolveSkinning() in vulkan_renderer.cpp
    uint jointOffset;

    // Three scalars, not a uint3: std430 gives a uint3 16-byte alignment and a 96-byte stride, while the C++
    // side packs glm::uvec3 at 80. Every instance past the first then reads across the boundary, which looks
    // like geometry collapsing toward the origin rather than like a layout bug
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

[[vk::binding(13,0)]]
StructuredBuffer<InstanceData> gInstances;

struct PushConstants
{
    // Where this draw's run starts in gInstances. Instances within a draw are addressed by SV_InstanceID on
    // top of it, so a run of N proxies sharing a mesh and material is one push and one draw rather than N of
    // each. The engine sorts proxies so those runs are contiguous - see VulkanRenderer::tick()
    uint instanceBase;
};

[[vk::push_constant]]
PushConstants gPush;

struct VSInput
{
    [[vk::location(0)]]
    float3 position : POSITION;

    [[vk::location(1)]]
    float3 normal : NORMAL;

    [[vk::location(2)]]
    float2 uv : TEXCOORD0;

    [[vk::location(3)]]
    float4 tangent : TANGENT;

    [[vk::location(4)]]
    float3 color : COLOR;
};

struct VSOutput
{
    float4 position : SV_Position;

    [[vk::location(0)]]
    float3 worldNormal : NORMAL;

    [[vk::location(1)]]
    float2 uv : TEXCOORD0;

    [[vk::location(2)]]
    float3 color : COLOR;

    [[vk::location(3)]]
    float3 worldPosition : TEXCOORD2;

    [[vk::location(4)]]
    float viewSpaceZ : TEXCOORD3;

    [[vk::location(5)]]
    float3 worldTangent : TANGENT;

    [[vk::location(6)]]
    float tangentSign : TEXCOORD4;
};

[shader("vertex")]
VSOutput vertexMain(VSInput input, uint instanceID : SV_InstanceID)
{
    VSOutput output;

    // Draws always pass firstInstance = 0, so instanceID is zero-based within this draw and the run's start
    // comes from the push constant. That avoids depending on whether SV_InstanceID maps to SPIR-V's
    // InstanceIndex (which includes firstInstance) or to the base-relative HLSL meaning
    float4x4 model = gInstances[gPush.instanceBase + instanceID].model;

    float4 worldPos = mul(model, float4(input.position, 1.0));

    output.position =
        mul(
            gCamera.viewProjection,
            worldPos);

    output.worldPosition = worldPos.xyz;
    output.viewSpaceZ = mul(gCamera.view, worldPos).z;

    output.worldNormal =
        normalize(
            mul(
                (float3x3)model,
                input.normal));

    output.worldTangent =
        normalize(
            mul(
                (float3x3)model,
                input.tangent.xyz));
    output.tangentSign = input.tangent.w;

    output.uv =
        input.uv;

    output.color = input.color;

    return output;
}

// No skinned variant here: skinning.slang poses into a vertex buffer once, so a posed instance arrives at
// vertexMain as an ordinary draw. The identity model matrix is per the glTF spec - a skinned node's own
// transform is not applied, so posed vertices are already in world space

// Shades one fragment, and hands back the geometry buffer's contents through @p outNormalRoughness
//
// Everything above the handoff to lighting.slang is texture fetches and TBN reconstruction
//
// An inout rather than a returned struct: the write has to land once the shading normal is final and before
// any debug view returns, and the zero at the top covers the paths that never get there
float4 shadeSurface(VSOutput input, inout float4 outNormalRoughness, inout float3 outIndirectDiffuse)
{
    // Zero-length normal = "nothing here", matching the attachment's clear. A fragment that discards or
    // returns before the normal is known must not leave the previous draw's value in place
    outNormalRoughness = float4(0.0, 0.0, 0.0, 0.0);
    outIndirectDiffuse = float3(0.0, 0.0, 0.0);

    float4 albedoSample = gAlbedo.Sample(input.uv);
    float alpha = albedoSample.a * gMaterial.tint.a;

    // The alpha test lives in fragmentMainCutout, not here: a reachable `discard` forces the whole pipeline
    // onto late-Z, so in the shared body every material paid for a branch almost none of them take

    // Vertex colour is the third albedo input, alongside the map and the material tint. It is what lets a
    // model painted in Blender carry per-part colour without splitting into one mesh per material.
    // Meshes written before .tmesh v7 left the field zeroed, so the reader forces those to white
    float3 albedo = albedoSample.rgb * gMaterial.tint.rgb * input.color;
    float metallic = clamp(gMaterial.metallic * channelOf(gMetallicMap.Sample(input.uv), gMaterial.metallicChannel), 0.0, 1.0);
    float roughness = clamp(gMaterial.roughness * channelOf(gRoughnessMap.Sample(input.uv), gMaterial.roughnessChannel), 0.045, 1.0);
    float occlusion = lerp(
        1.0,
        channelOf(gOcclusionMap.Sample(input.uv), gMaterial.occlusionChannel),
        clamp(gMaterial.occlusionStrength, 0.0, 1.0)
    );

    // Gram-Schmidt re-orthogonalize the interpolated tangent against the interpolated normal, then derive
    // the bitangent from the vertex-supplied handedness sign - standard TBN reconstruction per-pixel
    float3 N = normalize(input.worldNormal);
    float3 T = input.worldTangent - N * dot(N, input.worldTangent);
    float tangentLength = length(T);
    float handedness = input.tangentSign;

    // Common, not an edge case - glTF only requires a tangent alongside a normal map. Normalizing the zero
    // vector gives NaN, which spreads into N even for a flat normal map, so only ambient lands
    if (tangentLength < 1e-6 || abs(handedness) < 1e-6)
    {
        float3 reference = abs(N.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
        T = normalize(cross(reference, N));
        handedness = 1.0;
    }
    else
    {
        T = T / tangentLength;
    }

    float3 B = cross(N, T) * handedness;

    // Before the normal map perturbs N: a per-texel normal scatters the shadow offset across a flat floor,
    // which reads as blotchy self-shadowing
    float3 geometricNormal = N;

    float3 tangentNormal = gNormal.Sample(input.uv).rgb * 2.0 - 1.0;
    N = normalize(T * tangentNormal.x + B * tangentNormal.y + N * tangentNormal.z);

    // The normal-mapped normal, not the vertex one: a screen-space trace has to leave along the direction
    // shading actually used, or reflections disagree with the lit image. Roughness rides along because blur
    // cannot be recovered from depth
    outNormalRoughness = float4(N, roughness);

    float3 V = normalize(gCamera.position - input.worldPosition);

    Surface surface;
    surface.worldPosition = input.worldPosition;
    surface.normal = N;
    surface.geometricNormal = geometricNormal;
    surface.viewDirection = V;
    surface.albedo = albedo;
    surface.metallic = metallic;
    surface.roughness = roughness;
    surface.occlusion = occlusion;
    surface.viewSpaceZ = input.viewSpaceZ;
    surface.pixel = input.position.xy;

    ShadingResult lit = shadeSurfaceLighting(surface);
    outIndirectDiffuse = lit.indirectDiffuse;

    // A visualisation, not radiance - emission would be painting light onto a picture of the albedo. The
    // heatmap now writes the geometry buffer before returning, which is harmless: the term it feeds is zero
    // in every debug mode
    if (lit.isDebugView)
    {
        return float4(lit.color, 1.0);
    }

    float3 color = lit.color;

    // Emission is not lighting: it neither casts nor receives, so it lands after shadowing and outside the
    // albedo/metallic split entirely
    color += gMaterial.emissiveColor * gMaterial.emissiveIntensity * gEmissiveMap.Sample(input.uv).rgb;

    // Linear HDR out, deliberately unbounded and un-encoded. The scene renders into a float target and
    // renderer::TonemapPass maps the whole thing to the display once, at the end - doing it here would make
    // the curve per-material policy, and would blend transparent surfaces against already-tonemapped colour
    return float4(color, alpha);
}

// A struct, because Slang assigns `out` parameters by declaration order and ignores an SV_Target index
// written on one - which swapped the attachments and rendered the whole viewport as the geometry buffer
struct FragmentOutput
{
    float4 color : SV_Target0;
    float4 normalRoughness : SV_Target1;
    float4 indirectDiffuse : SV_Target2;
};

// Two entry points over one shading body, differing only in whether a discard is reachable
//
// A reachable `discard` forces the pipeline onto late-Z, so an occluded fragment pays for every fetch and
// lookup before being thrown away. A `bool` parameter is not enough - Slang keeps the branch and its OpKill
// - so the test has to be lexically absent from the fast path. Verify with
// `slangc mesh.slang -target spirv-assembly -entry fragmentMain -stage fragment | grep OpKill`
[shader("fragment")]
FragmentOutput fragmentMain(VSOutput input)
{
    FragmentOutput output;
    output.normalRoughness = float4(0.0, 0.0, 0.0, 0.0);

    float3 indirectDiffuse = float3(0.0, 0.0, 0.0);
    output.color = shadeSurface(input, output.normalRoughness, indirectDiffuse);
    output.indirectDiffuse = float4(indirectDiffuse, 1.0);
    return output;
}

[shader("fragment")]
FragmentOutput fragmentMainCutout(VSOutput input)
{
    // Sampled again here rather than reusing shadeSurface's fetch: the test has to happen before any shading
    // work for the discard to be worth anything, and the duplicate costs one texture read on the only
    // materials that opted into cutouts
    float4 albedoSample = gAlbedo.Sample(input.uv);
    float alpha = albedoSample.a * gMaterial.tint.a;
    if (gMaterial.alphaCutoff > 0.0 && alpha < gMaterial.alphaCutoff)
    {
        discard;
    }

    FragmentOutput output;
    output.normalRoughness = float4(0.0, 0.0, 0.0, 0.0);

    float3 indirectDiffuse = float3(0.0, 0.0, 0.0);
    output.color = shadeSurface(input, output.normalRoughness, indirectDiffuse);
    output.indirectDiffuse = float4(indirectDiffuse, 1.0);
    return output;
}
