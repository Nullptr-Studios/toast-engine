// Depth-only prepass
//
// Lays down opaque depth so the main pass rejects occluded fragments at the depth test. Worth more here than
// in most renderers: a material fragment costs six texture fetches, a 12-tap shadow lookup, clustered
// lights, parallax-corrected IBL and an 8-probe SH interpolation, all wasted behind a wall
//
// Only pays because non-cutout materials compile without a discard - a reachable one forces late-Z, and then
// the main pass shades everything anyway and this is pure added cost
//
// Vertex-only, so there is no colour attachment and nothing to write but depth

struct CameraUBO
{
    float4x4 view;
    float4x4 projection;
    float4x4 viewProjection;
};

[[vk::binding(0,0)]]
ConstantBuffer<CameraUBO> gCamera;

// Mirrors mesh.slang's InstanceData and VulkanRenderer::InstanceData. Three scalar pads rather than a uint3,
// which under std430 would take 16-byte alignment and change the stride - see the note in mesh.slang
struct InstanceData
{
    float4x4 model;
    uint jointOffset;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

[[vk::binding(13,0)]]
StructuredBuffer<InstanceData> gInstances;

struct PushConstants
{
    uint instanceBase;
};

[[vk::push_constant]]
PushConstants gPush;

struct VSInput
{
    [[vk::location(0)]]
    float3 position : POSITION;
};

struct VSOutput
{
    float4 position : SV_Position;
};

[shader("vertex")]
VSOutput vertexMain(VSInput input, uint instanceID : SV_InstanceID)
{
    VSOutput output;
    float4x4 model = gInstances[gPush.instanceBase + instanceID].model;
    output.position = mul(gCamera.viewProjection, mul(model, float4(input.position, 1.0)));
    return output;
}

// No skinned variant: skinning.slang poses the vertices before any pass records, so an instance arrives
// already deformed with an identity model matrix. It used to duplicate the material shader's path and had
// to deform identically, or the prepass rejected the character against its own bind pose
