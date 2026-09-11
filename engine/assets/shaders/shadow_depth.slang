// Depth-only pass used by renderer::ShadowPass to fill every shadow map in a frame
//
// One shader for all three light kinds - they differ only in the matrix, and gPush.viewIndex picks it. No
// layer selection here: each view renders into its own image view
//
// No fragment entry point on purpose, so a cutout material casts its full quad rather than its silhouette

// Must match renderer::shadows::k_max_shadow_views
static const uint kMaxShadowViews = 20;

struct ShadowUBO
{
    // World -> shadow clip space, one per view rendered this frame
    float4x4 viewProjection[kMaxShadowViews];
};

[[vk::binding(0,0)]]
ConstantBuffer<ShadowUBO> gShadow;

// Binding 1 used to be the renderer's shared joint pool: this shader carried its own copy of the skinning
// blend so a walking character's shadow would not stay in bind pose. skinning.slang poses the vertices
// before this pass records, so the stream arriving at vertexMain is already deformed and the copy is gone

// Mirrors renderer::VulkanRenderer::InstanceData, same layout mesh.slang declares. Indexed one-to-one with
// the frame's MeshInstanceProxy list rather than the packed visible-only set the colour pass reads - a shadow
// view draws casters the camera cannot see
struct InstanceData
{
    float4x4 model;
    uint jointOffset;
    // Three scalars, not a uint3: std430 gives a uint3 16-byte alignment and pushes the stride to 96 where
    // the C++ side is 80. Same note as mesh.slang's copy
    uint pad0;
    uint pad1;
    uint pad2;
};

[[vk::binding(2,0)]]
StructuredBuffer<InstanceData> gInstances;

struct PushConstants
{
    // Index of the first instance of this draw; SV_InstanceID counts from there. The model matrix used to
    // live here, which meant one pushConstants call and one draw per caster per view - twenty views over a few
    // hundred casters is a thousand draws a frame before anything is rendered
    uint instanceBase;
    // The *first* view of this pass; multiview adds SV_ViewID to it, since a cascade set and a cube's six
    // faces occupy consecutive slots. A single-view pass reads SV_ViewID as 0
    uint viewIndex;
};

[[vk::push_constant]]
PushConstants gPush;

// Only position is declared, though the bound vertex buffer is the mesh's full renderer::Vertex stream -
// unconsumed attributes cost nothing here
struct VSInput
{
    [[vk::location(0)]]
    float3 position : POSITION;
};

[shader("vertex")]
float4 vertexMain(VSInput input, uint instanceID : SV_InstanceID, uint viewID : SV_ViewID) : SV_Position
{
    InstanceData instance = gInstances[gPush.instanceBase + instanceID];
    return mul(gShadow.viewProjection[gPush.viewIndex + viewID], mul(instance.model, float4(input.position, 1.0)));
}

// There is no vertexMainSkinned any more - a posed instance reaches vertexMain above with its vertices
// already deformed and an identity model matrix. That also removes the one place this shader could disagree
// with the material shader about how a character is shaped, which read as a shadow detaching from its caster
