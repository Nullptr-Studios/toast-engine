// Compute skinning pre-pass - poses a skinned mesh once per frame instead of once per pass
//
// Every vertex shader that could draw a skinned mesh used to carry its own copy, so a character in the
// colour pass, the prepass and four cascades was skinned six times, and any divergence showed as the
// prepass rejecting the pose the main pass draws
//
// The larger reason is ray tracing: a BLAS is built from a vertex buffer, so vertex-shader skinning leaves
// it in bind pose forever - a T-posed traced shadow. Writing posed vertices to memory is what allows a refit
//
// Output is **world space**: per the glTF spec a skinned node's own transform is not applied, which is why
// the engine hands skinned proxies an identity model matrix

// Byte-addressed, not StructuredBuffer<Vertex>: renderer::Vertex is a vertex attribute stream packed to 60
// bytes, and the same fields under std430 come out at 80 - so a structured view reads every vertex after the
// first from the wrong address. SkinVertex is 24 against std430's 32. The C++ structs static_assert these
static const uint kVertexStride = 60;
static const uint kVertexPositionOffset = 0;
static const uint kVertexNormalOffset = 12;
static const uint kVertexUvOffset = 24;
static const uint kVertexTangentOffset = 32;
static const uint kVertexColorOffset = 48;

static const uint kSkinStride = 24;
static const uint kSkinJointsOffset = 0;
static const uint kSkinWeightsOffset = 8;

[[vk::binding(0,0)]]
ByteAddressBuffer gVertices;

[[vk::binding(1,0)]]
ByteAddressBuffer gSkinVertices;

[[vk::binding(2,0)]]
RWByteAddressBuffer gPosedVertices;

// The renderer's shared joint pool, the same buffer mesh.slang used to read. jointOffset picks this
// instance's slice, so two instances of one mesh in different poses write different output ranges from one
// source
[[vk::binding(3,0)]]
StructuredBuffer<float4x4> gJointMatrices;

struct PushConstants
{
    uint vertexCount;
    // First vertex this instance owns in gPosedVertices; one dispatch per instance, not per mesh
    uint outputBase;
    uint jointOffset;
    uint _pad0;
};

[[vk::push_constant]]
PushConstants gPush;

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 threadID : SV_DispatchThreadID)
{
    uint vertex = threadID.x;
    if (vertex >= gPush.vertexCount)
    {
        return;
    }

    uint sourceByte = vertex * kVertexStride;
    uint skinByte = vertex * kSkinStride;

    float3 position = gVertices.Load<float3>(sourceByte + kVertexPositionOffset);
    float3 normal = gVertices.Load<float3>(sourceByte + kVertexNormalOffset);
    float4 tangent = gVertices.Load<float4>(sourceByte + kVertexTangentOffset);

    // glTF permits joint indices as unsigned byte or short; the importer widens both to uint16, so four of
    // them occupy two dwords that have to be split by hand
    uint2 packedJoints = gSkinVertices.Load<uint2>(skinByte + kSkinJointsOffset);
    float4 weights = gSkinVertices.Load<float4>(skinByte + kSkinWeightsOffset);

    uint4 joints = uint4(
        packedJoints.x & 0xFFFF,
        packedJoints.x >> 16,
        packedJoints.y & 0xFFFF,
        packedJoints.y >> 16);

    float4x4 skinMatrix =
        weights.x * gJointMatrices[gPush.jointOffset + joints.x] +
        weights.y * gJointMatrices[gPush.jointOffset + joints.y] +
        weights.z * gJointMatrices[gPush.jointOffset + joints.z] +
        weights.w * gJointMatrices[gPush.jointOffset + joints.w];

    // Upper 3x3, not a proper inverse-transpose: a blend of rigid joint transforms carries no non-uniform
    // scale unless the skeleton was authored with one, and an inverse per vertex is the wrong trade
    float3x3 skinNormalMatrix = (float3x3)skinMatrix;

    uint outputByte = (gPush.outputBase + vertex) * kVertexStride;

    gPosedVertices.Store<float3>(outputByte + kVertexPositionOffset, mul(skinMatrix, float4(position, 1.0)).xyz);
    gPosedVertices.Store<float3>(outputByte + kVertexNormalOffset, normalize(mul(skinNormalMatrix, normal)));
    // w is the handedness sign, which no transform touches
    gPosedVertices.Store<float4>(
        outputByte + kVertexTangentOffset,
        float4(normalize(mul(skinNormalMatrix, tangent.xyz)), tangent.w));

    // Copied unchanged, but they still have to be written: the posed buffer is what every raster pass binds
    // as its vertex stream, so a field left unwritten reads as whatever the previous frame's instance put
    // there
    gPosedVertices.Store<float2>(outputByte + kVertexUvOffset, gVertices.Load<float2>(sourceByte + kVertexUvOffset));
    gPosedVertices.Store<float3>(outputByte + kVertexColorOffset, gVertices.Load<float3>(sourceByte + kVertexColorOffset));
}
