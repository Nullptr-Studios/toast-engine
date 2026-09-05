import fullscreen;

// Copies the post-process chain's final image into the output image
//
// The renderer owns this rather than any pass: it means every post-process pass has the same shape - read a
// view, write your own target, hand the view on - and none of them needs to know whether it happens to be
// last. Disabling the whole chain then shows the raw scene instead of a black screen, because this still
// runs and the chain's "final image" is simply the scene target

[[vk::binding(0,0)]]
Sampler2D gSource;

typedef FullscreenVSOutput VSOutput;

[shader("vertex")]
VSOutput vertexMain(uint vertexId : SV_VertexID)
{
    return fullscreenTriangle(vertexId);
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target0
{
    return float4(gSource.Sample(input.uv).rgb, 1.0);
}
