// Shared full-screen triangle, imported by every post-process pass
//
// One triangle rather than a quad: no vertex buffer, no index buffer, and no diagonal seam through the
// middle of the screen
//
// No Y flip. Camera::getProjection() negates proj[1][1], so NDC +Y and texture +V both run down the screen
// and uv maps straight onto clip space. Flipping again turns the image upside down *and* silently mirrors
// every depth lookup that reconstructs a position from it - which has cost a debugging session before

public struct FullscreenVSOutput
{
    public float4 position : SV_Position;

    [[vk::location(0)]]
    public float2 uv : TEXCOORD0;
};

public FullscreenVSOutput fullscreenTriangle(uint vertexId)
{
    FullscreenVSOutput output;
    output.uv = float2((vertexId << 1) & 2, vertexId & 2);
    output.position = float4(output.uv * 2.0 - 1.0, 0.0, 1.0);
    return output;
}
