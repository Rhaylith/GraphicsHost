# Prints which of ggml-vulkan's optional shader extensions the given glslc supports.
# These are the GGML_VULKAN_*_GLSLC_SUPPORT defines that ggml/vulkan-shaders-gen.vcxproj and ggml/ggml.vcxproj
# set by hand (the CMake build probes them at configure time). Re-run after changing the glslc / shaderc version.
param(
    [string]$Glslc = "D:\code\vcpkg\installed\x64-windows\tools\shaderc\glslc.exe",
    [string]$Tests = (Join-Path $PSScriptRoot "..\extern\stable-diffusion.cpp\ggml\src\ggml-vulkan\vulkan-shaders\feature-tests")
)

$features = [ordered]@{
    coopmat                = "GGML_VULKAN_COOPMAT_GLSLC_SUPPORT"
    coopmat2               = "GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT"
    coopmat2_decode_vector = "GGML_VULKAN_COOPMAT2_DECODE_VECTOR_GLSLC_SUPPORT"
    integer_dot            = "GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT"
    bfloat16               = "GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT"
    float_e2m1             = "GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT"
    float_e4m3             = "GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT"
}

& $Glslc --version | Select-Object -First 1
foreach ($name in $features.Keys) {
    $file = Join-Path $Tests "$name.comp"
    $out = & $Glslc -o NUL -fshader-stage=compute --target-env=vulkan1.3 $file 2>&1 | Out-String
    $supported = ($LASTEXITCODE -eq 0) -and ($out -notmatch "extension not supported")
    "{0,-52} {1}" -f $features[$name], ($(if ($supported) { "supported" } else { "NOT supported" }))
}
