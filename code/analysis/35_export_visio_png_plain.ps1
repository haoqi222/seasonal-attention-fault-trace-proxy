# Plain-COM Visio PNG export (pwsh 7 compatible, no Add-Type).
# Mirrors export_visio_png.ps1's COM steps: read-only open, 300 PPI raster
# setting, page export. Margin normalization is done separately in Python.
# ASCII-only source.

param(
    [Parameter(Mandatory = $true)][string]$VsdxPath,
    [Parameter(Mandatory = $true)][string]$PngPath,
    [ValidateRange(72, 1200)][int]$Ppi = 300,
    [string]$StatusPath = "<workspace_root>\.workbuddy\tmp_fig2_export2.json"
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $VsdxPath -PathType Leaf)) {
    throw "VSDX not found: $VsdxPath"
}
$VsdxPath = [IO.Path]::GetFullPath($VsdxPath)
if ([IO.Path]::GetExtension($VsdxPath) -ine '.vsdx') {
    throw "Expected a .vsdx source: $VsdxPath"
}
if ([IO.Path]::GetExtension($PngPath) -ine '.png') {
    throw "Expected a .png output: $PngPath"
}
$outputDir = Split-Path -Parent $PngPath
if (-not (Test-Path -LiteralPath $outputDir)) {
    [void](New-Item -ItemType Directory -Path $outputDir)
}

$hashBefore = (Get-FileHash -LiteralPath $VsdxPath -Algorithm SHA256).Hash
$visio = $null
$document = $null
$page = $null
try {
    $visio = New-Object -ComObject Visio.Application
    $visio.Visible = $false
    $visio.AlertResponse = 7
    # 2 = visOpenRO. Do not save or mutate the user's VSDX.
    $document = $visio.Documents.OpenEx($VsdxPath, 2)
    $page = $document.Pages.Item(1)
    # 3 = visRasterUseCustomResolution; 0 = visRasterPixelsPerInch
    $visio.Settings.SetRasterExportResolution(3, $Ppi, $Ppi, 0)
    $page.Export($PngPath)
}
finally {
    if ($document) { try { $document.Saved = 1; $document.Close() } catch {} }
    if ($visio) { try { $visio.Quit() } catch {} }
    if ($page) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($page) }
    if ($document) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($document) }
    if ($visio) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($visio) }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

if (-not (Test-Path -LiteralPath $PngPath -PathType Leaf)) {
    throw "PNG was not created: $PngPath"
}
$hashAfter = (Get-FileHash -LiteralPath $VsdxPath -Algorithm SHA256).Hash

[pscustomobject]@{
    PngPath = $PngPath
    SourceUnchanged = ($hashBefore -eq $hashAfter)
} | ConvertTo-Json | Out-File -FilePath $StatusPath -Encoding utf8
