param(
    [string]$OutputDirectory = "<workspace_root>\05_figures_tables\formal_v4",
    [string]$BaseName = "fig2_v4_methodological_framework_0903-02"
)

$ErrorActionPreference = 'Stop'

$helper = '<workspace_root>\.codex\skills\draw-academic-algorithm-flowcharts\scripts\visio_layout_helpers.ps1'
. $helper

$outputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
if (-not (Test-Path -LiteralPath $outputDirectory -PathType Container)) {
    throw "Output directory not found: $outputDirectory"
}
$vsdxPath = Join-Path $outputDirectory ($BaseName + '.vsdx')
if (Test-Path -LiteralPath $vsdxPath) {
    throw "Refusing to overwrite existing editable source: $vsdxPath"
}

$preExistingVisio = @(Get-Process VISIO -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id)
$visio = $null
$document = $null
$page = $null
$pageSheet = $null
$shapes = New-Object Collections.Generic.List[object]
$connectors = New-Object Collections.Generic.List[object]

try {
    $visio = New-Object -ComObject Visio.Application
    $visio.Visible = $false
    $visio.AlertResponse = 7
    $document = $visio.Documents.Add('')
    $page = $document.Pages.Item(1)
    $page.Name = 'Figure 2'
    $pageSheet = $page.PageSheet
    $pageSheet.CellsU('PageWidth').FormulaU = '11.6 in'
    $pageSheet.CellsU('PageHeight').FormulaU = '3.25 in'
    $pageSheet.CellsU('DrawingScale').FormulaU = '1 in'
    $pageSheet.CellsU('PageScale').FormulaU = '1 in'

    $context = New-AcademicLayoutContext -Application $visio -Page $page `
        -PageWidthIn 11.6 -PageHeightIn 3.25 -ReferenceWidth 1160 -ReferenceHeight 325

    # Figure 2 is intentionally limited to four high-level modules. Detailed
    # hyperparameters, ablations, inference mechanics, and secondary metrics
    # remain in the Methods and tables so the main logic is readable at a glance.
    $panelWidth = 230
    $panelY = 58
    $panelHeight = 205
    $xPositions = @(45, 325, 605, 885)
    $headers = @(
        'Data and target',
        'Spatial controls',
        'Model comparison',
        'Spatial evaluation'
    )
    $bodies = @(
        "Seasonal Landsat composites`nQFFD proxy masks",
        "Four blocked folds`nTrain-only preprocessing",
        "Temporal attention`nMatched U-Net and concatenation",
        "Four-seed fold AP`nEqual-weight fold summary"
    )

    for ($i = 0; $i -lt $xPositions.Count; $i++) {
        $body = Add-AcademicRectangleTL -Context $context -X $xPositions[$i] -Y $panelY `
            -Width $panelWidth -Height $panelHeight -Text '' `
            -FontSizePt 17 -FontName 'Times New Roman' -LineWeightPt $(if ($i -eq 2) { 1.5 } else { 1.0 })
        $body.Name = ('Stage{0}' -f ($i + 1))
        if ($i -eq 2) {
            Set-AcademicShapeCell $body 'FillForegnd' 'RGB(238,238,238)'
        }
        [void]$shapes.Add($body)

        $headerText = Add-AcademicTextTL -Context $context -X ($xPositions[$i] + 12) -Y 82 `
            -Width ($panelWidth - 24) -Height 48 -Text $headers[$i] `
            -FontSizePt 19 -FontName 'Times New Roman' -Bold
        $headerText.Name = ('Stage{0}_HeaderText' -f ($i + 1))
        [void]$shapes.Add($headerText)

        $bodyText = Add-AcademicTextTL -Context $context -X ($xPositions[$i] + 12) -Y 145 `
            -Width ($panelWidth - 24) -Height 76 -Text $bodies[$i] `
            -FontSizePt 17 -FontName 'Times New Roman'
        $bodyText.Name = ('Stage{0}_BodyText' -f ($i + 1))
        [void]$shapes.Add($bodyText)
    }

    $bodyShapes = @($shapes | Where-Object { $_.Name -match '^Stage[1-4]$' } | Sort-Object Name)
    for ($i = 0; $i -lt ($bodyShapes.Count - 1); $i++) {
        $connector = Connect-AcademicShapes -Context $context `
            -FromShape $bodyShapes[$i] -ToShape $bodyShapes[$i + 1] `
            -LineWeightPt 1.0
        $connector.Name = ('MainFlow_{0}_{1}' -f ($i + 1), ($i + 2))
        [void]$connectors.Add($connector)
    }

    [void]$document.SaveAs($vsdxPath)
    $document.Saved = $true
}
finally {
    if ($document) {
        try { $document.Close() } catch {}
    }
    if ($visio) {
        try { $visio.Quit() } catch {}
    }
    foreach ($item in $connectors) {
        if ($item -and [Runtime.InteropServices.Marshal]::IsComObject($item)) {
            try { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($item) } catch {}
        }
    }
    foreach ($item in $shapes) {
        if ($item -and [Runtime.InteropServices.Marshal]::IsComObject($item)) {
            try { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($item) } catch {}
        }
    }
    foreach ($item in @($pageSheet, $page, $document, $visio)) {
        if ($item -and [Runtime.InteropServices.Marshal]::IsComObject($item)) {
            try { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($item) } catch {}
        }
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

if (-not (Test-Path -LiteralPath $vsdxPath -PathType Leaf)) {
    throw "Visio source was not created: $vsdxPath"
}

[pscustomobject]@{
    VsdxPath = $vsdxPath
    PageWidthIn = 11.6
    PageHeightIn = 3.25
    NativeModuleShapes = 4
    GluedMainFlowConnectors = 3
    PreExistingVisioProcessIds = $preExistingVisio
} | ConvertTo-Json -Depth 3
