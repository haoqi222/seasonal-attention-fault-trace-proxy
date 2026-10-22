# Draw Figure 2 (seasonal attention architecture) as a NATIVE, EDITABLE Visio
# drawing. Layout mirrors the approved matplotlib draft (long arrows version):
#   (a) four seasonal frames -> 4-scale shared encoder -> per-scale seasonal
#       attention -> U-Net decoder stages -> 1x1 conv, with per-scale skip
#       connections and vertical down/up arrows;
#   (b) fusion block chain at one cell/scale.
# Page aspect 13.40 x 7.145 in matches the Word extent ratio 1.876.
# ASCII-only source: special glyphs are composed via [char] escapes.

param(
    [string]$OutputDirectory = "<workspace_root>\05_figures_tables\formal_v4",
    [string]$BaseName = "fig2_v4_seasonal_attention_architecture_0908-02",
    [string]$ResultPath = "<workspace_root>\.workbuddy\tmp_fig2_visio_result.json"
)

$ErrorActionPreference = 'Stop'

$ell   = [char]0x2113   # script small l
$times = [char]0x00D7   # multiplication sign
$em    = [char]0x2014   # em dash

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

# ---- palette (matches the author's original VSDX) ----
$blue = 'RGB(164,194,220)'   # #a4c2dc encoder/decoder/conv/F boxes, front frame
$gray = 'RGB(238,238,238)'   # #eeeeee fusion + four-head attention
$frGreen  = 'RGB(205,226,204)'   # #cde2cc back frame
$frYellow = 'RGB(246,226,163)'   # #f6e2a3
$frTan    = 'RGB(232,196,160)'   # #e8c4a0

$preExistingVisio = @(Get-Process VISIO -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id)
$visio = $null
$document = $null
$page = $null
$pageSheet = $null
$shapes = New-Object Collections.Generic.List[object]
$connectors = New-Object Collections.Generic.List[object]

function Add-Box2 ([object]$Ctx, [double]$X, [double]$Y, [double]$W, [double]$H,
                   [string]$Name, [string]$Fill, [double]$LineWeightPt = 1.0) {
    $box = Add-AcademicRectangleTL -Context $Ctx -X $X -Y $Y -Width $W -Height $H `
        -Text '' -FontSizePt 14 -FontName 'Times New Roman' -LineWeightPt $LineWeightPt
    if ($Fill -ne 'WHITE') { Set-AcademicShapeCell $box 'FillForegnd' $Fill }
    $box.Name = $Name
    [void]$script:shapes.Add($box)
    return $box
}

function Add-Txt ([object]$Ctx, [double]$X, [double]$Y, [double]$W, [double]$H,
                  [string]$Text, [string]$Name, [double]$FontSizePt = 14,
                  [switch]$Bold, [switch]$Italic) {
    $t = Add-AcademicTextTL -Context $Ctx -X $X -Y $Y -Width $W -Height $H `
        -Text $Text -FontSizePt $FontSizePt -FontName 'Times New Roman' -Bold:$Bold
    if ($Italic) {
        Set-AcademicShapeText $t -Text $Text -FontSizePt $FontSizePt -FontName 'Times New Roman' -Italic
    }
    $t.Name = $Name
    [void]$script:shapes.Add($t)
    return $t
}

# Two-line box label: bold title + regular detail, overlaid as text shapes.
function Add-BoxLabel ([object]$Ctx, [double]$BoxX, [double]$BoxTop, [double]$BoxW,
                       [string]$Title, [string]$Detail, [string]$Name,
                       [double]$TitleDy = 4, [double]$DetailDy = 29) {
    $tt = Add-Txt $Ctx $BoxX ($BoxTop + $TitleDy) $BoxW 24 $Title ($Name + '_Title') 15 -Bold
    if ($Detail -ne '') {
        $td = Add-Txt $Ctx $BoxX ($BoxTop + $DetailDy) $BoxW 24 $Detail ($Name + '_Detail') 14
    }
}

function Add-Conn ([object]$Ctx, $FromShape, $ToShape, [string]$Name) {
    $c = Connect-AcademicShapes -Context $Ctx -FromShape $FromShape -ToShape $ToShape -LineWeightPt 1.0
    $c.Name = $Name
    [void]$script:connectors.Add($c)
    return $c
}

try {
    $visio = New-Object -ComObject Visio.Application
    $visio.Visible = $false
    $visio.AlertResponse = 7
    $document = $visio.Documents.Add('')
    $page = $document.Pages.Item(1)
    $page.Name = 'Figure 2'
    $pageSheet = $page.PageSheet
    $pageSheet.CellsU('PageWidth').FormulaU = '13.40 in'
    $pageSheet.CellsU('PageHeight').FormulaU = '7.45 in'
    $pageSheet.CellsU('DrawingScale').FormulaU = '1 in'
    $pageSheet.CellsU('PageScale').FormulaU = '1 in'

    $ctx = New-AcademicLayoutContext -Application $visio -Page $page `
        -PageWidthIn 13.40 -PageHeightIn 7.45 -ReferenceWidth 1340 -ReferenceHeight 745

    # ================= panel (a) =================
    [void](Add-Txt $ctx 30 22 60 28 '(a)' 'PanelA_Label' 15 -Bold)

    # column headers
    [void](Add-Txt $ctx 230 22 190 28 'Shared encoder'   'Hdr_Enc'  15 -Bold)
    [void](Add-Txt $ctx 510 22 190 28 'Per-scale fusion' 'Hdr_Fuse' 15 -Bold)
    [void](Add-Txt $ctx 790 22 190 28 'U-Net decoder'    'Hdr_Dec'  15 -Bold)

    # rows: r1 top (scale 4) .. r4 bottom (scale 1); box h 56, v-gap 33
    $rowsY   = @(58, 147, 236, 325)
    $chDetail = @(
        "128 ch, H/8 $times W/8",
        "64 ch, H/4 $times W/4",
        "32 ch, H/2 $times W/2",
        "16 ch, H $times W"
    )   # index 0 -> scale 4 ... index 3 -> scale 1 (top to bottom)

    $enc = @(); $fuse = @(); $dec = @()
    for ($i = 0; $i -lt 4; $i++) {
        $scaleNo = 4 - $i
        $y = $rowsY[$i]
        $e = Add-Box2 $ctx 230 $y 190 56 ("EncBox$scaleNo") $blue
        [void](Add-BoxLabel $ctx 230 $y 190 "Scale $scaleNo" $chDetail[$i] "EncLab$scaleNo")
        $f = Add-Box2 $ctx 510 $y 190 56 ("FuseBox$scaleNo") $gray
        [void](Add-BoxLabel $ctx 510 $y 190 'Seasonal attention' '(b), per scale' "FuseLab$scaleNo")
        $d = Add-Box2 $ctx 790 $y 190 56 ("DecBox$scaleNo") $blue
        [void](Add-BoxLabel $ctx 790 $y 190 "Stage $scaleNo" $chDetail[$i] "DecLab$scaleNo")
        $enc += $e; $fuse += $f; $dec += $d
    }
    # $enc[0] = Scale 4 (top) ... $enc[3] = Scale 1 (bottom)

    # horizontal skip-connection arrows (long, 0.9 in gaps)
    for ($i = 0; $i -lt 4; $i++) {
        [void](Add-Conn $ctx $enc[$i] $fuse[$i] ("Skip_Enc_Fuse_{0}" -f (4 - $i)))
        [void](Add-Conn $ctx $fuse[$i] $dec[$i] ("Skip_Fuse_Dec_{0}" -f (4 - $i)))
    }
    # vertical arrows: encoder bottom-up, decoder top-down
    for ($i = 0; $i -lt 3; $i++) {
        [void](Add-Conn $ctx $enc[3 - $i] $enc[2 - $i] ("EncVert_{0}" -f $i))
        [void](Add-Conn $ctx $dec[$i] $dec[$i + 1] ("DecVert_{0}" -f $i))
    }

    # four seasonal frames stack (front frame center-y aligns with Scale 1 row)
    $frameSpec = @(
        @{ X = 25; Y = 276; Fill = $frGreen },
        @{ X = 39; Y = 290; Fill = $frYellow },
        @{ X = 53; Y = 304; Fill = $frTan },
        @{ X = 67; Y = 318; Fill = $blue }
    )
    $front = $null
    for ($i = 0; $i -lt 4; $i++) {
        $fr = Add-Box2 $ctx $frameSpec[$i].X $frameSpec[$i].Y 130 70 ("Frame$($i + 1)") $frameSpec[$i].Fill 0.75
        if ($i -eq 3) { $front = $fr }
    }
    [void](Add-Txt $ctx 0 400 230 56 "Four seasonal frames`n8 channels each" 'Frames_Label' 14)
    [void](Add-Conn $ctx $front $enc[3] 'Frames_To_Scale1')

    # 1x1 conv output box, center-y aligned with Scale 1 row
    $conv = Add-Box2 $ctx 1010 318 185 70 'OutConv' $blue
    [void](Add-BoxLabel $ctx 1010 318 185 "1 $times 1 conv" 'one logit per cell' 'ConvLab' 8 33)
    [void](Add-Conn $ctx $dec[3] $conv 'Stage1_To_Conv')

    # panel (a) caption
    $capA = "Solid horizontal arrows are skip connections; fused features enter every decoder stage."
    $capAShape = Add-Txt $ctx 230 400 850 32 $capA 'CapA' 14
    Set-AcademicShapeText $capAShape -Text $capA -FontSizePt 14 -FontName 'Times New Roman' -Italic

    # ================= panel (b) =================
    [void](Add-Txt $ctx 30 498 60 28 '(b)' 'PanelB_Label' 15 -Bold)

    $bY = 540; $bH = 90
    $b1 = Add-Box2 $ctx 60   $bY 160 $bH 'B_Tokens'  'WHITE'
    [void](Add-BoxLabel $ctx 60   $bY 160 'Seasonal tokens' "Z($ell, p): 4 $times C($ell)" 'B1Lab' 18 47)
    $b2 = Add-Box2 $ctx 280  $bY 140 $bH 'B_QKV'    'WHITE'
    [void](Add-BoxLabel $ctx 280  $bY 140 'Q / K / V' 'linear maps' 'B2Lab' 18 47)
    $b3 = Add-Box2 $ctx 480  $bY 190 $bH 'B_Attend' $gray
    [void](Add-BoxLabel $ctx 480  $bY 190 'Four-head attention' "4 $times 4 per head" 'B3Lab' 18 47)
    $b4 = Add-Box2 $ctx 730  $bY 150 $bH 'B_ResLN'  'WHITE'
    [void](Add-BoxLabel $ctx 730  $bY 150 'Residual + LN' '' 'B4Lab' 33 0)
    $b5 = Add-Box2 $ctx 930  $bY 190 $bH 'B_Concat' 'WHITE'
    [void](Add-BoxLabel $ctx 930  $bY 190 'Fixed-order concat' "1 $times 1 projection" 'B5Lab' 18 47)
    $b6 = Add-Box2 $ctx 1170 $bY 110 $bH 'B_F'      $blue
    [void](Add-BoxLabel $ctx 1170 $bY 110 "F($ell, p)" '' 'B6Lab' 33 0)

    Add-Conn $ctx $b1 $b2 'BFlow_1_2' | Out-Null
    Add-Conn $ctx $b2 $b3 'BFlow_2_3' | Out-Null
    Add-Conn $ctx $b3 $b4 'BFlow_3_4' | Out-Null
    Add-Conn $ctx $b4 $b5 'BFlow_4_5' | Out-Null
    Add-Conn $ctx $b5 $b6 'BFlow_5_6' | Out-Null

    $capB = "Fusion block at one spatial cell and scale; repeated independently at every encoder scale and spatial cell $em attention acts only along the seasonal axis."
    $capBShape = Add-Txt $ctx 60 660 1220 56 $capB 'CapB' 14
    Set-AcademicShapeText $capBShape -Text $capB -FontSizePt 14 -FontName 'Times New Roman' -Italic

    [void]$document.SaveAs($vsdxPath)
    $document.Saved = $true
}
finally {
    if ($document) { try { $document.Close() } catch {} }
    if ($visio) { try { $visio.Quit() } catch {} }
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
    PageWidthIn = 13.40
    PageHeightIn = 7.45
    RectTextShapes = $shapes.Count
    GluedConnectors = $connectors.Count
    PreExistingVisioProcessIds = $preExistingVisio
} | ConvertTo-Json -Depth 3 | Out-File -FilePath $ResultPath -Encoding utf8
