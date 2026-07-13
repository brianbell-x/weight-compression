param(
  [string]$Output = (Join-Path $PSScriptRoot 'lossless-compression-hero.mp4')
)

$ErrorActionPreference = 'Stop'
$fps = 30
$seconds = 5
$frameCount = $fps * $seconds
$tempRoot = [IO.Path]::GetFullPath($env:TEMP).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
$work = [IO.Path]::GetFullPath((Join-Path $env:TEMP 'compression-social-hero-frames'))
if (-not $work.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase)) {
  throw "Refusing to use a render directory outside the system temp folder: $work"
}

if (Test-Path -LiteralPath $work) {
  Remove-Item -LiteralPath $work -Recurse -Force
}
New-Item -ItemType Directory -Path $work | Out-Null

$counts = @(4, 6, 7, 6, 4)
$layers = @()
for ($i = 0; $i -lt $counts.Count; $i++) {
  $column = @()
  for ($j = 1; $j -le $counts[$i]; $j++) {
    $column += ,@(
      95.0 + (1035.0 - 95.0) * $i / ($counts.Count - 1)
      225.0 + (500.0 - 225.0) * $j / ($counts[$i] + 1)
    )
  }
  $layers += ,$column
}

function F([double]$value) {
  return $value.ToString('0.###', [Globalization.CultureInfo]::InvariantCulture)
}

function FrameProgress([int]$frame) {
  $time = $frame / [double]$fps
  if ($time -lt 0.5) { return 1.0 }
  if ($time -lt 2.5) {
    $t = ($time - 0.5) / 2.0
    return 1.0 - [Math]::Pow(1.0 - $t, 3.0)
  }
  return 1.0
}

for ($frame = 0; $frame -lt $frameCount; $frame++) {
  $e = FrameProgress $frame
  $scale = 1.0 - (1.0 - 980.0 / 1403.0) * $e
  $mapX = { param([double]$x) 70.0 + ($x - 70.0) * $scale }
  $front = & $mapX 1130.0
  $center = & $mapX 565.0
  $size = [Math]::Round(1403.0 - 423.0 * $e)
  $saved = [Math]::Round(423.0 * $e)
  $chipOpacity = [Math]::Max(0.0, [Math]::Min(1.0, ($e - 0.45) / 0.55))
  $radius = 8.0 - 1.4 * $e

  $parts = [Collections.Generic.List[string]]::new()
  $parts.Add('<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="675" viewBox="0 0 1200 675">')
  $parts.Add('<defs><style>.mono{font-family:Consolas,&quot;Courier New&quot;,monospace;font-weight:700}.black{font-family:&quot;Arial Black&quot;,Arial,sans-serif;font-weight:900}</style><pattern id="h" width="9" height="9" patternTransform="rotate(45)" patternUnits="userSpaceOnUse"><line x1="0" y1="0" x2="0" y2="9" stroke="#d92b1c" stroke-width="1.6"/></pattern></defs>')
  $parts.Add('<rect width="1200" height="675" fill="#f2f0ea"/>')
  $parts.Add('<text x="70" y="94" class="black" font-size="58" letter-spacing="-2" fill="#141412">LOSSLESS MODEL COMPRESSION</text>')
  $parts.Add('<line x1="70" y1="122" x2="1130" y2="122" stroke="#141412" stroke-width="1.5"/>')
  $parts.Add('<text x="70" y="158" class="mono" font-size="17" letter-spacing="2" fill="#57544c">GLM-5.2 753B &#183; BF16</text>')
  $parts.Add('<text x="1130" y="158" text-anchor="end" class="mono" font-size="17" letter-spacing="2" fill="#57544c">IN VRAM</text>')
  $parts.Add('<rect x="70" y="174" width="1060" height="360" fill="none" stroke="#141412" stroke-width="1.5" stroke-dasharray="8 8"/>')
  $parts.Add(('<rect x="{0}" y="175" width="{1}" height="358" fill="url(#h)"/>' -f (F $front), (F (1130.0 - $front))))
  $parts.Add(('<line x1="{0}" y1="175" x2="{0}" y2="533" stroke="#d92b1c" stroke-width="3"/>' -f (F $front)))

  $parts.Add('<g stroke="#57544c" stroke-opacity=".45" stroke-width="1.3">')
  for ($i = 0; $i -lt $layers.Count - 1; $i++) {
    foreach ($a in $layers[$i]) {
      foreach ($b in $layers[$i + 1]) {
        $parts.Add(('<line x1="{0}" y1="{1}" x2="{2}" y2="{3}"/>' -f (F (& $mapX $a[0])), (F $a[1]), (F (& $mapX $b[0])), (F $b[1])))
      }
    }
  }
  $parts.Add('</g><g fill="#141412">')
  foreach ($column in $layers) {
    foreach ($node in $column) {
      $parts.Add(('<circle cx="{0}" cy="{1}" r="{2}"/>' -f (F (& $mapX $node[0])), (F $node[1]), (F $radius)))
    }
  }
  $parts.Add('</g>')

  $parts.Add(('<rect x="{0}" y="318" width="230" height="78" fill="#f2f0ea" stroke="#141412" stroke-width="1.5"/>' -f (F ($center - 115.0))))
  $parts.Add(('<text x="{0}" y="372" text-anchor="middle" class="black" font-size="44" letter-spacing="-2" fill="#141412">{1:N0} GB</text>' -f (F $center), $size))
  $parts.Add(('<g opacity="{0}"><rect x="866" y="292" width="218" height="112" fill="#f2f0ea" stroke="#141412" stroke-width="1.5"/><text x="975" y="341" text-anchor="middle" class="black" font-size="39" letter-spacing="-2" fill="#d92b1c">&#8722;{1:N0} GB</text><text x="975" y="383" text-anchor="middle" class="mono" font-size="16" letter-spacing="1" fill="#141412">{2}% SMALLER</text></g>' -f (F $chipOpacity), $saved, (F (30.17 * $e))))
  $parts.Add('<text x="600" y="590" text-anchor="middle" class="mono" font-size="19" letter-spacing="1.5" fill="#141412">READ IN PLACE &#183; BIT-EXACT ROUND-TRIP &#183; NO QUALITY TRADEOFF</text>')
  $parts.Add('<text x="600" y="627" text-anchor="middle" class="mono" font-size="12" letter-spacing="1.3" fill="#57544c">1,403 GB &#8594; 980 GB &#183; 423 GB FREED &#183; 30.17% SMALLER</text>')
  $parts.Add('</svg>')

  $svgPath = Join-Path $work ('frame_{0:D4}.svg' -f $frame)
  [IO.File]::WriteAllText($svgPath, [string]::Join('', $parts), [Text.UTF8Encoding]::new($false))
}

& magick mogrify -background '#f2f0ea' -format png (Join-Path $work 'frame_*.svg')
if ($LASTEXITCODE -ne 0) { throw 'ImageMagick frame rendering failed.' }

& ffmpeg -y -framerate $fps -i (Join-Path $work 'frame_%04d.png') -vf 'scale=1280:720:flags=lanczos,format=yuv420p' -c:v libx264 -preset slow -crf 18 -movflags +faststart -an $Output
if ($LASTEXITCODE -ne 0) { throw 'FFmpeg encoding failed.' }

Remove-Item -LiteralPath $work -Recurse -Force
Write-Output $Output
