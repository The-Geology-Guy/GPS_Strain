# 04 — Basemap tile providers and app distribution: costs and licence terms

Research date: 2026-10-14. All prices are in USD and exclude tax. Sources were checked on official pages where possible; each claim has a URL. Items marked **UNVERIFIED** could not be confirmed on an official page.

## Usage assumptions

- Each map uses 16–64 raster tiles at zoom ~9–12. Tiles are 256 px, or 512 px "@2x".
- The **scenarios use 64 tiles × 20 maps per user per month**, the upper end of the range.

| Scenario | Users | Maps / month | Tile requests / month |
|---|---|---|---|
| (a) | 10 | 200 | 12,800 |
| (b) | 100 | 2,000 | 128,000 |
| (c) | 1,000 | 20,000 | 1,280,000 |

**The key issue is licensing, not price.** The developer is a for-profit LLC, and users **print** the output. Most providers treat any use by a for-profit entity as commercial, even when the app is free. Most also restrict print separately from on-screen display.

---

## 1. Stadia Maps (hosts Stamen Terrain / Terrain Background)

**Plans** (https://stadiamaps.com/pricing/):

| Plan | $/month | Credits / month | Overage | Commercial use |
|---|---|---|---|---|
| Free | 0 | 200,000 | none (hard stop, HTTP 429) | **No** |
| Starter | 20 | 1,000,000 | $0.03 / 1k credits | Yes |
| Standard | 80 | 7,500,000 | $0.02 / 1k | Yes |
| Professional | 250 | 25,000,000 | $0.015 / 1k | Yes |

New accounts get a 14-day Professional trial, then drop to Free.

**Credit schedule** ("Complete Features and Credit Schedule", pricing page):

| Request type | Credits |
|---|---|
| Raster basemap tile | 1 per tile |
| Vector basemap tile | 1 per tile |
| Static map | 20 per request |
| Cacheable static map | 2,000 per request |

- **@2x tiles:** The pricing page lists no separate cost for @2x (512 px) tiles. I assume they cost 1 credit like any tile. **UNVERIFIED.**
- **Tile format:** Raster tiles come at 256 px and 512 px "retina" sizes, at `tiles.stadiamaps.com/tiles/<style>/{z}/{x}/{y}{r}.png` (https://docs.stadiamaps.com/raster/).

**Commercial use and the free tier**

- The ToS §13 says: "Our free tier may only be used for non-commercial or evaluation purposes."
- The revised ToS, effective 2026-11-09, defines use by any for-profit organisation as "Commercial Use", even without revenue (https://stadiamaps.com/terms-of-service/versions/2026-11-09/).
- **An app published by an LLC therefore needs a paid plan, even if the app is free.**

**Desktop apps and API keys**

- ToS §13 (2026-11-09) permits use in "desktop applications" provided you "maintain an active paid subscription".
- Docs recommend API-key authentication for desktop and mobile apps. They warn: "Avoid shipping API keys in an app binary when possible", and suggest rotating keys and monitoring usage (https://docs.stadiamaps.com/authentication/).
- §8 forbids sharing one key "across multiple products". An exception covers a suite of similar applications created and sold by the account holder.

**Print**

- §8 restricts use "in print, film, television…" unless the use is strictly non-commercial or Stadia grants an explicit commercial licence.
- The sanctioned route is the **cacheable static maps** endpoint (`/static_cacheable/`, 2,000 credits per image). Under §30.3.2: "Images may be printed and redistributed, up to 5,000 copies per Image, including for commercial use." You need an active paid subscription when you generate the image (https://stadiamaps.com/faqs/, https://docs.stadiamaps.com/static-maps/).
- Static images are 64–4096 points per side, and `@2x` doubles the pixel count.
- **Conclusion:** stitching ordinary raster tiles into a printed PDF does not clearly fall under the print licence. Use cacheable static maps for printable output, or get a written licence from Stadia.
- **UNVERIFIED:** whether Stamen Terrain is available on the static endpoint. The docs point to the style library and don't confirm it.

**Caching and offline**

- Only client-side caching is allowed. Tiles may be kept no longer than the HTTP cache headers allow, or 7 days if there is no header. Server-side caching is prohibited.
- Offline caching is capped at 100 MB per device.
- Bulk downloading is prohibited (ToS §8).

**Attribution** (https://docs.stadiamaps.com/attribution/)

- Required text: "© Stadia Maps © OpenMapTiles © OpenStreetMap" plus "© Stamen Design" for Stamen styles.
- For print, attribution can go in a caption, footnote or acknowledgements. It must include the names and these URLs: stadiamaps.com/attribution, stamen.com and openstreetmap.org/copyright.
- Non-commercial academic use needs no special permission. Commercial print beyond the static-map licence means contacting support@stadiamaps.com.

## 2. Esri ArcGIS Location Platform

**Pricing** (https://location.arcgis.com/pricing/):

| Item | Free each month | Price after free tier |
|---|---|---|
| Basemap tiles | 2,000,000 | $0.15 / 1,000 |
| Basemap sessions | 1,000 | $4 / 1,000 |
| Static maps | 1,000 | $0.90 / 1,000 |

- Tile billing covers `basemaps-api.arcgis.com`, `ibasemaps-api.arcgis.com` and `static-map-tiles-api.arcgis.com`.
- Pay-as-you-go is **off by default**. Once the free tier runs out, access is disabled until the next cycle unless you enable it (https://location.arcgis.com/help/deployment/).
- A session lasts up to 12 hours and gives unlimited tiles from the Basemap Styles service. Session tokens may not be used programmatically "(e.g., exporting volumes of basemap tiles)" (E300 footnote 10).

**Account and token**

- A **Location Platform account and an access token (API key) are required**. ArcGIS Online and Enterprise accounts are not supported for these services (https://developers.arcgis.com/rest/static-basemap-tiles/, https://developers.arcgis.com/maplibre-gl-js/maps/raster-tile-basemaps/display-multiple-basemap-layers/).
- Static basemap tiles are **512 × 512 px**. Fewer requests are needed per map, but the tiles are vector-style basemaps (outdoor, streets and so on), not a terrain raster.

**Available raster services**

- The Location Platform service-data page lists only **World_Hillshade** and **World_Imagery** as tile services on `ibasemaps-api.arcgis.com` (https://developers.arcgis.com/rest/basemap-styles/service-data/).
- World_Terrain_Base and World_Topo_Map are **not listed**. An "arcgis/terrain" style exists in Basemap Styles (https://developers.arcgis.com/rest/basemap-styles/arcgis-terrain-base-style-get).
- **UNVERIFIED:** whether the classic World_Terrain_Base raster is available through Location Platform billing.

**Commercial distribution**

- With a Location Platform subscription you "can deploy unlimited commercial public and private applications" (https://location.arcgis.com/help/deployment/).
- E300 Product-Specific Terms footnote 89 lets the customer distribute "revenue-generating Value-Added" applications. Footnote 78 includes a "Commercial App Deployment" licence (https://www.esri.com/content/dam/esrisites/en-us/media/legal/product-specific-terms-of-use/e300.pdf, Nov 2025).
- The free tier is part of the same subscription. I found no rule against using it for a commercial app.

**server.arcgisonline.com without a key**

- The tiles answer without a token, but Esri staff say keyless use is legal **only if you hold an ArcGIS licence** (ArcGIS Online or Enterprise). They describe it as a "paper" licence governed by the E300 terms (https://community.esri.com/t5/arcgis-javascript-maps-sdk-questions/map-without-api-key/m-p/1371175).
- **Unlicensed keyless use from a distributed app is not authorised.** Use Location Platform with a token instead.

**Print, static maps and caching**

- The E300 terms and the pricing page say **nothing explicit about printing**. **UNVERIFIED.** Ask Esri before relying on it for publications.
- E300 footnote 24 restricts map caches to development, testing and demonstration of a prototype.
- I found no explicit offline or tile-retention allowance for Location Platform raster tiles. **UNVERIFIED.**
- **Attribution:** "Powered by Esri" plus data-provider attribution is required (https://developers.arcgis.com/documentation/mapping-and-location-services/faq).

## 3. MapTiler Cloud

**Plans** (https://www.maptiler.com/cloud/pricing/):

| Plan | $/month | Sessions | API requests | Map printing |
|---|---|---|---|---|
| Free | 0 (no card) | 5k | 100k | not listed |
| Flex | 30 | 25k | 500k | **10 prints/month, internal use only** |
| Custom | quote (annual contract) | custom | custom | internal or external |

- Flex overage is $0.15 per 1k API requests and $2.50 per 1k sessions. On Free, service pauses until the next month.
- **Request counting:** a 256 px raster tile counts as 1 request, and a 512 px raster tile (including HiDPI) counts as **4**. A static map counts as 15 requests and is a Flex feature.
- **Commercial use:** The terms §1.2 say "Usage of the Free Plan is limited to non-commercial use and research & development for commercial products" (https://www.maptiler.com/terms/cloud/).
- **Desktop apps** are a permitted use (§5.1). You must use your own API keys and monitor or limit them (§3.2).
- **Print** (§5.3): allowed only "for an internal purpose", up to A4 at 300 DPI, with circulation limited by plan. **Classroom handouts and theses printed by third parties are external use, so you'd need a Custom quote (price unknown).**
- **Caching:** only a temporary personal cache (§5.7). Server-side caching, screenshots and bulk download are prohibited (§6.3, §7.2).
- **Attribution** must be shown on screen while the map is displayed (§5.4).
- **Styles:** "Outdoor" is "a topographic map with contour lines and hillshade" (https://www.maptiler.com/maps/).

## 4. Mapbox

**Pricing** (https://www.mapbox.com/pricing):

| API | Free each month | Price per 1k after free tier |
|---|---|---|
| Static Tiles API (styled raster tiles, e.g. Outdoors) | 200,000 | $0.50 → $0.40 → $0.30 |
| Raster Tiles API (raster tilesets, e.g. satellite, terrain-RGB) | 750,000 | $0.25 → $0.20 → $0.15 |
| Static Images API | 50,000 | $1.00 → $0.80 → $0.60 |

**Print** (Product Terms, 2026-07-21: https://cdn.prod.website-files.com/609ed46055e27a02ffc0749b/6a60463142f6478d57642594_Mapbox%20Product%20Terms%20(July%2021%2C%202026).pdf):

- §1.7: "Customer may not use any Licensed Map Content in any printed or video media except expressly as set forth".
- Print rights apply only to **Mapbox Streets v8+** (§1.7.3). Studio use has a lifetime cap of 100 high-resolution copies (§1.7.2).
- **Outdoors in print needs print rights bought through Mapbox Sales.**

**Caching**

- Caching is off by default (§1.9(v)). Where allowed, it's limited to 30 days on the requesting device (§2.8.1).
- Distributing content "by using a screenshot or other static image" is prohibited.

**Desktop apps**

- A "Licensed Application" includes desktop apps (§3.31), and content must come directly from Mapbox APIs.
- There's no Mapbox SDK requirement for desktop. Mobile apps must use the Mobile SDKs (§2.9.1).

**Attribution:** "© Mapbox" (linked) and "© OpenStreetMap", plus "Improve this map" for Mapbox Maps (§1.4).

**UNVERIFIED:** whether @2x Static Tiles count as one request.

## 5. Thunderforest, OpenTopoMap and self-hosting

### Thunderforest (Landscape / Outdoors)

**Plans** (https://www.thunderforest.com/pricing/):

| Plan | $/month | Tiles / month |
|---|---|---|
| Hobby | free | 150k |
| Solo Developer | 125 (£95) | 1.5M |
| Small Business | 255 (£195) | 15M |
| Large Business | 525 (£395) | 150M |

- Static Map API and vector tile requests count as 10 tile requests each.
- `@2x` tiles are supported (https://www.thunderforest.com/docs/map-tiles-api/). **UNVERIFIED:** whether @2x counts as one request.

**Terms** (https://www.thunderforest.com/terms/):

- "Commercial use is permitted and encouraged." The free plan is offered "at our discretion", and the pricing page frames Hobby as for testing ideas. **UNVERIFIED:** whether a commercial app may stay on Hobby. Ask Thunderforest.
- Map images are licensed **CC-BY-SA 4.0**, and the data is ODbL.
- Tiles may be cached on-device for offline use, even beyond HTTP expiry. Proxies and bulk pre-downloading are not allowed below the Small Business plan.
- Print attribution: "Maps © www.thunderforest.com, Data © www.osm.org/copyright".
- **CC-BY-SA ShareAlike may require the printed maps to carry a CC-BY-SA licence.** That can conflict with journal copyright transfer.
- Thunderforest is the one commercial provider whose terms allow printing outright.

### OpenTopoMap

- Tiles are licensed CC-BY-SA 3.0. The terms say to "always name opentopomap.org", and "If you plan to use the tiles for bigger projects, please contact us" (https://wiki.openstreetmap.org/wiki/OpenTopoMap).
- It's a volunteer server with no SLA. **I couldn't load the official opentopomap.org/about usage policy (binary/blocked). UNVERIFIED.**
- Don't rely on it for a distributed app without permission.

### Self-hosting (most freedom; print rights are effectively unrestricted beyond attribution)

**Data sources**

- **USGS 3DEP** is US public domain: "USGS-authored or produced data and information are considered to be in the U.S. Public Domain" (https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits).
- **Copernicus DEM GLO-30/GLO-90** has a free worldwide licence. Public distribution needs the notice "© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved". Modified data should say "produced using Copernicus WorldDEM-30" (https://dataspace.copernicus.eu/node/1306). **UNVERIFIED:** the exact commercial-use wording of the ESA licence PDF (https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/DEM/resources/license/License-COPDEM-30.pdf).
- **Natural Earth** is public domain. Not re-verified this session.
- **Protomaps planet PMTiles** (OSM vector basemap) is about 120 GB at z0–15 and is an ODbL Produced Work, so OSM attribution is required. Regional extracts are available (https://docs.protomaps.com/basemaps/downloads).
- **Mapterhorn** provides Terrarium terrain PMTiles.

**Approach and cost**

- Pre-render a hillshade raster tile pyramid (z0–12) from GLO-90/GLO-30 or 3DEP with GDAL, then store it as PMTiles.
- My estimate for a global land-only z0–12 hillshade is ~30–100 GB. **This is my estimate, not a sourced figure.**
- Host it on **Cloudflare R2** (https://developers.cloudflare.com/r2/pricing/): $0.015 per GB-month after 10 GB free, Class B reads at $0.36 per million after 10M free, and **free egress**.
- **Hosting cost:**
  - 100 GB stored is ≈ **$1.35/month**. The 120 GB Protomaps planet is about $1.65.
  - Reads stay under the 10M free Class B requests even in scenario (c), so read cost is ≈ $0.
  - Total is about **$2–4/month in every scenario**, plus a domain (~$10–15/yr).
  - PMTiles range requests are one Class B operation each. A Worker or Pages proxy may add cost if you exceed Workers' free tier. **UNVERIFIED.**
- The real cost is engineering time: DEM processing, plus rendering vector data to static images (MapLibre Native, or Mapnik/Cairo, from Python).
- Alternatively, bundle a regional hillshade inside the app and skip the network entirely.

## 6. App distribution

| Item | Cost | Source |
|---|---|---|
| Apple Developer Program | **$99/yr**. Required for Mac App Store, Developer ID and notarization. Fee waivers are only for nonprofits, educational and government bodies. | https://developer.apple.com/support/compare-memberships/ |
| Mac App Store commission | **0% if the app is free.** For paid apps it's 15% under the Small Business Program (≤$1M proceeds; you must enrol), otherwise 30%. | https://developer.apple.com/app-store/small-business-program/ |
| macOS direct download | Developer ID signing + notarization. No extra cost beyond the $99. | as above |
| Microsoft Store (individual) | **Free** registration since 2025, in ~200 markets. The Store signs MSIX-packaged apps for free. | https://blogs.windows.com/windowsdeveloper/2025/09/10/free-developer-registration-for-individual-developers-on-microsoft-store/ |
| Microsoft Store (company account for an LLC) | **UNVERIFIED** in this session. Historically a one-time ~$99. Registering as an individual may be acceptable. | — |
| Azure Artifact Signing (formerly Trusted Signing) | **Basic $9.99/mo** (5,000 signatures, 1 profile). **Premium $99.99/mo** (100,000 signatures, 10 profiles). Overage $0.005 per signature. Needs a paid Azure subscription, billed full-month. Individuals must be in the US or Canada; organisations in the US, CA, EU, UK and others. | Azure Retail Prices API (https://prices.azure.com/api/retail/prices); https://learn.microsoft.com/en-us/azure/artifact-signing/quickstart; https://learn.microsoft.com/en-us/azure/artifact-signing/faq |
| OV code-signing certificate | From ~$219/yr (Sectigo/Comodo, reseller price) | https://www.ssldragon.com/ssl-certificates/code-signing/ |
| EV code-signing certificate | From ~$287/yr (reseller). Keys must sit on a FIPS token or HSM. Maximum validity is 460 days since 2026-03-01. | as above |

- SmartScreen reputation builds with download volume for signed files (Artifact Signing FAQ).
- A US-based LLC owner qualifies for Artifact Signing as an individual, or via the LLC with organisation validation.
- Price checks on the official Azure pricing page showed "$-" placeholders. The figures above come from Azure's official retail-price API.

## 7. Python in a sandboxed Mac App Store app

- Apple DTS says the Mac App Store allows third-party tools if "they have to be signed and packaged correctly", and that Apple has no Python-specific guidance (https://developer.apple.com/forums/thread/700908).
- A PyInstaller issue shows typical failures (https://github.com/pyinstaller/pyinstaller/issues/8029):
  - unsigned CFFI temporary binaries rejected by library validation under the sandbox
  - entitlement inheritance problems for child processes
  - App Review rejecting a bundled Tk build for deprecated API use
  - an outdated `libomp` from wheels
- BeeWare Briefcase documents DMG/.app output but makes no Mac App Store claim (https://briefcase.beeware.org/en/v0.3.24/_sources/about/faq.rst.txt).
- **Practical advice:** start with notarized direct download (DMG). Treat the Mac App Store as optional and expensive in engineering time. Avoid onefile/self-extracting builds, sign every nested .so/.dylib, and exclude Tk unless it's needed.

---

## Comparison table

| Provider | Terrain-ish style | Free tier usable by an LLC? | Entry paid plan | Tile counting | Print by end users | Offline cache |
|---|---|---|---|---|---|---|
| Stadia | Stamen Terrain / Terrain Background | **No** (non-commercial only) | $20/mo, 1M credits | 1 credit/tile (@2x assumed same); cacheable static = 2,000 | Only via cacheable static: ≤5,000 copies/image | ≤100 MB/device; ≤ cache headers / 7 days |
| Esri Location Platform | World_Hillshade raster; Terrain / Outdoor styles | **Yes** (commercial apps allowed) | PAYG $0.15/1k after 2M | per tile; 512 px static tiles | **Not addressed — ask Esri** | prototype caches only (UNVERIFIED for apps) |
| MapTiler | Outdoor (contours + hillshade) | **No** | Flex $30/mo, 500k requests | 256 px = 1, 512 px = 4 | Flex: 10 internal prints; external needs Custom | temporary personal cache only |
| Mapbox | Outdoors | Yes for display (free tiers) | PAYG | per tile | **Not allowed** for Outdoors without purchased print rights | ≤30 days, same device |
| Thunderforest | Landscape / Outdoors | Hobby 150k, at their discretion (UNVERIFIED for commercial) | $125/mo, 1.5M | per tile; static = 10 | Allowed (CC-BY-SA 4.0 + attribution; ShareAlike) | allowed on-device |
| OpenTopoMap | OpenTopoMap | CC-BY-SA 3.0; volunteer server; "contact us" for big projects | — | — | CC-BY-SA | not specified |
| Self-host (Copernicus/3DEP hillshade + Protomaps) | your own | n/a | ~$2–4/mo R2 | — | **Unrestricted** (attribution only; ODbL for OSM layers) | unlimited / can bundle |

## Monthly cost scenarios (64 tiles × 20 maps × users)

| Provider | (a) 10 users, 12.8k tiles | (b) 100 users, 128k tiles | (c) 1,000 users, 1.28M tiles |
|---|---|---|---|
| Stadia, raster tiles (display rights) | $20 (Starter) | $20 | $28.40 (Starter + 280k overage) |
| **Stadia, cacheable static** (1 image per map; needed for print) | 400k credits → **$20** | 4M credits → **$80** (Standard) | 40M credits → **$475** (Professional $250 + 15M × $0.015/1k) |
| Esri Location Platform, tiles | $0 | $0 | $0 (under 2M free). 512 px tiles need roughly ¼ the requests. |
| MapTiler Flex, 256 px tiles | $30 | $30 | $147 ($30 + 780k × $0.15/1k) |
| MapTiler Flex, 512 px tiles (×4) | $30 (51.2k) | $31.80 (512k) | $723 (5.12M) |
| MapTiler for external print | Custom quote (UNKNOWN) | Custom | Custom |
| Mapbox Static Tiles (Outdoors) | $0 | $0 | $540 (1.08M × $0.50/1k) |
| Mapbox print rights | sales quote (UNKNOWN) | sales quote | sales quote |
| Thunderforest | $0 Hobby* | $0 Hobby* (128k of 150k) | $125 (Solo Developer) |
| Self-host on R2 | ~$1.5–4 | ~$1.5–4 | ~$1.5–4 |

\* The Hobby plan is offered at Thunderforest's discretion and framed for testing. Budget for $125/mo if they say a commercial app must be on a paid plan.

**Fixed yearly distribution costs**

| Setup | Cost |
|---|---|
| macOS only (MAS and/or notarized DMG), free app | **$99/yr** |
| + Windows via Microsoft Store only (MSIX, signed by the Store) | **+$0** → **$99/yr** |
| + Windows direct download, signed with Azure Artifact Signing Basic | **+$119.88/yr** → **$218.88/yr** |
| + Windows direct download, OV certificate instead | **+~$219/yr** → ~$318/yr |
| + Windows direct download, EV certificate instead | **+~$287/yr** → ~$386/yr |
| Optional domain for self-hosted tiles | ~$10–15/yr (estimate) |
| Commission | $0 if the app is free. Paid apps pay 15% (Small Business) or 30%. |

## Bottom line

- **Cheapest compliant on-screen use:** Esri Location Platform ($0 in all three scenarios) or Stadia Starter ($20/mo).
- **Cheapest print-licensed use:**
  - Stadia cacheable static maps: $20 / $80 / $475 per month, with a cap of 5,000 copies per image.
  - Thunderforest (CC-BY-SA): $0–$125 per month.
  - Self-hosting public-domain or Copernicus hillshade: ~$2–4 per month, no print limits, but the most engineering.
- **Avoid for printed handouts and publications unless you buy extra rights:** Mapbox Outdoors, MapTiler (Flex print is internal and 10 copies only), and keyless `server.arcgisonline.com`.
- **Confirm Esri's print terms in writing** before relying on them.

## Unverified items to confirm

1. Whether Stadia counts an @2x tile as 1 credit, and whether Stamen Terrain is available on `static_cacheable`.
2. Esri print, static-export and offline rights for Location Platform tiles, and whether World_Terrain_Base is available on `ibasemaps-api`.
3. MapTiler Custom print pricing.
4. Mapbox print-rights pricing, and whether @2x counts as one request.
5. Whether Thunderforest's Hobby plan may be used by a commercial app.
6. OpenTopoMap's official usage policy (page unreadable).
7. The exact commercial wording of the Copernicus DEM licence.
8. The Microsoft Store fee for company accounts.
9. The self-hosted hillshade storage size (my estimate).
