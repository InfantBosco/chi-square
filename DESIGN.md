# Design System: The Investigative Paper
**Codename:** Broadside Paper UI  
**Application:** AI-Based Fake News Detector  
**Document Type:** Complete Design System Specification  
**Version:** 2.0  

---

## 1. Design Philosophy & Aesthetic Identity

### 1.1 Metaphor: "The Newsroom Investigation Board"
The design system fuses the tactile authenticity of traditional print journalism with the high-contrast precision of modern digital verification tools. Instead of sterile, generic SaaS templates or dark cyberpunk AI glows, the interface is styled as an **investigative desk**:
- **Tactile Paper Skeuomorphism:** Physical deckle-edge torn paper silhouettes, warm fibrous newsprint surfaces, translucent washi drafting tape, and heavy carbon press ink.
- **Forensic Clarity:** Monospace telemetry, calibrated confidence progress meters, and verification stamp seals reminiscent of official editorial approvals or redaction stamps.

### 1.2 System Dimensions
- **Density:** `6/10` (Editorial Balanced — generous reading margins paired with dense data chips).
- **Variance:** `7/10` (Asymmetric organic tape rotations, structured editorial card stack).
- **Motion:** `5/10` (Subtle tactile button depression, smooth linear confidence bar easing, zero bouncy gimmicks).

---

## 2. Color Token Architecture

The design system enforces a strict 3-layer token hierarchy: **Primitive → Semantic → Component**.

### 2.1 Primitive Neutrals (Paper & Press Ink)
All neutrals are calibrated to warm newsprint rather than cold blue-gray or pure black.

```css
:root {
    /* Paper Canvas & Surfaces */
    --bg-app:          #F6F4EE;  /* Warm newsprint desktop canvas */
    --bg-surface:      #FAF9F5;  /* Secondary editorial sheet */
    --bg-card:         #FDFAF5;  /* Primary paper document */
    --bg-subtle:       #EFECE4;  /* Inactive tab tray & recessed surfaces */
    
    /* Paper Fiber Borders */
    --border-color:    #E2DDD3;  /* Aged paper boundary stroke */
    --border-subtle:   #ECE8DF;  /* Delicate ruled column divider */
    --border-hover:    #D0C9BD;  /* Active hover border */
    --border-focus:    #1C1917;  /* High-contrast active carbon stroke */
    
    /* Press Ink Typography */
    --text-main:       #1C1917;  /* Carbon black printing ink */
    --text-muted:      #57534E;  /* Newspaper column body charcoal */
    --text-light:      #8C857B;  /* Metadata pencil graphite */
    
    /* Primary Action Contrast */
    --color-primary:       #1C1917;
    --color-primary-hover: #292524;
}
```

### 2.2 Semantic Veracity Accents
Colors signify verified truth, identified disinformation, or archival records. Saturation is held below 80% to maintain editorial gravitas.

```css
:root {
    /* Verified Real News (Official Press Stamp) */
    --real-bg:         #F0FDF4;  /* Pale mint verification sheet */
    --real-text:       #166534;  /* Deep forest press stamp ink */
    --real-border:     #BBF7D0;  /* Stamp stroke */
    --real-fill:       #16A34A;  /* Meter progress bar fill */
    --real-glow:       rgba(22, 163, 74, 0.10);
    
    /* Fabricated / Fake News (Redaction & Alarm Stamp) */
    --fake-bg:         #FEF2F2;  /* Pale crimson alarm sheet */
    --fake-text:       #991B1B;  /* Deep red ink stamp */
    --fake-border:     #FECACA;  /* Stamp stroke */
    --fake-fill:       #DC2626;  /* Meter progress bar fill */
    --fake-glow:       rgba(220, 38, 38, 0.10);

    /* Verified Ground Truth (Authoritative Archival Seal) */
    --truth-bg:        #FEFCE8;  /* Parchment archive yellow */
    --truth-text:      #854D0E;  /* Bronze seal text */
    --truth-border:    #FEF08A;  /* Warm gold rule */
    
    /* Washi Drafting Tape (Physical Adhesive) */
    --tape-bg:         rgba(246, 238, 220, 0.85); /* Frosted masking tape */
    --tape-border:     rgba(160, 138, 105, 0.55); /* Perforated dashed edge */
}
```

---

## 3. Typographic Architecture

### 3.1 Typeface Pairings
1. **Primary Editorial Sans:** `'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`  
   *Role:* Headlines, tab labels, body copy, and UI navigation. Selected for clean modern geometry paired with open humanist apertures.
2. **Forensic Monospace:** `'JetBrains Mono', 'SF Mono', Consolas, monospace`  
   *Role:* Telemetry values, confidence figures, latency metrics, token attribution scores, and report certificate IDs.

### 3.2 Typographic Hierarchy & Scale
| Role | Font Family | Size | Weight | Line Height | Tracking | Text Transform |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Masthead Title** | Sans | 2.25rem (36px) | 800 | 1.15 | -0.03em | None |
| **Section Header** | Sans | 1.25rem (20px) | 700 | 1.30 | -0.02em | None |
| **Card Header** | Sans | 0.94rem (15px) | 700 | 1.20 | -0.01em | None |
| **Verdict Stamp** | Sans | 0.94rem (15px) | 800 | 1.00 | +0.03em | UPPERCASE |
| **Confidence Num**| Sans | 2.50rem (40px) | 800 | 1.00 | -0.04em | Tabular |
| **Body Editorial** | Sans | 0.94rem (15px) | 400 | 1.55 | normal | Sentence case |
| **Pencil Metadata**| Sans | 0.72rem (11.5px)| 700 | 1.20 | +0.05em | UPPERCASE |
| **Forensic Chip** | Mono | 0.77rem (12.3px)| 500 | 1.25 | -0.01em | None |
| **Telemetry Badge**| Mono | 0.72rem (11.5px)| 500 | 1.20 | 0.00 | None |

### 3.3 Measure & Line Length
- Editorial text blocks are strictly constrained to a maximum of **65–70 characters per line** for optimal human readability.

---

## 4. Skeuomorphic Paper Geometry & Textures

### 4.1 The "Torn Paper Note" Silhouette
Cards mimic physical torn paper broadsides. Instead of standard rounded CSS corners, cards use organic SVG serrated edge masks at top and bottom.

```css
.card-torn-shadow-wrap {
    position: relative;
    margin-top: 40px;
    margin-bottom: 50px;
    /* Realistic Triple-Layer Paper Drop Shadow */
    filter: 
        drop-shadow(0 2px 4px rgba(45, 34, 20, 0.12))
        drop-shadow(0 10px 22px rgba(45, 34, 20, 0.15))
        drop-shadow(0 24px 44px rgba(45, 34, 20, 0.09));
    transition: filter 0.25s ease;
}

.card {
    position: relative;
    background-color: var(--bg-card);
    background-image: 
        radial-gradient(circle at 15% 15%, rgba(255, 255, 255, 0.6) 0%, transparent 60%),
        url('card-paper-texture.svg');
    background-repeat: repeat;
    background-size: auto, 300px 300px;
    border: none !important;
    border-radius: 0 !important;
    padding: 34px 32px 30px 32px;
}
```

### 4.2 Frosted Washi Drafting Tape Accent
Every primary paper sheet is "pinned" to the board with a semi-translucent piece of drafting tape featuring a slight organic tilt:

```css
.card-tape {
    position: absolute;
    top: -12px;
    left: 50%;
    transform: translateX(-50%) rotate(-1.5deg);
    width: 140px;
    height: 32px;
    background: var(--tape-bg);
    backdrop-filter: blur(2px);
    border-left: 2px dashed var(--tape-border);
    border-right: 2px dashed var(--tape-border);
    box-shadow: 
        0 1px 4px rgba(45, 34, 20, 0.18),
        inset 0 1px 1px rgba(255, 255, 255, 0.7);
    z-index: 10;
    pointer-events: none;
}

/* Counter-balanced angle on the result card */
.card-tape-result {
    transform: translateX(-50%) rotate(1.2deg);
}
```

---

## 5. Component Specifications

### 5.1 Application Header & Masthead
- **Brand Pill:** Inactive state pill containing a pulsing green status dot (`#16A34A`) confirming live system readiness.
- **Eyebrow Chip:** Small uppercase monospace chip reading `INTELLIGENT AGENT · REAL-TIME VERIFICATION`.

### 5.2 Multi-Modal Ingestion Tabs (Segmented Tray)
Recessed pill container (`--bg-subtle`) holding 3 modal options:
1. **Direct Text Analysis** (`tab-text`)
2. **Article Web Link** (`tab-url`)
3. **Screenshot / Photo OCR** (`tab-image`)

- **Active State:** Clean paper elevation (`background: #FFFFFF`), carbon text, subtle shadow (`0 2px 8px rgba(44, 38, 28, 0.08)`).
- **Inactive State:** Recessed charcoal text (`--text-muted`), transparent background.

### 5.3 Primary Input Controls
- **Textarea:** Clean newsprint paper background, no harsh black borders, relaxed line height, bottom-docked word and character counters.
- **URL Scraper Bar:** Integrated single-line input field paired with an embedded `Fetch & Clean` trigger button.
- **OCR Dropzone:** Dashed border container (`border: 2px dashed var(--border-hover)`) supporting drag-and-drop, camera uploads, and local WebAssembly character extraction progress meters.

### 5.4 Primary & Secondary Action Buttons
```css
.primary-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 12px 24px;
    background-color: var(--color-primary);
    color: #FFFFFF;
    font-weight: 700;
    font-size: 0.94rem;
    border-radius: var(--radius-sm);
    box-shadow: 0 2px 4px rgba(28, 25, 23, 0.15);
    transition: all 0.15s ease;
}

.primary-btn:hover {
    background-color: var(--color-primary-hover);
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(28, 25, 23, 0.20);
}

.primary-btn:active {
    transform: translateY(1px);
    box-shadow: 0 1px 2px rgba(28, 25, 23, 0.15);
}
```

### 5.5 The Verdict Card Header (Stamp + Percentage)
- **Verdict Stamp:** Bordered press stamp with uppercase tracking:
  - `REAL NEWS`: Forest green on pale mint wash (`--real-bg`, `--real-text`).
  - `FAKE NEWS`: Deep crimson on pale red wash (`--fake-bg`, `--fake-text`).
- **Confidence Number:** Large tabular typography (`2.5rem`, `800 weight`) paired with a label above.
- **Confidence Bar Meter:** Clean horizontal track with a dynamic colored fill bar transitioning smoothly to the target percentage.

### 5.6 Verified Ground Truth Callout
Rendered exclusively when a false claim conflicts with verified knowledge graph facts:
- **Badge:** `VERIFIED GROUND TRUTH` on archival gold parchment (`--truth-bg`, `--truth-text`).
- **Heading:** `"What is actually true:"` followed by the verified factual assertion.

### 5.7 Forensic Stylometry Bento Grid
A balanced 4-column data grid measuring textual linguistic markers:
- **Sensationalism Index:** Proportional score chip (`Low`, `Moderate`, `High`).
- **Punctuation Dramatism:** Exclamation and question mark density tracker.
- **Capitalization Ratio:** ALL-CAPS percentage.
- **Style Verdict:** Plain-English tag (`Sensationalist / Clickbait`, `Neutral / Conversational`, `Formal Journalistic`).

### 5.8 Explainable AI (XAI) Vocabulary Chips
Interactive keyword badges illustrating feature contribution:
- **Driving toward REAL:** Green pill tag with a `+` score.
- **Driving toward FAKE:** Crimson pill tag with a `-` score.

---

## 6. Motion Physics & State Transitions

1. **Active Button Feedback:**
   `transform: translateY(1px)` on `:active` with an immediate 50ms tactile response.
2. **Confidence Progress Bar Fill:**
   Hardware-accelerated CSS width transition:
   ```css
   transition: width 0.85s cubic-bezier(0.16, 1, 0.3, 1);
   ```
3. **Card Reveal Animation:**
   The result card reveals with a subtle upward fade (`opacity: 0 -> 1`, `transform: translateY(12px) -> translateY(0)` over 280ms).
4. **Live Status Pulsing:**
   Subtle opacity pulse (`opacity: 1 -> 0.4 -> 1` over 2s) on the header status beacon.

---

## 7. Layout Principles & Spacing Scale

### 7.1 Container Constraints
- **Primary Column:** Strictly constrained to `max-width: 860px` centered horizontally.
- **Viewport Padding:** `24px` on desktop, `16px` on mobile screens.

### 7.2 Spacing Hierarchy
```css
--space-xs: 4px;
--space-sm: 8px;
--space-md: 12px;
--space-lg: 16px;
--space-xl: 24px;
--space-2xl: 32px;
--space-3xl: 40px;
--space-4xl: 50px;
```

---

## 8. Strict Anti-Patterns & Banned Elements

1. **NO Neon AI Glows:** Glowing cyan, electric purple, and neon outlines are strictly forbidden.
2. **NO Pure Black (`#000000`):** Pure black destroys paper depth. Deep carbon black (`#1C1917`) is used exclusively.
3. **NO Cold Gray Backgrounds:** Blue-tinted grays are banned. The palette strictly adheres to warm newsprint creams and natural parchment fibers.
4. **NO Generic Circular Spinners:** Loading states employ stepped editorial progress bars and descriptive status text.
5. **NO Floating Overlapping Cards:** All elements occupy dedicated, non-overlapping spatial zones.
