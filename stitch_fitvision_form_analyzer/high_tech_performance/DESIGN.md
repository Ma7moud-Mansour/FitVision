---
name: High-Tech Performance
colors:
  surface: '#131313'
  surface-dim: '#131313'
  surface-bright: '#393939'
  surface-container-lowest: '#0e0e0e'
  surface-container-low: '#1c1b1b'
  surface-container: '#201f1f'
  surface-container-high: '#2a2a2a'
  surface-container-highest: '#353534'
  on-surface: '#e5e2e1'
  on-surface-variant: '#c1c6d7'
  inverse-surface: '#e5e2e1'
  inverse-on-surface: '#313030'
  outline: '#8b90a0'
  outline-variant: '#414755'
  surface-tint: '#adc6ff'
  primary: '#adc6ff'
  on-primary: '#002e69'
  primary-container: '#4b8eff'
  on-primary-container: '#00285c'
  inverse-primary: '#005bc1'
  secondary: '#d7ffc5'
  on-secondary: '#053900'
  secondary-container: '#2ff801'
  on-secondary-container: '#0f6d00'
  tertiary: '#ffb77d'
  on-tertiary: '#4d2600'
  tertiary-container: '#da7700'
  on-tertiary-container: '#432100'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#d8e2ff'
  primary-fixed-dim: '#adc6ff'
  on-primary-fixed: '#001a41'
  on-primary-fixed-variant: '#004493'
  secondary-fixed: '#79ff5b'
  secondary-fixed-dim: '#2ae500'
  on-secondary-fixed: '#022100'
  on-secondary-fixed-variant: '#095300'
  tertiary-fixed: '#ffdcc3'
  tertiary-fixed-dim: '#ffb77d'
  on-tertiary-fixed: '#2f1500'
  on-tertiary-fixed-variant: '#6e3900'
  background: '#131313'
  on-background: '#e5e2e1'
  surface-variant: '#353534'
typography:
  display-xl:
    fontFamily: Geist
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  display-xl-mobile:
    fontFamily: Geist
    fontSize: 36px
    fontWeight: '700'
    lineHeight: 42px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Geist
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
  headline-md:
    fontFamily: Geist
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  label-caps:
    fontFamily: Geist
    fontSize: 12px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.1em
  mono-metrics:
    fontFamily: Geist
    fontSize: 20px
    fontWeight: '700'
    lineHeight: 24px
    letterSpacing: 0.05em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  base: 4px
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 48px
  container-max: 1280px
  gutter: 20px
---

## Brand & Style

The design system is engineered to evoke the feeling of a high-end cockpit or a laboratory-grade performance lab. The brand personality is rooted in **precision, motivation, and cutting-edge intelligence**. It treats every pixel as a data point, prioritizing high legibility and a sophisticated aesthetic that mirrors the accuracy of the AI-powered computer vision technology.

The visual style is a fusion of **Glassmorphism** and **High-Tech Minimalism**. We utilize deep, layered backgrounds to create a sense of infinite digital space, while foreground elements are treated as translucent "heads-up display" (HUD) panels. The aesthetic avoids unnecessary decoration, focusing instead on structural integrity, crisp lines, and subtle luminescent signals that guide the athlete through their data.

## Colors

The palette is optimized for high-contrast environments and peak focus. 

- **Primary (Electric Blue):** Used for critical calls to action, active tracking states, and focal points of the AI analysis. It represents the "intelligence" of the system.
- **Success (Neon Green):** Reserved for positive performance metrics, rep completion, and perfect form alignment. It provides an immediate hit of dopaminergic feedback.
- **Warning (Deep Orange):** Dedicated to form correction and safety alerts. It is distinct from the primary blue to ensure immediate cognitive processing during high-intensity training.
- **Backgrounds (Deep Charcoal):** A tiered system of dark greys (#121212, #1C1C1E) provides a stable, low-strain foundation that makes the accent colors vibrate with energy.

## Typography

This design system utilizes a dual-font strategy to balance technical precision with extreme legibility. **Geist** is used for headlines, labels, and numeric data to reinforce the "developer-grade" technical nature of the AI. **Inter** is utilized for body copy and instructions to ensure maximum readability during physical exertion when the user may be viewing the screen from a distance.

Numerical data—such as rep counts and velocity metrics—should use the `mono-metrics` style to ensure characters do not jump horizontally as values change rapidly. All labels for metrics should be presented in `label-caps` for a distinct HUD aesthetic.

## Layout & Spacing

The layout utilizes a **12-column fluid grid** for desktop and a **4-column grid** for mobile. Because the user is often moving during a workout, the system employs aggressive safe-area margins and generous touch targets.

- **Desktop/Tablet:** Content is centered in a max-width container of 1280px.
- **Mobile:** Horizontal margins are set to 24px to prevent content from hitting screen edges on curved devices.
- **Spacing Rhythm:** Based on a 4px baseline. Components should generally use `md` (16px) or `lg` (24px) padding to maintain a "breathable" high-tech feel.

## Elevation & Depth

Hierarchy is achieved through **Glassmorphism** rather than traditional shadows. This mimics the semi-transparent overlays found in high-performance optics.

1.  **Level 0 (Floor):** Deep charcoal solid background (#121212).
2.  **Level 1 (Cards/Panels):** Background blur (20px to 40px) with a semi-transparent fill (White at 5-8% opacity).
3.  **Level 2 (Active/Focus):** Increased fill opacity (12%) and a **1px crisp border** using the Primary Blue at 30% opacity.
4.  **Accent Elevation:** Important metrics or AI-detected joints feature a subtle "outer glow" using a low-spread drop shadow with the accent color (e.g., 0px 0px 15px rgba(0, 122, 255, 0.4)).

## Shapes

The shape language of this design system is **structured and precise**. We use a "Soft" roundedness level (4px - 12px) rather than fully rounded or pill shapes to maintain a professional, engineered look.

- **Primary Containers:** 8px (`rounded-lg`) for standard cards and workout modules.
- **Interactive Elements:** 4px (default) for buttons and input fields to give them a "machined" appearance.
- **Data Visualizations:** Strict 90-degree angles are used for bar charts, while line graphs use subtle curves to represent fluid human movement.

## Components

- **Buttons:** Primary buttons use a solid Electric Blue fill with white text. Secondary buttons are "ghost" style with a 1px Blue border. All buttons should have a high-contrast hover state that increases the brightness of the blue.
- **Glass Cards:** The signature component. Must include a `backdrop-filter: blur(20px)` and a `border: 1px solid rgba(255,255,255,0.1)`. 
- **Metric Chips:** Small, high-contrast badges used for "BPM," "Power Output," or "Form Quality." They use a dark fill with the text and border colored by the metric's status (Green/Blue/Orange).
- **Form Correction Overlays:** These are dynamic, floating alerts that use the Warning Orange. They should include a pulsing glow effect to capture attention during movement.
- **Input Fields:** Dark, recessed backgrounds with a 1px border that glows Electric Blue upon focus. 
- **AI Vision Indicators:** Thin, 1px lines and circular "nodes" that track body joints. These nodes should have a slight "ping" animation when the system successfully locks onto a limb.