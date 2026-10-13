#!/usr/bin/env python3
"""Shared global CSS for the teleop GUI.

Extracted byte-for-byte from app.py (v3 refactor) so that both the main
SPA page (`index()` in app.py) and the standalone login page
(`pages/login.py`) can `ui.add_css(CSS)` without a circular import.
"""

# ═══════════════════════════════════════════════════════════════════════
# Styles
# ═══════════════════════════════════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg: #F5F5F7;
    --surface: rgba(255,255,255,0.72);
    --surface-solid: #FFFFFF;
    --border: rgba(0,0,0,0.06);
    --border-hover: rgba(0,0,0,0.12);
    --text: #1D1D1F;
    --text-2: #86868B;
    --text-3: #AEAEB2;
    --blue: #007AFF;
    --green: #34C759;
    --amber: #FF9500;
    --red: #FF3B30;
    --r: 20px;
}
*, *::before, *::after { box-sizing: border-box; }
body {
    font-family: 'Inter', -apple-system, system-ui, sans-serif !important;
    background: var(--bg) !important;
    color: var(--text);
    -webkit-font-smoothing: antialiased;
    overflow: hidden;
}
/* Responsive content wrapper — remove NiceGUI default padding */
.nicegui-content { max-width: 100% !important; overflow-x: hidden !important; padding: 0 !important; margin: 0 !important; gap: 0 !important; }
.q-page { padding: 0 !important; min-height: 0 !important; }
.q-page-container { padding: 0 !important; }
.q-layout { min-height: 0 !important; }
/* NiceGUI 2.x defaults: every element gets flex column layout with gap.
 * Nuke all gaps and restore block-level stacking for our layout. */
.nicegui-column, .nicegui-row, .nicegui-element { gap: 0 !important; }
body .q-page > div { gap: 0 !important; }
/* Main area: plain block column; children (panels) stack naturally.
 * Panels toggle display:none/block via go_stage() — do NOT use !important
 * on display here or it breaks page switching. */
#main-area-block { display: block !important; }
#main-area-block > div { gap: 0 !important; }
#main-area-block > div > div { gap: 0 !important; }

/* Ambient glow — subtle warm tones on white */
.ambient { position: fixed; inset: 0; z-index: 0; pointer-events: none; }
.ambient::before {
    content: ''; position: absolute; width: 600px; height: 600px;
    top: -150px; right: -100px;
    background: radial-gradient(circle, rgba(0,122,255,0.04) 0%, transparent 70%);
    animation: float1 22s ease-in-out infinite;
}
.ambient::after {
    content: ''; position: absolute; width: 500px; height: 500px;
    bottom: -100px; left: -80px;
    background: radial-gradient(circle, rgba(52,199,89,0.03) 0%, transparent 70%);
    animation: float2 28s ease-in-out infinite;
}
@keyframes float1 { 0%,100%{transform:translate(0,0)} 50%{transform:translate(60px,50px)} }
@keyframes float2 { 0%,100%{transform:translate(0,0)} 50%{transform:translate(-50px,-40px)} }

/* Glass — Apple frosted glass on white */
.glass {
    background: var(--surface);
    backdrop-filter: blur(40px) saturate(180%);
    -webkit-backdrop-filter: blur(40px) saturate(180%);
    border: 1px solid var(--border);
    border-radius: var(--r);
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    transition: all 0.3s ease;
}
.glass:hover {
    border-color: var(--border-hover);
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}

/* Stage stepper */
.stage-dot {
    width: 36px; height: 36px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 700;
    transition: all 0.4s cubic-bezier(0.34,1.56,0.64,1);
}
.stage-dot.locked { background: rgba(0,0,0,0.05); color: var(--text-3); }
.stage-dot.current { background: var(--blue); color: white; box-shadow: 0 0 16px rgba(0,122,255,0.25); transform: scale(1.1); }
.stage-dot.done { background: var(--green); color: white; }
.stage-line { height: 2px; width: 40px; background: rgba(0,0,0,0.06); transition: background 0.4s; }
.stage-line.done { background: var(--green); }

/* Status pill */
.pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 20px;
    font-size: 14px; font-weight: 600;
    background: rgba(0,0,0,0.04);
    transition: all 0.3s;
}
.pill.on { background: rgba(52,199,89,0.10); color: var(--green); }
.pill.off { color: var(--text-3); }

/* Dot */
.dot { width: 10px; height: 10px; border-radius: 50%; transition: all 0.3s; }
.dot.on { background: var(--green); box-shadow: 0 0 6px rgba(52,199,89,0.35); }
.dot.off { background: rgba(0,0,0,0.12); }
.dot.warn { background: var(--amber); box-shadow: 0 0 6px rgba(255,149,0,0.35); }
.dot.err { background: var(--red); box-shadow: 0 0 8px rgba(255,59,48,0.35); animation: blink 0.6s ease infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.4} }
@keyframes pulseRec { 0%,100%{opacity:1; transform:scale(1)} 50%{opacity:0.55; transform:scale(0.85)} }

/* E-STOP */
.estop {
    width: 72px; height: 72px; border-radius: 50%;
    background: linear-gradient(135deg, #FF3B30, #D70015);
    border: 3px solid rgba(255,59,48,0.25);
    color: white; font-size: 9px; font-weight: 800; letter-spacing: 0.3px;
    cursor: pointer; transition: all 0.12s;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 16px rgba(255,59,48,0.2);
}
.estop:hover { transform: scale(1.06); box-shadow: 0 6px 24px rgba(255,59,48,0.3); }
.estop:active { transform: scale(0.9); }

/* Splash screen */
.splash {
    position: fixed; inset: 0; z-index: 9999;
    background: white;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    animation: splashFadeOut 0.6s ease 2.5s forwards;
}
.splash-logo {
    font-size: 42px; font-weight: 800; color: #1D1D1F;
    letter-spacing: -1.5px;
    animation: splashPulse 2s ease-in-out infinite;
}
.splash-sub {
    font-size: 13px; font-weight: 500; color: #86868B;
    letter-spacing: 4px; margin-top: 8px;
}
.splash-bar {
    width: 120px; height: 3px; background: #E5E5EA; border-radius: 2px;
    margin-top: 32px; overflow: hidden;
}
.splash-bar-fill {
    height: 100%; width: 0%; background: var(--blue); border-radius: 2px;
    animation: splashLoad 2.2s ease-out forwards;
}
@keyframes splashPulse { 0%,100%{opacity:1} 50%{opacity:0.7} }
@keyframes splashLoad { 0%{width:0%} 100%{width:100%} }
@keyframes splashFadeOut { to{opacity:0; visibility:hidden; height:0; padding:0; margin:0;} }

/* Metric */
.metric-num {
    font-size: 44px; font-weight: 800; letter-spacing: -2px; line-height: 1;
    font-feature-settings: 'tnum'; font-variant-numeric: tabular-nums;
    transition: color 0.5s;
}
.metric-unit { font-size: 16px; font-weight: 400; opacity: 0.35; margin-left: 3px; }
.metric-lbl { font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; opacity: 0.5; margin-top: 6px; }

/* Radar */
.radar {
    width: 180px; height: 180px; border-radius: 50%; position: relative;
    display: flex; align-items: center; justify-content: center;
}
.radar::before {
    content: ''; position: absolute; inset: 0; border-radius: 50%;
    border: 2.5px solid var(--green);
    box-shadow: 0 0 24px rgba(48,209,88,0.2), inset 0 0 24px rgba(48,209,88,0.1);
    transition: all 0.5s;
}
.radar.caution::before { border-color: var(--amber); box-shadow: 0 0 30px rgba(255,159,10,0.25), inset 0 0 30px rgba(255,159,10,0.12); }
.radar.danger::before { border-color: var(--red); box-shadow: 0 0 40px rgba(255,69,58,0.3), inset 0 0 40px rgba(255,69,58,0.15); animation: radar-pulse 0.7s ease infinite; }
@keyframes radar-pulse { 0%,100%{box-shadow:0 0 30px rgba(255,69,58,0.3),inset 0 0 30px rgba(255,69,58,0.15)} 50%{box-shadow:0 0 50px rgba(255,69,58,0.4),inset 0 0 50px rgba(255,69,58,0.2)} }

/* Responsive: stack operate page on narrow screens */
@media (max-width: 1024px) {
    .operate-main-layout { flex-direction: column !important; height: auto !important; }
    .operate-main-layout > div:first-child { min-width: 0; min-height: 400px; }
    .operate-main-layout > div:last-child { max-width: 100% !important; min-width: 0 !important; }
}

/* Responsive typography: scale down on narrow screens */
@media (max-width: 900px) {
    .page-enter h1, .page-enter > div:first-child > div:first-child {
        font-size: 24px !important;
    }
}

/* Chain grid breakpoints are handled via inline auto-fit minmax */

/* Wizard step */
.wiz-circle {
    width: 48px; height: 48px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; font-weight: 700; transition: all 0.4s;
}
.wiz-circle.pending { background: rgba(255,255,255,0.06); color: var(--text-3); }
.wiz-circle.active { background: var(--blue); color: white; box-shadow: 0 0 24px rgba(10,132,255,0.3); }
.wiz-circle.done { background: var(--green); color: white; }
.wiz-line { flex: 1; height: 2px; background: rgba(0,0,0,0.06); margin: 0 8px; }
.wiz-line.done { background: var(--green); }

/* Page transitions */
.page-enter { animation: slideIn 0.35s cubic-bezier(0.25,0.46,0.45,0.94); }
@keyframes slideIn { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }

/* Process manager */
.proc-row {
    display: flex; align-items: center; gap: 12px;
    padding: 12px 16px; border-radius: 12px;
    background: rgba(0,0,0,0.02);
    margin-bottom: 6px;
    transition: background 0.2s;
}
.proc-row:hover { background: rgba(0,0,0,0.03); }

/* Quasar overrides */
.q-tab { text-transform: none !important; }
.q-tab-panels { background: transparent !important; }
.q-tab-panel { padding: 0 !important; }
body .q-field__control { background: rgba(0,0,0,0.03) !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.10); border-radius: 3px; }
"""
