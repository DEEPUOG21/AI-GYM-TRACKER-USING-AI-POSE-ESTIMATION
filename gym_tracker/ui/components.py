"""Reusable presentation components. Charts only consume recorded telemetry."""
from datetime import datetime
from html import escape

import altair as alt
import pandas as pd
import streamlit as st

TITLES = {"bicep_curl": "Bicep Curl", "push_up": "Push Up", "squat": "Squat",
          "shoulder_press": "Shoulder Press", "auto": "Auto Classify"}
COLORS = ["#d7ff3f", "#70dbf4", "#c5a5ff", "#ffb86b"]


def html(markup):
    st.markdown(markup, unsafe_allow_html=True)


def page_heading(kicker, title, description):
    html(f'<div class="eyebrow">{escape(kicker)}</div>')
    st.title(title)
    st.caption(description)


def section_heading(title, detail=""):
    html(f'<div class="section-heading"><h3>{escape(title)}</h3><span>{escape(detail)}</span></div>')


def empty_state(symbol, title, description):
    html(f'<div class="empty-state"><span class="empty-symbol">{escape(symbol)}</span>'
         f'<h3>{escape(title)}</h3><p>{escape(description)}</p></div>')


def format_duration(seconds):
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes:02d}:{seconds:02d}"


def session_label(snapshot):
    timestamp = datetime.fromisoformat(snapshot["started_at"]).strftime("%d %b · %H:%M UTC")
    return f'{timestamp} · {format_duration(snapshot["duration_seconds"])} · {snapshot["session_id"][:6]}'


def chart_style(chart):
    return (chart.configure_view(strokeWidth=0).configure(background="transparent")
            .configure_axis(gridColor="#252b34", domain=False, labelColor="#9ca6b4",
                            titleColor="#9ca6b4", labelFontSize=11, titleFontSize=11)
            .configure_legend(labelColor="#c4cbd5", titleColor="#c4cbd5"))


def session_charts(snapshot):
    events = [{"Seconds": float(timestamp), "Rep": i + 1, "Exercise": TITLES[m["exercise"]]}
              for m in snapshot["exercises"]
              for i, timestamp in enumerate(m["rep_timestamps_seconds"])]
    left, right = st.columns([1.65, 1], gap="large")
    with left:
        section_heading("Your rep timeline", "HOVER TO EXPLORE")
        if events:
            chart = (alt.Chart(pd.DataFrame(events)).mark_line(interpolate="step-after", point=True, strokeWidth=2)
                     .encode(x=alt.X("Seconds:Q", title="Time into session (s)"),
                             y=alt.Y("Rep:Q", title="Completed reps", axis=alt.Axis(tickMinStep=1)),
                             color=alt.Color("Exercise:N", scale=alt.Scale(range=COLORS), legend=alt.Legend(orient="bottom")),
                             tooltip=["Exercise:N", alt.Tooltip("Seconds:Q", format=".2f"), "Rep:Q"])
                     .properties(height=235).interactive())
            st.altair_chart(chart_style(chart), use_container_width=True)
        else:
            empty_state("↗", "Your first rep starts the story", "Completed repetitions will appear on this timeline.")
    with right:
        section_heading("Time in motion", "OBSERVED DURATION")
        rows = [{"Exercise": TITLES[m["exercise"]], "Seconds": m["duration_seconds"]}
                for m in snapshot["exercises"] if m["duration_seconds"] > 0]
        if snapshot["unassigned_duration_seconds"] > 0:
            rows.append({"Exercise": "Unassigned", "Seconds": snapshot["unassigned_duration_seconds"]})
        if rows:
            chart = (alt.Chart(pd.DataFrame(rows)).mark_arc(innerRadius=65, outerRadius=95, cornerRadius=4, padAngle=0.025)
                     .encode(theta=alt.Theta("Seconds:Q"),
                             color=alt.Color("Exercise:N", scale=alt.Scale(range=COLORS + ["#495262"]),
                                             legend=alt.Legend(orient="bottom")),
                             tooltip=["Exercise:N", alt.Tooltip("Seconds:Q", format=".2f")])
                     .properties(height=235))
            st.altair_chart(chart_style(chart), use_container_width=True)
        else:
            empty_state("◷", "Make time for movement", "Exercise time appears as frames are processed.")


def history_chart(sessions):
    rows = [{"Session": i + 1, "Reps": sum(m["reps"] for m in s["exercises"]),
             "Duration (s)": s["duration_seconds"], "Recorded": session_label(s)}
            for i, s in enumerate(sessions)]
    chart = (alt.Chart(pd.DataFrame(rows)).mark_bar(color=COLORS[0], size=max(6, min(42, 180 / len(rows))),
                                                  cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
             .encode(x=alt.X("Session:O", title="Recorded session"), y=alt.Y("Reps:Q", axis=alt.Axis(tickMinStep=1)),
                     tooltip=["Recorded:N", "Reps:Q", alt.Tooltip("Duration (s):Q", format=".1f")])
             .properties(height=240))
    st.altair_chart(chart_style(chart), use_container_width=True)


def motion_hero():
    """Decorative SVG illustration, never presented as live measurement."""
    html('''<section class="hero">
    <div class="hero-copy"><span class="eyebrow">THE INTELLIGENCE BEHIND YOUR MOVEMENT</span>
      <h1>Every move.<br><em>More meaning.</em></h1>
      <p>Turn your workout into something you can see.<br>Track your reps. Understand your effort. Find your next step.</p>
      <div class="hero-tags"><span>POSE TRACKING</span><span>EXERCISE RECOGNITION</span><span>SESSION COACHING</span></div>
    </div>
    <div class="motion-scene" aria-label="Decorative pose tracking illustration">
      <div class="scene-grid"></div><div class="scan-line"></div>
      <span class="scene-label">MOTION / VISION</span>
      <svg viewBox="0 0 380 340" role="img" aria-label="Stylized athlete with highlighted pose landmarks">
       <defs><linearGradient id="body" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#57644c"/><stop offset="1" stop-color="#17201b"/></linearGradient></defs>
       <ellipse cx="196" cy="315" rx="112" ry="13" fill="#d7ff3f" opacity=".06"/>
       <circle cx="193" cy="157" r="120" fill="none" stroke="#d7ff3f" stroke-opacity=".12" stroke-dasharray="3 9"/>
       <circle cx="200" cy="157" r="88" fill="none" stroke="#d7ff3f" stroke-opacity=".08"/>
       <path d="M174 89 Q173 111 159 124 L143 184 L165 209 L197 190 L222 136 L214 103Z" fill="url(#body)" stroke="#748267" stroke-width="1"/>
       <circle cx="205" cy="65" r="24" fill="url(#body)" stroke="#748267"/>
       <path d="M165 120 L117 156 L74 133 M213 110 L256 144 L303 108 M164 200 L128 252 L82 298 M186 199 L221 249 L267 298" fill="none" stroke="#374436" stroke-width="22" stroke-linecap="round" stroke-linejoin="round"/>
       <path d="M205 65 L190 110 L165 120 L117 156 L74 133 M190 110 L213 110 L256 144 L303 108 M165 120 L164 200 L186 199 L213 110 M164 200 L128 252 L82 298 M186 199 L221 249 L267 298" fill="none" stroke="#d7ff3f" stroke-width="1.7" stroke-linecap="round" opacity=".9"/>
       <g fill="#d7ff3f" stroke="#20251d" stroke-width="3"><circle cx="205" cy="65" r="5"/><circle cx="165" cy="120" r="5"/><circle cx="213" cy="110" r="5"/><circle cx="117" cy="156" r="5"/><circle cx="256" cy="144" r="5"/><circle cx="74" cy="133" r="5"/><circle cx="303" cy="108" r="5"/><circle cx="164" cy="200" r="5"/><circle cx="186" cy="199" r="5"/><circle cx="128" cy="252" r="5"/><circle cx="221" cy="249" r="5"/><circle cx="82" cy="298" r="5"/><circle cx="267" cy="298" r="5"/></g>
       <path d="M26 56 V30 H52 M329 30 H355 V56 M26 280 V307 H52 M329 307 H355 V280" fill="none" stroke="#d7ff3f" stroke-opacity=".5"/>
      </svg>
      <span class="scene-foot">BODY LANDMARKS <b>33</b><small>POSE VISUALIZATION</small></span>
    </div></section>''')
