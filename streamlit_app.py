"""QuantCritic multipage Streamlit application entry point."""

from __future__ import annotations

import streamlit as st

pages = [
    st.Page("app_pages/home.py", title="Home", icon="🛡️", default=True),
    st.Page("pages/1_Backtest.py", title="Backtest", icon="📈"),
    st.Page("pages/2_Audit.py", title="Audit", icon="🧪"),
    st.Page("pages/3_AI_Critic.py", title="AI Critic", icon="🤖"),
    st.Page("pages/4_Leaderboard.py", title="Leaderboard", icon="🏆"),
    st.Page("pages/5_Lookahead_Lab.py", title="Lookahead Lab", icon="🔬"),
    st.Page("pages/7_Strategy_Builder.py", title="Strategy Builder", icon="🧩"),
    st.Page("pages/8_About_Methodology.py", title="About & Methodology", icon="📚"),
]

pg = st.navigation(pages)
pg.run()
