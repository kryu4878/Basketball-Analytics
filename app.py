"""Streamlit dashboard for exploring sample NBA analytics data."""
import streamlit as st
import altair as alt

from src import analytics

st.set_page_config(page_title="Basketball Analytics Lab", layout="wide")
st.title("🏀 Basketball Analytics Lab")
st.write(
    "Explore sample NBA data, inspect player trends, and generate quick projections for upcoming games."
)

team = st.selectbox("Select a team", options=analytics.team_list())
team_summary = analytics.compute_team_summary(team)

summary_cols = st.columns(len(team_summary))
for col, (metric, value) in zip(summary_cols, team_summary.items()):
    col.metric(metric, value)

trend = analytics.team_trend(team)
st.subheader(f"{team} efficiency trend")
trend_chart = (
    alt.Chart(trend)
    .transform_fold(["offensive_rating", "defensive_rating", "pace"], as_=["Metric", "Value"])
    .mark_line(point=True)
    .encode(
        x="date:T",
        y="Value:Q",
        color="Metric:N",
        tooltip=["date:T", "Metric:N", "Value:Q"],
    )
)
st.altair_chart(trend_chart, use_container_width=True)

st.subheader("Player lookup")
query = st.text_input("Search by player or team")
player_results = analytics.search_players(query)
st.dataframe(player_results, use_container_width=True)

if not player_results.empty:
    player_name = st.selectbox("Choose a player for projections", options=player_results["player"])
    projection = analytics.player_projection(player_name)
    st.write("### Projected stat line")
    for key, value in projection.items():
        st.write(f"**{key}:** {value}")

st.subheader("Upcoming game projections")
predictions = analytics.project_upcoming_games()
st.dataframe(predictions, use_container_width=True)
st.caption(
    "Predictions come from a quick regression/classification pipeline built on the sample data set, so treat them as illustrative only."
)

