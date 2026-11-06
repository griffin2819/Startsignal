"""
NFL weekly signal report (v1)

Setup:   pip install nflreadpy pandas pyarrow
Run:     python nfl_report.py                 # latest week in the data
         python nfl_report.py --week 8        # a specific week
         python nfl_report.py --grade         # grade past flags against later weeks

Data: nflverse (CC-BY 4.0) -- credit "nflverse" in anything you publish.
Outputs: report_week<N>.md and scorecard.csv in the current folder.
"""
import argparse
import os
import sys

import pandas as pd

POSITIONS = ["RB", "WR", "TE"]  # used for trend signals
RANK_POS = ["QB", "RB", "WR", "TE"]  # used for rankings
RECENT = 2  # weeks counted as "recent"; everything earlier is the baseline
SCORECARD = "scorecard.csv"
# Waiver view assumes the season's top-N scorers at each position are already rostered.
# nflverse has no ownership data, so this is a heuristic; tune the numbers to your league size.
ROSTERED = {"QB": 15, "RB": 40, "WR": 50, "TE": 15}
# Per-position cutoffs for the usage lists (carries + targets per game). Receivers
# touch the ball less than running backs, so one shared threshold crowds them out.
MIN_DELTA = {"RB": 4.0, "WR": 3.0, "TE": 2.5}  # smallest change that counts
MIN_TOUCH = {"RB": 8.0, "WR": 5.0, "TE": 4.0}  # minimum recent (risers) / earlier (fallers) usage
NEEDED = ["player_id", "player_display_name", "position", "team", "week",
          "opponent_team", "carries", "targets", "target_share", "fantasy_points_ppr",
          "rushing_tds", "receiving_tds"]


def load_data(season: int) -> pd.DataFrame:
    import nflreadpy as nfl
    return nfl.load_player_stats([season]).to_pandas()


def prep(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in NEEDED if c not in df.columns]
    if missing:
        sys.exit(f"Missing columns {missing}. nflverse may have renamed them; "
                 "send me the error and I'll update the script.")
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"]
    df = df[df["position"].isin(RANK_POS)].copy()
    for c in ["carries", "targets", "target_share", "fantasy_points_ppr",
              "rushing_tds", "receiving_tds"]:
        df[c] = df[c].fillna(0)
    df["tds"] = df["rushing_tds"] + df["receiving_tds"]
    df["opps"] = df["carries"] + df["targets"]  # rushing + receiving opportunities
    return df


def build_signals(df: pd.DataFrame, week: int) -> pd.DataFrame:
    df = df[(df["week"] <= week) & df["position"].isin(POSITIONS)]
    recent = df[df["week"] > week - RECENT]
    base = df[df["week"] <= week - RECENT]

    # league-average fantasy points per opportunity, by position
    tot = df.groupby("position")[["fantasy_points_ppr", "opps"]].sum()
    ppo = (tot["fantasy_points_ppr"] / tot["opps"]).rename("ppo")

    ident = ["player_id", "player_display_name", "position", "team"]
    r = recent.groupby(ident).agg(
        g_rec=("week", "count"), opps_rec=("opps", "mean"),
        ts_rec=("target_share", "mean"), pts_rec=("fantasy_points_ppr", "mean"),
        tds_rec=("tds", "sum")).reset_index()
    b = base.groupby("player_id").agg(
        g_base=("week", "count"), opps_base=("opps", "mean"),
        ts_base=("target_share", "mean"), pts_base=("fantasy_points_ppr", "mean")
    ).reset_index()

    m = r.merge(b, on="player_id").merge(ppo, on="position")
    m = m[(m["g_rec"] == RECENT) & (m["g_base"] >= 3)].copy()  # skips bye/injury gaps
    m["opps_delta"] = m["opps_rec"] - m["opps_base"]
    m["ts_delta"] = m["ts_rec"] - m["ts_base"]
    # usage change measured against other players at the same position
    m["z_opps"] = m.groupby("position")["opps_delta"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else 0.0)
    # points above/below what their usage "should" produce at league-average efficiency
    m["resid"] = m["pts_rec"] - m["opps_rec"] * m["ppo"]
    return m


def flag(m: pd.DataFrame) -> dict:
    top = lambda d, col, asc=False: d.sort_values(col, ascending=asc).head(5)
    return {
        "Usage risers (more touches + targets)":
            top(m[(m.opps_delta >= m.position.map(MIN_DELTA))
                  & (m.opps_rec >= m.position.map(MIN_TOUCH))], "z_opps"),
        "Target-share risers":
            top(m[(m.ts_delta >= 0.06) & (m.ts_rec >= 0.15)], "ts_delta"),
        "Buy-low (heavy usage, low production)":
            top(m[(m.opps_rec >= 10) & (m.resid <= -4)], "resid", True),
        "Sell-high (TD-driven production)":
            top(m[(m.resid >= 6) & (m.tds_rec >= 2)], "resid"),
        "Usage fallers":
            top(m[(m.opps_delta <= -m.position.map(MIN_DELTA))
                  & (m.opps_base >= m.position.map(MIN_TOUCH))], "z_opps", True),
    }


KEY = {"Usage risers (more touches + targets)": ("opps_delta", "{:+.1f} opps/g"),
       "Target-share risers": ("ts_delta", "{:+.1%} target share"),
       "Buy-low (heavy usage, low production)": ("resid", "{:+.1f} pts/g vs usage"),
       "Sell-high (TD-driven production)": ("resid", "{:+.1f} pts/g vs usage"),
       "Usage fallers": ("opps_delta", "{:+.1f} opps/g")}


def groups_to_markdown(groups: list, season: int, week: int) -> str:
    out = [f"# NFL Trend Report: {season} Week {week}",
           f"_Compares the last {RECENT} games to each player's earlier baseline. "
           "Data: nflverse (CC-BY 4.0)._", ""]
    for g in groups:
        out.append(f"## {g['title']}")
        if not g["items"]:
            out += ["_No players met the threshold this week._", ""]
            continue
        for p in g["items"]:
            nxt = ""
            if p.get("opp"):
                rank = f" (#{p['oppRank']} defense vs {p['pos']})" if p.get("oppRank") else ""
                when = f"week {p['wk']}, " if p.get("wk") else ""
                nxt = f" Next: {when}vs {p['opp']}{rank}."
            inj = f" **{p['inj']}.**" if p.get("inj") else ""
            out.append(f"- **{p['name']}** ({p['pos']}, {p['team']}): {p['stat']}; "
                       f"{p['line']}. {p['why']}{inj}{nxt}")
        out.append("")
    return "\n".join(out)


def build_rankings(df):
    """Defense-vs-position, offense-by-position and player rankings (PPR/game)."""
    pts = "fantasy_points_ppr"

    def team_table(col):
        g = df.groupby([col, "week", "position"])[pts].sum().reset_index()
        g = g.groupby([col, "position"])[pts].mean().reset_index(name="ppg")
        out = {}
        for pos, d in g.groupby("position"):
            d = d.sort_values("ppg", ascending=False).reset_index(drop=True)
            out[pos] = [{"rank": i + 1, "team": r[col], "ppg": round(r.ppg, 1)}
                        for i, r in d.iterrows()]
        return out

    pl = df.groupby(["player_id", "player_display_name", "position", "team"]).agg(
        g=("week", "count"), ppg=(pts, "mean"), opps=("opps", "mean")).reset_index()
    players = {}
    for pos, d in pl[pl.g >= 3].groupby("position"):
        d = d.sort_values("ppg", ascending=False).head(25).reset_index(drop=True)
        players[pos] = [{"rank": i + 1, "id": r.player_id, "name": r.player_display_name, "team": r.team,
                         "g": int(r.g), "ppg": round(r.ppg, 1), "opps": round(r.opps, 1)}
                        for i, r in d.iterrows()]
    # defense rank 1 = allows the most points (best matchup); offense rank 1 = scores the most
    return {"defense": team_table("opponent_team"), "offense": team_table("team"),
            "players": players}


def build_waiver(df, week):
    """Likely-available players who are producing lately (last 4 games)."""
    pts = "fantasy_points_ppr"
    ident = ["player_id", "player_display_name", "position", "team"]
    s = df.groupby(ident).agg(gS=("week", "count"), ppgS=(pts, "mean"),
                              oppsS=("opps", "mean")).reset_index()
    l = df[df["week"] > week - 4].groupby("player_id").agg(
        g4=("week", "count"), ppg4=(pts, "mean"), opps4=("opps", "mean"),
        tds4=("tds", "sum")).reset_index()
    m = s.merge(l, on="player_id")
    m = m[m["g4"] >= 2].copy()
    m["srank"] = m[m["gS"] >= 3].groupby("position")["ppgS"].rank(
        ascending=False, method="first")
    out = {}
    for pos, d in m.groupby("position"):
        d = d[d["srank"].isna() | (d["srank"] > ROSTERED[pos])]
        if pos != "QB":
            d = d[d["opps4"] >= 5]  # filters out one-play TD flukes
        d = d.sort_values("ppg4", ascending=False).head(15)
        out[pos] = [{"id": r["player_id"], "name": r["player_display_name"], "team": r["team"],
                     "g4": int(r["g4"]), "ppg4": round(r["ppg4"], 1),
                     "ppgS": round(r["ppgS"], 1), "opps4": round(r["opps4"], 1),
                     "tds4": int(r["tds4"]),
                     "rank": None if pd.isna(r["srank"]) else int(r["srank"])}
                    for _, r in d.iterrows()]
    return out


def load_schedule(season):
    try:
        import nflreadpy as nfl
        sch = nfl.load_schedules([season]).to_pandas()
        if "game_type" in sch.columns:
            sch = sch[sch["game_type"] == "REG"]
        return sch[["week", "home_team", "away_team"]]
    except Exception as e:  # schedule problems shouldn't block the report
        print(f"Skipping schedule ({e})")
        return None


def next_matchups(sched, week):
    """team -> (opponent, week) for each team's next scheduled game after `week`.
    The game week is later than week + 1 when the team is on a bye."""
    m = {}
    if sched is not None:
        for _, g in sched[sched["week"] > week].sort_values("week").iterrows():
            m.setdefault(g["home_team"], (g["away_team"], int(g["week"])))
            m.setdefault(g["away_team"], (g["home_team"], int(g["week"])))
    return m


# ---- optional extras: each loader may fail without breaking the report ----
def _safe(label, fn, *a):
    try:
        return fn(*a)
    except Exception as e:
        print(f"Skipping {label} ({e})")
        return None


def _snaps(season):
    import nflreadpy as nfl
    sn = nfl.load_snap_counts([season]).to_pandas()
    ids = nfl.load_players().to_pandas()[["gsis_id", "pfr_id"]].dropna()
    sn = sn.merge(ids, left_on="pfr_player_id", right_on="pfr_id")
    sn = sn.rename(columns={"gsis_id": "player_id", "offense_pct": "snap_pct"})
    if sn["snap_pct"].max() <= 1.0:  # stored as a fraction
        sn["snap_pct"] *= 100
    return sn[["player_id", "week", "snap_pct"]].drop_duplicates(["player_id", "week"])


def _redzone(season):
    import nflreadpy as nfl
    cols = ["week", "yardline_100", "play_type", "receiver_player_id", "rusher_player_id"]
    p = nfl.load_pbp([season]).select(cols).to_pandas()
    p = p[p["yardline_100"] <= 20]
    t = p[p["play_type"] == "pass"].groupby(["receiver_player_id", "week"]).size()
    c = p[p["play_type"] == "run"].groupby(["rusher_player_id", "week"]).size()
    rz = pd.concat([t, c]).groupby(level=[0, 1]).sum().reset_index()
    rz.columns = ["player_id", "week", "rz"]
    return rz


def _injuries(season):
    import nflreadpy as nfl
    inj = nfl.load_injuries([season]).to_pandas()
    wk = int(inj["week"].max())
    d = {}
    for _, r in inj[inj["week"] == wk].iterrows():
        if pd.notna(r.get("report_status")):
            body = r.get("report_primary_injury")
            d[r["gsis_id"]] = r["report_status"] + (f" ({body})" if pd.notna(body) else "")
    return d, wk


def _depth(season):
    import nflreadpy as nfl
    d = nfl.load_depth_charts([season]).to_pandas()
    pos_c = next((c for c in ["pos_abb", "depth_position"] if c in d.columns), None)
    rank_c = next((c for c in ["pos_rank", "depth_team"] if c in d.columns), None)
    if not (pos_c and rank_c and "gsis_id" in d.columns):
        raise ValueError("unexpected depth chart columns")
    for t in ("week", "dt"):  # keep only the latest chart
        if t in d.columns:
            d = d[d[t] == d[t].max()]
            break
    return {r["gsis_id"]: f"{r[pos_c]}{int(float(r[rank_c]))}"
            for _, r in d.dropna(subset=["gsis_id", pos_c, rank_c]).iterrows()}


def attach_extras(df, season):
    df = df.copy()
    if "air_yards_share" not in df.columns:
        df["air_yards_share"] = float("nan")
    for label, fn in (("snap counts", _snaps), ("red-zone usage", _redzone)):
        x = _safe(label, fn, season)
        if x is not None:
            df = df.merge(x, on=["player_id", "week"], how="left")
    for c in ("snap_pct", "rz"):
        if c not in df.columns:
            df[c] = float("nan")
    return df


def build_sos(df, sched, week):
    """Schedule strength: average PPR points the upcoming opponents allow to each
    position this season. Rank 1 = softest schedule."""
    allowed = {pos: {r["team"]: r["ppg"] for r in rows}
               for pos, rows in build_rankings(df)["defense"].items()}
    out = {}
    for key, lo, hi in (("ros", week + 1, 99), ("next4", week + 1, week + 4),
                        ("playoffs", max(15, week + 1), 17)):  # fantasy playoffs
        games = sched[(sched["week"] >= lo) & (sched["week"] <= hi)]
        opps = {}
        for _, g in games.iterrows():
            opps.setdefault(g["home_team"], []).append(g["away_team"])
            opps.setdefault(g["away_team"], []).append(g["home_team"])
        out[key] = {}
        for pos, ppg in allowed.items():
            rows = [(t, sum(ppg[o] for o in os_ if o in ppg) / max(1, sum(o in ppg for o in os_)), len(os_))
                    for t, os_ in opps.items()]
            rows.sort(key=lambda x: -x[1])
            out[key][pos] = [{"rank": i + 1, "team": t, "avg": round(a, 1), "games": n}
                             for i, (t, a, n) in enumerate(rows)]
    return out


def build_players(df, injuries, depth, opp):
    num = lambda x: None if pd.isna(x) else round(float(x), 1)
    out = []
    for pid, g in df.sort_values("week").groupby("player_id"):
        if len(g) < 2:
            continue
        last = g.iloc[-1]
        log = [[int(r.week), r.opponent_team, num(r.fantasy_points_ppr), int(r.opps),
                num(r.target_share * 100), num(r.air_yards_share * 100),
                num(r.snap_pct), num(r.rz)] for r in g.itertuples()]
        out.append({"id": pid, "name": last["player_display_name"], "pos": last["position"],
                    "team": last["team"], "log": log, "inj": injuries.get(pid),
                    "depth": depth.get(pid), **opp(last["team"], last["position"])})
    return out


def write_json(flags, season, week, df, matchups, sched=None):
    import json
    df = attach_extras(df[df["week"] <= week], season)
    seas = build_rankings(df)
    last4 = build_rankings(df[df["week"] > week - 4])
    soft = {(r["team"], pos): r["rank"]
            for pos, rows in seas["defense"].items() for r in rows}  # 1 = softest

    def opp(team, pos):
        g = matchups.get(team)
        if not g:
            return {}
        o, wk = g
        d = {"opp": o, "oppRank": soft.get((o, pos))}
        if wk != week + 1:  # bye next week: say which week the next game is
            d["wk"] = wk
        return d

    inj = _safe("injuries", _injuries, season) or ({}, None)
    depth = _safe("depth charts", _depth, season) or {}

    rec_s = df[df["week"] > week - RECENT].groupby("player_id")["snap_pct"].mean()
    base_s = df[df["week"] <= week - RECENT].groupby("player_id")["snap_pct"].mean()

    def why(t, p):
        """One plain-English sentence explaining why the player was flagged."""
        if t.startswith("Usage"):
            kind = "up" if p.opps_delta > 0 else "down"
            s = f"Carries + targets {kind} from {p.opps_base:.0f} to {p.opps_rec:.0f} per game."
        elif t.startswith("Target-share"):
            s = f"Target share up from {p.ts_base * 100:.0f}% to {p.ts_rec * 100:.0f}%."
        elif t.startswith("Buy-low"):
            s = (f"Getting {p.opps_rec:.0f} carries + targets a game but scoring about "
                 f"{abs(p.resid):.0f} PPR points a game less than that usage usually brings.")
        else:
            s = (f"Scoring about {p.resid:.0f} PPR points a game more than that usage usually "
                 f"brings, with {int(p.tds_rec)} touchdowns in the last {RECENT} games.")
            if p.opps_delta >= MIN_DELTA.get(p.position, 4.0):
                s += (f" Usage is also up ({p.opps_base:.0f} to {p.opps_rec:.0f} per game), "
                      "so the role looks real, but the touchdown pace won't hold.")
        sb, sr = base_s.get(p.player_id), rec_s.get(p.player_id)
        if pd.notna(sb) and pd.notna(sr):
            s += f" Snap share {sb:.0f}% to {sr:.0f}%."
        return s

    groups = [{"title": t, "items": [
        {"id": p.player_id, "name": p.player_display_name, "pos": p.position, "team": p.team,
         "stat": KEY[t][1].format(p[KEY[t][0]]),
         "line": f"{p.pts_rec:.1f} PPR/g vs {p.pts_base:.1f} before",
         "why": why(t, p),
         "inj": inj[0].get(p.player_id),
         **opp(p.team, p.position)}
        for _, p in d.iterrows()]} for t, d in flags.items()]
    for rk in (seas, last4):
        for pos, rows in rk["players"].items():
            for r in rows:
                r.update(opp(r["team"], pos))
    waiver = build_waiver(df, week)
    for pos, rows in waiver.items():
        for r in rows:
            r.update(opp(r["team"], pos))
            r["inj"] = inj[0].get(r["id"])
    sos = build_sos(df, sched, week) if sched is not None else None
    with open("report.json", "w") as f:
        json.dump({"season": season, "week": week, "injWeek": inj[1],
                   "nextWeek": week + 1 if matchups else None, "groups": groups,
                   "rankings": {"season": seas, "last4": last4},
                   "waiver": waiver, "sos": sos}, f)
    with open("players.json", "w") as f:
        json.dump(build_players(df, inj[0], depth, opp), f)
    return groups


def log_scorecard(flags: dict, season: int, week: int) -> None:
    rows = []
    for signal, d in flags.items():
        for _, p in d.iterrows():
            rows.append({"season": season, "week": week, "signal": signal,
                         "player_id": p.player_id, "player": p.player_display_name,
                         "pts_base": round(p.pts_base, 2), "next_pts": None, "hit": None})
    new = pd.DataFrame(rows)
    if os.path.exists(SCORECARD):
        old = pd.read_csv(SCORECARD)
        old = old[~((old.season == season) & (old.week == week))]  # re-runs replace
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(SCORECARD, index=False)


def grade(df: pd.DataFrame, season: int) -> None:
    """v1 rule: 'up' signals hit if next game beats the player's baseline PPR avg;
    'down' signals (sell-high, fallers) hit if it falls short of it."""
    if not os.path.exists(SCORECARD):
        sys.exit("No scorecard.csv yet. Run a report first.")
    sc = pd.read_csv(SCORECARD)
    nxt = df[["player_id", "week", "fantasy_points_ppr"]].rename(
        columns={"week": "next_week", "fantasy_points_ppr": "np"})
    sc["next_week"] = sc["week"] + 1
    sc = sc.merge(nxt, on=["player_id", "next_week"], how="left")
    sc["next_pts"] = sc["np"].combine_first(sc["next_pts"])
    down = sc["signal"].str.startswith(("Sell-high", "Usage fallers"))
    sc["hit"] = sc.apply(lambda r: None if pd.isna(r.next_pts) else
                         int((r.next_pts < r.pts_base) if down[r.name]
                             else (r.next_pts > r.pts_base)), axis=1)
    sc.drop(columns=["next_week", "np"]).to_csv(SCORECARD, index=False)
    graded = sc.dropna(subset=["hit"])
    print(f"Graded {len(graded)} of {len(sc)} flags")
    if len(graded):
        print(graded.groupby("signal")["hit"].agg(["count", "mean"]).round(2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int)
    ap.add_argument("--grade", action="store_true")
    a = ap.parse_args()

    df = prep(load_data(a.season))
    if a.grade:
        return grade(df, a.season)
    week = a.week or int(df["week"].max())
    flags = flag(build_signals(df, week))
    sched = load_schedule(a.season)
    groups = write_json(flags, a.season, week, df, next_matchups(sched, week), sched)
    md = groups_to_markdown(groups, a.season, week)
    with open(f"report_week{week}.md", "w") as f:
        f.write(md)
    log_scorecard(flags, a.season, week)
    print(md)


if __name__ == "__main__":
    main()
