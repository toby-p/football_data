import pandas as pd
import os
import datetime

class footballData:
    def __init__(self, csv_dir):
        """Args:
            csv_dir (str): filepath location of CSV files downloaded from:
                http://www.football-data.co.uk/englandm.php
        """
        self.CSV_DIR = csv_dir
        self.csvs = [f for f in os.listdir(self.CSV_DIR) if f[-4:]==".csv"]
        data = self._join_multi_csvs(self.csvs)
        self.DF = data[0]
        self.csv_loading_errors = data[1]
        self._add_odds_cols()
        self._add_dt_cols()
        self.DF.sort_values(by="datetime", inplace=True)
        self.DF.reset_index(inplace=True, drop=True)

    def open_csv(self, filename, encoding="latin1"):
        """Open a CSV from the filepath; return a Pandas DataFrame."""
        fp = os.path.join(self.CSV_DIR, filename)
        return pd.read_csv(fp, encoding=encoding)

    def _join_multi_csvs(self, list_of_filenames):
        """Open multiple CSVs in same format and append them together into a
        single Pandas DataFrame."""
        list_of_dfs, errors = [], []
        for f in list_of_filenames:
            try:
                df = self.open_csv(f)
                # Drop empty columns & rows:
                df.dropna(axis=0, how="all", inplace=True)
                df.dropna(axis=1, how="all", inplace=True)
                list_of_dfs.append(df)

                # Sanity check that no. results in csv = n_teams*(n_teams-1)
                n_teams = len(df["HomeTeam"].unique())
                exp = n_teams*(n_teams-1)
                actual = len(df)
                if actual!=exp:
                    e = "Expected {} results, received {}".format(exp, actual)
                    errors.append({f:e})
            except Exception as e:
                errors.append({f:e})

        df = pd.concat(list_of_dfs)
        return df.reset_index(drop=True), errors

    def _add_odds_cols(self):
        for r in ["H", "A", "D"]:
            cols = [c for c in self.betting_columns if c[2]==r]
            for f in ["min", "max", "mean"]:
                c = "{}_{}".format(r, f)
                self.DF[c] = eval("self.DF[cols].{}(skipna=True, axis=1)".format(f))

    def _add_dt_cols(self):
        self.DF["datetime"] = pd.NaT
        fmts = ["%d/%m/%y", "%d/%m/%Y"]
        for f in fmts:
            self.DF["dtcol"] = pd.to_datetime(self.DF["Date"],
                                              format=f,
                                              errors="coerce")
            self.DF["datetime"].fillna(self.DF["dtcol"], inplace=True)
        self.DF.drop(columns=["dtcol"], inplace=True)
        self.DF["mm/yyyy"] = self.DF["datetime"].apply(lambda x: x.strftime("%m/%Y"))
        self.DF["Season"] = self.DF["mm/yyyy"].apply(self._mmyyyy_to_season)
        self.DF["str_dt"] = self.DF["datetime"].dt.strftime("%d/%m/%y")

    @staticmethod
    def _mmyyyy_to_season(mmyyyy):
        """Get the season from the `mm/yyyy` string."""
        y = mmyyyy[-4:]
        m = mmyyyy[:2]
        ms = {"01":1, "02":1, "03":1, "04":1, "05":1, "06":1,
              "08":0, "09":0, "10":0, "11":0, "12":0}
        y1 = int(y)-ms[m]
        y2 = y1+1
        return str(y1)[-2:]+"/"+str(y2)[-2:]

    def _basic_cols(func):
        """Decorator to remove all but the most important columns."""
        def remove_cols(*a, **kw):
            df = func(*a, **kw)
            cols = ["Div", "Season", "HomeTeam", "AwayTeam", "FTR", "FTHG",
                    "FTAG", "mm/yyyy", "datetime", "str_dt"]
            for r in ["H", "D", "A"]:
                for f in ["min", "max", "mean"]:
                    cols.append("{}_{}".format(r, f))
            return df[cols]
        return remove_cols

    @_basic_cols
    def subset(self, teams=None, seasons=None, divisions=None, min_date=None,
               max_date=None,**kwargs):
        """Get a subset of the main DataFrame based on the criteria supplied.

        Args:
            teams (str/list of str):
            seasons (str/list of str):
            divisions (str/list of str):
            min_date, max_date (str): date limits in format `DD/MM/YY`

        Kwargs:
            df (Pandas DataFrame): optionally supply a DataFrame to subset,
                otherwise the class instance's `DF` attribute will be used.
        """
        if "df" in kwargs:
            df = kwargs["df"]
        else:
            df = self.DF.copy()

        if isinstance(teams, str):
            teams = [teams]
        if teams!=None:
            df = df[(df["HomeTeam"].isin(teams)) | (df["AwayTeam"].isin(teams))]

        if isinstance(seasons, str):
            seasons = [seasons]
        if seasons!=None:
            df = df[df["Season"].isin(seasons)]

        if isinstance(divisions, str):
            divisions = [divisions]
        if divisions!=None:
            df = df[df["Div"].isin(divisions)]

        if min_date:
            dt = datetime.datetime.strptime(min_date, "%d/%m/%y")
            df = df[df["datetime"]>=dt]

        if max_date:
            dt = datetime.datetime.strptime(max_date, "%d/%m/%y")
            df = df[df["datetime"]<=dt]

        return df.reset_index(drop=True)

    def previous_x_results(self, x, team, match_date, same_season=True):
        """Get the previous `x` number of results preceding the match date
        for the given team.

        Args:
            x (int): number of matches prior to `match_date` to return.
            team (str): name of team.
            match_date (str): date of the target match in format `DD/MM/YY`.
                The `x` previous matches AND this this match will be returned.
                If an exact match date is not found the closest previous match
                will be used.
            same_season (bool): if True limit results to the same season as the
                target match (i.e. don't return matches from previous season).
        """
        df = self.subset(teams=team, max_date=match_date)
        if same_season:
            season = df.loc[df.index.max(), "Season"]
            df = self.subset(seasons=season, df=df)
        min_ix = df.index.max()-x
        if min_ix<0:
            min_ix = 0
        return df.iloc[min_ix:df.index.max()+1].reset_index(drop=True)

    @property
    def columns(self):
        return sorted(list(self.DF.columns))

    @property
    def not_na_summary(self):
        """Get a summary of the NaN counts by column."""
        new = pd.DataFrame(self.DF.notnull().sum(),
                           columns=["count"]).sort_values(by="count",
                                                          ascending=False)
        new["percent"] = new["count"]/len(self.DF)
        return new

    @property
    def teams(self):
        away = set(self.DF["AwayTeam"])
        home = set(self.DF["HomeTeam"])
        return sorted(list(home|away))

    @property
    def df_full_columns_only(self):
        return self.DF.dropna(axis=1).copy()

    @property
    def betting_companies(self):
        companies = []
        possible = [s[:2] for s in self.DF.columns]
        cols = self.DF.columns
        for c in possible:
            if c+"A" in cols and c+"D" in cols and c+"H" in cols:
                companies.append(c)
        return sorted(list(set(companies)))

    @property
    def betting_columns(self):
        bc = self.betting_companies
        endings = ["A", "H", "D"]
        return [c+e for c in bc for e in endings]

    @property
    def betting_columns_not_na_summary(self):
        not_nas = self.not_na_summary
        betting_cols = self.betting_columns
        return not_nas.loc[betting_cols].sort_values(by="percent",
                                                     ascending=False)

    @_basic_cols
    def team_data(self, team):
        df = self.DF[(self.DF["HomeTeam"]==team) | (self.DF["AwayTeam"]==team)]
        return df.reset_index(drop=True)

    # Pre-filtered DataFrames:
    # ==========================================================================
    @_basic_cols
    def home_wins(self, **kwargs):
        """DataFrame of all games which resulted in a home team win.

        Kwargs:
            Args from `subset`.
        """
        subset = self.subset(**kwargs)
        df = subset[subset["FTR"]=="H"]
        return df.reset_index(drop=True)

    @_basic_cols
    def away_wins(self, **kwargs):
        """DataFrame of all games which resulted in an away team win.

        Kwargs:
            Args from `subset`.
        """
        subset = self.subset(**kwargs)
        df = subset[subset["FTR"]=="A"]
        return df.reset_index(drop=True)

    @_basic_cols
    def draws(self, **kwargs):
        """DataFrame of all games which resulted in a draw.

        Kwargs:
            Args from `subset`.
        """
        subset = self.subset(**kwargs)
        df = subset[subset["FTR"]=="D"]
        return df.reset_index(drop=True)

    def _make_count_df(self, result, **kwargs):
        """Meta function to make team results count DFs.

        Args:
            result (str): either `home_wins`, `away_wins`, `home_losses`,
                `away_losses`, `home_draw`, `away_draw`.

        Kwargs:
            Args from `subset`.
        """
        dfs = {"home_wins":self.home_wins(**kwargs),
               "away_wins":self.away_wins(**kwargs),
               "home_losses":self.away_wins(**kwargs),
               "away_losses":self.home_wins(**kwargs),
               "home_draw":self.draws(**kwargs),
               "away_draw":self.draws(**kwargs)}
        cols = {"home_wins":"HomeTeam", "away_wins":"AwayTeam",
                "home_losses":"HomeTeam", "away_losses":"AwayTeam",
                "home_draw":"HomeTeam", "away_draw":"AwayTeam"}
        d, c = dfs[result], cols[result]
        df = pd.DataFrame(d.groupby([c])[c].count())
        df.rename(columns={c:"count"}, inplace=True)
        df.sort_values(by="count", ascending=False, inplace=True)
        df.reset_index(inplace=True)
        df.rename(columns={c:"Team"}, inplace=True)
        return df

    def team_results(self, **kwargs):
        """DataFrame summarising results by team.

        Kwargs:
            Args from `subset`.
        """
        hw = self._make_count_df("home_wins", **kwargs).rename(columns={"count":"Home_Wins"})
        aw = self._make_count_df("away_wins", **kwargs).rename(columns={"count":"Away_Wins"})
        hl = self._make_count_df("home_losses", **kwargs).rename(columns={"count":"Home_Losses"})
        al = self._make_count_df("away_losses", **kwargs).rename(columns={"count":"Away_Losses"})
        hd = self._make_count_df("home_draw", **kwargs).rename(columns={"count":"Home_Draws"})
        ad = self._make_count_df("away_draw", **kwargs).rename(columns={"count":"Away_Draws"})
        df = pd.merge(hw, aw, left_on="Team", right_on="Team")
        df = pd.merge(df, hl, left_on="Team", right_on="Team")
        df = pd.merge(df, al, left_on="Team", right_on="Team")
        df = pd.merge(df, hd, left_on="Team", right_on="Team")
        df = pd.merge(df, ad, left_on="Team", right_on="Team")

        df["Draws"] = df["Home_Draws"] + df["Away_Draws"]
        df["Wins"] = df["Home_Wins"] + df["Away_Wins"]
        df["Losses"] = df["Home_Losses"] + df["Away_Losses"]
        df["Total_Games"] = df["Wins"] + df["Losses"] + df["Draws"]
        df["Points"] = df["Draws"] + df["Wins"]*3

        df.sort_values(by=["Points", "Wins", "Draws"],
                       ascending=[False, False, False], inplace=True)
        return df.reset_index(drop=True)

    def goal_difference(self, x, team, match_date, same_season=True):
        """Get a team's goal difference summary in the `x` games prior to the
        match date.

        Args:
            x (int): number of matches prior to `match_date` to return.
            team (str): name of team.
            match_date (str): date of the target match in format `DD/MM/YY`.
                The `x` matches prior to this match will be returned (but NOT
                this match). If an exact match date is not found the closest
                previous match will be used.
            same_season (bool): if True limit results to the same season as the
                target match (i.e. don't return matches from previous season).
        """
        df = self.previous_x_results(x=x, team=team, match_date=match_date,
                                     same_season=same_season)
        d = {"scored_home":int(df.loc[df["HomeTeam"]==team, "FTHG"].sum()),
             "conceded_home":int(df.loc[df["HomeTeam"]==team, "FTAG"].sum()),
             "scored_away":int(df.loc[df["AwayTeam"]==team, "FTAG"].sum()),
             "conceded_away":int(df.loc[df["AwayTeam"]==team, "FTHG"].sum())
            }
        d["scored_total"] = d["scored_home"] + d["scored_away"]
        d["conceded_total"] = d["conceded_home"] + d["conceded_away"]
        d["GD_home"] = d["scored_home"] - d["conceded_home"]
        d["GD_away"] = d["scored_away"] - d["conceded_away"]
        d["GD_total"] = d["GD_home"] + d["GD_away"]

        return d
