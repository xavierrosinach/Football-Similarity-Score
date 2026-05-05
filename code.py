"""
Script para calcular jugadores similares a partir de estadísticas agregadas
de una temporada.

El flujo general es:
1. Leer datos base de jugadores, estadísticas y partidos.
2. Filtrar la temporada actual.
3. Agregar estadísticas por jugador.
4. Generar métricas avanzadas.
5. Limpiar y preparar el dataframe para modelado.
6. Calcular matrices de similaridad por posición general.
7. Obtener los jugadores más similares para un jugador o para todo un equipo.
"""

import os
import zipfile
import warnings
from datetime import datetime
import shutil
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURACIÓN GLOBAL
# ============================================================

warnings.simplefilter(action="ignore", category=PerformanceWarning)

ACT_SEASON = "2526"

ZIP_DATA_PATH = os.path.join(os.getcwd(), "data", "zipped-data.zip")
INPUT_DATA_PATH = os.path.join(os.getcwd(), "data", "input")
OUTPUT_DATA_PATH = os.path.join(os.getcwd(), "data", "output")

os.makedirs(INPUT_DATA_PATH, exist_ok=True)
with zipfile.ZipFile(ZIP_DATA_PATH, 'r') as zip_ref:
    zip_ref.extractall(INPUT_DATA_PATH)

PLAYER_INFO_FILE = os.path.join(INPUT_DATA_PATH, "Player.csv")
PLAYER_STATS_FILE = os.path.join(INPUT_DATA_PATH, "StatsPlayer.csv")
MATCH_INFO_FILE = os.path.join(INPUT_DATA_PATH, "Match.csv")


# ============================================================
# LECTURA DE DATOS GLOBALES
# ============================================================

player_info_df = pd.read_csv(PLAYER_INFO_FILE, sep=";")
player_stats_df = pd.read_csv(PLAYER_STATS_FILE, sep=";")
match_info_df = pd.read_csv(MATCH_INFO_FILE, sep=";")


# ============================================================
# ORDEN FINAL DE COLUMNAS
# ============================================================

list_to_order_df = [
    "Player", "DateBirth", "Height", "PrefFoot", "Position", "Role",
    "MinutesPlayed", "Matches", "Elo", "OppElo", "Score", "OppScore",
    "TouchesPer90", "PassesPer90", "AccuratePassesPer90", "PassAccuracy",
    "PassesPerTouch", "TouchesPerPass", "PossessionLostPer90",
    "PossessionLossRate", "UnsuccessfulTouchesPer90",
    "UnsuccessfulTouchRate", "DispossessedPer90", "DispossessedRate",
    "OwnHalfPassesPer90", "AccurateOwnHalfPassesPer90",
    "OwnHalfPassAccuracy", "OwnHalfPassShare", "OppositionHalfPassesPer90",
    "AccurateOppositionHalfPassesPer90", "OppositionHalfPassAccuracy",
    "OppositionHalfPassShare", "LongBallsPer90", "AccurateLongBallsPer90",
    "LongBallAccuracy", "LongBallShare", "ProgressiveFieldTiltPer90",
    "KeyPassesPer90", "KeyPassesPerPass", "GoalAssistsPer90",
    "AssistConversion", "ExpectedAssistsPer90", "ExpectedAssistsPerKeyPass",
    "ExpectedAssistsPerPass", "CrossesPer90", "AccurateCrossesPer90",
    "CrossAccuracy", "CrossShare", "AccurateCrossShare",
    "BigChancesCreatedPer90", "BigChancesMissedPer90",
    "BigChancesPerKeyPass", "TotalShotsPer90", "ShotsOnTargetPer90",
    "ShotsOffTargetPer90", "BlockedShotsPer90", "GoalsPer90",
    "ExpectedGoalsPer90", "ExpectedGoalsOnTargetPer90", "ShotAccuracy",
    "GoalConversion", "BigChanceMissRate", "ExpectedGoalsPerShot",
    "ExpectedGoalsOnTargetPerShotOnTarget", "GoalsMinusxGPer90",
    "GoalsMinusxGOTPer90", "BlockedShotRate", "ShotsOnTargetRate",
    "ShotsOffTargetRate", "GoalsPerShotOnTarget", "HitWoodworkPer90",
    "HitWoodworkRate", "OffsidesPer90", "OffsideRate", "ShotMeanX",
    "ShotMeanY", "ShotMeanLength", "ShotTotalLenghtPer90",
    "ShotShareZoneBack", "ShotShareZoneCentre", "ShotShareZoneLeft",
    "ShotShareZoneRight", "DuelsWonPer90", "DuelsLostPer90",
    "DuelWinRate", "AerialWonPer90", "AerialLostPer90", "AerialWinRate",
    "ContestsPer90", "ContestsWonPer90", "ContestWinRate",
    "ChallengesLostPer90", "ChallengesLostRate", "TacklesPer90",
    "TacklesWonPer90", "TackleAccuracy", "InterceptionsPer90",
    "InterceptionShare", "ClearancesPer90", "ClearanceShare",
    "OutfielderBlocksPer90", "BlockShare", "BallRecoveriesPer90",
    "BallRecoveryRate", "DefensiveActionsPer90", "DefensiveActionSuccess",
    "LastManTacklesPer90", "LastManTackleRate", "ClearanceOffLinePer90",
    "ClearanceOffLineRate", "ErrorsLeadToShotPer90",
    "ErrorsLeadToShotRate", "ErrorsLeadToGoalPer90",
    "ErrorsLeadToGoalRate", "GoalsConcededPerDefAction", "SavesPer90",
    "SaveRate", "SavedShotsInsideBoxPer90", "SavedShotsInsideBoxRate",
    "CrossesNotClaimedPer90", "CrossesNotClaimedRate", "HighClaimsPer90",
    "HighClaimRate", "PunchesPer90", "PunchRate",
    "KeeperSweeperActionsPer90", "AccurateKeeperSweeperActionsPer90",
    "KeeperSweeperAccuracy", "GoalKicksPer90", "GoalKicksRate",
    "PenaltyFacedPer90", "PenaltySavesPer90", "PenaltySaveRate",
    "PenaltyGoalsConcededPer90", "PenaltyGoalConcededRate",
    "GoalsConcededPer90", "GoalsPreventedPer90",
    "GoalsPreventedDiffPer90", "CleanSheetsPer90", "FoulsPer90",
    "YellowCardsPer90", "RedCardsPer90", "CardsPerFoul",
    "YellowCardsPerFoul", "RedCardsPerFoul", "FoulsPerDefAction",
    "WasFouledPer90", "WasFouledRate", "PenaltyWonPer90",
    "PenaltyWonRate", "PenaltyMissedPer90", "PenaltyMissRate",
    "PenaltyConcededPer90", "PenaltyConcededRate", "OwnGoalsPer90",
    "OwnGoalRate", "CornersWonPer90", "CornersWonRate",
    "CornersLostPer90", "CornersLostRate", "CornersTakenPer90",
    "CornersTakenRate", "PassMeanAngle", "PassMeanLength",
    "PassTotalLengthPer90", "PassMeanX", "PassMeanY", "PassPercZoneBack",
    "PassPercZoneCenter", "PassPercZoneLeft", "PassPercZoneRight",
    "PassShareZoneBack", "PassShareZoneCenter", "PassShareZoneLeft",
    "PassShareZoneRight", "PassPercDirBackward",
    "PassPercDirBackwardLeft", "PassPercDirBackwardRight",
    "PassPercDirForward", "PassPercDirForwardLeft",
    "PassPercDirForwardRight", "PassPercDirLeft", "PassPercDirRight",
    "PassShareDirBackward", "PassShareDirBackwardLeft",
    "PassShareDirBackwardRight", "PassShareDirForward",
    "PassShareDirForwardLeft", "PassShareDirForwardRight",
    "PassShareDirLeft", "PassShareDirRight"
]


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def df_safe_div(num, den, ndigits=4):
    """
    Realiza una división segura entre dos valores o columnas.

    Evita divisiones por cero y valores nulos.
    Si el denominador es 0 o hay valores nulos, devuelve NaN.

    Parameters
    ----------
    num : scalar, pd.Series o array-like
        Numerador.
    den : scalar, pd.Series o array-like
        Denominador.
    ndigits : int
        Número de decimales para redondear.

    Returns
    -------
    np.ndarray
        Resultado redondeado de la división.
    """

    result = np.where(
        (den != 0) & (~pd.isna(num)) & (~pd.isna(den)),
        num / den,
        np.nan
    )

    return np.round(result, ndigits)


# ============================================================
# CREACIÓN DE MÉTRICAS
# ============================================================

def player_new_metrics(player_stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea nuevas métricas avanzadas sobre el dataframe de jugadores.

    La función calcula:
    - Ratios de pase y posesión.
    - Ratios de creación.
    - Métricas de finalización.
    - Métricas defensivas.
    - Métricas de portero.
    - Métricas disciplinarias.
    - Métricas por 90 minutos.

    Parameters
    ----------
    player_stats_df : pd.DataFrame
        Dataframe de estadísticas de jugadores.

    Returns
    -------
    pd.DataFrame
        Dataframe con métricas adicionales.
    """

    df = player_stats_df.copy()

    # Factor para transformar métricas absolutas a métricas por 90 minutos.
    factor_90 = df_safe_div(90, df["MinutesPlayed"])

    # -----------------------------
    # Pase y posesión
    # -----------------------------
    df["PassAccuracy"] = df_safe_div(df["AccuratePasses"], df["Passes"])
    df["OwnHalfPassAccuracy"] = df_safe_div(df["AccurateOwnHalfPasses"], df["OwnHalfPasses"])
    df["OppositionHalfPassAccuracy"] = df_safe_div(df["AccurateOppositionHalfPasses"], df["OppositionHalfPasses"])
    df["LongBallAccuracy"] = df_safe_div(df["AccurateLongBalls"], df["LongBalls"])

    df["OwnHalfPassShare"] = df_safe_div(df["OwnHalfPasses"], df["Passes"])
    df["OppositionHalfPassShare"] = df_safe_div(df["OppositionHalfPasses"], df["Passes"])
    df["LongBallShare"] = df_safe_div(df["LongBalls"], df["Passes"])

    df["ProgressiveFieldTilt"] = df_safe_div(
        df["OppositionHalfPasses"],
        df["OppositionHalfPasses"] + df["OwnHalfPasses"]
    )

    df["PassesPerTouch"] = df_safe_div(df["Passes"], df["Touches"])
    df["TouchesPerPass"] = df_safe_div(df["Touches"], df["Passes"])
    df["PossessionLossRate"] = df_safe_div(df["PossessionLost"], df["Touches"])
    df["UnsuccessfulTouchRate"] = df_safe_div(df["UnsuccessfulTouches"], df["Touches"])
    df["DispossessedRate"] = df_safe_div(df["Dispossessed"], df["Touches"])

    # -----------------------------
    # Creación
    # -----------------------------
    df["KeyPassesPerPass"] = df_safe_div(df["KeyPasses"], df["Passes"])
    df["AssistConversion"] = df_safe_div(df["GoalAssists"], df["KeyPasses"])
    df["ExpectedAssistsPerKeyPass"] = df_safe_div(df["ExpectedAssists"], df["KeyPasses"])
    df["ExpectedAssistsPerPass"] = df_safe_div(df["ExpectedAssists"], df["Passes"])
    df["CrossAccuracy"] = df_safe_div(df["AccurateCrosses"], df["Crosses"])
    df["CrossShare"] = df_safe_div(df["Crosses"], df["Passes"])
    df["AccurateCrossShare"] = df_safe_div(df["AccurateCrosses"], df["Passes"])
    df["BigChancesPerKeyPass"] = df_safe_div(df["BigChancesCreated"], df["KeyPasses"])

    # -----------------------------
    # Precisión de pase por zona
    # -----------------------------
    df["PassPercZoneBack"] = df_safe_div(df["PassAccZoneBack"], df["PassZoneBack"])
    df["PassPercZoneCenter"] = df_safe_div(df["PassAccZoneCenter"], df["PassZoneCenter"])
    df["PassPercZoneLeft"] = df_safe_div(df["PassAccZoneLeft"], df["PassZoneLeft"])
    df["PassPercZoneRight"] = df_safe_div(df["PassAccZoneRight"], df["PassZoneRight"])

    # -----------------------------
    # Distribución de pases por zona
    # -----------------------------
    total_zone_passes = (
        df["PassZoneBack"]
        + df["PassZoneCenter"]
        + df["PassZoneLeft"]
        + df["PassZoneRight"]
    )

    df["PassShareZoneBack"] = df_safe_div(df["PassZoneBack"], total_zone_passes)
    df["PassShareZoneCenter"] = df_safe_div(df["PassZoneCenter"], total_zone_passes)
    df["PassShareZoneLeft"] = df_safe_div(df["PassZoneLeft"], total_zone_passes)
    df["PassShareZoneRight"] = df_safe_div(df["PassZoneRight"], total_zone_passes)

    # -----------------------------
    # Precisión de pase por dirección
    # -----------------------------
    df["PassPercDirBackward"] = df_safe_div(df["PassAccDirBackward"], df["PassDirBackward"])
    df["PassPercDirBackwardLeft"] = df_safe_div(df["PassAccDirBackwardLeft"], df["PassDirBackwardLeft"])
    df["PassPercDirBackwardRight"] = df_safe_div(df["PassAccDirBackwardRight"], df["PassDirBackwardRight"])
    df["PassPercDirForward"] = df_safe_div(df["PassAccDirForward"], df["PassDirForward"])
    df["PassPercDirForwardLeft"] = df_safe_div(df["PassAccDirForwardLeft"], df["PassDirForwardLeft"])
    df["PassPercDirForwardRight"] = df_safe_div(df["PassAccDirForwardRight"], df["PassDirForwardRight"])
    df["PassPercDirLeft"] = df_safe_div(df["PassAccDirLeft"], df["PassDirLeft"])
    df["PassPercDirRight"] = df_safe_div(df["PassAccDirRight"], df["PassDirRight"])

    # -----------------------------
    # Distribución de pase por dirección
    # -----------------------------
    total_dir_passes = (
        df["PassDirBackward"]
        + df["PassDirBackwardLeft"]
        + df["PassDirBackwardRight"]
        + df["PassDirForward"]
        + df["PassDirForwardLeft"]
        + df["PassDirForwardRight"]
        + df["PassDirLeft"]
        + df["PassDirRight"]
    )

    df["PassShareDirBackward"] = df_safe_div(df["PassDirBackward"], total_dir_passes)
    df["PassShareDirBackwardLeft"] = df_safe_div(df["PassDirBackwardLeft"], total_dir_passes)
    df["PassShareDirBackwardRight"] = df_safe_div(df["PassDirBackwardRight"], total_dir_passes)
    df["PassShareDirForward"] = df_safe_div(df["PassDirForward"], total_dir_passes)
    df["PassShareDirForwardLeft"] = df_safe_div(df["PassDirForwardLeft"], total_dir_passes)
    df["PassShareDirForwardRight"] = df_safe_div(df["PassDirForwardRight"], total_dir_passes)
    df["PassShareDirLeft"] = df_safe_div(df["PassDirLeft"], total_dir_passes)
    df["PassShareDirRight"] = df_safe_div(df["PassDirRight"], total_dir_passes)

    # -----------------------------
    # Finalización
    # -----------------------------
    df["ShotAccuracy"] = df_safe_div(df["ShotsOnTarget"], df["TotalShots"])
    df["GoalConversion"] = df_safe_div(df["Goals"], df["TotalShots"])
    df["BigChanceMissRate"] = df_safe_div(df["BigChancesMissed"], df["BigChancesCreated"])
    df["ExpectedGoalsPerShot"] = df_safe_div(df["ExpectedGoals"], df["TotalShots"])
    df["ExpectedGoalsOnTargetPerShotOnTarget"] = df_safe_div(
        df["ExpectedGoalsOnTarget"],
        df["ShotsOnTarget"]
    )

    df["GoalsMinusxG"] = df["Goals"] - df["ExpectedGoals"]
    df["GoalsMinusxGOT"] = df["Goals"] - df["ExpectedGoalsOnTarget"]

    df["BlockedShotRate"] = df_safe_div(df["BlockedShots"], df["TotalShots"])
    df["ShotsOnTargetRate"] = df_safe_div(df["ShotsOnTarget"], df["TotalShots"])
    df["ShotsOffTargetRate"] = df_safe_div(df["ShotsOffTarget"], df["TotalShots"])
    df["GoalsPerShotOnTarget"] = df_safe_div(df["Goals"], df["ShotsOnTarget"])
    df["HitWoodworkRate"] = df_safe_div(df["HitWoodwork"], df["TotalShots"])

    # -----------------------------
    # Distribución de resultados de tiro
    # -----------------------------
    total_shots = df["MissedShots"] + df["ShotsOnPost"] + df["SavedShots"]

    df["MissedShotRate"] = df_safe_div(df["MissedShots"], total_shots)
    df["PostShotRate"] = df_safe_div(df["ShotsOnPost"], total_shots)
    df["SavedShotRate"] = df_safe_div(df["SavedShots"], total_shots)

    # -----------------------------
    # Distribución de tiros por zona
    # -----------------------------
    total_shot_zones = (
        df["ShotZoneBack"]
        + df["ShotZoneCentre"]
        + df["ShotZoneLeft"]
        + df["ShotZoneRight"]
    )

    df["ShotShareZoneBack"] = df_safe_div(df["ShotZoneBack"], total_shot_zones)
    df["ShotShareZoneCentre"] = df_safe_div(df["ShotZoneCentre"], total_shot_zones)
    df["ShotShareZoneLeft"] = df_safe_div(df["ShotZoneLeft"], total_shot_zones)
    df["ShotShareZoneRight"] = df_safe_div(df["ShotZoneRight"], total_shot_zones)

    # -----------------------------
    # Duelos
    # -----------------------------
    df["DuelWinRate"] = df_safe_div(df["DuelsWon"], df["DuelsWon"] + df["DuelsLost"])
    df["AerialWinRate"] = df_safe_div(df["AerialWon"], df["AerialWon"] + df["AerialLost"])
    df["ContestWinRate"] = df_safe_div(df["ContestsWon"], df["Contests"])
    df["ChallengesLostRate"] = df_safe_div(df["ChallengesLost"], df["Contests"])

    # -----------------------------
    # Defensa
    # -----------------------------
    df["DefensiveActions"] = (
        df["Tackles"]
        + df["Interceptions"]
        + df["Clearances"]
        + df["OutfielderBlocks"]
    )

    df["DefensiveActionSuccess"] = df_safe_div(
        df["TacklesWon"] + df["Interceptions"] + df["Clearances"],
        df["DefensiveActions"]
    )

    df["TackleAccuracy"] = df_safe_div(df["TacklesWon"], df["Tackles"])
    df["InterceptionShare"] = df_safe_div(df["Interceptions"], df["DefensiveActions"])
    df["ClearanceShare"] = df_safe_div(df["Clearances"], df["DefensiveActions"])
    df["BlockShare"] = df_safe_div(df["OutfielderBlocks"], df["DefensiveActions"])
    df["BallRecoveryRate"] = df_safe_div(df["BallRecoveries"], df["DefensiveActions"])
    df["LastManTackleRate"] = df_safe_div(df["LastManTackles"], df["Tackles"])
    df["ClearanceOffLineRate"] = df_safe_div(df["ClearanceOffLine"], df["Clearances"])
    df["ErrorsLeadToShotRate"] = df_safe_div(df["ErrorsLeadToShot"], df["DefensiveActions"])
    df["ErrorsLeadToGoalRate"] = df_safe_div(df["ErrorsLeadToGoal"], df["DefensiveActions"])
    df["GoalsConcededPerDefAction"] = df_safe_div(df["GoalsConceded"], df["DefensiveActions"])

    # -----------------------------
    # Portero
    # -----------------------------
    df["SaveRate"] = df_safe_div(df["Saves"], df["Saves"] + df["GoalsConceded"])
    df["SavedShotsInsideBoxRate"] = df_safe_div(df["SavedShotsInsideBox"], df["Saves"])
    df["CrossesNotClaimedRate"] = df_safe_div(
        df["CrossesNotClaimed"],
        df["CrossesNotClaimed"] + df["HighClaims"]
    )

    df["HighClaimRate"] = df_safe_div(
        df["HighClaims"],
        df["HighClaims"] + df["Punches"] + df["CrossesNotClaimed"]
    )

    df["PunchRate"] = df_safe_div(
        df["Punches"],
        df["HighClaims"] + df["Punches"] + df["CrossesNotClaimed"]
    )

    df["KeeperSweeperAccuracy"] = df_safe_div(
        df["AccurateKeeperSweeperActions"],
        df["KeeperSweeperActions"]
    )

    df["PenaltySaveRate"] = df_safe_div(df["PenaltySaves"], df["PenaltyFaced"])
    df["PenaltyGoalConcededRate"] = df_safe_div(df["PenaltyGoalsConceded"], df["PenaltyFaced"])
    df["GoalsPreventedDiff"] = df["GoalsPrevented"] - df["GoalsConceded"]

    # -----------------------------
    # Disciplina
    # -----------------------------
    df["CardsPerFoul"] = df_safe_div(df["YellowCards"] + df["RedCards"], df["Fouls"])
    df["YellowCardsPerFoul"] = df_safe_div(df["YellowCards"], df["Fouls"])
    df["RedCardsPerFoul"] = df_safe_div(df["RedCards"], df["Fouls"])
    df["FoulsPerDefAction"] = df_safe_div(df["Fouls"], df["DefensiveActions"])
    df["WasFouledRate"] = df_safe_div(df["WasFouled"], df["Touches"])
    df["OffsideRate"] = df_safe_div(df["Offsides"], df["TotalShots"])

    # -----------------------------
    # Balones parados
    # -----------------------------
    df["CornersWonRate"] = df_safe_div(df["CornersWon"], df["Touches"])
    df["CornersLostRate"] = df_safe_div(df["CornersLost"], df["DefensiveActions"])
    df["CornersTakenRate"] = df_safe_div(df["CornersTaken"], df["Touches"])
    df["GoalKicksRate"] = df_safe_div(df["GoalKicks"], df["Touches"])

    # -----------------------------
    # Penaltis
    # -----------------------------
    df["PenaltyWonRate"] = df_safe_div(df["PenaltyWon"], df["Touches"])
    df["PenaltyMissRate"] = df_safe_div(df["PenaltyMissed"], df["PenaltyWon"])
    df["PenaltyConcededRate"] = df_safe_div(df["PenaltyConceded"], df["DefensiveActions"])

    # -----------------------------
    # Diferencias
    # -----------------------------
    df["GoalsMinusGoalsConceded"] = df["Goals"] - df["GoalsConceded"]
    df["ShotsMinusGoals"] = df["TotalShots"] - df["Goals"]
    df["OwnGoalRate"] = df_safe_div(df["OwnGoals"], df["GoalsConceded"])

    # -----------------------------
    # Métricas por 90 minutos
    # -----------------------------
    cols_exclude = [
        "Match", "Team", "Player", "ShirtNumber", "Position",
        "PassMeanAngle", "PassMeanLength", "PassMeanX", "PassMeanY",
        "ShotMeanX", "ShotMeanY", "MeanGoalY", "MeanGoalZ",
        "ShotMeanLength", "GoalIniX", "GoalIniY", "GoalFinalY",
        "GoalFinalZ", "GoalMeanLength", "ShotZoneBackLength",
        "ShotZoneCentreLength", "ShotZoneLeftLength",
        "ShotZoneRightLength"
    ]

    per90_cols = df.columns.difference(cols_exclude)

    per90_dict = {
        f"{col}Per90": df[col] * factor_90
        for col in per90_cols
    }

    df = pd.concat([df, pd.DataFrame(per90_dict)], axis=1)

    return df


# ============================================================
# AGREGACIÓN DE DATOS POR JUGADOR
# ============================================================

def player_aggregate_data(player_stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega estadísticas de partido a nivel jugador.

    Primero elimina columnas informativas y métricas de ratio.
    Luego suma o promedia las columnas según corresponda.
    Finalmente recalcula las métricas avanzadas sobre el dataframe agregado.
    """

    player_agg = player_stats_df.copy()

    cols_to_drop = [c for c in ["Match", "Team", "Opponent"] if c in player_agg.columns]
    player_agg = player_agg.drop(columns=cols_to_drop)

    cols_per90 = [c for c in player_agg.columns if c.endswith("Per90")]
    player_agg = player_agg.drop(columns=cols_per90)

    ratio_cols = [
        "Match", "Team", "PassAccuracy", "PassesPerTouch", "TouchesPerPass",
        "PossessionLossRate", "UnsuccessfulTouchRate", "DispossessedRate",
        "OwnHalfPassAccuracy", "OwnHalfPassShare",
        "OppositionHalfPassAccuracy", "OppositionHalfPassShare",
        "LongBallAccuracy", "LongBallShare", "KeyPassesPerPass",
        "AssistConversion", "ExpectedAssistsPerKeyPass",
        "ExpectedAssistsPerPass", "CrossAccuracy", "CrossShare",
        "AccurateCrossShare", "BigChanceMissRate", "BigChancesPerKeyPass",
        "ShotAccuracy", "ShotsOnTargetRate", "ShotsOffTargetRate",
        "BlockedShotRate", "GoalConversion", "GoalsPerShotOnTarget",
        "ExpectedGoalsPerShot", "ExpectedGoalsOnTargetPerShotOnTarget",
        "HitWoodworkRate", "OffsideRate", "DuelWinRate", "AerialWinRate",
        "ContestWinRate", "ChallengesLostRate", "TackleAccuracy",
        "DefensiveActionSuccess", "InterceptionShare", "ClearanceShare",
        "BlockShare", "BallRecoveryRate", "LastManTackleRate",
        "ClearanceOffLineRate", "ErrorsLeadToShotRate",
        "ErrorsLeadToGoalRate", "GoalsConcededPerDefAction", "SaveRate",
        "SavedShotsInsideBoxRate", "CrossesNotClaimedRate", "HighClaimRate",
        "PunchRate", "KeeperSweeperAccuracy", "GoalKicksRate",
        "PenaltySaveRate", "PenaltyGoalConcededRate", "CardsPerFoul",
        "YellowCardsPerFoul", "RedCardsPerFoul", "FoulsPerDefAction",
        "WasFouledRate", "PenaltyWonRate", "PenaltyMissRate",
        "PenaltyConcededRate", "OwnGoalRate", "CornersWonRate",
        "CornersLostRate", "CornersTakenRate", "GoalsMinusxG",
        "GoalsMinusxGOT", "GoalsMinusGoalsConceded", "ShotsMinusGoals",
        "PassPercZoneBack", "PassPercZoneCenter", "PassPercZoneLeft",
        "PassPercZoneRight", "PassPercDirBackward",
        "PassPercDirBackwardLeft", "PassPercDirBackwardRight",
        "PassPercDirForward", "PassPercDirForwardLeft",
        "PassPercDirForwardRight", "PassPercDirLeft", "PassPercDirRight",
        "TotalGoals"
    ]

    ratio_cols = [c for c in ratio_cols if c in player_agg.columns]
    player_agg = player_agg.drop(columns=ratio_cols)

    # Cada fila equivale a una aparición del jugador en un partido.
    player_agg["Matches"] = 1

    def mode_or_nan(s):
        """
        Devuelve la moda de una serie o NA si está vacía.

        Nota:
        Esta función estaba en el notebook original, pero no se usa
        en la lógica posterior. Se mantiene para no alterar estructura.
        """
        s = s.dropna()

        if s.empty:
            return pd.NA

        return s.mode().iloc[0]

    agg_dict = {}

    for col in player_agg.columns:
        if col == "Player":
            continue
        elif col in [
            "PassMeanAngle", "PassMeanLength", "PassMeanX", "PassMeanY",
            "ShotMeanX", "ShotMeanY", "MeanGoalY", "MeanGoalZ",
            "ShotMeanLength", "GoalIniX", "Elo", "OppElo",
            "GoalIniY", "GoalFinalY", "GoalFinalZ", "GoalMeanLength",
            "ShotZoneBackLength", "ShotZoneCentreLength",
            "ShotZoneLeftLength", "ShotZoneRightLength"
        ]:
            agg_dict[col] = "mean"
        else:
            agg_dict[col] = "sum"

    player_agg_df = player_agg.groupby("Player", as_index=False).agg(agg_dict)

    final_df = player_new_metrics(player_stats_df=player_agg_df)

    return final_df


# ============================================================
# LIMPIEZA FINAL
# ============================================================

def final_df_cleaning(season_player_df: pd.DataFrame, min_minutes: int = 500) -> pd.DataFrame:
    """
    Limpia el dataframe final agregado y lo prepara para calcular similaridades.

    Pasos:
    - Filtra jugadores por minutos mínimos.
    - Elimina jugadores sin fecha de nacimiento, altura o pie preferido.
    - Ordena columnas.
    - Calcula edad.
    - Crea posición general.
    - Imputa valores numéricos por media de posición.
    - Aplica one-hot encoding.
    """

    df = season_player_df.copy()

    df = df[df["MinutesPlayed"] >= min_minutes]
    df = df.dropna(subset=["DateBirth", "Height", "PrefFoot"])

    df = df[list_to_order_df]

    df["DateBirth"] = pd.to_datetime(df["DateBirth"], format="%d/%m/%Y", errors="coerce")

    today = pd.Timestamp.today()
    df.insert(2, "Age", (today - df["DateBirth"]).dt.days // 365)

    df = df.drop(columns="DateBirth")

    df["Role"] = df["Position"] + " - " + df["Role"]

    df.insert(
        5,
        "GeneralPos",
        np.where(
            df["Position"].isin(["LW", "ST", "RW"]),
            "Attacker",
            np.where(
                df["Position"].isin(["LB", "CB", "RB"]),
                "Defender",
                np.where(
                    df["Position"].isin(["DM", "AM"]),
                    "Midfielder",
                    "Goalkeeper"
                )
            )
        )
    )

    num_cols = df.select_dtypes(include=[np.number]).columns

    df[num_cols] = df.groupby("Position")[num_cols].transform(
        lambda x: x.fillna(x.mean())
    )

    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != "Player"]

    df_encoded = pd.get_dummies(
        df,
        columns=cat_cols,
        drop_first=False,
        dtype=int
    )

    return df_encoded.set_index("Player")


# ============================================================
# MATRIZ DE SIMILARIDAD
# ============================================================

def compute_similarity_matrix(df: pd.DataFrame, metric: str = "cosine", scale: bool = True, weights: dict = None) -> pd.DataFrame:
    """
    Calcula una matriz de similaridad entre jugadores.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe numérico con jugadores como índice.
    metric : str
        Métrica de similaridad. Valores soportados:
        - "cosine"
        - "euclidean"
    scale : bool
        Si True, aplica StandardScaler antes de calcular la similaridad.
    weights : dict
        Diccionario opcional de pesos por columna.

    Returns
    -------
    pd.DataFrame
        Matriz cuadrada de similaridad jugador-jugador.
    """

    X = df.copy()

    if weights is not None:
        for col, weight in weights.items():
            if col in X.columns:
                X[col] = X[col] * weight

    if scale:
        scaler = StandardScaler()
        X_values = scaler.fit_transform(X)
    else:
        X_values = X.values

    if metric == "cosine":
        sim_matrix = cosine_similarity(X_values)
    elif metric == "euclidean":
        dist_matrix = euclidean_distances(X_values)
        sim_matrix = 1 / (1 + dist_matrix)
    else:
        raise ValueError("metric debe ser 'cosine' o 'euclidean'.")

    sim_df = pd.DataFrame(
        sim_matrix,
        index=df.index,
        columns=df.index
    )

    return sim_df


# ============================================================
# PROCESAMIENTO GLOBAL DE DATOS
# ============================================================

def all_data_proc() -> list:
    """
    Prepara todos los datos necesarios para calcular similaridades.

    Devuelve cuatro matrices de similaridad:
    1. Porteros.
    2. Defensas.
    3. Centrocampistas.
    4. Atacantes.
    """

    season_matches = match_info_df[
        match_info_df["Season"].astype(str) == str(ACT_SEASON)
    ].reset_index(drop=True)

    season_player_stats = player_stats_df[
        player_stats_df["Match"].isin(season_matches["ID"].unique().tolist())
    ].reset_index(drop=True)

    season_player_info = player_info_df[
        player_info_df["ID"].isin(player_stats_df["Player"].unique().tolist())
    ].reset_index(drop=True)

    season_player_info = season_player_info[
        ["ID", "DateBirth", "Height", "PrefFoot", "Position", "Role"]
    ]

    season_player_stats = season_player_stats.drop(
        columns=["ShirtNumber", "Position"]
    )

    season_matches = season_matches[
        [
            "ID", "HomeTeam", "AwayTeam", "HomeScore", "AwayScore",
            "HomeElo", "AwayElo"
        ]
    ]

    home_season_matches = season_matches.rename(
        columns={
            "ID": "Match",
            "HomeTeam": "Team",
            "AwayTeam": "Opponent",
            "HomeScore": "Score",
            "AwayScore": "OppScore",
            "HomeElo": "Elo",
            "AwayElo": "OppElo"
        }
    )

    away_season_matches = season_matches.rename(
        columns={
            "ID": "Match",
            "AwayTeam": "Team",
            "HomeTeam": "Opponent",
            "AwayScore": "Score",
            "HomeScore": "OppScore",
            "AwayElo": "Elo",
            "HomeElo": "OppElo"
        }
    )

    season_matches = pd.concat(
        [home_season_matches, away_season_matches]
    ).sort_values(
        by=["Match", "Team"]
    ).reset_index(drop=True)

    season_matches_stats = season_player_stats.merge(
        season_matches,
        on=["Match", "Team"],
        how="left"
    )

    season_player_agg_stats = player_aggregate_data(
        player_stats_df=season_matches_stats
    )

    season_player_info = season_player_info.rename(columns={"ID": "Player"})

    season_player_df = season_player_info.merge(
        season_player_agg_stats,
        how="inner",
        on="Player"
    )

    final_player_df = final_df_cleaning(
        season_player_df=season_player_df
    )

    # -----------------------------
    # Porteros
    # -----------------------------
    only_goalkeepers = final_player_df[
        final_player_df["GeneralPos_Goalkeeper"] == 1
    ]

    only_goalkeepers = only_goalkeepers.loc[
        :,
        only_goalkeepers.nunique(dropna=False) > 1
    ]

    only_goalkeepers = only_goalkeepers.dropna(axis=1)

    sim_matrix_goalkeepers = compute_similarity_matrix(
        df=only_goalkeepers
    )

    sim_matrix_goalkeepers.index.name = None
    sim_matrix_goalkeepers.columns.name = None

    # -----------------------------
    # Defensas
    # -----------------------------
    only_defenders = final_player_df[
        final_player_df["GeneralPos_Defender"] == 1
    ]

    only_defenders = only_defenders.loc[
        :,
        only_defenders.nunique(dropna=False) > 1
    ]

    only_defenders = only_defenders.dropna(axis=1)

    sim_matrix_defenders = compute_similarity_matrix(
        df=only_defenders
    )

    sim_matrix_defenders.index.name = None
    sim_matrix_defenders.columns.name = None

    # -----------------------------
    # Centrocampistas
    # -----------------------------
    only_midfielders = final_player_df[
        final_player_df["GeneralPos_Midfielder"] == 1
    ]

    only_midfielders = only_midfielders.loc[
        :,
        only_midfielders.nunique(dropna=False) > 1
    ]

    only_midfielders = only_midfielders.dropna(axis=1)

    sim_matrix_midfielders = compute_similarity_matrix(
        df=only_midfielders
    )

    sim_matrix_midfielders.index.name = None
    sim_matrix_midfielders.columns.name = None

    # -----------------------------
    # Atacantes
    # -----------------------------
    only_forward = final_player_df[
        final_player_df["GeneralPos_Attacker"] == 1
    ]

    only_forward = only_forward.loc[
        :,
        only_forward.nunique(dropna=False) > 1
    ]

    only_forward = only_forward.dropna(axis=1)

    sim_matrix_attackers = compute_similarity_matrix(
        df=only_forward
    )

    sim_matrix_attackers.index.name = None
    sim_matrix_attackers.columns.name = None

    return [
        sim_matrix_goalkeepers,
        sim_matrix_defenders,
        sim_matrix_midfielders,
        sim_matrix_attackers
    ]


# ============================================================
# SIMILARES DE UN JUGADOR
# ============================================================

def get_player_similar_players(player_id: str, player_info_df: pd.DataFrame, list_sim_matrix: list, top_n: int = 50, ignore_team: bool = False) -> pd.DataFrame:
    """
    Obtiene el top N de jugadores más similares a un jugador concreto.

    Parameters
    ----------
    player_id : str
        ID del jugador objetivo.
    player_info_df : pd.DataFrame
        Dataframe global de información de jugadores.
    list_sim_matrix : list
        Lista de matrices de similaridad por posición.
    top_n : int
        Número de jugadores similares a devolver.
    ignore_team : bool
        Si True, excluye jugadores del mismo equipo.

    Returns
    -------
    pd.DataFrame
        Dataframe con jugadores similares y su puntuación.
    """

    if player_id in list_sim_matrix[0].columns:
        df = list_sim_matrix[0].copy()
    elif player_id in list_sim_matrix[1].columns:
        df = list_sim_matrix[1].copy()
    elif player_id in list_sim_matrix[2].columns:
        df = list_sim_matrix[2].copy()
    elif player_id in list_sim_matrix[3].columns:
        df = list_sim_matrix[3].copy()
    else:
        return pd.DataFrame()

    similar_players = df[[player_id]]

    similar_players = similar_players.sort_values(
        by=player_id,
        ascending=False
    )

    similar_players = similar_players.reset_index().rename(
        columns={
            "index": "Player",
            player_id: "Similarity"
        }
    )

    min_val = similar_players["Similarity"].min()
    max_val = similar_players["Similarity"].max()

    similar_players["Similarity"] = (
        (similar_players["Similarity"] - min_val)
        / (max_val - min_val)
    )

    similar_players = similar_players[
        similar_players["Player"] != player_id
    ]

    if ignore_team:
        player_team = player_info_df[
            player_info_df["ID"] == player_id
        ]["Team"].iloc[0]

        list_players = player_info_df[
            player_info_df["Team"] == player_team
        ]["ID"].unique().tolist()

        similar_players = similar_players[
            ~similar_players["Player"].isin(list_players)
        ]

    return similar_players.head(top_n)


# ============================================================
# SIMILARES DE TODOS LOS JUGADORES DE UN EQUIPO
# ============================================================

def get_team_similar_players(team_id: str, ignore_team: bool = False, top_n: int = 5) -> pd.DataFrame:
    """
    Obtiene los jugadores más similares para cada jugador de un equipo.

    Devuelve un dataframe pivoteado:
    - Una fila por jugador del club.
    - Columnas Top1, Top2, ..., TopN.
    """

    list_data = all_data_proc()

    player_id_dict = dict(
        zip(player_info_df["ID"], player_info_df["Name"])
    )

    list_to_concat = []

    for player_id in player_info_df.loc[
        player_info_df["Team"] == team_id,
        "ID"
    ].unique():

        top_similar = get_player_similar_players(
            player_id=player_id,
            player_info_df=player_info_df,
            list_sim_matrix=list_data,
            ignore_team=ignore_team,
            top_n=top_n
        )

        if not top_similar.empty:
            top_similar["ClubPlayer"] = player_id
            list_to_concat.append(top_similar)

    result = pd.concat(list_to_concat, ignore_index=True)

    result["ClubPlayer"] = result["ClubPlayer"].map(player_id_dict)
    result["SimilarPlayer"] = result["Player"].map(player_id_dict)
    result["Similarity"] = round(result["Similarity"] * 100, 2)

    result = result[
        ["ClubPlayer", "SimilarPlayer", "Similarity"]
    ]

    df_out = result.sort_values(
        ["ClubPlayer", "Similarity"],
        ascending=[True, False]
    )

    df_out["rank"] = df_out.groupby("ClubPlayer").cumcount() + 1

    df_out["SimilarText"] = (
        df_out["SimilarPlayer"]
        + " ("
        + df_out["Similarity"].round(2).astype(str)
        + "%)"
    )

    df_pivot = df_out.pivot(
        index="ClubPlayer",
        columns="rank",
        values="SimilarText"
    )

    df_pivot.columns = [
        f"Top{int(col)}"
        for col in df_pivot.columns
    ]

    df_pivot = df_pivot.reset_index()

    return df_pivot


# ============================================================
# EJECUCIÓN DEL SCRIPT
# ============================================================

def main():

    best_teams = {
        "4DKRD": "bayern_munchen",
        "U02BO": "arsenal",
        "M148I": "barcelona",
        "JVKVM": "real_madrid",
        "VRU3P": "paris_saint_germain",
        "V6IU4": "manchester_united",
        "WWENZ": "manchester_city",
        "30355": "liverpool",
        "6YTSQ": "inter",
        "KLERD": "borussia_dortmund",
        "17WKF": "aston_villa",
        "J1TMZ": "newcastle",
        "EGCJZ": "atletico_madrid",
        "2MT4Z": "brentford",
        "F731X": "chelsea",
        "ZYOY5": "milan",
        "A3NMP": "sporting_cp",
        "QFK5P": "porto",
        "GYHDA": "brighton",
        "NP921": "everton"
    }

    top_n = 5

    for team_id, file_name in best_teams.items():
        print(f"Searching the top {top_n} most similar players for team {file_name}")
        similar_players_df = get_team_similar_players(team_id=team_id, top_n=top_n)

        similar_players_df.to_csv(
            os.path.join(OUTPUT_DATA_PATH, f"{file_name}.csv"),
            index=False,
            sep=";"
        )

    shutil.rmtree(INPUT_DATA_PATH)


if __name__ == "__main__":
    main()