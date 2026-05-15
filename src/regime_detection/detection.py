from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .constants import REGIME_STATE_FEATURES


def run_detection_model(
    normalized: pd.DataFrame,
    gap_window: int = 60,
    cusum_threshold: float = 0.08,
    change_point_window: int = 30,
    change_point_z_threshold: float = 3.0,
    change_point_method: str = "pelt",
    change_point_lookback: int = 180,
    change_point_min_size: int = 20,
    change_point_penalty: float | None = None,
    change_point_confirmation_days: int = 5,
    change_point_cooldown_days: int = 20,
    regime_state_count: int = 4,
    regime_state_model: str = "hmm",
    hmm_max_iter: int = 25,
    hmm_transition_prior: float = 2.0,
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for _, group in normalized.sort_values(["strategy_name", "date"]).groupby("strategy_name"):
        df = group.copy()
        df["rolling_gap_mean_60d"] = df["gap"].rolling(gap_window, min_periods=10).mean()
        df["rolling_gap_vol_60d"] = df["gap"].rolling(gap_window, min_periods=10).std()
        df["gap_z"] = (df["gap"] - df["rolling_gap_mean_60d"]) / df["rolling_gap_vol_60d"].replace(0.0, np.nan)
        df["gap_z"] = df["gap_z"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df["cusum_value"], df["cusum_signal"] = _negative_cusum(df["residual_return"], cusum_threshold)
        df["gap_z_alert"] = df["gap_z"] < -2.0
        df["drawdown_alert"] = df["rolling_drawdown"] < -0.1
        df["volatility_alert"] = (df["rolling_vol_60d"] > 0.0) & (df["rolling_vol_20d"] > (df["rolling_vol_60d"] * 1.5))
        df["negative_residual_10d"] = df["residual_return"].rolling(10, min_periods=5).mean() < 0.0
        df["below_normal_10d"] = (df["gap_pct"] < 0.0).rolling(10, min_periods=5).sum() >= 7
        df["sharpe_deterioration_alert"] = (df["rolling_sharpe_20d"] < 0.0) & (
            df["rolling_sharpe_20d"] < df["rolling_sharpe_60d"]
        )
        change_points = _change_point_detection(
            df,
            method=change_point_method,
            rolling_window=change_point_window,
            threshold=change_point_z_threshold,
            lookback=change_point_lookback,
            min_size=change_point_min_size,
            penalty=change_point_penalty,
            confirmation_days=change_point_confirmation_days,
            cooldown_days=change_point_cooldown_days,
        )
        df["change_point_score"] = change_points["change_point_score"]
        df["change_point_signal"] = change_points["change_point_signal"]
        df["change_point_method"] = change_points["change_point_method"]
        df["change_point_age_days"] = change_points["change_point_age_days"]
        df["change_point_count"] = change_points["change_point_count"]
        df["change_point_date"] = change_points["change_point_date"]
        regime = _hidden_state_regimes(
            df,
            regime_state_count,
            model_type=regime_state_model,
            max_iter=hmm_max_iter,
            transition_prior=hmm_transition_prior,
        )
        df["hmm_regime"] = regime["hmm_regime"]
        df["hmm_regime_probability"] = regime["hmm_regime_probability"]
        df["hmm_regime_state"] = regime["hmm_regime_state"]
        df["hmm_regime_method"] = regime["hmm_regime_method"]
        df["regime_alert_level"] = df.apply(_alert_level, axis=1)
        outputs.append(df)
    return pd.concat(outputs, ignore_index=True)


def _negative_cusum(residuals: pd.Series, threshold: float) -> tuple[pd.Series, pd.Series]:
    values: list[float] = []
    signals: list[int] = []
    running = 0.0
    for value in residuals.fillna(0.0):
        running = min(0.0, running + float(value))
        signal = int(abs(running) >= threshold)
        values.append(running)
        signals.append(signal)
        if signal:
            running = 0.0
    return pd.Series(values, index=residuals.index), pd.Series(signals, index=residuals.index)


def _alert_level(row: pd.Series) -> str:
    stress_count = int(row.get("drawdown_alert", False)) + int(row.get("volatility_alert", False))
    stress_count += int(row.get("negative_residual_10d", False)) + int(row.get("below_normal_10d", False))
    stress_count += int(row.get("sharpe_deterioration_alert", False))

    change_point = int(row.get("change_point_signal", 0)) == 1
    hidden_stress = str(row.get("hmm_regime", "")) in {"crisis", "stress"}

    if row["gap_z"] < -3.0 and (row["cusum_signal"] == 1 or stress_count >= 2 or change_point or hidden_stress):
        return "red"
    if row["gap_z"] < -2.5 or row["cusum_signal"] == 1 or change_point or (row["gap_z"] < -2.0 and stress_count >= 2):
        return "orange"
    if row["gap_z"] < -2.0 or stress_count >= 3 or hidden_stress:
        return "yellow"
    return "green"


def _rolling_change_point_score(df: pd.DataFrame, window: int) -> pd.Series:
    series = pd.to_numeric(df["residual_return"], errors="coerce").fillna(0.0)
    left_mean = series.shift(window).rolling(window, min_periods=max(10, window // 2)).mean()
    right_mean = series.rolling(window, min_periods=max(10, window // 2)).mean()
    left_vol = series.shift(window).rolling(window, min_periods=max(10, window // 2)).std()
    right_vol = series.rolling(window, min_periods=max(10, window // 2)).std()
    pooled_vol = np.sqrt((left_vol**2 + right_vol**2) / 2.0).replace(0.0, np.nan)
    mean_shift = ((right_mean - left_mean).abs() / pooled_vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    vol_ratio = (right_vol / left_vol.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    vol_shift = np.log(vol_ratio).abs().fillna(0.0)
    return (mean_shift + vol_shift).fillna(0.0)


def _change_point_detection(
    df: pd.DataFrame,
    method: str,
    rolling_window: int,
    threshold: float,
    lookback: int,
    min_size: int,
    penalty: float | None,
    confirmation_days: int,
    cooldown_days: int,
) -> pd.DataFrame:
    requested_method = str(method or "pelt").lower()
    if requested_method == "rolling":
        score = _rolling_change_point_score(df, rolling_window)
        return pd.DataFrame(
            {
                "change_point_score": score,
                "change_point_signal": (score >= threshold).astype(int),
                "change_point_method": "rolling_mean_vol_shift",
                "change_point_age_days": np.nan,
                "change_point_count": 0,
                "change_point_date": pd.NaT,
            },
            index=df.index,
        )
    if requested_method in {"bayesian", "bayes"}:
        return _online_bayesian_change_points(
            dates=pd.to_datetime(df["date"]),
            residuals=pd.to_numeric(df["residual_return"], errors="coerce"),
            threshold=threshold,
            lookback=lookback,
            min_size=min_size,
            confirmation_days=confirmation_days,
            cooldown_days=cooldown_days,
            index=df.index,
        )
    return _online_pelt_change_points(
        dates=pd.to_datetime(df["date"]),
        residuals=pd.to_numeric(df["residual_return"], errors="coerce"),
        threshold=threshold,
        lookback=lookback,
        min_size=min_size,
        penalty=penalty,
        confirmation_days=confirmation_days,
        cooldown_days=cooldown_days,
        index=df.index,
    )


def _online_bayesian_change_points(
    dates: pd.Series,
    residuals: pd.Series,
    threshold: float,
    lookback: int,
    min_size: int,
    confirmation_days: int,
    cooldown_days: int,
    index: pd.Index,
) -> pd.DataFrame:
    values = residuals.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    lookback = max(int(lookback), 2 * int(min_size), 20)
    min_size = max(int(min_size), 5)
    confirmation_days = max(int(confirmation_days), 0)
    cooldown_days = max(int(cooldown_days), 0)
    threshold = float(threshold)

    scores = np.zeros(len(values), dtype=float)
    signals = np.zeros(len(values), dtype=int)
    ages = np.full(len(values), np.nan)
    counts = np.zeros(len(values), dtype=int)
    change_dates = pd.Series(pd.NaT, index=index, dtype="datetime64[ns]")
    last_signal_idx = -cooldown_days - 1

    for end_idx in range(len(values)):
        start_idx = max(0, end_idx - lookback + 1)
        window = values[start_idx : end_idx + 1]
        if len(window) < 2 * min_size:
            continue

        standardized = _standardize_for_change_points(window)
        split, score = _latest_bayesian_mean_shift(standardized, min_size=min_size)
        if split is None:
            continue
        counts[end_idx] = 1

        age = len(standardized) - split
        global_breakpoint_idx = start_idx + split
        scores[end_idx] = score
        ages[end_idx] = float(age)
        change_dates.iloc[end_idx] = dates.iloc[global_breakpoint_idx]
        confirmed_age = age - min_size
        if 0 <= confirmed_age <= confirmation_days and score >= threshold and end_idx - last_signal_idx > cooldown_days:
            signals[end_idx] = 1
            last_signal_idx = end_idx

    return pd.DataFrame(
        {
            "change_point_score": scores,
            "change_point_signal": signals,
            "change_point_method": "online_bayesian_mean_shift",
            "change_point_age_days": ages,
            "change_point_count": counts,
            "change_point_date": change_dates,
        },
        index=index,
    )


def _online_pelt_change_points(
    dates: pd.Series,
    residuals: pd.Series,
    threshold: float,
    lookback: int,
    min_size: int,
    penalty: float | None,
    confirmation_days: int,
    cooldown_days: int,
    index: pd.Index,
) -> pd.DataFrame:
    values = residuals.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    lookback = max(int(lookback), 2 * int(min_size), 20)
    min_size = max(int(min_size), 5)
    confirmation_days = max(int(confirmation_days), 0)
    cooldown_days = max(int(cooldown_days), 0)
    threshold = float(threshold)

    scores = np.zeros(len(values), dtype=float)
    signals = np.zeros(len(values), dtype=int)
    ages = np.full(len(values), np.nan)
    counts = np.zeros(len(values), dtype=int)
    change_dates = pd.Series(pd.NaT, index=index, dtype="datetime64[ns]")
    last_signal_idx = -cooldown_days - 1

    # Score each row from a trailing window only, so historical backtests match live scoring.
    for end_idx in range(len(values)):
        start_idx = max(0, end_idx - lookback + 1)
        window = values[start_idx : end_idx + 1]
        if len(window) < 2 * min_size:
            continue

        standardized = _standardize_for_change_points(window)
        resolved_penalty = float(penalty) if penalty is not None else max(threshold**2, np.log(len(standardized)) * 2.0)
        latest_breakpoint, score = _latest_penalized_mean_shift(
            standardized,
            min_size=min_size,
            penalty=resolved_penalty,
        )
        if latest_breakpoint is None:
            continue
        counts[end_idx] = 1

        age = len(standardized) - latest_breakpoint
        global_breakpoint_idx = start_idx + latest_breakpoint

        scores[end_idx] = score
        ages[end_idx] = float(age)
        change_dates.iloc[end_idx] = dates.iloc[global_breakpoint_idx]
        confirmed_age = age - min_size
        if 0 <= confirmed_age <= confirmation_days and score >= threshold and end_idx - last_signal_idx > cooldown_days:
            signals[end_idx] = 1
            last_signal_idx = end_idx

    return pd.DataFrame(
        {
            "change_point_score": scores,
            "change_point_signal": signals,
            "change_point_method": "online_pelt_mean_shift",
            "change_point_age_days": ages,
            "change_point_count": counts,
            "change_point_date": change_dates,
        },
        index=index,
    )


def _standardize_for_change_points(values: np.ndarray) -> np.ndarray:
    centered = values - np.nanmedian(values)
    scale = np.nanstd(centered)
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.zeros_like(centered, dtype=float)
    return centered / scale


def _latest_penalized_mean_shift(values: np.ndarray, min_size: int, penalty: float) -> tuple[int | None, float]:
    n = len(values)
    if n < 2 * min_size or np.nanstd(values) <= 1e-12:
        return None, 0.0

    prefix_sum = np.concatenate(([0.0], np.cumsum(values)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(values**2)))
    candidates = np.arange(min_size, n - min_size + 1)
    full_cost = _segment_sse(prefix_sum, prefix_sq, 0, n)
    left_cost = _segment_sse_vectorized(prefix_sum, prefix_sq, 0, candidates)
    right_cost = _segment_sse_vectorized(prefix_sum, prefix_sq, candidates, n)
    gains = full_cost - left_cost - right_cost
    accepted = gains >= penalty
    if not bool(accepted.any()):
        return None, 0.0

    accepted_candidates = candidates[accepted]
    accepted_gains = gains[accepted]
    best_position = int(accepted_gains.argmax())
    best_split = int(accepted_candidates[best_position])
    score = float(np.sqrt(max(float(accepted_gains[best_position]), 0.0)))
    return best_split, score


def _latest_bayesian_mean_shift(values: np.ndarray, min_size: int) -> tuple[int | None, float]:
    n = len(values)
    if n < 2 * min_size or np.nanstd(values) <= 1e-12:
        return None, 0.0

    full_log_evidence = _normal_inverse_gamma_log_evidence(values)
    best_split: int | None = None
    best_log_bayes_factor = 0.0
    for split in range(min_size, n - min_size + 1):
        log_bayes_factor = (
            _normal_inverse_gamma_log_evidence(values[:split])
            + _normal_inverse_gamma_log_evidence(values[split:])
            - full_log_evidence
        )
        if log_bayes_factor > best_log_bayes_factor:
            best_log_bayes_factor = float(log_bayes_factor)
            best_split = split

    if best_split is None:
        return None, 0.0
    return best_split, float(np.sqrt(max(2.0 * best_log_bayes_factor, 0.0)))


def _normal_inverse_gamma_log_evidence(values: np.ndarray) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    mean = float(np.mean(values))
    centered_sse = float(((values - mean) ** 2).sum())
    mu_0 = 0.0
    kappa_0 = 1.0
    alpha_0 = 2.0
    beta_0 = 1.0
    kappa_n = kappa_0 + n
    alpha_n = alpha_0 + n / 2.0
    beta_n = beta_0 + 0.5 * centered_sse + (kappa_0 * n * (mean - mu_0) ** 2) / (2.0 * kappa_n)
    return (
        math.lgamma(alpha_n)
        - math.lgamma(alpha_0)
        + alpha_0 * math.log(beta_0)
        - alpha_n * math.log(max(beta_n, 1e-12))
        + 0.5 * (math.log(kappa_0) - math.log(kappa_n))
        - n / 2.0 * math.log(math.pi)
    )


def _segment_sse(prefix_sum: np.ndarray, prefix_sq: np.ndarray, start: int, end: int) -> float:
    count = end - start
    if count <= 0:
        return 0.0
    total = prefix_sum[end] - prefix_sum[start]
    squared = prefix_sq[end] - prefix_sq[start]
    return float(max(squared - (total * total / count), 0.0))


def _segment_sse_vectorized(prefix_sum: np.ndarray, prefix_sq: np.ndarray, start: int | np.ndarray, end: int | np.ndarray) -> np.ndarray:
    count = np.asarray(end) - np.asarray(start)
    total = prefix_sum[end] - prefix_sum[start]
    squared = prefix_sq[end] - prefix_sq[start]
    return np.maximum(squared - (total * total / count), 0.0)


def _hidden_state_regimes(
    df: pd.DataFrame,
    requested_states: int,
    model_type: str = "hmm",
    max_iter: int = 25,
    transition_prior: float = 2.0,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    state_count = max(2, min(int(requested_states), 5, len(df) // 25 if len(df) >= 50 else 1))
    if state_count < 2:
        out["hmm_regime"] = "normal"
        out["hmm_regime_probability"] = 1.0
        out["hmm_regime_state"] = 0
        out["hmm_regime_method"] = "insufficient_rows"
        return out

    features = _regime_state_feature_frame(df)
    if (features.std() == 0.0).all():
        out["hmm_regime"] = "normal"
        out["hmm_regime_probability"] = 1.0
        out["hmm_regime_state"] = 0
        out["hmm_regime_method"] = "constant_features"
        return out

    if str(model_type or "hmm").lower() in {"hmm", "gaussian_hmm", "true_hmm"}:
        hmm = _gaussian_hmm_regimes(df, features, state_count, max_iter=max_iter, transition_prior=transition_prior)
        if hmm is not None:
            return hmm

    return _gaussian_mixture_regimes(df, features, state_count)


def _regime_state_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    for column in REGIME_STATE_FEATURES:
        if column not in features.columns:
            features[column] = 0.0
    return features[REGIME_STATE_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _gaussian_mixture_regimes(df: pd.DataFrame, features: pd.DataFrame, state_count: int) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    scaled = StandardScaler().fit_transform(features)
    model = GaussianMixture(n_components=state_count, covariance_type="diag", random_state=7, reg_covar=1e-5)
    model.fit(scaled)
    probabilities = model.predict_proba(scaled)
    raw_states = probabilities.argmax(axis=1)
    smoothed_states = _smooth_states(raw_states)
    state_labels = _label_states(df, smoothed_states)
    smoothed_probs = probabilities[np.arange(len(df)), smoothed_states]

    out["hmm_regime"] = [state_labels[state] for state in smoothed_states]
    out["hmm_regime_probability"] = smoothed_probs
    out["hmm_regime_state"] = smoothed_states
    out["hmm_regime_method"] = "gaussian_mixture_smoothed"
    return out


def _gaussian_hmm_regimes(
    df: pd.DataFrame,
    features: pd.DataFrame,
    state_count: int,
    max_iter: int,
    transition_prior: float,
) -> pd.DataFrame | None:
    scaled = StandardScaler().fit_transform(features)
    if len(scaled) < state_count * 10:
        return None

    try:
        init_model = GaussianMixture(n_components=state_count, covariance_type="diag", random_state=7, reg_covar=1e-5)
        init_model.fit(scaled)
        responsibilities = init_model.predict_proba(scaled)
    except Exception:
        return None

    start_prob, transition, means, variances = _initialize_hmm_parameters(
        scaled,
        responsibilities,
        transition_prior=max(float(transition_prior), 0.0),
    )
    log_likelihood = -np.inf
    for _ in range(max(int(max_iter), 1)):
        log_emission = _gaussian_log_emissions(scaled, means, variances)
        current_log_likelihood, log_alpha = _forward_log_prob(log_emission, start_prob, transition)
        log_beta = _backward_log_prob(log_emission, transition)
        gamma = np.exp(log_alpha + log_beta - current_log_likelihood)
        xi_sum = _expected_transition_counts(log_alpha, log_beta, log_emission, transition, current_log_likelihood)

        start_prob = _normalize_vector(gamma[0] + 1e-3)
        transition = _normalize_rows(xi_sum + 1e-3 + np.eye(state_count) * max(float(transition_prior), 0.0))
        weights = gamma.sum(axis=0).clip(min=1e-8)
        means = (gamma.T @ scaled) / weights[:, None]
        variances = np.einsum("nk,nkp->kp", gamma, (scaled[:, None, :] - means[None, :, :]) ** 2) / weights[:, None]
        variances = np.clip(variances, 1e-5, None)

        if abs(current_log_likelihood - log_likelihood) < 1e-4:
            log_likelihood = current_log_likelihood
            break
        log_likelihood = current_log_likelihood

    log_emission = _gaussian_log_emissions(scaled, means, variances)
    log_likelihood, log_alpha = _forward_log_prob(log_emission, start_prob, transition)
    log_beta = _backward_log_prob(log_emission, transition)
    probabilities = np.exp(log_alpha + log_beta - log_likelihood)
    states = _viterbi_path(log_emission, start_prob, transition)
    state_labels = _label_states(df, states)

    out = pd.DataFrame(index=df.index)
    out["hmm_regime"] = [state_labels[state] for state in states]
    out["hmm_regime_probability"] = probabilities[np.arange(len(df)), states]
    out["hmm_regime_state"] = states
    out["hmm_regime_method"] = "gaussian_hmm_em"
    return out


def _initialize_hmm_parameters(
    x: np.ndarray,
    responsibilities: np.ndarray,
    transition_prior: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state_count = responsibilities.shape[1]
    states = responsibilities.argmax(axis=1)
    start_prob = _normalize_vector(responsibilities[0] + 1e-3)
    transition = np.full((state_count, state_count), 1e-3)
    transition += np.eye(state_count) * transition_prior
    for left, right in zip(states[:-1], states[1:]):
        transition[int(left), int(right)] += 1.0
    transition = _normalize_rows(transition)

    weights = responsibilities.sum(axis=0).clip(min=1e-8)
    means = (responsibilities.T @ x) / weights[:, None]
    variances = np.einsum("nk,nkp->kp", responsibilities, (x[:, None, :] - means[None, :, :]) ** 2) / weights[:, None]
    variances = np.clip(variances, 1e-5, None)
    return start_prob, transition, means, variances


def _gaussian_log_emissions(x: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - means[None, :, :]
    return -0.5 * (np.log(2.0 * np.pi * variances).sum(axis=1)[None, :] + ((diff**2) / variances[None, :, :]).sum(axis=2))


def _forward_log_prob(log_emission: np.ndarray, start_prob: np.ndarray, transition: np.ndarray) -> tuple[float, np.ndarray]:
    log_start = np.log(start_prob.clip(min=1e-12))
    log_transition = np.log(transition.clip(min=1e-12))
    log_alpha = np.empty_like(log_emission)
    log_alpha[0] = log_start + log_emission[0]
    for idx in range(1, len(log_emission)):
        log_alpha[idx] = log_emission[idx] + _logsumexp(log_alpha[idx - 1][:, None] + log_transition, axis=0)
    return float(_logsumexp(log_alpha[-1], axis=0)), log_alpha


def _backward_log_prob(log_emission: np.ndarray, transition: np.ndarray) -> np.ndarray:
    log_transition = np.log(transition.clip(min=1e-12))
    log_beta = np.zeros_like(log_emission)
    for idx in range(len(log_emission) - 2, -1, -1):
        log_beta[idx] = _logsumexp(log_transition + log_emission[idx + 1][None, :] + log_beta[idx + 1][None, :], axis=1)
    return log_beta


def _expected_transition_counts(
    log_alpha: np.ndarray,
    log_beta: np.ndarray,
    log_emission: np.ndarray,
    transition: np.ndarray,
    log_likelihood: float,
) -> np.ndarray:
    state_count = transition.shape[0]
    log_transition = np.log(transition.clip(min=1e-12))
    xi_sum = np.zeros((state_count, state_count), dtype=float)
    for idx in range(len(log_emission) - 1):
        log_xi = (
            log_alpha[idx][:, None]
            + log_transition
            + log_emission[idx + 1][None, :]
            + log_beta[idx + 1][None, :]
            - log_likelihood
        )
        xi_sum += np.exp(log_xi)
    return xi_sum


def _viterbi_path(log_emission: np.ndarray, start_prob: np.ndarray, transition: np.ndarray) -> np.ndarray:
    log_start = np.log(start_prob.clip(min=1e-12))
    log_transition = np.log(transition.clip(min=1e-12))
    state_count = transition.shape[0]
    scores = np.empty_like(log_emission)
    backpointers = np.zeros((len(log_emission), state_count), dtype=int)
    scores[0] = log_start + log_emission[0]
    for idx in range(1, len(log_emission)):
        candidates = scores[idx - 1][:, None] + log_transition
        backpointers[idx] = candidates.argmax(axis=0)
        scores[idx] = log_emission[idx] + candidates.max(axis=0)

    states = np.zeros(len(log_emission), dtype=int)
    states[-1] = int(scores[-1].argmax())
    for idx in range(len(log_emission) - 2, -1, -1):
        states[idx] = backpointers[idx + 1, states[idx + 1]]
    return states


def _normalize_vector(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(values))
    if total <= 0.0 or not np.isfinite(total):
        return np.full(len(values), 1.0 / len(values))
    return values / total


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    totals = values.sum(axis=1, keepdims=True)
    fallback = np.full_like(values, 1.0 / values.shape[1])
    return np.divide(values, totals, out=fallback, where=totals > 0.0)


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    max_value = np.max(values, axis=axis, keepdims=True)
    stable = np.exp(values - max_value).sum(axis=axis, keepdims=True)
    result = max_value + np.log(stable.clip(min=1e-300))
    return np.squeeze(result, axis=axis)


def _smooth_states(states: np.ndarray, min_run: int = 3) -> np.ndarray:
    smoothed = states.copy()
    run_start = 0
    while run_start < len(smoothed):
        run_end = run_start + 1
        while run_end < len(smoothed) and smoothed[run_end] == smoothed[run_start]:
            run_end += 1
        if run_end - run_start < min_run:
            replacement = smoothed[run_start - 1] if run_start > 0 else (smoothed[run_end] if run_end < len(smoothed) else smoothed[run_start])
            smoothed[run_start:run_end] = replacement
        run_start = run_end
    return smoothed


def _label_states(df: pd.DataFrame, states: np.ndarray) -> dict[int, str]:
    frame = df.copy()
    frame["state"] = states
    stress = (
        -frame["gap_z"].fillna(0.0)
        - frame["rolling_drawdown"].fillna(0.0) * 5.0
        + frame["rolling_vol_60d"].fillna(0.0) * 50.0
        - frame["rolling_sharpe_60d"].fillna(0.0) * 0.2
        + frame.get("cds_z", pd.Series(0.0, index=frame.index)).fillna(0.0)
    )
    frame["stress_score"] = stress
    ordered_states = frame.groupby("state")["stress_score"].mean().sort_values().index.tolist()
    labels_by_rank = ["normal", "transition", "stress", "crisis", "crisis"]
    return {state: labels_by_rank[min(rank, len(labels_by_rank) - 1)] for rank, state in enumerate(ordered_states)}
