import numpy as np

DEFAULT_AIRSPEED = 23

def projected_wind(v_E, v_N, airspeed=DEFAULT_AIRSPEED):
    """Finds wind magnitude projected onto 2D ground speed vector, thus the lateral component is undetermined."""
    v_norm = np.sqrt(v_N**2 + v_E**2)
    return airspeed - v_norm

def calc_beta(v_E, v_N):
    """Returns the angle in radians from North"""
    return np.atan2(-v_E, -v_N) + np.pi

def maximum_gust(ts, wind, window_secs=3):
    """Returns the maximum of `window_secs` sliding rectangular window mean on `wind`."""
    assert np.all(np.diff(ts) > 0), "Timestamps are not monotonically increasing"

    # Find the end indices since timestamps are non-uniform
    end_idxs = np.searchsorted(ts, ts + window_secs)
    # Remove all the end indices that are out of bounds
    end_idxs = end_idxs[end_idxs < len(ts)]
    
    max_gust = 0.0
    for begin, end in enumerate(end_idxs):
        max_gust = max(max_gust, np.mean(np.abs(wind[begin:end])))
        
    return max_gust

def exp_wt(c):
    return c

def equal_wt(n):
    return 1/n

def mcclain_wt(prev, alpha=0.1):
    return prev / (1 + prev - alpha)

def learning_wt(n, alpha=1.0):
    # Slow learner, https://www.youtube.com/watch?v=sctfLwN-r5o
    # Slowly decreasing weight
    return 1 / (n**alpha)

def wind_gaussian_wt(diff, decay_rate = 1.0):
    """Gaussian weight for wind estimators.
    Uses `diff` to determine how valid the new observation is. 
    `decay_rate` is a hyperparameter for the width of the Gaussian, 
    the higher the value, the more quickly it decays if diff is large
    """
    return np.exp(-decay_rate*diff**2)
def update_estimator(xhat, x, gamma):
    """Update estimator `xhat` with new observation `x` with weight `gamma`"""
    return (1-gamma) * xhat + gamma * x

def update_wind_angle_estimator(ahat, beta, what, wproj, r1 = 1.0, r2 = 1.0):
    """Update wind angle estimator

    Args:
        ahat:   current wind angle estimation
        beta:   ground speed angle from N
        what:   current wind magnitude estimation
        wproj:  projected wind speed
        r1:     decay rate for angle update weight when flying into the wind
        r2:     decay rate for angle update weight when flying perpendicularly to the wind

    Returns:
        what: new value of estimator
    """
    # FIXME: Angle update must account for the fact that a measurement of 350 degrees is the same as a measurement of -10 degrees
    #   So the estimator should use the one that is closest to the current estimate of ahat. Not fixed due to shortage of time
    #   Additionally, the two possible observation weights, when delta is 0 or n*pi/2 should have a better activation function.
    #   Currently the activation between z1 and z2 is just discrete.
    wdelta = (what - np.abs(wproj))
    z1 = wind_gaussian_wt(wdelta, decay_rate=r1)
    z2 = wind_gaussian_wt(wproj, decay_rate=r2)

    # Pick one method to simplify the estimator update
    if z1 > z2:
        # If projected wind magnitude is close to the existing estimator (z1 > z2), then beta is a good update for alpha
        z = z1
        alpha = beta
    else:
        # If projected wind magnitude is close to zero relative to ahat (z2 > z1), then a perpendicular airflow gives a good update for alpha
        # In the latter case, there are two valid updates, but it is best to choose the sign of the solution closest to the current estimate
        z = z2
        alpha = beta + np.sign(ahat - beta) * np.pi/2
        alpha = np.mod(alpha, np.pi)

    return update_estimator(ahat, alpha, z)

def update_wind_mag_estimator(what, beta, ahat, wproj, ra = 2.0, rw = 1.0, h=1.0):
    """Update wind magnitude estimator

    Args:
        what:   current wind magnitude estimation
        beta:   ground speed angle from N
        ahat:   current wind angle estimation
        wproj:  projected wind speed
        ra:     decay rate for angle update
        rw:     decay rate for wind magnitude update
        h:      time step

    Returns:
        (what, z) (Tuple): new value of estimator and confidence in the observation
    """
    # Wind magnitude is significantly easier, as this is simply weighted by delta, and the projected wind speed is a good update when delta is small
    adelta = beta - ahat
    wupdate = np.abs(wproj/np.cos(adelta))
    wdelta = what - wupdate

    za = wind_gaussian_wt(np.sin(adelta), decay_rate=ra)
    zw = wind_gaussian_wt(wdelta, decay_rate=rw)
    z = za*zw

    return update_estimator(what, wupdate, h*z), z


def get_fw_timeranges(df_status, df_armed):
    """Returns a list of 2-tuples, containing start and end timestamps valid for fixed wing wind analysis"""
    fw_mask = (df_status["vtol_in_rw_mode"] == 0) & (df_status["vtol_in_trans_mode"] == 0)
    fw_ints = fw_mask.astype(np.int8)

    fw_starts = np.diff(fw_ints, prepend=fw_ints[0]) > 0
    fw_stops = np.diff(fw_ints, append=fw_ints[len(fw_ints)-1]) < 0

    ts_starts = df_status["timestamp"][fw_starts].values
    ts_stops = df_status["timestamp"][fw_stops].values

    assert ts_starts.shape[0] == ts_stops.shape[0], f"FW start and stop timestamps not matching with shape: {ts_starts.shape[0]} != {ts_stops.shape[0]}"

    #FIXME: Armed conditionals. I don't know, but drone should always be armed if operational, so am I correct in throwing an error if its not?
    armed_ts = df_armed[df_armed["armed"] == 1]["timestamp"].values
    assert ts_starts[0] > armed_ts[0], f"First FW timestamp is earlier than armed timestamp: {ts_starts[0]} < {armed_ts[0]}"
    assert ts_starts[-1] < armed_ts[-1], f"Last FW timestamp is later than armed timestamp: {ts_starts[-1]} > {armed_ts[-1]}"


    return zip(ts_starts, ts_stops)