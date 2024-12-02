from functions import *

from os import path, makedirs
import pandas as pd
from sys import argv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["axes.formatter.limits"] = (-99, 99) # Remove scientific notation offset on axes
mpl.rcParams["axes.formatter.useoffset"] = False

DATA_FOLDER = Path('case_flight_logs')
GREEN = "\u001B[32m"
RESET = "\u001B[0m"

# Hyperparameters
RWM = 1.0 # Mean wind update - magnitude pessimism factor
RWA = 3.0 # Mean wind update - angle pessimism factor
RA1 = 1.0 # Wind angle update - delta = 0 pessimism factor
RA2 = 1.0 # Wind angle update - delta = n*pi/2 pessimism factor

if not path.exists("plots"):
    makedirs("plots")

file_prefixes = ["case_circular_flight_", "case_realistic_flight_"]

prefix = file_prefixes[0]
ahat0 = np.deg2rad(270)
what0 = 7

if len(argv) >= 2:
    choice = int(argv[1])
    if choice == 1:
        # just the default prefix defined above
        pass
    elif choice == 2:
        prefix = file_prefixes[1]
        ahat0 = np.deg2rad(225)
        what0 = 8
        
print(f"\n{GREEN}Description:{RESET}")
print(f"Selected prefix: {prefix}")
print(f"Airspeed: {DEFAULT_AIRSPEED}")
print(f"Forecasted Wind Direction: {np.rad2deg(ahat0)}")
print(f"Forecasted Wind Speed: {what0}")

df_armed = pd.read_csv(DATA_FOLDER / (prefix + "actuator_armed_0.csv"))
df_dist = pd.read_csv(DATA_FOLDER / (prefix + "distance_sensor_0.csv"))
df_gps = pd.read_csv(DATA_FOLDER / (prefix + "vehicle_gps_position_0.csv"))
df_status = pd.read_csv(DATA_FOLDER / (prefix + "vtol_vehicle_status_0.csv"))

# Plot the flight status parameters
fig, axes = plt.subplots(3, 1, sharex=True)
axes[0].plot(df_status["timestamp"] * 1e-6, df_status["vtol_in_rw_mode"])
axes[0].set_title("Rotary Wing mode")
axes[1].plot(df_status["timestamp"] * 1e-6, df_status["vtol_in_trans_mode"])
axes[1].set_title("Transition mode")
axes[2].plot(df_armed["timestamp"] * 1e-6, df_armed["armed"])
axes[2].set_title("Armed")
axes[2].set_xlabel("Time since startup [s]")
plt.tight_layout()
plt.savefig("plots" / Path(f"{prefix}status.png"))
plt.clf()

trs = get_fw_timeranges(df_status, df_armed)

for tr in trs:
    suffix = f"_{int(tr[0]*1e-6)}-{int(tr[1]*1e-6)}.png"

    # Raw data
    data = df_gps[(df_gps["timestamp"] > tr[0]) & (df_gps["timestamp"] < tr[1])]
    ts = data["timestamp"].values * 1e-6

    dts = np.diff(ts)
    dts = np.insert(dts, 0, dts[0])

    betas = calc_beta(data["vel_e_m_s"], data["vel_n_m_s"])
    proj_winds = projected_wind(data["vel_n_m_s"], data["vel_e_m_s"])
    max_proj_gust = maximum_gust(ts, proj_winds)

    # Plot raw data
    fig, axes = plt.subplots(3, 1, sharex=True)
    axes[0].plot(ts, data["vel_n_m_s"], label="N")
    axes[0].plot(ts, data["vel_e_m_s"], label="E")
    axes[0].set_title(r"Ground velocity, $\vec v$")
    axes[0].set_ylabel("Speed [m/s]")
    axes[0].legend()

    axes[1].plot(ts, np.rad2deg(betas))
    axes[1].set_title(r"Ground velocity angle, $\beta$")
    axes[1].set_ylim(0, 360)
    axes[1].set_ylabel(r"Angle [$^\circ$]")

    axes[2].plot(ts, proj_winds)
    axes[2].set_title(r"Head/Projected wind, $w_p$")
    axes[2].set_ylabel("Speed [m/s]")
    axes[2].set_xlabel("Time [s]")

    plt.tight_layout()
    plt.savefig("plots" / Path(f"{prefix}raw_data{suffix}"))
    plt.clf()

    # Run estimators
    whats = np.zeros(len(data)+1)
    wconf = np.zeros(len(data)+1)
    whats[0] = what0
    wconf[0] = 1.0
    ahats = np.zeros(len(data)+1)
    ahats[0] = ahat0
    for idx in range(len(data)):
        n = idx + 2
        vel_E = data.iloc[idx]['vel_e_m_s']
        vel_N = data.iloc[idx]['vel_n_m_s']

        beta = calc_beta(vel_E, vel_N)
        whats[idx+1], wconf[idx+1] = update_wind_mag_estimator(whats[idx], beta, ahats[idx], proj_winds.iloc[idx], RWA, RWM, dts[idx])
        # ahats[idx+1] = update_wind_angle_estimator(ahats[idx], beta, whats[idx], proj_winds.iloc[idx], RA1, RA2)
        ahats[idx+1] = ahat0

    max_est_gust = maximum_gust(ts, whats[1:])

    # Plot estimators
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(ts, whats[1:])
    axes[0].set_title(r"Estimated Wind Magnitude, $\hat w$")
    axes[0].set_ylabel("Speed [m/s]")
    # axes[1].plot(ts, np.rad2deg(ahats[1:]))
    # axes[1].set_title(r"Estimated Wind Angle, $\hat \alpha$")
    # axes[1].set_ylabel(r"Angle from N [$^\circ$]")
    axes[1].plot(ts, wconf[1:])
    axes[1].set_title(r"Wind magnitude update confidence, $z$")
    axes[1].set_ylabel(r"Weight")
    axes[1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig("plots" / Path(f"{prefix}estimators{suffix}"))
    plt.clf()

    # Plot frequencies, not sure what to use frequencies for yet, but could be cool
    ft = np.fft.fft(proj_winds)
    mean_dt = np.mean(np.diff(ts))
    plt.plot(np.fft.fftfreq(proj_winds.shape[0], mean_dt), np.log10(np.abs(ft)))
    plt.ylabel("Log10 Amplitude")
    plt.xlabel("Frequency [Hz]")

    plt.savefig("plots" / Path(f"{prefix}freqs{suffix}"))
    plt.clf()
    
    print(f"\n{GREEN}Time interval (seconds since bootup): {tr[0]*1e-6} - {tr[1]*1e-6}{RESET}")
    print(f"Max 3-second gust of head/projected wind: {max_proj_gust}")
    print(f"Max 3-second gust of estimated wind field: {max_est_gust}")
    print(f"Mean of head/projected wind: {np.mean(np.abs(proj_winds))}")
    print(f"Mean of estimated wind field: {np.mean(whats[1:])}")
    print("Saved figures in 'plots' folder")