"""
Author: Tim Hermans (tim-hermans@hotmail.com).
"""
from datetime import timedelta

import numpy as np
import pandas as pd


STARTDATE = pd.to_datetime('01-01-2021')

SIGNAL_LABELS = {
    'P_sol_dem': 'Zon->Verbruik',
    'P_sol_bat': 'Zon->Batterij',
    'P_sol_net': 'Zon->Net',
    'P_bat_dem': 'Bat->Verbruik',
    'P_net_dem': 'Net->Verbruik',
    'P_net_bat': 'Net->Batterij',
    'pQ_bat': 'Batterijlading',
    'P_demand': 'Totaal Verbruik',
    'P_solar': 'Zonne-energie',
}


class EnergySimulator(object):
    def __init__(
            self, P_demand, P_solar, P_net_max,
            Q_bat_max, P_bat_charge_max, P_bat_discharge_max, efficiency_bat,
            pQ_bat_init=0, dt=1, day_start_hour=6, day_end_hour=18):
        """
        All values are in W or Wh.
        """
        assert len(P_demand) == len(P_solar)

        # Get time vector.
        time = np.arange(0, 365 * 24, dt)
        assert len(time) == len(P_demand)

        # Set input attributes.
        self.P_demand = P_demand
        self.P_solar = P_solar
        self.P_net_max = P_net_max
        self.Q_bat_max = Q_bat_max
        self.P_bat_charge_max = P_bat_charge_max
        self.P_bat_discharge_max = P_bat_discharge_max
        self.efficiency_bat = efficiency_bat
        self.pQ_bat_init = pQ_bat_init
        self.dt = dt
        self.day_start_hour = day_start_hour
        self.day_end_hour = day_end_hour
        self.time = time
        pass

    def do_timestep(self, t, p_dem, p_sol, pQ_bat):
        # Some logicals.
        bat_is_full = pQ_bat > 99.99
        hour_of_the_day = t % 24
        is_night = hour_of_the_day < self.day_start_hour or hour_of_the_day >= self.day_end_hour

        # Maximum rate for unloading battery at this point (averaged).
        # Can only unload the amount of kWh left divided by the dt.
        Q_bat = pQ_bat / 100 * self.Q_bat_max
        p_bat_max = min(self.P_bat_discharge_max, Q_bat / self.dt)

        # Same for charging rate.
        P_bat_charge_max = min(self.P_bat_charge_max, (self.Q_bat_max - Q_bat) / self.dt)

        # Determine the flow of energy (power).
        p_sol_dem = min(p_sol, p_dem)
        p_sol_bat = min(p_sol - p_sol_dem, P_bat_charge_max) if not bat_is_full else 0
        p_sol_net = p_sol - p_sol_dem - p_sol_bat

        if is_night:
            # First, let net handle it.
            p_net_dem = min(p_dem - p_sol_dem, self.P_net_max)

            # If there is a shortage of energy, use battery.
            p_bat_dem = min(p_dem - p_sol_dem - p_net_dem, p_bat_max) * self.efficiency_bat / 100

            # If there is net left, load battery if not full.
            p_net_bat = min(P_bat_charge_max - p_sol_bat, self.P_net_max - p_net_dem) if not bat_is_full else 0

        else:
            # First unload battery.
            p_bat_dem = min(p_dem - p_sol_dem, p_bat_max) * self.efficiency_bat / 100

            # If not enough, use net.
            p_net_dem = min(p_dem - p_sol_dem - p_bat_dem, self.P_net_max)

            # During the day, we won't charge the battery using the net.
            p_net_bat = 0

        # Recompute how much we need from the net.
        p_net_dem = p_dem - p_sol_dem - p_bat_dem
        if p_net_dem > self.P_net_max:
            pass
            # raise ValueError('Net power exceeds maximum: fuse break.')

        # Compute how much the battery will charge during the current timestep.
        eta = self.efficiency_bat
        dQ_bat = ((p_sol_bat + p_net_bat) * eta / 100 - p_bat_dem * 100 / eta) * self.dt
        pQ_bat_next = min(pQ_bat + dQ_bat / self.Q_bat_max * 100, 100)

        # Collect all in list.
        output = [
            p_sol_dem,
            p_sol_bat,
            p_sol_net,
            p_bat_dem,
            p_net_dem,
            p_net_bat,
            pQ_bat_next
        ]

        if any(np.array(output) < -1e-10):
            raise AssertionError

        return output

    def get_labels(self):
        """
        Get the labels for the numbers in the list returned by self.do_timestep().
        """
        labels = [
            'P_sol_dem',
            'P_sol_bat',
            'P_sol_net',
            'P_bat_dem',
            'P_net_dem',
            'P_net_bat',
            'pQ_bat',
        ]
        return labels

    def run(self):
        time = self.time
        output = []
        pQ_bat_current = self.pQ_bat_init
        for t, p_dem, p_sol in zip(time, self.P_demand, self.P_solar):
            out_i = self.do_timestep(t, p_dem, p_sol, pQ_bat_current)
            pQ_bat_next = out_i[-1]
            out_i[-1] = pQ_bat_current
            output.append(out_i)

            # When we go to the next timestep, the next becomes the current.
            pQ_bat_current = pQ_bat_next

        output = np.array(output).T  # (n_outputs, n_time)
        output = dict(zip(self.get_labels(), output))
        output['time'] = time

        return output


def convert_time_unit(t, unit):
    """
    Input in hours.
    """
    # Startdate
    if unit == 'Vermogen [kW]':
        # Per hour.
        t = STARTDATE + pd.to_timedelta(t, unit='hours')
        xlabel = 'Tijd'
    elif unit == 'Energie/dag [kWh]':
        # Per day.
        t = pd.date_range(STARTDATE, STARTDATE + timedelta(days=365), freq='d')
        xlabel = 'Dag'
    elif unit == 'Energie/week [kWh]':
        t = np.arange(52) + 1
        xlabel = 'Week'
    elif unit == 'Energie/maand [kWh]':
        t = np.arange(12) + 1
        xlabel = 'Maand'
    else:
        raise NotImplementedError(f'Unknown input for input-unit: {unit}.')
    return t, xlabel


def convert_bat_charge(p, unit, dt=1):
    """
    Input in %.
    """
    if unit == 'Vermogen [kW]':
        # Keep as is (per hour).
        y = p
    elif unit == 'Energie/dag [kWh]':
        # Average days.
        y = np.mean(segment_1d(p, segment_length=int(24/dt)), axis=-1)
    elif unit == 'Energie/week [kWh]':
        # Average weeks.
        y = np.mean(segment_1d(p, segment_length=int(7*24/dt)), axis=-1)
    elif unit == 'Energie/maand [kWh]':
        # Average months.
        t_end_month = pd.date_range(STARTDATE, STARTDATE + timedelta(days=365 - 1), freq='m')
        lastdate = STARTDATE - timedelta(days=1)
        start_idx = 0
        yy = []
        for d in t_end_month:
            hours = int((d - lastdate).total_seconds()/3600)
            stop_idx = start_idx + int(hours / dt)
            yy.append(np.mean(p[start_idx: stop_idx]))
            lastdate = d
            start_idx = stop_idx
        y = np.asarray(yy)
    else:
        raise NotImplementedError(f'Unknown input for input-unit: {unit}.')

    return y


def convert_power_unit(p, unit, dt=1):
    """
    Input in W.
    """
    # To kW.
    p = p / 1000

    if unit == 'Vermogen [kW]':
        # Already in kW.
        y = p
    elif unit == 'Energie/dag [kWh]':
        # Integrate days.
        y = np.sum(segment_1d(p, segment_length=int(24/dt)), axis=-1) * dt
    elif unit == 'Energie/week [kWh]':
        # Integrate weeks.
        y = np.sum(segment_1d(p, segment_length=int(7*24/dt)), axis=-1) * dt
    elif unit == 'Energie/maand [kWh]':
        # Integrate months.
        t_end_month = pd.date_range(STARTDATE, STARTDATE + timedelta(days=365 - 1), freq='m')
        lastdate = STARTDATE - timedelta(days=1)
        start_idx = 0
        yy = []
        for d in t_end_month:
            hours = int((d - lastdate).total_seconds()/3600)
            stop_idx = start_idx + int(hours / dt)
            yy.append(np.sum(p[start_idx: stop_idx])*dt)
            lastdate = d
            start_idx = stop_idx
        y = np.asarray(yy)
    else:
        raise NotImplementedError(f'Unknown input for input-unit: {unit}.')

    return y


def get_demand(total_energy, season_ratio, day_ratio, dt):
    """
    Model power demand over time (one year).

    Args:
        total_energy (float): Total energy consumption in a year (kWh).
        season_ratio (float): Ratio of peak to minimum energy consumption (on monthly basis).
        day_ratio (float): Ratio of peak to minimum energy consumption (on daily basis).
        dt (float): Timestep (hours).

    Returns:
        t (np.ndarray): time vector (in hours).
        P_demand (np.ndarray): array with the power demand in Watts per unit of time (dt) for one year,
            starting at Januray at midnight.
    """
    # Hours in a year.
    total_hours = 365*24

    # Average power.
    power_avg = total_energy*1000/total_hours  # kWh -> W.

    # Time vector.
    t = np.arange(0, total_hours, dt)

    # Add seasonality.
    low_point_hours = t[int(len(t) * 0.6)]  # Bit later than the middle of the year (could be more specific).
    w = 2*np.pi/total_hours  # Period of a year
    phi = 1.5*np.pi - w*low_point_hours
    amplitude = power_avg * (season_ratio - 1) / (1 + season_ratio)
    monthly_fluctuations = amplitude * np.sin(w*t + phi)

    # Add daily fluctuations.
    low_point_hours = 0  # starting at midnight.
    w = 2*np.pi/24  # Period of a day
    phi = 1.5*np.pi - w*low_point_hours
    amplitude = power_avg * (day_ratio - 1) / (1 + day_ratio)
    daily_fluctuations = amplitude * np.sin(w*t + phi)

    # Scale/modulate the daily fluctuations by the monthly fluctuations.
    daily_fluctuations *= (power_avg + monthly_fluctuations)/power_avg

    # Total estimated power.
    P_demand = power_avg + monthly_fluctuations + daily_fluctuations

    # Check if power demand over one year equals the energt consumption.
    total_energy_effective = np.sum(P_demand)*dt/1000  # [kWh]
    assert np.abs(total_energy_effective - total_energy)/total_energy < 1e-8

    return t, P_demand


def get_solar_power(Wp, season_ratio, dt):
    """
    Model solar power generation over time (one year).

    Args:
        Wp (float): Total Watt peak of all solar panels combined.
        season_ratio (float): Ratio of the maximum solar energy over the minimum daily solar energy in a year.
        dt (float): Timestep (hours).

    Returns:
        t (np.ndarray): time vector (in hours).
        P_solar (np.array): array with the generated solar power in Watts per unit of time (dt) for one year,
            starting at Januray at midnight.
    """
    # Hours in a year.
    total_hours = 365 * 24

    # Time vector.
    t = np.arange(0, total_hours, dt)

    # Add seasonality.
    low_point_hours = t[int(len(t) * 0)]  # Start of the year.
    w = 2 * np.pi / total_hours  # Period of a year
    phi = 1.5 * np.pi - w * low_point_hours
    amplitude = 0.5*Wp*(1 - 1/season_ratio)
    monthly_fluctuations = amplitude * np.sin(w * t + phi)

    # Add daily fluctuations.
    low_point_hours = 0  # starting at midnight.
    w = 2 * np.pi / 24  # Period of a day
    phi = 1.5 * np.pi - w * low_point_hours
    daily_fluctuations = np.clip(np.sin(w * t + phi), 0, 1)

    # Total solar power is max Wp and is modulated each day between 0 and 1.
    P_solar = daily_fluctuations*(monthly_fluctuations + Wp - max(monthly_fluctuations))

    # Check.
    assert abs(max(P_solar) - Wp) < 1e8

    return t, P_solar


def segment_1d(x, segment_length):
    assert x.ndim == 1
    segment_length = int(segment_length)
    n_segs = len(x) // segment_length
    x_seg = np.reshape(x[:n_segs*segment_length], (n_segs, segment_length))
    return x_seg


def get_datetime_array(startdate, enddate):
    sdate = pd.to_datetime(startdate)
    edate = pd.to_datetime(enddate)
    return pd.date_range(sdate, edate - timedelta(days=1), freq='d')
