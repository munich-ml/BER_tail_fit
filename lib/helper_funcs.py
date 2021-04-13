# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 15:59:00 2019

@author: hsteffens
"""

import json, os, time, warnings
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from copy import deepcopy
from enum import Enum
from scipy import special
from scipy.optimize import curve_fit


# Enum constants
class FILE_ID(Enum):
    JITTER =      {"type":"Jitter",       "version":"2021-04-10"}
    _SEP = "."

    @staticmethod
    # use method to receive file id. e.g: FILE_ID.get_id(FILE_ID.CONFIG)
    def get_id(fid):
        return fid.value["type"] + FILE_ID._SEP.value + fid.value["version"]

    @staticmethod
    def get_sep():
        return FILE_ID._SEP.value


class JsonEnc(json.JSONEncoder):
    """
    Extends the standard JSONEncoder to support additional datatypes.
    
    Keywords strings as dict keys are used to identify instances of the 
    additional types.
    
    Additional datatype  | keyword
    ---------------------|------------
    pandas DataFrame     | @DataFrame
    pandas Series        | @Series
    numpy array          | @np.array
    datetime.datetime    | @datetime
    datetime.timedelta   | @timedelta
    
    Of course, the regular JSON datatypes are supported, too:
        int, float, str, bool, None, list, (tuple), dict
        
    Example usage:
        # Encode data object to json_str
        json_str = json.dumps(data, cls=JsonEnc)
        
        # Decode json_str to a data object
        data_copy = json.loads(json_str, cls=JsonDec)
        
    """
    def default(self, obj):
        if isinstance(obj, pd.DataFrame):
            return {"@DataFrame": {"columns": list(obj.columns),
                                   "index": list(obj.index),
                                   "data": obj.values.tolist()}}
        
        if isinstance(obj, pd.Series):
            return {"@Series": {"name": obj.name,
                                "index": list(obj.index),
                                "data": obj.values.tolist()}}
        
        if isinstance(obj, np.ndarray):
            return {"@np.array": obj.tolist()}
        
        if isinstance(obj, dt.datetime):
            return {"@datetime": obj.strftime('%Y-%m-%d %H:%M:%S.%f')}

        if isinstance(obj, dt.timedelta):
            return {"@timedelta": obj.total_seconds()}

        return json.JSONEncoder.default(self, obj)


class JsonDec(json.JSONDecoder):
    """
    Extends the standard JSONDecoder to support additional datatypes.
    
    Additional types are recognized by dict key keywords, which are injected 
    by the JsonEnc.
    
    Additional datatype  | keyword
    ---------------------|------------
    pandas DataFrame     | @DataFrame
    pandas Series        | @Series
    numpy array          | @np.array
    datetime.datetime    | @datetime
    datetime.timedelta   | @timedelta
    
    Of course, the regular JSON datatypes are supported, too:
        int, float, str, bool, None, list, (tuple), dict
        
    Example usage:
        # Encode data object to json_str
        json_str = json.dumps(data, cls=JsonEnc)
        
        # Decode json_str to a data object
        data_copy = json.loads(json_str, cls=JsonDec)
        
    """
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=JsonDec.custom_hook, *args, **kwargs)
    
    @staticmethod
    def custom_hook(dct):
        if len(dct) == 1:  # add. datatypes are coded in dict of len=1
            if "@np.array" in dct:
                return np.array(dct["@np.array"])
            
            if "@DataFrame" in dct:
                return pd.DataFrame(data=dct["@DataFrame"]["data"],
                                    columns=dct["@DataFrame"]["columns"],
                                    index=dct["@DataFrame"]["index"])
            
            if "@Series" in dct:
                return pd.Series(data=dct["@Series"]["data"],
                                 name=dct["@Series"]["name"],
                                 index=dct["@Series"]["index"])
            
            if "@datetime" in dct:
                return dt.datetime.strptime(dct["@datetime"],
                                            '%Y-%m-%d %H:%M:%S.%f')
            
            if "@timedelta" in dct:
                return dt.timedelta(seconds=dct["@timedelta"])
            
        return dct


class JitterEstimator():
    @staticmethod
    def jitter_func_inv(ber, sigma, mu):
        """
        Jitter model function based on scipy inverse complemetary error function 
        input <np.array> ber: bit error ratio data from the HTester
        input <float> sigma: Width of the gaussian distribution. 
        input <float> mu: Offset of the gaussian distribution. 
        return <np.array> horz_offs: sample points within the unit interval
        scipy inverse complemetary error function doc:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.erfcinv.html
        """
        return special.erfcinv(4*ber) * np.sqrt(2) * sigma + mu

    @staticmethod
    def jitter_func(x, sigma, mu):
        """
        Jitter model function based on scipy complemetary error function 
        input <np.array> x: sample points within the unit interval  (-0.5 ... 0.5)
        input <float> sigma: Width of the gaussian distribution. 
        input <float> mu: Offset of the gaussian distribution. 
        return <np.array> BER: bit error ratio
        scipy inverse complemetary error function doc:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.erfc.html
        """
        return special.erfc((x - mu) / (np.sqrt(2)*sigma)) / 4
    
    
    def __init__(self, d):
        """
        JitterEstimator class

        :input <dict> d, having at least two items:
        'setup': {'chs': [8, 9], 'horStepCnt': 32, 'verStepSize': 0, 'verScale': 2, 'targetBER': 1e-06
        'eyescan': {8: DataFrame8, 9: DataFrama9}
        """
        self.d = d


    def __repr__(self):
        s = "JitterEstimator\nsetup:"
        for key, value in self.d["setup"].items():
            if "BER" in key or "linerate" in key:
                s += "\n- {}: {}".format(key, value)
        
        if "jitter" in self.d:
            jtable = pd.DataFrame(self.d["jitter"])
            if len(jtable):
                s += "\njitter fit table:\n" + str(jtable)
            else:
                s += "\nEmpty jitter fit table"

        return s


    def fit(self, specifiedBER=None, thresholdBER=None):
        """
        Apply Gaussian tail fit to internal eyescan data (self.d["eyescan"])
        :input <float> specifiedBER: BER (bit error ratio) to which the TJ (total jitter) is estimated, typically 1e-12
        :input <float> thresholdBER: BER values below thresholdBER are used for fitting, typically 1e-3
        """
        if specifiedBER is not None:
            self.d["setup"]["BERs"] = specifiedBER 
        
        if thresholdBER is not None:
            self.d["setup"]["BERt"] = thresholdBER 
        
        self.d["fit"] = {}
        self.d["jitter"] = {}
        
        for ch, ber in self.d["eyescan"].items():
            fit = {"success": True}
            for side in ("left", "right"):
                if side == "left":
                    mask = ber.index < 0
                    initial = ( 0.02, -0.3)      # inital guess for sigma and mu
                    bounds = ((0, -0.5), (1, 0)) # bounds (min, max)
                else:
                    mask = ber.index >= 0
                    initial = (-0.02,  0.3)      # sigma is fitted to negative values on the right
                    bounds = ((-1, 0), (0, 0.5)) # bounds (min, max)

                ss_ber = ber[mask]    # single sided BER
                if sum(ss_ber < self.d["setup"]["BERt"]) >= 2:  # at least two samples required for fitting  
                    ss_ber = ss_ber[ss_ber < self.d["setup"]["BERt"]]
                else:    # force fit by using the two samples with highes BER
                    ss_ber.sort_values(inplace=True)
                    ss_ber = ss_ber.iloc[:2]
                            
                try:
                    with warnings.catch_warnings(): # suppress the OptimizeWarning tha curve_fit sometimes raises
                        warnings.simplefilter("ignore")
                        (sigma, mu), _ = curve_fit(self.jitter_func_inv, ss_ber.values, ss_ber.index, 
                                                   p0=initial, bounds=bounds)
                except Exception as e:
                    print(e)
                    fit["success"] = False
                else:
                    fit[side] = {"sigma": sigma, "mu": mu}

            jitter = {}
            if fit["success"]:                    
                jitter["RJrms"] = fit["left"]["sigma"] - fit["right"]["sigma"]       # '-' because right sigma is negative 
                jitter["RJpp"] = self.jitter_func_inv(self.d["setup"]["BERs"], sigma=jitter["RJrms"], mu=0)
                jitter["DJpp"] = 0.5 + fit["left"]["mu"] + 0.5 - fit["right"]["mu"]
                jitter["TJpp"] = jitter["RJpp"] + jitter["DJpp"]
                eye_left = self.jitter_func_inv(self.d["setup"]["BERs"], sigma=fit["left"]["sigma"], mu=fit["left"]["mu"])
                eye_right= self.jitter_func_inv(self.d["setup"]["BERs"], sigma=fit["right"]["sigma"], mu=fit["right"]["mu"])
                jitter["center"] = (eye_left + eye_right) /2
                    
            self.d["jitter"][ch] = jitter
            self.d["fit"][ch] = fit


    def to_json(self, path_=None):
        """ Stores the JitterEstimator instance into a json using the custom JsonEnc """
        if path_ is None:
            fn = time.strftime('%Y%m%d_%H%M%S') + "_jitter.json"
            path_ = os.path.join(fn)

        dct = deepcopy(FILE_ID.JITTER.value)   # file meto data, like type and version
        dct["content"] = self.d
        with open(path_, "w") as f:
            json.dump(dct, f, cls=JsonEnc)


    @classmethod
    def from_json(class_, path_):
        """ Alternative constructor from json file using the custom JsonDev """
        with open(path_, "r") as f:
            dct = json.load(f, cls=JsonDec)
            
        # check meta
        if dct["type"] != FILE_ID.JITTER.value["type"]:
            msg = "Can't open file type '{}' when '{}' is expected".format(
                dct["type"], FILE_ID.JITTER.value["type"])
            raise Exception(msg)
        
        return class_(dct["content"])


def plot_jitter_fit(fns, filesDir, exclude_chs=None, refit=False, figsize=(12, 3.5)):
    """
    Creates 'plt' plots from jitter.json Files
    :input fns <str> filename or <list> of <str> filenames
    :input filesDir <str> directory of the fns
    :input exclude_chs <list> of <int>: excludes channels from the plot
    :input refit <bool> refits the Gaussian to the eyescan data if True
    :input figsize <tuple>
    """
    if type(fns) == str:   # this allows the fns to be a <str> filename or <list> of filenames
        fns = [fns]

    if exclude_chs is None:
        exclude_chs = []
    else:
        exclude_chs = [str(ch) for ch in exclude_chs]  # make sure list entries are str type

    for fn in fns:
        jitter = JitterEstimator.from_json(os.path.join(filesDir, fn))
        if refit:
            jitter.fit()
        BERs = jitter.d["setup"]["BERs"]
        BERt = jitter.d["setup"]["BERt"]
        y_scale = (BERs, 1)
        x_scale = (-0.5, 0.5)

        for ch, linerate in jitter.d["setup"]["linerates"].items():
            if ch not in exclude_chs:
                try:
                    tj = jitter.d["jitter"][ch]["TJpp"]
                    dj = jitter.d["jitter"][ch]["DJpp"]
                    rj = jitter.d["jitter"][ch]["RJpp"]
                    rj_rms = jitter.d["jitter"][ch]["RJrms"]
                    center = jitter.d["jitter"][ch]["center"]
                    meas = jitter.d["eyescan"][ch]
                    fit = jitter.d["fit"][ch]

                    plt.figure(figsize=figsize)
                    plt.semilogy(meas.index, meas.values, "bo", label="measurements")
                    plt.semilogy(x_scale, [BERt]*2, ":m", label="threshold={:.0e}".format(BERt))
                    plt.semilogy(x_scale, [BERs]*2, ":c", label="BERs={:.0e}".format(BERs))

                    for key, values in fit.items():
                        if key in ("left", "right"):
                            label = {"left": None, "right": "fitted Gaussian"}[key]
                            x = {"left": np.linspace(-0.5, 0), "right": np.linspace(0, 0.5)}[key] 
                            y = JitterEstimator.jitter_func(x, sigma=values["sigma"], mu=values["mu"])
                            plt.semilogy(x, y, "b:", label=label)

                            label = {"left": None, "right": "DJpp={:.3f}UI".format(dj)}[key]
                            x = {"left": [-0.5, values["mu"]], "right": [values["mu"], 0.5]}[key]
                            plt.fill_between(x, [0.25]*len(x), [BERt]*len(x), color="m", linewidth=0, label=label)

                            label = {"left": None, "right": "RJpp={:.3f}UI".format(rj)}[key]
                            x_start = JitterEstimator.jitter_func_inv(BERs, sigma=values["sigma"], mu=values["mu"])
                            x = np.linspace(x_start, values["mu"])
                            y = np.minimum(BERt, JitterEstimator.jitter_func(x, sigma=values["sigma"], mu=values["mu"]))
                            plt.fill_between(x, y, color="c", linewidth=0, label=label)

                    plt.fill_between([10], [1], color="white", label="RJrms={:.3f}UI".format(rj_rms))
                    plt.fill_between([10], [1], color="white", label="TJpp={:.3f}UI".format(tj))
                    plt.semilogy([center]*2, y_scale, "k", label="center={:.3f}UI".format(center))

                    plt.ylim(y_scale), plt.xlim(x_scale)
                    plt.ylabel("BER"), plt.xlabel("x [UI]")
                    plt.title("C{} {}Mb/s {}".format(ch, linerate, fn))
                    plt.legend(loc='upper center'), plt.grid();
                except Exception as e:
                    print("Exception during {}, C{} occured:\n{}".format(fn, ch, e))


def plot_jitter_overlay(fns, filesDir, exclude_chs=None, refit=False, figsize=(12, 5)):
    """
    Creates 'plt' plots from jitter.json Files
    :input fns <str> filename or <list> of <str> filenames
    :input filesDir <str> directory of the fns
    :input exclude_chs <list> of <int>: excludes channels from the plot
    :input refit <bool> refits the Gaussian to the eyescan data if True
    :input figsize <tuple>
    """
    class ColorNames():
        COLORS = ['orangered', 'orange', 'blue', 'skyblue', 'limegreen', 'lime', 'blueviolet', 'magenta', 'navy', 'royalblue', 'red', 'gold', 'green', 'yellowgreen', 'maroon', 'salmon', 'darkgrey', 'silver', 'peru', 'cyan', 'teal']
        def __init__(self):
            self.i = -1

        def same(self):
            """Returns the color name <str> of the current index"""
            if self.i == -1:
                self.i = 0
            return self.COLORS[self.i % len(self.COLORS)]

        def next(self):
            """Inclrements the index and returns the color name <str> of the current index""" 
            self.i += 1
            return self.same()


    plt.figure(figsize=figsize)
    colors = ColorNames()

    if type(fns) == str:   # this allows the fns to be a <str> filename or <list> of filenames
        fns = [fns]

    if exclude_chs is None:
        exclude_chs = []
    else:
        exclude_chs = [str(ch) for ch in exclude_chs]  # make sure list entries are str type


    for fn in fns:
        jitter = JitterEstimator.from_json(os.path.join(filesDir, fn))
        if refit:
            jitter.fit()
        BERs = jitter.d["setup"]["BERs"]
        BERt = jitter.d["setup"]["BERt"]
        y_scale = (BERs, 1)
        x_scale = (-0.5, 0.5)

        for ch, linerate in jitter.d["setup"]["linerates"].items():
            if ch not in exclude_chs:            
                try:
                    label = "{} C{} {}Mb/s".format(fn[:15], ch, linerate)
                    meas = jitter.d["eyescan"][ch]

                    high = meas[meas > BERt]
                    plt.semilogy(high.index, high.values, marker=".", color=colors.next(), linewidth=0)
                    low = meas[meas <= BERt]
                    plt.semilogy(low.index, low.values, marker="x", markersize=8, color=colors.same(), linewidth=0, label=label)

                    for key, values in jitter.d["fit"][ch].items():
                        if key in ("left", "right"):
                            x = {"left": np.linspace(-0.5, 0), "right": np.linspace(0, 0.5)}[key] 
                            y = JitterEstimator.jitter_func(x, sigma=values["sigma"], mu=values["mu"])
                            plt.semilogy(x, y, ":", color=colors.same())

                except Exception as e:
                    print("Exception during {}, C{} occured:\n{}".format(fn, ch, e))

        plt.ylim(y_scale), plt.xlim(x_scale)
    plt.ylabel("BER"), plt.xlabel("x [UI]")
    plt.title("C{} {}Mb/s {}".format(ch, linerate, fn))
    plt.legend(loc='upper center'), plt.grid();


if __name__ == "__main__":
    pass

