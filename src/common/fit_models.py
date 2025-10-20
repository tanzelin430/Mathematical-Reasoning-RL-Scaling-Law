import numpy as np
import pandas as pd

class FitLogErrRate:
    def __init__(self, L, r, N0_k, r_e0, N0_e0):
        self.L = L
        self.r = r
        self.N0_k = N0_k
        self.r_e0 = r_e0
        self.N0_e0 = N0_e0
    
    @staticmethod
    def _logistic_k(N, L, r, N0_k):
        """k(N) = L / (1 + exp(-r * (N - N0_k)))"""
        return L / (1 + np.exp(-r * (N - N0_k)))
    
    @staticmethod
    def _logistic_e0(N, L, r_e0, N0_e0):
        """E0(N) = L / (1 + exp(r_e0 * (N - N0_e0)))"""
        return L / (1 + np.exp(r_e0 * (N - N0_e0)))

    # Basic model
    @staticmethod
    def predict_log10_errrate_by_params_log10E(L, r, N0_k, r_e0, N0_e0, N, log10_E):
        """dual Logistic fitting: log10(ErrRate) = -k(N) * log10(E) + E0(N)
        Where: k(N) = L / (1 + exp(-r * (N - N0_k)))
               _E0(N) = L / (1 + exp(r_e0 * (N - N0_e0)))
               E0(N) = log(1 + _E0(N))
        """
        # Shared parameter: L
        # k(N) parameters: r, N0_k
        # E0(N) parameters: r_e0, N0_e0
        k = FitLogErrRate._logistic_k(N, L, r, N0_k)
        E0 = FitLogErrRate._logistic_e0(N, L, r_e0, N0_e0)
        E0 = np.log(1 + E0)
        return -k * log10_E + E0

    # static predict
    @staticmethod
    def predict_reward_by_params_log10E(L, r, N0_k, r_e0, N0_e0, N, log10_E):
        log10_errrate = FitLogErrRate.predict_log10_errrate_by_params_log10E(L, r, N0_k, r_e0, N0_e0, N, log10_E)
        def log10_errrate_to_reward(log10_errrate):
            return 1 - 10 ** log10_errrate
        return log10_errrate_to_reward(log10_errrate)
    
    @staticmethod
    def predict_errrate_by_params_log10E(L, r, N0_k, r_e0, N0_e0, N, log10_E):
        log10_errrate = FitLogErrRate.predict_log10_errrate_by_params_log10E(L, r, N0_k, r_e0, N0_e0, N, log10_E)
        def exp10(log10):
            return 10 ** log10
        return exp10(log10_errrate)
    
    # instance predict
    def predict_reward_log10E(self, N, log10_E):
        return FitLogErrRate.predict_reward_by_params_log10E(self.L, self.r, self.N0_k, self.r_e0, self.N0_e0, N, log10_E)
    
    def predict_errrate_log10E(self, N, log10_E):
        return FitLogErrRate.predict_errrate_by_params_log10E(self.L, self.r, self.N0_k, self.r_e0, self.N0_e0, N, log10_E)

    # for fitting
    @staticmethod
    def model(data, L, r, N0_k, r_e0, N0_e0):
        N, log10_E = data
        return FitLogErrRate.predict_log10_errrate_by_params_log10E(L, r, N0_k, r_e0, N0_e0, N, log10_E)

    # df["reward"] = predict_reward_df(df, "N", "E")
    def predict_reward_df(self, df, N_column, E_column):
        # apply predict_reward
        return df.apply(lambda row: self.predict_reward_log10E(row[N_column], np.log10(row[E_column])), axis=1)
    
    # df["errrate"] = predict_errrate_df(df, "N", "E")
    def predict_errrate_df(self, df, N_column, E_column):
        # apply predict_errrate
        return df.apply(lambda row: self.predict_errrate_log10E(row[N_column], np.log10(row[E_column])), axis=1)



class FitLogErrRateElegant:
    """
    log10(ErrRate) = -k(N) * log10(E) + E0(N)
    where
      k(N)  = log10(1 + c * s(N))          # 对数包裹 S 形（更稳边界、更解耦）
      E0(N) = L0 / (1 + exp(r_e0 * (N - N0_e0)))   # 保留对 E0 的 Logistic（也可换）
    可选 s(N):
      - 'logistic':      s = sigma(a*(N - N0_k))
      - 'richards':      s = (1 + Q * exp(-B*(N - N0_k)))**(-1/nu)
      - 'weibull_cdf':   s = 1 - exp(-(N/lmbd)**beta)   (N>0 时常用)
      - 'tanh01':        s = 0.5*(1 + tanh(a*(N - N0_k)))
    """
    def __init__(self, s_type='logistic',
                 # k(N) params
                 c=0.05, a=1e-9, N0_k=3e9, Q=10.0, B=1e-9, nu=1.0, lmbd=4e9, beta=2.0,
                 # E0(N) params
                 L0=0.05, r_e0=8e-10, N0_e0=3e9):
        self.s_type = s_type
        self.c = c
        self.a = a
        self.N0_k = N0_k
        self.Q = Q
        self.B = B
        self.nu = nu
        self.lmbd = lmbd
        self.beta = beta
        self.L0 = L0
        self.r_e0 = r_e0
        self.N0_e0 = N0_e0

    # --- s(N) families ---
    @staticmethod
    def _sigma(x):
        # 数值稳定的 sigmoid
        x = np.asarray(x)
        out = np.empty_like(x, dtype=float)
        pos = x >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        e = np.exp(x[neg])
        out[neg] = e / (1.0 + e)
        return out

    def _s_of_N(self, N):
        N = np.asarray(N, dtype=float)
        if self.s_type == 'logistic':
            return self._sigma(self.a * (N - self.N0_k))
        elif self.s_type == 'richards':
            # s(N) = (1 + Q * exp(-B*(N - N0_k)))^(-1/nu)
            return (1.0 + self.Q * np.exp(-self.B * (N - self.N0_k))) ** (-1.0 / self.nu)
        elif self.s_type == 'weibull_cdf':
            # s(N) = 1 - exp(-(N/lmbd)^beta)
            x = np.maximum(N, 0.0) / self.lmbd
            return 1.0 - np.exp(-np.power(x, self.beta))
        elif self.s_type == 'tanh01':
            return 0.5 * (1.0 + np.tanh(self.a * (N - self.N0_k)))
        else:
            raise ValueError(f"Unknown s_type: {self.s_type}")

    # --- components ---
    def k_of_N(self, N):
        sN = self._s_of_N(N)
        return np.log10(1.0 + self.c * sN)  # 关键改写

    def E0_of_N(self, N):
        # 保留 Logistic（与你原来一致；也可换成更通用的分段/有界多项式）
        return self.L0 / (1.0 + np.exp(self.r_e0 * (N - self.N0_e0)))

    # --- core predictions ---
    def predict_log10_errrate(self, N, log10_E):
        k = self.k_of_N(N)
        E0 = self.E0_of_N(N)
        return -k * log10_E + E0

    def predict_errrate(self, N, E):
        log10_E = np.log10(E)
        return np.power(10.0, self.predict_log10_errrate(N, log10_E))

    def predict_reward(self, N, E):
        return 1.0 - self.predict_errrate(N, E)

    # --- static helpers (for curve_fit) ---
    @staticmethod
    def model(data, c, a, N0_k, L0, r_e0, N0_e0):
        """
        简洁参数版（s_type='logistic'）用于 curve_fit：
          data: (N, log10_E)
          params: c, a, N0_k (for k) + L0, r_e0, N0_e0 (for E0)
        """
        N, log10_E = data
        # s(N)
        sN = FitLogErrRateElegant._sigma(a * (N - N0_k))
        # k(N)=log10(1 + c*s(N))
        k = np.log10(1.0 + c * sN)
        # E0(N) Logistic
        E0 = L0 / (1.0 + np.exp(r_e0 * (N - N0_e0)))
        return -k * log10_E + E0

    # --- df helpers ---
    def predict_reward_df(self, df, N_column, E_column):
        return df.apply(lambda row: self.predict_reward(row[N_column], row[E_column]), axis=1)

    def predict_errrate_df(self, df, N_column, E_column):
        return df.apply(lambda row: self.predict_errrate(row[N_column], row[E_column]), axis=1)

class FitErrShiftedPower:
    """
    Err(N,E) = B * (Nc/N)^{alpha_N} * (E + E00*(Nc/N)^{xi})^{-alpha_E}
    R(N,E)   = 1 - Err(N,E)

    推荐在 log10 空间拟合： y = log10(Err), X = (N, log10_E)
    参数顺序（固定 Nc 版的 model）： [B, alpha_N, alpha_E, E00, xi]
    参数顺序（可拟合 Nc 版的 model）： [B, Nc, alpha_N, alpha_E, E00, xi]
    """

    def __init__(self, Nc=3.0e9, B=0.2, alpha_N=0.2, alpha_E=0.06, E00=3.0e8, xi=0.5):
        self.Nc = float(Nc)
        self.B = float(B)
        self.alpha_N = float(alpha_N)
        self.alpha_E = float(alpha_E)
        self.E00 = float(E00)
        self.xi = float(xi)

    # ---------- core ----------
    def err(self, N, E):
        N = np.asarray(N, float)
        E = np.asarray(E, float)
        Nc_over_N = self.Nc / N
        E0N = self.E00 * np.power(Nc_over_N, self.xi)
        En = E + E0N
        return self.B * np.power(Nc_over_N, self.alpha_N) * np.power(En, -self.alpha_E)

    def reward(self, N, E):
        return 1.0 - self.err(N, E)

    # ---------- predict log10(Err) ----------
    def predict_log10_err(self, N, E=None, log10_E=None):
        N = np.asarray(N, float)
        if log10_E is None:
            if E is None:
                raise ValueError("Provide either E or log10_E.")
            log10_E = np.log10(np.asarray(E, float))
        else:
            log10_E = np.asarray(log10_E, float)

        Nc_over_N = self.Nc / N
        log10_B = np.log10(self.B)
        E0N = self.E00 * np.power(Nc_over_N, self.xi)
        E_val = np.power(10.0, log10_E)
        # log10(E + E0N) = log10(10^{log10_E} + E0N) = log10_E + log10(1 + E0N/10^{log10_E})
        log10_Eplus = log10_E + np.log10(1.0 + E0N / np.maximum(E_val, 1e-300))

        return log10_B + self.alpha_N * np.log10(Nc_over_N) - self.alpha_E * log10_Eplus

    # ---------- curve_fit models (一致性保证) ----------
    @staticmethod
    def model_log10_fixed_Nc(data, B, alpha_N, alpha_E, E00, xi, Nc=3.0e9):
        N, log10_E = data
        Nc_over_N = Nc / N
        log10_B = np.log10(B)
        E0N = E00 * np.power(Nc_over_N, xi)
        E_val = np.power(10.0, log10_E)
        log10_Eplus = log10_E + np.log10(1.0 + E0N / np.maximum(E_val, 1e-300))
        return log10_B + alpha_N * np.log10(Nc_over_N) - alpha_E * log10_Eplus

    @staticmethod
    def model_log10_free_Nc(data, B, Nc, alpha_N, alpha_E, E00, xi):
        N, log10_E = data
        Nc_over_N = Nc / N
        log10_B = np.log10(B)
        E0N = E00 * np.power(Nc_over_N, xi)
        E_val = np.power(10.0, log10_E)
        log10_Eplus = log10_E + np.log10(1.0 + E0N / np.maximum(E_val, 1e-300))
        return log10_B + alpha_N * np.log10(Nc_over_N) - alpha_E * log10_Eplus
    
    # df["reward"] = predict_reward_df(df, "N", "E")
    def predict_reward_df(self, df, N_column, E_column):
        # apply predict_reward
        return df.apply(lambda row: 1-np.exp(self.predict_log10_err(N=row[N_column], E=row[E_column])), axis=1)
    
    # df["errrate"] = predict_errrate_df(df, "N", "E")
    def predict_errrate_df(self, df, N_column, E_column):
        # apply predict_errrate
        return df.apply(lambda row: np.exp(self.predict_log10_err(N=row[N_column], E=row[E_column])), axis=1)


    # ---------- helpers ----------
    @classmethod
    def from_fixed_params(cls, Nc, B, alpha_N, alpha_E, E00, xi):
        return cls(Nc=Nc, B=B, alpha_N=alpha_N, alpha_E=alpha_E, E00=E00, xi=xi)

    @staticmethod
    def format_params_fixed(p):
        B, alpha_N, alpha_E, E00, xi = p
        return (f"B={B:.3g}, alpha_N={alpha_N:.3g}, alpha_E={alpha_E:.3g}, "
                f"E00={E00:.3e}, xi={xi:.3g}")

    @staticmethod
    def format_params_free(p):
        B, Nc, alpha_N, alpha_E, E00, xi = p
        return (f"B={B:.3g}, Nc={Nc:.3e}, alpha_N={alpha_N:.3g}, alpha_E={alpha_E:.3g}, "
                f"E00={E00:.3e}, xi={xi:.3g}")
