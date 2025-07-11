
"""
Core model definitions for capacity forecasting and surrogate modeling.

Modules:

  • train_pbm_surrogate_for_PI_RNN
      Train a RandomForest surrogate on PBM simulation pickles
      (used to inject PBM capacity drop predictions
      into the PI-RNN during training).

  • CustomRNNCellWithSurrogate & MultiStepPIRNN
      PI-RNN architecture that combines RNN hidden updates with
      PBM surrogate injection and supports optional MC-dropout.

  • BaselineMultiStepRNN
      Simple recurrent benchmark model without physics injection.

  • GPRBaseline
      Empirical + Gaussian Process model as a baseline.

  • PBMSurrogate
      Flexible, horizon-specific PBM surrogates for
      single-step capacity and capacity drop, plus recursive
      multi-step forecasts.

Usage:
    from models import (
        train_pbm_surrogate_for_PI_RNN,
        MultiStepPIRNN,
        BaselineMultiStepRNN,
        GPRBaseline,
        PBMSurrogate
    )
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import curve_fit
import GPy


# —————————————————————————————
# 1. Train PBM Surrogate (for PI-RNN)
# Use 'train_pbm_surrogate_for_PI_RNN' to obtain a RandomForest surrogate (rf_model) and its scaler (scaler_sim).
# This surrogate (rf_model, scaler_sim) is distinct from the PBMSurrogate class below and is used
# internally to inject physics-based predictions into the PI-RNN model during training.
# —————————————————————————————
def train_pbm_surrogate_for_PI_RNN(file_paths, sim_features, sim_target, seed=40):
    """
    Load simulation data, fit a PBM surrogate.
    Returns: (surrogate_model, scaler_sim)
    """
    # 1) load & concat
    sim_dfs = [pd.read_pickle(fp) for fp in file_paths]
    sim_df  = pd.concat(sim_dfs, ignore_index=True)

    # 2) features / target
    X = sim_df[sim_features].values
    y = sim_df[sim_target].values

    # 3) scale
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 4) train RF
    rf = RandomForestRegressor(n_estimators=200, random_state=seed)
    rf.fit(X_scaled, y)

    # 5) monkey-patch predict to zero out NaNs
    _orig_predict = rf.predict
    def _safe_predict(X_new):
        X_clean = np.nan_to_num(X_new, nan=0.0)
        return _orig_predict(X_clean)
    rf.predict = _safe_predict

    return rf, scaler

# —————————————————————————————
### PI-RNN  
# —————————————————————————————
class CustomRNNCellWithSurrogate(nn.Module):
    def __init__(self, input_size, hidden_size, surrogate_model, dropout_rate: float = 0.0):
        super().__init__()
        self.input_size     = input_size
        self.hidden_size    = hidden_size
        self.surrogate_model= surrogate_model

        self.W_ih     = nn.Linear(input_size, hidden_size)
        self.W_hh     = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

        # dropout layer: identity if rate=0 -> only activate for UQ
        self.dropout  = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.pbm_weight = nn.Parameter(torch.randn(1, hidden_size))
        self.fc         = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden):
        # PBM surrogate injection (unchanged)
        raw_feats = x[:, : self.input_size].detach().cpu().numpy()
        pbm_out   = self.surrogate_model.predict(raw_feats)
        pbm_out   = torch.tensor(pbm_out, dtype=torch.float32, device=x.device)
        pbm_out   = pbm_out.unsqueeze(1).expand(-1, self.hidden_size)
        h_pbm     = self.pbm_weight * pbm_out

        # standard RNN update
        h_next = self.activation(self.W_ih(x) + self.W_hh(hidden) + h_pbm)

        # **dropout** (no-op if dropout_rate=0)
        h_next = self.dropout(h_next)

        capacity_drop = self.fc(h_next)
        return h_next, capacity_drop


class MultiStepPIRNN(nn.Module):
    def __init__(self, input_size, hidden_size, surrogate_model, dropout_rate: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        # pass dropout_rate down into the cell
        self.rnn_cell    = CustomRNNCellWithSurrogate(input_size, hidden_size, surrogate_model, dropout_rate)
        self.input_size  = input_size

    def forward(self, x, current_capacity, forecast_steps):
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        next_capacity = current_capacity.clone().to(x.device)
        preds = []
        for t in range(forecast_steps):
            inp = torch.cat((x[:, t, :], next_capacity), dim=1)
            h, drop = self.rnn_cell(inp, h)
            next_capacity = next_capacity - drop
            preds.append(next_capacity.squeeze(-1))
        return torch.stack(preds, dim=1)



# —————————————————————————————
### Baseline RNN  
# —————————————————————————————
class BaselineRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BaselineRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ih = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden):
        h_next = self.activation(self.W_ih(x) + self.W_hh(hidden))
        capacity_drop = self.fc(h_next)
        return h_next, capacity_drop

class BaselineMultiStepRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BaselineMultiStepRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = BaselineRNNCell(input_size, hidden_size)
        self.input_size = input_size

    def forward(self, x, current_capacity, forecast_steps):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        next_capacity = current_capacity.clone().to(x.device)
        predictions = []
        for t in range(forecast_steps):
            current_input = torch.cat((x[:, t, :], next_capacity), dim=1)
            h, capacity_drop = self.rnn_cell(current_input, h)
            next_capacity = next_capacity - capacity_drop
            predictions.append(next_capacity.squeeze(-1))
        return torch.stack(predictions, dim=1)
    

# —————————————————————————————
### Baseline GPR
# —————————————————————————————
class GPRBaseline:
    def __init__(self, initial_points=9):
        self.initial_points   = initial_points
        self.empirical_params = {}
        self.gpr_models       = {}

    @staticmethod
    def empirical_model(x, a, b, c):
        return a + b * np.exp(c * x)

    def fit(self, cell_data, cell_key):
        # Force consistent key ordering: (Cell, Group)
        cell_key = (cell_key[0], cell_key[1])
        ip = self.initial_points

        cell_data = cell_data.sort_values('RPT Number')
        first_9 = cell_data.iloc[:ip]
        X_train = first_9[['RPT Number']].values.flatten()
        y_train = first_9['Capacity'].values

        try:
            popt, _ = curve_fit(
                self.empirical_model,
                X_train,
                y_train,
                p0=[np.mean(y_train), -1, -0.1],
                # bounds=([-10, -10, -10], [10, 10, 10]),
                maxfev=10000
            )
        except RuntimeError:
            popt = [np.mean(y_train), 0, 0]

        self.empirical_params[cell_key] = popt

        residuals = y_train - self.empirical_model(X_train, *popt)
        kernel = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0)
        gpr = GPy.models.GPRegression(X_train[:, None], residuals[:, None], kernel)
        gpr.optimize()

        self.gpr_models[cell_key] = gpr

    def predict(self, cell_data, cell_key):
        cell_key = (cell_key[0], cell_key[1])
        ip = self.initial_points

        forecast_data = cell_data.sort_values('RPT Number').iloc[ip:]
        rpt_vals = forecast_data['RPT Number'].values
        y_true   = forecast_data['Capacity'].values

        popt = self.empirical_params[cell_key]
        emp_preds = self.empirical_model(rpt_vals, *popt)
        gpr = self.gpr_models[cell_key]
        res_preds, _ = gpr.predict(rpt_vals[:, None])

        return y_true, emp_preds + res_preds.flatten()

    def predict_horizon(self, cell_data, cell_key, steps):
        cell_key = (cell_key[0], cell_key[1])
        ip = self.initial_points

        cell_data = cell_data.sort_values('RPT Number')
        if len(cell_data) < ip + steps:
            return [], []

        rpt_vals = cell_data['RPT Number'].values[ip:ip+steps]
        y_true = cell_data['Capacity'].values[ip:ip+steps]

        popt = self.empirical_params[cell_key]
        emp_preds = self.empirical_model(rpt_vals, *popt)
        gpr = self.gpr_models[cell_key]
        res_preds, _ = gpr.predict(rpt_vals[:, None])

        return y_true, emp_preds + res_preds.flatten()




# —————————————————————————————
### PBM Surrogate
# —————————————————————————————
class PBMSurrogate:
    def __init__(
        self,
        features,
        capacity_target="Capacity",
        drop_target="Capacity_Drop_Ah",
        n_estimators=50,
        random_state=40
    ):
        self.features        = features
        self.capacity_target = capacity_target
        self.drop_target     = drop_target

        # single-step capacity surrogate
        self.scaler_cap = MinMaxScaler()
        self.model_cap  = RandomForestRegressor(
                              n_estimators=n_estimators,
                              random_state=random_state
                          )

        # single-step drop surrogate
        self.scaler_drop = MinMaxScaler()
        self.model_drop  = RandomForestRegressor(
                              n_estimators=n_estimators,
                              random_state=random_state
                          )

        # containers for multi-step drop surrogates
        self.scalers_h   = {}  # horizon h → MinMaxScaler
        self.models_h    = {}  # horizon h → RandomForestRegressor

    def load_simulation_data(self, file_paths):
        sim_dfs = [pd.read_pickle(fp) for fp in file_paths]
        return pd.concat(sim_dfs, ignore_index=True)

    # -- Single-step capacity model --
    def fit_capacity(self, sim_df):
        X = sim_df[self.features].values
        y = sim_df[self.capacity_target].values
        Xs = self.scaler_cap.fit_transform(X)
        self.model_cap.fit(Xs, y)

    def predict_capacity(self, df):
        X  = df[self.features].values
        Xs = self.scaler_cap.transform(X)
        return self.model_cap.predict(Xs)

    # -- Single-step drop model --
    def fit_drop(self, sim_df):
        X = sim_df[self.features].values
        y = sim_df[self.drop_target].values
        Xs = self.scaler_drop.fit_transform(X)
        self.model_drop.fit(Xs, y)

    def predict_drop(self, df):
        X  = df[self.features].values
        Xs = self.scaler_drop.transform(X)
        return self.model_drop.predict(Xs)

    # -- Multi-step drop surrogate (horizon‐specific) --
    def fit_horizon(self, sim_df, h):
        Xh, yh = [], []
        for i in range(len(sim_df) - h):
            block = sim_df[self.features].iloc[i : i+h].values.flatten()
            Xh.append(block)
            yh.append(sim_df[self.drop_target].iloc[i + h])
        Xh = np.vstack(Xh); yh = np.array(yh)

        scaler_h = MinMaxScaler().fit(Xh)
        model_h  = RandomForestRegressor(
                       n_estimators=self.model_drop.n_estimators,
                       random_state=self.model_drop.random_state
                   )
        model_h.fit(scaler_h.transform(Xh), yh)

        self.scalers_h[h] = scaler_h
        self.models_h[h]  = model_h

    def predict_capacity_multi(self, df, steps):
        """
        Use the horizon-specific drop model to predict the drop at t+steps,
        then reconstruct capacity as cap[t] - drop_pred.
        """
        scaler_h = self.scalers_h[steps]
        model_h  = self.models_h[steps]

        caps = df[self.capacity_target].values
        n    = len(df)
        y_pred = []
        for i in range(n - steps):
            block = df[self.features].iloc[i : i+steps].values.flatten()
            drop_pred = model_h.predict(
                scaler_h.transform(block.reshape(1, -1))
            )[0]
            y_pred.append(caps[i] - drop_pred)

        return np.array(y_pred)





