# apps/analytics/management/commands/tune_model_optuna.py
import yaml
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import structlog
import logging
from django.core.management.base import BaseCommand, CommandParser
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# **استيراد الدالة الصحيحة**
from apps.analytics.models.train import (
    _prepare_numeric_df,
    create_target_fixed_horizon, # <-- الدالة الجديدة
    load_training_data,
)
from apps.market_data.models import Asset

logger = structlog.get_logger(__name__)

def objective_binary(
    trial: optuna.Trial,
    config: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> float:
    """
    الدالة الهدف لتحسين نموذج تصنيف ثنائي (long-only or short-only).
    الهدف هو تعظيم AUC.
    """
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': 1000,
        'seed': config['training'].get('seed', 42),
        'n_jobs': -1,
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'is_unbalance': True, # السماح لـ LightGBM بالتعامل مع عدم التوازن
    }
    
    model = lgb.LGBMClassifier(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    preds_proba = model.predict_proba(X_val.values)[:, 1]
    auc_score = roc_auc_score(y_val, preds_proba)
    return auc_score

class Command(BaseCommand):
    help = "Optimizes hyperparameters for a specialized binary model (long or short) using the fixed_horizon target."

    def add_arguments(self, parser: CommandParser):
        parser.add_argument("--config", type=str, required=True, help="Path to the base YAML config file.")
        parser.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials to run.")
        parser.add_argument(
            "--direction",
            type=str,
            required=True,
            choices=['long', 'short'],
            help="The direction to optimize for ('long' or 'short')."
        )

    def handle(self, *args, **options):
        structlog.configure(processors=[structlog.dev.ConsoleRenderer()])
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            handler = logging.StreamHandler()
            root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)

        with open(options["config"], 'r') as f:
            config = yaml.safe_load(f)

        direction = options["direction"]
        self.stdout.write(self.style.SUCCESS(f"--- Starting Hyperparameter Tuning for {direction.upper()}-ONLY Model ---"))
        
        train_cfg = config['training']
        target_cfg = config['target_creation']
        
        self.stdout.write("Loading and preparing data...")
        asset = Asset.objects.get(symbol=train_cfg['asset_symbol'])
        X, prices_df = load_training_data(asset, train_cfg['timeframe'])
        
        self.stdout.write(f"Creating specialized binary target for '{direction}' using fixed_horizon...")
        
        # 1. إنشاء الهدف متعدد الفئات أولاً
        multi_class_target = create_target_fixed_horizon(
            prices_df['close'],
            target_cfg['horizon_steps'],
            target_cfg['return_threshold']
        )
        
        # 2. تحويله إلى هدف ثنائي بناءً على الاتجاه
        if direction == 'long':
            y = (multi_class_target == 2).astype(int) # 1 if BUY (2), 0 otherwise
        else: # short
            y = (multi_class_target == 0).astype(int) # 1 if SELL (0), 0 otherwise

        data = X.join(y.rename('target')).dropna(subset=['target'])

        if data.empty or data['target'].sum() == 0:
            raise CommandError(f"No positive samples found for {direction} model. Cannot run tuning.")

        X_final = _prepare_numeric_df(data.drop(columns='target'))
        y_final = data['target'].astype(int)
        
        X_train, _, y_train, _ = train_test_split(X_final, y_final, test_size=0.1, shuffle=False)
        X_train_part, X_val, y_train_part, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

        self.stdout.write(f"Data prepared. Training set size: {len(X_train_part)}, Validation set size: {len(X_val)}")
        self.stdout.write(f"Class distribution in validation set:\n{y_val.value_counts(normalize=True).round(3)}")
        self.stdout.write("Starting Optuna optimization (maximizing AUC)...")

        study = optuna.create_study(direction='maximize')
        
        study.optimize(
            lambda trial: objective_binary(trial, config, X_train_part, y_train_part, X_val, y_val), 
            n_trials=options["n_trials"],
            show_progress_bar=True
        )

        best_trial = study.best_trial
        self.stdout.write(self.style.SUCCESS("\n--- Tuning Complete ---"))
        self.stdout.write(f"Best trial (AUC): {best_trial.value:.4f}")
        self.stdout.write(f"Best hyperparameters for {direction.upper()}-ONLY model (ready for YAML):")
        
        best_params = best_trial.params
        output_params = {
            f'lightgbm_{direction}': {
                'objective': 'binary',
                'metric': 'auc',
                'n_estimators': 1000,
                'boosting_type': 'gbdt',
                'is_unbalance': True,
                **best_params
            }
        }
        self.stdout.write(yaml.dump(output_params, indent=2, sort_keys=False))
        self.stdout.write(self.style.WARNING(
            f"\nNext step: Copy this parameter block into your config file."
        ))

        