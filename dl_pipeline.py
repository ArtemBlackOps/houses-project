from copy import deepcopy
from dataclasses import dataclass, field
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold



def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Map функций активации и оптимизаторов ---
ACTIVATION_MAP = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'elu': nn.ELU,
    'gelu': nn.GELU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'selu': nn.SELU
}

OPTIMIZER_MAP = {
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
    'sgd': torch.optim.SGD,
    'rmsprop': torch.optim.RMSprop
}


@dataclass
class CategoricalConfig:
    """Конфигурация для эмбеддингов категориальных признаков."""
    cat_cols: List[str]
    # Словарь {col_name: num_unique_values}
    cat_dims: Dict[str, int]
    # Фиксированная размерность эмбеддинга или None (тогда рассчитывается автоматически)
    emb_drop: float = 0.0


@dataclass
class DLConfig:
    """Конфигурация гиперпараметров модели и процесса обучения."""
    # Архитектура MLP
    hidden_units: List[int] = field(default_factory=lambda: [64, 32])
    activation: str = 'relu'
    use_batchnorm: bool = True
    dropout_rates: Union[float, List[float]] = 0.0
    
    # Категориальные фичи (Задание со звездочкой)
    cat_config: Optional[CategoricalConfig] = None
    
    # Параметры обучения
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 20
    optimizer_name: str = 'adamw'
    
    # Scheduler
    use_scheduler: bool = True
    scheduler_type: str = 'cosine'  # 'cosine' или 'step'

    # Early Stopping
    early_stopping: bool = True
    patience: int = 20  # Сколько эпох ждем без улучшений val_loss
    
    # Задача и Loss
    task: str = 'binary'  # 'binary', 'multiclass', 'regression'
    num_classes: int = 1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class TabularDataset(Dataset):
    """Датасет для совместной обработки численных и категориальных признаков."""
    def __init__(self, X_num: np.ndarray, X_cat: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long) if X_cat is not None else None
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        item = {'num': self.X_num[idx]}
        if self.X_cat is not None:
            item['cat'] = self.X_cat[idx]
        if self.y is not None:
            item['y'] = self.y[idx]
        return item


class FlexibleMLP(nn.Module):
    """Универсальная нейросеть с поддержкой BatchNorm, Dropout, Embeddings и разной глубины."""
    def __init__(self, in_features_num: int, config: DLConfig):
        super().__init__()
        self.config = config
        
        # 1. Обработка Embeddings для категориальных фичей
        self.embeddings = nn.ModuleList()
        total_emb_dim = 0
        
        if config.cat_config is not None:
            for col in config.cat_config.cat_cols:
                num_classes = config.cat_config.cat_dims[col]
                # Формула rule-of-thumb для размерности эмбеддинга
                emb_dim = min(50, (num_classes + 1) // 2)
                emb_dim = max(2, emb_dim)
                self.embeddings.append(nn.Embedding(num_classes + 1, emb_dim))  # +1 для неизвестных/out-of-bounds
                total_emb_dim += emb_dim
                
            self.emb_drop = nn.Dropout(config.cat_config.emb_drop)
        else:
            self.emb_drop = nn.Identity()

        # Входная размерность первого линейного слоя = число численных + сумма размерностей эмбеддингов
        current_dim = in_features_num + total_emb_dim
        
        # 2. Нормализация Dropout списком или единым числом
        if isinstance(config.dropout_rates, float):
            dropouts = [config.dropout_rates] * len(config.hidden_units)
        else:
            dropouts = config.dropout_rates
            assert len(dropouts) == len(config.hidden_units), "Длина dropout_rates должна совпадать с hidden_units"

        act_cls = ACTIVATION_MAP.get(config.activation.lower(), nn.ReLU)

        # 3. Сборка скрытых слоев
        layers = []
        for hidden_dim, drop_rate in zip(config.hidden_units, dropouts):
            layers.append(nn.Linear(current_dim, hidden_dim))
            
            if config.use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
                
            layers.append(act_cls())
            
            if drop_rate > 0:
                layers.append(nn.Dropout(drop_rate))
                
            current_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        
        # 4. Выходной слой
        out_dim = config.num_classes if config.task == 'multiclass' else 1
        self.head = nn.Linear(current_dim, out_dim)

    def forward(self, x_num: torch.Tensor, x_cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_inputs = [x_num]
        
        if x_cat is not None and len(self.embeddings) > 0:
            emb_outs = []
            for i, emb_layer in enumerate(self.embeddings):
                emb_outs.append(emb_layer(x_cat[:, i]))
            x_emb = torch.cat(emb_outs, dim=1)
            x_emb = self.emb_drop(x_emb)
            x_inputs.append(x_emb)
            
        x = torch.cat(x_inputs, dim=1)
        x = self.mlp(x)
        out = self.head(x)
        return out


class DLTrainer:
    """Класс для управления обучением, валидацией и инференсом."""

    def __init__(self, config: DLConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model: Optional[FlexibleMLP] = None
        self.criterion = self._get_loss_fn()

    def _get_loss_fn(self):
        if self.config.task == 'binary':
            return nn.BCEWithLogitsLoss()
        elif self.config.task == 'multiclass':
            return nn.CrossEntropyLoss()
        else:
            return nn.MSELoss()

    def fit(
        self,
        X_train_num: np.ndarray,
        y_train: np.ndarray,
        X_val_num: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        X_train_cat: Optional[np.ndarray] = None,
        X_val_cat: Optional[np.ndarray] = None,
        seed: int = 42,
    ) -> Dict[str, List[float]]:

        seed_everything(seed)

        # Подготовка данных
        train_dataset = TabularDataset(X_train_num, X_train_cat, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )

        val_loader = None
        if X_val_num is not None and y_val is not None:
            val_dataset = TabularDataset(X_val_num, X_val_cat, y_val)
            val_loader = DataLoader(
                val_dataset, batch_size=self.config.batch_size, shuffle=False
            )

        # Инициализация модели
        in_features = X_train_num.shape[1]
        self.model = FlexibleMLP(in_features, self.config).to(self.device)

        opt_cls = OPTIMIZER_MAP.get(
            self.config.optimizer_name.lower(), torch.optim.AdamW
        )
        optimizer = opt_cls(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        scheduler = None
        if self.config.use_scheduler:
            if self.config.scheduler_type == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.config.epochs
                )
            elif self.config.scheduler_type == 'step':
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=5, gamma=0.5
                )

        history = {'train_loss': [], 'val_loss': []}

        # --- Переменные для Early Stopping ---
        best_val_loss = float('inf')
        best_model_weights = None
        patience_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            # --- Training ---
            self.model.train()
            running_loss = 0.0
            for batch in train_loader:
                x_num = batch['num'].to(self.device)
                x_cat = (
                    batch['cat'].to(self.device) if 'cat' in batch else None
                )
                y = batch['y'].to(self.device)

                if self.config.task == 'binary':
                    y = y.unsqueeze(1)
                elif self.config.task == 'multiclass':
                    y = y.long()
                else:
                    y = y.unsqueeze(1)

                optimizer.zero_grad()
                preds = self.model(x_num, x_cat)
                loss = self.criterion(preds, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * x_num.size(0)

            if scheduler is not None:
                scheduler.step()

            epoch_train_loss = running_loss / len(train_dataset)
            history['train_loss'].append(epoch_train_loss)

            # --- Validation ---
            epoch_val_loss = None
            if val_loader is not None:
                self.model.eval()
                val_running_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        x_num = batch['num'].to(self.device)
                        x_cat = (
                            batch['cat'].to(self.device)
                            if 'cat' in batch
                            else None
                        )
                        y = batch['y'].to(self.device)

                        if self.config.task == 'binary':
                            y = y.unsqueeze(1)
                        elif self.config.task == 'multiclass':
                            y = y.long()
                        else:
                            y = y.unsqueeze(1)

                        preds = self.model(x_num, x_cat)
                        loss = self.criterion(preds, y)
                        val_running_loss += loss.item() * x_num.size(0)

                epoch_val_loss = val_running_loss / len(val_loader.dataset)
                history['val_loss'].append(epoch_val_loss)

                # --- Логика Early Stopping ---
                if self.config.early_stopping:
                    if epoch_val_loss < best_val_loss:
                        best_val_loss = epoch_val_loss
                        best_model_weights = deepcopy(self.model.state_dict())
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.config.patience:
                            # Восстанавливаем лучшие веса
                            if best_model_weights is not None:
                                self.model.load_state_dict(best_model_weights)
                            break

        # Если Early Stopping не сработал раньше, загружаем лучшую модель под конец
        if (
            self.config.early_stopping
            and val_loader is not None
            and best_model_weights is not None
        ):
            self.model.load_state_dict(best_model_weights)

        return history

    def predict(
        self, X_num: np.ndarray, X_cat: Optional[np.ndarray] = None
    ) -> np.ndarray:
        self.model.eval()
        dataset = TabularDataset(X_num, X_cat)
        loader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=False
        )

        preds_list = []
        with torch.no_grad():
            for batch in loader:
                x_num = batch['num'].to(self.device)
                x_cat = (
                    batch['cat'].to(self.device) if 'cat' in batch else None
                )
                out = self.model(x_num, x_cat)

                if self.config.task == 'binary':
                    probs = torch.sigmoid(out)
                    preds_list.append(probs.cpu().numpy())
                elif self.config.task == 'multiclass':
                    probs = torch.softmax(out, dim=1)
                    preds_list.append(probs.cpu().numpy())
                else:
                    preds_list.append(out.cpu().numpy())

        return np.vstack(preds_list)


class PyTorchMLPWrapper:
    """ Обертка для DLTrainer, поддерживающая передачу валидационной выборки."""

    def __init__(self, config):
        self.config = config
        self.trainer = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_tr = X_train.values if hasattr(X_train, 'values') else X_train
        y_tr = y_train.values if hasattr(y_train, 'values') else y_train

        X_v = (
            (X_val.values if hasattr(X_val, 'values') else X_val)
            if X_val is not None
            else None
        )
        y_v = (
            (y_val.values if hasattr(y_val, 'values') else y_val)
            if y_val is not None
            else None
        )

        self.trainer = DLTrainer(self.config)
        self.trainer.fit(
            X_train_num=X_tr, y_train=y_tr, X_val_num=X_v, y_val=y_v
        )
        return self

    def predict_proba(self, X):
        X_arr = X.values if hasattr(X, 'values') else X
        probs = self.trainer.predict(X_arr).reshape(-1, 1)
        return np.hstack([1 - probs, probs])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs > 0.5).astype(int)

    def score(self, X, y):
        preds = self.predict(X)
        y_arr = y.values if hasattr(y, 'values') else y
        return accuracy_score(y_arr, preds)