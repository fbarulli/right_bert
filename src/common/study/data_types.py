from __future__ import annotations
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TrialData:
    """Data class for storing trial information."""
    
    trial_id: int
    number: int
    state: str
    value: Optional[float] = None
    params: Dict[str, Any] = None
    datetime_start: Optional[datetime] = None
    datetime_complete: Optional[datetime] = None
    user_attrs: Dict[str, Any] = None
    system_attrs: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values for None fields."""
        if self.params is None:
            self.params = {}
        if self.user_attrs is None:
            self.user_attrs = {}
        if self.system_attrs is None:
            self.system_attrs = {}
    
    @classmethod
    def from_optuna_trial(cls, trial) -> TrialData:
        """Create a TrialData instance from an Optuna Trial object."""
        return cls(
            trial_id=trial._trial_id,
            number=trial.number,
            state=trial.state,
            value=trial.value,
            params=trial.params,
            datetime_start=trial.datetime_start,
            datetime_complete=trial.datetime_complete,
            user_attrs=trial.user_attrs,
            system_attrs=trial.system_attrs
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the trial data to a dictionary."""
        return {
            'trial_id': self.trial_id,
            'number': self.number,
            'state': self.state,
            'value': self.value,
            'params': self.params,
            'datetime_start': self.datetime_start.isoformat() if self.datetime_start else None,
            'datetime_complete': self.datetime_complete.isoformat() if self.datetime_complete else None,
            'user_attrs': self.user_attrs,
            'system_attrs': self.system_attrs
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrialData:
        """Create a TrialData instance from a dictionary."""
        # Handle datetime conversion
        datetime_start = None
        if data.get('datetime_start'):
            datetime_start = datetime.fromisoformat(data['datetime_start'])
        
        datetime_complete = None
        if data.get('datetime_complete'):
            datetime_complete = datetime.fromisoformat(data['datetime_complete'])
        
        return cls(
            trial_id=data['trial_id'],
            number=data['number'],
            state=data['state'],
            value=data.get('value'),
            params=data.get('params', {}),
            datetime_start=datetime_start,
            datetime_complete=datetime_complete,
            user_attrs=data.get('user_attrs', {}),
            system_attrs=data.get('system_attrs', {})
        )


__all__ = ['TrialData']
