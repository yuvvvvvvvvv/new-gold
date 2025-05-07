import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path

class ConfigurationError(Exception):
    """Custom exception for configuration errors"""
    pass

class ConfigLoader:
    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load and validate the YAML configuration file"""
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            self._override_with_env_vars()
            self._validate_config()
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing YAML configuration: {e}")
    
    def _override_with_env_vars(self) -> None:
        """Override configuration values with environment variables"""
        env_mappings = {
            'MT5_LOGIN': ('mt5', 'credentials', 'login'),
            'MT5_PASSWORD': ('mt5', 'credentials', 'password'),
            'MT5_SERVER': ('mt5', 'credentials', 'server'),
            'NEWS_API_KEY': ('apis', 'news', 'api_key'),
            'TELEGRAM_TOKEN': ('apis', 'telegram', 'token'),
            'TELEGRAM_CHAT_ID': ('apis', 'telegram', 'chat_id')
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                self._set_nested_value(self.config, config_path, env_value)
    
    def _set_nested_value(self, d: dict, path: tuple, value: Any) -> None:
        """Set a value in a nested dictionary using a path tuple"""
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = value
    
    def _validate_config(self) -> None:
        """Validate the configuration structure and required fields"""
        required_sections = ['mt5', 'apis', 'trading', 'indicators', 'ai_model', 'logging']
        for section in required_sections:
            if section not in self.config:
                raise ConfigurationError(f"Missing required section: {section}")
        
        self._validate_mt5_config()
        self._validate_trading_config()
        self._validate_indicators_config()
        self._validate_ai_model_config()
        self._validate_logging_config()
    
    def _validate_mt5_config(self) -> None:
        """Validate MT5 configuration section"""
        required_fields = [
            ('credentials', 'login'),
            ('credentials', 'password'),
            ('credentials', 'server'),
            ('symbol', 'name'),
            ('symbol', 'timeframe')
        ]
        self._check_required_fields('mt5', required_fields)
    
    def _validate_trading_config(self) -> None:
        """Validate trading configuration section"""
        required_fields = [
            ('risk_management', 'max_risk_per_trade'),
            ('risk_management', 'max_daily_drawdown'),
            ('schedule', 'trading_hours'),
            ('circuit_breakers', 'volatility_threshold')
        ]
        self._check_required_fields('trading', required_fields)
    
    def _validate_indicators_config(self) -> None:
        """Validate technical indicators configuration section"""
        required_fields = [
            ('moving_averages', 'ema_fast'),
            ('moving_averages', 'ema_slow'),
            ('oscillators', 'rsi', 'period'),
            ('volatility', 'atr', 'period')
        ]
        self._check_required_fields('indicators', required_fields)
    
    def _validate_ai_model_config(self) -> None:
        """Validate AI model configuration section"""
        required_fields = [
            ('lstm', 'sequence_length'),
            ('lstm', 'units'),
            ('features', 'price'),
            ('training', 'optimizer')
        ]
        self._check_required_fields('ai_model', required_fields)
    
    def _validate_logging_config(self) -> None:
        """Validate logging configuration section"""
        required_fields = [
            ('level',),
            ('file', 'path'),
            ('format',)
        ]
        self._check_required_fields('logging', required_fields)
    
    def _check_required_fields(self, section: str, fields: list) -> None:
        """Check if required fields exist in a configuration section"""
        section_config = self.config.get(section, {})
        for field_path in fields:
            current = section_config
            for key in field_path:
                if not isinstance(current, dict) or key not in current:
                    raise ConfigurationError(
                        f"Missing required field: {section}.{'.'.join(field_path)}"
                    )
                current = current[key]
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation"""
        current = self.config
        for key in keys:
            if not isinstance(current, dict):
                return default
            current = current.get(key, default)
            if current is None:
                return default
        return current
    
    def get_mt5_credentials(self) -> Dict[str, str]:
        """Get MT5 credentials with validation"""
        credentials = self.get('mt5', 'credentials')
        if not all(credentials.get(k) for k in ['login', 'password', 'server']):
            raise ConfigurationError("Incomplete MT5 credentials")
        return credentials
    
    def get_trading_parameters(self) -> Dict[str, Any]:
        """Get trading parameters with validation"""
        return self.get('trading', default={})
    
    def get_indicator_parameters(self) -> Dict[str, Any]:
        """Get technical indicator parameters"""
        return self.get('indicators', default={})
    
    def get_ai_model_parameters(self) -> Dict[str, Any]:
        """Get AI model parameters"""
        return self.get('ai_model', default={})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.get('logging', default={})

# Example usage
if __name__ == '__main__':
    try:
        config = ConfigLoader()
        print("Configuration loaded successfully")
        
        # Example: Access configuration values
        mt5_creds = config.get_mt5_credentials()
        print(f"MT5 Symbol: {config.get('mt5', 'symbol', 'name')}")
        print(f"Risk per trade: {config.get('trading', 'risk_management', 'max_risk_per_trade')}")
        
    except ConfigurationError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")