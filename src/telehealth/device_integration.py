"""
Phase X-A: Home EEG Device Ecosystem Integration
Support for consumer and medical-grade home EEG devices
"""

import json
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
import logging
from enum import Enum
import aiohttp
import websockets
import serial_asyncio
import bluetooth
from abc import ABC, abstractmethod

class DeviceType(Enum):
    """Supported home EEG device types"""
    EMOTIV_EPOC_X = "emotiv_epoc_x"
    EMOTIV_INSIGHT = "emotiv_insight"
    MUSE_2 = "muse_2"
    MUSE_S = "muse_s"
    DREEM_HEADBAND = "dreem_headband"
    NEUROSITY_CROWN = "neurosity_crown"
    OPENBCI_CYTON = "openbci_cyton"
    ENOBIO_32 = "enobio_32"
    COGNIONICS_QUICK_30 = "cognionics_quick_30"
    GTEC_UNICORN = "gtec_unicorn"

class DeviceStatus(Enum):
    """Home EEG device connection status"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    ERROR = "error"
    LOW_BATTERY = "low_battery"
    POOR_SIGNAL = "poor_signal"
    CALIBRATING = "calibrating"
    MAINTENANCE = "maintenance"

class ConnectionType(Enum):
    """Device connection methods"""
    BLUETOOTH = "bluetooth"
    WIFI = "wifi"
    USB = "usb"
    WIRELESS_DONGLE = "wireless_dongle"
    CLOUD_API = "cloud_api"

@dataclass
class DeviceCapabilities:
    """Device technical specifications"""
    channel_count: int
    sampling_rate_hz: int
    resolution_bits: int
    max_session_minutes: int
    has_accelerometer: bool = False
    has_gyroscope: bool = False
    has_ppg: bool = False
    supports_impedance: bool = False
    supports_realtime_processing: bool = True
    battery_life_hours: float = 8.0
    wireless_range_meters: int = 10

@dataclass
class DeviceConfiguration:
    """Device-specific configuration"""
    device_id: str
    device_type: DeviceType
    firmware_version: str
    channel_map: Dict[str, str]  # device channel -> standard 10-20 position
    sampling_rate: int
    notch_filters: List[float]
    bandpass_filter: Tuple[float, float]
    impedance_threshold_kohm: float = 50.0
    signal_quality_threshold: float = 0.8
    auto_reconnect: bool = True
    data_encryption: bool = True

@dataclass
class SignalQuality:
    """Real-time signal quality metrics"""
    timestamp: datetime
    channel_qualities: Dict[str, float]  # 0.0-1.0 quality score per channel
    overall_quality: float
    impedances_kohm: Dict[str, float]
    artifacts_detected: List[str]
    signal_to_noise_ratio: float
    contact_quality: Dict[str, str]  # good/fair/poor per channel

@dataclass
class DeviceSession:
    """Home EEG recording session"""
    session_id: str
    device_id: str
    participant_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    signal_quality: Optional[SignalQuality] = None
    data_file_path: Optional[str] = None
    session_notes: str = ""
    artifacts_count: int = 0
    data_loss_percentage: float = 0.0
    completed_successfully: bool = False

class BaseDeviceDriver(ABC):
    """Abstract base class for home EEG device drivers"""

    def __init__(self, device_config: DeviceConfiguration):
        self.config = device_config
        self.status = DeviceStatus.DISCONNECTED
        self.logger = logging.getLogger(f'device_{device_config.device_type.value}')
        self.current_session: Optional[DeviceSession] = None
        self.signal_buffer: List[np.ndarray] = []
        self.quality_buffer: List[SignalQuality] = []

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to device"""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from device"""
        pass

    @abstractmethod
    async def start_streaming(self) -> bool:
        """Start EEG data streaming"""
        pass

    @abstractmethod
    async def stop_streaming(self) -> bool:
        """Stop EEG data streaming"""
        pass

    @abstractmethod
    async def get_signal_quality(self) -> SignalQuality:
        """Get current signal quality"""
        pass

    @abstractmethod
    async def set_configuration(self, config: Dict[str, Any]) -> bool:
        """Update device configuration"""
        pass

    async def get_device_info(self) -> Dict[str, Any]:
        """Get device information"""
        return {
            'device_id': self.config.device_id,
            'device_type': self.config.device_type.value,
            'firmware_version': self.config.firmware_version,
            'status': self.status.value,
            'channel_count': len(self.config.channel_map),
            'sampling_rate': self.config.sampling_rate
        }

class EmotivDriver(BaseDeviceDriver):
    """Emotiv EPOC X and Insight driver"""

    def __init__(self, device_config: DeviceConfiguration):
        super().__init__(device_config)
        self.cortex_session = None
        self.websocket = None
        self.auth_token = None

    async def connect(self) -> bool:
        """Connect to Emotiv device via Cortex API"""
        try:
            self.status = DeviceStatus.CONNECTING

            # Connect to Cortex websocket
            cortex_url = "wss://localhost:6868"
            self.websocket = await websockets.connect(cortex_url)

            # Authenticate
            auth_success = await self._authenticate()
            if not auth_success:
                return False

            # Query devices
            devices = await self._query_devices()
            if not devices:
                return False

            # Connect to specific device
            device_id = self.config.device_id
            if device_id not in [d['id'] for d in devices]:
                self.logger.error(f"Device {device_id} not found")
                return False

            # Create session
            session_id = await self._create_session(device_id)
            if not session_id:
                return False

            self.cortex_session = session_id
            self.status = DeviceStatus.CONNECTED

            self.logger.info(f"Connected to Emotiv device {device_id}")
            return True

        except Exception as e:
            self.logger.error(f"Emotiv connection failed: {e}")
            self.status = DeviceStatus.ERROR
            return False

    async def _authenticate(self) -> bool:
        """Authenticate with Cortex API"""
        try:
            # In production, use proper Emotiv credentials
            auth_request = {
                "jsonrpc": "2.0",
                "method": "requestAccess",
                "params": {
                    "clientId": "emotiv_neurofeedback_app",
                    "clientSecret": "cortex_client_secret_here"
                },
                "id": 1
            }

            await self.websocket.send(json.dumps(auth_request))
            response = await self.websocket.recv()
            result = json.loads(response)

            if 'result' in result:
                self.auth_token = result['result']['accessGranted']
                return True

            return False

        except Exception as e:
            self.logger.error(f"Emotiv authentication failed: {e}")
            return False

    async def _query_devices(self) -> List[Dict[str, Any]]:
        """Query available Emotiv devices"""
        try:
            query_request = {
                "jsonrpc": "2.0",
                "method": "queryHeadsets",
                "params": {},
                "id": 2
            }

            await self.websocket.send(json.dumps(query_request))
            response = await self.websocket.recv()
            result = json.loads(response)

            return result.get('result', [])

        except Exception as e:
            self.logger.error(f"Device query failed: {e}")
            return []

    async def _create_session(self, device_id: str) -> Optional[str]:
        """Create Cortex session"""
        try:
            session_request = {
                "jsonrpc": "2.0",
                "method": "createSession",
                "params": {
                    "cortexToken": self.auth_token,
                    "headset": device_id,
                    "status": "active"
                },
                "id": 3
            }

            await self.websocket.send(json.dumps(session_request))
            response = await self.websocket.recv()
            result = json.loads(response)

            return result.get('result', {}).get('id')

        except Exception as e:
            self.logger.error(f"Session creation failed: {e}")
            return None

    async def disconnect(self) -> bool:
        """Disconnect from Emotiv device"""
        try:
            if self.cortex_session:
                # Close session
                close_request = {
                    "jsonrpc": "2.0",
                    "method": "updateSession",
                    "params": {
                        "cortexToken": self.auth_token,
                        "session": self.cortex_session,
                        "status": "close"
                    },
                    "id": 4
                }

                await self.websocket.send(json.dumps(close_request))
                await self.websocket.recv()

            if self.websocket:
                await self.websocket.close()

            self.status = DeviceStatus.DISCONNECTED
            return True

        except Exception as e:
            self.logger.error(f"Emotiv disconnection failed: {e}")
            return False

    async def start_streaming(self) -> bool:
        """Start EEG streaming from Emotiv device"""
        try:
            subscribe_request = {
                "jsonrpc": "2.0",
                "method": "subscribe",
                "params": {
                    "cortexToken": self.auth_token,
                    "session": self.cortex_session,
                    "streams": ["eeg", "mot", "dev"]
                },
                "id": 5
            }

            await self.websocket.send(json.dumps(subscribe_request))
            response = await self.websocket.recv()
            result = json.loads(response)

            if 'result' in result:
                self.status = DeviceStatus.STREAMING
                # Start background task to receive data
                asyncio.create_task(self._receive_data())
                return True

            return False

        except Exception as e:
            self.logger.error(f"Streaming start failed: {e}")
            return False

    async def stop_streaming(self) -> bool:
        """Stop EEG streaming"""
        try:
            unsubscribe_request = {
                "jsonrpc": "2.0",
                "method": "unsubscribe",
                "params": {
                    "cortexToken": self.auth_token,
                    "session": self.cortex_session,
                    "streams": ["eeg", "mot", "dev"]
                },
                "id": 6
            }

            await self.websocket.send(json.dumps(unsubscribe_request))
            await self.websocket.recv()

            self.status = DeviceStatus.CONNECTED
            return True

        except Exception as e:
            self.logger.error(f"Streaming stop failed: {e}")
            return False

    async def _receive_data(self):
        """Receive streaming data from Emotiv device"""
        try:
            while self.status == DeviceStatus.STREAMING:
                response = await self.websocket.recv()
                data = json.loads(response)

                if 'eeg' in data:
                    # Process EEG data
                    eeg_data = np.array(data['eeg'][1:])  # Skip timestamp
                    self.signal_buffer.append(eeg_data)

                    # Keep buffer size manageable
                    if len(self.signal_buffer) > 1000:
                        self.signal_buffer = self.signal_buffer[-500:]

                elif 'dev' in data:
                    # Process device status
                    await self._process_device_status(data['dev'])

        except Exception as e:
            self.logger.error(f"Data reception error: {e}")
            self.status = DeviceStatus.ERROR

    async def _process_device_status(self, dev_data: List[Any]):
        """Process device status information"""
        # Emotiv device status processing
        battery_level = dev_data[1] if len(dev_data) > 1 else 100

        if battery_level < 20:
            self.status = DeviceStatus.LOW_BATTERY

    async def get_signal_quality(self) -> SignalQuality:
        """Get current signal quality from Emotiv device"""
        if not self.signal_buffer:
            return SignalQuality(
                timestamp=datetime.now(),
                channel_qualities={},
                overall_quality=0.0,
                impedances_kohm={},
                artifacts_detected=[],
                signal_to_noise_ratio=0.0,
                contact_quality={}
            )

        # Calculate signal quality metrics
        recent_data = self.signal_buffer[-100:] if len(self.signal_buffer) >= 100 else self.signal_buffer

        channel_qualities = {}
        contact_quality = {}

        for i, (channel, position) in enumerate(self.config.channel_map.items()):
            if i < len(recent_data[0]):
                channel_data = [sample[i] for sample in recent_data]

                # Simple quality metrics
                std_dev = np.std(channel_data)
                quality_score = min(1.0, max(0.0, 1.0 - (std_dev / 100.0)))

                channel_qualities[position] = quality_score
                contact_quality[position] = "good" if quality_score > 0.8 else "fair" if quality_score > 0.5 else "poor"

        overall_quality = np.mean(list(channel_qualities.values())) if channel_qualities else 0.0

        return SignalQuality(
            timestamp=datetime.now(),
            channel_qualities=channel_qualities,
            overall_quality=overall_quality,
            impedances_kohm={},  # Emotiv doesn't provide impedance
            artifacts_detected=[],
            signal_to_noise_ratio=10.0,  # Placeholder
            contact_quality=contact_quality
        )

    async def set_configuration(self, config: Dict[str, Any]) -> bool:
        """Update Emotiv device configuration"""
        # Emotiv configuration is mostly fixed
        # Can adjust some settings via Cortex API
        return True

class MuseDriver(BaseDeviceDriver):
    """Muse 2 and Muse S driver"""

    def __init__(self, device_config: DeviceConfiguration):
        super().__init__(device_config)
        self.ble_device = None
        self.characteristic = None

    async def connect(self) -> bool:
        """Connect to Muse device via Bluetooth LE"""
        try:
            self.status = DeviceStatus.CONNECTING

            # In production, use proper BLE library like bleak
            # This is a simplified implementation

            # Scan for Muse devices
            device_found = await self._scan_for_device()
            if not device_found:
                return False

            # Connect via BLE
            connection_success = await self._connect_ble()
            if not connection_success:
                return False

            self.status = DeviceStatus.CONNECTED
            self.logger.info(f"Connected to Muse device {self.config.device_id}")
            return True

        except Exception as e:
            self.logger.error(f"Muse connection failed: {e}")
            self.status = DeviceStatus.ERROR
            return False

    async def _scan_for_device(self) -> bool:
        """Scan for Muse device"""
        # Simplified BLE scanning
        # In production, use proper BLE discovery
        return True

    async def _connect_ble(self) -> bool:
        """Connect via Bluetooth LE"""
        # Simplified BLE connection
        # In production, use proper BLE connection handling
        return True

    async def disconnect(self) -> bool:
        """Disconnect from Muse device"""
        try:
            if self.ble_device:
                # Disconnect BLE
                pass

            self.status = DeviceStatus.DISCONNECTED
            return True

        except Exception as e:
            self.logger.error(f"Muse disconnection failed: {e}")
            return False

    async def start_streaming(self) -> bool:
        """Start EEG streaming from Muse"""
        try:
            # Send start command to Muse
            # Muse uses specific BLE characteristics for control

            self.status = DeviceStatus.STREAMING
            asyncio.create_task(self._receive_muse_data())
            return True

        except Exception as e:
            self.logger.error(f"Muse streaming start failed: {e}")
            return False

    async def stop_streaming(self) -> bool:
        """Stop EEG streaming"""
        try:
            # Send stop command to Muse
            self.status = DeviceStatus.CONNECTED
            return True

        except Exception as e:
            self.logger.error(f"Muse streaming stop failed: {e}")
            return False

    async def _receive_muse_data(self):
        """Receive streaming data from Muse"""
        try:
            while self.status == DeviceStatus.STREAMING:
                # Receive BLE notifications
                # Parse Muse data format
                await asyncio.sleep(0.004)  # 256 Hz sampling

        except Exception as e:
            self.logger.error(f"Muse data reception error: {e}")
            self.status = DeviceStatus.ERROR

    async def get_signal_quality(self) -> SignalQuality:
        """Get signal quality from Muse device"""
        # Muse provides built-in signal quality indicators
        # This would parse the actual quality data from the device

        return SignalQuality(
            timestamp=datetime.now(),
            channel_qualities={
                'TP9': 0.85,
                'AF7': 0.92,
                'AF8': 0.88,
                'TP10': 0.81
            },
            overall_quality=0.87,
            impedances_kohm={},
            artifacts_detected=[],
            signal_to_noise_ratio=12.5,
            contact_quality={
                'TP9': 'good',
                'AF7': 'good',
                'AF8': 'good',
                'TP10': 'good'
            }
        )

    async def set_configuration(self, config: Dict[str, Any]) -> bool:
        """Update Muse configuration"""
        # Muse has limited configuration options
        return True

class OpenBCIDriver(BaseDeviceDriver):
    """OpenBCI Cyton board driver"""

    def __init__(self, device_config: DeviceConfiguration):
        super().__init__(device_config)
        self.serial_connection = None

    async def connect(self) -> bool:
        """Connect to OpenBCI via serial/USB"""
        try:
            self.status = DeviceStatus.CONNECTING

            # Connect via serial port
            port = f"/dev/ttyUSB{self.config.device_id}"  # Linux
            # port = f"COM{self.config.device_id}"  # Windows

            reader, writer = await serial_asyncio.open_serial_connection(
                url=port,
                baudrate=115200
            )

            self.serial_connection = (reader, writer)

            # Send reset command
            writer.write(b'v')
            response = await reader.read(100)

            if b'OpenBCI' in response:
                self.status = DeviceStatus.CONNECTED
                self.logger.info(f"Connected to OpenBCI on {port}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"OpenBCI connection failed: {e}")
            self.status = DeviceStatus.ERROR
            return False

    async def disconnect(self) -> bool:
        """Disconnect from OpenBCI"""
        try:
            if self.serial_connection:
                reader, writer = self.serial_connection
                writer.write(b's')  # Stop streaming
                writer.close()
                await writer.wait_closed()

            self.status = DeviceStatus.DISCONNECTED
            return True

        except Exception as e:
            self.logger.error(f"OpenBCI disconnection failed: {e}")
            return False

    async def start_streaming(self) -> bool:
        """Start streaming from OpenBCI"""
        try:
            reader, writer = self.serial_connection
            writer.write(b'b')  # Start streaming

            self.status = DeviceStatus.STREAMING
            asyncio.create_task(self._receive_openbci_data())
            return True

        except Exception as e:
            self.logger.error(f"OpenBCI streaming start failed: {e}")
            return False

    async def stop_streaming(self) -> bool:
        """Stop streaming"""
        try:
            reader, writer = self.serial_connection
            writer.write(b's')  # Stop streaming

            self.status = DeviceStatus.CONNECTED
            return True

        except Exception as e:
            self.logger.error(f"OpenBCI streaming stop failed: {e}")
            return False

    async def _receive_openbci_data(self):
        """Receive data from OpenBCI"""
        try:
            reader, writer = self.serial_connection

            while self.status == DeviceStatus.STREAMING:
                # Read OpenBCI packet (33 bytes)
                packet = await reader.read(33)

                if len(packet) == 33 and packet[0] == 0xA0:
                    # Parse 8-channel data
                    channels_data = []
                    for i in range(8):
                        start_idx = 2 + i * 3
                        raw_value = int.from_bytes(
                            packet[start_idx:start_idx+3],
                            byteorder='big',
                            signed=True
                        )
                        # Convert to microvolts
                        voltage = raw_value * 4.5 / (2**23 - 1) / 24 * 1000000
                        channels_data.append(voltage)

                    self.signal_buffer.append(np.array(channels_data))

                    # Keep buffer manageable
                    if len(self.signal_buffer) > 1000:
                        self.signal_buffer = self.signal_buffer[-500:]

        except Exception as e:
            self.logger.error(f"OpenBCI data reception error: {e}")
            self.status = DeviceStatus.ERROR

    async def get_signal_quality(self) -> SignalQuality:
        """Get signal quality for OpenBCI"""
        if not self.signal_buffer:
            return SignalQuality(
                timestamp=datetime.now(),
                channel_qualities={},
                overall_quality=0.0,
                impedances_kohm={},
                artifacts_detected=[],
                signal_to_noise_ratio=0.0,
                contact_quality={}
            )

        # Calculate quality from recent data
        recent_data = np.array(self.signal_buffer[-250:]) if len(self.signal_buffer) >= 250 else np.array(self.signal_buffer)

        channel_qualities = {}
        contact_quality = {}

        standard_positions = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']

        for i, position in enumerate(standard_positions[:recent_data.shape[1]]):
            channel_data = recent_data[:, i]

            # Quality based on signal characteristics
            signal_std = np.std(channel_data)
            signal_range = np.ptp(channel_data)

            # Good quality: reasonable variance, not saturated
            quality_score = 0.0
            if 5 < signal_std < 100 and signal_range < 200:
                quality_score = min(1.0, signal_std / 50.0)

            channel_qualities[position] = quality_score
            contact_quality[position] = "good" if quality_score > 0.7 else "fair" if quality_score > 0.4 else "poor"

        overall_quality = np.mean(list(channel_qualities.values())) if channel_qualities else 0.0

        return SignalQuality(
            timestamp=datetime.now(),
            channel_qualities=channel_qualities,
            overall_quality=overall_quality,
            impedances_kohm={},  # OpenBCI doesn't provide impedance directly
            artifacts_detected=[],
            signal_to_noise_ratio=15.0,
            contact_quality=contact_quality
        )

    async def set_configuration(self, config: Dict[str, Any]) -> bool:
        """Configure OpenBCI settings"""
        try:
            reader, writer = self.serial_connection

            # Set sampling rate if specified
            if 'sampling_rate' in config:
                if config['sampling_rate'] == 125:
                    writer.write(b'~6')  # 125 Hz
                elif config['sampling_rate'] == 250:
                    writer.write(b'~5')  # 250 Hz

                await asyncio.sleep(0.1)

            return True

        except Exception as e:
            self.logger.error(f"OpenBCI configuration failed: {e}")
            return False

class HomeEEGDeviceManager:
    """
    Manager for home EEG device ecosystem
    Handles multiple device types and unified interface
    """

    def __init__(self, config_path: str = None):
        """Initialize home EEG device manager"""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()

        # Device management
        self.registered_devices: Dict[str, DeviceConfiguration] = {}
        self.active_drivers: Dict[str, BaseDeviceDriver] = {}
        self.device_capabilities: Dict[DeviceType, DeviceCapabilities] = {}
        self.active_sessions: Dict[str, DeviceSession] = {}

        # Initialize device capabilities
        self._initialize_device_capabilities()

        self.logger.info("Home EEG Device Manager initialized")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load device manager configuration"""
        default_config = {
            'auto_discovery': True,
            'connection_timeout_seconds': 30,
            'signal_quality_check_interval': 5,
            'data_buffer_size_mb': 100,
            'auto_reconnect_attempts': 3,
            'session_timeout_minutes': 120,
            'min_signal_quality': 0.6,
            'supported_devices': [
                'emotiv_epoc_x', 'emotiv_insight', 'muse_2', 'muse_s',
                'openbci_cyton', 'neurosity_crown', 'dreem_headband'
            ]
        }

        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)

        return default_config

    def _setup_logging(self) -> logging.Logger:
        """Setup device manager logging"""
        logger = logging.getLogger('home_eeg_devices')
        logger.setLevel(logging.INFO)

        log_dir = Path('logs/telehealth')
        log_dir.mkdir(parents=True, exist_ok=True)

        fh = logging.FileHandler(log_dir / f'devices_{datetime.now().strftime("%Y%m%d")}.log')
        fh.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        return logger

    def _initialize_device_capabilities(self):
        """Initialize capabilities for supported devices"""
        self.device_capabilities.update({
            DeviceType.EMOTIV_EPOC_X: DeviceCapabilities(
                channel_count=14,
                sampling_rate_hz=256,
                resolution_bits=16,
                max_session_minutes=720,  # 12 hours
                has_accelerometer=True,
                has_gyroscope=True,
                supports_impedance=False,
                battery_life_hours=12.0,
                wireless_range_meters=12
            ),
            DeviceType.EMOTIV_INSIGHT: DeviceCapabilities(
                channel_count=5,
                sampling_rate_hz=256,
                resolution_bits=16,
                max_session_minutes=480,  # 8 hours
                has_accelerometer=True,
                has_gyroscope=True,
                supports_impedance=False,
                battery_life_hours=8.0,
                wireless_range_meters=10
            ),
            DeviceType.MUSE_2: DeviceCapabilities(
                channel_count=4,
                sampling_rate_hz=256,
                resolution_bits=12,
                max_session_minutes=600,  # 10 hours
                has_accelerometer=True,
                has_gyroscope=True,
                has_ppg=False,
                supports_impedance=False,
                battery_life_hours=10.0,
                wireless_range_meters=8
            ),
            DeviceType.MUSE_S: DeviceCapabilities(
                channel_count=4,
                sampling_rate_hz=256,
                resolution_bits=12,
                max_session_minutes=720,  # 12 hours
                has_accelerometer=True,
                has_gyroscope=True,
                has_ppg=True,
                supports_impedance=False,
                battery_life_hours=12.0,
                wireless_range_meters=10
            ),
            DeviceType.OPENBCI_CYTON: DeviceCapabilities(
                channel_count=8,
                sampling_rate_hz=250,
                resolution_bits=24,
                max_session_minutes=1440,  # 24 hours (powered)
                has_accelerometer=False,
                has_gyroscope=False,
                supports_impedance=True,
                battery_life_hours=26.0,
                wireless_range_meters=15
            ),
            DeviceType.NEUROSITY_CROWN: DeviceCapabilities(
                channel_count=8,
                sampling_rate_hz=256,
                resolution_bits=24,
                max_session_minutes=600,  # 10 hours
                has_accelerometer=True,
                has_gyroscope=True,
                supports_impedance=True,
                battery_life_hours=10.0,
                wireless_range_meters=12
            )
        })

    async def register_device(self, device_config: DeviceConfiguration) -> bool:
        """Register a new home EEG device"""
        try:
            # Validate device configuration
            if not self._validate_device_config(device_config):
                return False

            # Store configuration
            self.registered_devices[device_config.device_id] = device_config

            self.logger.info(f"Registered device {device_config.device_id} ({device_config.device_type.value})")
            return True

        except Exception as e:
            self.logger.error(f"Device registration failed: {e}")
            return False

    def _validate_device_config(self, config: DeviceConfiguration) -> bool:
        """Validate device configuration"""
        # Check if device type is supported
        if config.device_type not in self.device_capabilities:
            self.logger.error(f"Unsupported device type: {config.device_type}")
            return False

        capabilities = self.device_capabilities[config.device_type]

        # Validate sampling rate
        if config.sampling_rate > capabilities.sampling_rate_hz:
            self.logger.error(f"Sampling rate {config.sampling_rate} exceeds device capability")
            return False

        # Validate channel mapping
        if len(config.channel_map) > capabilities.channel_count:
            self.logger.error(f"Channel count {len(config.channel_map)} exceeds device capability")
            return False

        return True

    async def connect_device(self, device_id: str) -> bool:
        """Connect to registered device"""
        try:
            if device_id not in self.registered_devices:
                self.logger.error(f"Device {device_id} not registered")
                return False

            device_config = self.registered_devices[device_id]

            # Create appropriate driver
            driver = self._create_device_driver(device_config)
            if not driver:
                return False

            # Connect to device
            success = await driver.connect()
            if success:
                self.active_drivers[device_id] = driver
                self.logger.info(f"Connected to device {device_id}")

            return success

        except Exception as e:
            self.logger.error(f"Device connection failed: {e}")
            return False

    def _create_device_driver(self, config: DeviceConfiguration) -> Optional[BaseDeviceDriver]:
        """Create appropriate device driver"""
        try:
            if config.device_type in [DeviceType.EMOTIV_EPOC_X, DeviceType.EMOTIV_INSIGHT]:
                return EmotivDriver(config)
            elif config.device_type in [DeviceType.MUSE_2, DeviceType.MUSE_S]:
                return MuseDriver(config)
            elif config.device_type == DeviceType.OPENBCI_CYTON:
                return OpenBCIDriver(config)
            # Add more drivers as needed
            else:
                self.logger.error(f"No driver available for {config.device_type}")
                return None

        except Exception as e:
            self.logger.error(f"Driver creation failed: {e}")
            return None

    async def start_session(self,
                           device_id: str,
                           participant_id: str,
                           session_duration_minutes: int = 60) -> Optional[str]:
        """Start EEG recording session"""
        try:
            if device_id not in self.active_drivers:
                self.logger.error(f"Device {device_id} not connected")
                return None

            driver = self.active_drivers[device_id]

            # Check signal quality before starting
            quality = await driver.get_signal_quality()
            if quality.overall_quality < self.config.get('min_signal_quality', 0.6):
                self.logger.warning(f"Signal quality {quality.overall_quality} below threshold")
                # Could return None here to prevent session start

            # Start streaming
            success = await driver.start_streaming()
            if not success:
                return None

            # Create session record
            session_id = f"HOME_{device_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            session = DeviceSession(
                session_id=session_id,
                device_id=device_id,
                participant_id=participant_id,
                start_time=datetime.now(),
                signal_quality=quality
            )

            self.active_sessions[session_id] = session
            driver.current_session = session

            self.logger.info(f"Started session {session_id} for participant {participant_id}")

            # Schedule session timeout
            asyncio.create_task(self._monitor_session(session_id, session_duration_minutes))

            return session_id

        except Exception as e:
            self.logger.error(f"Session start failed: {e}")
            return None

    async def stop_session(self, session_id: str) -> bool:
        """Stop EEG recording session"""
        try:
            if session_id not in self.active_sessions:
                self.logger.error(f"Session {session_id} not found")
                return False

            session = self.active_sessions[session_id]
            driver = self.active_drivers.get(session.device_id)

            if driver:
                # Stop streaming
                await driver.stop_streaming()

                # Update session record
                session.end_time = datetime.now()
                session.duration_seconds = (session.end_time - session.start_time).total_seconds()
                session.completed_successfully = True

                # Save session data
                await self._save_session_data(session_id)

            # Remove from active sessions
            del self.active_sessions[session_id]

            self.logger.info(f"Stopped session {session_id}")
            return True

        except Exception as e:
            self.logger.error(f"Session stop failed: {e}")
            return False

    async def _monitor_session(self, session_id: str, duration_minutes: int):
        """Monitor session for automatic timeout"""
        try:
            await asyncio.sleep(duration_minutes * 60)

            if session_id in self.active_sessions:
                self.logger.info(f"Session {session_id} timeout - stopping automatically")
                await self.stop_session(session_id)

        except Exception as e:
            self.logger.error(f"Session monitoring error: {e}")

    async def _save_session_data(self, session_id: str):
        """Save session data to file"""
        try:
            session = self.active_sessions[session_id]
            driver = self.active_drivers.get(session.device_id)

            if driver and driver.signal_buffer:
                # Save EEG data
                data_dir = Path('data/telehealth/sessions')
                data_dir.mkdir(parents=True, exist_ok=True)

                data_file = data_dir / f"{session_id}_eeg_data.npz"
                eeg_data = np.array(driver.signal_buffer)

                np.savez_compressed(
                    data_file,
                    eeg_data=eeg_data,
                    sampling_rate=driver.config.sampling_rate,
                    channel_map=driver.config.channel_map,
                    session_metadata=asdict(session)
                )

                session.data_file_path = str(data_file)

                self.logger.info(f"Saved session data to {data_file}")

        except Exception as e:
            self.logger.error(f"Session data save failed: {e}")

    async def get_device_status(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get current device status"""
        try:
            if device_id not in self.active_drivers:
                return None

            driver = self.active_drivers[device_id]
            device_info = await driver.get_device_info()
            signal_quality = await driver.get_signal_quality()

            return {
                **device_info,
                'signal_quality': asdict(signal_quality),
                'current_session': driver.current_session.session_id if driver.current_session else None
            }

        except Exception as e:
            self.logger.error(f"Status check failed: {e}")
            return None

    async def get_all_device_status(self) -> Dict[str, Any]:
        """Get status of all devices"""
        status_summary = {}

        for device_id in self.active_drivers:
            status_summary[device_id] = await self.get_device_status(device_id)

        return {
            'devices': status_summary,
            'active_sessions': len(self.active_sessions),
            'total_registered': len(self.registered_devices),
            'last_updated': datetime.now().isoformat()
        }

    async def disconnect_device(self, device_id: str) -> bool:
        """Disconnect device"""
        try:
            if device_id not in self.active_drivers:
                return True

            driver = self.active_drivers[device_id]

            # Stop any active session
            if driver.current_session:
                await self.stop_session(driver.current_session.session_id)

            # Disconnect device
            success = await driver.disconnect()

            if success:
                del self.active_drivers[device_id]

            return success

        except Exception as e:
            self.logger.error(f"Device disconnection failed: {e}")
            return False

    async def auto_discover_devices(self) -> List[Dict[str, Any]]:
        """Auto-discover available home EEG devices"""
        discovered_devices = []

        try:
            # Scan for different device types
            # This would implement actual device discovery protocols

            # Example discoveries (placeholder)
            if self.config.get('auto_discovery', True):
                discovered_devices.extend([
                    {
                        'device_id': 'EPOC_001',
                        'device_type': DeviceType.EMOTIV_EPOC_X.value,
                        'connection_type': ConnectionType.WIFI.value,
                        'signal_strength': 85,
                        'battery_level': 78
                    },
                    {
                        'device_id': 'MUSE_002',
                        'device_type': DeviceType.MUSE_2.value,
                        'connection_type': ConnectionType.BLUETOOTH.value,
                        'signal_strength': 92,
                        'battery_level': 65
                    }
                ])

            self.logger.info(f"Discovered {len(discovered_devices)} devices")

        except Exception as e:
            self.logger.error(f"Device discovery failed: {e}")

        return discovered_devices