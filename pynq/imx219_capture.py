#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# -*- coding: utf-8 -*-
#
# Portions of the IMX219 initialization sequence are based on
# Linux kernel drivers/media/i2c/imx219.c.
# Copyright (C) 2019, Raspberry Pi (Trading) Ltd

"""
Kria KV260 + Raspberry Pi Camera V2 (IMX219) capture utility.

Pipeline:
    IMX219 (2-lane RAW10, 3280x2464)
      -> MIPI CSI-2 Rx Subsystem
      -> AXI4-Stream Subset Converter
      -> raw10_to_y8_pack
      -> AXI VDMA S2MM
      -> PS DDR

Jupyter:
    from imx219_capture import CaptureConfig, capture_and_show

    config = CaptureConfig(
        bit_file="design_1.bit",
        hwh_file="design_1.hwh",
        test_pattern=0x0000,  # 0x0000: real image, 0x0002: color bars
    )
    frames, report = capture_and_show(config)

Command line:
    python3 imx219_capture.py
    python3 imx219_capture.py --test-pattern
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pynq import MMIO, Overlay, allocate


@dataclass(frozen=True)
class CaptureConfig:
    """Capture parameters."""

    bit_file: str | Path = "design_1.bit"
    hwh_file: str | Path = "design_1.hwh"

    width: int = 3280
    height: int = 2464
    frame_count: int = 4
    bytes_per_pixel: int = 1

    # 0x0000: real image, 0x0002: IMX219 color-bar test pattern
    test_pattern: int = 0x0000

    exposure: int = 0x0640
    analog_gain: int = 0x00
    digital_gain: int = 0x0100

    capture_wait_sec: float = 1.0
    display_downsample: int = 4

    i2c_mux_address: int = 0x74
    i2c_mux_camera_channel: int = 0x04
    imx219_address: int = 0x10

    @property
    def stride(self) -> int:
        return self.width * self.bytes_per_pixel

    @property
    def frame_size(self) -> int:
        return self.stride * self.height

    @property
    def total_size(self) -> int:
        return self.frame_size * self.frame_count


class I2CBus:
    """Thin wrapper around PYNQ AxiIIC."""

    def __init__(self, axi_iic: Any):
        self.iic = axi_iic
        self.ffi = axi_iic._ffi

    def send(self, address: int, values: list[int]) -> int:
        values = [int(v) & 0xFF for v in values]
        if not values:
            raise ValueError("I2C送信データが空です")

        buf = self.ffi.new("unsigned char[]", bytes(values))
        sent = self.iic.send(int(address), buf, len(values), 0)
        self.iic.wait()

        if sent != len(values):
            raise RuntimeError(
                "I2C write length mismatch: "
                f"address=0x{address:02X}, expected={len(values)}, sent={sent}"
            )
        return sent

    def receive(self, address: int, length: int) -> list[int]:
        if length <= 0:
            raise ValueError("I2C読み出し長は1以上にしてください")

        buf = self.ffi.new("unsigned char[]", int(length))
        received = self.iic.receive(int(address), buf, int(length), 0)
        self.iic.wait()

        if received != length:
            raise RuntimeError(
                "I2C read length mismatch: "
                f"address=0x{address:02X}, expected={length}, received={received}"
            )
        return [int(buf[i]) for i in range(length)]


class IMX219:
    """IMX219 register access and initialization."""

    def __init__(self, bus: I2CBus, address: int = 0x10):
        self.bus = bus
        self.address = address

    def _set_pointer(self, reg: int) -> None:
        reg &= 0xFFFF
        self.bus.send(
            self.address,
            [(reg >> 8) & 0xFF, reg & 0xFF],
        )

    def read8(self, reg: int) -> int:
        self._set_pointer(reg)
        return self.bus.receive(self.address, 1)[0]

    def read16(self, reg: int) -> int:
        self._set_pointer(reg)
        data = self.bus.receive(self.address, 2)
        return (data[0] << 8) | data[1]

    def write8(self, reg: int, value: int) -> int:
        reg &= 0xFFFF
        value &= 0xFF
        return self.bus.send(
            self.address,
            [(reg >> 8) & 0xFF, reg & 0xFF, value],
        )

    def write16(self, reg: int, value: int) -> int:
        reg &= 0xFFFF
        value &= 0xFFFF
        return self.bus.send(
            self.address,
            [
                (reg >> 8) & 0xFF,
                reg & 0xFF,
                (value >> 8) & 0xFF,
                value & 0xFF,
            ],
        )

    def stop_stream(self) -> None:
        self.write8(0x0100, 0x00)
        time.sleep(0.05)

    def start_stream(self) -> None:
        self.write8(0x0100, 0x01)
        time.sleep(0.05)
        mode = self.read8(0x0100)
        if mode != 0x01:
            raise RuntimeError(
                f"IMX219 stream on failed: MODE_SELECT=0x{mode:02X}"
            )

    def configure(self, config: CaptureConfig) -> None:
        """Configure 2-lane RAW10 full-resolution operation."""

        print("\nConfiguring IMX219...")
        self.stop_stream()

        common_regs8 = [
            (0x30EB, 0x05),
            (0x30EB, 0x0C),
            (0x300A, 0xFF),
            (0x300B, 0xFF),
            (0x30EB, 0x05),
            (0x30EB, 0x09),

            (0x455E, 0x00),
            (0x471E, 0x4B),
            (0x4767, 0x0F),
            (0x4750, 0x14),
            (0x4540, 0x00),
            (0x47B4, 0x14),
            (0x4713, 0x30),
            (0x478B, 0x10),
            (0x478F, 0x10),
            (0x4793, 0x10),
            (0x4797, 0x0E),
            (0x479B, 0x0E),

            (0x0170, 0x01),  # X_ODD_INC
            (0x0171, 0x01),  # Y_ODD_INC
            (0x0128, 0x00),  # automatic D-PHY timing
        ]

        for reg, value in common_regs8:
            self.write8(reg, value)

        # External clock: 24 MHz
        self.write16(0x012A, 0x1800)

        # 2-lane PLL, matching the current Vivado MIPI configuration.
        self.write8(0x0301, 0x05)      # VTPXCK_DIV
        self.write8(0x0303, 0x01)      # VTSYCK_DIV
        self.write8(0x0304, 0x03)      # PREPLLCK_VT_DIV
        self.write8(0x0305, 0x03)      # PREPLLCK_OP_DIV
        self.write16(0x0306, 0x0039)   # PLL_VT_MPY = 57
        self.write8(0x0309, 0x0A)      # OPPXCK_DIV
        self.write8(0x030B, 0x01)      # OPSYCK_DIV
        self.write16(0x030C, 0x0072)   # PLL_OP_MPY = 114
        self.write8(0x0114, 0x01)      # 2 data lanes

        # 3280 x 2464 crop and output.
        self.write16(0x0164, 0x0000)
        self.write16(0x0166, 0x0CCF)
        self.write16(0x0168, 0x0000)
        self.write16(0x016A, 0x099F)
        self.write16(0x016C, 0x0CD0)
        self.write16(0x016E, 0x09A0)

        # Binning disabled.
        self.write8(0x0174, 0x00)
        self.write8(0x0175, 0x00)

        # RAW10.
        self.write16(0x018C, 0x0A0A)

        # Frame timing.
        self.write16(0x0160, 0x0DC6)
        self.write16(0x0162, 0x0D78)

        # Orientation and exposure.
        self.write8(0x0172, 0x00)
        self.write8(0x0157, config.analog_gain)
        self.write16(0x0158, config.digital_gain)
        self.write16(0x015A, config.exposure)

        # Test-pattern dimensions and selection.
        self.write16(0x0624, 0x0CD0)
        self.write16(0x0626, 0x09A0)
        self.write16(0x0600, config.test_pattern)

        # Keep standby until VDMA is ready.
        self.stop_stream()

    def verify(self, config: CaptureConfig) -> dict[str, int]:
        values = {
            "CHIP_ID": self.read16(0x0000),
            "MODE_SELECT": self.read8(0x0100),
            "CSI_LANE_MODE": self.read8(0x0114),
            "X_OUTPUT_SIZE": self.read16(0x016C),
            "Y_OUTPUT_SIZE": self.read16(0x016E),
            "CSI_DATA_FORMAT": self.read16(0x018C),
            "TEST_PATTERN": self.read16(0x0600),
            "PLL_VT_MPY": self.read16(0x0306),
            "PLL_OP_MPY": self.read16(0x030C),
        }

        expected = {
            "CHIP_ID": 0x0219,
            "MODE_SELECT": 0x00,
            "CSI_LANE_MODE": 0x01,
            "X_OUTPUT_SIZE": config.width,
            "Y_OUTPUT_SIZE": config.height,
            "CSI_DATA_FORMAT": 0x0A0A,
            "TEST_PATTERN": config.test_pattern,
            "PLL_VT_MPY": 57,
            "PLL_OP_MPY": 114,
        }

        print("\nIMX219 verification:")
        for name, actual in values.items():
            print(f"  {name:<16} = 0x{actual:X}")
            if actual != expected[name]:
                raise RuntimeError(
                    f"IMX219 verification failed: {name}, "
                    f"expected=0x{expected[name]:X}, actual=0x{actual:X}"
                )

        print("  result           = OK")
        return values


class VDMAS2MM:
    """AXI VDMA S2MM register control."""

    S2MM_DMACR = 0x30
    S2MM_DMASR = 0x34
    S2MM_VSIZE = 0xA0
    S2MM_HSIZE = 0xA4
    S2MM_FRMDLY_STRIDE = 0xA8
    S2MM_START_ADDR1 = 0xAC

    def __init__(self, base_address: int, address_range: int):
        self.mmio = MMIO(base_address, address_range)

    def read(self, offset: int) -> int:
        return int(self.mmio.read(offset))

    def write(self, offset: int, value: int) -> None:
        self.mmio.write(int(offset), int(value) & 0xFFFFFFFF)

    @staticmethod
    def decode_status(status: int) -> dict[str, int]:
        return {
            "Halted": (status >> 0) & 1,
            "VDMAIntErr": (status >> 4) & 1,
            "VDMASlvErr": (status >> 5) & 1,
            "VDMADecErr": (status >> 6) & 1,
            "SOFEarlyErr": (status >> 7) & 1,
            "EOLEarlyErr": (status >> 8) & 1,
            "SOFLateErr": (status >> 11) & 1,
            "FrmCnt_Irq": (status >> 12) & 1,
            "DlyCnt_Irq": (status >> 13) & 1,
            "Err_Irq": (status >> 14) & 1,
            "EOLLateErr": (status >> 15) & 1,
            "IRQFrameCntSts": (status >> 16) & 0xFF,
            "IRQDelayCntSts": (status >> 24) & 0xFF,
        }

    def print_status(self, label: str) -> int:
        control = self.read(self.S2MM_DMACR)
        status = self.read(self.S2MM_DMASR)

        print(f"\n{label}")
        print(f"  S2MM_DMACR = 0x{control:08X}")
        print(f"  S2MM_DMASR = 0x{status:08X}")

        for name, value in self.decode_status(status).items():
            print(f"  {name:<15} = {value}")

        return status

    def reset(self, timeout_sec: float = 1.0) -> None:
        self.write(self.S2MM_DMACR, 0x00000004)

        start = time.time()
        while self.read(self.S2MM_DMACR) & 0x4:
            if time.time() - start > timeout_sec:
                raise TimeoutError("VDMA reset bit did not clear")
            time.sleep(0.001)

        # Clear sticky status bits.
        self.write(self.S2MM_DMASR, 0xFFFFFFFF)
        time.sleep(0.01)

    def configure_and_start(
        self,
        config: CaptureConfig,
        frame_addresses: list[int],
    ) -> None:
        self.reset()

        for index, address in enumerate(frame_addresses):
            self.write(self.S2MM_START_ADDR1 + 4 * index, address)

        self.write(self.S2MM_FRMDLY_STRIDE, config.stride)
        self.write(
            self.S2MM_HSIZE,
            config.width * config.bytes_per_pixel,
        )

        # Run/Stop=1, Circular_Park=1.
        self.write(self.S2MM_DMACR, 0x00000003)

        # VSIZE must be written last.
        self.write(self.S2MM_VSIZE, config.height)
        time.sleep(0.05)

    def stop(self) -> None:
        control = self.read(self.S2MM_DMACR)
        self.write(self.S2MM_DMACR, control & ~0x1)
        time.sleep(0.02)

    def active_errors(self) -> list[str]:
        decoded = self.decode_status(self.read(self.S2MM_DMASR))
        error_names = [
            "VDMAIntErr",
            "VDMASlvErr",
            "VDMADecErr",
            "SOFEarlyErr",
            "EOLEarlyErr",
            "SOFLateErr",
            "Err_Irq",
            "EOLLateErr",
        ]
        return [name for name in error_names if decoded[name] != 0]


def _resolve_and_check_files(
    config: CaptureConfig,
) -> tuple[Path, Path]:
    bit_path = Path(config.bit_file).expanduser().resolve()
    hwh_path = Path(config.hwh_file).expanduser().resolve()

    print("BIT file:", bit_path)
    print("HWH file:", hwh_path)

    if not bit_path.exists():
        raise FileNotFoundError(f"bitファイルがありません: {bit_path}")
    if not hwh_path.exists():
        raise FileNotFoundError(f"hwhファイルがありません: {hwh_path}")

    return bit_path, hwh_path


def _select_camera_mux(bus: I2CBus, config: CaptureConfig) -> None:
    bus.send(config.i2c_mux_address, [config.i2c_mux_camera_channel])
    readback = bus.receive(config.i2c_mux_address, 1)[0]

    print("\nI2C mux:")
    print(f"  address  = 0x{config.i2c_mux_address:02X}")
    print(f"  readback = 0x{readback:02X}")

    if readback != config.i2c_mux_camera_channel:
        raise RuntimeError(
            "I2C mux mismatch: "
            f"expected=0x{config.i2c_mux_camera_channel:02X}, "
            f"actual=0x{readback:02X}"
        )

    print("  result   = OK")


def _frame_statistics(frames: np.ndarray) -> list[dict[str, float | int]]:
    statistics: list[dict[str, float | int]] = []

    print("\n============================================================")
    print("Captured frame statistics")
    print("============================================================")

    for index, frame in enumerate(frames):
        info: dict[str, float | int] = {
            "index": index,
            "min": int(frame.min()),
            "max": int(frame.max()),
            "mean": float(frame.mean()),
            "std": float(frame.std()),
            "nonzero_pixels": int(np.count_nonzero(frame)),
            "nonzero_rows": int(
                np.count_nonzero(np.any(frame != 0, axis=1))
            ),
        }
        statistics.append(info)

        print(
            f"frame {index}: "
            f"min={info['min']}, max={info['max']}, "
            f"mean={info['mean']:.4f}, std={info['std']:.4f}, "
            f"nonzero_pixels={info['nonzero_pixels']}, "
            f"nonzero_rows={info['nonzero_rows']}"
        )

    return statistics


def capture_frames(
    config: CaptureConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Capture frames and return a normal NumPy copy.

    Returned shape:
        (frame_count, height, width)
    """

    config = config or CaptureConfig()
    bit_path, _ = _resolve_and_check_files(config)

    gc.collect()

    print("\nLoading Overlay...")
    overlay = Overlay(
        str(bit_path),
        download=True,
        ignore_version=True,
    )

    required_ips = {
        "axi_iic_0",
        "axi_vdma_0",
        "mipi_csi2_rx_subsyst_0",
    }
    missing = required_ips - set(overlay.ip_dict)
    if missing:
        raise RuntimeError(f"HWHに必要なIPがありません: {sorted(missing)}")

    bus = I2CBus(overlay.axi_iic_0)
    _select_camera_mux(bus, config)

    sensor = IMX219(bus, config.imx219_address)
    sensor.configure(config)
    sensor_report = sensor.verify(config)

    vdma_info = overlay.ip_dict["axi_vdma_0"]
    vdma = VDMAS2MM(
        vdma_info["phys_addr"],
        vdma_info["addr_range"],
    )

    print("\nCapture configuration:")
    print(f"  width       = {config.width}")
    print(f"  height      = {config.height}")
    print(f"  stride      = {config.stride} bytes")
    print(f"  frame size  = {config.frame_size} bytes")
    print(f"  total size  = {config.total_size} bytes")
    print(f"  total size  = {config.total_size / (1024 * 1024):.3f} MiB")

    capture_buffer = allocate(
        shape=(config.total_size,),
        dtype=np.uint8,
        cacheable=False,
    )

    capture_status = 0

    try:
        capture_buffer[:] = 0
        base_address = int(capture_buffer.physical_address)

        frame_addresses = [
            base_address + index * config.frame_size
            for index in range(config.frame_count)
        ]

        print("\nFrame addresses:")
        for index, address in enumerate(frame_addresses):
            print(f"  frame {index}: 0x{address:08X}")
            if address % 8 != 0:
                raise RuntimeError(
                    f"Frame address is not 8-byte aligned: 0x{address:08X}"
                )

        vdma.configure_and_start(config, frame_addresses)
        vdma.print_status("VDMA before sensor stream on")

        sensor.start_stream()
        print(
            f"\nCapturing for {config.capture_wait_sec:.1f} seconds..."
        )
        time.sleep(config.capture_wait_sec)

        capture_status = vdma.print_status("VDMA after capture")

        # Copy before releasing the physically contiguous buffer.
        frames = np.array(capture_buffer, copy=True).reshape(
            config.frame_count,
            config.height,
            config.width,
        )

    finally:
        try:
            sensor.stop_stream()
            print(
                "\nIMX219 stopped: "
                f"MODE_SELECT=0x{sensor.read8(0x0100):02X}"
            )
        except Exception as error:
            print("IMX219 stop warning:", repr(error))

        try:
            vdma.stop()
            vdma.print_status("VDMA after stop")
        except Exception as error:
            print("VDMA stop warning:", repr(error))

        try:
            capture_buffer.freebuffer()
        except Exception:
            pass

    errors = vdma.active_errors()
    if errors:
        print("\nVDMA errors detected:", errors)
    else:
        print("\nVDMA transfer errors: none")

    statistics = _frame_statistics(frames)

    report: dict[str, Any] = {
        "sensor": sensor_report,
        "vdma_status": capture_status,
        "vdma_errors": errors,
        "statistics": statistics,
    }

    return frames, report


def show_best_frame(
    frames: np.ndarray,
    config: CaptureConfig | None = None,
) -> int:
    """Display the frame having the largest standard deviation."""

    config = config or CaptureConfig()

    best_index = max(
        range(frames.shape[0]),
        key=lambda index: float(frames[index].std()),
    )
    frame = frames[best_index]

    display_low = float(np.percentile(frame, 1))
    display_high = float(np.percentile(frame, 99))

    if display_high <= display_low:
        display_low = 0.0
        display_high = 255.0

    downsample = max(1, int(config.display_downsample))

    print("\nBest frame index =", best_index)
    print("Display range:", display_low, "to", display_high)

    plt.figure(figsize=(14, 8))
    plt.imshow(
        frame[::downsample, ::downsample],
        cmap="gray",
        vmin=display_low,
        vmax=display_high,
        aspect="auto",
    )

    source_name = (
        "real image" if config.test_pattern == 0 else "test pattern"
    )
    plt.title(
        f"IMX219 {source_name} - frame {best_index} "
        f"(1/{downsample} sampling)"
    )
    plt.axis("off")
    plt.show()

    return best_index


def capture_and_show(
    config: CaptureConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convenience wrapper for Jupyter."""

    config = config or CaptureConfig()
    frames, report = capture_frames(config)
    show_best_frame(frames, config)
    return frames, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture IMX219 frames through the KV260 PL pipeline."
    )
    parser.add_argument(
        "--bit",
        default="design_1.bit",
        help="Path to design_1.bit",
    )
    parser.add_argument(
        "--hwh",
        default="design_1.hwh",
        help="Path to design_1.hwh",
    )
    parser.add_argument(
        "--test-pattern",
        action="store_true",
        help="Use the IMX219 color-bar test pattern",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=1.0,
        help="Capture duration in seconds",
    )
    args = parser.parse_args()

    config = CaptureConfig(
        bit_file=args.bit,
        hwh_file=args.hwh,
        test_pattern=0x0002 if args.test_pattern else 0x0000,
        capture_wait_sec=args.wait,
    )

    capture_and_show(config)


if __name__ == "__main__":
    main()
