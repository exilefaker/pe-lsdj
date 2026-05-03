"""
PyBoy SRAM helpers for LSDJ streaming.

LSDJ's 32KB save RAM is split into four 8KB banks, addressed flat as 0x0000–0x7FFF
in constants.py. PyBoy exposes banked SRAM as memory[bank, 0xA000 + offset].
"""

SRAM_BANK_SIZE = 0x2000   # 8KB per bank
SRAM_BASE      = 0xA000   # GB address of SRAM window


def read_sram(pyboy, flat_addr: int) -> int:
    bank   = flat_addr // SRAM_BANK_SIZE
    offset = flat_addr %  SRAM_BANK_SIZE
    return pyboy.memory[bank, SRAM_BASE + offset]


def write_sram(pyboy, flat_addr: int, value: int) -> None:
    bank   = flat_addr // SRAM_BANK_SIZE
    offset = flat_addr %  SRAM_BANK_SIZE
    pyboy.memory[bank, SRAM_BASE + offset] = value


def read_sram_range(pyboy, flat_start: int, length: int) -> list[int]:
    return [read_sram(pyboy, flat_start + i) for i in range(length)]


def write_sram_range(pyboy, flat_start: int, values: list[int]) -> None:
    for i, v in enumerate(values):
        write_sram(pyboy, flat_start + i, v)
