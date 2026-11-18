"""Pack a Llama config's split-half RoPE coefficients for the T2 HBM port."""

import argparse
import json
import math
from pathlib import Path
import struct


def inverse_frequencies(config):
    head_dim = config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"]
    params = config.get("rope_parameters") or config.get("rope_scaling") or {}
    theta = params.get("rope_theta", config.get("rope_theta", 10000.0))
    rope_type = params.get("rope_type", params.get("type", "default"))
    partial_factor = params.get("partial_rotary_factor", config.get("partial_rotary_factor", 1.0))
    if head_dim <= 0 or head_dim % 2 or partial_factor != 1.0 or theta <= 0:
        raise ValueError("T2 needs a positive even head dimension and full-head rotation")

    frequencies = [theta ** (-lane / head_dim) for lane in range(0, head_dim, 2)]
    if rope_type == "default":
        return frequencies
    if rope_type != "llama3":
        raise ValueError(f"Unsupported RoPE variant {rope_type!r}")

    factor = params["factor"]
    low = params["low_freq_factor"]
    high = params["high_freq_factor"]
    context = params["original_max_position_embeddings"]
    if factor <= 0 or context <= 0 or not 0 < low < high:
        raise ValueError("Invalid Llama 3 RoPE scaling parameters")

    scaled = []
    for frequency in frequencies:
        cycles = context * frequency / math.tau
        if cycles >= high:
            scaled.append(frequency)
        elif cycles <= low:
            scaled.append(frequency / factor)
        else:
            blend = (cycles - low) / (high - low)
            scaled.append((1 - blend) * frequency / factor + blend * frequency)
    return scaled


def q15(value):
    magnitude = math.floor(abs(value) * (1 << 15) + 0.5)
    signed = -magnitude if value < 0 else magnitude
    return max(-32768, min(32767, signed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="checkpoint config.json")
    parser.add_argument("output", type=Path, help="binary HBM image")
    parser.add_argument("--positions", type=int, default=2048, help="hardware sequence limit")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        frequencies = inverse_frequencies(config)
        if not 0 < args.positions <= config.get("max_position_embeddings", 65535):
            raise ValueError("Position count is outside the checkpoint context")
        if len(frequencies) * 4 % 32:
            raise ValueError("Each coefficient row must contain whole 32-byte beats")
        row = struct.Struct(f"<{len(frequencies) * 2}h")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("wb") as output:
            for pos in range(args.positions):
                angles = [pos * frequency for frequency in frequencies]
                cosines = [q15(math.cos(angle)) for angle in angles]
                sines = [q15(math.sin(angle)) for angle in angles]
                output.write(row.pack(*(cosines + sines)))
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
