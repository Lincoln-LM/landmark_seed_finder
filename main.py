import os
import glob
import tomllib
import importlib
import json
import numpy as np
import pyopencl as cl
from numba_pokemon_prngs.data.personal import PERSONAL_INFO_LA, PersonalInfo8LA
from numba_pokemon_prngs.data.encounter import ENCOUNTER_INFORMATION_LA
from numba_pokemon_prngs.data.encounter.encounter_area_la import SlotLA
from numba_pokemon_prngs.enums import LATime, LAWeather
from numba_pokemon_prngs.data import SPECIES_EN, NATURES_EN, GENDER_SYMBOLS
from numba_pokemon_prngs.xorshift import Xoroshiro128PlusRejection
from pla_pid_iv.util import (
    ec_pid_matrix,
    ec_pid_const,
    generate_fix_init_spec,
    xoroshiro128plus_next,
)
from pla_pid_iv.util.pa8 import PA8
import pla_pid_iv.pla_reverse.pla_reverse as pla_reverse
import resources


def read_pa8_files():
    pa8_files = glob.glob("*.pa8")
    file_info = []

    for file_name in pa8_files:
        # Extract map index and ID from the file name [Order Caught]-[Map Index]-[Landmark ID].pa8
        catch_number, map_index, id_number = file_name.split("-")
        id_number = id_number.split(".")[0]
        file_info.append((int(catch_number), int(map_index), id_number, file_name))

    return file_info


def read_config(file_path):
    with open(file_path, "rb") as config_file:
        config = tomllib.load(config_file)

    return config


def get_name_en(species: int, form: int = 0, is_alpha: bool = False) -> str:
    return f"{'Alpha ' if is_alpha else ''}{SPECIES_EN[species]}{f'-{form}' if form else ''}"


MAPS = [
    "obsidianfieldlands",
    "crimsonmirelands",
    "cobaltcoastlands",
    "coronethighlands",
    "alabastericelands",
]

os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

context = cl.create_some_context()
queue = cl.CommandQueue(context)


def main():
    config = read_config("config.toml")
    shiny_rolls = config.get("shiny_rolls", 1)
    max_gap = config.get("max_gap", 4)
    look_for_shiny = config.get("shiny_only", True)
    look_for_alpha = config.get("alpha_only", False)
    max_advances = config.get("max_advances", 4000)

    print(
        f"Config info: {shiny_rolls=} {max_gap=} {look_for_shiny=} {look_for_alpha=} {max_advances=}"
    )

    # Read the pa8 files and parse their names
    file_info = read_pa8_files()

    # Loop over each pa8 file
    for catch_number, map_index, identifier, file_name in file_info:
        print(f"\nProcessing {file_name} (Map: {map_index}, ID: {identifier})")

        with (importlib.resources.files(resources) / f"{MAPS[map_index]}.json").open(
            "r"
        ) as f:
            landmark_data = json.load(f)[identifier]

        activation_rate = landmark_data["activationRate"]
        item_reward_min = landmark_data["itemRewardMin"]
        item_reward_max = landmark_data["itemRewardMax"]
        multiple_unique_items = len(landmark_data["rewardTable"]) > 1
        encounter_table = ENCOUNTER_INFORMATION_LA[map_index + 1][
            np.uint64(landmark_data["encounterTable"])
        ]

        print(f"\nProcessing Map: {MAPS[map_index]}")
        print(f"\nActivation Rate: {activation_rate}%")
        print(f"Reward Count: {item_reward_min}-{item_reward_max}")

        print("\nReward Table:")
        for reward in landmark_data["rewardTable"]:
            print(f"{reward['item']}: {reward['probability']}%")

        print("\nEncounter Table:")
        for slot in encounter_table.slots:
            slot = np.rec.array(slot, dtype=SlotLA.dtype)
            print(
                f"{get_name_en(slot.species, slot.form, slot.is_alpha)} - Lv. {slot.min_level}-{slot.max_level}"
            )

        mon: PA8 = np.fromfile(file_name, dtype=PA8.dtype).view(np.recarray)[0]
        personal_info: PersonalInfo8LA = PERSONAL_INFO_LA[mon.species]
        if mon.form:
            personal_info = PERSONAL_INFO_LA[
                personal_info.form_stats_index + mon.form - 1
            ]
        is_alpha = bool(mon._16 & 32)
        mon_ivs = (
            mon.iv32 & 0x1F,
            (mon.iv32 >> 5) & 0x1F,
            (mon.iv32 >> 10) & 0x1F,
            (mon.iv32 >> 20) & 0x1F,
            (mon.iv32 >> 25) & 0x1F,
            (mon.iv32 >> 15) & 0x1F,
        )

        fixed_seed_low = (mon.encryption_constant - 0x229D6A5B) & 0xFFFFFFFF

        CONSTANTS = {
            "SHINY_ROLLS": shiny_rolls,
            "XORO_CONST": pla_reverse.matrix.vec_to_int(
                ec_pid_const(fixed_seed_low, shiny_rolls)
            ),
            "SEED_MAT": ",".join(
                str(pla_reverse.matrix.vec_to_int(row))
                for row in pla_reverse.matrix.generalized_inverse(
                    ec_pid_matrix(shiny_rolls)
                )
            ),
            "NULL_SPACE": ",".join(
                str(pla_reverse.matrix.vec_to_int(row))
                for row in pla_reverse.matrix.nullspace(ec_pid_matrix(shiny_rolls))
            ),
            "PID": mon.pid,
            "FIXED_SEED_LOW": fixed_seed_low,
        }

        fixed_seed_program = cl.Program(
            context,
            pla_reverse.shaders.build_shader_code(
                "fixed_seed_ec_pid_shader", CONSTANTS
            ),
        ).build()

        host_results = np.zeros(128, np.uint64)
        host_count = np.zeros(1, np.int32)
        device_results = cl.Buffer(
            context,
            cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=host_results,
        )
        device_count = cl.Buffer(
            context,
            cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=host_count,
        )

        fixed_seed_program.find_fixed_seeds(
            queue,
            (2**16,),
            None,
            device_count,
            device_results,
        )

        cl.enqueue_copy(queue, host_results, device_results)
        cl.enqueue_copy(queue, host_count, device_count)

        host_results = host_results[: host_count[0]]
        print(f"{host_count[0]} candidate fixed seeds found")
        valid_fixed_seeds = []
        for fixed_seed in host_results:
            (
                shiny,
                encryption_constant,
                pid,
                ivs,
                ability,
                gender,
                nature,
                height,
                weight,
            ) = generate_fix_init_spec(
                fixed_seed,
                personal_info.gender_ratio,
                shiny_rolls,
                False,
                3 if is_alpha else 0,
                is_alpha,
                mon.tid | (mon.sid << np.uint32(16)),
            )
            if all(iv0 == iv1 for iv0, iv1 in zip(ivs, mon_ivs)):
                valid_fixed_seeds.append(fixed_seed)
        print(f"{len(valid_fixed_seeds)} valid fixed seeds found")
        for i, fixed_seed in enumerate(valid_fixed_seeds):
            print(f"{i}: {fixed_seed=:016X}")

        landmark_seeds = []
        for fixed_seed in valid_fixed_seeds:
            for gap in range(2, max_gap + 1):
                landmark_seed_to_fixed_seed_matrix = np.zeros((64, 64), np.uint64)

                seed0, seed1 = np.uint64(0), np.uint64(0x82A2B175229D6A5B)
                for _ in range(gap):
                    seed0, seed1 = xoroshiro128plus_next(seed0, seed1)
                xoro_const = (int(seed0) & 0xFFFFFFFF) | (
                    (int(seed1) & 0xFFFFFFFF) << 32
                )
                for bit in range(64):
                    seed0, seed1 = np.uint64(1 << bit), np.uint64(0)
                    for _ in range(gap):
                        seed0, seed1 = xoroshiro128plus_next(seed0, seed1)
                    for rand_bit in range(32):
                        landmark_seed_to_fixed_seed_matrix[bit, rand_bit] = (
                            int(seed0) >> rand_bit
                        ) & 1
                        landmark_seed_to_fixed_seed_matrix[bit, rand_bit + 32] = (
                            int(seed1) >> rand_bit
                        ) & 1

                CONSTANTS = {
                    "SEED_MAT": ",".join(
                        str(pla_reverse.matrix.vec_to_int(row))
                        for row in pla_reverse.matrix.generalized_inverse(
                            landmark_seed_to_fixed_seed_matrix
                        )
                    ),
                    "NULL_SPACE": ",".join(
                        str(pla_reverse.matrix.vec_to_int(row))
                        for row in pla_reverse.matrix.nullspace(
                            landmark_seed_to_fixed_seed_matrix
                        )
                    ),
                    "TARGET_RAND": fixed_seed,
                    "GAP": gap,
                    "XORO_CONST": xoro_const,
                }

                landmark_seed_program = cl.Program(
                    context,
                    pla_reverse.shaders.build_shader_code(
                        "generic_next_64_shader", CONSTANTS
                    ),
                ).build()

                host_results = np.zeros(128, np.uint64)
                host_count = np.zeros(1, np.int32)
                device_results = cl.Buffer(
                    context,
                    cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR,
                    hostbuf=host_results,
                )
                device_count = cl.Buffer(
                    context,
                    cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                    hostbuf=host_count,
                )

                landmark_seed_program.find_seeds(
                    queue,
                    (2**16, 2**16),
                    None,
                    device_count,
                    device_results,
                )

                cl.enqueue_copy(queue, host_results, device_results)
                cl.enqueue_copy(queue, host_count, device_count)

                host_results = host_results[: host_count[0]]
                landmark_seeds.extend(
                    (landmark_seed, fixed_seed) for landmark_seed in host_results
                )

        print(f"{len(landmark_seeds)} candidate landmark seeds found")
        valid_landmark_seeds = []
        for landmark_seed, fixed_seed in landmark_seeds:
            landmark_rng = Xoroshiro128PlusRejection(landmark_seed)
            has_encounter = landmark_rng.next_rand(100) < activation_rate
            if not has_encounter:
                continue
            encounter_slot = landmark_rng.next()  # TODO: validate
            if fixed_seed != landmark_rng.next():
                continue
            valid_landmark_seeds.append(landmark_seed)

        print(f"{len(valid_landmark_seeds)} valid landmark seeds found")
        for i, landmark_seed in enumerate(valid_landmark_seeds):
            print(f"{i}: {landmark_seed=:016X}")

            result_file_path = (
                f"{catch_number}-{map_index}-{identifier}-{landmark_seed:08X}.txt"
            )
            print(f"Writing results to {result_file_path}")

            target_found = False
            target_advance = None

            landmark_rng = Xoroshiro128PlusRejection(0)
            advance = 0
            with open(result_file_path, "w+", encoding="utf-8") as result_file:
                while advance < max_advances:
                    landmark_rng.re_init(landmark_seed)
                    has_encounter = landmark_rng.next_rand(100) < activation_rate
                    if has_encounter:
                        encounter_slot = encounter_table.calc_slot(
                            landmark_rng.next() * 5.421010862427522e-20,
                            np.int64(LATime.DAY),
                            np.int64(LAWeather.SUNNY),
                        )
                        fixed_seed = landmark_rng.next()
                        if encounter_slot.min_level != encounter_slot.max_level:
                            level = (
                                landmark_rng.next_rand(
                                    encounter_slot.max_level
                                    - encounter_slot.min_level
                                    + 1
                                )
                                + encounter_slot.min_level
                            )
                        else:
                            level = encounter_slot.min_level
                        (
                            shiny,
                            encryption_constant,
                            pid,
                            ivs,
                            ability,
                            gender_idx,
                            nature_idx,
                            height,
                            weight,
                        ) = generate_fix_init_spec(
                            np.uint64(fixed_seed),
                            personal_info.gender_ratio,
                            shiny_rolls,
                            False,
                            encounter_slot.guaranteed_ivs,
                            encounter_slot.is_alpha,
                            mon.tid | (mon.sid << np.uint32(16)),
                        )

                        # matches shiny/alpha filters
                        if (not look_for_shiny or shiny) and (
                            not look_for_alpha or encounter_slot.is_alpha
                        ):
                            target_found = True
                            target_advance = advance - 1

                        gender = GENDER_SYMBOLS[gender_idx]
                        nature = NATURES_EN[nature_idx]
                        iv_str = "/".join(str(iv) for iv in ivs)

                        result_file.write(
                            f"Encounter {advance=}: {get_name_en(encounter_slot.species, encounter_slot.form, encounter_slot.is_alpha)} {shiny=} {level=} {encryption_constant=:08X} {pid=:08X}\n{iv_str=} {ability=} {gender=} {nature=}\n{height=}\n{weight=}\n"
                        )
                    if item_reward_min != item_reward_max:
                        reward_count = (
                            landmark_rng.next_rand(
                                item_reward_max - item_reward_min + 1
                            )
                            + item_reward_min
                        )
                    else:
                        reward_count = item_reward_min
                    if has_encounter:
                        reward_count = 10
                    for _ in range(reward_count):
                        # TODO: handle and log rewards
                        if multiple_unique_items:
                            landmark_rng.next_rand(100)
                    landmark_seed = np.uint64(landmark_rng.next())
                    advance += 1
            if target_found:
                print(f"\nTarget found after {target_advance} advances")
            else:
                print("\nNo target found in advance range")


if __name__ == "__main__":
    main()
